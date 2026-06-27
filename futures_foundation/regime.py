"""Market-regime HMM — a reusable foundation-level market-state summary.

A Hidden Markov Model fit on per-bar observations (volatility / efficiency
features, or the Chronos embedding) discovers the market's natural regimes
(trend / range / chop / …) as hidden states. Its causal filtered state
posteriors are a compact "what regime are we in NOW" summary that downstream
heads (XGBoost) consume as extra features — the embedding tells the head what
the bar looks like; the regime posteriors tell it the persistent context.

Strategy-agnostic: the strategy declares WHICH of its features are regime
observations (via a `regime_obs(keys)` hook); this module fits/decodes the HMM.
No strategy specifics live here.

WHERE IT SITS:
    Chronos embed ─┐
                   ├─► RegimeHMM ─► state posteriors [K] ─┐
    vol features ──┘                                      ▼
       [embed | strategy features | K posteriors] ─► XGBoost
The posteriors are CONCATENATED into the head's input — the head sees the full
embedding, the strategy features AND the regime summary.

LEAK DISCIPLINE (the whole reason this is a class, not a one-liner):
  - scaler + PCA + HMM are FIT ON TRAIN BARS ONLY (a boolean train mask).
  - decode uses CAUSAL FORWARD FILTERING  P(s_t | o_1..t) — NOT hmmlearn's
    smoothing `predict_proba`, which conditions on the whole sequence (future
    bars). The bot can reproduce the filter incrementally bar-by-bar.
  - the HMM is UNSUPERVISED (never sees labels), so it cannot leak the target.

KEY CONVENTION: keys are the pipeline's (item_id, bar_index, ...) tuples.
keys[i][0] = stream id (e.g. 'NQ@3min'), keys[i][1] = bar index. Both
directions of a bar share one observation, so the HMM runs on UNIQUE bars per
stream in time order, then scatters each bar's posterior back to its signals.
"""
from __future__ import annotations

import numpy as np

# The EXACT volatility features the HMM observes — the validated set that gave
# +0.0084 AUC (shuffle-clean, not overfit). These are EXISTING features the
# labeler already computes; the regime layer only SELECTS them by name (adds NO
# new properties), decoupled from the Chronos context window.
REGIME_FEATURE_NAMES = (
    'adx', 'vol_surge_5', 'vol_surge_20', 'atr_expand_5', 'atr_expand_20',
    'range_vs_atr', 'retstd', 'std20atr',
)


def select_regime_observations(feat, feat_names):
    """Pick the EXISTING volatility/regime columns (by canonical name) out of a
    labeler's feature matrix -> (obs [N, d], names). No new properties: this is
    a column selection on features the labeler already computed. Returns the
    columns in REGIME_FEATURE_NAMES order, skipping any the labeler lacks."""
    feat = np.asarray(feat, np.float32)
    idx = {n: i for i, n in enumerate(feat_names)}
    cols, names = [], []
    for n in REGIME_FEATURE_NAMES:
        if n in idx:
            cols.append(idx[n]); names.append(n)
    if not cols:
        raise ValueError("no regime feature columns found in labeler features — "
                         "labeler.feature_names() must expose some of "
                         f"{REGIME_FEATURE_NAMES}")
    return feat[:, cols], names


# number of auto-derived context observations (see context_observations)
N_CONTEXT_OBS = 6


def context_observations(contexts):
    """AUTO regime observations from each Chronos CONTEXT WINDOW (log-closes) —
    the same input Chronos sees, read-only (the window is NOT modified). No
    strategy hook: works for any labeler whose contexts are price windows.

    Per window -> [N, 6]: realized-vol, mean|ret|, Kaufman efficiency ratio
    (1=clean trend, 0=chop), recent/full vol ratio (expansion), lag-1 return
    autocorr (persistence), normalized net drift. These are the classical
    trend-vs-range-vs-chop discriminators.
    """
    out = np.zeros((len(contexts), N_CONTEXT_OBS), np.float32)
    for n, w in enumerate(contexts):
        w = np.asarray(w, float)
        r = np.diff(w)
        if r.size < 4:
            continue
        sd = r.std() + 1e-9
        eff = abs(w[-1] - w[0]) / (np.abs(r).sum() + 1e-9)
        q = max(4, r.size // 4)
        vratio = (r[-q:].std() + 1e-9) / sd
        ac = (np.corrcoef(r[:-1], r[1:])[0, 1]
              if r[:-1].std() > 0 and r[1:].std() > 0 else 0.0)
        drift = (w[-1] - w[0]) / (sd * np.sqrt(r.size))
        out[n] = (sd, np.abs(r).mean(), eff, vratio,
                  ac if np.isfinite(ac) else 0.0, drift)
    return out


# ---------------------------------------------------------------------------
# causal HMM math (own impl — version-proof vs hmmlearn internals, and
# guarantees FILTERING, not smoothing)
# ---------------------------------------------------------------------------
def _diag_logprob(X, means, covars_diag):
    """Diagonal-Gaussian emission log-prob -> [T, K]."""
    T = X.shape[0]
    K = means.shape[0]
    out = np.empty((T, K), float)
    for k in range(K):
        d2 = ((X - means[k]) ** 2 / covars_diag[k]).sum(1)
        out[:, k] = -0.5 * (d2 + np.log(2 * np.pi * covars_diag[k]).sum())
    return out


def _forward_filter(framelogprob, startprob, transmat):
    """Causal forward filtering -> P(s_t | o_1..t), [T, K]. No future peek."""
    from scipy.special import logsumexp
    T, K = framelogprob.shape
    lt = np.log(transmat + 1e-300)
    la = np.empty((T, K))
    la[0] = np.log(startprob + 1e-300) + framelogprob[0]
    la[0] -= logsumexp(la[0])
    for t in range(1, T):
        pred = logsumexp(la[t - 1][:, None] + lt, axis=0)
        la[t] = pred + framelogprob[t]
        la[t] -= logsumexp(la[t])
    return np.exp(la)


# ---------------------------------------------------------------------------
def _stream_order(keys):
    """Group signals into per-stream ordered UNIQUE bars.
    Returns: streams = {stream_id: [(bar_index, first_signal_row), ...] sorted},
    and bar_rows = {(stream_id, bar_index): [all signal rows on that bar]}."""
    first = {}
    bar_rows = {}
    for n, k in enumerate(keys):
        sid, bar = k[0], int(k[1])
        bar_rows.setdefault((sid, bar), []).append(n)
        first.setdefault((sid, bar), n)
    streams = {}
    for (sid, bar), n in first.items():
        streams.setdefault(sid, []).append((bar, n))
    for sid in streams:
        streams[sid].sort()
    return streams, bar_rows


class RegimeHMM:
    """Unsupervised regime HMM with leak-safe, causal posterior decoding.

    n_states   number of regimes to discover (3 ≈ trend/range/chop; 4 default).
    pca_dim    reduce observations to this many dims before the HMM ONLY when
               the observation dim exceeds it (e.g. the 256-d embedding). Small
               vol-feature sets (D ≤ pca_dim) are used as-is.
    """

    def __init__(self, n_states=4, pca_dim=10, n_iter=50, seed=0):
        self.n_states = int(n_states)
        self.pca_dim = int(pca_dim)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self._scaler = None
        self._pca = None
        self._startprob = None
        self._transmat = None
        self._means = None
        self._covars = None          # diagonal [K, D']
        self._fitted = False

    # -- internal: observation -> HMM input space (scaler [+ PCA]) -----------
    def _project(self, obs):
        Z = self._scaler.transform(obs)
        if self._pca is not None:
            Z = self._pca.transform(Z)
        return np.asarray(Z, float)

    def fit(self, keys, obs, train_mask):
        """Fit scaler(+PCA)+HMM on TRAIN unique bars (train_mask True rows),
        using per-stream contiguous sequences. obs: [N, D]."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from hmmlearn.hmm import GaussianHMM

        obs = np.asarray(obs, float)
        train_mask = np.asarray(train_mask, bool)
        streams, _ = _stream_order(keys)
        tr_rows = [n for sid in streams for (bar, n) in streams[sid]
                   if train_mask[n]]
        if len(tr_rows) < self.n_states * 10:
            raise ValueError(f"too few train bars ({len(tr_rows)}) for "
                             f"{self.n_states} states")
        self._scaler = StandardScaler().fit(obs[tr_rows])
        Zt = self._scaler.transform(obs[tr_rows])
        if obs.shape[1] > self.pca_dim:
            self._pca = PCA(n_components=self.pca_dim,
                            random_state=self.seed).fit(Zt)
            Zt = self._pca.transform(Zt)
        else:
            self._pca = None
        # per-stream TRAIN sequences (HMM needs contiguous lengths)
        Zall = self._project(obs)
        seqs, lengths = [], []
        for sid in streams:
            rows = [n for (bar, n) in streams[sid] if train_mask[n]]
            if len(rows) > 5:
                seqs.append(Zall[rows]); lengths.append(len(rows))
        hmm = GaussianHMM(n_components=self.n_states, covariance_type='diag',
                          n_iter=self.n_iter, random_state=self.seed, tol=1e-3)
        hmm.fit(np.vstack(seqs), lengths)
        self._startprob = np.asarray(hmm.startprob_, float)
        self._transmat = np.asarray(hmm.transmat_, float)
        self._means = np.asarray(hmm.means_, float)
        self._covars = np.array([np.diag(c) if np.ndim(c) == 2 else np.asarray(c)
                                 for c in hmm.covars_], float)
        self._fitted = True
        return self

    def transform(self, keys, obs):
        """Causal filtered posteriors for EVERY signal -> [N, n_states].
        Each stream is filtered over its full ordered unique-bar sequence; the
        bar's posterior is scattered to all signals on that bar."""
        if not self._fitted:
            raise RuntimeError("RegimeHMM.transform before fit")
        obs = np.asarray(obs, float)
        Zall = self._project(obs)
        streams, bar_rows = _stream_order(keys)
        post = np.zeros((len(keys), self.n_states), np.float32)
        for sid, order in streams.items():
            rows = [n for (bar, n) in order]
            flp = _diag_logprob(Zall[rows], self._means, self._covars)
            f = _forward_filter(flp, self._startprob, self._transmat)
            for r, (bar, _n) in enumerate(order):
                for n in bar_rows[(sid, bar)]:
                    post[n] = f[r]
        return post

    def viterbi(self, keys, obs):
        """Most-likely state per signal (DIAGNOSTIC ONLY — uses hmmlearn's
        Viterbi, which is a full-sequence decode; do not use as a live feature).
        -> [N] int."""
        if not self._fitted:
            raise RuntimeError("RegimeHMM.viterbi before fit")
        from hmmlearn.hmm import GaussianHMM
        m = GaussianHMM(n_components=self.n_states, covariance_type='diag')
        m.startprob_, m.transmat_ = self._startprob, self._transmat
        m.means_, m.covars_ = self._means, self._covars
        obs = np.asarray(obs, float)
        Zall = self._project(obs)
        streams, bar_rows = _stream_order(keys)
        vit = np.full(len(keys), -1, int)
        for sid, order in streams.items():
            rows = [n for (bar, n) in order]
            v = m.predict(Zall[rows])
            for r, (bar, _n) in enumerate(order):
                for n in bar_rows[(sid, bar)]:
                    vit[n] = v[r]
        return vit

    def state_names(self):
        """Order states by mean of HMM observation dim 0 — a stable label so
        bundles/contracts can reference 'regime_0'..'regime_{K-1}' consistently."""
        return [f'regime_{i}' for i in range(self.n_states)]

    # -- serialization ------------------------------------------------------
    def params_dict(self):
        """Frozen params the serve-path needs to reproduce the causal filter."""
        return {
            'n_states': self.n_states, 'pca_dim': self.pca_dim,
            'scaler_mean': self._scaler.mean_.tolist(),
            'scaler_scale': self._scaler.scale_.tolist(),
            'pca_components': (None if self._pca is None
                               else self._pca.components_.tolist()),
            'pca_mean': (None if self._pca is None
                         else self._pca.mean_.tolist()),
            'startprob': self._startprob.tolist(),
            'transmat': self._transmat.tolist(),
            'means': self._means.tolist(),
            'covars_diag': self._covars.tolist(),
        }

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        return str(path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
