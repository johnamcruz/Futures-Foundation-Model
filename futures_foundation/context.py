"""Context heads — named market-understanding handles on the foundation.

Five forward-looking, close-only context targets (close-only because the
foundation's Bolt context is close-only — a head must be able to see the
inputs that define its target), and `ContextHeads`: XGBoost probes trained
ONCE on pre-cutoff foundation embeddings, frozen thereafter, exposing
`ctx_*` features downstream models can fuse by name.

Capability evidence (Phase-0 probe, full pre-2023 corpus, shuffle +
trivial-baseline controls): the frozen embedding knows future VOLATILITY
beyond trivial trailing stats (vol percentile r=0.52 vs 0.41 trivial;
expansion AUC 0.78 vs 0.70); knows structure/range at trivial-matching
level; does not know direction. See scripts/probe_context_heads.py.

Leak discipline (hard requirement): heads train only on bars whose
FORWARD label window ends before HEADS_CUTOFF. Downstream signal training
that consumes ctx_* features must use folds at/after HEADS_CUTOFF — the
chronos evaluate/produce seam enforces this.

Pre-registered gates (decided before any training run): a head ships in
`transform()` only if it clears its gate on the pre-cutoff validation
slice. Failing heads stay in the bundle (metrics recorded) but are
excluded from transform by default.

Process contract: this module is torch-free; XGBoost only. Embeddings
arrive via `futures_foundation.foundation.embed_bars` (subprocess seam).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HEADS_CUTOFF = pd.Timestamp('2023-01-01', tz='UTC')
MAX_LABEL_HORIZON = 20            # bars; longest forward window of any head

# Pre-registered probe gates — change only BEFORE a run, never after.
GATE_REG_PEARSON = 0.05
GATE_CLF_AUC = 0.55

#: (name, kind) — kind: 'reg' | 'clf'. Feature name downstream = 'ctx_<name>'.
HEAD_SPECS = [
    ('fwd_return', 'reg'),
    ('vol_expansion', 'clf'),
    ('volatility', 'reg'),
    ('structure', 'clf'),
    ('range_pos', 'reg'),
]


def compute_context_labels(close: pd.Series) -> pd.DataFrame:
    """All five forward-looking labels from a close series. NaN where a
    trailing or forward window is unavailable — never filled.

      fwd_return     reg  20-bar fwd log-return / trailing 200-bar std of
                          20-bar returns, clipped +/-4
      vol_expansion  clf  fwd 20-bar realized vol > 1.5x trailing 200-bar
                          median of 20-bar realized vol
      volatility     reg  fwd 10-bar realized-vol percentile vs trailing
                          100 bars' 10-bar vols, continuous [0,1]
      structure      clf  fwd 20-bar close max/min vs trailing 12-bar close
                          max/min: both higher = 1, both lower = 0, mixed NaN
      range_pos      reg  close at t+10 within trailing 20-bar close range
    """
    lc = np.log(close)
    r1 = lc.diff()

    out = pd.DataFrame(index=close.index)

    fwd20 = lc.shift(-20) - lc
    sigma20 = lc.diff(20).rolling(200, min_periods=50).std()
    out['fwd_return'] = (fwd20 / sigma20.replace(0, np.nan)).clip(-4, 4)

    v10 = r1.rolling(10).std()
    v20 = r1.rolling(20).std()

    fwd_v20 = v20.shift(-20)
    med_v20 = v20.rolling(200, min_periods=50).median()
    ve = (fwd_v20 > 1.5 * med_v20).astype(float)
    ve[fwd_v20.isna() | med_v20.isna()] = np.nan
    out['vol_expansion'] = ve

    # percentile of fwd 10-bar vol within the trailing 100 bars' v10 dist
    fwd_v10 = v10.shift(-10).to_numpy()
    v10a = v10.to_numpy()
    pct = np.full(len(v10a), np.nan)
    W = 100
    if len(v10a) > W:
        sw = np.lib.stride_tricks.sliding_window_view(v10a, W)
        tgt = fwd_v10[W - 1:]
        with np.errstate(invalid='ignore'):
            ranks = np.nanmean(sw < tgt[:, None], axis=1)
        bad = np.isnan(tgt) | np.isnan(sw).any(axis=1)
        ranks[bad] = np.nan
        pct[W - 1:] = ranks
    out['volatility'] = pct

    ref_hi = close.rolling(12).max()
    ref_lo = close.rolling(12).min()
    fwd_hi = close.rolling(20).max().shift(-20)   # covers t+1..t+20
    fwd_lo = close.rolling(20).min().shift(-20)
    st = pd.Series(np.nan, index=close.index)
    valid = ref_hi.notna() & ref_lo.notna() & fwd_hi.notna() & fwd_lo.notna()
    st[valid & (fwd_hi > ref_hi) & (fwd_lo > ref_lo)] = 1.0
    st[valid & (fwd_hi < ref_hi) & (fwd_lo < ref_lo)] = 0.0
    out['structure'] = st                          # mixed = NaN sentinel

    rh = close.rolling(20).max()
    rl = close.rolling(20).min()
    width = (rh - rl).replace(0, np.nan)
    out['range_pos'] = ((close.shift(-10) - rl) / width).clip(0, 1)

    return out


def _fit_head(kind, X, y, seed, n_estimators):
    import xgboost as xgb
    common = dict(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                  random_state=seed, n_jobs=0)
    if kind == 'reg':
        return xgb.XGBRegressor(objective='reg:squarederror', **common).fit(X, y)
    return xgb.XGBClassifier(objective='binary:logistic',
                             eval_metric='logloss', **common).fit(X, y)


def _score_head(kind, model, X, y):
    if kind == 'reg':
        p = model.predict(X)
        if p.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(p, y)[0, 1])
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


@dataclass
class ContextHeads:
    """Frozen XGBoost context heads on foundation embeddings.

    fit() trains every head and applies the pre-registered gate on the
    validation slice; transform() emits ONLY the heads that passed (or an
    explicit `include` override for ablations). save()/load() round-trip
    via joblib with full training metadata.
    """
    seed: int = 0
    n_estimators: int = 400
    models: dict = field(default_factory=dict)      # name -> fitted model
    metrics: dict = field(default_factory=dict)     # name -> metrics dict
    meta: dict = field(default_factory=dict)

    @property
    def active_names(self):
        """ctx_* feature names emitted by transform(), in HEAD_SPECS order."""
        return [f'ctx_{n}' for n, _ in HEAD_SPECS
                if self.metrics.get(n, {}).get('passed')]

    def fit(self, E_tr, labels_tr, E_va, labels_va, verbose=True):
        """Train all heads on (E_tr, labels_tr); gate on (E_va, labels_va).
        Rows with NaN labels are dropped per head (structure's mixed
        sentinel). Returns self."""
        for name, kind in HEAD_SPECS:
            ytr = np.asarray(labels_tr[name], np.float32)
            yva = np.asarray(labels_va[name], np.float32)
            m_tr, m_va = ~np.isnan(ytr), ~np.isnan(yva)
            if m_tr.sum() < 500 or m_va.sum() < 100:
                self.metrics[name] = dict(kind=kind, passed=False,
                                          skipped='too few rows',
                                          n_train=int(m_tr.sum()),
                                          n_val=int(m_va.sum()))
                continue
            model = _fit_head(kind, E_tr[m_tr], ytr[m_tr], self.seed,
                              self.n_estimators)
            score = _score_head(kind, model, E_va[m_va], yva[m_va])
            gate = GATE_REG_PEARSON if kind == 'reg' else GATE_CLF_AUC
            passed = bool(score > gate)
            self.models[name] = model
            self.metrics[name] = dict(
                kind=kind, metric='pearson_r' if kind == 'reg' else 'auc',
                score=score, gate=gate, passed=passed,
                n_train=int(m_tr.sum()), n_val=int(m_va.sum()))
            if verbose:
                flag = '✅ PASS' if passed else '❌ FAIL (excluded)'
                print(f"  [ctx_{name:<13}] "
                      f"{self.metrics[name]['metric']}={score:+.3f} "
                      f"gate>{gate}  {flag}")
        return self

    def transform(self, E, include=None):
        """[N, D_MODEL] embeddings -> [N, n_active] ctx features (float32),
        column order = active_names. `include` overrides the gate (list of
        bare head names) for ablation studies."""
        names = (include if include is not None
                 else [n for n, _ in HEAD_SPECS
                       if self.metrics.get(n, {}).get('passed')])
        if not names:
            return np.zeros((len(E), 0), np.float32)
        cols = []
        for name in names:
            kind = dict(HEAD_SPECS)[name]
            model = self.models[name]
            if kind == 'reg':
                cols.append(model.predict(E).astype(np.float32))
            else:
                cols.append(model.predict_proba(E)[:, 1].astype(np.float32))
        return np.column_stack(cols)

    def context_at(self, close, indices, ctx: int = 128, batch: int = 64,
                   include=None) -> pd.DataFrame:
        """Per-candle market readout — the live-inference entry point.

        For each decision bar index: causal log-close window -> foundation
        embedding (subprocess) -> named ctx_* features. The bot can call
        this every bar to know the current regime/volatility/structure
        state without ever touching embeddings directly.

        -> DataFrame indexed by `indices`, columns = active_names.
        """
        from .foundation import embed_bars
        E = embed_bars(close, indices, ctx=ctx, batch=batch)
        names = ([f'ctx_{n}' for n in include] if include is not None
                 else self.active_names)
        return pd.DataFrame(self.transform(E, include=include),
                            index=np.asarray(indices), columns=names)

    def htf_context_at(self, ts, close, indices, htf: str = '1h',
                       ctx: int = 128, batch: int = 64,
                       include=None) -> pd.DataFrame:
        """Per-candle HTF market readout for intraday trading — the same
        heads, fed a higher-timeframe close series, STRICTLY CAUSAL.

        For a decision bar at base-TF time ts[i], the HTF window contains
        only HTF buckets that FULLY ENDED at or before ts[i] (the
        cross-TF lookahead bug class: never read an HTF bar that hasn't
        closed). Conservative by construction: ts is treated as bar-open
        time, so the current (still-forming) HTF bucket is never visible.

        ts:      tz-aware timestamps of the base bars (1m/3m/5m), sorted.
        close:   base-TF closes aligned to ts.
        indices: decision-bar integer positions into ts/close.
        htf:     pandas offset ('1h', '4h', ...). Columns are suffixed,
                 e.g. 'ctx_volatility_1h'.

        -> DataFrame indexed by `indices`; rows with insufficient HTF
        history are NaN.

        NOTE: heads are trained on base-TF windows; HTF windows are a
        transfer application. Bolt embeds log-close shapes scale-free and
        the labels are bar-count-relative, but validate per strategy on
        the honest ruler before relying on it.
        """
        ts = pd.DatetimeIndex(ts)
        c = np.asarray(close, dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64)
        # bucket each base bar; HTF close = last base close in bucket
        bucket = ts.floor(htf)
        df = pd.DataFrame({'b': bucket, 'c': c})
        htf_close = df.groupby('b', sort=True)['c'].last()
        bucket_end = (htf_close.index + pd.Timedelta(htf)).asi8
        hc = np.log(htf_close.to_numpy())
        # completed buckets at decision bar i: bucket_end <= ts[i]
        n_done = np.searchsorted(bucket_end, ts.asi8[idx], side='right')
        names = ([f'ctx_{n}' for n in include] if include is not None
                 else self.active_names)
        out = pd.DataFrame(np.nan, index=idx,
                           columns=[f'{n}_{htf}' for n in names])
        ok = n_done >= ctx
        if ok.any():
            windows = np.stack([hc[d - ctx:d] for d in n_done[ok]]).astype(
                np.float32)
            # dedupe identical windows (many base bars share an HTF state)
            uniq, inv = np.unique(windows, axis=0, return_inverse=True)
            from .foundation import embed
            E = embed(uniq, batch=batch)
            out.loc[ok] = self.transform(E, include=include)[inv]
        return out

    def save(self, path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(models=self.models, metrics=self.metrics,
                         meta=self.meta, seed=self.seed,
                         n_estimators=self.n_estimators), path)
        return str(path)

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(seed=blob.get('seed', 0),
                  n_estimators=blob.get('n_estimators', 400))
        obj.models = blob['models']
        obj.metrics = blob['metrics']
        obj.meta = blob.get('meta', {})
        return obj

    def describe(self) -> str:
        active = ', '.join(self.active_names) or '(none passed)'
        return (f"ContextHeads[active: {active}] "
                f"meta={json.dumps(self.meta, default=str)[:200]}")
