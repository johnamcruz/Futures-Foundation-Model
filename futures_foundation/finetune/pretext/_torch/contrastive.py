"""Contrastive market-regime refinement with two causal, label-free modes.

``kaufman`` (production) groups completed input windows by scale-free Kaufman Efficiency Ratio:
chop, directional trend, or an unlabeled transition band. ``temporal`` preserves the historical
time-neighborhood experiment for controlled comparisons. Neither mode reads future bars or trade
outcomes. Both operate through a disposable projection head and are evaluated on the transferable
encoder embedding with deterministic validation plus Probe Atlas retention gates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _encode_channels, _apply_control, BaseTrainer


def _random_crop_resize(x, crop_max=0.2, gen=None):
    """Mantis RandomCropResize (multichannel): crop a random 0..crop_max fraction off the time axis
    (random start), interpolate back to the original length. Preserves trend SHAPE. One view."""
    B, C, L = x.shape
    cr = float(torch.empty(1, device=x.device).uniform_(
        0.0, float(crop_max), generator=gen).item())
    cl = max(8, int(L * (1.0 - cr)))
    start = int(torch.randint(0, L - cl + 1, (1,), device=x.device,
                              generator=gen).item())
    return F.interpolate(x[:, :, start:start + cl], size=L, mode='linear', align_corners=False)


class ContrastiveTrendNet(nn.Module):
    """Mantis encoder + channel adapter + SimCLR-style projection head (Mantis 'prj'). embed(x) =
    per-channel encode + concat (the SAME embedding downstream consumes); forward(x) = L2-normalized
    projection for the contrastive loss (head discarded after)."""

    def __init__(self, C=5, new_channels=8, proj_dim=128, model_id='paris-noah/Mantis-8M'):
        super().__init__()
        from .common import load_mantis
        from mantis.adapters import LinearChannelCombiner
        self.encoder = load_mantis(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C = C
        emb = hidden * self.new_c
        self.prj = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, emb), nn.GELU(),
                                 nn.Linear(emb, proj_dim))

    def embed(self, x):                                    # [B,C,L] -> [B, new_c*hidden]
        a = self.adapter(x)
        return _encode_channels(self.encoder, a)

    def forward(self, x):                                  # [B,C,L] -> [B, proj_dim] (normalized)
        return F.normalize(self.prj(self.embed(x)), dim=1)


def _snap_to_starts(starts, target):
    """Nearest valid window start for each target position (both 1-D int64, starts SORTED).
    Returns (snapped_starts, |snapped-target| distance) — the caller decides tolerance. Snapping
    handles stream boundaries in the concatenated corpus: a target that falls in a gap/next
    stream snaps far away and gets dropped by the tolerance check."""
    j = torch.searchsorted(starts, target).clamp(0, len(starts) - 1)
    jm = (j - 1).clamp_min(0)
    pick_lo = (target - starts[jm]).abs() < (starts[j] - target).abs()
    j = torch.where(pick_lo, jm, j)
    s = starts[j]
    return s, (s - target).abs()


def _zscore_clamp(w, clamp):
    """Per-window per-channel z-score + clamp — the pipeline's universal standardize."""
    m = w.mean(2, keepdim=True)
    s = w.std(2, keepdim=True) + 1e-6
    return ((w - m) / s).clamp(-clamp, clamp)


def _vol_sigma(w_raw, close_ch=3):
    """Data-driven per-window volatility: mean |Δclose| / mean |close| of the RAW window.
    Scale-free across tickers/TFs; high = chaotic/noisy window. A weight source, NOT a label."""
    c = w_raw[:, min(close_ch, w_raw.shape[1] - 1), :]
    return c.diff(dim=1).abs().mean(1) / c.abs().mean(1).clamp_min(1e-9)


def _kaufman_efficiency(w_raw, close_ch=3):
    """Causal Kaufman Efficiency Ratio and direction for completed raw windows.

    ER = net displacement / total traveled path and is bounded in [0, 1]. It is
    scale-free across instruments and timeframes and uses no bar beyond the input
    window. Direction is deliberately consulted only for efficient trend windows.
    """
    c = w_raw[:, min(close_ch, w_raw.shape[1] - 1), :]
    displacement = c[:, -1] - c[:, 0]
    path = c.diff(dim=1).abs().sum(1)
    efficiency = (displacement.abs() / path.clamp_min(1e-9)).clamp(0.0, 1.0)
    return efficiency, displacement.sign()


def _kaufman_regime(w_raw, chop=0.25, trend=0.50, close_ch=3):
    """Map causal windows to chop=0, uptrend=1, downtrend=2, transition=-1.

    The gap between ``chop`` and ``trend`` prevents uncertain middle-efficiency
    windows from receiving a brittle pseudo-label; they retain only their paired
    augmented view as a positive.
    """
    if not 0.0 <= chop < trend <= 1.0:
        raise ValueError("Kaufman thresholds must satisfy 0 <= chop < trend <= 1")
    efficiency, direction = _kaufman_efficiency(w_raw, close_ch)
    regime = torch.full_like(direction, -1, dtype=torch.long)
    regime[efficiency <= chop] = 0
    regime[(efficiency >= trend) & (direction > 0)] = 1
    regime[(efficiency >= trend) & (direction < 0)] = 2
    return regime, efficiency


def _augment(x, gen, noise=0.10, scale=0.20, tmask=0.15, crop_max=0.2):
    """One stochastic view: crop-resize (trend-shape preserving) + gaussian noise + per-channel
    scale jitter + contiguous time-mask. Input is standardized (unit-ish sigma), so `noise` is
    in sigma units."""
    v = _random_crop_resize(x, crop_max, gen)
    if noise > 0:
        v = v + noise * torch.randn(v.shape, device=v.device, generator=gen)
    if scale > 0:
        sc = 1.0 + scale * (2 * torch.rand((v.shape[0], v.shape[1], 1), device=v.device,
                                           generator=gen) - 1)
        v = v * sc
    if tmask > 0:
        L = v.shape[2]
        mlen = max(1, int(L * tmask))
        t0 = int(torch.randint(0, L - mlen + 1, (1,), device=v.device, generator=gen).item())
        v = v.clone()
        v[:, :, t0:t0 + mlen] = 0.0
    return v


def _weighted_supcon(z, group, pos_ok, positions, w_row, temperature, far_min):
    """SupCon over the stacked batch. z:[N,D] L2-normalized; group: rows of the same anchor
    family (views + its temporal positives) are mutual POSITIVES; pos_ok=False rows (failed
    snaps) act as plain negatives. Pairs from DIFFERENT groups closer than `far_min` bars are
    EXCLUDED (neither positive nor negative). w_row: per-row anchor σ-weight."""
    N = z.shape[0]
    sim = (z @ z.t()) / temperature
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    same_group = (group[:, None] == group[None, :]) & ~eye
    near = (positions[:, None] - positions[None, :]).abs() < far_min
    excluded = eye | (near & ~same_group)                       # near-but-not-positive: drop
    sim = sim.masked_fill(excluded, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = same_group & pos_ok[None, :]                          # failed snaps can't be positives
    cnt = pos.sum(1)
    loss_row = -(logp * pos).sum(1) / cnt.clamp_min(1)
    valid = (cnt > 0) & pos_ok                                  # anchor rows with >=1 positive
    if not valid.any():
        return z.sum() * 0.0
    w = w_row[valid]
    return (loss_row[valid] * w).sum() / w.sum().clamp_min(1e-9)


def _regime_supcon(z, instance, regime, w_row, temperature):
    """SupCon using causal regime compatibility plus same-window view identity.

    Recognized chop/up/down rows share positives across the batch. Transition rows
    (label -1) are never forced together and learn only from their second augmented
    view. All other rows remain negatives, including opposing trend directions.
    """
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_instance = instance[:, None] == instance[None, :]
    known = (regime[:, None] >= 0) & (regime[None, :] >= 0)
    same_regime = known & (regime[:, None] == regime[None, :])
    positive = (same_instance | same_regime) & ~eye
    sim = (z @ z.t()) / temperature
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    count = positive.sum(1)
    loss_row = -(logp * positive).sum(1) / count.clamp_min(1)
    valid = count > 0
    # Prevent the naturally common chop class from dominating the gradient. Known regimes
    # receive inverse-frequency weights within the batch; transition instances retain weight 1.
    balanced = torch.ones_like(w_row)
    recognized = regime >= 0
    if recognized.any():
        counts = torch.bincount(regime[recognized], minlength=3).float().clamp_min(1)
        present = counts > 1  # each anchor has two views, so absent classes remain at clamp=1
        scales = recognized.sum().float() / (present.sum().clamp_min(1) * counts)
        balanced[recognized] = scales[regime[recognized]]
        balanced = balanced / balanced.mean().clamp_min(1e-9)
    weights = (w_row * balanced)[valid]
    return (loss_row[valid] * weights).sum() / weights.sum().clamp_min(1e-9)


class _ContrastiveTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=64, pos_deltas=(2, 16, 64), far_min=512,
                 new_channels=8, proj_dim=128, temperature=0.1, aug_noise=0.10, aug_scale=0.20,
                 aug_tmask=0.15, crop_max=0.2, vol_weight=1.0, w_clip=4.0, metrics_n=768,
                 regime_key='temporal', kaufman_chop=0.25, kaufman_trend=0.50,
                 model_id='paris-noah/Mantis-8M', backbone_ckpt=None, clamp=10.0, **base):
        super().__init__(big, tr, va, amp=False, **base)        # InfoNCE runs fp32
        self.seq = int(seq)
        self.pos_deltas = [int(d) for d in pos_deltas]
        self.far_min = int(far_min)
        self.tol = [max(4, d // 2) for d in self.pos_deltas]    # snap tolerance per scale
        self.new_channels, self.proj_dim = new_channels, proj_dim
        self.temperature, self.crop_max, self.clamp = temperature, crop_max, clamp
        self.aug_noise, self.aug_scale, self.aug_tmask = aug_noise, aug_scale, aug_tmask
        self.vol_weight, self.w_clip = float(vol_weight), float(w_clip)
        if regime_key not in {'temporal', 'kaufman'}:
            raise ValueError(f"unsupported contrastive regime_key={regime_key!r}")
        self.regime_key = regime_key
        self.kaufman_chop, self.kaufman_trend = float(kaufman_chop), float(kaufman_trend)
        self.metrics_n = int(metrics_n)
        self.model_id, self.backbone_ckpt = model_id, backbone_ckpt
        self.C = int(self.big_t.shape[1])
        # Some legacy callers constructed starts for their historical context+horizon
        # parent length rather than this trainer's explicit ``seq``.  Exclude any
        # tail start that cannot hold a full window; otherwise random sampling makes
        # the same run intermittently fail at the corpus boundary.
        last_start = len(self.big_t) - self.seq
        self.tr_sorted = torch.sort(self.tr[self.tr <= last_start])[0]  # snapping needs sorted
        self.va_sorted = torch.sort(self.va[self.va <= last_start])[0]
        self.tr_sorted_sampling = self._sampling_for_values(self.tr_sorted, 'tr')
        self.va_sorted_sampling = self._sampling_for_values(self.va_sorted, 'va')
        if not len(self.tr_sorted) or not len(self.va_sorted):
            raise ValueError("contrastive train/validation starts contain no complete windows")

    def build_net(self):
        net = ContrastiveTrendNet(C=self.C, new_channels=self.new_channels,
                                  proj_dim=self.proj_dim, model_id=self.model_id).to(self.dev)
        if self.backbone_ckpt:                                  # warm-start from stage-2 seq2seq
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    # ------------------------------------------------------------------ batch construction
    def _windows_at(self, pos):
        """Gather raw windows [n, C, seq] ending-exclusive at pos+seq for absolute positions."""
        rows = pos[:, None] + torch.arange(self.seq, device=self.dev)[None, :]
        return self.big_t[rows].permute(0, 2, 1).contiguous()

    def _sorted(self, starts):
        return self.tr_sorted if starts is self.tr else self.va_sorted

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
        ss = self._sorted(starts)
        sampling = self.tr_sorted_sampling if starts is self.tr else self.va_sorted_sampling
        b_idx = self.sample_indices(ss, generator=gen, sampling=sampling)
        s = ss[b_idx]                                           # [B] anchor start positions
        raw_a = self._windows_at(s)
        sigma = _vol_sigma(raw_a)                               # data-driven anchor volatility
        regime, efficiency = _kaufman_regime(
            raw_a, self.kaufman_chop, self.kaufman_trend)
        anchors = _zscore_clamp(raw_a, self.clamp)
        anchors = _apply_control(anchors, self.control)
        if self.regime_key == 'kaufman':
            return anchors, [], None, None, s, sigma, regime, efficiency
        pos_s, pos_ok = [], []
        for d, tol in zip(self.pos_deltas, self.tol):
            ps, dist = _snap_to_starts(ss, s + d)
            pos_s.append(ps)
            pos_ok.append(dist <= tol)
        positives = [_zscore_clamp(self._windows_at(p), self.clamp) for p in pos_s]
        # corrupt ONLY the input (controls destroy the temporal structure the loss feeds on)
        positives = [_apply_control(p, self.control) for p in positives]
        return (anchors, positives, torch.stack(pos_ok), torch.stack(pos_s), s, sigma,
                regime, efficiency)

    def _sigma_weights(self, sigma):
        """σ_t -> per-anchor weight: high-vol DOWN-weighted (w = (med/σ)^vol_weight, mean-1
        normalized, clipped). vol_weight=0 disables (all-equal)."""
        if self.vol_weight <= 0:
            return torch.ones_like(sigma)
        med = sigma.median().clamp_min(1e-9)
        w = (med / sigma.clamp_min(1e-9)) ** self.vol_weight
        w = w.clamp(1.0 / self.w_clip, self.w_clip)
        return w / w.mean().clamp_min(1e-9)

    def compute_loss(self, batch):
        anchors, positives, pos_ok, pos_s, s, sigma, regime, _efficiency = batch
        B = anchors.shape[0]
        g = self.gen
        v1 = _augment(anchors, g, self.aug_noise, self.aug_scale, self.aug_tmask, self.crop_max)
        v2 = _augment(anchors, g, self.aug_noise, self.aug_scale, self.aug_tmask, self.crop_max)
        if self.regime_key == 'kaufman':
            z = self.net(torch.cat([v1, v2], 0))
            ids = torch.arange(B, device=self.dev).repeat(2)
            labels = regime.repeat(2)
            return _regime_supcon(z, ids, labels,
                                  self._sigma_weights(sigma).repeat(2), self.temperature)
        X = torch.cat([v1, v2] + positives, 0)                  # [(2+K)B, C, seq]
        z = self.net(X)                                         # L2-normalized projections
        ids = torch.arange(B, device=self.dev)
        K = len(positives)
        group = torch.cat([ids] * (2 + K), 0)
        ok = torch.cat([torch.ones(2 * B, dtype=torch.bool, device=self.dev),
                        pos_ok.reshape(-1)], 0)
        positions = torch.cat([s, s] + [p for p in pos_s], 0)
        w_row = self._sigma_weights(sigma).repeat(2 + K)
        return _weighted_supcon(z, group, ok, positions, w_row, self.temperature, self.far_min)

    # ------------------------------------------------------------------ spec validation (A-E)
    @torch.no_grad()
    def _regime_metrics(self, n=768):
        """The requirement doc's structural metrics, on the DOWNSTREAM embedding (net.embed):
          A smooth     temporal consistency: cos(z_t, z_nearest) - cos(z_t, z_random)  (want >0)
          B sil        emergent structure: k-means silhouette of the embedding cloud   (want >0)
          C scale_span multi-scale: sim(short Δ) - sim(long Δ) with monotone ordering  (want >=0)
          D vol_ratio  noise robustness: dispersion(high-σ) / dispersion(low-σ)        (want ~1)
          E drift      temporal stability: early-vs-late centroid shift / cloud radius (want <1)
        Computed on a seeded fixed sample of VAL windows — diagnostics + the stage gate, never a
        trading metric (the SHIP gate stays the stage-2 benchmark, judged elsewhere)."""
        ss = self.va_sorted
        g = torch.Generator(device=self.dev)
        g.manual_seed(20260704)
        idx = torch.sort(torch.randint(0, len(ss) - 1, (min(n, len(ss) - 1),),
                                       device=self.dev, generator=g))[0]
        s = ss[idx]
        z = F.normalize(self.net.embed(_zscore_clamp(self._windows_at(s), self.clamp)), dim=1)
        # A: nearest valid neighbor vs random pair
        s1, d1 = _snap_to_starts(ss, s + 1)
        z1 = F.normalize(self.net.embed(_zscore_clamp(self._windows_at(s1), self.clamp)), dim=1)
        okA = d1 <= 4
        perm = torch.randperm(len(z), device=self.dev, generator=g)
        smooth = float((z[okA] * z1[okA]).sum(1).mean() - (z * z[perm]).sum(1).mean())
        # C: similarity across the positive scales (short/medium/long)
        sims = []
        for d, tol in zip(self.pos_deltas, self.tol):
            pd, dd = _snap_to_starts(ss, s + d)
            zk = F.normalize(self.net.embed(_zscore_clamp(self._windows_at(pd), self.clamp)),
                             dim=1)
            okd = dd <= tol
            sims.append(float((z[okd] * zk[okd]).sum(1).mean()) if okd.any() else float('nan'))
        finite = [x for x in sims if x == x]
        scale_mono = bool(all(a >= b - 1e-6 for a, b in zip(finite, finite[1:])))
        scale_span = (finite[0] - finite[-1]) if len(finite) >= 2 else float('nan')
        # B: emergent cluster structure (CPU sklearn on the sample)
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            zc = z.float().cpu().numpy()
            lab = KMeans(n_clusters=6, n_init=4, random_state=0).fit_predict(zc)
            sil = float(silhouette_score(zc, lab)) if len(set(lab)) > 1 else 0.0
        except Exception:
            sil = float('nan')
        # D: high-vol vs low-vol dispersion (collapse/domination check)
        sig = _vol_sigma(self._windows_at(s))
        q1, q2 = torch.quantile(sig.float(), torch.tensor([1 / 3, 2 / 3], device=sig.device))
        lo, hi = z[sig <= q1], z[sig >= q2]

        def _disp(a):
            return float((a - a.mean(0, keepdim=True)).norm(dim=1).mean()) if len(a) > 1 else 0.0
        vol_ratio = _disp(hi) / max(_disp(lo), 1e-9)
        # E: early-vs-late half stability
        half = len(z) // 2
        c_all = z.mean(0, keepdim=True)
        radius = float((z - c_all).norm(dim=1).mean())
        drift = float((z[:half].mean(0) - z[half:].mean(0)).norm()) / max(radius, 1e-9)
        raw = self._windows_at(s)
        regime, efficiency = _kaufman_regime(
            raw, self.kaufman_chop, self.kaufman_trend)
        sim = z @ z.t()
        eye = torch.eye(len(z), dtype=torch.bool, device=self.dev)
        known = (regime[:, None] >= 0) & (regime[None, :] >= 0)
        same = known & (regime[:, None] == regime[None, :]) & ~eye
        different = known & (regime[:, None] != regime[None, :])
        regime_margin = (float(sim[same].mean() - sim[different].mean())
                         if same.any() and different.any() else float('nan'))
        return {'smooth': smooth, 'sil': sil, 'scale_span': float(scale_span),
                'scale_mono': scale_mono, 'vol_ratio': float(vol_ratio), 'drift': drift,
                'kaufman_margin': regime_margin,
                'kaufman_known_frac': float((regime >= 0).float().mean()),
                'kaufman_er_mean': float(efficiency.mean())}

    @torch.no_grad()
    def val_eval(self):
        """FIXED-batch val loss (seeded, RNG save/restored -> stable early-stop) + the spec's
        A-E regime-geometry metrics on the downstream embedding."""
        self.net.eval()
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(20260704)
        vgen = torch.Generator(device=self.dev)
        vgen.manual_seed(20260704)
        try:
            tot = 0.0
            nb = min(10, max(1, len(self.va) // self.batch))
            for _ in range(nb):
                tot += float(self.compute_loss(self.make_batch(self.va, gen=vgen)))
            mx = self._regime_metrics(self.metrics_n)
            b = self.make_batch(self.va, gen=vgen)
            mx['std'] = float(self.net.embed(b[0]).std(0).mean())
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        self.net.train()
        return tot / nb, mx

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"A.smooth={extra['smooth']:+.3f} B.sil={extra['sil']:+.3f} "
                  f"C.span={extra['scale_span']:+.3f}{'✓' if extra['scale_mono'] else '✗'} "
                  f"D.vol={extra['vol_ratio']:.2f} E.drift={extra['drift']:.2f} "
                  f"K.ERmargin={extra['kaufman_margin']:+.3f} "
                  f"std={extra['std']:.3f}{'  *' if improved else ''}", flush=True)


def regime_gate(extra):
    """The requirement doc's success definition as an explicit gate on the A-E metrics.
    Thresholds are stated heuristics for 'smooth, structured market-state geometry' vs noise:
      A smooth > 0.05 (adjacent windows measurably closer than random)
      B sil    > 0.05 (clusters emerge above the no-structure ~0 line)
      C monotone scale ordering with span >= 0 (multi-scale similarity preserved)
      D vol_ratio in [0.5, 2.0] (high-vol neither collapsed nor dominating)
      E drift  < 1.0 (early/late centroids shift less than the cloud radius)"""
    checks = {
        'A_temporal_consistency': extra['smooth'] > 0.05,
        'B_emergent_structure': extra['sil'] == extra['sil'] and extra['sil'] > 0.05,
        'C_multi_scale': bool(extra['scale_mono']) and extra['scale_span'] >= 0,
        'D_noise_robustness': 0.5 <= extra['vol_ratio'] <= 2.0,
        'E_temporal_stability': extra['drift'] < 1.0,
    }
    return all(checks.values()), checks


def train_ssl_contrastive(big, train_starts, val_starts, *, seq=64, pos_deltas=(2, 16, 64),
                          far_min=512, new_channels=8, proj_dim=128, temperature=0.1,
                          aug_noise=0.10, aug_scale=0.20, aug_tmask=0.15, crop_max=0.2,
                          vol_weight=1.0, w_clip=4.0, metrics_n=768, epochs=60,
                          regime_key='temporal', kaufman_chop=0.25, kaufman_trend=0.50,
                          steps_per_epoch=200, batch=256, lr=2e-4, weight_decay=0.05,
                          patience=8, device=None, model_id='paris-noah/Mantis-8M',
                          backbone_ckpt=None, control='real', seed=0, clamp=10.0, grad_clip=1.0,
                          verbose=True, ckpt_path=None, resume=False, freeze_encoder_layers=0,
                          lora_r=0, lora_alpha=16.0, lora_dropout=0.0,
                          log_every_steps=25, **_ignore):
    """Causal regime contrastive refinement using Kaufman or temporal grouping.

    Returns the best encoder state plus structural/Kaufman diagnostics. The projection head is
    training-only, while the full trainer sidecar permits a true crash resume.
    """
    return _ContrastiveTrainer(
        big, train_starts, val_starts, seq=seq, pos_deltas=pos_deltas, far_min=far_min,
        new_channels=new_channels, proj_dim=proj_dim, temperature=temperature,
        aug_noise=aug_noise, aug_scale=aug_scale, aug_tmask=aug_tmask, crop_max=crop_max,
        vol_weight=vol_weight, w_clip=w_clip, metrics_n=metrics_n, model_id=model_id,
        regime_key=regime_key, kaufman_chop=kaufman_chop, kaufman_trend=kaufman_trend,
        backbone_ckpt=backbone_ckpt, clamp=clamp, epochs=epochs,
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers, lora_r=lora_r,
        lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        log_every_steps=log_every_steps).fit()
