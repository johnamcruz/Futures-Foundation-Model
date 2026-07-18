"""Stage-2.7 trainer: NEXT-LEG + PATH — stage 2.6 (nextleg, the GRADUATED checkpoint) plus ONE
new target: how rough the newborn leg's ride is.

WHY (a measured hole, not a guess). nextleg teaches HOW FAR (t1) and HOW LONG (t2) a leg runs —
never HOW IT GETS THERE. Measured on the production pool (ES+NQ@3min, 130,994 pivots):
    predicting "does this leg's path stop me out"
      pure stop geometry (risk/ATR, 1 feature)   AUC 0.5184
      the nextleg embedding (1280-d)             AUC 0.5572   <- +0.039 of real path structure
      embedding + geometry                       AUC 0.5572   <- geometry adds NOTHING it lacks
The encoder already holds some path information and geometry explains none of it. r1 asks for
the rest of it directly. 0.5572 is the number a 2.7 checkpoint must beat on that probe.

THE TARGET — r1 = deepest pullback WITHIN the newborn leg / that leg's OWN extent.
Pure candle highs/lows over a ratio of two distances from the SAME leg: unitless, scale-free,
instrument- and timeframe-agnostic — exactly the property that lets t1/t2 live in bars.
NO ATR. NO cost. NO entry, stop, or R. FFM learns candles; the strategy layer owns risk.
A drawdown in R-units would smuggle the stop into the pretext and make this a supervised label
aimed AT the downstream event — the shape that lost in turn-electra. This is generic path
knowledge, aimed NEAR the event.

SEPARATE MODULE ON PURPOSE: `nextleg` is the shipped backbone (AI_Models/mantis_ssl_nextleg.pt).
2.7 imports it and never edits it, so 2.6 stays byte-reproducible and the A/B is honest.
The leg head is discarded after training -> the checkpoint stays encoder-only, so a 2.7 ckpt
drops into every 2.6 consumer unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_path import leg_retrace                     # noqa: F401  torch-free path math
from .nextleg import NextLegNet, _NextLegTrainer, _alternating_fractals


def _leg_path_targets(big, k, leg_cap, retrace_cap=2.0):
    """Stage-2.6 targets + the path target, from the same alternating fractal sequence:
        t1 = extreme_{i+1} - c_i     bars until the newborn leg ends      (log1p)
        t2 = extreme_{i+2} - o_n     bars the counter-leg lasts           (log1p)
        r1 = the newborn leg's retrace fraction                           (ratio, see leg_retrace)
    -> (confirms [N], targets [N,3], ok [N]). t1/t2 are computed identically to 2.6."""
    h, l = big[:, 1], big[:, 2]
    seq = _alternating_fractals(h, l, k)
    confirms, tgts, oks = [], [], []
    for i in range(len(seq) - 2):
        o_i, c_i, d = seq[i]
        o_n, _, _ = seq[i + 1]
        o_nn, _, _ = seq[i + 2]
        t1, t2 = o_n - c_i, o_nn - o_n
        r1 = leg_retrace(h, l, o_i, o_n, d, cap=retrace_cap)
        ok = ((t1 > 0) and (t2 > 0) and (t1 <= leg_cap) and (t2 <= leg_cap)
              and bool(np.isfinite(r1)))                   # unresolved r1 -> drop the anchor
        confirms.append(c_i)
        tgts.append((np.log1p(max(t1, 0)), np.log1p(max(t2, 0)),
                     r1 if np.isfinite(r1) else 0.0))
        oks.append(ok)
    return (np.asarray(confirms, np.int64), np.asarray(tgts, np.float32),
            np.asarray(oks, bool))


class NextLegPathNet(NextLegNet):
    """2.6's net with a 3-output leg head (t1, t2, r1). Encoder untouched -> the saved
    checkpoint is encoder-only and drops into every 2.6 consumer."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        emb = self.decoder[0].in_features
        self.leg_head = nn.Sequential(nn.Linear(emb, emb // 4), nn.GELU(), nn.Linear(emb // 4, 3))


class _NextLegPathTrainer(_NextLegTrainer):
    """2.6's trainer with r1 added. Anchors, batching, normalization, the candle ANCHOR loss and
    every anti-drift guard are INHERITED unchanged — the only deltas are the 3rd target column,
    its loss term, and its diagnostic."""

    def __init__(self, big, tr, va, *, retrace_w=1.0, retrace_cap=2.0, **kw):
        self.retrace_w, self.retrace_cap = float(retrace_w), float(retrace_cap)
        # 2.6's __init__ owns the anchor/split/leak-safety logic and we want ALL of it, unchanged
        # and un-forked (the primitives-divergence trap). It resolves targets via the module-level
        # nextleg._leg_targets, so swap that symbol for the 3-column version for the duration of
        # the super() call ONLY, and restore it in `finally`. nextleg.py itself is never edited.
        from . import nextleg as _nl
        _orig = _nl._leg_targets
        _nl._leg_targets = (lambda b, k, cap, _rc=float(retrace_cap):
                            _leg_path_targets(b, k, cap, retrace_cap=_rc))
        try:
            super().__init__(big, tr, va, **kw)
        finally:
            _nl._leg_targets = _orig
        if self.verbose:
            r = self._tgt_tr[:, 2].cpu().numpy()
            print(f"  [nextleg_path] STAGE 2.7 retrace_w={self.retrace_w:g} | r1 "
                  f"p25={np.percentile(r, 25):.3f} med={np.median(r):.3f} "
                  f"p75={np.percentile(r, 75):.3f} p95={np.percentile(r, 95):.3f} "
                  f"(0=clean one-way leg)", flush=True)

    def build_net(self):
        net = NextLegPathNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                             model_id=self.model_id, aux_dim=0).to(self.dev)
        if self.backbone_ckpt:                              # warm-start from the 2.6 encoder
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    def compute_loss(self, batch):
        ctx, cand_t, leg_t = batch
        candles, legs = self.net.forward_all(ctx)
        cand_loss = F.mse_loss(candles.float(), cand_t)
        leg_loss = F.smooth_l1_loss(legs[:, :2].float(), leg_t[:, :2])     # 2.6's term, unchanged
        ret_loss = F.smooth_l1_loss(legs[:, 2].float(), leg_t[:, 2])       # r1 weighted APART:
        #   t1/t2 live in log1p-bars (~3-5), r1 in [0,2] — one smooth_l1 over all three would
        #   let the bar terms swamp the path gradient.
        return (self.mse_weight * cand_loss + self.leg_w * leg_loss
                + self.retrace_w * ret_loss)

    @torch.no_grad()
    def val_eval(self):
        """2.6's val_eval + the r1 diagnostic. NOT super() + a second pass: preds and targets must
        come from the SAME batch or every correlation is measured across mismatched samples."""
        self.net.eval()
        tot_c = tot_p = tot_l = tot_r = 0.0
        preds, tgts = [], []
        nb = min(20, max(1, len(self.va) // self.batch))
        for _ in range(nb):
            ctx, cand_t, leg_t = self.make_batch(self.va)
            candles, legs = self.net.forward_all(ctx)
            tot_c += float(F.mse_loss(candles.float(), cand_t))
            tot_p += float((cand_t ** 2).mean())
            tot_l += float(F.smooth_l1_loss(legs[:, :2].float(), leg_t[:, :2]))
            tot_r += float(F.smooth_l1_loss(legs[:, 2].float(), leg_t[:, 2]))
            preds.append(legs.float().cpu()); tgts.append(leg_t.cpu())
        P, T = torch.cat(preds), torch.cat(tgts)
        corr = [float(np.corrcoef(P[:, j].numpy(), T[:, j].numpy())[0, 1]) for j in (0, 1, 2)]
        mae = [float(np.expm1(P[:, j].numpy()).mean() - np.expm1(T[:, j].numpy()).mean())
               for j in (0, 1)]                                     # bars, bias diagnostic
        estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        skill = 1.0 - (tot_c / nb) / max(tot_p / nb, 1e-12)
        # Select the checkpoint on every objective used for training. The original 2.7 code
        # reported retrace correlation but omitted retrace loss from early stopping/checkpointing.
        vloss = (self.mse_weight * (tot_c / nb) + self.leg_w * (tot_l / nb)
                 + self.retrace_w * (tot_r / nb))
        return vloss, {'skill': skill, 'leg_corr1': corr[0], 'leg_corr2': corr[1],
                       'leg_bias1': mae[0], 'leg_bias2': mae[1], 'std': estd,
                       'retrace_loss': tot_r / nb,
                       'retrace_corr': corr[2],
                       'retrace_bias': float(P[:, 2].numpy().mean() - T[:, 2].numpy().mean())}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} skill={extra['skill']:+.3f} "
                  f"legR={extra['leg_corr1']:+.3f}/{extra['leg_corr2']:+.3f} "
                  f"retR={extra['retrace_corr']:+.3f} emb_std={extra['std']:.4f}"
                  f"{'  *' if improved else ''}", flush=True)


def train_ssl_nextleg_path(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                           context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                           steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                           device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                           control='real', seed=0, clamp=10.0, grad_clip=1.0, verbose=True,
                           ckpt_path=None, resume=False, freeze_encoder_layers=0, std_guard=1.6,
                           leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0,
                           retrace_w=1.0, retrace_cap=2.0, **_ignore):
    """NEXT-LEG + PATH SSL -> (best_encoder_state, history). History adds 'retrace_corr' (the
    learning diagnostic for the path target) to 2.6's 'skill'/'leg_corr1'/'leg_corr2'/'std'.

    Signature MIRRORS train_ssl_nextleg (+retrace_w/retrace_cap): the orchestrator hands every
    task the WHOLE shared cfg, so a trainer must name what it wants and swallow the rest in
    **_ignore. A **kw passthrough forwards cfg keys like `seq` down to BaseTrainer -> TypeError.

    GATES a 2.7 candidate must clear before it replaces 2.6 anywhere:
      1. retrace_corr materially > 0        — it actually learned the path target
      2. skill / leg_corr1 / leg_corr2 not degraded vs 2.6 — no drift damage to what works
      3. stop-race probe AUC > 0.5572       — the hole this was built for actually closed
      4. trend-lifecycle scorecard >= 2.6 bars — re-baseline first: banked probe bars were
         measured under the OLD shuffled probe split (fixed 1e4bf45), so they are not comparable
      5. downstream WR@3R / meanR >= 2.6 at the deploy operating points
    """
    t = _NextLegPathTrainer(big, train_starts, val_starts,
                            horizons=horizons, context_lengths=context_lengths,
                            new_channels=new_channels, model_id=model_id,
                            backbone_ckpt=backbone_ckpt, clamp=clamp, leg_cap=leg_cap,
                            leg_w=leg_w, leg_k=leg_k, mse_weight=mse_weight,
                            retrace_w=retrace_w, retrace_cap=retrace_cap,
                            epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
                            weight_decay=weight_decay, patience=patience, device=device,
                            seed=seed, grad_clip=grad_clip, verbose=verbose, control=control,
                            ckpt_path=ckpt_path, resume=resume,
                            freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard)
    return t.fit()
