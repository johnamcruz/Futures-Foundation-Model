"""Stage-2.8 NEXT-LEG + ordered future path-race trainer.

This module subclasses stage-2.6 and never edits it. The saved artifact is still encoder-only.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_race import RACE_LEVELS, ordered_adverse_curve
from .nextleg import NextLegNet, _NextLegTrainer, _alternating_fractals


def _leg_race_targets(big, k, leg_cap, race_levels=RACE_LEVELS, race_cap=2.0):
    """Stage-2.6 t1/t2 plus the future-only ordered adverse curve.

    Targets are anchored at confirmation. Race bars are strictly ``confirm+1 .. next origin``;
    t2 still resolves at the following origin. The inherited two-leg reserve and trainer assertion
    therefore bound every future value inside its temporal split.
    """
    h, l, c = big[:, 1], big[:, 2], big[:, 3]
    seq = _alternating_fractals(h, l, k)
    confirms, tgts, oks = [], [], []
    for i in range(len(seq) - 2):
        _o_i, c_i, d = seq[i]
        o_n, _, _ = seq[i + 1]
        o_nn, _, _ = seq[i + 2]
        t1, t2 = o_n - c_i, o_nn - o_n
        race = ordered_adverse_curve(h, l, c, c_i, o_n, d, race_levels, race_cap)
        ok = ((t1 > 0) and (t2 > 0) and (t1 <= leg_cap) and (t2 <= leg_cap)
              and bool(np.isfinite(race).all()))
        confirms.append(c_i)
        tgts.append((np.log1p(max(t1, 0)), np.log1p(max(t2, 0)),
                     *np.nan_to_num(race, nan=0.0).tolist()))
        oks.append(ok)
    return (np.asarray(confirms, np.int64), np.asarray(tgts, np.float32),
            np.asarray(oks, bool))


class NextLegRaceNet(NextLegNet):
    """NextLeg net with t1/t2 plus one output per ordered progress level."""

    def __init__(self, *a, race_levels=RACE_LEVELS, **kw):
        super().__init__(*a, **kw)
        self.race_levels = tuple(race_levels)
        emb = self.decoder[0].in_features
        self.leg_head = nn.Sequential(
            nn.Linear(emb, emb // 4), nn.GELU(),
            nn.Linear(emb // 4, 2 + len(self.race_levels)))


class _NextLegRaceTrainer(_NextLegTrainer):
    def __init__(self, big, tr, va, *, race_w=1.0, race_cap=2.0,
                 race_levels=RACE_LEVELS, **kw):
        self.race_w = float(race_w)
        self.race_cap = float(race_cap)
        self.race_levels = tuple(float(q) for q in race_levels)
        # Reuse every anchor, normalization, batching, reserve, and leak guard from production
        # NextLeg. Only target assembly is swapped during construction; the parent is restored even
        # if initialization fails, so stage-2.6 remains byte-reproducible.
        from . import nextleg as _nl
        original = _nl._leg_targets
        _nl._leg_targets = (lambda b, k, cap, _lv=self.race_levels, _rc=self.race_cap:
                            _leg_race_targets(b, k, cap, _lv, _rc))
        try:
            super().__init__(big, tr, va, **kw)
        finally:
            _nl._leg_targets = original
        if self.verbose:
            med = np.median(self._tgt_tr[:, 2:].cpu().numpy(), axis=0)
            print(f"  [nextleg_race] race_w={self.race_w:g} levels={self.race_levels} "
                  f"median={np.round(med, 3).tolist()} (future-only, ordered)", flush=True)

    def build_net(self):
        net = NextLegRaceNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                             model_id=self.model_id, aux_dim=0,
                             race_levels=self.race_levels).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        self.net = net

    def compute_loss(self, batch):
        ctx, cand_t, leg_t = batch
        candles, legs = self.net.forward_all(ctx)
        cand_loss = F.mse_loss(candles.float(), cand_t)
        leg_loss = F.smooth_l1_loss(legs[:, :2].float(), leg_t[:, :2])
        race_loss = F.smooth_l1_loss(legs[:, 2:].float(), leg_t[:, 2:])
        return (self.mse_weight * cand_loss + self.leg_w * leg_loss
                + self.race_w * race_loss)

    @torch.no_grad()
    def val_eval(self):
        """Validation and checkpoint selection include the new objective."""
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
            tot_r += float(F.smooth_l1_loss(legs[:, 2:].float(), leg_t[:, 2:]))
            preds.append(legs.float().cpu()); tgts.append(leg_t.cpu())
        P, T = torch.cat(preds), torch.cat(tgts)
        corr = [float(np.corrcoef(P[:, j].numpy(), T[:, j].numpy())[0, 1])
                for j in range(P.shape[1])]
        mae = [float(np.expm1(P[:, j].numpy()).mean() - np.expm1(T[:, j].numpy()).mean())
               for j in (0, 1)]
        estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        skill = 1.0 - (tot_c / nb) / max(tot_p / nb, 1e-12)
        vloss = (self.mse_weight * (tot_c / nb) + self.leg_w * (tot_l / nb)
                 + self.race_w * (tot_r / nb))
        extra = {'skill': skill, 'leg_corr1': corr[0], 'leg_corr2': corr[1],
                 'leg_bias1': mae[0], 'leg_bias2': mae[1], 'std': estd,
                 'race_corr': float(np.mean(corr[2:])),
                 'race_corrs': tuple(float(x) for x in corr[2:])}
        return vloss, extra

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            rc = '/'.join(f'{x:+.3f}' for x in extra['race_corrs'])
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"skill={extra['skill']:+.3f} "
                  f"legR={extra['leg_corr1']:+.3f}/{extra['leg_corr2']:+.3f} "
                  f"raceR={rc} emb_std={extra['std']:.4f}"
                  f"{'  *' if improved else ''}", flush=True)


def train_ssl_nextleg_race(big, train_starts, val_starts, *,
                           horizons=(5, 10, 20, 25),
                           context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                           steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05,
                           patience=8, device=None, model_id='paris-noah/Mantis-8M',
                           backbone_ckpt=None, control='real', seed=0, clamp=10.0,
                           grad_clip=1.0, verbose=True, ckpt_path=None, resume=False,
                           freeze_encoder_layers=0, std_guard=1.6, leg_cap=256, leg_w=1.0,
                           leg_k=2, mse_weight=1.0, race_w=0.25, race_cap=2.0,
                           race_levels=RACE_LEVELS, **_ignore):
    """Train an encoder on NextLeg plus a future-only ordered adverse path curve."""
    t = _NextLegRaceTrainer(
        big, train_starts, val_starts,
        horizons=horizons, context_lengths=context_lengths, new_channels=new_channels,
        model_id=model_id, backbone_ckpt=backbone_ckpt, clamp=clamp, leg_cap=leg_cap,
        leg_w=leg_w, leg_k=leg_k, mse_weight=mse_weight, race_w=race_w,
        race_cap=race_cap, race_levels=race_levels, epochs=epochs,
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard)
    return t.fit()
