"""Stage-2.8 NEXT-LEG + ordered future path-race trainer.

This module subclasses stage-2.6 and never edits it. The saved artifact is still encoder-only.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_race import RACE_LEVELS, ordered_adverse_curve
from .forecast import _ForecastTrainer
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
    def __init__(self, big, tr, va, *, leg_cap=256, leg_w=1.0, leg_k=2,
                 mse_weight=1.0, race_w=1.0, race_cap=2.0,
                 race_levels=RACE_LEVELS, **fw):
        self.race_w = float(race_w)
        self.race_cap = float(race_cap)
        self.race_levels = tuple(float(q) for q in race_levels)
        self.leg_w, self.mse_weight = float(leg_w), float(mse_weight)
        tr_np = tr if isinstance(tr, np.ndarray) else np.asarray(tr)
        va_np = va if isinstance(va, np.ndarray) else np.asarray(va)
        # Override only NextLeg's anchor initializer. Its current assertion confuses the small
        # candle-decoder tensor (max_ctx+25) with the assembler's temporal reserve
        # (max_ctx+2*leg_cap). Everything else—network lineage, make_batch normalization/control,
        # optimizer, shared fit loop, and encoder-only checkpoint—is inherited unchanged.
        _ForecastTrainer.__init__(self, big, tr, va, **fw)
        confirms, tgts, ok = _leg_race_targets(
            np.asarray(big, np.float32), int(leg_k), int(leg_cap),
            self.race_levels, self.race_cap)
        confirms, tgts = confirms[ok], tgts[ok]
        starts = confirms - self.max_ctx + 1
        horizon = np.expm1(tgts[:, 0]) + np.expm1(tgts[:, 1])
        max_future = float(horizon.max()) if len(horizon) else 0.0
        reserved_future = 2 * int(leg_cap)
        assert max_future < reserved_future + 1, (
            f'TEMPORAL LEAK: race target reads {max_future:.0f} bars ahead but the task permits '
            f'only {reserved_future} (2*leg_cap)')
        # tr/va contain only starts whose COMPLETE task.reserve() window stays within one stream,
        # temporal split, and the pre-2026 region. Membership is the actual boundary purge.
        m_tr, m_va = np.isin(starts, tr_np), np.isin(starts, va_np)
        if m_tr.sum() < 1000 or m_va.sum() < 200:
            raise ValueError(f'nextleg_race: too few resolved pivot anchors '
                             f'(train={int(m_tr.sum())}, val={int(m_va.sum())})')
        self._replace_start_pool('tr', starts[m_tr])
        self._replace_start_pool('va', starts[m_va])
        self._tgt_tr = torch.as_tensor(tgts[m_tr], device=self.dev)
        self._tgt_va = torch.as_tensor(tgts[m_va], device=self.dev)
        if self.verbose:
            med = np.median(self._tgt_tr[:, 2:].cpu().numpy(), axis=0)
            print(f"  [nextleg_race] anchors train={len(self.tr):,} val={len(self.va):,} "
                  f"k={leg_k} cap={leg_cap} | race_w={self.race_w:g} levels={self.race_levels} "
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
        # MPS requires contiguous operands here (CUDA accepts the strided column slices).
        leg_loss = F.smooth_l1_loss(
            legs[:, :2].float().contiguous(), leg_t[:, :2].contiguous())
        race_loss = F.smooth_l1_loss(
            legs[:, 2:].float().contiguous(), leg_t[:, 2:].contiguous())
        return (self.mse_weight * cand_loss + self.leg_w * leg_loss
                + self.race_w * race_loss)

    @torch.no_grad()
    def val_eval(self):
        """Validation and checkpoint selection include the new objective on a FIXED pivot set.

        Pivot difficulty varies substantially, so resampling validation anchors every epoch made
        the composite loss too noisy to identify the best encoder. Save/restore the training RNG
        and replay the same validation batches each epoch; training sampling is unaffected.
        """
        self.net.eval()
        gen_state = self.gen.get_state()
        self.gen.manual_seed(20260718)
        tot_c = tot_p = tot_l = tot_r = 0.0
        preds, tgts = [], []
        nb = min(20, max(1, len(self.va) // self.batch))
        try:
            for _ in range(nb):
                ctx, cand_t, leg_t = self.make_batch(self.va)
                candles, legs = self.net.forward_all(ctx)
                tot_c += float(F.mse_loss(candles.float(), cand_t))
                tot_p += float((cand_t ** 2).mean())
                tot_l += float(F.smooth_l1_loss(
                    legs[:, :2].float().contiguous(), leg_t[:, :2].contiguous()))
                tot_r += float(F.smooth_l1_loss(
                    legs[:, 2:].float().contiguous(), leg_t[:, 2:].contiguous()))
                preds.append(legs.float().cpu()); tgts.append(leg_t.cpu())
            P, T = torch.cat(preds), torch.cat(tgts)
            corr = [float(np.corrcoef(P[:, j].numpy(), T[:, j].numpy())[0, 1])
                    for j in range(P.shape[1])]
            mae = [float(np.expm1(P[:, j].numpy()).mean() - np.expm1(T[:, j].numpy()).mean())
                   for j in (0, 1)]
            estd = float(self.net.embed(self.make_batch(self.va)[0]).std(0).mean())
        finally:
            self.gen.set_state(gen_state)
            self.net.train()
        candle_loss, leg_loss, race_loss = tot_c / nb, tot_l / nb, tot_r / nb
        skill = 1.0 - candle_loss / max(tot_p / nb, 1e-12)
        vloss = (self.mse_weight * candle_loss + self.leg_w * leg_loss
                 + self.race_w * race_loss)
        extra = {'skill': skill, 'candle_loss': candle_loss, 'leg_loss': leg_loss,
                 'race_loss': race_loss, 'leg_corr1': corr[0], 'leg_corr2': corr[1],
                 'leg_bias1': mae[0], 'leg_bias2': mae[1], 'std': estd,
                 'race_corr': float(np.mean(corr[2:])),
                 'race_corrs': tuple(float(x) for x in corr[2:])}
        return vloss, extra

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            rc = '/'.join(f'{x:+.3f}' for x in extra['race_corrs'])
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"loss(c/l/r)={extra['candle_loss']:.3f}/{extra['leg_loss']:.3f}/"
                  f"{extra['race_loss']:.3f} "
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
                           leg_k=2, mse_weight=1.0, target_reserve=None,
                           race_w=0.25, race_cap=2.0,
                           race_levels=RACE_LEVELS, **_ignore):
    """Train an encoder on NextLeg plus a future-only ordered adverse path curve."""
    t = _NextLegRaceTrainer(
        big, train_starts, val_starts,
        horizons=horizons, context_lengths=context_lengths, new_channels=new_channels,
        model_id=model_id, backbone_ckpt=backbone_ckpt, clamp=clamp, leg_cap=leg_cap,
        leg_w=leg_w, leg_k=leg_k, mse_weight=mse_weight, race_w=race_w,
        target_reserve=target_reserve,
        race_cap=race_cap, race_levels=race_levels, epochs=epochs,
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard)
    return t.fit()
