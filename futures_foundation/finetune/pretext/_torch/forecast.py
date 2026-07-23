"""Stage-2 trainer: MULTI-HORIZON, VARIABLE-CONTEXT candle seq2seq (ANTI-SHORTCUT). Predict the
future CANDLE (OHLCV) at each horizon as a move FROM 'now' (context-standardized), so 'copy now'
== predict-zero (punished). Reports per-horizon skill so we can see whether the far horizons learn."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _encode_channels, _apply_control, _gather_batch, BaseTrainer


class MultiHorizonForecastNet(nn.Module):
    """Mantis encoder + channel adapter + a MULTI-HORIZON candle decoder. A variable-length context
    is encoded; the decoder predicts the future CANDLE (OHLCV) at EACH horizon -> [B, C, n_horizons]."""

    def __init__(self, C=5, new_channels=8, horizons=(5, 10, 20, 25),
                 model_id='paris-noah/Mantis-8M', aux_dim=0):
        super().__init__()
        from .common import load_mantis
        from mantis.adapters import LinearChannelCombiner
        self.encoder = load_mantis(model_id)
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        self.new_c = min(new_channels, C)
        self.adapter = LinearChannelCombiner(num_channels=C, new_num_channels=self.new_c)
        self.C, self.horizons = C, tuple(int(h) for h in horizons)
        self.nH = len(self.horizons)
        emb = hidden * self.new_c
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, C * self.nH))
        # OPTIONAL aux head sized by the forecast OBJECTIVE (e.g. direction logits) — a LINEAR task head
        # off the same embedding, so the objective's gradient shapes the ENCODER. Discarded after
        # training. aux_dim=0 = candle-only (original stage-2).
        self.aux_head = nn.Linear(emb, aux_dim) if aux_dim > 0 else None

    def embed(self, x):                                   # [B,C,L] -> [B, new_c*hidden]
        a = self.adapter(x)
        return _encode_channels(self.encoder, a)

    def forward(self, ctx):                               # -> (candles [B,C,nH], aux [B,aux_dim] or None)
        e = self.embed(ctx)
        candles = self.decoder(e).view(-1, self.C, self.nH)
        return candles, (self.aux_head(e) if self.aux_head is not None else None)


class _ForecastTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
                 new_channels=8, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                 compile_model=False, clamp=10.0, objective='candle_mse', dir_weight=0.0,
                 dir_close_ch=3, **base):
        super().__init__(big, tr, va, **base)
        self.hlist = [int(h) for h in horizons]
        self.clens = [int(x) for x in context_lengths]
        self.max_ctx, self.h_max = max(self.clens), max(self.hlist)
        self.parent = self.max_ctx + self.h_max
        self.h_off = torch.as_tensor([h - 1 for h in self.hlist], dtype=torch.long, device=self.dev)
        self.clens_t = torch.as_tensor(self.clens, dtype=torch.long, device=self.dev)
        self.new_channels, self.model_id = new_channels, model_id
        self.backbone_ckpt, self.compile_model, self.clamp = backbone_ckpt, compile_model, clamp
        self.C = int(self.big_t.shape[1])
        from .forecast_objectives import get_forecast_objective
        self.obj = get_forecast_objective(objective)                # pluggable forecast supervision
        self.dir_weight = float(dir_weight)                         # aux-head loss weight (0 = candle-only)
        self.close_ch = min(int(dir_close_ch), self.C - 1)          # OHLCV close = index 3

    def build_net(self):
        net = MultiHorizonForecastNet(C=self.C, new_channels=self.new_channels, horizons=self.hlist,
                                      model_id=self.model_id,
                                      aux_dim=self.obj.aux_dim(len(self.hlist))).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location='cpu'))
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
        b_idx = self.sample_indices(starts, generator=gen)
        w = _gather_batch(self.big_t, starts, b_idx, self.parent)     # [B,C,max_ctx+h_max] real
        L = int(self.clens_t[torch.randint(0, len(self.clens_t), (1,), device=self.dev,
                                           generator=gen)].item())
        ctx_raw = w[:, :, self.max_ctx - L:self.max_ctx]             # [B,C,L] context ending at 'now'
        fut_raw = w[:, :, self.max_ctx:]                            # [B,C,h_max] future candles
        m = ctx_raw.mean(2, keepdim=True); s = ctx_raw.std(2, keepdim=True) + 1e-6
        cs = ((ctx_raw - m) / s).clamp(-self.clamp, self.clamp)     # standardized context (input)
        fs = ((fut_raw - m) / s).clamp(-self.clamp, self.clamp)     # standardized future candles
        target = fs[:, :, self.h_off] - cs[:, :, -1:]               # [B,C,nH] move FROM now (anti-shortcut)
        return _apply_control(cs, self.control), target             # corrupt ONLY the input

    def compute_loss(self, batch):
        model_ctx, target = batch
        candles, aux = self.net(model_ctx)
        return self.obj.loss(candles, aux, target, self.close_ch, self.dir_weight)

    @torch.no_grad()
    def val_eval(self):
        from .forecast_objectives import dir_acc as _dir_acc
        self.net.eval(); tot_obj = 0.0; tot_mse = 0.0; ptot = 0.0
        toth = torch.zeros(len(self.hlist), device=self.dev)        # per-horizon: is 20/25 learning?
        ptoth = torch.zeros(len(self.hlist), device=self.dev)
        dacc = torch.zeros(len(self.hlist), device=self.dev)        # per-horizon directional accuracy
        nb = min(20, max(1, len(self.va) // self.batch))
        # Fixed validation draws make early stopping compare model epochs rather than
        # different random windows/context lengths. Training keeps its independent RNG.
        vgen = torch.Generator(device=self.dev); vgen.manual_seed(20260704)
        embed_ctx = None
        for _ in range(nb):
            mc, tg = self.make_batch(self.va, gen=vgen)
            if embed_ctx is None:
                embed_ctx = mc
            with self.amp_ctx():
                candles, aux = self.net(mc)                         # net ALWAYS returns (candles, aux)
                # Checkpoint selection must use the SAME objective as training. Previously every
                # forecast variant (direction/quantile/bins/mixture) early-stopped on candle MSE,
                # silently discarding the epoch best at its added supervision.
                obj_loss = self.obj.loss(candles, aux, tg, self.close_ch, self.dir_weight)
            se = (candles.float() - tg) ** 2
            tot_obj += float(obj_loss); tot_mse += float(se.mean()); ptot += float((tg ** 2).mean())
            toth += se.mean(dim=(0, 1)); ptoth += (tg ** 2).mean(dim=(0, 1))
            dacc += _dir_acc(candles, tg, self.close_ch)            # universal (comparable across objectives)
        estd = float(self.net.embed(embed_ctx).std(0).mean())
        self.net.train()
        vloss, candle_mse, ploss = tot_obj / nb, tot_mse / nb, ptot / nb
        skill = float(1.0 - candle_mse / ploss) if ploss > 1e-12 else 0.0
        skill_h = (1.0 - toth / ptoth.clamp_min(1e-12)).cpu().tolist()
        dir_h = (dacc / nb).cpu().tolist()                          # dir_acc>0.5 = learning direction
        return vloss, {'candle_mse': candle_mse, 'persist_loss': ploss, 'skill': skill,
                       'skill_per_h': dict(zip(self.hlist, skill_h)),
                       'dir_acc': float(sum(dir_h) / len(dir_h)),
                       'dir_acc_per_h': dict(zip(self.hlist, dir_h)), 'std': estd}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            ph = ' '.join(f"h{h}={s:+.2f}" for h, s in extra['skill_per_h'].items())
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"mse={extra['candle_mse']:.4f} persist={extra['persist_loss']:.4f} "
                  f"skill={extra['skill']:+.3f} [{ph}] "
                  f"dir={extra['dir_acc']:.3f} emb_std={extra['std']:.4f}{'  *' if improved else ''}",
                  flush=True)


def train_ssl_forecast(big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
                       context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
                       steps_per_epoch=200, batch=512, lr=1e-4, weight_decay=0.05, patience=8,
                       device=None, model_id='paris-noah/Mantis-8M', backbone_ckpt=None,
                       compile_model=False, control='real', seed=0, amp_dtype='fp16',
                       grad_clip=1.0, clamp=10.0, verbose=True,
                       ckpt_path=None, resume=False, freeze_encoder_layers=0,
                       objective='candle_mse', dir_weight=0.0, dir_close_ch=3,
                       lora_r=0, lora_alpha=16.0, lora_dropout=0.0,
                       log_every_steps=25, **_ignore):
    """Multi-horizon / variable-context candle seq2seq. Returns (best_encoder_state, history) with
    'val_loss', 'persist_loss', 'skill', 'skill_per_h', 'std' (+ 'dir_acc' if dir_weight>0). Warm-start
    from stage-1. OPTIONAL: dir_weight>0 adds a direction-head BCE term (sign of the fwd close move) to
    the candle MSE — trains the encoder to be direction-aware (WR-relevant); 0 = original behavior."""
    return _ForecastTrainer(big, train_starts, val_starts, horizons=horizons,
                            context_lengths=context_lengths, new_channels=new_channels,
                            model_id=model_id, backbone_ckpt=backbone_ckpt, compile_model=compile_model,
                            clamp=clamp, epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
                            lr=lr, weight_decay=weight_decay, patience=patience, device=device,
                            seed=seed, grad_clip=grad_clip, amp_dtype=amp_dtype, verbose=verbose,
                            control=control, ckpt_path=ckpt_path, resume=resume,
                            freeze_encoder_layers=freeze_encoder_layers, objective=objective,
                            dir_weight=dir_weight, dir_close_ch=dir_close_ch, lora_r=lora_r,
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                            log_every_steps=log_every_steps).fit()
