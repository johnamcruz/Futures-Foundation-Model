"""Shared TORCH layer for the SSL pretext trainers.

Window helpers (encode / standardize / control-corrupt / gather), the frozen-embedding + ONNX
primitives, and a BaseTrainer that owns the SHARED training loop (epoch loop, AMP, grad-clip,
cosine LR, early-stop on val loss, best-ENCODER-state snapshot). Each pretext trainer
(mask / forecast / contrastive) subclasses BaseTrainer and provides only build_net / make_batch /
compute_loss / val_eval — no copied loops. torch imports live under this subpackage only (loaded
lazily), so the orchestrator + task registry stay torch-free.
"""
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_identity import MANTIS_MODEL_REVISION


def load_mantis(model_id='paris-noah/Mantis-8M'):
    """Load the exact public base used by training and every vanilla probe baseline."""
    from mantis.architecture import Mantis8M
    revision = os.environ.get('MANTIS_MODEL_REVISION', MANTIS_MODEL_REVISION)
    return Mantis8M.from_pretrained(model_id, revision=revision)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _enc(encoder, x1):
    """Encode one channel [B,1,L] -> [B, hidden], interpolating the window to Mantis's native
    seq_len (512) first so it ALWAYS sees its pretrained patch size."""
    L = int(getattr(encoder, 'seq_len', 512))
    if x1.shape[-1] != L:
        x1 = F.interpolate(x1, size=L, mode='linear', align_corners=False)
    return encoder(x1)


def _encode_channels(encoder, x):
    """Encode ``[B,C,L]`` with one Mantis call and concatenate channel embeddings.

    Mantis treats every input series independently, so folding channels into the
    batch is mathematically equivalent to calling the encoder C times and
    concatenating the results.  It removes C-1 Python/MPS dispatch round trips,
    which is material for local training, while preserving output order.
    """
    B, C, L = x.shape
    emb = _enc(encoder, x.reshape(B * C, 1, L))
    return emb.reshape(B, C * emb.shape[1])


def _standardize(x):                                     # per-window per-channel z-score
    m = x.mean(dim=2, keepdim=True)
    s = x.std(dim=2, keepdim=True)
    return (x - m) / (s + 1e-6)


def _time_shuffle(x):
    """Permute the time axis independently per sample -> destroys temporal order, keeps the exact
    value set. Used for the SHUFFLE control."""
    B, C, T = x.shape
    perm = torch.argsort(torch.rand(B, T, device=x.device), 1)
    return torch.gather(x, 2, perm[:, None, :].expand(B, C, T))


def _apply_control(x, control):
    """Corrupt ONLY the model INPUT per the apples-to-apples control: 'shuffle' scrambles the time
    axis, 'random' replaces with noise, else (real) passes through. The target/trend-key is always
    computed from the REAL context by the caller -> real must beat shuffle/random. Shared by every
    pretext trainer so the control logic lives in one place."""
    if control == 'shuffle':
        return _time_shuffle(x)
    if control == 'random':
        return torch.randn_like(x)
    return x


def _gather_batch(big, starts, b_idx, length):
    """big [T, C] -> windows [B, C, length] for the start positions starts[b_idx]."""
    s = starts[b_idx]                                    # [B]
    rows = s[:, None] + torch.arange(length, device=big.device)[None, :]   # [B, length]
    return big[rows].permute(0, 2, 1).contiguous()       # [B, C, length]


def _plain_encoder_state(state):
    """Accept an ordinary Mantis checkpoint or a composite related-series checkpoint."""
    from .related_series import plain_encoder_state
    return plain_encoder_state(state)


# ----------------------------------------------------------------- frozen embedding (probe / cache)
@torch.no_grad()
def embed_encoder(big, starts, seq, *, ckpt=None, model_id='paris-noah/Mantis-8M',
                  device=None, batch=512, max_windows=20000, seed=0):
    """Frozen ENCODER-ONLY embeddings of clean (per-window z-scored) windows — the quantity that
    transfers downstream via backbone_ckpt. Each OHLCV channel is encoded independently and
    concatenated -> [M, C*hidden]. ckpt=None -> vanilla Mantis (probe baseline)."""
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = load_mantis(model_id)
    if ckpt:
        enc.load_state_dict(_plain_encoder_state(torch.load(ckpt, map_location='cpu')))
    enc = enc.to(dev).eval()
    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    s = np.asarray(starts, np.int64)
    if len(s) > max_windows:
        s = np.sort(np.random.default_rng(seed).choice(s, max_windows, replace=False))
    s_t = torch.as_tensor(s, device=dev)
    out = []
    for b in range(0, len(s_t), batch):
        win = _gather_batch(big_t, s_t, torch.arange(b, min(b + batch, len(s_t)), device=dev), seq)
        win = _standardize(win)                          # [B, C, seq]
        emb = _encode_channels(enc, win)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32), s


def embed_window_chunks(chunks, *, ckpt=None, model_id='paris-noah/Mantis-8M', device=None,
                        batch=512):
    """Yield frozen embeddings for an iterable of bounded ``[N,C,seq]`` chunks.

    The encoder is loaded once and retained across chunks.  This is the safe path
    for multi-million-window diagnostics: callers can stream source windows and
    write each result to a memmap without materializing either full array in RAM.
    """
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = load_mantis(model_id)
    if ckpt:
        enc.load_state_dict(_plain_encoder_state(torch.load(ckpt, map_location='cpu')))
    enc = enc.to(dev).eval()

    def _iterator():
        with torch.no_grad():
            for windows in chunks:
                X = torch.as_tensor(np.asarray(windows, np.float32))
                out = []
                for b in range(0, len(X), batch):
                    w = _standardize(X[b:b + batch].to(dev))
                    emb = _encode_channels(enc, w)
                    out.append(emb.float().cpu().numpy())
                yield (np.concatenate(out) if out else
                       np.zeros((0, 0), np.float32))

    return _iterator()


def embed_windows(windows, *, ckpt=None, model_id='paris-noah/Mantis-8M', device=None, batch=512):
    """Frozen ENCODER-ONLY embeddings of pre-extracted windows [N, C, seq] -> [N, C*hidden]. The
    head-only/cached downstream primitive: backbone frozen, embed ONCE, then a cheap head trains
    on the cache."""
    return next(embed_window_chunks((windows,), ckpt=ckpt, model_id=model_id,
                                    device=device, batch=batch))


class _EncoderONNX(nn.Module):
    """ONNX-exportable wrapper reproducing embed_windows EXACTLY: per-window standardize ->
    per-channel interpolate to native length -> encode -> concat. Raw window [B,C,seq] in,
    embedding [B, C*hidden] out (standardize baked in so the bot feeds RAW OHLCV).

    Channels are folded into the BATCH ([B,C,seq] -> [B*C,1,seq]) so the encoder appears ONCE
    in the traced graph. The per-channel python loop traced C copies of the transformer into
    the ONNX (5x nodes, defeated ORT fusion — profiled at 22% Transpose / 57% shape-plumbing).
    reshape(B, C*hidden) on the [B*C, hidden] output reproduces cat(dim=-1) block order exactly."""

    def __init__(self, encoder, C):
        super().__init__()
        self.encoder = encoder
        self.C = int(C)

    def forward(self, w):                                     # [B, C, seq] raw OHLCV
        w = _standardize(w)
        return _encode_channels(self.encoder, w)


def _ort_optimize_graph(path):
    """Offline onnxruntime graph optimization: constant-fold the tracer's shape-plumbing
    (Shape/Gather/Unsqueeze/Constant), eliminate redundant Transposes, fuse LayerNorm/attention.
    Saves the optimized graph over `path`. EXTENDED level emits some ORT-specific fused ops —
    fine for our serve path (the bot runs onnxruntime), and it's the level that kills the
    Transpose overhead. Best-effort: on any failure the un-optimized (still correct) file stands."""
    try:
        import tempfile

        import onnxruntime as ort
        fd, tmp = tempfile.mkstemp(suffix='.onnx', dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = tmp
        ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        os.replace(tmp, path)
        print(f"[onnx] ORT-optimized encoder graph saved -> {path}", flush=True)
    except Exception as e:                                    # pragma: no cover
        print(f"[onnx] ORT offline optimization skipped: {e}", flush=True)


def export_encoder_onnx(path, *, ckpt=None, C=5, seq=64,
                        model_id='paris-noah/Mantis-8M', device='cpu'):
    """Export the frozen encoder (standardize+interp+encode) to ONNX: raw window [B,C,seq] ->
    embedding [B, C*hidden]. Matches embed_windows numerically (parity-tested).

    The graph holds ONE encoder (channels batched, see _EncoderONNX) and is post-processed by
    onnxruntime offline optimization (constant folding + transpose elimination + fusion)."""
    enc = load_mantis(model_id)
    if ckpt:
        enc.load_state_dict(_plain_encoder_state(torch.load(ckpt, map_location='cpu')))
    enc = enc.to(device).eval()
    m = _EncoderONNX(enc, C).to(device).eval()
    dummy = torch.randn(2, int(C), int(seq), device=device)   # >1 row so std is well-defined
    # Mantis calls torch.diff internally (aten::diff has no ONNX symbolic) -> swap for an
    # equivalent slice-subtract during export so the traced graph is exportable. Restored after.
    _orig_diff = torch.diff

    def _diff_traceable(x, n=1, dim=-1, *, axis=None, prepend=None, append=None):
        d = axis if axis is not None else dim
        for _ in range(int(n)):
            x = x.narrow(d, 1, x.size(d) - 1) - x.narrow(d, 0, x.size(d) - 1)
        return x
    torch.diff = _diff_traceable
    try:
        torch.onnx.export(m, dummy, path, input_names=['window'], output_names=['embedding'],
                          dynamic_axes={'window': {0: 'batch'}, 'embedding': {0: 'batch'}},
                          do_constant_folding=True,
                          opset_version=17, dynamo=False)      # legacy tracer (dynamo chokes on Mantis)
    finally:
        torch.diff = _orig_diff
    _ort_optimize_graph(path)
    return path


def _atomic_save(obj, path):
    """Crash-safe save: write to a temp file then os.replace (atomic) -> a Colab disconnect can
    never leave a half-written checkpoint."""
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_meta(path, best_val, epoch):
    import json
    tmp = str(path) + '.meta.json.tmp'
    with open(tmp, 'w') as f:
        json.dump({'best_val': float(best_val), 'epoch': int(epoch)}, f)
    os.replace(tmp, str(path) + '.meta.json')


def _read_meta_best(path):
    import json
    mp = str(path) + '.meta.json'
    if os.path.exists(mp):
        try:
            return float(json.load(open(mp)).get('best_val', 1e18))
        except Exception:
            return 1e18
    return 1e18


def _freeze_encoder(encoder, n_layers):
    """Anti-forgetting: freeze the input tokenizer + the first n_layers transformer blocks of a
    Mantis encoder when REFINING a warm-started encoder, so the bulk of learned structure can't
    drift. Later blocks + the channel adapter (+ projection) stay trainable, so the embedding can
    still adapt. n_layers<=0 -> no freeze. Robust to V1 (vit_unit) / V2 (transf_unit) paths."""
    if not n_layers or int(n_layers) <= 0:
        return 0
    n = int(n_layers)
    tok = getattr(encoder, 'tokgen_unit', None)              # input patch/scalar tokenizer (general)
    if tok is not None:
        for p in tok.parameters():
            p.requires_grad = False
    unit = getattr(encoder, 'vit_unit', None) or getattr(encoder, 'transf_unit', None)
    tr = getattr(unit, 'transformer', None) if unit is not None else None
    layers = getattr(tr, 'layers', None) if tr is not None else None
    frozen = 0
    if layers is not None:
        for blk in list(layers)[:n]:
            for p in blk.parameters():
                p.requires_grad = False
            frozen += 1
    return frozen


# ================================================================= BASE TRAINER (shared loop)
class BaseTrainer:
    """Shared SSL training loop for every pretext. Subclass and implement:
      * build_net()            -> set self.net (+ warm-start from backbone_ckpt)
      * make_batch(starts)     -> a batch object for compute_loss / accumulated over steps
      * compute_loss(batch)    -> scalar loss (called inside the AMP context)
      * val_eval()             -> (val_loss: float, extra: dict incl 'std' for the collapse guard)
    fit() owns: the epoch/step loop, AMP (cuda) + grad-clip, cosine LR, early-stop on val_loss,
    and the best-ENCODER-state snapshot. Subclasses may override make_optimizer / log_line."""

    def __init__(self, big, train_starts, val_starts, *, epochs=60, steps_per_epoch=200, batch=512,
                 lr=1e-4, weight_decay=0.05, patience=8, device=None, seed=0, grad_clip=None,
                 amp=True, amp_dtype='fp16', verbose=True, control='real',
                 ckpt_path=None, resume=False, freeze_encoder_layers=0, std_guard=0.0,
                 lora_r=0, lora_alpha=16.0, lora_dropout=0.0, log_every_steps=25):
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        self.ckpt_path, self.resume = ckpt_path, resume    # progressive best-save + resume (real run only)
        self.freeze_encoder_layers = freeze_encoder_layers  # anti-forgetting: freeze first N enc layers
        self.std_guard = float(std_guard or 0.0)           # >0: HALT when emb_std exceeds it (drift guard)
        self.lora_r, self.lora_alpha = int(lora_r or 0), float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.log_every_steps = max(0, int(log_every_steps or 0))
        self.dev = device or ('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        self.big_t = torch.as_tensor(np.asarray(big, np.float32), device=self.dev)
        self._tr_source_values, self._tr_source_groups = self._source_arrays(train_starts)
        self._va_source_values, self._va_source_groups = self._source_arrays(val_starts)
        self.tr = torch.as_tensor(self._tr_source_values, device=self.dev)
        self.va = torch.as_tensor(self._va_source_values, device=self.dev)
        self._tr_sampling = self._sampling_spec(self._tr_source_groups)
        self._va_sampling = self._sampling_spec(self._va_source_groups)
        self.epochs, self.steps_per_epoch, self.batch = epochs, steps_per_epoch, batch
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.grad_clip, self.verbose, self.control = grad_clip, verbose, control
        self.use_amp = (self.dev == 'cuda') and amp                # contrastive runs fp32 (amp=False)
        _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
        self.amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if self.use_amp else (lambda: _nullctx())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.net = None

    @staticmethod
    def _source_arrays(source):
        """Return ordinary starts plus optional source ids from a WindowStartPool.

        This is deliberately duck-typed so the shared torch layer does not import
        the torch-free orchestrator. Plain arrays preserve historical sampling.
        """
        values = np.asarray(getattr(source, 'starts', source), dtype=np.int64)
        groups = getattr(source, 'group_ids', None)
        if groups is not None:
            groups = np.asarray(groups, dtype=np.int32)
            if len(groups) != len(values):
                raise ValueError('sampling group ids are not aligned with window starts')
        return values, groups

    def _sampling_spec(self, groups):
        """Compact stream-first sampling plan for one start array.

        Starts are assembled stream-by-stream, so each source occupies one contiguous
        range. Selecting a range uniformly and then an offset within it reproduces
        Chronos's choose-source-then-example mixture without an O(N) multinomial over
        every window at every optimizer step.
        """
        if groups is None or not len(groups):
            return None
        changes = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
        offsets = changes[:-1].astype(np.int64)
        counts = np.diff(changes).astype(np.int64)
        run_groups = groups[offsets]
        if len(np.unique(run_groups)) != len(run_groups):
            raise ValueError('uniform-stream groups must occupy contiguous start ranges')
        return (torch.as_tensor(offsets, device=self.dev),
                torch.as_tensor(counts, device=self.dev))

    def _sampling_for_values(self, values, split):
        """Inherit source membership after a pretext filters legal start anchors."""
        vals = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values,
                          dtype=np.int64)
        source_values = getattr(self, f'_{split}_source_values')
        source_groups = getattr(self, f'_{split}_source_groups')
        if source_groups is None or not len(vals):
            return None
        idx = np.searchsorted(source_values, vals)
        if (idx >= len(source_values)).any() or not np.array_equal(source_values[idx], vals):
            raise ValueError(f'{split} pretext anchors are not a subset of legal window starts')
        return self._sampling_spec(source_groups[idx])

    def _replace_start_pool(self, split, values):
        """Replace a start tensor while retaining its original source-mixture policy."""
        vals = np.asarray(values, dtype=np.int64)
        tensor = torch.as_tensor(vals, device=self.dev)
        setattr(self, split, tensor)
        setattr(self, f'_{split}_sampling', self._sampling_for_values(vals, split))
        return tensor

    def sample_indices(self, starts, *, generator=None, sampling=None, size=None):
        """Sample batch indices, stream-first when an opt-in mixture is attached."""
        generator = generator or self.gen
        size = self.batch if size is None else int(size)
        if sampling is None:
            sampling = (self._tr_sampling if starts is self.tr else
                        self._va_sampling if starts is self.va else None)
        if sampling is None:
            return torch.randint(0, len(starts), (size,), device=self.dev, generator=generator)
        offsets, counts = sampling
        group_idx = torch.randint(0, len(offsets), (size,), device=self.dev, generator=generator)
        # All production streams have far fewer than 2**24 starts, so float32
        # resolution makes the standard floor(U*N) draw effectively uniform on MPS/CUDA.
        within = torch.floor(torch.rand(size, device=self.dev, generator=generator)
                             * counts[group_idx]).to(torch.long)
        return offsets[group_idx] + within

    # ---- hooks (subclass implements) ----
    def build_net(self):
        raise NotImplementedError

    def make_batch(self, starts):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def val_eval(self):
        raise NotImplementedError

    def make_optimizer(self):
        return torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],
                                 lr=self.lr, weight_decay=self.weight_decay)

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    def _encoder(self):
        net = self.net
        return net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder

    def _stateful_net(self):
        """Unwrapped task network used only for crash-resume trainer state."""
        return self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod

    def _params(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def snapshot_state(self):
        """Checkpoint hook; related-series trainers override to retain their fusion module."""
        from .lora import merged_state_dict
        return merged_state_dict(self._encoder())

    def load_snapshot_state(self, state):
        """Inverse checkpoint hook used by crash-safe resume."""
        from .lora import load_plain_state_dict
        load_plain_state_dict(self._encoder(), state)

    def fit(self):
        """Run the shared loop -> (best_encoder_state, history). Progressively saves the best
        ENCODER to ckpt_path (crash-safe) and resumes from it — real run only (controls never
        touch the checkpoint). freeze_encoder_layers anchors early layers against drift."""
        self.build_net()
        from .lora import inject_mantis_lora
        if self.lora_r:
            stats = inject_mantis_lora(self._encoder(), rank=self.lora_r,
                                       alpha=self.lora_alpha, dropout=self.lora_dropout)
            if self.verbose:
                print(f"  [lora] r={self.lora_r} alpha={self.lora_alpha:g} "
                      f"modules={stats['modules']} trainable={stats['trainable']:,}/"
                      f"{stats['total']:,} ({stats['percent']:.2f}%)", flush=True)
        save_ok = bool(self.ckpt_path) and self.control == 'real'   # controls never touch the ckpt
        best, best_state, resume_payload, start_epoch = 1e18, None, None, 0
        trainer_path = str(self.ckpt_path) + '.trainer.pt' if save_ok else None
        if self.resume and save_ok and os.path.exists(self.ckpt_path):
            if trainer_path and os.path.exists(trainer_path):
                resume_payload = torch.load(trainer_path, map_location='cpu')
                self._stateful_net().load_state_dict(resume_payload['model_state'])
                best = float(resume_payload['best_val'])
                start_epoch = int(resume_payload['epoch']) + 1
                best_state = self.snapshot_state()
                if 'generator_state' in resume_payload:
                    self.gen.set_state(resume_payload['generator_state'])
                if self.verbose:
                    print(f"  [resume] restored full trainer state from epoch "
                          f"{start_epoch}/{self.epochs} (best_val={best:.4f})", flush=True)
            else:
                # Backward-compatible recovery for checkpoints created before full trainer
                # sidecars existed. The encoder is safe, but task heads restart from scratch.
                self.load_snapshot_state(torch.load(self.ckpt_path, map_location='cpu'))
                best = _read_meta_best(self.ckpt_path)
                best_state = self.snapshot_state()
                if self.verbose:
                    print(f"  [resume] loaded legacy encoder-only checkpoint {self.ckpt_path} "
                          f"(best_val={best:.4f}; task head restarted)", flush=True)
        nfz = _freeze_encoder(self._encoder(), self.freeze_encoder_layers)   # anti-forgetting
        if nfz and self.verbose:
            ntr = sum(p.requires_grad for p in self.net.parameters())
            print(f"  [freeze] tokenizer + first {nfz} encoder layers frozen ({ntr} trainable tensors)",
                  flush=True)
        opt = self.make_optimizer()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        if resume_payload is not None:
            opt.load_state_dict(resume_payload['optimizer_state'])
            sched.load_state_dict(resume_payload['scheduler_state'])
        bad = int(resume_payload.get('bad_epochs', 0)) if resume_payload else 0
        history = ([resume_payload['best_history_row']]
                   if resume_payload and resume_payload.get('best_history_row') else [])
        for ep in range(start_epoch, self.epochs):
            self.net.train(); tr_tot = 0.0; ep_started = time.monotonic()
            for step in range(self.steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                with self.amp_ctx():
                    loss = self.compute_loss(self.make_batch(self.tr))
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    self.scaler.step(opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    opt.step()
                tr_tot += float(loss.detach())
                completed = step + 1
                if (self.verbose and self.log_every_steps and
                        (completed == 1 or completed % self.log_every_steps == 0
                         or completed == self.steps_per_epoch)):
                    elapsed = time.monotonic() - ep_started
                    rate = completed / max(elapsed, 1e-9)
                    print(f"  [step] epoch={ep + 1}/{self.epochs} "
                          f"step={completed}/{self.steps_per_epoch} "
                          f"loss={tr_tot / completed:.4f} rate={rate:.2f}step/s "
                          f"elapsed={elapsed:.1f}s", flush=True)
            sched.step()
            if self.dev == 'cuda':
                torch.cuda.empty_cache()
            vloss, extra = self.val_eval()
            history.append({'epoch': ep, 'train_loss': tr_tot / self.steps_per_epoch,
                            'val_loss': vloss, **extra})
            if self.std_guard and extra.get('std', 0.0) > self.std_guard:
                # DRIFT GUARD: embedding std past the ceiling = the representation is drifting off
                # the data (the unanchored-discrimination failure mode). HALT NOW and do NOT save
                # this epoch — val often keeps micro-improving while drift bakes in, so waiting for
                # early-stop would keep crowning drifted epochs as "best".
                if self.verbose:
                    print(f"  [std-guard] emb_std {extra['std']:.3f} > {self.std_guard:.2f} at "
                          f"ep{ep} — HALTED (best checkpoint kept from before the breach)",
                          flush=True)
                break
            improved = vloss < best - 1e-5
            if improved:
                best, bad = vloss, 0
                best_state = self.snapshot_state()
                if save_ok:                                          # progressive best-save (crash-safe)
                    _atomic_save(best_state, self.ckpt_path)
                    _write_meta(self.ckpt_path, best, ep)
                    # The public checkpoint remains encoder-only. This separate sidecar retains
                    # the disposable task head plus optimizer/scheduler/RNG for a true resume.
                    _atomic_save({
                        'schema': 'ffm_ssl_trainer_resume_v1',
                        'model_state': self._stateful_net().state_dict(),
                        'optimizer_state': opt.state_dict(),
                        'scheduler_state': sched.state_dict(),
                        'generator_state': self.gen.get_state(),
                        'best_val': best,
                        'epoch': ep,
                        'bad_epochs': bad,
                        'best_history_row': history[-1],
                    }, trainer_path)
            else:
                bad += 1
            self.log_line(ep, tr_tot / self.steps_per_epoch, vloss, extra, improved)
            if bad >= self.patience:
                break
        if best_state is None:
            raise RuntimeError("training ended without a valid encoder checkpoint")
        return best_state, history
