"""Domain-adapt Chronos-Bolt on futures data (self-supervised forecasting).

STRATEGY-AGNOSTIC generic infra: produces a futures-domain-adapted
`amazon/chronos-bolt-tiny` checkpoint by continuing its native forecasting
objective on our own bars. No strategy, no labels — it only makes the BACKBONE
better at futures price dynamics, which benefits every downstream
Chronos+XGBoost selection model. The deliverable is an A/B: the fine-tuned Bolt
vs vanilla Bolt on the honest-ruler walk-forward, to settle whether domain
adaptation improves embedding quality.

Distinct from the two sibling fine-tune paths in this package:
  - `finetune.py`      — SUPERVISED: backbone + classification head, CE on
                          strategy labels (task fine-tune).
  - `_ft/` (run_ft.py) — T5 forecasting domain-adapt (tokenizer/seq2seq path);
                          its `bolt.yaml` is scaffold-only, `--bolt` errors out.
  - THIS module        — BOLT forecasting domain-adapt. Bolt is patch-based and
                          tokenizer-FREE, so it needs a simpler sliding-window
                          collator, not the T5 data path. The model is already
                          trainable: `ChronosBoltModelForForecasting` is a
                          `T5PreTrainedModel` whose forward(context, target)
                          returns a quantile `loss` (chronos/chronos_bolt.py).

REPRESENTATION (matters for transfer): downstream `backbone.embed()` feeds
LOG-PRICE windows of length 128 (the *_chronos labelers use `lp=np.log(c)`,
`C.append(lp[i-CTX+1:i+1])`). So we domain-adapt on the SAME representation —
context = log(close) windows of `context_length` (default 128 to match embed
exactly) — so the encoder adapts to precisely what we later embed.

Torch/chronos are imported INSIDE run() (lazy) — never at module top — mirroring
backbone.py so the package stays import-safe in torch-free contexts.

CLI:
    python3 -m pipelines.chronos.bolt_finetune                 # 6tk, 3m+5m, 20k steps
    python3 -m pipelines.chronos.bolt_finetune --smoke         # 2-step sanity
    python3 -m pipelines.chronos.bolt_finetune \
        --tickers ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN --tfs 3min,5min  # widest domain
Then (per the wiring gap — NOT auto-applied — see backbone.stamp_active_source):
    export CHRONOS_FT_CKPT=<printed path>
    python3 colabs/supertrend_chronos.py                       # A/B vs vanilla
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# pipelines/chronos/bolt_finetune.py -> parents: [chronos, pipelines, REPO_ROOT]
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / 'data'
OUT_DIR = _ROOT / 'temp' / 'chronos_bolt_ft'
MODEL_ID = 'amazon/chronos-bolt-tiny'

# Downstream signal models run on these 6 (equity + metal). data/ also has
# CL/ZB/ZN (5min-only) — pass them via --tickers for wider domain coverage.
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']


def _build_windows(context_length, prediction_length, stride, tfs, tickers,
                   months):
    """Sliding (context, target) windows over log(close), per ticker/tf.

    Returns (contexts, targets) float32:
      contexts: [N, context_length]    — log-price history (matches embed input)
      targets : [N, prediction_length] — the next prediction_length log-prices
    Causal by construction (target is strictly future bars; windows never cross
    ticker/tf boundaries). Missing ticker/tf files are skipped, not fatal — so
    e.g. CL/ZB/ZN (5min-only) contribute only their 5min windows.
    """
    ctxs, tgts = [], []
    span = context_length + prediction_length
    for tk in tickers:
        for tf in tfs:
            p = DATA_DIR / f'{tk}_{tf}.csv'
            if not p.exists():
                print(f"  skip {tk}_{tf} (no file)")
                continue
            df = pd.read_csv(p, usecols=['datetime', 'close'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            if months and months > 0:
                df = df[df['datetime'] >= df['datetime'].max()
                        - pd.DateOffset(months=months)]
            lp = np.log(df['close'].to_numpy(np.float64))
            lp = lp[np.isfinite(lp)]
            n = len(lp)
            if n < span:
                continue
            idx = np.arange(0, n - span + 1, stride)
            for i in idx:
                ctxs.append(lp[i:i + context_length])
                tgts.append(lp[i + context_length:i + span])
            print(f"  {tk}_{tf}: {len(idx)} windows ({n:,} bars)")
    if not ctxs:
        raise RuntimeError("no windows built — check data/ and --tfs")
    return np.asarray(ctxs, np.float32), np.asarray(tgts, np.float32)


def run(steps=20000, context_length=128, prediction_length=64, stride=32,
        tfs=('3min', '5min'), tickers=TICKERS, months=0, lr=1e-3,
        batch_size=32, save_steps=None, smoke=False):
    """Domain-adapt bolt-tiny on the given futures bars; save checkpoint.
    Returns the checkpoint Path (HF format — backbone.embed loads it directly)."""
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments
    from chronos import BaseChronosPipeline

    if smoke:
        steps, months, stride = 2, 1, 256
    save_steps = save_steps or steps

    print("=== Chronos-Bolt domain-adapt (futures, self-supervised) ===")
    print(f"  base model   : {MODEL_ID}")
    print(f"  representation: log(close)  ctx={context_length}  "
          f"pred={prediction_length}  stride={stride}")
    print(f"  tickers={list(tickers)}  tfs={list(tfs)}  "
          f"months={months or 'ALL'}  steps={steps}")

    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID)
    model = pipe.model                       # ChronosBoltModelForForecasting
    cfg_pred = model.chronos_config.prediction_length
    if prediction_length > cfg_pred:
        print(f"  ⚠ requested pred={prediction_length} > model native "
              f"{cfg_pred}; clamping to {cfg_pred}")
        prediction_length = cfg_pred

    print("  building windows ...")
    ctxs, tgts = _build_windows(context_length, prediction_length, stride,
                                tfs, tickers, months)
    print(f"  total windows: {len(ctxs):,}  "
          f"(ctx {ctxs.shape}, tgt {tgts.shape})")

    class WinDS(Dataset):
        def __len__(self):
            return len(ctxs)

        def __getitem__(self, i):
            return {'context': torch.from_numpy(ctxs[i]),
                    'target': torch.from_numpy(tgts[i])}

    def collate(batch):
        return {'context': torch.stack([b['context'] for b in batch]),
                'target': torch.stack([b['target'] for b in batch])}

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUT_DIR / 'out'),
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=max(1, steps // 100),
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,         # keep 'context'/'target' keys
        use_cpu=(device == 'cpu'),
        fp16=False, bf16=False,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=WinDS(), data_collator=collate)
    print(f"  device={device} | training {steps} steps ...")
    trainer.train()

    final = OUT_DIR / 'checkpoint-final'
    model.save_pretrained(str(final))
    print(f"\nDONE — domain-adapted Bolt checkpoint: {final}")
    print(f"\n  ⚠ To A/B vs vanilla Bolt, export BEFORE the walk-forward:")
    print(f"\n    export CHRONOS_FT_CKPT={final}\n")
    print(f"    python3 colabs/supertrend_chronos.py   # REAL vs SHUFFLE on FT-Bolt")
    print(f"  Then unset CHRONOS_FT_CKPT and re-run for the vanilla baseline.")
    print(f"  ⚠ Without the export, downstream silently uses vanilla {MODEL_ID}.")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--context-length', type=int, default=128,
                    help='must match downstream embed CTX (128) for cleanest transfer')
    ap.add_argument('--prediction-length', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32, help='bars between windows')
    ap.add_argument('--tfs', default='3min,5min')
    ap.add_argument('--tickers', default=','.join(TICKERS))
    ap.add_argument('--months', type=int, default=0, help='0 = all available')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(steps=a.steps, context_length=a.context_length,
        prediction_length=a.prediction_length, stride=a.stride,
        tfs=tuple(a.tfs.split(',')), tickers=tuple(a.tickers.split(',')),
        months=a.months, lr=a.lr, batch_size=a.batch_size, smoke=a.smoke)


if __name__ == '__main__':
    main()
