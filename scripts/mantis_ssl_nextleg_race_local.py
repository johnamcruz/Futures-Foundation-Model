"""Local MPS rejection screen for Stage-2.8 NextLeg Race.

This is intentionally separate from the Colab launcher. It trains the same raw-candle,
future-only objective and preserves the same temporal reserve, but defaults to the production
ES/NQ 3-minute universe and a short run sized for a 16 GB M1. A local winner is only a candidate:
Probe Atlas and the anchored Pivot Trend workflow remain mandatory before a full Colab run.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from futures_foundation.finetune import ssl  # noqa: E402


def _csv_tuple(name, default):
    value = os.environ.get(name, default)
    return tuple(x.strip() for x in value.split(',') if x.strip())


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available; use the Colab launcher or a supported Apple GPU')

    data_dir = os.environ.get('DATA_DIR', str(ROOT / 'data'))
    warm_ckpt = os.environ.get(
        'WARM_CKPT', str(ROOT / 'checkpoints' / 'mantis_ssl_nextleg.pt'))
    out_path = os.environ.get(
        'OUT_PATH', str(ROOT / 'temp' / 'mantis_ssl_nextleg_race_local.pt'))
    tickers = _csv_tuple('TICKERS', 'ES,NQ')
    tfs = _csv_tuple('TFS', '3min')

    batch = int(os.environ.get('BATCH', '128'))
    epochs = int(os.environ.get('EPOCHS', '12'))
    steps = int(os.environ.get('STEPS', '100'))
    lr = float(os.environ.get('LR', '5e-5'))
    freeze = int(os.environ.get('FREEZE_ENCODER_LAYERS', '3'))
    race_w = float(os.environ.get('RACE_W', '0.10'))
    resume = os.environ.get('RESUME', '0') == '1'

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)
    if not os.path.isfile(warm_ckpt):
        raise FileNotFoundError(warm_ckpt)
    if os.path.abspath(out_path) == os.path.abspath(warm_ckpt):
        raise ValueError('OUT_PATH must not overwrite production NextLeg')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f'[local-race] MPS | {tickers} x {tfs} | batch={batch} epochs={epochs} steps={steps}')
    print(f'[local-race] warm={warm_ckpt}')
    print(f'[local-race] freeze={freeze} lr={lr:g} race_w={race_w:g} -> {out_path}')
    print('[local-race] holdout >= 2026-01-01 excluded; local result is a rejection screen only')

    verdict = ssl.loop_ssl(
        data_dir=data_dir, out_path=out_path, tickers=tickers, tfs=tfs,
        pretext='nextleg_race', backbone_ckpt=warm_ckpt,
        horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
        leg_k=2, leg_cap=256, leg_w=1.0, mse_weight=1.0,
        race_w=race_w, race_cap=2.0, race_levels=(0.25, 0.50, 0.75, 1.00),
        new_channels=3, batch=batch, epochs=epochs, steps_per_epoch=steps,
        lr=lr, weight_decay=0.0, patience=5, clamp=10.0, grad_clip=1.0,
        val_frac=0.1, holdout_start='2026-01-01', controls=(), probe=False,
        resume=resume, freeze_encoder_layers=freeze, device='mps', seed=0)

    print('\n[local-race] verdict')
    for key, value in verdict.items():
        if key not in ('history', 'epochs'):
            print(f'  {key:>24}: {value}')
    print(f'\nCandidate: {out_path}')
    print('Next: build its checkpoint-specific Probe Atlas cache; do not promote from SSL loss alone.')


if __name__ == '__main__':
    main()
