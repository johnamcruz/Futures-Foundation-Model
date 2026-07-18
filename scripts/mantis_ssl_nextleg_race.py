# ==============================================================================
# MANTIS SSL STAGE 2.8 — NEXT-LEG + ORDERED FUTURE PATH RACE (Colab GPU)
# ==============================================================================
# New additive experiment. Production nextleg code/checkpoint are never overwritten.
# The target is a four-point, candle-only curve:
#   adverse excursion BEFORE the newborn leg first reaches 25/50/75/100% of its own extent.
# It begins strictly after pivot confirmation, preserves event ordering, and contains no ATR,
# stop, target, R multiple, cost, or strategy label.

# ======================================= CELL 1 — SETUP ========================================
import os
import subprocess

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.chdir('/content')
FFM_BRANCH = os.environ.get('FFM_BRANCH', 'ssl/stage-2.8-nextleg-race')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print(f'Cloning FFM repo ({FFM_BRANCH})...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(['git', 'clone', '--branch', FFM_BRANCH,
                    'https://github.com/johnamcruz/Futures-Foundation-Model.git',
                    '/content/Futures-Foundation-Model'], capture_output=True, text=True)
if r.returncode:
    print(r.stderr)
    raise RuntimeError(f'git clone failed for {FFM_BRANCH!r}')
os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')
os.system('pip install mantis-tsfm -q 2>&1 | tail -1')
import sympy.printing  # noqa: F401
from futures_foundation.finetune import ssl  # noqa: E402


# ======================================= CELL 2 — CONFIG =======================================
import torch

DATA_DIR = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
WARM_CKPT = os.environ.get(
    'WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg.pt')
OUT_PATH = os.environ.get(
    'OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg_race.pt')
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'

LEG_K = int(os.environ.get('LEG_K', '2'))
LEG_CAP = int(os.environ.get('LEG_CAP', '256'))
LEG_W = float(os.environ.get('LEG_W', '1.0'))
# Stage-2.7's weight 1.0 damaged retained regime/volatility capabilities. Begin conservatively;
# raceR must still learn, and a later A/B may raise this only if retention remains intact.
RACE_W = float(os.environ.get('RACE_W', '0.25'))
RACE_CAP = float(os.environ.get('RACE_CAP', '2.0'))
RACE_LEVELS = tuple(float(x) for x in os.environ.get(
    'RACE_LEVELS', '0.25,0.50,0.75,1.00').split(','))
HORIZONS = (5, 10, 20, 25)
CONTEXT_LENGTHS = (64, 100, 150, 200)
NEW_CHANNELS = int(os.environ.get('NEW_CHANNELS', '3'))
BATCH = int(os.environ.get('BATCH', '512'))
EPOCHS = int(os.environ.get('EPOCHS', '120'))
STEPS = int(os.environ.get('STEPS', '200'))
LR = float(os.environ.get('LR', '0.0001188117389055629'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0.0'))
PATIENCE = int(os.environ.get('PATIENCE', '8'))
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))
RESUME = os.environ.get('RESUME', '0') == '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(DATA_DIR)
if not os.path.isfile(WARM_CKPT):
    raise FileNotFoundError(WARM_CKPT)
assert os.path.abspath(OUT_PATH) != os.path.abspath(WARM_CKPT)
assert not OUT_PATH.endswith('/mantis_ssl_nextleg.pt'), 'production checkpoint is protected'
assert RACE_W > 0 and tuple(sorted(RACE_LEVELS)) == RACE_LEVELS
print(f'Device={DEVICE} | warm={WARM_CKPT} | output={OUT_PATH}')
print(f'nextleg k={LEG_K} cap={LEG_CAP} | race levels={RACE_LEVELS} w={RACE_W}')
print(f'holdout >= {HOLDOUT_START} excluded; reserve covers both future legs')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='nextleg_race', backbone_ckpt=WARM_CKPT,
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS,
    leg_k=LEG_K, leg_cap=LEG_CAP, leg_w=LEG_W, mse_weight=1.0,
    race_w=RACE_W, race_cap=RACE_CAP, race_levels=RACE_LEVELS,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS,
    steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
    clamp=10.0, grad_clip=1.0, val_frac=0.1, holdout_start=HOLDOUT_START,
    controls=(), probe=True, resume=RESUME,
    freeze_encoder_layers=FREEZE_ENCODER_LAYERS, device=DEVICE, seed=0)

print('\n' + '=' * 64 + '\nSTAGE 2.8 NEXTLEG_RACE VERDICT\n' + '=' * 64)
for key, value in verdict.items():
    if key not in ('history', 'epochs'):
        print(f'{key:>24}: {value}')
print(f'candidate encoder -> {OUT_PATH}')
print('Do not promote unless: raceR learns; NextLeg retention is preserved; exact stop-race probe,')
print('trend lifecycle, and anchored Pivot Trend all beat mantis_ssl_nextleg.pt.')
