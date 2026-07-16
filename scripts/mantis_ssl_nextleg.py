# ==============================================================================
# MANTIS SSL STAGE 2.6 — NEXT-LEG FORECASTING (Colab GPU)
# ==============================================================================
#
# Teach the FOUNDATION what trends look like: from a context ending at a CONFIRMED fractal
# pivot, predict the market's own next pattern element in BARS (instrument/TF-agnostic; NO ATR,
# no derived indicators — user spec 2026-07-16):
#   t1 = how many bars the NEWBORN leg runs (real leg vs fizzle — trend-start quality)
#   t2 = how many bars the COUNTER-leg lasts (the retest — the geometry that kills entries)
# Pure SSL: targets from the deterministic PURE fractal detector (k-bar candle comparisons) on
# future price. Candle-seq2seq loss stays as the ANCHOR (mse_weight=1 — anti-drift, banked
# load-bearing). Warm-start from ctr_seq2seq, freeze 2 blocks (the reorder recipe).
#
# GATE (before ANY WR pipeline): the trend-lifecycle scorecard — the new checkpoint must BEAT
# the banked per-direction probes (start 0.7689/0.7635, end 0.7519/0.7603). colabs/mantis/
# trend_probe.py on the new ckpt. Only a scorecard win earns the teach-head/WF re-run.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ main, install) ==========
import os, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (main)...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(['git', 'clone', '--branch', 'main',
                    'https://github.com/johnamcruz/Futures-Foundation-Model.git',
                    '/content/Futures-Foundation-Model'], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); raise RuntimeError('git clone failed')
print('Cloned')

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')
os.system('pip install mantis-tsfm -q 2>&1 | tail -1')
try:
    import sympy.printing  # noqa: F401  (torch symbolic-shape print crash guard)
except Exception as _e:
    print(f'[warn] sympy.printing preload skipped: {_e}')

try:
    from futures_foundation.finetune import ssl, ssl_data  # noqa: F401
    print('FFM + SSL modules import OK')
except ImportError as e:
    print(f'Import failed: {e}\nRestarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ======================================= CELL 2 — CONFIG + pre-flight ==========================
import os, torch

DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
# WARM = the promoted base (ctr_seq2seq). OUT = a DISTINCT file — the base is NEVER overwritten.
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt')
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg.pt')

TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']    # all 9
TFS     = ['1min', '3min', '5min', '15min']                          # all 4 TFs (fractal corpus)
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── NEXT-LEG knobs (bars; NO ATR anywhere) ──
LEG_K   = int(os.environ.get('LEG_K', '2'))       # pure fractal k (raw candle comparisons)
LEG_CAP = int(os.environ.get('LEG_CAP', '256'))   # unresolved legs beyond this are EXCLUDED
LEG_W   = float(os.environ.get('LEG_W', '1.0'))   # leg loss weight vs the candle anchor
def _int_tuple(env, default):
    v = os.environ.get(env)
    return tuple(int(x) for x in v.split(',') if x.strip()) if v else default
HORIZONS        = _int_tuple('HORIZONS', (5, 10, 20, 25))            # candle ANCHOR horizons
CONTEXT_LENGTHS = _int_tuple('CONTEXT_LENGTHS', (64, 100, 150, 200)) # variable context (base recipe)

# ── TRAINING (the reorder-sweep winner recipe — banked) ──
NEW_CHANNELS = int(os.environ.get('NEW_CHANNELS', '3'))
BATCH   = 512
EPOCHS  = int(os.environ.get('EPOCHS', '120'))
STEPS   = 200
LR      = float(os.environ.get('LR', '0.0001188117389055629'))       # ~1.19e-4 (reorder winner)
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0.0'))
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))   # learn ON TOP of the base
CONTROLS = ()          # fast iteration; re-enable ('shuffle','random') for a fresh anti-shortcut pass
PROBE = True           # probe vs vanilla (diagnostic); THE gate = the lifecycle scorecard (separate)
RESUME  = os.environ.get('RESUME', '0') == '1'
SEED = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (warm-start encoder) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
_base = '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt'
assert os.path.abspath(OUT_PATH) != os.path.abspath(_base), 'OUT must never overwrite the base'
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm <- {WARM_CKPT}')
print(f'   NEXT-LEG (bars, NO ATR): k={LEG_K} cap={LEG_CAP} leg_w={LEG_W} | anchor horizons={HORIZONS}')
print(f'   ctx={CONTEXT_LENGTHS} nc={NEW_CHANNELS} frz={FREEZE_ENCODER_LAYERS} lr={LR:.2e} '
      f'EPOCHS={EPOCHS} BATCH={BATCH}')
print(f'   OUTPUT -> {OUT_PATH}   (base ctr_seq2seq UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='nextleg', backbone_ckpt=WARM_CKPT,
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS,
    leg_k=LEG_K, leg_cap=LEG_CAP, leg_w=LEG_W, mse_weight=1.0,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    weight_decay=WEIGHT_DECAY, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP,
    val_frac=VAL_FRAC, holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 2.6 (next-leg) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print('\nNEXT: the GATE — run colabs/mantis/trend_probe.py with CKPT=' + OUT_PATH)
print('      must BEAT banked per-dir probes: start 0.7689/0.7635, end 0.7519/0.7603.')
print('      Only then: teach-head/WF re-run (BACKBONE_CKPT=this) — never before.')
