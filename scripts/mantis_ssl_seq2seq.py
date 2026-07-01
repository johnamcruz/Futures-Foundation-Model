# ==============================================================================
# MANTIS SSL STAGE 2 — SEQ2SEQ FORECASTING FT on raw OHLCV (Colab GPU)
# ==============================================================================
#
# A SECOND self-supervised stage that CONTINUES (fine-tunes) the masked-SSL encoder
# (scripts/mantis_ssl_pretrain.py -> mantis_ssl_ohlcv.pt) with a CAUSAL SEQ2SEQ
# FORECASTING pretext: encode a CONTEXT window of past bars, predict the NEXT
# `horizon` bars (full OHLCV). To forecast the future the encoder MUST model forward
# dynamics — momentum continuation, volatility persistence, trend vs. reversal — i.e.
# the trend-prediction the downstream buy/sell classifier needs. Learning the data
# better as a forecaster is the path to better trend prediction.
#
# PRICE-ACTION ONLY: raw OHLCV (open/high/low/close/volume). No derived/handcraft
# features anywhere in this stage — same corpus as stage 1.
#
# WARM-START: starts from the stage-1 checkpoint (BACKBONE_CKPT) — this is an FT on
# top of the current SSL encoder, not a fresh pretrain.
#
# OUTPUT: an adapted ENCODER checkpoint. Downstream classifier finetuning starts from
# it exactly like stage 1:  BACKBONE_CKPT=<this .pt>  (WF/produce), or
# build_model(..., backbone_ckpt=<this .pt>).
#
# Validity gates (identical rig to stage 1):
#   * TIME-SPLIT val forecast MSE early-stop      (generalize forward; 2026 EXCLUDED)
#   * PROBE vs vanilla = the GATE: frozen embedding must predict regime/vol/structure +
#     forward buy/sell move BETTER than the un-adapted backbone
#   * REAL vs SHUFFLE vs RANDOM = probe-based diagnostic (only REAL has a predictable
#     continuation; time-scrambled / noise do not)
# ==============================================================================


# ==============================================================================
# CELL 1 — SETUP  (clone FFM @ mantis branch, install, mantis-tsfm)
# ==============================================================================
import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (mantis branch)...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(
    ['git', 'clone', '--branch', 'mantis',
     'https://github.com/johnamcruz/Futures-Foundation-Model.git',
     '/content/Futures-Foundation-Model'],
    capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); raise RuntimeError('git clone failed')
print('Cloned')

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')
os.system('pip install mantis-tsfm -q 2>&1 | tail -1')
os.system('pip install optuna -q 2>&1 | tail -1')      # OPTUNA_SCAN / probe-gate tuning

try:
    from futures_foundation.finetune import ssl, ssl_data
    print('FFM + SSL modules import OK')
except ImportError as e:
    print(f'Import failed: {e}\nRestarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ==============================================================================
# CELL 2 — CONFIGURATION  (Drive paths + hyperparameters + pre-flight)
# ==============================================================================
import os, torch

# ── PATHS (Drive) ──
# DATA_DIR must contain the raw CSVs named  {TICKER}_{TF}.csv  with columns
# datetime,open,high,low,close,volume  (e.g. ES_3min.csv).  SAME corpus as stage 1.
DATA_DIR  = '/content/drive/MyDrive/Futures Data'
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_ohlcv.pt'      # stage-1 encoder (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'    # stage-2 adapted encoder

# ── CORPUS (identical to stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1                    # last 10% of each stream's pre-2026 bars = val

# ── SEQ2SEQ FORECASTING ──
SEQ          = 64      # context window (bars); interpolated to Mantis's native 512 internally
HORIZON      = 16      # forecast horizon: predict the next HORIZON bars from the context
NEW_CHANNELS = 8       # channel-combiner output (OHLCV=5 -> NEW_CHANNELS)
MAX_JITTER   = 16      # forward bars reserved for the buy/sell probe (in-stream)

# ── TRAINING (GPU-max) ──
BATCH        = 1024    # drop if OOM (512 / 768).
EPOCHS       = 60
STEPS        = 200     # steps per epoch
LR           = 1e-4
PATIENCE     = 8
# ── STABILITY (anti-blowup) ──
# A FLAT/compressed context window has ~0 std, so the forward delta target (fut-last)/ctx_std
# explodes -> diverging loss (train ~5e4). These bound it: clamp the standardized values, clip
# the gradient norm. Lower CLAMP (e.g. 6) = more conservative; raise only if losses stay tame.
CLAMP        = 10.0
GRAD_CLIP    = 1.0
CONTROLS     = ['shuffle', 'random']   # probe-based diagnostics (did temporal order help)
COMPILE      = False   # torch.compile(encoder) — try True on A100/L4 for extra speed
SEED         = 0

# ── CHANNEL-WEIGHTED LOSS SWEEP (price-path) ──
# Loop over per-channel loss weights (O,H,L,C,V) and rank them by the 10-fold FORWARD-predictive
# probe score. No Optuna — a simple, deterministic, fully-visible sweep. Each candidate trains a
# full encoder + saves its own checkpoint, so a Colab disconnect only loses the CURRENT candidate.
# PRICE-PATH hypothesis: emphasize close, de-emphasize near-unpredictable volume, for trend focus.
CANDIDATES = {
    'equal':   [1.0, 1.0, 1.0, 1.0, 1.0],   # = v1 baseline
    'pp_mod':  [1.0, 1.0, 1.0, 2.0, 0.25],  # price-path moderate
    'pp_novol':[1.0, 1.0, 1.0, 2.0, 0.0],   # price-path, drop volume
    'close3':  [1.0, 1.0, 1.0, 3.0, 0.0],   # close-heavy
}
PROBE_FOLDS  = 10     # k-fold CV per probe -> robust score for RANKING candidates (skip=1)
PROBE        = True   # probe regime/vol/structure + forward buy/sell move vs vanilla

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT: fail in seconds if a path is wrong (before any GPU) ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(
        f'WARM_CKPT (stage-1 encoder) not found:\n  {WARM_CKPT}\n'
        f'Run scripts/mantis_ssl_pretrain.py first, or point WARM_CKPT at the masked-SSL ckpt.')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(
        f'No {{TICKER}}_{{TF}}.csv files found under {DATA_DIR}.\n'
        f'Expected e.g. {DATA_DIR}/ES_3min.csv with columns '
        f'datetime,open,high,low,close,volume.')
print(f'✅ PRE-FLIGHT: found {len(found)}/{len(TICKERS)*len(TFS)} CSVs under {DATA_DIR}')
print(f'   seq2seq FORECAST | warm-start <- {WARM_CKPT}')
print(f'   corpus {TICKERS} x {TFS} | SEQ={SEQ} HORIZON={HORIZON} BATCH={BATCH} EPOCHS={EPOCHS}')
print(f'   channel-weight SWEEP: {list(CANDIDATES)} | probe_folds={PROBE_FOLDS} | controls on winner only')
print(f'   OUTPUT base -> {OUT_PATH}  (per-candidate: <base>_<tag>.pt)')


# ==============================================================================
# CELL 3 — CHANNEL-WEIGHT SWEEP (deterministic, visible) -> rank by 10-fold forward_score
#   phase 1: each candidate = full forecast FT (NO controls, fast) -> 10-fold probe -> save ckpt
#   phase 2: run shuffle/random CONTROLS on the WINNER only (validity, 10-fold)
# ==============================================================================
from pathlib import Path

def _ck(tag):                                            # per-candidate checkpoint path
    p = Path(OUT_PATH)
    return str(p.with_name(p.stem + f'_{tag}').with_suffix('.pt'))

_common = dict(
    data_dir=DATA_DIR, tickers=TICKERS, tfs=TFS,
    pretext='forecast', horizon=HORIZON, backbone_ckpt=WARM_CKPT,   # FT on stage-1 encoder
    grad_clip=GRAD_CLIP, clamp=CLAMP,                               # stability
    seq=SEQ, max_jitter=MAX_JITTER, new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    probe=PROBE, probe_folds=PROBE_FOLDS, n_trials=0, max_iters=1,  # no Optuna
    device=device.type, compile_model=COMPILE, seed=SEED)

# ---- phase 1: sweep (no controls) ----
results = {}
for tag, w in CANDIDATES.items():
    out = _ck(tag)
    print(f"\n{'#'*70}\n#  CANDIDATE '{tag}'  weights(O,H,L,C,V)={w}  -> {out}\n{'#'*70}", flush=True)
    v = ssl.loop_ssl(out_path=out, channel_weights=w, controls=(), **_common)   # NO controls = fast
    results[tag] = dict(weights=w, ckpt=out, forward_score=v.get('forward_score'),
                        fwd_absmove_delta=v.get('fwd_absmove_delta'),
                        fwd_dir_delta=v.get('fwd_dir_delta'),
                        forecast_skill=v.get('forecast_skill'),
                        mean_core_delta=v['history'][-1].get('mean_core_delta'))

# ---- ranking table ----
ranked = sorted(results.items(), key=lambda kv: (kv[1]['forward_score'] or -9), reverse=True)
print("\n" + "=" * 88)
print("  CHANNEL-WEIGHT SWEEP — ranked by 10-fold forward_score (fwd move-size + direction vs vanilla)")
print("=" * 88)
print(f"  {'cand':<9}{'weights (O,H,L,C,V)':<26}{'fwd_score':>10}{'fwd_size':>10}{'fwd_dir':>9}"
      f"{'skill':>8}{'core':>9}")
for tag, r in ranked:
    print(f"  {tag:<9}{str(r['weights']):<26}{(r['forward_score'] or 0):>+10.4f}"
          f"{(r['fwd_absmove_delta'] or 0):>+10.4f}{(r['fwd_dir_delta'] or 0):>+9.4f}"
          f"{(r['forecast_skill'] or 0):>+8.3f}{(r['mean_core_delta'] or 0):>+9.4f}")
best_tag, best = ranked[0]
print("=" * 88)
print(f"  WINNER: '{best_tag}'  weights={best['weights']}  forward_score={best['forward_score']:+.4f}")

# ---- phase 2: controls on the WINNER only (shuffle/random validity, 10-fold) ----
win_out = _ck(best_tag + '_final')
print(f"\n{'#'*70}\n#  CONTROLS on winner '{best_tag}' (shuffle/random validity) -> {win_out}\n{'#'*70}",
      flush=True)
vv = ssl.loop_ssl(out_path=win_out, channel_weights=best['weights'],
                  controls=('shuffle', 'random'), **_common)
print("\n" + "=" * 60 + "\n  WINNER VERDICT (with controls)\n" + "=" * 60)
for k, val in vv.items():
    if k != 'history':
        print(f'  {k:>22}: {val}')
print(f"\nBEST stage-2 encoder -> {win_out}")
print(f"All candidate ckpts  -> " + ", ".join(_ck(t) for t in CANDIDATES))
print("\nDownstream: BACKBONE_CKPT=<best .pt>  (compare vs stage-1 + v1 on the edge ruler)")
