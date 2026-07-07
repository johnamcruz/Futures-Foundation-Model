# ==============================================================================
# MANTIS SSL STAGE 2 — MULTI-HORIZON, VARIABLE-CONTEXT CANDLE seq2seq (Colab GPU)
# ==============================================================================
#
# The stage-2 pretext that teaches the encoder PRICE-ACTION DYNAMICS AT MULTIPLE SCALES.
# Warm-starts from the stage-1 masked-SSL encoder (mantis_ssl_ohlcv.pt) and continues
# training with a forecasting objective:
#
#   * VARIABLE CONTEXT  — each step samples a context length (64/100/150/200 bars), so the
#     encoder learns short AND long history (scale-invariant; interpolated to Mantis's 512).
#   * MULTI-HORIZON     — predict the future CANDLE (OHLCV) at EACH horizon 5/10/20/25 bars ahead
#     (near AND far), forcing multi-timescale trend understanding.
#   * CANDLE TARGET      — predict the actual future candles, CONTEXT-STANDARDIZED (per-channel
#     z-score by the context's own mean/std). NO ATR / NO R (that's volatility-relative). All
#     channels equal; the model uses AND predicts all OHLCV (incl volume).
#
# Anti-shortcut: target = move FROM now (copy-now == predict-zero, punished). All channels equal.
# clamp + grad-clip keep training stable. Price-action ONLY (raw OHLCV) — no derived features.
#
# OUTPUT: an adapted ENCODER checkpoint. Downstream uses it exactly like stage-1:
#   BACKBONE_CKPT=<this .pt>  (WF/produce / benchmark), or build_model(..., backbone_ckpt=<this>).
# The scorecard: does OHLCV-only (embedding, no handcraft) predict trend BETTER than stage-1.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ main, install) ==========
import os, subprocess
# Reduce CUDA fragmentation OOM (must be set BEFORE torch inits CUDA -> after a runtime restart).
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

# torch.utils._sympy references sympy.printing WITHOUT importing it -> stringifying a dynamic-shape
# expr (the long-horizon windows hit torch's symbolic-shape path) crashes with "module 'sympy' has
# no attribute 'printing'". Force the submodule import so torch can format symbolic exprs.
try:
    import sympy.printing  # noqa: F401
except Exception as _e:      # pragma: no cover
    print(f'[warn] sympy.printing preload skipped: {_e}')

try:
    from futures_foundation.finetune import ssl, ssl_data
    print('FFM + SSL modules import OK')
except ImportError as e:
    print(f'Import failed: {e}\nRestarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ======================================= CELL 2 — CONFIG + pre-flight ==========================
import os, torch

# ── PATHS (Drive) — DEFAULT = the REORDER's forecast-last step (STEP 2). Env-overridable. ──
# REORDER (mask->contrastive->forecast): warm from the contrastive-from-mask regime checkpoint,
#   FREEZE early blocks (=3) so forecast learns ON TOP of the regime instead of erasing it, and
#   write a DISTINCT file so the shipped seq2seq stays put. Copy-and-run as-is for step 2.
# Old default lineage (mask->forecast) if ever needed:
#   WARM_CKPT=.../mantis_ssl_ohlcv.pt OUT_PATH=.../mantis_ssl_seq2seq.pt FREEZE_ENCODER_LAYERS=0
DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
# WARM_CKPT resolved AFTER EXTEND_FROM (below): in extension mode it defaults to EXTEND_FROM (the
# base = ctr_seq2seq) so build_net ALSO warm-starts from the base, not regime; fresh mode -> regime.
WARM_CKPT = os.environ.get('WARM_CKPT', '')
_WARM_DEFAULT = '/content/drive/MyDrive/AI_Models/mantis_ssl_regime.pt'   # fresh stage-2 reorder warm
# DEFAULT OUT = the TUNED reorder (Optuna sweep winner, trial 3) — a DISTINCT file so the manual
# freeze=3 anchor (mantis_ssl_seq2seq_reordered.pt, 52.6%) is NEVER overwritten. The two are the
# freeze-2-vs-3 A/B: this tuned freeze=2 winner vs the anchor freeze=3, both vs seq2seq.
# DEFAULT OUT = the LONG-HORIZON extension file (plain run = warm-start ctr_seq2seq + reach to 75).
# DISTINCT from the short-horizon ext (ctr_seq2seq_ext.pt) so this can't resume/overwrite it. For the
# short-horizon RoBERTa ext set OUT_PATH=...ctr_seq2seq_ext.pt HORIZONS=5,10,20,25 .
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_lh75_seq2seq.pt')

# ── CORPUS (same as stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── MULTI-HORIZON / VARIABLE-CONTEXT candle forecast (sweep-winner) ──
def _int_tuple(env, default):                 # "10,25,50,75" -> (10,25,50,75); unset -> default
    v = os.environ.get(env)
    return tuple(int(x) for x in v.split(',') if x.strip()) if v else default
HORIZONS        = _int_tuple('HORIZONS', (10, 25, 50, 75))   # DEFAULT = long-horizon extension (reach
#                                       to 75 = halfway to the 150-bar trade horizon; 50 anchors it)
CONTEXT_LENGTHS = _int_tuple('CONTEXT_LENGTHS', (100, 150, 200, 200))  # longer context to support it
# LONG-HORIZON EXTENSION (build on what ctr_seq2seq knows -> reach FURTHER): the base only forecasts
# 25 bars ahead but a trade runs 150 (VERT) — extend the horizons so the encoder learns how moves
# DEVELOP over the range where runners form. Keep the count at 4 (head shape matches the warm-start)
# and 50 as the stepping-stone anchor for 75 (halfway to the trade horizon; further = too noisy).
#   HORIZONS=10,25,50,75 CONTEXT_LENGTHS=100,150,200,200 \
#   EXTEND_FROM=.../mantis_ssl_ctr_seq2seq.pt OUT_PATH=.../mantis_ssl_lh75_seq2seq.pt EPOCHS=120
# Judge on the 2025 dry-run vs ctr_seq2seq + the trend_eff/range_expand probes; one-shot 2026 if it wins.
#
# ── DIRECTION TEACHER (fakeout vs trend) — the CONTINUE-vs-REVERSE run ──────────────────────────
# The pivot's weakness is direction×regime: fading a move that KEEPS GOING (real trend) vs one that
# REVERSES (fakeout). candle_mse at h50/h75 is noisy (the exact far candle is ~random); but the SIGN
# of the far move IS learnable and IS the fakeout-vs-trend signal. candle_direction adds BCE on
# sign(fwd close move) at each horizon -> at h50/h75 that BCE literally teaches "does this continue or
# reverse over the next 50-75 bars." Causal (past-only), in the forecast family (not a new objective).
#   OBJECTIVE=candle_direction DIR_WEIGHT=0.3 HORIZONS=10,25,50,75 CONTEXT_LENGTHS=100,150,200,200 \
#   EXTEND_FROM=.../mantis_ssl_ctr_seq2seq.pt OUT_PATH=.../mantis_ssl_lhdir_seq2seq.pt EPOCHS=120
# Then feed it downstream with more context: pivot MV_SEQ=128|200 on this backbone; the TEST is
# WR@3R AMONG counter-trend entries. (Run the plain long-horizon candle_mse FIRST as the baseline.)
# DEFAULTS = the REORDER-sweep winner (trial 3): candle_mse / nc=3 / wd=0 / freeze=2 / lr=1.19e-4.
OBJECTIVE       = os.environ.get('OBJECTIVE', 'candle_mse')    # candle_direction = direction teacher;
#                                             forecast_dist objectives: candle_quantile | candle_bins
NEW_CHANNELS    = int(os.environ.get('NEW_CHANNELS', '3'))
DIR_WEIGHT      = float(os.environ.get('DIR_WEIGHT', '0.0'))   # >0 (~0.3) for candle_direction
# ── PRETEXT: 'forecast' (candle_mse/direction) | 'forecast_dist' (DISTRIBUTIONAL — the richer
# fakeout-vs-trend teacher). forecast_dist predicts the DISTRIBUTION of the far candle (quantile/bins)
# -> at h50/h75 where the point forecast is noise, the distribution's SHAPE carries continue-vs-reverse.
# mse_weight KEEPS the reconstruction anchor (default 1.0 -> NO drift, unlike ELECTRA). Recipe:
#   PRETEXT=forecast_dist OBJECTIVE=candle_quantile QUANTILE_TAUS=bolt9 HORIZONS=10,25,50,75 \
#   EXTEND_FROM=.../mantis_ssl_ctr_seq2seq.pt OUT_PATH=.../mantis_ssl_fdist_lh.pt EPOCHS=120
PRETEXT       = os.environ.get('PRETEXT', 'forecast')
MSE_WEIGHT    = float(os.environ.get('MSE_WEIGHT', '1.0'))     # forecast_dist reconstruction ANCHOR (keep!)
QUANTILE_TAUS = os.environ.get('QUANTILE_TAUS', 'bolt9')       # candle_quantile: 'lohi'(2) | 'bolt9'(9)
BINS_K        = int(os.environ.get('BINS_K', '41'))            # candle_bins resolution
# ── forecast_dist config = WHAT THE SCAN TAUGHT US, not guesses ──
# The forecast_dist Optuna scan came back FLAT at short horizons (redundant with stage-2), so it never
# crowned a winner. What it DID establish: (1) the base hyperparams are the forecast sweep-winner (trial
# 3: lr 1.19e-4 / nc3 / frz2 / wd0 — already the defaults above), and (2) the reconstruction ANCHOR is
# load-bearing — the scan's mse_weight=0 arm is the drift trap ([[electra]] lesson). So we KEEP
# mse_weight=1.0 and the spread-carrying head (candle_quantile / bolt9 — 'spread SHAPE is the signal').
# The NEW variable the scan never tried = LONG horizons; that's the actual bet.
if PRETEXT == 'forecast_dist' and OBJECTIVE == 'candle_mse':  # forecast default -> dist default
    OBJECTIVE = 'candle_quantile'                             # Bolt pinball (spread shape = the signal)

# ── TRAINING (sweep-winner; BATCH MATCHES THE SWEEP so the tuned LR transfers) ──
BATCH   = 512         # PARITY with the sweep (lr was tuned at 512; 1024 would need a different lr)
EPOCHS  = int(os.environ.get('EPOCHS', '120'))  # DEFAULT = the RoBERTa extension budget (ADDITIONAL
#                       epochs on resume). The original stage-2 run used 60 — a WALL,
#                       not convergence (the identical harness hit it still improving) -> the RoBERTa
#                       extension uses EPOCHS=120 (ADDITIONAL, on top of the base's 60) with
#                       EXTEND_FROM (below); patience still governs.
STEPS   = 200         # steps/epoch
LR      = float(os.environ.get('LR', '0.0001188117389055629'))   # reorder-sweep winner (trial 3, ~1.19e-4)
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0.0'))       # trial 3 (env-overridable)
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0                    # stability
CONTROLS = ()                                  # skip shuffle/random retrains (fast iteration; judge
                                               # offline). Re-enable ('shuffle','random') for a fresh
                                               # anti-shortcut check on a NEW pretext/target change.
PROBE = True                                    # probe vs vanilla (diagnostic, not a gate)
SEED = 0

# ── CRASH-SAFE + ANTI-FORGETTING (best saved PROGRESSIVELY to OUT_PATH; Colab-disconnect resilient) ──
RESUME  = os.environ.get('RESUME', '0') == '1'   # resume from the best saved to OUT_PATH
# DEFAULT = 3 for the REORDER's forecast-last step: freeze the first 3 (of 6) blocks so forecast
# learns on top of the regime instead of erasing it. Set FREEZE_ENCODER_LAYERS=0 for the old
# default lineage (mask->forecast full fine-tune, the original sweep winner).
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))   # trial 3 = freeze=2

# ── RoBERTa EXTENSION MODE — "was the base undertrained?" ──
# EXTEND_FROM=<promoted base .pt> continues that checkpoint's OWN training (same recipe) with a
# bigger epoch budget, into a DISTINCT OUT_PATH (the extended model is a NEW CANDIDATE — it must
# win the 2025 dry-run like everything else; the promoted base is never overwritten):
#   EXTEND_FROM=/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt \
#   OUT_PATH=/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq_ext.pt  EPOCHS=120
# NOTE: on resume the epoch counter restarts at 0, so EPOCHS = ADDITIONAL epochs on top of the
# base's 60 (EPOCHS=120 -> up to +120 more). patience(8) still stops early if the base was
# ALREADY converged — that's the cheap answer to "was it undertrained".
# Mechanics: OUT_PATH seeded with a COPY of EXTEND_FROM, RESUME forced on -> trainer loads it.
# DEFAULT = the extension run (copy-paste-and-run). Set EXTEND_FROM='' to run the original
# stage-2 recipe from WARM_CKPT instead.
EXTEND_FROM = os.environ.get('EXTEND_FROM',
                             '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt')
if EXTEND_FROM:
    import shutil
    if not os.path.exists(EXTEND_FROM):
        raise FileNotFoundError(f'EXTEND_FROM not found: {EXTEND_FROM}')
    if os.path.abspath(OUT_PATH) == os.path.abspath(EXTEND_FROM):
        raise SystemExit('❌ EXTEND_FROM == OUT_PATH — the extension must write a NEW file.')
    if not os.path.exists(OUT_PATH):
        shutil.copy2(EXTEND_FROM, OUT_PATH)
        print(f'[extend] seeded {OUT_PATH} <- copy of {EXTEND_FROM}')
    else:
        print(f'[extend] OUT_PATH exists — resuming the extension already in progress')
    RESUME = True
# Resolve WARM_CKPT default now that EXTEND_FROM is known: extension -> the base (ctr_seq2seq) so the
# build_net warm-start matches the resume base; fresh run (EXTEND_FROM='') -> the regime reorder warm.
WARM_CKPT = WARM_CKPT or EXTEND_FROM or _WARM_DEFAULT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (warm-start encoder) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
# In EXTENSION mode the EFFECTIVE base is EXTEND_FROM (resume-loads it over the WARM_CKPT backbone,
# common.py fit(): build_net warm-starts WARM_CKPT, then resume overwrites with the EXTEND_FROM copy).
_base = EXTEND_FROM if EXTEND_FROM else WARM_CKPT
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | '
      f'BASE (continues from) <- {_base}'
      + (f'  [WARM_CKPT {os.path.basename(WARM_CKPT)} is loaded then overwritten by resume]'
         if EXTEND_FROM else ''))
print(f'   pretext={PRETEXT} obj={OBJECTIVE} lr={LR:.2e} nc={NEW_CHANNELS} frz={FREEZE_ENCODER_LAYERS} '
      f'wd={WEIGHT_DECAY} BATCH={BATCH} EPOCHS={EPOCHS}'
      + (f' | DIST mse_w={MSE_WEIGHT} taus={QUANTILE_TAUS} bins_k={BINS_K}'
         if PRETEXT == 'forecast_dist' else ''))
print(f'   horizons={HORIZONS} context_lengths={CONTEXT_LENGTHS}')
_live = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'
_safe = '   (live mantis_ssl_seq2seq.pt UNTOUCHED)' if os.path.abspath(OUT_PATH) != os.path.abspath(_live) else '   ⚠️ WRITES THE LIVE seq2seq'
print(f'   OUTPUT -> {OUT_PATH}{_safe}')


# ======================================= CELL 3 — TRAIN (single run, no Optuna) ================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext=PRETEXT, backbone_ckpt=WARM_CKPT,                  # forecast | forecast_dist, warm-start
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS, objective=OBJECTIVE,
    mse_weight=MSE_WEIGHT, quantile_taus=QUANTILE_TAUS, bins_k=BINS_K,   # forecast_dist knobs (anchor)
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    weight_decay=WEIGHT_DECAY, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP, val_frac=VAL_FRAC,
    holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS, dir_weight=DIR_WEIGHT,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 2 (multi-horizon seq2seq) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k != 'history':
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print(f'report           -> {OUT_PATH}.report.json')
print('\nDownstream: BACKBONE_CKPT=<this .pt>  — then benchmark OHLCV-only WR@3R vs stage-1.')
