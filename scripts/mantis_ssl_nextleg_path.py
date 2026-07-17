# ==============================================================================
# MANTIS SSL STAGE 2.7 — NEXT-LEG + PATH (Colab GPU)
# ==============================================================================
#
# Stage 2.6 (nextleg, the GRADUATED backbone) teaches HOW FAR and HOW LONG a leg runs — both in
# bars. It never teaches HOW THE LEG GETS THERE. 2.7 adds exactly ONE target:
#
#   r1 = the deepest pullback WITHIN the newborn leg / that leg's OWN extent
#
# A ratio of two price distances from the SAME leg -> unitless, scale-free, instrument- and
# TF-agnostic, read straight off candle highs/lows. NO ATR. NO cost. NO entry/stop/R.
# FFM learns candles; the strategy layer owns risk. Pure SSL — the target is the market's own
# structure applied to future price, exactly like t1/t2.
#
# WHY (a MEASURED hole, not a guess). On the production pool (ES+NQ@3min, 130,994 pivots),
# predicting whether a leg's path takes out a stop:
#     pure stop geometry (risk/ATR, 1 feature)   AUC 0.5184
#     the nextleg embedding (1280-d)             AUC 0.5572   <- +0.039 of real path structure
#     embedding + geometry                       AUC 0.5572   <- geometry adds NOTHING it lacks
# So real path structure exists, geometry explains none of what the encoder holds, and the
# encoder plainly has only part of it. 2.6 was never asked for the rest. This asks.
#
# Expressing the same idea in R-units would smuggle the strategy's stop into the pretext and
# reproduce the shape that lost in turn-electra (objective aimed AT the downstream event).
# This stays a FORECASTER of the market's own pattern unit, aimed NEAR the event.
#
# GATES — ALL must pass before 2.7 replaces 2.6 anywhere:
#   1. retrace_corr materially > 0            (training log; it learned the target at all)
#   2. skill / legR not degraded vs 2.6       (drift damage killed turn-electra AND lc512)
#   3. probe_atlas pred_stopped_out > 0.5473  (THE hole. Geometry-only floor = 0.5184)
#      and pred_vol_expand NOT below 0.842    (the strongest clean signal — do not dent it)
#   4. error_mining missed_winners > 0.5174   (the recall blind spot)
#   5. trend-lifecycle scorecard >= 2.6 — RE-BASELINE 2.6 FIRST: the banked 0.7689/0.7635 bars
#      were measured under the OLD SHUFFLED probe split (fixed in 1e4bf45). A contiguous-split
#      number is NOT comparable to them. Re-run 2.6 through trend_probe.py before judging 2.7.
#   6. downstream WR@3R / meanR >= 2.6 at the deploy operating points
#
# 2.6 IS NEVER OVERWRITTEN: OUT_PATH is a distinct file, asserted below.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM, install) =================
import os, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.chdir('/content')

# The 2.7 pretext ships on its own branch until merged. Cloning 'main' before the merge gives
# KeyError: 'nextleg_path' from the registry (fail-fast, working as intended) — point this at the
# branch to test pre-merge, or leave it on main once the PR lands.
FFM_BRANCH = os.environ.get('FFM_BRANCH', 'ssl/stage-2.7-nextleg-path')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print(f'Cloning FFM repo ({FFM_BRANCH})...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(['git', 'clone', '--branch', FFM_BRANCH,
                    'https://github.com/johnamcruz/Futures-Foundation-Model.git',
                    '/content/Futures-Foundation-Model'], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); raise RuntimeError(f'git clone failed (branch {FFM_BRANCH!r})')
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
# WARM = stage 2.6 (nextleg — the graduated backbone). 2.7 learns the PATH on top of it.
# OUT  = a DISTINCT file. 2.6 is the live lineage and is NEVER overwritten.
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg.pt')
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg_path.pt')

TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']    # all 9
TFS     = ['1min', '3min', '5min', '15min']                          # all 4 TFs (fractal corpus)
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── NEXT-LEG knobs (bars; NO ATR anywhere) — identical to 2.6 so the A/B is one lever ──
LEG_K   = int(os.environ.get('LEG_K', '2'))       # pure fractal k (raw candle comparisons)
LEG_CAP = int(os.environ.get('LEG_CAP', '256'))   # unresolved legs beyond this are EXCLUDED
LEG_W   = float(os.environ.get('LEG_W', '1.0'))   # leg loss weight vs the candle anchor
# ── THE 2.7 LEVER (the only difference vs 2.6) ──
RETRACE_W   = float(os.environ.get('RETRACE_W', '1.0'))   # path-target weight. 0.0 == stage 2.6.
RETRACE_CAP = float(os.environ.get('RETRACE_CAP', '2.0')) # r1 clip (a tiny-extent leg can blow up)
def _int_tuple(env, default):
    v = os.environ.get(env)
    return tuple(int(x) for x in v.split(',') if x.strip()) if v else default
HORIZONS        = _int_tuple('HORIZONS', (5, 10, 20, 25))            # candle ANCHOR horizons
CONTEXT_LENGTHS = _int_tuple('CONTEXT_LENGTHS', (64, 100, 150, 200)) # variable context (base recipe)

# ── TRAINING (2.6's recipe, unchanged — one lever at a time) ──
NEW_CHANNELS = int(os.environ.get('NEW_CHANNELS', '3'))
BATCH   = 512
EPOCHS  = int(os.environ.get('EPOCHS', '120'))
STEPS   = 200
LR      = float(os.environ.get('LR', '0.0001188117389055629'))       # ~1.19e-4 (reorder winner)
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0.0'))
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))   # learn ON TOP of 2.6
CONTROLS = ()          # fast iteration; re-enable ('shuffle','random') for a fresh anti-shortcut pass
PROBE = True           # probe vs vanilla (diagnostic); THE gates are probe_atlas + the scorecard
RESUME  = os.environ.get('RESUME', '0') == '1'
SEED = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (stage 2.6 encoder) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
for _protected in ('/content/drive/MyDrive/AI_Models/mantis_ssl_nextleg.pt',
                   '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt'):
    assert os.path.abspath(OUT_PATH) != os.path.abspath(_protected), \
        f'OUT_PATH must never overwrite a promoted checkpoint: {_protected}'
assert RETRACE_W > 0, 'RETRACE_W=0 is stage 2.6 — run scripts/mantis_ssl_nextleg.py instead'
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm <- {WARM_CKPT}')
print(f'   NEXT-LEG (bars, NO ATR): k={LEG_K} cap={LEG_CAP} leg_w={LEG_W} | anchor horizons={HORIZONS}')
print(f'   *** 2.7 PATH TARGET: retrace_w={RETRACE_W} cap={RETRACE_CAP} '
      f'(unitless giveback/extent — pure candles) ***')
print(f'   ctx={CONTEXT_LENGTHS} nc={NEW_CHANNELS} frz={FREEZE_ENCODER_LAYERS} lr={LR:.2e} '
      f'EPOCHS={EPOCHS} BATCH={BATCH}')
print(f'   OUTPUT -> {OUT_PATH}   (stage 2.6 nextleg UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='nextleg_path', backbone_ckpt=WARM_CKPT,
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS,
    leg_k=LEG_K, leg_cap=LEG_CAP, leg_w=LEG_W, mse_weight=1.0,
    retrace_w=RETRACE_W, retrace_cap=RETRACE_CAP,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    weight_decay=WEIGHT_DECAY, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP,
    val_frac=VAL_FRAC, holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 2.7 (next-leg + PATH) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (2.6 nextleg untouched)')
print("""
NEXT — the gates, in order. Do NOT skip to the WR pipeline.

  A. LEARNING CHECK (free, from the log above): retrace_corr materially > 0, and
     skill / legR NOT below 2.6's. retrace_corr>0 with a dead stop-race probe = 2.7 dies.

  B. RE-EMBED the production pool with this checkpoint, then:
       probe_atlas.py   pred_stopped_out  must beat  0.5473   <- THE hole (geom floor 0.5184)
                        pred_vol_expand   must NOT fall below 0.8420  <- drift-damage kill switch
                        ret_* retention   must NOT regress
       error_mining.py  missed_winners    must beat  0.5174   <- the recall blind spot
     Both append to the ledger keyed by checkpoint -> the comparison is longitudinal.

  C. trend_probe.py — RE-BASELINE 2.6 FIRST. The banked 0.7689/0.7635 bars were measured under
     the OLD SHUFFLED probe split (fixed 1e4bf45); a contiguous-split score is NOT comparable.
     Run 2.6 AND 2.7 through it under the same split, then compare.

  D. Only after A-C: teach-head / WF re-run with BACKBONE_CKPT=this.
""")
