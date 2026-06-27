"""Export a produce() bundle to ONNX for in-process bot inference (no torch /
xgboost daemons; sidesteps the macOS libomp collision). Two parts:

  - signal / risk XGBoost heads -> ONNX (onnxmltools convert_xgboost, in-process)
  - Chronos encoder -> ONNX (torch) via the extractors.chronos.onnx_encoder
    SUBPROCESS — torch must not share a process with xgboost.

Every export is PARITY-CHECKED against the joblib bundle before it's accepted.
Files are written next to the joblib: <stem>_signal_head.onnx,
<stem>_risk_head.onnx (if a risk head exists), <stem>_encoder.onnx.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PARITY_TOL = 1e-5          # XGBoost head proba/raw (ULP-level drift floor)


def _extract_proba(outs, ncols):
    """onnxmltools classifier output is [label, proba] or a ZipMap list."""
    for o in outs:
        if getattr(o, 'ndim', 0) == 2 and o.shape[1] == ncols:
            return o
        if isinstance(o, list) and o and isinstance(o[0], dict):
            ks = sorted(o[0])
            return np.array([[d[k] for k in ks] for d in o], np.float32)
    return None


def _bake_platt_calibration(model, platt):
    """Rewrite a convert_xgboost ONNX so its [N,2] 'probabilities' output is
    Platt-calibrated: col1 = sigmoid(A·logit(clip(p1))+B), col0 = 1-col1.
    Mirrors XGBHead._apply_platt exactly (same clip, same logit) so the ONNX
    stays parity-identical to the calibrated joblib. Binary heads only."""
    from onnx import helper, numpy_helper
    A, B = platt
    eps = 1e-6
    g = model.graph
    OLD, RAW = 'probabilities', 'cal_raw_probabilities'
    if not any(o == OLD for nd in g.node for o in nd.output):
        raise RuntimeError("signal-head ONNX has no 'probabilities' output to calibrate")
    for nd in g.node:                                  # producer: OLD -> RAW
        for i, o in enumerate(nd.output):
            if o == OLD:
                nd.output[i] = RAW
    old_vi = next((vi for vi in g.output if vi.name == OLD), None)
    keep = [vi for vi in g.output if vi.name != OLD]
    del g.output[:]; g.output.extend(keep)

    def c(name, arr):
        return helper.make_node('Constant', [], [name],
                                value=numpy_helper.from_array(np.asarray(arr, np.float32), name + '_v'))
    def ci(name, arr):
        return helper.make_node('Constant', [], [name],
                                value=numpy_helper.from_array(np.asarray(arr, np.int64), name + '_v'))
    N = helper.make_node
    g.node.extend([
        ci('cal_i1', [1]), c('cal_eps', eps), c('cal_1me', 1 - eps),
        c('cal_A', [[A]]), c('cal_B', [[B]]), c('cal_one', [[1.0]]),
        N('Gather', [RAW, 'cal_i1'], ['cal_p1'], axis=1),         # [N,1]
        N('Clip', ['cal_p1', 'cal_eps', 'cal_1me'], ['cal_p1c']),
        N('Sub', ['cal_one', 'cal_p1c'], ['cal_om']),            # 1 - p1
        N('Log', ['cal_p1c'], ['cal_lp']), N('Log', ['cal_om'], ['cal_lom']),
        N('Sub', ['cal_lp', 'cal_lom'], ['cal_logit']),
        N('Mul', ['cal_logit', 'cal_A'], ['cal_az']),
        N('Add', ['cal_az', 'cal_B'], ['cal_z']),
        N('Sigmoid', ['cal_z'], ['cal_c1']),
        N('Sub', ['cal_one', 'cal_c1'], ['cal_c0']),
        N('Concat', ['cal_c0', 'cal_c1'], [OLD], axis=1),         # [N,2]
    ])
    g.output.append(old_vi if old_vi is not None else
                    helper.make_tensor_value_info(OLD, 1, [None, 2]))
    return model


def export_bundle_onnx(bundle, output_path, *, verbose=True, samples=50, seed=42):
    """Export the bundle's heads + encoder to ONNX next to output_path,
    each parity-checked vs the joblib. Returns {name: (path, delta_or_msg, ok)}."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort

    stem = Path(output_path).with_suffix('')
    n = int(bundle['feat_dim'])
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((samples, n)).astype(np.float32)
    results = {}

    # ── signal head (XGBClassifier) ──────────────────────────────────────
    sig = Path(f"{stem}_signal_head.onnx")
    onx = convert_xgboost(bundle['signal_head']._clf,
                          initial_types=[('input', FloatTensorType([None, n]))],
                          target_opset=15)
    platt = getattr(bundle['signal_head'], '_platt', None)
    if platt is not None:                              # bake calibration in
        onx = _bake_platt_calibration(onx, platt)
    sig.write_bytes(onx.SerializeToString())
    ref = bundle['signal_head'].predict_proba(X)       # already calibrated
    proba = _extract_proba(
        ort.InferenceSession(str(sig), providers=['CPUExecutionProvider'])
        .run(None, {'input': X}), ref.shape[1])
    d = float(np.abs(proba[:, 1] - ref[:, 1]).max()) if proba is not None else float('inf')
    results['signal_head'] = (str(sig), d, d < PARITY_TOL)

    # ── risk head (XGBRegressor, binary labelers only) ───────────────────
    if bundle.get('risk_head') is not None:
        rsk = Path(f"{stem}_risk_head.onnx")
        onx = convert_xgboost(bundle['risk_head']._reg,
                              initial_types=[('input', FloatTensorType([None, n]))],
                              target_opset=15)
        rsk.write_bytes(onx.SerializeToString())
        ref_r = bundle['risk_head']._reg.predict(X)
        raw = (ort.InferenceSession(str(rsk), providers=['CPUExecutionProvider'])
               .run(None, {'input': X})[0].squeeze())
        d = float(np.abs(raw - ref_r).max())
        results['risk_head'] = (str(rsk), d, d < PARITY_TOL)

    # ── Chronos encoder (torch SUBPROCESS — no xgboost in that process) ──
    enc = Path(f"{stem}_encoder.onnx")
    ck = str(bundle['chronos_ckpt'])
    ctx = int(bundle['ctx_window'])
    from . import backbone as _bb
    root = str(_bb._ROOT)
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    r = subprocess.run(
        [sys.executable, '-m', 'futures_foundation.extractors.chronos.onnx_encoder',
         ck, str(ctx), str(enc)],
        cwd=root, env=env, capture_output=True, text=True)
    enc_ok = r.returncode == 0
    msg = (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])
    results['encoder'] = (str(enc), msg, enc_ok)

    # ── volume encoder (opt-in): a 2nd encoder ONNX with the volume pool. The
    # bot runs it on the volume window and concatenates, same as the price
    # encoder. Kept SEPARATE (the bot chains the two), parity-checked like price.
    ve = bundle.get('volume_embed')
    if ve:
        venc = Path(f"{stem}_volume_encoder.onnx")
        rv = subprocess.run(
            [sys.executable, '-m', 'futures_foundation.extractors.chronos.onnx_encoder',
             ck, str(ctx), str(venc), ve.get('pool', 'meanreg')],
            cwd=root, env=env, capture_output=True, text=True)
        venc_ok = rv.returncode == 0
        vmsg = (rv.stdout.strip().splitlines()[-1] if rv.stdout.strip()
                else rv.stderr[-400:])
        results['volume_encoder'] = (str(venc), vmsg, venc_ok)

    if verbose:
        print("\n[onnx] export + parity (vs joblib):")
        for k, (p, d, ok) in results.items():
            print(f"  {('✓' if ok else '✗')} {k:12s} {p}  {d}")
        if not all(ok for _, _, ok in results.values()):
            print("  ⚠ one or more ONNX exports FAILED parity — do NOT ship.")
    return results


def main():
    """CLI: export an existing produce() joblib bundle to ONNX + parity-check.
    Exits non-zero if any export fails parity vs the joblib."""
    import argparse
    import joblib
    ap = argparse.ArgumentParser(
        description="Export a produce() joblib bundle to ONNX (XGBoost heads + "
                    "Chronos encoder), each parity-checked vs the joblib.")
    ap.add_argument('bundle', help='joblib bundle from produce.py')
    ap.add_argument('--samples', type=int, default=50, help='parity sample rows')
    args = ap.parse_args()
    bundle = joblib.load(args.bundle)
    res = export_bundle_onnx(bundle, args.bundle, verbose=True, samples=args.samples)
    raise SystemExit(0 if all(ok for _, _, ok in res.values()) else 1)


if __name__ == '__main__':
    main()
