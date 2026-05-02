"""
Checkpoint diagnostic — feature weight spread + fusion split

Inspects a HybridStrategyModel done.pt checkpoint to verify the model
is learning discriminative representations, not treating all inputs equally.

Two diagnostics:
  1. Strategy feature weight spread — are some features 2-3x more important?
  2. Fusion layer input split — is the backbone dominating over context/strategy?

Usage:
  python3 utility/diagnostic_checkpoint.py path/to/F1_done.pt
  python3 utility/diagnostic_checkpoint.py path/to/F*_done.pt   # all folds + summary table

Optional: pass --features to label the strategy features by name
  python3 utility/diagnostic_checkpoint.py F*_done.pt --features zone_is_bullish,wick_rejection,...
"""
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, '.')
from futures_foundation.model import FFMConfig
from futures_foundation.finetune.model import HybridStrategyModel

SEP = '=' * 62


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'next_fold_state' in ckpt:
        state = ckpt['next_fold_state']
        test_metrics = ckpt.get('test_metrics', {})
    elif 'model_state' in ckpt:
        state = ckpt['model_state']
        test_metrics = {}
    else:
        state = ckpt
        test_metrics = {}

    feat_dim = state['strategy_projection.0.weight'].shape[1]
    cfg = FFMConfig()
    model = HybridStrategyModel(ffm_config=cfg, num_strategy_features=feat_dim, num_labels=2, use_context=True)
    model.load_state_dict(state)
    model.eval()
    return model, test_metrics


def get_feature_names(n, provided=None):
    if provided and len(provided) >= n:
        return provided[:n]
    base = provided or []
    return base + [f'feature_{i}' for i in range(len(base), n)]


def analyze(path, feature_names=None):
    print(f'\n{SEP}')
    print(f'  {path}')
    print(SEP)

    model, tm = load_model(path)

    if tm:
        p80  = tm.get('prec_at_80', tm.get('precision', 0))
        n80  = tm.get('n_at_80', 0)
        prec = tm.get('precision', 0)
        rec  = tm.get('recall', 0)
        print(f'  Test  P:{prec:.3f} R:{rec:.3f} | P@80:{p80:.3f}(N={n80})')

    # --- Feature weight spread ---
    strat_w = model.strategy_projection[0].weight.detach().numpy()  # [hidden, N_feats]
    importance = np.abs(strat_w).mean(axis=0)
    fi_min, fi_max = importance.min(), importance.max()
    spread = fi_max / fi_min
    status = '✅ DIFFERENTIATED' if spread >= 1.3 else '❌ UNDIFFERENTIATED'

    names = get_feature_names(len(importance), feature_names)
    max_name_len = max(len(n) for n in names)
    col = max(max_name_len, 16)

    print(f'\n  Feature spread: {fi_min:.4f}–{fi_max:.4f}  ratio={spread:.2f}x  {status}')
    print()
    ranked = sorted(zip(importance, names), reverse=True)
    for imp, name in ranked:
        bar = '█' * int(imp / fi_max * 24)
        marker = ' ◄' if imp == fi_max else (' ▼' if imp == fi_min else '')
        print(f'    {name:{col}s} {imp:.4f}  {bar}{marker}')

    # --- Fusion split ---
    # Fusion input: [backbone(256) | context(15) | strategy(projected)]
    fusion_w = model.fusion[0].weight.detach().numpy()
    backbone_dim = 256
    context_dim  = 15
    b_imp = np.abs(fusion_w[:, :backbone_dim]).mean()
    c_imp = np.abs(fusion_w[:, backbone_dim:backbone_dim + context_dim]).mean()
    s_imp = np.abs(fusion_w[:, backbone_dim + context_dim:]).mean()
    total = b_imp + c_imp + s_imp
    bp, cp, sp = b_imp / total, c_imp / total, s_imp / total
    bb_status = '✅' if bp > 0.35 else '⚠️ '

    print(f'\n  Fusion split: {bb_status} Backbone:{bp:.1%}  Context:{cp:.1%}  Strategy:{sp:.1%}')
    print(f'  (healthy: backbone >35% — model is leveraging pretrained representations)')

    return {
        'path': path,
        'spread': spread,
        'top_feature': ranked[0][1],
        'bottom_feature': ranked[-1][1],
        'backbone_pct': bp,
        'p80': tm.get('prec_at_80', None),
        'n80': tm.get('n_at_80', None),
    }


def summary_table(results):
    print(f'\n{SEP}')
    print(f'  FOLD COMPARISON SUMMARY')
    print(SEP)
    print(f'  {"File":32s} {"Spread":>8} {"Backbone":>9} {"P@80":>7} {"N@80":>6}')
    print(f'  {"-"*32} {"-"*8} {"-"*9} {"-"*7} {"-"*6}')
    for r in results:
        name = r['path'].split('/')[-1]
        p80  = f"{r['p80']:.3f}" if r['p80'] is not None else '   —  '
        n80  = str(r['n80']) if r['n80'] is not None else '—'
        sp_flag = '✅' if r['spread'] >= 1.3 else '❌'
        bb_flag = '✅' if r['backbone_pct'] > 0.35 else '⚠️ '
        print(f'  {name:32s} {sp_flag}{r["spread"]:5.2f}x  {bb_flag}{r["backbone_pct"]:6.1%}  {p80:>7}  {n80:>6}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checkpoint diagnostic — feature spread + fusion split')
    parser.add_argument('files', nargs='*', default=['temp/F*_done.pt'], help='Path(s) or glob pattern(s) to done.pt files')
    parser.add_argument('--features', type=str, default='', help='Comma-separated strategy feature names (optional)')
    args = parser.parse_args()

    feature_names = [f.strip() for f in args.features.split(',')] if args.features else None

    paths = []
    for p in args.files:
        expanded = sorted(glob.glob(p))
        paths.extend(expanded if expanded else [p])

    if not paths:
        print('No checkpoint files found.')
        print('Usage: python3 utility/diagnostic_checkpoint.py path/to/F*_done.pt')
        sys.exit(1)

    results = []
    for path in paths:
        try:
            r = analyze(path, feature_names=feature_names)
            results.append(r)
        except Exception as e:
            print(f'  ERROR loading {path}: {e}')

    if len(results) > 1:
        summary_table(results)
