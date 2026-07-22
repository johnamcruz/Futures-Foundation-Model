"""Final-model train + held-out OOS (finetune/produce.py) — torch-free.

Uses a synthetic labeler + the 'logistic' classifier so the produce step (train <
cutoff, score the holdout once, SHUFFLE control) is exercised end-to-end through
the subprocess worker, no torch.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import produce
from futures_foundation.finetune.classifier import Classifier, register_classifier


class SyntheticLabeler:
    n_classes = 2

    def __init__(self, n_bars=1600, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'SYN', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        idx = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        return [self.W[i, 0] for i in idx], self.y[idx], [(int(i),) for i in idx]

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[0]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def mv_feature_names(self):
        return [f'ch{i}' for i in range(self.C)]

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


def test_train_final_oos_beats_shuffle():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    out = produce.train_final(lab, classifier='logistic', holdout_start='2023-06-01',
                              seed=0, verbose=False)
    for k in ('oos_auc', 'best_val_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle',
              'n_train', 'n_oos', 'oos_trades', 'beats_shuffle'):
        assert k in out
    assert out['n_oos'] > 0 and out['oos_trades'] > 0
    assert out['oos_auc'] > 0.8                     # learnable signal holds OOS
    assert out['beats_shuffle'] is True
    assert out['edge_shuffle'] >= produce.PASS_LIFT_MARGIN_R


def test_train_final_insufficient_oos_raises():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    with pytest.raises(ValueError):
        produce.train_final(lab, classifier='logistic', holdout_start='2099-01-01',
                            verbose=False)


# ---- operating-point / WR-by-score readouts (the '1-2 A+ trades/day' deploy view) -----------
class _RLabeler:
    """Minimal labeler: each key carries its realized R at k[4]; evaluate returns R for taken."""
    def evaluate(self, keys, preds):
        return np.array([k[4] for k, p in zip(keys, preds) if p == 1], float)


def _ranked_oos():
    # 5 wins (+2R) then 5 losses (-1R); proba ranks the wins first; 10 trades across 5 days
    R = [2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    keys = [('ES@3min', i, 1, 0.0, r) for i, r in enumerate(R)]
    proba = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.40, 0.30, 0.20, 0.10, 0.05])
    ts = pd.to_datetime([f'2026-01-0{d}' for d in [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]], utc=True)
    return _RLabeler(), keys, proba, ts


def test_operating_points_top_slice_is_all_winners():
    lab, keys, proba, ts = _ranked_oos()
    rows = produce.operating_points(lab, keys, proba, ts, rates=(5, 2, 1))
    by = {r['rate']: r for r in rows}
    assert by[1]['days'] == 5
    assert by[1]['n'] == 5 and by[1]['wr3R'] == 1.0 and by[1]['meanR'] == pytest.approx(2.0)
    assert by[2]['n'] == 10 and by[2]['wr3R'] == pytest.approx(0.5)   # 2/day*5 -> all 10
    # tighter rate = higher (or equal) WR — the whole point of the operating-point view
    assert by[1]['wr3R'] >= by[2]['wr3R']


def test_operating_points_uses_exact_strategy_win_truth():
    class _ExactLab(_RLabeler):
        def win_truth(self, keys):
            return np.array([k[5] for k in keys], bool)

    # Both exits made money, but only the first touched the target. WR must not count the
    # positive vertical-barrier mark as an exact target-first win.
    keys = [('ES@3min', 1, 1, 0.0, 2.95, True),
            ('ES@3min', 2, 1, 0.0, 0.40, False)]
    rows = produce.operating_points(_ExactLab(), keys, np.array([.9, .8]),
                                    pd.to_datetime(['2025-01-02', '2025-01-02'], utc=True),
                                    rates=(2,))
    assert rows[0]['wr3R'] == pytest.approx(.5)
    assert rows[0]['meanR'] == pytest.approx(1.675)


def test_target_operating_points_reports_each_strategy_rung():
    class _LadderLab(_RLabeler):
        def evaluate_targets(self, keys, preds):
            return {
                2.0: np.array([k[4] for k, p in zip(keys, preds) if p == 1], float),
                3.0: np.array([k[5] for k, p in zip(keys, preds) if p == 1], float),
            }

        def target_win_truth(self, keys):
            return {
                2.0: np.array([k[6] for k in keys], bool),
                3.0: np.array([k[7] for k in keys], bool),
            }

    keys = [
        ('NQ@3min', 1, 1, 0.0, 1.97, 2.97, True, True),
        ('NQ@3min', 2, 1, 0.0, 1.97, -1.03, True, False),
        ('NQ@3min', 3, 1, 0.0, -1.03, -1.03, False, False),
        ('NQ@3min', 4, 1, 0.0, -1.03, -1.03, False, False),
    ]
    score = np.array([.9, .8, .2, .1])
    ts = pd.to_datetime(['2026-01-02'] * 4, utc=True)
    rows = produce.target_operating_points(_LadderLab(), keys, score, ts, rates=(2,))
    assert rows[0]['n'] == 2
    assert rows[0]['targets']['2']['hit_rate'] == 1.0
    assert rows[0]['targets']['2']['meanR'] == pytest.approx(1.97)
    assert rows[0]['targets']['3']['hit_rate'] == .5
    assert rows[0]['targets']['3']['meanR'] == pytest.approx(.97)


def test_target_operating_points_is_optional_for_legacy_labelers():
    lab, keys, score, ts = _ranked_oos()
    assert produce.target_operating_points(lab, keys, score, ts) is None


def test_selection_target_audit_threads_standardized_score_to_strategy_hook():
    class _AuditLab:
        def selection_target_audit(self, keys, score, ts):
            return {'n': len(keys), 'top': int(np.argmax(score)), 'days': len(set(ts))}

    keys = [('NQ@3min', 1), ('NQ@3min', 2)]
    score = np.array([.2, .9])
    ts = [np.datetime64('2025-01-02'), np.datetime64('2025-01-03')]
    assert produce.selection_target_audit(_AuditLab(), keys, score, ts) == {
        'n': 2, 'top': 1, 'days': 2,
    }
    assert produce.selection_target_audit(object(), keys, score, ts) is None


def test_wr_by_score_bands_are_monotone():
    lab, keys, proba, ts = _ranked_oos()
    bands = produce.wr_by_score(lab, keys, proba, ts)
    assert sum(b['n'] for b in bands) == len(keys)          # bands partition the OOS set
    wrs = [b['wr3R'] for b in bands if b['n'] > 0]
    assert wrs[0] >= wrs[-1]                                 # top band cleaner than bottom
    assert bands[0]['wr3R'] == 1.0                           # top decile = pure winners here
    for b in bands:                                          # trades/day = n / days(=5)
        assert b['per_day'] == pytest.approx(b['n'] / 5)


def test_fixed_probability_bands_expose_the_point_eight_to_point_nine_economics():
    lab, keys, proba, ts = _ranked_oos()
    rows = produce.wr_by_probability(lab, keys, proba, ts)
    assert sum(row['n'] for row in rows) == len(keys)
    band = next(row for row in rows if row['lo'] == .8 and row['hi'] == .9)
    assert band['n'] == 2
    assert band['wr3R'] == 1.0 and band['meanR'] == pytest.approx(2.0)
    assert band['per_day'] == pytest.approx(.4)


def test_alignment_breakdown_counter_vs_aligned():
    # counter pivots: model score separates winners (score high -> R=+2) from losers; aligned all
    # mediocre. The readout must show counter's TOP-score WR >> counter's base WR (sighted gating).
    class _AlignLab(_RLabeler):
        def htf_alignment(self, keys):
            return np.array([k[5] for k in keys])            # alignment stashed at k[5]
    n = 120
    keys, proba, ts = [], [], []
    for i in range(n):
        counter = i < 60
        good = (i % 2 == 0)
        r = 2.0 if good else -1.0
        keys.append(('ES@3min', i, 1, 0.0, r, -1 if counter else 1))
        proba.append((0.9 if good else 0.1) if counter else 0.5)   # score separates ONLY counter
        ts.append(np.datetime64('2026-01-0%d' % ((i % 5) + 1)))
    ab = produce.alignment_breakdown(_AlignLab(), keys, np.array(proba), ts)
    assert set(ab) == {'counter', 'aligned'}
    assert ab['counter']['base_wr3R'] == pytest.approx(0.5)
    top = ab['counter']['ops'][-1]                           # tightest rate = highest scores
    assert top['wr3R'] == 1.0                                # sighted: top counter picks all win
    assert ab['aligned']['base_wr3R'] == pytest.approx(0.5)
    # no htf_alignment on the labeler -> None (backward-compatible)
    assert produce.alignment_breakdown(_RLabeler(), keys, np.array(proba), ts) is None


def test_regime_breakdown_reports_exact_top_decile_without_filtering_pool():
    class _RegimeLab(_RLabeler):
        def regime_bucket(self, keys):
            return np.array([k[5] for k in keys], object)

        def win_truth(self, keys):
            return np.array([k[6] for k in keys], bool)

    keys, score = [], []
    for regime in ('normal', 'expanded'):
        for i in range(100):
            exact = i < 20
            # Exact winners rank first inside each regime; positive unresolved exits deliberately
            # remain non-wins so the audit cannot inflate target-first WR.
            r = 2.95 if exact else (0.25 if i < 40 else -1.05)
            keys.append(('ES@3min', i, 1, 0.0, r, regime, exact))
            score.append(1.0 - i / 100)
    out = produce.regime_breakdown(_RegimeLab(), keys, np.asarray(score))
    assert set(out) == {'normal', 'expanded'}
    for row in out.values():
        assert row['n'] == 100 and row['top10_n'] == 10
        assert row['base_wr3R'] == pytest.approx(.20)
        assert row['top10_wr3R'] == pytest.approx(1.0)


def test_operating_points_empty_is_safe():
    assert produce.operating_points(_RLabeler(), [], np.array([]), []) == []
    assert produce.wr_by_score(_RLabeler(), [], np.array([]), []) == []
    assert produce.selection_concentration([], np.array([]), []) == []


def test_selection_concentration_exposes_single_stream_dependency():
    keys = ([('CL@3min', i, 1, 0.0, 2.0) for i in range(8)]
            + [('ES@3min', 100 + i, 1, 0.0, 2.0) for i in range(2)])
    score = np.linspace(1.0, .1, 10)
    ts = pd.to_datetime(['2025-01-02'] * 10, utc=True)
    row = produce.selection_concentration(keys, score, ts, rates=(5,))[0]
    assert row['n'] == 5 and row['active_streams'] == 1
    assert row['max_stream_share'] == 1.0 and row['stream_hhi'] == 1.0
    assert row['top_streams'] == {'CL@3min': 1.0}


def test_keyed_classifier_receives_eval_keys_for_auxiliary_audit():
    seen = []

    @register_classifier('_test_keyed_eval_audit')
    class _KeyedAuditClassifier(Classifier):
        def __init__(self, **cfg):
            self.cfg = cfg

        def featurize(self, labeler, keys):
            raise NotImplementedError

        def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0, keys_tr=None,
                        keys_val=None, keys_eval=None):
            seen.append((list(keys_tr), list(keys_val), list(keys_eval)))
            self._forecast_metrics = {'oos': {'aux': {'auc': .6}}}
            self._chop_metrics = {'oos': {'name': 'CHOP20', 'auc': .55}}
            self._chop_platt = (.8, -.1)
            self._chop_percentiles = {'ES@3min': {'p50': .4, 'p75': .6, 'p90': .8}}
            self._chop_abstention_audit = {'oos': {'p99': {'keep_75': {'mean_r': 1.2}}}}
            self._chop_fusion = {'architecture': 'confidence_tapered_separate_tower_v3',
                                 'fusion_alpha': .12, 'hard_gate': False}
            return (np.linspace(.1, .9, len(Xval)),
                    np.linspace(.1, .9, len(Xeval)), .55)

    train_keys = [('ES@3min', i, 1, 0.0, 2.0) for i in range(12)]
    val_keys = [('ES@3min', 100 + i, 1, 0.0, 2.0) for i in range(4)]
    eval_keys = [('ES@3min', 200 + i, 1, 0.0, 2.0 if i % 2 else -1.0)
                 for i in range(6)]
    out = produce._fit_score(
        '_test_keyed_eval_audit', {'requires_keys': True}, _RLabeler(),
        np.zeros((12, 2)), np.arange(12) % 2,
        np.zeros((4, 2)), np.arange(4) % 2,
        np.zeros((6, 2)), eval_keys, np.arange(6) % 2, 0, False,
        keys_tr=train_keys, keys_val=val_keys)
    assert len(seen) == 2  # real and label-shuffle control
    assert seen[0][2] == eval_keys and seen[1][2] == eval_keys
    assert out['forecast_metrics'] == {'oos': {'aux': {'auc': .6}}}
    assert out['chop_metrics']['oos']['auc'] == .55
    assert out['chop_platt'] == (.8, -.1)
    assert out['chop_percentiles']['ES@3min']['p75'] == .6
    assert out['chop_abstention_audit']['oos']['p99']['keep_75']['mean_r'] == 1.2
    assert out['chop_fusion']['architecture'] == 'confidence_tapered_separate_tower_v3'


def test_ladder_signal_contract_shape(tmp_path):
    # the reach-ladder produce must write a ladder-shaped signal.json: entry_signal=p_3r, both head
    # outputs, reach_targets, and calibration baked into the onnx (bot reads p_3r as-is).
    import json
    out = dict(oos_auc=0.6, oos_meanR=0.63, shuffle_meanR=None, edge_shuffle=None,
               oos_trades=1000, n_train=100, n_oos=50, platt=None,
               chop_fusion={'architecture': 'confidence_tapered_separate_tower_v3',
                            'fusion_alpha': .12, 'hard_gate': False},
               entry_thresholds={'top0.1pct': 4.2, 'top1pct': 3.3, 'top10pct': 2.8},
               val_percentiles={'ES@3min': {'p50': 1.7, 'p90': 3.1, 'p99': 4.4}})
    ck = {'rank': 'expected_reach', 'reach_targets': [2.0, 3.0, 4.0, 6.0, 8.0], 'head': 'mlp'}
    lab = SyntheticLabeler(n_bars=200)
    produce._emit(out, 'mantis_frozen', ck, lab, None, None, 5, 1, ['c0'], ['ES'], ['3min'],
                  '2026-01-01', True, str(tmp_path / 'model'), verbose=False)
    c = json.loads((tmp_path / 'model_signal.json').read_text())
    assert c['head_type'] == 'reach_ladder'
    assert c['entry_signal'] == 'expected_reach'           # the VALIDATED entry signal (80% WR tiers)
    assert c['entry_rule'] == 'enter if expected_reach >= entry_thresholds[tier]'
    assert c['entry_thresholds'] == {'top0.1pct': 4.2, 'top1pct': 3.3, 'top10pct': 2.8}
    assert c['head_outputs'] == ['p_3r', 'expected_reach']
    assert c['reach_targets'] == [2.0, 3.0, 4.0, 6.0, 8.0]
    assert c['calibration']['baked_into_onnx'] is True
    assert c['auxiliary_metrics']['chop_fusion']['fusion_alpha'] == .12
    # STANDARDIZED 0-100 SCORE (2026-07-17): the ladder MUST carry the per-stream percentile scale
    # too — same 0-100 axis as the single head, ranking expected_reach, so a head swap needs no bot
    # change. Regression guard for the wiring gap where only the single head emitted it.
    ss = c['score_scale']
    assert ss['kind'] == 'per_stream_val_percentile' and ss['signal'] == 'expected_reach'
    assert ss['percentiles'] == {'ES@3min': {'p50': 1.7, 'p90': 3.1, 'p99': 4.4}}
    assert ss['p_min'] == 1.7                               # VAL-median E[R] backstop
    assert 'expected_reach' in ss['rule']


def test_contract_window_recipe_for_multi_tf(tmp_path):
    # a labeler that declares MV_AGG=(1,5) gets an explicit window_recipe in signal.json (how the
    # bot builds the [10, seq] input); single-TF labelers get None (backward-compatible).
    import json
    out = dict(oos_auc=0.6, oos_meanR=0.5, shuffle_meanR=None, edge_shuffle=None,
               oos_trades=10, n_train=100, n_oos=50, platt=None)
    lab = SyntheticLabeler(n_bars=200)
    lab.MV_AGG, lab.MV_SEQ, lab.MV_MODE = (1, 5), 64, 'ohlcv_agg1-5'
    produce._emit(out, 'mantis_frozen', {'head': 'mlp'}, lab, None, None, 10, 64,
                  None, ['ES'], ['3min'], '2026-01-01', True, str(tmp_path / 'm'), verbose=False)
    c = json.loads((tmp_path / 'm_signal.json').read_text())
    r = c['window_recipe']
    assert r['factors'] == [1, 5] and r['mv_seq'] == 64
    assert 'O=first' in r['aggregation'] and 'anchored' in r['aggregation']
    # single-TF -> no recipe (bot keeps feeding the plain [5, seq] window)
    lab2 = SyntheticLabeler(n_bars=200)
    produce._emit(dict(out), 'mantis_frozen', {'head': 'mlp'}, lab2, None, None, 5, 64,
                  None, ['ES'], ['3min'], '2026-01-01', True, str(tmp_path / 'm2'), verbose=False)
    assert json.loads((tmp_path / 'm2_signal.json').read_text())['window_recipe'] is None


def test_ticker_generalization_requires_every_ticker_and_rate_positive():
    def rows(values):
        return [dict(rate=r, meanR=v, rate_met=True)
                for r, v in zip((5, 3, 2, 1), values)]

    good = produce.ticker_generalization({
        'ES': rows((.1, .2, .3, .4)),
        'NQ': rows((.2, .3, .4, .5)),
    })
    assert good['passed'] is True and good['failures'] == []

    bad = produce.ticker_generalization({
        'ES': rows((.1, .2, .3, .4)),
        'NQ': rows((.2, -.01, .4, .5)),
    })
    assert bad['passed'] is False
    assert bad['failures'] == ['NQ@3/day']


def test_ticker_generalization_fails_missing_or_unsustainable_rate():
    out = produce.ticker_generalization({
        'GC': [dict(rate=5, meanR=.2, rate_met=True),
               dict(rate=3, meanR=.3, rate_met=False)],
    })
    assert out['passed'] is False
    assert out['failures'] == ['GC@3/day', 'GC@2/day', 'GC@1/day']


def test_train_final_writes_signal_contract(tmp_path):
    import json
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    out = produce.train_final(lab, classifier='logistic', holdout_start='2023-06-01',
                              seed=0, export_onnx=True,
                              output_path=str(tmp_path / 'mantis_fractal'), verbose=False)
    assert 'artifacts' in out
    cpath = out['artifacts']['contract']
    c = json.loads(open(cpath).read())
    # self-describing contract shape (mirrors pipeline.produce's signal.json)
    for k in ('contract_version', 'role', 'input', 'channel_names', 'ft_config',
              'n_classes', 'oos_metrics', 'train_scope', 'content_sha'):
        assert k in c
    assert c['input']['channels'] == lab.C
    assert c['channel_names'] == [f'ch{i}' for i in range(lab.C)]
    assert c['oos_metrics']['oos_auc'] is not None
    # logistic has no ONNX export -> onnx/sha are null but the contract still writes
    assert c['onnx'] is None and c['content_sha'] is None
