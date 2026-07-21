"""Contract tests for the local clean SSL production lineage."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'mantis_ssl_clean_pipeline.py'
SPEC = importlib.util.spec_from_file_location('mantis_ssl_clean_pipeline', SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)
PRETRAIN_SCRIPT = ROOT / 'scripts' / 'mantis_ssl_pretrain.py'
PRETRAIN_SPEC = importlib.util.spec_from_file_location('mantis_ssl_pretrain', PRETRAIN_SCRIPT)
pretrain = importlib.util.module_from_spec(PRETRAIN_SPEC)
sys.modules[PRETRAIN_SPEC.name] = pretrain
PRETRAIN_SPEC.loader.exec_module(pretrain)


def test_clean_pipeline_order_and_parent_chain():
    assert tuple(stage.name for stage in pipeline.STAGES) == (
        'mask', 'contrastive', 'seq2seq', 'nextleg')
    assert tuple(stage.parent for stage in pipeline.STAGES) == (
        None, 'mask', 'contrastive', 'seq2seq')
    assert pipeline.HOLDOUT_START == '2026-01-01'


def test_clean_pipeline_uses_banked_seq2seq_and_nextleg_recipes():
    contrastive = pipeline._stage_config('contrastive')
    assert contrastive['regime_key'] == 'kaufman'
    assert contrastive['kaufman_chop'] == 0.25
    assert contrastive['kaufman_trend'] == 0.50
    assert contrastive['lr'] == 5e-5
    assert contrastive['vol_weight'] == 0.0
    assert contrastive['freeze_encoder_layers'] == 2
    assert contrastive['crop_max'] == 0.0 and contrastive['aug_tmask'] == 0.0
    seq = pipeline._stage_config('seq2seq')
    assert seq['pretext'] == 'forecast'
    assert seq['horizons'] == (5, 10, 20, 25)
    assert seq['context_lengths'] == (64, 100, 150, 200)
    assert seq['holdout_start'] == '2026-01-01'
    assert seq['lora_r'] == 8 and seq['lora_alpha'] == 16
    assert seq['sampling_mode'] == 'bar_proportional'
    assert seq['log_every_steps'] == 25
    assert seq['lr'] == 4e-5 and seq['freeze_encoder_layers'] == 2
    leg = pipeline._stage_config('nextleg')
    assert leg['pretext'] == 'nextleg' and leg['leg_k'] == 2 and leg['leg_cap'] == 256
    assert leg['holdout_start'] == '2026-01-01'
    assert leg['lr'] == 3e-5 and leg['freeze_encoder_layers'] == 2


def test_clean_pipeline_stage_learning_rates_are_overridable(monkeypatch):
    monkeypatch.setenv('CONTRASTIVE_LR', '6e-5')
    monkeypatch.setenv('SEQ2SEQ_LR', '5e-5')
    monkeypatch.setenv('NEXTLEG_LR', '4e-5')
    assert pipeline._stage_config('contrastive')['lr'] == 6e-5
    assert pipeline._stage_config('seq2seq')['lr'] == 5e-5
    assert pipeline._stage_config('nextleg')['lr'] == 4e-5


def test_clean_pipeline_uniform_stream_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv('SAMPLING_MODE', 'uniform_stream')
    for name in ('contrastive', 'seq2seq', 'nextleg'):
        assert pipeline._stage_config(name)['sampling_mode'] == 'uniform_stream'


def test_completed_checkpoint_cannot_cross_sampling_recipes(tmp_path):
    report = tmp_path / 'stage.pt.report.json'
    report.write_text(json.dumps({'config': {'sampling_mode': 'bar_proportional'}}))
    pipeline._assert_completed_stage_recipe(report, sampling_mode='bar_proportional')
    with pytest.raises(RuntimeError, match='choose a new --out-dir'):
        pipeline._assert_completed_stage_recipe(report, sampling_mode='uniform_stream')


def test_legacy_report_is_treated_as_bar_proportional(tmp_path):
    report = tmp_path / 'legacy.pt.report.json'
    report.write_text(json.dumps({'config': {}}))
    pipeline._assert_completed_stage_recipe(report, sampling_mode='bar_proportional')


def test_existing_child_must_match_canonical_parent_hash(tmp_path):
    paths = pipeline._stage_map(tmp_path)
    parent = paths['contrastive']
    seq = paths['seq2seq']
    parent.write_bytes(b'contrastive-parent')
    seq.write_bytes(b'seq-child')
    lineage = Path(str(seq) + '.data_provenance.json')
    lineage.write_text(json.dumps({
        'stage': 'seq2seq',
        'parent': {'sha256': pipeline.sha256(parent)},
    }))
    pipeline._assert_stage_parent(seq, pipeline.STAGES[2], paths)

    lineage.write_text(json.dumps({
        'stage': 'seq2seq',
        'parent': {'sha256': 'stale-contrastive-parent'},
    }))
    with pytest.raises(RuntimeError, match='wrong parent'):
        pipeline._assert_stage_parent(seq, pipeline.STAGES[2], paths)


def test_stage_verdict_requires_current_probe_and_both_gates(tmp_path):
    report = tmp_path / 'stage.pt.report.json'
    good = {
        'probe': {'split_schema': pipeline.PROBE_SPLIT_SCHEMA},
        'verdict': {'all_pass': True, 'representation_pass': True, 'beats_controls': True},
    }
    report.write_text(json.dumps(good))
    pipeline._assert_stage_verdict(report)
    report.write_text(json.dumps({**good, 'verdict': {
        'all_pass': False, 'representation_pass': True, 'beats_controls': False,
    }}))
    with pytest.raises(RuntimeError, match='beats_controls=False'):
        pipeline._assert_stage_verdict(report)
    report.write_text(json.dumps({'probe': {}, 'verdict': {'all_pass': True}}))
    with pytest.raises(RuntimeError, match='stale probe split'):
        pipeline._assert_stage_verdict(report)


def test_mps_batches_fit_16gb_and_each_stage_has_distinct_output():
    batches = {stage.name: stage.batch['mps'] for stage in pipeline.STAGES}
    assert batches == {'mask': 256, 'contrastive': 64, 'seq2seq': 128, 'nextleg': 128}
    assert len({stage.filename for stage in pipeline.STAGES}) == len(pipeline.STAGES)


def test_batch_changes_preserve_the_sample_budget_and_oom_fallback():
    for stage in pipeline.STAGES:
        batch = stage.batch['mps']
        steps = pipeline._steps_for(stage, batch, {})
        assert batch * steps >= stage.samples_per_epoch
        fallback = batch // 2
        assert pipeline._steps_for(stage, fallback, {}) == steps * 2


def test_explicit_step_override_wins_over_sample_budget():
    stage = pipeline.STAGES[0]
    assert pipeline._steps_for(stage, 512, {'STEPS_PER_EPOCH': '7'}) == 7
    assert pipeline._steps_for(stage, 512, {'MASK_STEPS': '9',
                                            'STEPS_PER_EPOCH': '7'}) == 9


def test_probe_atlas_progress_compares_each_completed_stage(tmp_path):
    atlas = tmp_path / 'probe_atlas'; atlas.mkdir()
    for i, stage in enumerate(pipeline.STAGES[:2]):
        payload = {'checkpoint_sha256': str(i), 'probes': {
            'ret': {'family': 'retention', 'auc': 0.60 + i * 0.02},
            'pred': {'family': 'prediction', 'auc': 0.52 + i * 0.03}}}
        (atlas / f'{stage.name}.json').write_text(__import__('json').dumps(payload))
    progress = pipeline._refresh_atlas_progress(tmp_path)
    assert [row['stage'] for row in progress['stages']] == ['mask', 'contrastive']
    assert progress['stages'][1]['deltas_vs_previous'] == {'pred': 0.03, 'ret': 0.02}
    assert (tmp_path / 'probe_atlas_progress.json').is_file()


def test_probe_atlas_retention_gate_rejects_broad_or_single_probe_forgetting():
    base = {'stage': 'mask', 'retention_mean_auc': 0.80,
            'prediction_mean_auc': 0.70, 'deltas_vs_previous': {}}
    good = {'stage': 'contrastive', 'retention_mean_auc': 0.795,
            'prediction_mean_auc': 0.695,
            'deltas_vs_previous': {'trend_context': -0.02, 'ret_squeeze': 0.01}}
    pipeline._assert_atlas_retention({'stages': [base, good]}, 'contrastive')

    broad = {**good, 'prediction_mean_auc': 0.68}
    with pytest.raises(RuntimeError, match='prediction_mean_auc'):
        pipeline._assert_atlas_retention({'stages': [base, broad]}, 'contrastive')

    narrow = {**good, 'deltas_vs_previous': {'trend_context': -0.04}}
    with pytest.raises(RuntimeError, match='trend_context'):
        pipeline._assert_atlas_retention({'stages': [base, narrow]}, 'contrastive')


def test_clean_pipeline_probe_dependencies_are_public_scripts():
    source = SCRIPT.read_text()
    assert 'ROOT / "colabs"' not in source
    assert 'ROOT / "scripts" / "probe_atlas.py"' in source
    assert 'ROOT / "scripts" / "generate_trend_labels.py"' in source
    assert 'ROOT / "colabs"' not in source
    assert '"DATA_DIR": str(data_dir)' in source


def test_probe_atlas_receives_external_data_directory(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'mask.pt'
    checkpoint.write_bytes(b'checkpoint')
    labels = tmp_path / 'labels.npz'
    labels.write_bytes(b'labels')
    data_dir = tmp_path / 'drive-data'
    data_dir.mkdir()
    captured = {}

    def run_logged(_command, *, env, log_path):
        captured.update(env)
        result = Path(env['ATLAS_OUT'])
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({
            'checkpoint_sha256': env['CKPT_SHA256'], 'probes': {},
        }))
        return 0, False

    monkeypatch.setattr(pipeline, '_run_logged', run_logged)
    pipeline._run_probe_atlas(
        pipeline.STAGES[0], checkpoint, out_dir=tmp_path, device='cuda',
        python=Path(sys.executable), labels=labels, data_dir=data_dir)
    assert captured['DATA_DIR'] == str(data_dir)


def test_master_preserves_virtualenv_python_for_children():
    source = SCRIPT.read_text()
    assert 'Path(sys.executable).absolute()' in source
    assert 'Path(sys.executable).resolve()' not in source


def test_logged_child_survives_durable_log_eio(tmp_path, monkeypatch):
    class BrokenLog:
        def write(self, _text):
            raise OSError(5, 'Input/output error')

        def close(self):
            raise OSError(5, 'Input/output error')

    real_open = Path.open
    durable = tmp_path / 'drive' / 'stage.log'

    def flaky_open(path, *args, **kwargs):
        if path == durable:
            return BrokenLog()
        return real_open(path, *args, **kwargs)

    fallback = tmp_path / 'runtime'
    monkeypatch.setattr(Path, 'open', flaky_open)
    code, oom = pipeline._run_logged(
        [sys.executable, '-c', 'print("still-running")'],
        env={**__import__('os').environ, 'FFM_LOCAL_LOG_DIR': str(fallback)},
        log_path=durable)
    assert code == 0 and oom is False
    assert 'still-running' in (fallback / durable.name).read_text()


def test_mask_controls_default_to_eight_epochs_and_resume_is_explicit(monkeypatch):
    parser = pipeline._parser()
    args = parser.parse_args([])
    assert args.control_epochs == 8
    assert args.reuse_mask_real is False
    monkeypatch.setenv('REUSE_MASK_REAL', '1')
    assert pipeline._parser().parse_args([]).reuse_mask_real is True


def test_mask_preflight_allows_checkpoint_only_finalization(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    checkpoint = tmp_path / 'mask.pt'
    checkpoint.write_bytes(b'saved-real')
    streams = tuple((ticker, timeframe) for ticker in pretrain.TICKERS
                    for timeframe in pretrain.TIMEFRAMES)
    monkeypatch.setattr(pretrain, 'seal_continuous_streams', lambda *args, **kwargs: {
        'streams': [{'ticker': ticker, 'timeframe': timeframe}
                    for ticker, timeframe in streams],
    })
    args = pretrain._parser().parse_args([
        '--data-dir', str(data_dir), '--out', str(checkpoint),
        '--reuse-real-checkpoint',
    ])
    _, resolved, _ = pretrain._preflight(args)
    assert resolved == checkpoint.resolve()


def test_mask_preflight_requires_checkpoint_for_reuse(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    streams = tuple((ticker, timeframe) for ticker in pretrain.TICKERS
                    for timeframe in pretrain.TIMEFRAMES)
    monkeypatch.setattr(pretrain, 'seal_continuous_streams', lambda *args, **kwargs: {
        'streams': [{'ticker': ticker, 'timeframe': timeframe}
                    for ticker, timeframe in streams],
    })
    args = pretrain._parser().parse_args([
        '--data-dir', str(data_dir), '--out', str(tmp_path / 'missing.pt'),
        '--reuse-real-checkpoint',
    ])
    with pytest.raises(SystemExit, match='REAL checkpoint does not exist'):
        pretrain._preflight(args)
