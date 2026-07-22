"""Contract tests for the local clean SSL production lineage."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


def test_non_mask_controls_are_explicit_and_bounded(monkeypatch):
    monkeypatch.setenv('SSL_CONTROLS', 'shuffle,random')
    monkeypatch.setenv('CONTROL_EPOCHS', '8')
    cfg = pipeline._stage_config('nextleg')
    assert cfg['controls'] == ('shuffle', 'random')
    assert cfg['control_epochs'] == 8


def test_unknown_non_mask_control_fails_closed(monkeypatch):
    monkeypatch.setenv('SSL_CONTROLS', 'shuffle,oracle')
    with pytest.raises(ValueError, match='unsupported SSL controls'):
        pipeline._stage_config('nextleg')


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


def test_existing_child_can_be_bound_to_an_explicit_external_parent(tmp_path):
    """A partial NextLeg lineage must record and recheck the exact supplied Seq2Seq hash."""
    paths = pipeline._stage_map(tmp_path)
    canonical_parent = paths['seq2seq']
    canonical_parent.write_bytes(b'wrong-local-parent')
    external_parent = tmp_path / 'downloaded-seq2seq.pt'
    external_parent.write_bytes(b'exact-external-parent')
    child = paths['nextleg']
    child.write_bytes(b'uniform-nextleg-child')
    lineage = Path(str(child) + '.data_provenance.json')
    lineage.write_text(json.dumps({
        'stage': 'nextleg',
        'parent': {'sha256': pipeline.sha256(external_parent)},
    }))

    pipeline._assert_stage_parent(
        child, pipeline.STAGES[-1], paths, parent_path=external_parent)
    with pytest.raises(RuntimeError, match='wrong parent'):
        pipeline._assert_stage_parent(child, pipeline.STAGES[-1], paths)


def test_partial_lineage_cli_requires_explicit_parent_surface():
    args = pipeline._parser().parse_args([
        '--start-stage', 'nextleg',
        '--parent-checkpoint', '/tmp/seq2seq.pt',
        '--sampling-mode', 'uniform_stream',
        '--controls', 'shuffle,random',
    ])
    assert args.start_stage == 'nextleg'
    assert args.parent_checkpoint == '/tmp/seq2seq.pt'
    assert args.sampling_mode == 'uniform_stream'
    assert args.controls == 'shuffle,random'


def test_partial_lineage_reuses_only_hash_and_provenance_matched_atlas(tmp_path):
    source = tmp_path / 'source'
    atlas = source / 'probe_atlas'
    atlas.mkdir(parents=True)
    output = tmp_path / 'candidate'
    output.mkdir()
    parent = tmp_path / 'seq2seq.pt'
    parent.write_bytes(b'fixed-seq2seq')
    provenance = {'schema': 'test', 'streams': {'NQ@3min': {'sha256': 'abc'}}}
    labels = atlas / 'trend_lifecycle_labels_pre2026.npz'
    np.savez(labels, ts=np.array(['2025-06-01'], dtype='datetime64[ns]'))
    Path(str(labels) + '.provenance.json').write_text(json.dumps({
        'holdout_start': pipeline.HOLDOUT_START,
        'data_provenance_sha256': pipeline._provenance_sha256(provenance),
    }))
    (atlas / 'seq2seq.json').write_text(json.dumps({
        'schema': 'ffm_probe_atlas_v2',
        'scope': '9x4_strategy_agnostic',
        'fit': '<2024', 'eval': '2025',
        'checkpoint_sha256': pipeline.sha256(parent),
    }))
    pool = {'schema': 'ffm_probe_atlas_pool_v1', 'rows': 1, 'pool_sha256': 'pool'}
    (atlas / 'seq2seq_emb.npy.pool.json').write_text(json.dumps(pool))

    reused = pipeline._reuse_atlas_parent(
        source_dir=source, out_dir=output, stage=pipeline.STAGES[2],
        checkpoint=parent, provenance=provenance)
    assert reused == labels
    assert json.loads((output / 'probe_atlas/seq2seq.json').read_text())[
        'checkpoint_sha256'] == pipeline.sha256(parent)
    assert json.loads((output / 'probe_atlas/seq2seq_emb.npy.pool.json').read_text()) == pool

    wrong_parent = tmp_path / 'wrong.pt'
    wrong_parent.write_bytes(b'not-seq2seq')
    with pytest.raises(RuntimeError, match='different checkpoint'):
        pipeline._reuse_atlas_parent(
            source_dir=source, out_dir=tmp_path / 'wrong-output',
            stage=pipeline.STAGES[2], checkpoint=wrong_parent, provenance=provenance)


def test_atlas_parent_child_pool_identity_must_match(tmp_path):
    atlas = tmp_path / 'probe_atlas'
    atlas.mkdir()
    parent = atlas / 'seq2seq_emb.npy.pool.json'
    child = atlas / 'nextleg_emb.npy.pool.json'
    parent.write_text(json.dumps({'pool_sha256': 'same', 'rows': 12}))
    child.write_text(parent.read_text())
    pipeline._assert_atlas_pool_match(tmp_path, 'seq2seq', 'nextleg')
    child.write_text(json.dumps({'pool_sha256': 'different', 'rows': 12}))
    with pytest.raises(RuntimeError, match='pool mismatch'):
        pipeline._assert_atlas_pool_match(tmp_path, 'seq2seq', 'nextleg')


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


def test_saved_seq2seq_report_is_revalidated_without_retraining(tmp_path):
    report = tmp_path / 'seq.pt.report.json'
    report.write_text(json.dumps({
        'config': {'pretext': 'forecast'},
        'probe': {
            'split_schema': pipeline.PROBE_SPLIT_SCHEMA,
            'mean_core_delta': 0.0429668,
            'descriptive_delta': 0.0574667,
            'fwd_absmove_delta': -0.0006,
            'fwd_dir_delta': -0.0055,
            'learns_regime_vol_structure': True,
        },
        'history': [{'std': 1.0057, 'forecast_skill': 0.058, 'gate_ok': False}],
        'verdict': {
            'all_pass': False, 'representation_pass': False,
            'beats_controls': True, 'real_delta': 0.0429668,
        },
    }))
    refreshed = pipeline._revalidate_stage_report(report)
    assert refreshed['all_pass'] is True
    assert refreshed['representation_pass'] is True
    pipeline._assert_stage_verdict(report)


def test_checkpoint_recovery_distinguishes_interrupted_training_from_finalization(tmp_path):
    checkpoint = tmp_path / 'seq.pt'
    assert pipeline._checkpoint_recovery_flags(checkpoint) == (False, False)
    checkpoint.write_bytes(b'progressive-best')
    assert pipeline._checkpoint_recovery_flags(checkpoint) == (True, False)
    Path(str(checkpoint) + '.real_complete.json').write_text('{}')
    assert pipeline._checkpoint_recovery_flags(checkpoint) == (False, True)


def test_failed_report_with_completed_real_falls_through_to_revalidation(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'nextleg.pt'
    report = tmp_path / 'nextleg.pt.report.json'
    checkpoint.write_bytes(b'checkpoint')
    report.write_text('{}')
    Path(str(checkpoint) + '.real_complete.json').write_text('{}')
    monkeypatch.setattr(pipeline, '_revalidate_stage_report', lambda _path: {})
    monkeypatch.setattr(
        pipeline, '_assert_stage_verdict',
        lambda _path: (_ for _ in ()).throw(RuntimeError('beats_controls=False')))

    reason = pipeline._completed_stage_revalidation_reason(checkpoint, report)

    assert reason == 'beats_controls=False'


def test_failed_report_without_completed_real_remains_fatal(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'nextleg.pt'
    report = tmp_path / 'nextleg.pt.report.json'
    checkpoint.write_bytes(b'partial')
    report.write_text('{}')
    monkeypatch.setattr(pipeline, '_revalidate_stage_report', lambda _path: {})
    monkeypatch.setattr(
        pipeline, '_assert_stage_verdict',
        lambda _path: (_ for _ in ()).throw(RuntimeError('failed verdict')))

    with pytest.raises(RuntimeError, match='failed verdict'):
        pipeline._completed_stage_revalidation_reason(checkpoint, report)


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
