"""Contract tests for the local clean SSL production lineage."""
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'mantis_ssl_clean_pipeline.py'
SPEC = importlib.util.spec_from_file_location('mantis_ssl_clean_pipeline', SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def test_clean_pipeline_order_and_parent_chain():
    assert tuple(stage.name for stage in pipeline.STAGES) == (
        'mask', 'contrastive', 'seq2seq', 'nextleg')
    assert tuple(stage.parent for stage in pipeline.STAGES) == (
        None, 'mask', 'contrastive', 'seq2seq')
    assert pipeline.HOLDOUT_START == '2026-01-01'


def test_clean_pipeline_uses_banked_seq2seq_and_nextleg_recipes():
    seq = pipeline._stage_config('seq2seq')
    assert seq['pretext'] == 'forecast'
    assert seq['horizons'] == (5, 10, 20, 25)
    assert seq['context_lengths'] == (64, 100, 150, 200)
    assert seq['holdout_start'] == '2026-01-01'
    assert seq['lora_r'] == 8 and seq['lora_alpha'] == 16
    assert seq['log_every_steps'] == 25
    leg = pipeline._stage_config('nextleg')
    assert leg['pretext'] == 'nextleg' and leg['leg_k'] == 2 and leg['leg_cap'] == 256
    assert leg['holdout_start'] == '2026-01-01'


def test_mps_batches_fit_16gb_and_each_stage_has_distinct_output():
    batches = {stage.name: stage.batch['mps'] for stage in pipeline.STAGES}
    assert batches == {'mask': 256, 'contrastive': 32, 'seq2seq': 128, 'nextleg': 128}
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


def test_master_preserves_virtualenv_python_for_children():
    source = SCRIPT.read_text()
    assert 'Path(sys.executable).absolute()' in source
    assert 'Path(sys.executable).resolve()' not in source
