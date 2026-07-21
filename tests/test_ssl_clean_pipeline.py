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
    leg = pipeline._stage_config('nextleg')
    assert leg['pretext'] == 'nextleg' and leg['leg_k'] == 2 and leg['leg_cap'] == 256
    assert leg['holdout_start'] == '2026-01-01'


def test_mps_batches_fit_16gb_and_each_stage_has_distinct_output():
    batches = {stage.name: stage.batch['mps'] for stage in pipeline.STAGES}
    assert batches == {'mask': 256, 'contrastive': 32, 'seq2seq': 128, 'nextleg': 128}
    assert len({stage.filename for stage in pipeline.STAGES}) == len(pipeline.STAGES)
