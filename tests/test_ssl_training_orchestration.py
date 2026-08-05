from __future__ import annotations

import json
from pathlib import Path
import sys

from ml_training_loop import Phase
from ml_training_loop.skills import NoopSkillBootstrapper

from futures_foundation.orchestration.ssl import (
    ArtifactContract,
    CommandStage,
    SslTrainingWorkflow,
    load_ssl_workflow,
    run_ssl_training,
)


def _writer(path: Path) -> Path:
    script = path / "write_artifact.py"
    script.write_text(
        """
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps({
    "schema": sys.argv[2],
    "status": sys.argv[3],
}) + "\\n")
""".lstrip()
    )
    return script


def _stage(name: str, script: Path, output: Path, status: str = "complete"):
    schema = f"ffm_test_{name}_v1"
    return CommandStage(
        name=name,
        command=(sys.executable, str(script), str(output), schema, status),
        artifacts=(
            ArtifactContract(
                path=output,
                kind="json",
                expected={"schema": schema, "status": "complete"},
            ),
        ),
    )


def test_ssl_workflow_reuses_commands_and_returns_authenticated_receipts(tmp_path):
    writer = _writer(tmp_path)
    workflow = SslTrainingWorkflow(
        name="masked-ssl-smoke",
        repository_root=tmp_path,
        state_root=tmp_path / "state",
        stages=(
            _stage("preflight", writer, tmp_path / "preflight.json"),
            _stage("train", writer, tmp_path / "train.json"),
            _stage("probe_atlas", writer, tmp_path / "atlas.json"),
        ),
    )

    result = run_ssl_training(
        workflow,
        run_id="masked-smoke",
        skills=NoopSkillBootstrapper(),
    )

    assert result.phase is Phase.COMPLETE
    assert [receipt.stage for receipt in result.receipts] == [
        "preflight",
        "train",
        "probe_atlas",
    ]
    for receipt in result.receipts:
        assert receipt.outputs["returncode"] == 0
        assert Path(receipt.outputs["log_path"]).is_file()
        artifact = next(iter(receipt.outputs["artifact_evidence"].values()))
        assert artifact["kind"] == "json"
        assert len(artifact["sha256"]) == 64
        assert receipt.outputs["command"] == list(
            workflow.stages[result.receipts.index(receipt)].command
        )
    saved = json.loads(
        (tmp_path / "state" / "masked-smoke" / "state.json").read_text()
    )
    assert saved["phase"] == "COMPLETE"


def test_ssl_workflow_blocks_before_next_stage_on_artifact_contract_failure(tmp_path):
    writer = _writer(tmp_path)
    workflow = SslTrainingWorkflow(
        name="invalid-ssl-smoke",
        repository_root=tmp_path,
        state_root=tmp_path / "state",
        stages=(
            _stage(
                "preflight",
                writer,
                tmp_path / "preflight.json",
                status="invalid",
            ),
            _stage("train", writer, tmp_path / "train.json"),
        ),
    )

    result = run_ssl_training(
        workflow,
        run_id="invalid-smoke",
        skills=NoopSkillBootstrapper(),
    )

    assert result.phase is Phase.BLOCKED
    assert [receipt.stage for receipt in result.receipts] == ["preflight"]
    assert "status" in result.message
    assert not (tmp_path / "train.json").exists()


def test_ssl_workflow_loads_existing_ffm_commands_from_json(tmp_path):
    config = tmp_path / "workflow.json"
    config.write_text(json.dumps({
        "schema": "ffm_ssl_training_workflow_v1",
        "name": "volume-ssl",
        "repository_root": ".",
        "state_root": "runs",
        "stages": [{
            "name": "train",
            "command": ["python", "scripts/chronos/chronos2_ssl_volume_structure.py"],
            "artifacts": [{
                "path": "output/report.json",
                "kind": "json",
                "expected": {
                    "schema": "ffm_chronos2_volume_structure_ssl_v3",
                    "status": "complete",
                },
            }, {
                "path": "output/checkpoint",
                "kind": "directory",
                "required_files": ["adapter_config.json"],
            }],
        }],
    }))

    workflow = load_ssl_workflow(config)

    assert workflow.name == "volume-ssl"
    assert workflow.repository_root == tmp_path.resolve()
    assert workflow.state_root == (tmp_path / "runs").resolve()
    assert workflow.stages[0].command[1].endswith(
        "chronos2_ssl_volume_structure.py"
    )
    assert workflow.stages[0].artifacts[1].required_files == (
        "adapter_config.json",
    )
