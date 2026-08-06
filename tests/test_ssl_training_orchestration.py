from __future__ import annotations

import json
from pathlib import Path
import sys

from ml_training_loop import Phase
from ml_training_loop.integrations.reasoning import CodexExecution
from ml_training_loop.skills import NoopSkillBootstrapper

from futures_foundation.orchestration.ssl import (
    ArtifactContract,
    CommandStage,
    MetricRequirement,
    ReportGate,
    SslReasoningConfig,
    SslScientificContext,
    StageRevision,
    SslTrainingWorkflow,
    build_ssl_reasoning_adapter,
    load_ssl_workflow,
    run_ssl_training,
)


class _CodexExecutor:
    def __init__(self):
        self.requests = []

    @property
    def identity(self):
        return {"kind": "fake-ffm-codex-v1"}

    def execute(self, request):
        self.requests.append(request)
        assert "$ml-diagnose-experiment" in request.prompt
        assert "$ml-design-experiment" in request.prompt
        assert "$ml-train-representation" in request.prompt
        assert "causal Volume-Structure SSL" in request.prompt
        assert "sealed 2026 holdout remains inaccessible" in request.prompt
        assert "parent checkpoint sha256:abc123" in request.prompt
        return CodexExecution(
            command=("codex", "exec"),
            returncode=0,
            stdout="",
            stderr="",
            response=json.dumps(
                {
                    "decision": "REVISE",
                    "rationale": "The native representation lift gate failed.",
                    "config_override_json": json.dumps(
                        {
                            "declared_revision_id": "retention_safe_capacity",
                        }
                    ),
                }
            ),
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
    saved = json.loads((tmp_path / "state" / "masked-smoke" / "state.json").read_text())
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
    config.write_text(
        json.dumps(
            {
                "schema": "ffm_ssl_training_workflow_v1",
                "name": "volume-ssl",
                "repository_root": ".",
                "state_root": "runs",
                "max_revisions_per_stage": 1,
                "reasoning": {
                    "provider": "codex",
                    "scientific_context": {
                        "objective": "learn causal volume structure",
                        "causal_setup": ["OHLCV-only windows"],
                        "frozen_constraints": ["sealed holdout is inaccessible"],
                        "experiment_ledger": [],
                    },
                },
                "stages": [
                    {
                        "name": "train",
                        "command": [
                            "python",
                            "scripts/chronos/chronos2_ssl_volume_structure.py",
                        ],
                        "artifacts": [
                            {
                                "path": "output/report.json",
                                "kind": "json",
                                "expected": {
                                    "schema": "ffm_chronos2_volume_structure_ssl_v3",
                                    "status": "complete",
                                },
                            },
                            {
                                "path": "output/checkpoint",
                                "kind": "directory",
                                "required_files": ["adapter_config.json"],
                            },
                        ],
                        "gate": {
                            "report_path": "output/report.json",
                            "requirements": [
                                {
                                    "field": "representation.native_lift",
                                    "operator": ">=",
                                    "value": 0.05,
                                }
                            ],
                        },
                        "revisions": [
                            {
                                "candidate_id": "lower_lr",
                                "rationale": "Reduce forgetting.",
                                "command": ["python", "scripts/retrain.py"],
                                "changed_settings": {"learning_rate": 0.00001},
                            }
                        ],
                    }
                ],
            }
        )
    )

    workflow = load_ssl_workflow(config)

    assert workflow.name == "volume-ssl"
    assert workflow.repository_root == tmp_path.resolve()
    assert workflow.state_root == (tmp_path / "runs").resolve()
    assert workflow.stages[0].command[1].endswith("chronos2_ssl_volume_structure.py")
    assert workflow.stages[0].artifacts[1].required_files == ("adapter_config.json",)
    assert workflow.reasoning.scientific_context.required_skills == (
        "ml-diagnose-experiment",
        "ml-design-experiment",
        "ml-train-representation",
    )
    assert workflow.stages[0].gate.requirements[0].field == (
        "representation.native_lift"
    )
    assert workflow.stages[0].revisions[0].candidate_id == "lower_lr"


def test_ssl_gate_uses_skill_directed_codex_reasoning_for_one_revision(tmp_path):
    writer = tmp_path / "write_lift.py"
    writer.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'status':'complete','representation':{'native_lift':float(sys.argv[2])},"
        "'sealed_2026_touched':False}))\n"
    )
    report = tmp_path / "report.json"
    context = SslScientificContext(
        objective="causal Volume-Structure SSL",
        causal_setup=("OHLCV-only windows", "native Chronos LoRA checkpoint"),
        frozen_constraints=(
            "sealed 2026 holdout remains inaccessible",
            "parent checkpoint sha256:abc123",
            "REAL, time-SHUFFLE, RANDOM controls remain matched",
        ),
        experiment_ledger=(),
    )
    workflow = SslTrainingWorkflow(
        name="volume-ssl-reasoning",
        repository_root=tmp_path,
        state_root=tmp_path / "state",
        reasoning=SslReasoningConfig(scientific_context=context),
        stages=(
            CommandStage(
                name="probe_atlas",
                command=(sys.executable, str(writer), str(report), "0.01"),
                artifacts=(
                    ArtifactContract(
                        path=report,
                        kind="json",
                        expected={"status": "complete", "sealed_2026_touched": False},
                    ),
                ),
                gate=ReportGate(
                    report_path=report,
                    requirements=(
                        MetricRequirement("representation.native_lift", ">=", 0.05),
                    ),
                ),
                revisions=(
                    StageRevision(
                        candidate_id="retention_safe_capacity",
                        rationale="Increase LoRA capacity without changing frozen lineage.",
                        command=(sys.executable, str(writer), str(report), "0.08"),
                        changed_settings={"lora_rank": {"from": 8, "to": 16}},
                    ),
                ),
            ),
        ),
        max_revisions_per_stage=1,
    )
    executor = _CodexExecutor()

    result = run_ssl_training(
        workflow,
        run_id="volume-reasoning",
        skills=NoopSkillBootstrapper(),
        reasoning=build_ssl_reasoning_adapter(workflow, executor=executor),
    )

    assert result.phase is Phase.COMPLETE
    assert result.attempts == {"probe_atlas": 2}
    assert len(result.revisions) == 1
    assert result.revisions[0].config_override["changed_settings"] == {
        "lora_rank": {"from": 8, "to": 16}
    }
    assert len(executor.requests) == 1
