"""Run existing FFM SSL commands through the shared training state machine.

This module owns orchestration only. Data loading, objectives, optimization,
checkpoint creation, Probe Atlas, and promotion evidence remain implemented by
their existing FFM commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ml_training_loop import (
    Decision,
    GateResult,
    StageReceipt,
    StageSpec,
    TrainingLoop,
    TrainingPlan,
)
from ml_training_loop.adapters import DictAdapterRegistry
from ml_training_loop.cli import DEFAULT_BUNDLE
from ml_training_loop.interfaces import (
    GateRequest,
    ReasoningAdapter,
    SkillBootstrapper,
    StageRequest,
)
from ml_training_loop.skills import BundledSkillBootstrapper
from ml_training_loop.stores import JsonRunStore


_STAGE_ADAPTER = "ffm-command"
_GATE_ADAPTER = "ffm-artifact-contract"
_ARTIFACT_KINDS = frozenset({"file", "directory", "json"})


@dataclass(frozen=True)
class ArtifactContract:
    """One artifact that must exist and authenticate after a command."""

    path: Path
    kind: str = "file"
    expected: Mapping[str, Any] = field(default_factory=dict)
    required_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact kind: {self.kind}")
        if self.kind != "directory" and self.required_files:
            raise ValueError("required_files apply only to directory artifacts")
        if self.kind != "json" and self.expected:
            raise ValueError("expected fields apply only to JSON artifacts")


@dataclass(frozen=True)
class CommandStage:
    """An existing FFM command plus its observable completion contract."""

    name: str
    command: tuple[str, ...]
    artifacts: tuple[ArtifactContract, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    required_skills: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.command:
            raise ValueError("command stage requires a name and command")
        if not self.artifacts:
            raise ValueError("command stage requires at least one artifact contract")
        if any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("command arguments must be non-empty strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class SslTrainingWorkflow:
    """Frozen FFM declaration for preflight, SSL, and representation evidence."""

    name: str
    repository_root: Path
    state_root: Path
    stages: tuple[CommandStage, ...]
    max_revisions_per_stage: int = 0

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if not self.name or not self.stages:
            raise ValueError("SSL workflow requires a name and stages")
        if len(names) != len(set(names)):
            raise ValueError("SSL workflow stage names must be unique")
        if not self.repository_root.expanduser().resolve().is_dir():
            raise ValueError("repository_root must be an existing directory")
        if self.max_revisions_per_stage < 0:
            raise ValueError("max revisions must be nonnegative")


def _resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_ssl_workflow(path: Path) -> SslTrainingWorkflow:
    """Load the stable FFM JSON interface for an existing-command workflow."""
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SSL workflow is unreadable: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("SSL workflow root must be a JSON object")
    if payload.get("schema") != "ffm_ssl_training_workflow_v1":
        raise ValueError("unsupported SSL workflow schema")
    base = source.parent
    try:
        stages = tuple(
            CommandStage(
                name=item["name"],
                command=tuple(item["command"]),
                artifacts=tuple(
                    ArtifactContract(
                        path=Path(contract["path"]),
                        kind=contract.get("kind", "file"),
                        expected=contract.get("expected", {}),
                        required_files=tuple(contract.get("required_files", ())),
                    )
                    for contract in item["artifacts"]
                ),
                environment=item.get("environment", {}),
                timeout_seconds=item.get("timeout_seconds"),
                required_skills=tuple(item.get("required_skills", ())),
            )
            for item in payload["stages"]
        )
        return SslTrainingWorkflow(
            name=payload["name"],
            repository_root=_resolve_from(base, payload.get("repository_root", ".")),
            state_root=_resolve_from(base, payload.get("state_root", ".ml-training-loop")),
            stages=stages,
            max_revisions_per_stage=int(payload.get("max_revisions_per_stage", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid SSL workflow contract: {error}") from error


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_artifact(contract: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    path = Path(contract["path"])
    kind = contract["kind"]
    if kind == "directory":
        if not path.is_dir():
            return f"directory is missing: {path}", {}
        for relative in contract.get("required_files", ()):
            if not (path / relative).is_file():
                return f"required file is missing: {path / relative}", {}
    else:
        if not path.is_file():
            return f"file is missing: {path}", {}
        if kind == "json":
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return f"JSON is unreadable: {path}: {error}", {}
            if not isinstance(payload, dict):
                return f"JSON root is not an object: {path}", {}
            for field, expected in contract.get("expected", {}).items():
                actual = payload.get(field)
                if actual != expected:
                    return (
                        f"field {field!r} expected {expected!r} "
                        f"but received {actual!r}: {path}",
                        {},
                    )
    return None, {
        "kind": kind,
        "sha256": _tree_sha256(path) if path.is_dir() else _file_sha256(path),
    }


class _CommandAdapter:
    def __init__(self, *, repository_root: Path, state_root: Path) -> None:
        self._repository_root = repository_root.expanduser().resolve()
        self._state_root = state_root.expanduser().resolve()

    def execute(self, request: StageRequest) -> StageReceipt:
        config = {**request.stage.config, **request.config_override}
        command = tuple(config["command"])
        environment = os.environ.copy()
        environment.update(config.get("environment", {}))
        log_path = (
            self._state_root
            / request.run_id
            / "logs"
            / f"{request.stage.name}.attempt-{request.attempt}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            result = subprocess.run(
                command,
                cwd=self._repository_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=config.get("timeout_seconds"),
            )
        artifact_evidence = {}
        artifact_failures = {}
        if result.returncode == 0:
            for contract in config["artifacts"]:
                failure, evidence = _inspect_artifact(contract)
                path = contract["path"]
                if failure is None:
                    artifact_evidence[path] = evidence
                else:
                    artifact_failures[path] = failure
        return StageReceipt(
            stage=request.stage.name,
            attempt=request.attempt,
            status="complete",
            outputs={
                "command": list(command),
                "returncode": result.returncode,
                "log_path": str(log_path),
                "artifacts": [item["path"] for item in config["artifacts"]],
                "artifact_evidence": artifact_evidence,
                "artifact_failures": artifact_failures,
            },
        )


class _ArtifactContractGate:
    def evaluate(self, request: GateRequest) -> GateResult:
        outputs = request.receipt.outputs
        returncode = outputs.get("returncode")
        if returncode != 0:
            return GateResult(
                Decision.BLOCKED,
                f"{request.stage.name} command exited with status {returncode}",
                {"log_path": outputs.get("log_path")},
            )

        evidence: dict[str, Any] = {"artifacts": {}}
        for contract in request.stage.config["artifacts"]:
            path = Path(contract["path"])
            failure, measured = _inspect_artifact(contract)
            if failure is not None:
                return GateResult(
                    Decision.BLOCKED,
                    f"{request.stage.name} artifact {failure}",
                    {"path": str(path), "log_path": outputs.get("log_path")},
                )
            saved = outputs.get("artifact_evidence", {}).get(str(path))
            if saved != measured:
                return GateResult(
                    Decision.BLOCKED,
                    f"{request.stage.name} artifact identity drifted: {path}",
                    {"path": str(path), "log_path": outputs.get("log_path")},
                )
            evidence["artifacts"][str(path)] = measured
        return GateResult(
            Decision.PROCEED,
            f"{request.stage.name} command and artifact contracts passed",
            evidence,
        )

def _artifact_config(contract: ArtifactContract, repository_root: Path) -> dict[str, Any]:
    path = contract.path.expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return {
        "path": str(path.resolve()),
        "kind": contract.kind,
        "expected": dict(contract.expected),
        "required_files": list(contract.required_files),
    }


def _plan(workflow: SslTrainingWorkflow) -> TrainingPlan:
    root = workflow.repository_root.expanduser().resolve()
    stages = tuple(
        StageSpec(
            name=stage.name,
            stage_adapter=_STAGE_ADAPTER,
            gate_adapter=_GATE_ADAPTER,
            config={
                "command": list(stage.command),
                "environment": dict(stage.environment),
                "timeout_seconds": stage.timeout_seconds,
                "artifacts": [
                    _artifact_config(contract, root)
                    for contract in stage.artifacts
                ],
            },
            required_skills=stage.required_skills,
        )
        for stage in workflow.stages
    )
    return TrainingPlan(
        name=workflow.name,
        stages=stages,
        required_skills=(
            "ml-rigor-workflow",
            "ml-audit-data-labels",
            "ml-train-representation",
            "ml-validate-temporal",
        ),
        max_revisions_per_stage=workflow.max_revisions_per_stage,
    )


def run_ssl_training(
    workflow: SslTrainingWorkflow,
    *,
    run_id: str,
    skills: SkillBootstrapper | None = None,
    reasoning: ReasoningAdapter | None = None,
):
    """Run or resume an FFM SSL workflow through the shared state machine."""
    repository_root = workflow.repository_root.expanduser().resolve()
    state_root = workflow.state_root.expanduser().resolve()
    command_adapter = _CommandAdapter(
        repository_root=repository_root,
        state_root=state_root,
    )
    loop = TrainingLoop(
        adapters=DictAdapterRegistry(
            stages={_STAGE_ADAPTER: command_adapter},
            gates={_GATE_ADAPTER: _ArtifactContractGate()},
        ),
        store=JsonRunStore(state_root),
        skills=skills or BundledSkillBootstrapper(DEFAULT_BUNDLE),
        reasoning=reasoning,
    )
    return loop.run(_plan(workflow), run_id=run_id)
