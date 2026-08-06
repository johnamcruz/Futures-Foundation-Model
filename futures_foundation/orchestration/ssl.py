"""Run existing FFM SSL commands through the shared training state machine.

This module owns orchestration only. Data loading, objectives, optimization,
checkpoint creation, Probe Atlas, and promotion evidence remain implemented by
their existing FFM commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ml_training_loop import (
    Decision,
    GateResult,
    ReasoningOutcome,
    Revision,
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
from ml_training_loop.integrations import CodexCliReasoningAdapter
from ml_training_loop.skills import BundledSkillBootstrapper
from ml_training_loop.stores import JsonRunStore


_STAGE_ADAPTER = "ffm-command"
_GATE_ADAPTER = "ffm-artifact-contract"
_ARTIFACT_KINDS = frozenset({"file", "directory", "json"})
_OPERATORS = frozenset({">=", ">", "<=", "<", "=="})


@dataclass(frozen=True)
class MetricRequirement:
    field: str
    operator: str
    value: float

    def __post_init__(self) -> None:
        if not self.field or self.operator not in _OPERATORS:
            raise ValueError("metric requirement field or operator is invalid")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, numbers.Real)
            or not math.isfinite(self.value)
        ):
            raise ValueError("metric requirement value must be finite numeric")


@dataclass(frozen=True)
class ReportGate:
    report_path: Path
    requirements: tuple[MetricRequirement, ...]
    failure_decision: Decision = Decision.REVISE

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("report gate requires at least one metric")
        if self.failure_decision not in {Decision.REVISE, Decision.STOP}:
            raise ValueError("report gate failure decision must be REVISE or STOP")


@dataclass(frozen=True)
class StageRevision:
    candidate_id: str
    rationale: str
    command: tuple[str, ...]
    changed_settings: Mapping[str, Any]
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.rationale or not self.command:
            raise ValueError("stage revision requires id, rationale, and command")
        if not self.changed_settings:
            raise ValueError("stage revision requires changed_settings")


@dataclass(frozen=True)
class SslScientificContext:
    objective: str
    causal_setup: tuple[str, ...]
    frozen_constraints: tuple[str, ...]
    experiment_ledger: tuple[Mapping[str, Any], ...]
    required_skills: tuple[str, ...] = (
        "ml-diagnose-experiment",
        "ml-design-experiment",
        "ml-train-representation",
    )

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("SSL scientific objective must be non-empty")
        for name in ("causal_setup", "frozen_constraints", "required_skills"):
            values = getattr(self, name)
            if not values or any(not item.strip() for item in values):
                raise ValueError(f"SSL scientific context {name} is invalid")
        if any(
            not isinstance(item, Mapping) or not item for item in self.experiment_ledger
        ):
            raise ValueError("SSL experiment ledger entries must be objects")


@dataclass(frozen=True)
class SslReasoningConfig:
    provider: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 1800
    scientific_context: SslScientificContext | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"codex", "predeclared"}:
            raise ValueError("unsupported SSL reasoning provider")
        if self.provider == "codex" and self.scientific_context is None:
            raise ValueError("Codex SSL reasoning requires scientific_context")
        if self.timeout_seconds <= 0:
            raise ValueError("SSL reasoning timeout_seconds must be positive")


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
    gate: ReportGate | None = None
    revisions: tuple[StageRevision, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.command:
            raise ValueError("command stage requires a name and command")
        if not self.artifacts:
            raise ValueError("command stage requires at least one artifact contract")
        if any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("command arguments must be non-empty strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        revision_ids = tuple(item.candidate_id for item in self.revisions)
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("stage revision candidate ids must be unique")


@dataclass(frozen=True)
class SslTrainingWorkflow:
    """Frozen FFM declaration for preflight, SSL, and representation evidence."""

    name: str
    repository_root: Path
    state_root: Path
    stages: tuple[CommandStage, ...]
    max_revisions_per_stage: int = 0
    reasoning: SslReasoningConfig | None = None

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
        if self.reasoning is not None and self.max_revisions_per_stage <= 0:
            raise ValueError("SSL reasoning requires a positive revision budget")


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
                gate=(
                    None
                    if item.get("gate") is None
                    else ReportGate(
                        report_path=Path(item["gate"]["report_path"]),
                        requirements=tuple(
                            MetricRequirement(
                                field=requirement["field"],
                                operator=requirement["operator"],
                                value=requirement["value"],
                            )
                            for requirement in item["gate"]["requirements"]
                        ),
                        failure_decision=Decision(
                            item["gate"].get("failure_decision", "REVISE")
                        ),
                    )
                ),
                revisions=tuple(
                    StageRevision(
                        candidate_id=revision["candidate_id"],
                        rationale=revision["rationale"],
                        command=tuple(revision["command"]),
                        changed_settings=revision["changed_settings"],
                        environment=revision.get("environment", {}),
                        timeout_seconds=revision.get("timeout_seconds"),
                    )
                    for revision in item.get("revisions", ())
                ),
            )
            for item in payload["stages"]
        )
        raw_reasoning = payload.get("reasoning")
        reasoning = None
        if raw_reasoning is not None:
            raw_context = raw_reasoning.get("scientific_context")
            context = None
            if raw_context is not None:
                context = SslScientificContext(
                    objective=raw_context["objective"],
                    causal_setup=tuple(raw_context["causal_setup"]),
                    frozen_constraints=tuple(raw_context["frozen_constraints"]),
                    experiment_ledger=tuple(raw_context["experiment_ledger"]),
                    required_skills=tuple(
                        raw_context.get(
                            "required_skills",
                            (
                                "ml-diagnose-experiment",
                                "ml-design-experiment",
                                "ml-train-representation",
                            ),
                        )
                    ),
                )
            reasoning = SslReasoningConfig(
                provider=raw_reasoning.get("provider", "codex"),
                model=raw_reasoning.get("model", "gpt-5.6-sol"),
                reasoning_effort=raw_reasoning.get("reasoning_effort", "medium"),
                timeout_seconds=raw_reasoning.get("timeout_seconds", 1800),
                scientific_context=context,
            )
        return SslTrainingWorkflow(
            name=payload["name"],
            repository_root=_resolve_from(base, payload.get("repository_root", ".")),
            state_root=_resolve_from(
                base, payload.get("state_root", ".ml-training-loop")
            ),
            stages=stages,
            max_revisions_per_stage=int(payload.get("max_revisions_per_stage", 0)),
            reasoning=reasoning,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid SSL workflow contract: {error}") from error


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
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
        gate = request.stage.config.get("report_gate")
        if gate is not None:
            report = json.loads(Path(gate["report_path"]).read_text())
            failures = []
            measured = {}
            for requirement in gate["requirements"]:
                actual = _nested_value(report, requirement["field"])
                measured[requirement["field"]] = actual
                if not _compare(
                    actual,
                    requirement["operator"],
                    requirement["value"],
                ):
                    failures.append({**requirement, "actual": actual})
            if failures:
                return GateResult(
                    Decision(gate["failure_decision"]),
                    f"{request.stage.name} representation gate failed",
                    {"metrics": measured, "failures": failures},
                )
        return GateResult(
            Decision.PROCEED,
            f"{request.stage.name} command and artifact contracts passed",
            evidence,
        )


def _nested_value(payload: Mapping[str, Any], field: str) -> float:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"report metric is missing: {field}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"report metric must be numeric: {field}")
    return float(value)


def _compare(actual: float, operator: str, expected: float) -> bool:
    return {
        ">=": actual >= expected,
        ">": actual > expected,
        "<=": actual <= expected,
        "<": actual < expected,
        "==": actual == expected,
    }[operator]


def _artifact_config(
    contract: ArtifactContract, repository_root: Path
) -> dict[str, Any]:
    path = contract.path.expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return {
        "path": str(path.resolve()),
        "kind": contract.kind,
        "expected": dict(contract.expected),
        "required_files": list(contract.required_files),
    }


def _revision_override(revision: StageRevision) -> dict[str, Any]:
    return {
        "command": list(revision.command),
        "environment": dict(revision.environment),
        "timeout_seconds": revision.timeout_seconds,
        "changed_settings": dict(revision.changed_settings),
    }


def _report_gate_config(
    gate: ReportGate | None,
    artifacts: tuple[ArtifactContract, ...],
    repository_root: Path,
) -> dict[str, Any] | None:
    if gate is None:
        return None
    path = gate.report_path.expanduser()
    if not path.is_absolute():
        path = repository_root / path
    path = path.resolve()
    declared_reports = {
        (
            item.path.expanduser()
            if item.path.expanduser().is_absolute()
            else repository_root / item.path.expanduser()
        ).resolve()
        for item in artifacts
        if item.kind == "json"
    }
    if path not in declared_reports:
        raise ValueError("report gate path must be a declared JSON artifact")
    return {
        "report_path": str(path),
        "requirements": [
            {"field": item.field, "operator": item.operator, "value": item.value}
            for item in gate.requirements
        ],
        "failure_decision": gate.failure_decision.value,
    }


def _plan(workflow: SslTrainingWorkflow) -> TrainingPlan:
    root = workflow.repository_root.expanduser().resolve()
    reasoning_policy = None
    if workflow.reasoning is not None:
        context = workflow.reasoning.scientific_context
        reasoning_policy = {
            "provider": workflow.reasoning.provider,
            "model": workflow.reasoning.model,
            "reasoning_effort": workflow.reasoning.reasoning_effort,
            "timeout_seconds": workflow.reasoning.timeout_seconds,
            "scientific_context": (
                None
                if context is None
                else {
                    "objective": context.objective,
                    "causal_setup": list(context.causal_setup),
                    "frozen_constraints": list(context.frozen_constraints),
                    "experiment_ledger": [
                        dict(item) for item in context.experiment_ledger
                    ],
                    "required_skills": list(context.required_skills),
                }
            ),
        }
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
                    _artifact_config(contract, root) for contract in stage.artifacts
                ],
                "report_gate": _report_gate_config(
                    stage.gate,
                    stage.artifacts,
                    root,
                ),
                "reasoning_policy": reasoning_policy,
                "revision_ladder": [
                    {
                        "candidate_id": item.candidate_id,
                        "rationale": item.rationale,
                        **_revision_override(item),
                    }
                    for item in stage.revisions
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
            *(
                (
                    "ml-diagnose-experiment",
                    "ml-design-experiment",
                )
                if workflow.reasoning is not None
                else ()
            ),
        ),
        max_revisions_per_stage=workflow.max_revisions_per_stage,
    )


def build_ssl_reasoning_adapter(
    workflow: SslTrainingWorkflow,
    *,
    executor=None,
) -> ReasoningAdapter | None:
    """Build the bounded, skill-directed Codex adapter for an SSL workflow."""
    config = workflow.reasoning
    if config is None or config.provider == "predeclared":
        return None
    context = config.scientific_context
    assert context is not None

    def prompt_builder(request) -> str:
        return (
            "Use $ml-diagnose-experiment first to classify the failure and "
            "localize the first failed representation boundary. Then use "
            "$ml-design-experiment to select one smallest falsifying revision, "
            "and $ml-train-representation to preserve native checkpoint utility, "
            "retention, controls, and auxiliary-head independence. Select exactly "
            "one unused candidate from stage.config.revision_ladder by returning "
            '{"declared_revision_id": <candidate_id>}. Do not invent commands '
            "or change the frozen objective, data, temporal roles, sealed holdout, "
            "parent checkpoint, controls, evidence gates, or artifact contract. "
            "Return STOP when no candidate tests the diagnosed boundary and "
            "BLOCKED only for integrity, causality, lineage, or executable faults.\n\n"
            "Authenticated SSL scientific context:\n"
            + json.dumps(
                {
                    "objective": context.objective,
                    "causal_setup": list(context.causal_setup),
                    "frozen_constraints": list(context.frozen_constraints),
                    "experiment_ledger": [
                        dict(item) for item in context.experiment_ledger
                    ],
                    "required_skills": list(context.required_skills),
                },
                indent=2,
                sort_keys=True,
            )
        )

    def candidates(request) -> dict[str, dict[str, Any]]:
        used = {
            revision.config_override.get("candidate_id")
            for revision in request.prior_revisions
        }
        return {
            item["candidate_id"]: {
                "command": item["command"],
                "environment": item["environment"],
                "timeout_seconds": item["timeout_seconds"],
                "changed_settings": item["changed_settings"],
                "candidate_id": item["candidate_id"],
            }
            for item in request.stage.config.get("revision_ladder", ())
            if item["candidate_id"] not in used
        }

    selected: dict[str, dict[str, Any]] = {}

    def validate(revision: Revision) -> None:
        if set(revision.config_override) != {"declared_revision_id"}:
            raise ValueError("Codex must select one declared SSL revision")
        available = candidates(_active_request[0])
        candidate_id = revision.config_override["declared_revision_id"]
        if candidate_id not in available:
            raise ValueError("Codex selected an unknown or used SSL revision")
        selected.clear()
        selected.update(available[candidate_id])

    _active_request = [None]
    adapter = CodexCliReasoningAdapter(
        repository_root=workflow.repository_root,
        receipt_root=workflow.state_root / "reasoning",
        prompt_builder=prompt_builder,
        revision_validator=validate,
        executor=executor,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        sandbox="read-only",
        timeout_seconds=config.timeout_seconds,
    )

    class _Adapter:
        def revise(self, request):
            _active_request[0] = request
            outcome = adapter.revise(request)
            if outcome.decision is not Decision.REVISE:
                return outcome
            return ReasoningOutcome(
                decision=Decision.REVISE,
                rationale=outcome.rationale,
                revision=Revision(
                    stage=request.stage.name,
                    rationale=outcome.rationale,
                    config_override=dict(selected),
                ),
                evidence=outcome.evidence,
            )

    return _Adapter()


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
        reasoning=(
            reasoning
            if reasoning is not None
            else build_ssl_reasoning_adapter(workflow)
        ),
    )
    return loop.run(_plan(workflow), run_id=run_id)
