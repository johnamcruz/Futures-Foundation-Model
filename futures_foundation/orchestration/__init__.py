"""FFM-owned adapters for the shared ML training-loop state machine."""

from .ssl import (
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

__all__ = [
    "ArtifactContract",
    "CommandStage",
    "MetricRequirement",
    "ReportGate",
    "SslReasoningConfig",
    "SslScientificContext",
    "StageRevision",
    "SslTrainingWorkflow",
    "build_ssl_reasoning_adapter",
    "load_ssl_workflow",
    "run_ssl_training",
]
