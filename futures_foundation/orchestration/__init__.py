"""FFM-owned adapters for the shared ML training-loop state machine."""

from .ssl import (
    ArtifactContract,
    CommandStage,
    SslTrainingWorkflow,
    load_ssl_workflow,
    run_ssl_training,
)

__all__ = [
    "ArtifactContract",
    "CommandStage",
    "SslTrainingWorkflow",
    "load_ssl_workflow",
    "run_ssl_training",
]
