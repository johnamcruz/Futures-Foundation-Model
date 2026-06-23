from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainingConfig:
    """Generic hyperparameters for strategy labeling / evaluation runs.

    NOTE: the torch walk-forward trainer was retired with the from-scratch
    FFM backbone (see git tag `ffm-transformer-final`). The learned-backbone
    fields (freeze_ratio, warm_start_mode, backbone_lr_multiplier,
    focal-loss knobs, checkpoint-tier patience, ...) were removed with it.
    What remains parameterizes labeling/eval only; walk-forward *training*
    now lives in futures_foundation.pipeline (Chronos-Bolt embeddings + XGBoost).
    """

    # ── Sequence ──
    seq_len: int = 96

    # ── Batch ──
    batch_size: int = 256

    # ── Optimisation (generic) ──
    epochs: int = 40
    lr: float = 5e-5

    # ── Early stopping (generic) ──
    patience: int = 15            # epochs without val improvement

    # ── Economic checkpoint selection (borrow #3, opt-in default-OFF) ──
    # When True AND borrow #1's realized_r is available (the labeler emits a
    # `direction` column), checkpoint selection can use the CAGR·√Sortino
    # product objective (_econ_combined_objective / _val_econ_objective).
    econ_selection: bool = False
    econ_patience: int = 10       # epochs without econ-objective improvement

    # ── Output ──
    num_labels: int = 2           # 2 = noise/signal; 3 = sell/hold/buy

    # ── Evaluation ──
    baseline_wr: Dict[str, float] = field(default_factory=dict)
