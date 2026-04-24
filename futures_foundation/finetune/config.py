from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainingConfig:
    """All hyperparameters for walk-forward strategy fine-tuning.

    Passed to run_walk_forward() and stored in checkpoint files so that
    resume detection works correctly (config hash changes → fresh start).
    """

    # ── Sequence ──
    seq_len: int = 96

    # ── Batch ──
    batch_size: int = 256
    sig_per_batch: int = 8        # target signal windows per batch

    # ── Optimisation ──
    epochs: int = 40
    lr: float = 5e-5
    freeze_ratio: float = 0.66    # fraction of backbone layers to freeze

    # ── Warm start ──
    # 'selective' (default): transfer backbone weights only, cold-start strategy heads
    #   so heads re-calibrate to the new fold's regime from scratch.
    # 'full': transfer entire model (original behaviour).
    warm_start_mode: str = 'selective'
    # LR multiplier applied to backbone params when warm-starting (option 2).
    # Keeps backbone knowledge stable while heads adapt at full speed.
    # Set to 1.0 to disable layerwise LR.
    backbone_lr_multiplier: float = 0.1

    # ── Loss ──
    risk_weight: float = 0.1      # risk-head loss coefficient
    miss_penalty: float = 1.0     # class weight for signal class
    false_penalty: float = 1.0    # class weight for noise class
    focal_gamma: float = 1.0
    focal_smoothing: float = 0.10

    # ── Early stopping ──
    patience: int = 15            # epochs without val_loss improvement
    max_ratio: float = 2.5        # val_loss / train_loss ceiling
    ratio_patience: int = 8       # consecutive epochs above max_ratio

    # ── Output ──
    num_labels: int = 2           # 2 = noise/signal; 3 = sell/hold/buy

    # ── Evaluation ──
    baseline_wr: Dict[str, float] = field(default_factory=dict)
