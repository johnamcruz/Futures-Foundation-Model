import torch
import torch.nn as nn

from ..config import FFMConfig
from ..model import FFMBackbone


class HybridStrategyModel(nn.Module):
    """
    FFM backbone fused with a strategy-specific feature projection.

    Architecture:
        FFM Backbone (frozen lower layers)
             │  → CLS embedding (hidden_size)
             │
        ┌────┴──────────────────────────────────────┐
        │                                            │
        │   Strategy features (num_strategy_features)
        │        → Linear(64) → GELU → Linear(64)
        │                            │
        └───── cat ──────────────────┘
                 │ (hidden_size + 64)
             fusion: Linear → GELU → LayerNorm → Dropout
                 │ (hidden_size)
        ┌────────┼────────┬──────────────┐
        │        │        │              │
    signal    risk    confidence
     head     head      head

    The model is strategy-agnostic: only num_strategy_features changes between
    different strategy fine-tunes.
    """

    def __init__(
        self,
        ffm_config: FFMConfig,
        num_strategy_features: int,
        num_labels: int = 2,
        risk_weight: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.risk_weight = risk_weight

        self.backbone = FFMBackbone(ffm_config)

        self.strategy_projection = nn.Sequential(
            nn.Linear(num_strategy_features, 64),
            nn.GELU(),
            nn.Dropout(ffm_config.hidden_dropout_prob),
            nn.Linear(64, 64),
        )

        combined_dim = ffm_config.hidden_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, ffm_config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(ffm_config.hidden_size),
            nn.Dropout(ffm_config.hidden_dropout_prob),
        )

        self.signal_head = nn.Sequential(
            nn.Linear(ffm_config.hidden_size, ffm_config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(ffm_config.hidden_dropout_prob),
            nn.Linear(ffm_config.hidden_size // 2, num_labels),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(ffm_config.hidden_size, ffm_config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(ffm_config.hidden_dropout_prob),
            nn.Linear(ffm_config.hidden_size // 2, 1),
            nn.Softplus(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(ffm_config.hidden_size, ffm_config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(ffm_config.hidden_dropout_prob),
            nn.Linear(ffm_config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def load_backbone(self, path: str) -> None:
        """Load pretrained backbone weights (raw backbone state_dict, no prefix)."""
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f'  ⚠ Backbone unexpected keys: {len(unexpected)}')
        print(f'  ✅ Backbone loaded — {len(state_dict)} tensors, {len(missing)} missing')

    def freeze_backbone(self, freeze_ratio: float = 0.66) -> None:
        """Freeze the bottom fraction of backbone layer groups."""
        groups = self.backbone.get_layer_groups()
        num_freeze = int(len(groups) * freeze_ratio)
        frozen = trainable = 0
        for i, (_name, params) in enumerate(groups):
            for p in params:
                p.requires_grad = i >= num_freeze
                if i < num_freeze:
                    frozen += p.numel()
                else:
                    trainable += p.numel()
        head_params = sum(
            p.numel()
            for m in [self.strategy_projection, self.fusion,
                      self.signal_head, self.risk_head, self.confidence_head]
            for p in m.parameters()
        )
        trainable += head_params
        print(f'  Frozen {num_freeze}/{len(groups)} layers | '
              f'{frozen:,} frozen, {trainable:,} trainable')

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features,
        strategy_features,
        candle_types=None,
        time_of_day=None,
        day_of_week=None,
        instrument_ids=None,
        session_ids=None,
        attention_mask=None,
    ):
        embedding = self.backbone(
            features=features,
            candle_types=candle_types,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            instrument_ids=instrument_ids,
            session_ids=session_ids,
            attention_mask=attention_mask,
            output_sequence=False,
        )
        strat_embed = self.strategy_projection(strategy_features)
        fused = self.fusion(torch.cat([embedding, strat_embed], dim=-1))
        return {
            'signal_logits':    self.signal_head(fused),
            'risk_predictions': self.risk_head(fused),
            'confidence':       self.confidence_head(fused).squeeze(-1),
        }
