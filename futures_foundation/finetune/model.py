import torch
import torch.nn as nn

from ..config import FFMConfig
from ..model import FFMBackbone

# 4 pretraining head output sizes — must match FFMConfig defaults
_CONTEXT_HEAD_SIZES = {
    'regime':     4,
    'volatility': 4,
    'structure':  2,
    'range':      5,
}
_CONTEXT_DIM = sum(_CONTEXT_HEAD_SIZES.values())  # 15


def _make_head(config: FFMConfig, num_labels: int) -> nn.Sequential:
    """Two-layer MLP head identical to FFMForPretraining._make_head."""
    return nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size // 2),
        nn.GELU(),
        nn.Dropout(config.hidden_dropout_prob),
        nn.Linear(config.hidden_size // 2, num_labels),
    )


class HybridStrategyModel(nn.Module):
    """
    FFM backbone fused with strategy features + optional frozen pretraining context.

    Architecture (use_context=True):
        FFM Backbone (frozen lower layers)
             │  → CLS embedding (hidden_size=256)
             │
        ┌────┴─────────────────────────────────────────────────┐
        │                                                       │
        │  Frozen context heads (loaded from best_pretrained.pt)│
        │    regime(4) + volatility(4) + structure(2) + range(5)│
        │    → softmax → 15-dim context vector                  │
        │                                                       │
        │  Strategy features (num_strategy_features)           │
        │        → Linear(64) → GELU → Linear(64)              │
        │                                    │                  │
        └─────────── cat ────────────────────┘
                     │ (256 + 15 + 64 = 335)
                fusion: Linear → GELU → LayerNorm → Dropout
                     │ (256)
             ┌───────┤
         signal    risk
          head     head

    With use_context=False the 15-dim context is omitted (combined_dim=320),
    matching the original architecture.
    """

    def __init__(
        self,
        ffm_config: FFMConfig,
        num_strategy_features: int,
        num_labels: int = 2,
        risk_weight: float = 0.1,
        use_context: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.risk_weight = risk_weight
        self.use_context = use_context

        self.backbone = FFMBackbone(ffm_config)

        if use_context:
            self.regime_head     = _make_head(ffm_config, _CONTEXT_HEAD_SIZES['regime'])
            self.volatility_head = _make_head(ffm_config, _CONTEXT_HEAD_SIZES['volatility'])
            self.structure_head  = _make_head(ffm_config, _CONTEXT_HEAD_SIZES['structure'])
            self.range_head      = _make_head(ffm_config, _CONTEXT_HEAD_SIZES['range'])
            context_dim = _CONTEXT_DIM
        else:
            context_dim = 0

        self.strategy_projection = nn.Sequential(
            nn.Linear(num_strategy_features, 64),
            nn.GELU(),
            nn.Dropout(ffm_config.hidden_dropout_prob),
            nn.Linear(64, 64),
        )

        combined_dim = ffm_config.hidden_size + context_dim + 64
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

    def load_pretrained(self, path: str) -> None:
        """
        Load backbone + 4 context head weights from best_pretrained.pt.
        Context heads are frozen after loading.
        """
        state_dict = torch.load(path, map_location='cpu', weights_only=True)

        backbone_sd = {
            k[len('backbone.'):]: v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        if unexpected:
            print(f'  ⚠ Backbone unexpected keys: {len(unexpected)}')

        if self.use_context:
            head_attrs = [
                ('regime_head',     self.regime_head),
                ('volatility_head', self.volatility_head),
                ('structure_head',  self.structure_head),
                ('range_head',      self.range_head),
            ]
            for attr_name, head_module in head_attrs:
                prefix = f'{attr_name}.'
                head_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                head_module.load_state_dict(head_sd, strict=True)
                for p in head_module.parameters():
                    p.requires_grad = False

        total = len(backbone_sd)
        print(
            f'  ✅ Pretrained loaded — {total} backbone tensors'
            + (' | context heads frozen' if self.use_context else '')
        )

    def load_backbone(self, path: str) -> None:
        """Load backbone-only weights (raw backbone state_dict, no prefix)."""
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f'  ⚠ Backbone unexpected keys: {len(unexpected)}')
        print(f'  ✅ Backbone loaded — {len(state_dict)} tensors, {len(missing)} missing')

    def freeze_backbone(self, freeze_ratio: float = 0.66) -> None:
        """Freeze the bottom fraction of backbone layer groups. Context heads stay frozen."""
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
                      self.signal_head, self.risk_head]
            for p in m.parameters()
        )
        trainable += head_params
        context_frozen = 0
        if self.use_context:
            context_frozen = sum(
                p.numel()
                for m in [self.regime_head, self.volatility_head,
                          self.structure_head, self.range_head]
                for p in m.parameters()
            )
            frozen += context_frozen
        print(f'  Frozen {num_freeze}/{len(groups)} backbone layers | '
              f'{frozen:,} frozen ({context_frozen:,} context heads), {trainable:,} trainable')

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

        if self.use_context:
            with torch.no_grad():
                ctx = torch.cat([
                    torch.softmax(self.regime_head(embedding),     dim=-1),
                    torch.softmax(self.volatility_head(embedding), dim=-1),
                    torch.softmax(self.structure_head(embedding),  dim=-1),
                    torch.softmax(self.range_head(embedding),      dim=-1),
                ], dim=-1)  # (batch, 15)
            strat_embed = self.strategy_projection(strategy_features)
            fused = self.fusion(torch.cat([embedding, ctx, strat_embed], dim=-1))
        else:
            strat_embed = self.strategy_projection(strategy_features)
            fused = self.fusion(torch.cat([embedding, strat_embed], dim=-1))

        signal_logits = self.signal_head(fused)
        probs = torch.softmax(signal_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return {
            'signal_logits':    signal_logits,
            'risk_predictions': self.risk_head(fused),
            'confidence':       confidence,
        }
