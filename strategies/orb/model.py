"""
Hybrid ORB Model
================

FFM Backbone (256-dim market context) + ORB Features (20→64-dim breakout info)
→ Fusion (320→256) → Signal Head (BUY/SELL/HOLD) + Risk Head (max_rr)

Usage:
    from futures_foundation import FFMConfig
    from strategies.orb import HybridORBModel

    config = FFMConfig()
    model = HybridORBModel(config)
    model.load_backbone("path/to/best_backbone.pt")
    model.freeze_backbone(freeze_ratio=0.33)

    out = model(features=X_seq, orb_features=orb_last_bar)
"""

import torch
import torch.nn as nn

from futures_foundation import FFMConfig, FFMBackbone

from .features import NUM_ORB_FEATURES


class HybridORBModel(nn.Module):
    """
    Hybrid architecture combining pretrained FFM backbone with
    ORB-specific features for breakout signal prediction.

    Architecture:
        FFM Backbone: (batch, seq, 42) → (batch, 256)  [market context]
        ORB Projection: (batch, 20) → (batch, 64)       [breakout info]
        Fusion: (batch, 320) → (batch, 256)
        Signal Head: (batch, 256) → (batch, 3)           [HOLD/BUY/SELL]
        Risk Head: (batch, 256) → (batch, 1)             [max_rr estimate]
    """

    def __init__(self, config: FFMConfig, num_orb_features: int = NUM_ORB_FEATURES,
                 num_labels: int = 3, num_risk_targets: int = 1,
                 risk_weight: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.num_risk_targets = num_risk_targets
        self.risk_weight = risk_weight
        self.hidden_size = config.hidden_size

        # FFM backbone (pretrained)
        self.backbone = FFMBackbone(config)

        # ORB feature projection: 20 → 64
        self.orb_projection = nn.Sequential(
            nn.Linear(num_orb_features, 64),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(64, 64),
        )

        # Fusion: 256 + 64 = 320 → 256
        combined_dim = config.hidden_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # Signal head: BUY / SELL / HOLD
        self.signal_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_labels),
        )

        # Risk head: max_rr prediction
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_risk_targets),
            nn.Softplus(),
        )

    def load_backbone(self, path: str):
        """Load pretrained backbone weights from pretraining checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k.replace("backbone.", "")] = v
            elif not any(k.startswith(p) for p in [
                "regime_head", "volatility_head", "structure_head", "range_head"]):
                backbone_state[k] = v
        self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone from {path}")

    def freeze_backbone(self, freeze_ratio: float = 0.66):
        """Freeze bottom portion of backbone layers."""
        groups = self.backbone.get_layer_groups()
        num_freeze = int(len(groups) * freeze_ratio)
        frozen, trainable = 0, 0
        for i, (name, params) in enumerate(groups):
            freeze = i < num_freeze
            for p in params:
                p.requires_grad = not freeze
                if freeze:
                    frozen += p.numel()
                else:
                    trainable += p.numel()

        head_params = (sum(p.numel() for p in self.orb_projection.parameters())
                       + sum(p.numel() for p in self.fusion.parameters())
                       + sum(p.numel() for p in self.signal_head.parameters())
                       + sum(p.numel() for p in self.risk_head.parameters()))
        trainable += head_params
        print(f"Frozen {num_freeze}/{len(groups)} groups "
              f"({frozen:,} frozen, {trainable:,} trainable)")

    def trainable_parameters(self):
        """Return only parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features, orb_features, time_of_day=None, day_of_week=None,
                instrument_ids=None, session_ids=None, attention_mask=None):
        # Backbone: (batch, seq, 42) → (batch, 256)
        embedding = self.backbone(
            features=features, time_of_day=time_of_day, day_of_week=day_of_week,
            instrument_ids=instrument_ids, session_ids=session_ids,
            attention_mask=attention_mask, output_sequence=False)

        # ORB features: (batch, 20) → (batch, 64)
        orb_embed = self.orb_projection(orb_features)

        # Fuse: (batch, 320) → (batch, 256)
        combined = torch.cat([embedding, orb_embed], dim=-1)
        fused = self.fusion(combined)

        signal_logits = self.signal_head(fused)
        risk_predictions = self.risk_head(fused)

        return {
            "signal_logits": signal_logits,
            "risk_predictions": risk_predictions,
            "embedding": embedding,
            "fused": fused,
        }
