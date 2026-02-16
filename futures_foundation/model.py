"""
FFM Model — Transformer backbone with pretraining and fine-tuning heads.

Architecture:
    Input Projection → [CLS] + Positional/Temporal Encoding
    → Transformer Encoder (N layers)
    → CLS Pooling → Market Context Embedding (backbone output)
    → Task-specific heads (pretraining or fine-tuning)

HuggingFace compatible: supports save_pretrained / from_pretrained / push_to_hub.
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import FFMConfig


# =============================================================================
# Building Blocks
# =============================================================================


class TemporalEncoding(nn.Module):
    """
    Combined positional + temporal encoding.

    Adds three signals:
        1. Learnable positional embeddings (bar position in sequence)
        2. Time-of-day encoding (sine/cosine, captures intraday patterns)
        3. Day-of-week embedding (captures weekly seasonality)
    """

    def __init__(self, config: FFMConfig):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.time_of_day_proj = nn.Linear(2, config.hidden_size)
        self.day_of_week_embeddings = nn.Embedding(7, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, time_of_day=None, day_of_week=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            time_of_day: (batch, seq_len) float in [0, 1] — fraction of 24h day
            day_of_week: (batch, seq_len) long in [0, 6]
        """
        seq_len = hidden_states.size(1)
        device = hidden_states.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embeddings(position_ids)

        if time_of_day is not None:
            tod_rad = time_of_day.unsqueeze(-1) * 2 * math.pi
            tod_enc = torch.cat([torch.sin(tod_rad), torch.cos(tod_rad)], dim=-1)
            hidden_states = hidden_states + self.time_of_day_proj(tod_enc)

        if day_of_week is not None:
            hidden_states = hidden_states + self.day_of_week_embeddings(day_of_week)

        return self.dropout(self.layernorm(hidden_states))


class CategoricalEmbeddings(nn.Module):
    """
    Embeddings for categorical metadata: instrument and session type.

    Added to hidden states so the model learns instrument-specific and
    session-specific patterns while sharing the core representation.
    """

    def __init__(self, config: FFMConfig):
        super().__init__()
        self.instrument_embeddings = nn.Embedding(
            config.num_instruments, config.hidden_size
        )
        self.session_embeddings = nn.Embedding(
            config.num_sessions, config.hidden_size
        )

    def forward(self, hidden_states, instrument_ids=None, session_ids=None):
        if instrument_ids is not None:
            inst_emb = self.instrument_embeddings(instrument_ids)
            hidden_states = hidden_states + inst_emb.unsqueeze(1)
        if session_ids is not None:
            hidden_states = hidden_states + self.session_embeddings(session_ids)
        return hidden_states


# =============================================================================
# Backbone (the reusable part)
# =============================================================================


class FFMBackbone(PreTrainedModel):
    """
    Futures Foundation Model backbone.

    Takes a sequence of derived feature bars and produces a market context
    embedding via CLS token pooling.

    Input:  (batch, seq_len, num_features) + optional metadata
    Output: (batch, hidden_size) — the CLS token embedding
    """

    config_class = FFMConfig

    def __init__(self, config: FFMConfig):
        super().__init__(config)
        self.config = config

        # Input projection: features → hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Encodings
        self.temporal_encoding = TemporalEncoding(config)
        self.categorical_embeddings = CategoricalEmbeddings(config)

        # Transformer encoder (pre-norm for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
            enable_nested_tensor=False,
        )

        # Output layer norm
        self.output_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=config.initializer_range)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        time_of_day: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        session_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, num_features) — derived OHLCV features
            time_of_day: (batch, seq_len) — fraction of day [0, 1]
            day_of_week: (batch, seq_len) — day index [0, 6]
            instrument_ids: (batch,) — instrument index
            session_ids: (batch, seq_len) — session type index
            attention_mask: (batch, seq_len) — 1 for valid, 0 for padding
            output_sequence: if True, return full sequence; else return CLS only

        Returns:
            If output_sequence=False: (batch, hidden_size) — CLS embedding
            If output_sequence=True: (batch, seq_len+1, hidden_size)
        """
        batch_size, seq_len, _ = features.shape

        # Project input features
        hidden_states = self.input_projection(features)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        # Adjust metadata tensors for CLS prefix
        if time_of_day is not None:
            cls_pad = torch.zeros(batch_size, 1, device=time_of_day.device)
            time_of_day = torch.cat([cls_pad, time_of_day], dim=1)
        if day_of_week is not None:
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.long, device=day_of_week.device)
            day_of_week = torch.cat([cls_pad, day_of_week], dim=1)
        if session_ids is not None:
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.long, device=session_ids.device)
            session_ids = torch.cat([cls_pad, session_ids], dim=1)

        # Add encodings
        hidden_states = self.temporal_encoding(hidden_states, time_of_day, day_of_week)
        hidden_states = self.categorical_embeddings(hidden_states, instrument_ids, session_ids)

        # Attention mask
        src_key_padding_mask = None
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)
            src_key_padding_mask = full_mask == 0

        # Transformer encoder
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.output_layernorm(hidden_states)

        if output_sequence:
            return hidden_states
        return hidden_states[:, 0, :]  # CLS token

    def get_layer_groups(self):
        """
        Returns layer groups for differential learning rates / freezing.
        Bottom → top: input/embeddings, transformer layers, output norm.
        """
        groups = []

        # Group 0: Input + embeddings
        g0 = (
            list(self.input_projection.parameters())
            + list(self.temporal_encoding.parameters())
            + list(self.categorical_embeddings.parameters())
            + [self.cls_token]
        )
        groups.append(("input_embeddings", g0))

        # Groups 1..N: Transformer layers
        for i, layer in enumerate(self.encoder.layers):
            groups.append((f"transformer_layer_{i}", list(layer.parameters())))

        # Final: output norm
        groups.append(("output_norm", list(self.output_layernorm.parameters())))

        return groups


# =============================================================================
# Pretraining Model (backbone + 4 task heads)
# =============================================================================


class FFMForPretraining(PreTrainedModel):
    """
    Futures Foundation Model with multi-task pretraining heads.

    Four simultaneous classification tasks:
        1. Regime: Trending Up / Trending Down / Rotational / Volatile
        2. Volatility: Low / Normal / Elevated / Extreme
        3. Structure: Bullish / Bearish / Mixed
        4. Range Position: Quintile (0-20%, ..., 80-100%)

    Loss uses learnable uncertainty weighting (Kendall et al., 2018).
    """

    config_class = FFMConfig

    def __init__(self, config: FFMConfig):
        super().__init__(config)
        self.backbone = FFMBackbone(config)

        # Task heads — 2-layer MLPs
        self.regime_head = self._make_head(config, config.num_regime_labels)
        self.volatility_head = self._make_head(config, config.num_volatility_labels)
        self.structure_head = self._make_head(config, config.num_structure_labels)
        self.range_head = self._make_head(config, config.num_range_labels)

        # Learnable task weights (log variance)
        self.task_log_vars = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _make_head(config: FFMConfig, num_labels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_labels),
        )

    def forward(
        self,
        features: torch.Tensor,
        time_of_day: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        session_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regime_labels: Optional[torch.Tensor] = None,
        volatility_labels: Optional[torch.Tensor] = None,
        structure_labels: Optional[torch.Tensor] = None,
        range_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Returns dict with:
            - 'loss': combined weighted loss (if any labels provided)
            - 'regime_logits', 'volatility_logits', etc.
            - 'embedding': backbone CLS embedding
        """
        embedding = self.backbone(
            features=features,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            instrument_ids=instrument_ids,
            session_ids=session_ids,
            attention_mask=attention_mask,
            output_sequence=False,
        )

        # Task predictions
        regime_logits = self.regime_head(embedding)
        volatility_logits = self.volatility_head(embedding)
        structure_logits = self.structure_head(embedding)
        range_logits = self.range_head(embedding)

        output = {
            "embedding": embedding,
            "regime_logits": regime_logits,
            "volatility_logits": volatility_logits,
            "structure_logits": structure_logits,
            "range_logits": range_logits,
        }

        # Compute loss with uncertainty weighting
        labels_and_logits = [
            (regime_labels, regime_logits),
            (volatility_labels, volatility_logits),
            (structure_labels, structure_logits),
            (range_labels, range_logits),
        ]

        loss_fn = nn.CrossEntropyLoss()
        total_loss = torch.tensor(0.0, device=features.device)
        num_tasks = 0

        for i, (labels, logits) in enumerate(labels_and_logits):
            if labels is not None:
                task_loss = loss_fn(logits, labels)
                precision = torch.exp(-self.task_log_vars[i])
                total_loss = total_loss + precision * task_loss + self.task_log_vars[i]
                num_tasks += 1

        if num_tasks > 0:
            output["loss"] = total_loss / num_tasks

        return output


# =============================================================================
# Fine-Tuning Classification Model
# =============================================================================


class FFMForClassification(PreTrainedModel):
    """
    FFM with a classification head for downstream strategy fine-tuning.

    Examples:
        - ORB: BUY / SELL / HOLD (num_labels=3)
        - ICT CISD: Bullish / Bearish / None (num_labels=3)
        - Regime: Trend Up / Down / Rotational / Volatile (num_labels=4)
    """

    config_class = FFMConfig

    def __init__(self, config: FFMConfig, num_labels: int = 3):
        super().__init__(config)
        self.num_labels = num_labels
        self.backbone = FFMBackbone(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_labels),
        )

    def load_backbone(self, path: str):
        """Load pretrained backbone weights from file."""
        state_dict = torch.load(path, map_location="cpu")
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k.replace("backbone.", "")] = v
            elif not any(
                k.startswith(prefix)
                for prefix in [
                    "regime_head", "volatility_head",
                    "structure_head", "range_head", "task_log_vars",
                ]
            ):
                backbone_state[k] = v
        self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone from {path}")

    def freeze_backbone(self, freeze_ratio: float = 0.66):
        """Freeze the bottom portion of the backbone for fine-tuning."""
        groups = self.backbone.get_layer_groups()
        num_to_freeze = int(len(groups) * freeze_ratio)

        frozen, trainable = 0, 0
        for i, (name, params) in enumerate(groups):
            freeze = i < num_to_freeze
            for p in params:
                p.requires_grad = not freeze
                if freeze:
                    frozen += p.numel()
                else:
                    trainable += p.numel()

        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        trainable += classifier_params

        print(
            f"Frozen {num_to_freeze}/{len(groups)} layer groups "
            f"({frozen:,} frozen, {trainable:,} trainable)"
        )

    def trainable_parameters(self):
        """Returns only trainable parameters (for optimizer)."""
        return (p for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features: torch.Tensor,
        time_of_day: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        session_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedding = self.backbone(
            features=features,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            instrument_ids=instrument_ids,
            session_ids=session_ids,
            attention_mask=attention_mask,
            output_sequence=False,
        )

        logits = self.classifier(embedding)
        output = {"logits": logits, "embedding": embedding}

        if labels is not None:
            output["loss"] = nn.CrossEntropyLoss()(logits, labels)

        return output