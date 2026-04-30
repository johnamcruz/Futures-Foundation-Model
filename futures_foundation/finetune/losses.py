import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Cross-entropy with focal down-weighting of easy examples and label smoothing."""

    def __init__(self, gamma: float = 1.0, weight=None, label_smoothing: float = 0.10):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs    = F.log_softmax(logits, dim=-1)
        ce           = -(smooth_targets * log_probs).sum(dim=-1)
        w = self.weight.to(logits.dtype) if self.weight is not None else None
        pt           = torch.exp(-F.cross_entropy(logits, targets, weight=w, reduction='none'))
        focal_weight = (1 - pt) ** self.gamma
        loss         = focal_weight * ce
        if w is not None:
            loss = loss * w[targets]
        return loss.mean()
