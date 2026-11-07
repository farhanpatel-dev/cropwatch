"""Losses for segmentation with ignored pixels.

Cross-entropy scores each pixel on its own, so a rare class (a few % of pixels) barely moves it.
Dice scores the overlap of the predicted and true region per class, then averages the classes,
so every class counts equally however small it is. We add the two together.
"""

import torch
import torch.nn.functional as F
from torch import nn

from cropwatch_ml.datasets import IGNORE_INDEX


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - mean soft Dice over classes. Pixels equal to IGNORE_INDEX are left out.

    logits: N x C x H x W, target: N x H x W (class ids or IGNORE_INDEX).
    Dice for one class = 2 * overlap / (predicted + true), summed over the whole batch.
    eps keeps a class that is absent from both prediction and target near Dice 1 (no penalty).
    """
    num_classes = logits.shape[1]
    valid = (target != IGNORE_INDEX).unsqueeze(1)  # N x 1 x H x W
    probs = logits.float().softmax(1) * valid
    onehot = F.one_hot(target.masked_fill(~valid.squeeze(1), 0), num_classes)
    onehot = onehot.permute(0, 3, 1, 2).float() * valid
    dims = (0, 2, 3)
    overlap = (probs * onehot).sum(dims)
    total = probs.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * overlap + eps) / (total + eps)).mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy (optionally class-weighted) + dice_weight * Dice."""

    def __init__(self, class_weights: torch.Tensor | None = None, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (target != IGNORE_INDEX).any():
            # Every pixel ignored: cross-entropy would be 0/0 = NaN. Contribute nothing instead.
            return logits.sum() * 0.0
        loss = self.ce(logits.float(), target)
        if self.dice_weight:
            loss = loss + self.dice_weight * dice_loss(logits, target)
        return loss
