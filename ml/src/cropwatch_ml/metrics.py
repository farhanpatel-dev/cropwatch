"""Segmentation metrics. IoU = overlap / union, per class, ignoring IGNORE_INDEX pixels."""

import torch

from cropwatch_ml.datasets import IGNORE_INDEX


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """num_classes x num_classes counts; rows = true class, columns = predicted class."""
    keep = target != IGNORE_INDEX
    idx = target[keep].long() * num_classes + pred[keep].long()
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_per_class(cm: torch.Tensor) -> list[float | None]:
    """IoU for each class; None if the class is absent from both target and prediction."""
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    return [None if u == 0 else float(t / u) for t, u in zip(tp, union, strict=True)]


def mean_iou(ious: list[float | None], include_background: bool) -> float:
    vals = [v for i, v in enumerate(ious) if v is not None and (include_background or i != 0)]
    return sum(vals) / len(vals) if vals else 0.0
