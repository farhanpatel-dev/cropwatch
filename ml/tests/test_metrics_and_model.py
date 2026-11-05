import torch

from cropwatch_ml.datasets import IGNORE_INDEX
from cropwatch_ml.metrics import confusion_matrix, iou_per_class, mean_iou
from cropwatch_ml.models import UNet


def test_iou_ignores_pixels_and_handles_absent_classes() -> None:
    target = torch.tensor([[0, 0, 1, IGNORE_INDEX]])
    pred = torch.tensor([[0, 1, 1, 2]])  # the class-2 guess sits on an ignored pixel
    ious = iou_per_class(confusion_matrix(pred, target, num_classes=4))
    assert ious[0] == 0.5  # 1 overlap / (2 true + 1 predicted - 1)
    assert ious[1] == 0.5
    assert ious[2] is None and ious[3] is None
    assert mean_iou(ious, include_background=False) == 0.5


def test_unet_output_shape() -> None:
    model = UNet(in_channels=3, num_classes=4, base=8, depth=4).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 4, 64, 64)
