import torch

from cropwatch_ml.datasets import IGNORE_INDEX
from cropwatch_ml.losses import CEDiceLoss, dice_loss


def _logits_for(target: torch.Tensor, num_classes: int = 4, scale: float = 20.0) -> torch.Tensor:
    """Very confident logits that predict exactly `target` (ignored pixels -> class 0)."""
    t = target.masked_fill(target == IGNORE_INDEX, 0)
    return torch.nn.functional.one_hot(t, num_classes).permute(0, 3, 1, 2).float() * scale


def test_dice_perfect_vs_wrong() -> None:
    # Big enough that the eps smoothing term is negligible, as on real 256x256 tiles.
    target = torch.tensor([[[0, 0, 1, 1], [0, 2, 2, 3]]]).repeat(1, 32, 32)
    assert dice_loss(_logits_for(target), target) < 0.01
    wrong = (target + 1) % 4
    assert dice_loss(_logits_for(wrong), target) > 0.9


def test_all_ignored_batch_gives_zero_not_nan() -> None:
    target = torch.full((2, 8, 8), IGNORE_INDEX)
    loss = CEDiceLoss(class_weights=torch.ones(4))(torch.randn(2, 4, 8, 8), target)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_ignored_pixels_do_not_change_loss() -> None:
    target = torch.tensor([[[0, 1, 1, 3]]])
    logits = torch.randn(1, 4, 1, 4)
    with_ignore = torch.cat([target, torch.full((1, 1, 2), IGNORE_INDEX)], dim=2)
    junk = torch.randn(1, 4, 1, 2) * 50  # whatever the model says on ignored pixels
    logits_ext = torch.cat([logits, junk], dim=3)
    loss_fn = CEDiceLoss(class_weights=torch.tensor([0.5, 1.0, 2.0, 3.0]))
    assert torch.allclose(loss_fn(logits, target), loss_fn(logits_ext, with_ignore), atol=1e-5)
