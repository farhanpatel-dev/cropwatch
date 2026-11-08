import torch

import random

from cropwatch_ml.datasets import jitter_colour


def test_zero_strength_changes_nothing() -> None:
    image = torch.rand(3, 16, 16)
    assert torch.allclose(jitter_colour(image, 0.0), image, atol=1e-6)


def test_output_stays_in_range_and_shape() -> None:
    random.seed(0)
    image = torch.rand(3, 16, 16)
    for _ in range(20):
        out = jitter_colour(image, 0.5)
        assert out.shape == image.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_grey_image_stays_grey() -> None:
    # Saturation only scales the distance from grey, so a grey image has no colour to change.
    random.seed(1)
    out = jitter_colour(torch.full((3, 8, 8), 0.4), 0.2)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])
