import torch

import pytest

from cropwatch_ml.models import MODELS, build_model


@pytest.mark.parametrize("name", MODELS)
def test_output_shape_matches_input(name: str) -> None:
    model = build_model(name, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 256))
    assert out.shape == (2, 4, 256, 256)


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError):
        build_model("nope", num_classes=4)
