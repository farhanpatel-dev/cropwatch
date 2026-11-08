import json
from pathlib import Path

import numpy as np
import pytest

from app.model import ModelInfo, Segmenter, load_model_info, normalise, softmax

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
HAS_MODEL = (MODEL_DIR / "model.onnx").exists()


def make_info(tile: int = 256) -> ModelInfo:
    return ModelInfo(
        name="fake",
        version="0",
        mlflow_run_id="none",
        tile_size=tile,
        input_channels=3,
        classes=[{"id": i, "name": n} for i, n in enumerate("abcd")],
        mean=[0.5, 0.25, 0.0],
        std=[0.5, 0.25, 1.0],
        input_name="image",
        output_name="logits",
    )


class RedChannelModel:
    """Fake model: class = normalised red value (0..3) per pixel, with a clear margin.

    With mean 0.5 / std 0.5, red 0 -> -1 and red 255 -> 1, so the class comes from
    the pixel's own red value. A stitching or normalisation bug changes the output.
    """

    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        x = feed["image"]
        assert x.dtype == np.float32 and x.shape[1] == 3
        cls = np.clip(np.round((x[:, 0] + 1) * 1.5), 0, 3).astype(int)  # N x T x T
        logits = np.zeros((x.shape[0], 4, *x.shape[2:]), np.float32)
        np.put_along_axis(logits, cls[:, None], 5.0, axis=1)
        return [logits]


def expected_class(red: np.ndarray) -> np.ndarray:
    norm = (red.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.clip(np.round((norm + 1) * 1.5), 0, 3).astype(np.uint8)


def test_normalise_formula_and_layout() -> None:
    tiles = np.zeros((2, 4, 4, 3), np.uint8)
    tiles[..., 0], tiles[..., 1], tiles[..., 2] = 255, 0, 51
    x = normalise(tiles, [0.5, 0.25, 0.0], [0.5, 0.25, 1.0])
    assert x.shape == (2, 3, 4, 4) and x.dtype == np.float32
    assert np.allclose(x[:, 0], 1.0)  # (1 - 0.5) / 0.5
    assert np.allclose(x[:, 1], -1.0)  # (0 - 0.25) / 0.25
    assert np.allclose(x[:, 2], 0.2)  # 51 / 255


def test_softmax_sums_to_one_and_handles_large_values() -> None:
    logits = np.array([[[[1000.0]], [[0.0]], [[-1000.0]]]], np.float32)
    p = softmax(logits)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(("h", "w"), [(50, 70), (256, 256), (512, 512), (600, 333)])
def test_predict_stitches_per_pixel(h: int, w: int) -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    model = RedChannelModel()
    classes, conf = Segmenter(make_info(), model, batch_size=4).predict(image)
    assert classes.shape == conf.shape == (h, w)
    assert classes.dtype == np.uint8 and conf.dtype == np.float16
    assert np.array_equal(classes, expected_class(image[..., 0]))
    # winning logit 5 vs three zeros: softmax = e^5 / (e^5 + 3)
    assert np.allclose(conf, np.exp(5) / (np.exp(5) + 3), atol=1e-3)


def test_predict_rejects_wrong_input() -> None:
    seg = Segmenter(make_info(), RedChannelModel())
    with pytest.raises(ValueError):
        seg.predict(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        seg.predict(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ValueError):
        seg.predict(np.zeros((10, 10, 3), np.float32))


def test_load_model_info_reports_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "model_info.json"
    path.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError, match="missing field"):
        load_model_info(path)


@pytest.mark.skipif(not HAS_MODEL, reason="model.onnx not present (not in Git)")
def test_real_model_matches_direct_onnx_on_one_tile() -> None:
    import onnxruntime as ort

    info = load_model_info(MODEL_DIR / "model_info.json")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    classes, conf = Segmenter.from_dir(MODEL_DIR).predict(image)

    # Same tile, normalised by hand with the numbers from the file.
    x = (image.astype(np.float32) / 255.0 - np.array(info.mean, np.float32)) / np.array(
        info.std, np.float32
    )
    x = x.transpose(2, 0, 1)[None]
    session = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    (logits,) = session.run(None, {"image": x.astype(np.float32)})
    assert np.array_equal(classes, logits[0].argmax(axis=0))
    assert set(np.unique(classes)) <= set(range(info.num_classes))
    assert float(conf.min()) >= 0.25 - 1e-3 and float(conf.max()) <= 1.0  # 4 classes
