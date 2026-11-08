"""Run the exported ONNX model on a whole image: tiles -> normalise -> model -> stitch.

Reads three files from the model directory (written by ml/scripts/export_onnx.py):
  model.onnx       input "image" N x 3 x T x T float32, output "logits" N x C x T x T
  model_info.json  name, version, tile size, classes, preprocessing mean/std
  classes.yaml     (used later by the API; not read here)

Normalisation must match training exactly (docs/api.md, "Input normalisation"): uint8 RGB
-> / 255 -> (x - mean[c]) / std[c] -> CHW. mean/std always come from model_info.json; a
wrong value raises no error, it silently gives bad predictions.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.tiling import OVERLAP, batched, iter_tiles, paste, plan_tiles


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    mlflow_run_id: str
    tile_size: int
    input_channels: int
    classes: list[dict]
    mean: list[float]
    std: list[float]
    input_name: str
    output_name: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_model_info(path: str | Path) -> ModelInfo:
    """Read model_info.json; fail early and clearly if a required field is missing."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        info = ModelInfo(
            name=raw["name"],
            version=str(raw["version"]),
            mlflow_run_id=raw["mlflow_run_id"],
            tile_size=int(raw["tile_size"]),
            input_channels=int(raw["input_channels"]),
            classes=raw["classes"],
            mean=[float(v) for v in raw["preprocessing"]["mean"]],
            std=[float(v) for v in raw["preprocessing"]["std"]],
            input_name=raw.get("input", {}).get("name", "image"),
            output_name=raw.get("output", {}).get("name", "logits"),
        )
    except KeyError as e:
        raise ValueError(f"model_info.json is missing field {e}") from e
    if info.input_channels != 3 or len(info.mean) != 3 or len(info.std) != 3:
        raise ValueError("model_info.json: expected 3 input channels with 3 mean/std values")
    if any(s <= 0 for s in info.std):
        raise ValueError("model_info.json: std values must be positive")
    return info


def normalise(tiles: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """N x T x T x 3 uint8 RGB -> N x 3 x T x T float32, normalised as in training."""
    x = tiles.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """Probabilities that sum to 1 along `axis` (max subtracted first, so exp cannot overflow)."""
    z = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


class Session(Protocol):
    """The part of onnxruntime.InferenceSession we use (lets tests pass a fake model)."""

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list: ...


class Segmenter:
    """Whole-image segmentation with the exported model.

    predict() returns two H x W maps for the input image:
      classes     uint8, class id per pixel (argmax of the logits)
      confidence  float16, softmax probability of that class (0-1); float16 halves memory
                  for large images and is precise enough for per-polygon means.
    """

    def __init__(
        self, info: ModelInfo, session: Session, batch_size: int = 16, overlap: int = OVERLAP
    ) -> None:
        self.info = info
        self.session = session
        self.batch_size = batch_size
        self.overlap = overlap

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path,
        info_path: str | Path,
        batch_size: int = 16,
        overlap: int = OVERLAP,
    ) -> "Segmenter":
        import onnxruntime as ort  # imported here so tests with a fake session do not need it

        info = load_model_info(info_path)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(info, session, batch_size, overlap)

    @classmethod
    def from_dir(cls, model_dir: str | Path, batch_size: int = 16) -> "Segmenter":
        model_dir = Path(model_dir)
        return cls.from_paths(model_dir / "model.onnx", model_dir / "model_info.json", batch_size)

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected H x W x 3 uint8 RGB, got {image.shape} {image.dtype}")
        h, w = image.shape[:2]
        t = self.info.tile_size
        plan = plan_tiles(h, w, t, self.overlap)
        classes = np.zeros((h, w), np.uint8)
        confidence = np.zeros((h, w), np.float16)
        crops = iter_tiles(image, plan, t)
        for batch in batched(plan, self.batch_size):
            tiles = np.stack([next(crops) for _ in batch])
            x = normalise(tiles, self.info.mean, self.info.std)
            (logits,) = self.session.run([self.info.output_name], {self.info.input_name: x})
            probs = softmax(np.asarray(logits, np.float32))
            tile_classes = probs.argmax(axis=1).astype(np.uint8)
            tile_conf = probs.max(axis=1).astype(np.float16)
            for i, tile in enumerate(batch):
                paste(tile, tile_classes[i], classes)
                paste(tile, tile_conf[i], confidence)
        return classes, confidence
