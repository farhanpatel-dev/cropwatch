"""Agriculture-Vision 2021: RGB image + target mask pairs for segmentation.

Folder layout (one split, e.g. ml/data/agvision2021_subset/val):
    images/rgb/<key>.jpg              512x512 RGB
    masks/<key>.png                   0 = outside valid area, 255 = valid
    boundaries/<key>.png              0 = outside field boundary, 255 = inside
    labels/<class_name>/<key>.png     0/255, one binary mask per class (classes can overlap)
key = <field_id>_<x1>-<y1>-<x2>-<y2>

Target mask rules (see CLAUDE.md):
    - start as background (0)
    - paint our classes in PAINT_ORDER, so the rarer class wins where labels overlap
    - pixels outside masks or boundaries become IGNORE_INDEX (excluded from loss and IoU)
    - all other dataset labels stay background
"""

import torch
from torch.utils.data import Dataset

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IGNORE_INDEX = 255
TILE_SIZE = 256
# Later names overwrite earlier ones: weed_cluster (rarest) wins, drydown (most common) loses.
PAINT_ORDER = ("drydown", "nutrient_deficiency", "weed_cluster")


def load_classes(path: str | Path) -> dict[str, int]:
    """Read classes.yaml and return {name: id}, including background."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {c["name"]: int(c["id"]) for c in data["classes"]}


def field_id(key: str) -> str:
    """'XF6FPLB2I_6774-3675-7286-4187' -> 'XF6FPLB2I'."""
    return key.split("_", 1)[0]


def list_keys(split_dir: str | Path) -> list[str]:
    """All image keys in a split folder, sorted."""
    return sorted(p.stem for p in (Path(split_dir) / "images" / "rgb").glob("*.jpg"))


def build_target(
    labels: dict[str, np.ndarray], valid: np.ndarray, class_ids: dict[str, int]
) -> np.ndarray:
    """Combine per-class binary masks into one uint8 class-id mask.

    labels: {class_name: bool array HxW}; names not in class_ids are ignored (stay background).
    valid:  bool array HxW, False where masks==0 or boundaries==0.
    """
    target = np.zeros(valid.shape, dtype=np.uint8)
    for name in PAINT_ORDER:
        if name in labels and name in class_ids:
            target[labels[name]] = class_ids[name]
    target[~valid] = IGNORE_INDEX
    return target


def split_by_field(
    keys: Sequence[str], n_val_fields: int, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split keys so that every field is entirely in train or entirely in val."""
    by_field: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_field[field_id(k)].append(k)
    fields = sorted(by_field)
    if not 0 < n_val_fields < len(fields):
        raise ValueError(f"n_val_fields must be between 1 and {len(fields) - 1}")
    val_fields = set(random.Random(seed).sample(fields, n_val_fields))
    train = sorted(k for f in fields if f not in val_fields for k in by_field[f])
    val = sorted(k for f in sorted(val_fields) for k in by_field[f])
    return train, val


def save_split(path: str | Path, train: list[str], val: list[str], seed: int) -> None:
    data = {
        "seed": seed,
        "train_fields": sorted({field_id(k) for k in train}),
        "val_fields": sorted({field_id(k) for k in val}),
        "train": train,
        "val": val,
    }
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def load_split(path: str | Path) -> tuple[list[str], list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["train"], data["val"]


def read_rgb(split_dir: Path, key: str) -> np.ndarray:
    """HxWx3 uint8."""
    with Image.open(split_dir / "images" / "rgb" / f"{key}.jpg") as im:
        return np.asarray(im.convert("RGB"))


def read_binary(path: Path) -> np.ndarray:
    """HxW bool (pixel > 0)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def read_target(split_dir: Path, key: str, class_ids: dict[str, int]) -> np.ndarray:
    """HxW uint8 class-id mask with IGNORE_INDEX outside the valid area."""
    valid = read_binary(split_dir / "masks" / f"{key}.png") & read_binary(
        split_dir / "boundaries" / f"{key}.png"
    )
    labels = {
        name: read_binary(split_dir / "labels" / name / f"{key}.png")
        for name in PAINT_ORDER
        if name in class_ids
    }
    return build_target(labels, valid, class_ids)


def compute_mean_std(split_dir: str | Path, keys: Sequence[str]) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of RGB values scaled to 0-1, over valid pixels only.

    Use training keys only, so validation data does not leak into preprocessing.
    """
    split_dir = Path(split_dir)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for key in keys:
        valid = read_binary(split_dir / "masks" / f"{key}.png") & read_binary(
            split_dir / "boundaries" / f"{key}.png"
        )
        px = read_rgb(split_dir, key)[valid].astype(np.float64) / 255.0  # N x 3
        total += px.sum(0)
        total_sq += (px**2).sum(0)
        count += len(px)
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return mean.tolist(), std.tolist()


def jitter_colour(image: torch.Tensor, strength: float) -> torch.Tensor:
    """Random brightness, contrast and saturation change of a 3xHxW image in 0-1."""

    def factor() -> float:
        return random.uniform(1 - strength, 1 + strength)

    image = image * factor()  # brightness
    mean = image.mean()
    image = (image - mean) * factor() + mean  # contrast: stretch around the average
    grey = image.mean(0, keepdim=True)
    image = (image - grey) * factor() + grey  # saturation: move away from / toward grey
    return image.clamp(0.0, 1.0)


class AgriVisionDataset(Dataset):
    """(image, target) pairs as tensors: image float32 3xTxT, target int64 TxT.

    mode="train": one random TxT crop per image per epoch.
    mode="val":   every non-overlapping TxT tile of every image (fixed, so runs are comparable).
    cache=True:   read every image and target once and keep them in memory (fast epochs).
    augment=True: random horizontal/vertical flips and 90-degree rotations (train mode only).
    colour_jitter=j: with augment, scale brightness, contrast and saturation each by a random
                  factor in [1-j, 1+j] (0 = off). The target is unchanged: colour does not move
                  pixels, it only makes the model rely less on the exact colours of train fields.
    """

    def __init__(
        self,
        split_dir: str | Path,
        keys: Sequence[str],
        classes_path: str | Path,
        mode: str = "train",
        tile_size: int = TILE_SIZE,
        image_size: int = 512,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        cache: bool = False,
        augment: bool = False,
        colour_jitter: float = 0.0,
    ) -> None:
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        if image_size % tile_size:
            raise ValueError("image_size must be a multiple of tile_size")
        self.split_dir = Path(split_dir)
        self.keys = list(keys)
        self.class_ids = load_classes(classes_path)
        self.mode = mode
        self.tile = tile_size
        self.image_size = image_size
        self.tiles_per_side = image_size // tile_size
        self.mean = torch.tensor(mean or [0.0, 0.0, 0.0]).view(3, 1, 1)
        self.std = torch.tensor(std or [1.0, 1.0, 1.0]).view(3, 1, 1)
        self.augment = augment and mode == "train"
        self.colour_jitter = colour_jitter if self.augment else 0.0
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if cache:
            for key in self.keys:
                self._cache[key] = self._read(key)

    def _read(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key in self._cache:
            return self._cache[key]
        rgb = read_rgb(self.split_dir, key)
        if rgb.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"{key}: expected {self.image_size}px image, got {rgb.shape[:2]}")
        return rgb, read_target(self.split_dir, key, self.class_ids)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.keys)
        return len(self.keys) * self.tiles_per_side**2

    def _crop_origin(self, index: int) -> tuple[str, int, int]:
        if self.mode == "train":
            key = self.keys[index]
            top = random.randint(0, self.image_size - self.tile)
            left = random.randint(0, self.image_size - self.tile)
            return key, top, left
        per_image = self.tiles_per_side**2
        key = self.keys[index // per_image]
        row, col = divmod(index % per_image, self.tiles_per_side)
        return key, row * self.tile, col * self.tile

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key, top, left = self._crop_origin(index)
        rgb, target = self._read(key)
        rgb = rgb[top : top + self.tile, left : left + self.tile]
        target = target[top : top + self.tile, left : left + self.tile]
        if self.augment:  # same flip/rotation for image and target, so they stay aligned
            if random.random() < 0.5:
                rgb, target = rgb[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                rgb, target = rgb[::-1], target[::-1]
            k = random.randint(0, 3)
            rgb, target = np.rot90(rgb, k), np.rot90(target, k)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        if self.colour_jitter:
            image = jitter_colour(image, self.colour_jitter)
        image = (image - self.mean) / self.std
        return image, torch.from_numpy(
            target.astype(np.int64)
        )  # astype copies (no negative strides)
