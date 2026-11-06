import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from cropwatch_ml.datasets import AgriVisionDataset

KEY = "FIELD1_0-0-64-64"


def _make_split(root: Path) -> Path:
    """A 64x64 fake image: red square in one corner, labelled weed_cluster there."""
    weed = np.zeros((64, 64), dtype=bool)
    weed[4:28, 36:60] = True  # top-right block, not symmetric under flips/rotations
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[weed] = (255, 0, 0)
    for sub in ["images/rgb", "masks", "boundaries", "labels/weed_cluster"]:
        (root / sub).mkdir(parents=True)
    Image.fromarray(rgb).save(root / "images/rgb" / f"{KEY}.jpg", quality=100)
    full = np.full((64, 64), 255, dtype=np.uint8)
    Image.fromarray(full).save(root / "masks" / f"{KEY}.png")
    Image.fromarray(full).save(root / "boundaries" / f"{KEY}.png")
    Image.fromarray(weed.astype(np.uint8) * 255).save(root / "labels/weed_cluster" / f"{KEY}.png")
    classes = {"classes": [{"id": 0, "name": "background"}, {"id": 3, "name": "weed_cluster"}]}
    (root / "classes.yaml").write_text(yaml.safe_dump(classes))
    return root


def test_augmented_image_and_target_stay_aligned(tmp_path: Path) -> None:
    root = _make_split(tmp_path)
    ds = AgriVisionDataset(
        root,
        [KEY],
        root / "classes.yaml",
        "train",
        tile_size=64,
        image_size=64,
        cache=True,
        augment=True,
    )
    random.seed(0)
    seen = set()
    for _ in range(40):
        image, target = ds[0]
        red = image[0].numpy() > 0.5
        weed = target.numpy() == 3
        assert (red != weed).mean() < 0.01  # JPEG may blur a few edge pixels
        seen.add(tuple(np.argwhere(weed).mean(0).round().astype(int)))
    assert len(seen) > 1  # augmentation actually moved the block
