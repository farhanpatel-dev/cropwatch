"""Plot RGB images next to their target masks, to check the data looks right.

Run from the repo root:
    ml\\.venv\\Scripts\\python.exe ml\\scripts\\plot_samples.py

Picks images that contain each of our classes (plus one background-only image) from the
train part of the split, and saves one PNG. Grey = ignored (outside mask or field boundary).
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from cropwatch_ml.datasets import IGNORE_INDEX, load_classes, load_split, read_rgb, read_target

ML_DIR = Path(__file__).resolve().parents[1]
IGNORE_COLOUR = "#808080"
BACKGROUND_COLOUR = "#000000"


def class_colours(classes_path: Path) -> dict[int, str]:
    data = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    return {
        int(c["id"]): (BACKGROUND_COLOUR if c["colour"] == "transparent" else c["colour"])
        for c in data["classes"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=ML_DIR / "data/agvision2021_subset/val")
    parser.add_argument("--split-file", type=Path, default=ML_DIR / "configs/split_day1.json")
    parser.add_argument("--classes", type=Path, default=ML_DIR / "configs/classes.yaml")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ML_DIR / "data/plots/samples.png")
    args = parser.parse_args()

    class_ids = load_classes(args.classes)
    id_to_name = {v: k for k, v in class_ids.items()}
    train_keys, _ = load_split(args.split_file)
    rng = random.Random(args.seed)
    rng.shuffle(train_keys)

    # Find a few images per class (and one with only background).
    wanted = {cid: [] for cid in id_to_name if cid != 0}
    background_only: list[str] = []
    targets: dict[str, np.ndarray] = {}
    for key in train_keys:
        if all(len(v) >= args.per_class for v in wanted.values()) and background_only:
            break
        target = read_target(args.split_dir, key, class_ids)
        present = set(np.unique(target).tolist()) - {0, IGNORE_INDEX}
        for cid in present:
            if len(wanted[cid]) < args.per_class:
                wanted[cid].append(key)
                targets[key] = target
                break
        else:
            if not present and not background_only:
                background_only.append(key)
                targets[key] = target
    keys = [k for cid in sorted(wanted) for k in wanted[cid]] + background_only

    colours = class_colours(args.classes)
    lut = [colours.get(i, BACKGROUND_COLOUR) for i in range(max(colours) + 1)]
    cmap = ListedColormap(lut + [IGNORE_COLOUR])
    ignore_slot = len(lut)

    fig, axes = plt.subplots(len(keys), 3, figsize=(10, 3.2 * len(keys)))
    for row, key in zip(np.atleast_2d(axes), keys, strict=True):
        rgb = read_rgb(args.split_dir, key)
        target = targets[key].astype(np.int64)
        shown = np.where(target == IGNORE_INDEX, ignore_slot, target)
        overlay = rgb.copy()
        for cid in range(1, len(lut)):
            overlay[target == cid] = (
                0.5 * overlay[target == cid]
                + 0.5 * np.array(matplotlib.colors.to_rgb(lut[cid])) * 255
            ).astype(np.uint8)
        names = sorted(id_to_name[c] for c in set(np.unique(target).tolist()) - {0, IGNORE_INDEX})
        row[0].imshow(rgb)
        row[0].set_title(key, fontsize=8)
        row[1].imshow(shown, cmap=cmap, vmin=0, vmax=ignore_slot, interpolation="nearest")
        row[1].set_title(", ".join(names) or "background only", fontsize=8)
        row[2].imshow(overlay)
        row[2].set_title("overlay", fontsize=8)
        for ax in row:
            ax.axis("off")

    handles = [Patch(color=lut[i], label=f"{i} {id_to_name[i]}") for i in range(len(lut))]
    handles.append(Patch(color=IGNORE_COLOUR, label=f"{IGNORE_INDEX} ignore"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"saved {len(keys)} samples to {args.out}")


if __name__ == "__main__":
    main()
