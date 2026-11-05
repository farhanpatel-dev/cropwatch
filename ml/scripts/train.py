"""Overfit test: train the U-Net on a few images until it memorises them, logging to MLflow.

If the model cannot drive the loss close to zero on ~20 images, something is broken
(data, loss, model or optimiser). Real training on held-out data comes later.

Run from the repo root (Compose stack running):
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    ml\\.venv\\Scripts\\python.exe ml\\scripts\\train.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np

from cropwatch_ml.datasets import (
    IGNORE_INDEX,
    PAINT_ORDER,
    AgriVisionDataset,
    compute_mean_std,
    load_classes,
    load_split,
    read_target,
)
from cropwatch_ml.metrics import confusion_matrix, iou_per_class, mean_iou
from cropwatch_ml.models import UNet

ML_DIR = Path(__file__).resolve().parents[1]


def pick_images(
    split_dir: Path, keys: list[str], class_ids: dict[str, int], n: int, seed: int
) -> tuple[list[str], dict[str, int]]:
    """About n/3 images per anomaly class, so every class is tested.

    Keys are shuffled with a fixed seed. Each image is counted for the rarest of our classes
    it contains, so the rare classes fill their quota first. Returns (keys, images per class).
    """
    rarest_first = [c for c in reversed(PAINT_ORDER) if c in class_ids]
    quota = {c: n // len(rarest_first) for c in rarest_first}
    for c in rarest_first[: n % len(rarest_first)]:
        quota[c] += 1
    counts = dict.fromkeys(rarest_first, 0)
    shuffled = sorted(keys)
    random.Random(seed).shuffle(shuffled)
    chosen = []
    for key in shuffled:
        present = set(np.unique(read_target(split_dir, key, class_ids)).tolist())
        for c in rarest_first:
            if class_ids[c] in present and counts[c] < quota[c]:
                chosen.append(key)
                counts[c] += 1
                break
        if len(chosen) == n:
            break
    return chosen, counts


def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str) -> torch.Tensor:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            cm += confusion_matrix(model(x).argmax(1), y, num_classes)
    model.train()
    return cm.cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=ML_DIR / "data/agvision2021_subset/val")
    p.add_argument("--split-file", type=Path, default=ML_DIR / "configs/split_day1.json")
    p.add_argument("--classes", type=Path, default=ML_DIR / "configs/classes.yaml")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help="cosine: lr falls smoothly from --lr to ~0 over all epochs",
    )
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", default="cropwatch-overfit")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_ids = load_classes(args.classes)
    id_to_name = {v: k for k, v in class_ids.items()}
    num_classes = len(class_ids)

    train_keys, _ = load_split(args.split_file)
    keys, per_class = pick_images(args.split_dir, train_keys, class_ids, args.n_images, args.seed)
    mean, std = compute_mean_std(args.split_dir, train_keys)  # all train fields, not just these
    print(f"device {device} | {len(keys)} images {per_class} | mean {mean} std {std}")

    # Fixed tiles (4 per image), loaded once into memory: same data every epoch, easy to memorise.
    ds = AgriVisionDataset(args.split_dir, keys, args.classes, mode="val", mean=mean, std=std)
    xs, ys = zip(*(ds[i] for i in range(len(ds))), strict=True)
    tiles = TensorDataset(torch.stack(xs), torch.stack(ys))
    train_loader = DataLoader(tiles, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(tiles, batch_size=args.batch_size)

    model = UNet(in_channels=3, num_classes=num_classes, base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # All-background baseline on the same tiles.
    all_y = torch.stack(ys)
    base_ious = iou_per_class(confusion_matrix(torch.zeros_like(all_y), all_y, num_classes))

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **{k: v for k, v in vars(args).items() if not isinstance(v, Path)},
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "torch": torch.__version__,
                "model": "unet",
                "params": n_params,
                "tiles": len(tiles),
                "images_per_class": json.dumps(per_class),
                "tile_size": ds.tile,
                "input_channels": 3,
                "norm_mean": json.dumps([round(m, 4) for m in mean]),
                "norm_std": json.dumps([round(s, 4) for s in std]),
                "miou_includes_background": "logged both: miou_with_bg and miou_no_bg",
            }
        )
        mlflow.log_metrics(
            {
                "baseline_miou_with_bg": mean_iou(base_ious, True),
                "baseline_miou_no_bg": mean_iou(base_ious, False),
            }
        )
        mlflow.log_artifact(str(args.classes))
        print(f"mlflow run {run.info.run_id} | params {n_params:,} | tiles {len(tiles)}")

        start = time.time()
        step = 0
        for epoch in range(1, args.epochs + 1):
            total, count = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
                count += len(x)
                step += 1
            mlflow.log_metrics(
                {"train_loss": total / count, "lr": opt.param_groups[0]["lr"]}, step=epoch
            )
            if scheduler is not None:
                scheduler.step()

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                cm = evaluate(model, eval_loader, num_classes, device)
                ious = iou_per_class(cm)
                metrics = {
                    "miou_with_bg": mean_iou(ious, True),
                    "miou_no_bg": mean_iou(ious, False),
                    **{f"iou_{id_to_name[i]}": v for i, v in enumerate(ious) if v is not None},
                }
                mlflow.log_metrics(metrics, step=epoch)
                print(
                    f"epoch {epoch:4d} | loss {total / count:.4f} | "
                    f"mIoU no bg {metrics['miou_no_bg']:.3f} with bg {metrics['miou_with_bg']:.3f} "
                    f"| {time.time() - start:.0f}s",
                    flush=True,
                )

        elapsed = time.time() - start
        mlflow.log_metrics({"train_seconds": elapsed, "steps": step})
        if device == "cuda":
            mlflow.log_metric("peak_gpu_mem_mb", torch.cuda.max_memory_allocated() / 1024**2)

        per_class = {id_to_name[i]: v for i, v in enumerate(ious)}
        summary = {
            "run_id": run.info.run_id,
            "images": keys,
            "final_iou": per_class,
            "baseline_iou": {id_to_name[i]: v for i, v in enumerate(base_ious)},
        }
        names = [id_to_name[i] for i in range(num_classes)]
        # rows = true class, columns = predicted class (pixel counts, ignored pixels excluded)
        cm_table = {"rows_true_cols_pred": names, "counts": cm.tolist()}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.json").write_text(json.dumps(summary, indent=1))
            (Path(tmp) / "confusion_matrix.json").write_text(json.dumps(cm_table, indent=1))
            torch.save(model.state_dict(), Path(tmp) / "model_state.pt")
            mlflow.log_artifacts(tmp)

        print(f"\ndone in {elapsed:.0f}s | run {run.info.run_id}")
        print(f"{'class':22s} {'final IoU':>9s} {'baseline':>9s}")
        for i in range(num_classes):
            fmt = lambda v: "   n/a" if v is None else f"{v:9.3f}"  # noqa: E731
            print(f"{id_to_name[i]:22s} {fmt(ious[i])} {fmt(base_ious[i])}")
        print("\nconfusion matrix (% of each true class; rows = true, columns = predicted)")
        print(" " * 22 + "".join(f"{n[:10]:>11s}" for n in names))
        for i, row in enumerate(cm.tolist()):
            total_row = sum(row) or 1
            print(f"{names[i]:22s}" + "".join(f"{100 * v / total_row:10.1f}%" for v in row))
        print(f"tracking: {os.environ.get('MLFLOW_TRACKING_URI', mlflow.get_tracking_uri())}")


if __name__ == "__main__":
    main()
