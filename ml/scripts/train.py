"""Train the U-Net and log everything to MLflow.

--mode overfit (default): memorise ~20 images (80 fixed tiles) and score on the same tiles.
    A pipeline check: if the loss cannot get close to zero, something is broken.
--mode full: train on all train fields (random 256 crops, flips/rotations) and score on the
    held-out val fields (4 fixed tiles per image). These are the honest numbers.
    The best epoch by val mIoU (no background) is saved as best_model_state.pt.

Run from the repo root (Compose stack running):
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    ml\\.venv\\Scripts\\python.exe ml\\scripts\\train.py --mode full
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
import copy
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


def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, num_classes: int, device: str
) -> tuple[torch.Tensor, float]:
    """Confusion matrix and mean loss over the loader (model in eval mode)."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += loss_fn(logits, y).item() * len(x)
            count += len(x)
            cm += confusion_matrix(logits.argmax(1), y, num_classes)
    model.train()
    return cm.cpu(), total / count


def iou_table(ious, base_ious, id_to_name) -> str:
    def fmt(v):
        return "      n/a" if v is None else f"{v:9.3f}"

    lines = [f"{'class':22s} {'IoU':>9s} {'baseline':>9s}"]
    lines += [f"{id_to_name[i]:22s} {fmt(ious[i])} {fmt(base_ious[i])}" for i in range(len(ious))]
    return "\n".join(lines)


def cm_table(cm: torch.Tensor, names: list[str]) -> str:
    lines = ["confusion matrix (% of each true class; rows = true, columns = predicted)"]
    lines.append(" " * 22 + "".join(f"{n[:10]:>11s}" for n in names))
    for i, row in enumerate(cm.tolist()):
        total_row = sum(row) or 1
        lines.append(f"{names[i]:22s}" + "".join(f"{100 * v / total_row:10.1f}%" for v in row))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["overfit", "full"], default="overfit")
    p.add_argument("--split-dir", type=Path, default=ML_DIR / "data/agvision2021_subset/val")
    p.add_argument("--split-file", type=Path, default=ML_DIR / "configs/split_day1.json")
    p.add_argument("--classes", type=Path, default=ML_DIR / "configs/classes.yaml")
    p.add_argument("--n-images", type=int, default=20, help="overfit mode only")
    p.add_argument("--epochs", type=int, default=None, help="default: 600 overfit, 150 full")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help="cosine: lr falls smoothly from --lr to ~0 over all epochs",
    )
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=None, help="default: 10 overfit, 5 full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", default=None, help="default: cropwatch-<mode>")
    args = p.parse_args()
    full = args.mode == "full"
    args.epochs = args.epochs or (150 if full else 600)
    args.eval_every = args.eval_every or (5 if full else 10)
    args.experiment = args.experiment or f"cropwatch-{args.mode}"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_ids = load_classes(args.classes)
    id_to_name = {v: k for k, v in class_ids.items()}
    names = [id_to_name[i] for i in range(len(class_ids))]
    num_classes = len(class_ids)

    train_keys, val_keys = load_split(args.split_file)
    mean, std = compute_mean_std(args.split_dir, train_keys)  # train fields only
    common = dict(classes_path=args.classes, mean=mean, std=std, cache=True)

    if full:
        train_ds = AgriVisionDataset(
            args.split_dir, train_keys, mode="train", augment=True, **common
        )
        eval_ds = AgriVisionDataset(args.split_dir, val_keys, mode="val", **common)
        eval_name = f"val ({len(val_keys)} images, {len(eval_ds)} tiles, 6 held-out fields)"
        extra_params = {"train_images": len(train_keys), "val_images": len(val_keys)}
    else:
        keys, per_class = pick_images(
            args.split_dir, train_keys, class_ids, args.n_images, args.seed
        )
        ds = AgriVisionDataset(args.split_dir, keys, mode="val", **common)
        xs, ys = zip(*(ds[i] for i in range(len(ds))), strict=True)
        train_ds = eval_ds = TensorDataset(torch.stack(xs), torch.stack(ys))
        eval_name = f"the same {len(eval_ds)} training tiles"
        extra_params = {"images_per_class": json.dumps(per_class), "tiles": len(eval_ds)}
    print(f"device {device} | mode {args.mode} | scoring on {eval_name}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=full)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    model = UNet(in_channels=3, num_classes=num_classes, base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # All-background baseline on the scoring tiles.
    all_y = torch.cat([y for _, y in eval_loader])
    base_ious = iou_per_class(confusion_matrix(torch.zeros_like(all_y), all_y, num_classes))

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **{k: v for k, v in vars(args).items() if not isinstance(v, Path)},
                **extra_params,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "torch": torch.__version__,
                "model": "unet",
                "params": n_params,
                "tile_size": 256,
                "input_channels": 3,
                "augment": "flips+rot90" if full else "none",
                "norm_mean": json.dumps([round(m, 4) for m in mean]),
                "norm_std": json.dumps([round(s, 4) for s in std]),
                "scored_on": eval_name,
            }
        )
        mlflow.log_metrics(
            {
                "baseline_miou_with_bg": mean_iou(base_ious, True),
                "baseline_miou_no_bg": mean_iou(base_ious, False),
            }
        )
        mlflow.log_artifact(str(args.classes))
        mlflow.log_artifact(str(args.split_file))
        print(f"mlflow run {run.info.run_id} | params {n_params:,}", flush=True)

        start = time.time()
        best = {"miou_no_bg": -1.0, "epoch": 0, "state": None, "cm": None, "ious": None}
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
            mlflow.log_metrics(
                {"train_loss": total / count, "lr": opt.param_groups[0]["lr"]}, step=epoch
            )
            if scheduler is not None:
                scheduler.step()

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                cm, eval_loss = evaluate(model, eval_loader, loss_fn, num_classes, device)
                ious = iou_per_class(cm)
                metrics = {
                    "eval_loss": eval_loss,
                    "miou_with_bg": mean_iou(ious, True),
                    "miou_no_bg": mean_iou(ious, False),
                    **{f"iou_{id_to_name[i]}": v for i, v in enumerate(ious) if v is not None},
                }
                mlflow.log_metrics(metrics, step=epoch)
                if metrics["miou_no_bg"] > best["miou_no_bg"]:
                    best = {
                        "miou_no_bg": metrics["miou_no_bg"],
                        "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict()),
                        "cm": cm,
                        "ious": ious,
                    }
                print(
                    f"epoch {epoch:4d} | train loss {total / count:.4f} | eval loss "
                    f"{eval_loss:.4f} | mIoU no bg {metrics['miou_no_bg']:.3f} with bg "
                    f"{metrics['miou_with_bg']:.3f} | best {best['miou_no_bg']:.3f} "
                    f"@{best['epoch']} | {time.time() - start:.0f}s",
                    flush=True,
                )

        elapsed = time.time() - start
        mlflow.log_metrics(
            {
                "train_seconds": elapsed,
                "best_epoch": best["epoch"],
                "best_miou_no_bg": best["miou_no_bg"],
                "best_miou_with_bg": mean_iou(best["ious"], True),
                **{
                    f"best_iou_{id_to_name[i]}": v
                    for i, v in enumerate(best["ious"])
                    if v is not None
                },
            }
        )
        if device == "cuda":
            mlflow.log_metric("peak_gpu_mem_mb", torch.cuda.max_memory_allocated() / 1024**2)

        summary = {
            "run_id": run.info.run_id,
            "mode": args.mode,
            "scored_on": eval_name,
            "best_epoch": best["epoch"],
            "best_iou": {id_to_name[i]: v for i, v in enumerate(best["ious"])},
            "final_iou": {id_to_name[i]: v for i, v in enumerate(ious)},
            "baseline_iou": {id_to_name[i]: v for i, v in enumerate(base_ious)},
            "norm_mean": mean,
            "norm_std": std,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "summary.json").write_text(json.dumps(summary, indent=1))
            (tmp / "confusion_matrix_best.json").write_text(
                json.dumps({"rows_true_cols_pred": names, "counts": best["cm"].tolist()})
            )
            torch.save(model.state_dict(), tmp / "model_state.pt")
            torch.save(best["state"], tmp / "best_model_state.pt")
            mlflow.log_artifacts(str(tmp))

        print(f"\ndone in {elapsed:.0f}s | run {run.info.run_id} | scored on {eval_name}")
        print(f"\nfinal epoch {args.epochs}:\n{iou_table(ious, base_ious, id_to_name)}")
        print(f"\nbest epoch {best['epoch']}:\n{iou_table(best['ious'], base_ious, id_to_name)}")
        print(f"\nbest epoch {cm_table(best['cm'], names)}")
        print(f"\ntracking: {os.environ.get('MLFLOW_TRACKING_URI', mlflow.get_tracking_uri())}")


if __name__ == "__main__":
    main()
