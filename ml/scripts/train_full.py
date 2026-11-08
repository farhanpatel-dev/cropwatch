"""Train a U-Net on the full Agriculture-Vision 2021 train split, score on the val split.

--model unet           our own U-Net from scratch (run e9ed5ab1).
--model unet-resnet34  U-Net with an ImageNet-pretrained ResNet34 encoder.

The official split is by field (no field is in both train/ and val/), so we use it as is.
- Train: one random 256 crop per image per epoch, flips and 90-degree rotations.
- Images are read from disk by worker processes each step (the full set does not fit in RAM).
- Loss: cross-entropy (class-weighted) + Dice, ignored pixels (255) left out.
- Every epoch: score on a fixed sample of val images (fast). At the end: load the best epoch
  and score it on ALL val images. Those final numbers are the ones to report.
- Mean/std and class weights come from a fixed sample of train images only.
- best.pt / last.pt are saved to ml/checkpoints/<run_id>/ after every eval, so a crash
  does not lose the best model. Everything is also logged to MLflow at the end.

Run from the repo root (Compose stack running):
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    ml\\.venv\\Scripts\\python.exe ml\\scripts\\train_full.py
Short trial (check speed and GPU memory first):
    ml\\.venv\\Scripts\\python.exe ml\\scripts\\train_full.py --max-train-images 2000 ^
        --val-images 200 --epochs 1 --skip-final-val --experiment cropwatch-trial
"""

import torch
from torch.utils.data import DataLoader

import argparse
import gc
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
    AgriVisionDataset,
    list_keys,
    load_classes,
    read_rgb,
    read_target,
)
from cropwatch_ml.losses import CEDiceLoss
from cropwatch_ml.metrics import confusion_matrix, iou_per_class, mean_iou
from cropwatch_ml.models import MODELS, build_model

ML_DIR = Path(__file__).resolve().parents[1]


def sample_stats(
    split_dir: Path, keys: list[str], class_ids: dict[str, int]
) -> tuple[list[float], list[float], list[float]]:
    """Mean/std of RGB (0-1, valid pixels only) and pixel share of each class, in one pass."""
    num_classes = len(class_ids)
    total = np.zeros(3)
    total_sq = np.zeros(3)
    count = 0
    class_px = np.zeros(num_classes)
    for key in keys:
        target = read_target(split_dir, key, class_ids)
        valid = target != IGNORE_INDEX
        px = read_rgb(split_dir, key)[valid].astype(np.float64) / 255.0
        total += px.sum(0)
        total_sq += (px**2).sum(0)
        count += len(px)
        class_px += np.bincount(target[valid], minlength=num_classes)[:num_classes]
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return mean.tolist(), std.tolist(), (class_px / class_px.sum()).tolist()


def class_weights_from(freq: list[float], power: float = 0.5) -> list[float]:
    """1 / (pixel share ** power), scaled to average 1. Rare classes get bigger weights.
    power 1 = full inverse frequency, 0.5 = 1/sqrt (keeps a 1% class from getting 100x the
    weight of background), smaller = softer (fewer false alarms on background), 0 = equal."""
    w = 1 / np.maximum(freq, 1e-6) ** power
    return (w / w.mean()).tolist()


def evaluate(
    model: torch.nn.Module, loader: DataLoader, num_classes: int, device: str, amp: bool
) -> torch.Tensor:
    """Confusion matrix (rows = true, columns = predicted) over the loader."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            cm += confusion_matrix(model(x).argmax(1), y, num_classes)
    model.train()
    return cm.cpu()


def baseline_cm(cm: torch.Tensor) -> torch.Tensor:
    """What the confusion matrix would be if the model said 'background' everywhere."""
    base = torch.zeros_like(cm)
    base[:, 0] = cm.sum(1)
    return base


def scores(cm: torch.Tensor, names: list[str], prefix: str = "") -> dict[str, float]:
    ious = iou_per_class(cm)
    out = {
        f"{prefix}miou_with_bg": mean_iou(ious, True),
        f"{prefix}miou_no_bg": mean_iou(ious, False),
    }
    out.update({f"{prefix}iou_{names[i]}": v for i, v in enumerate(ious) if v is not None})
    return out


def iou_table(cm: torch.Tensor, names: list[str]) -> str:
    ious, base = iou_per_class(cm), iou_per_class(baseline_cm(cm))
    true_px = cm.sum(1).tolist()

    def fmt(v):
        return "      n/a" if v is None else f"{v:9.3f}"

    lines = [f"{'class':22s} {'IoU':>9s} {'baseline':>9s} {'true px':>13s}"]
    lines += [
        f"{n:22s} {fmt(ious[i])} {fmt(base[i])} {true_px[i]:13,d}" for i, n in enumerate(names)
    ]
    lines.append(
        f"{'mIoU no bg / with bg':22s} {mean_iou(ious, False):9.3f} / {mean_iou(ious, True):.3f}"
        f"  (baseline {mean_iou(base, False):.3f} / {mean_iou(base, True):.3f})"
    )
    return "\n".join(lines)


def cm_table(cm: torch.Tensor, names: list[str]) -> str:
    lines = ["confusion matrix (% of each true class; rows = true, columns = predicted)"]
    lines.append(" " * 22 + "".join(f"{n[:10]:>11s}" for n in names))
    for i, row in enumerate(cm.tolist()):
        total_row = sum(row) or 1
        lines.append(f"{names[i]:22s}" + "".join(f"{100 * v / total_row:10.1f}%" for v in row))
    return "\n".join(lines)


def main() -> None:
    data = ML_DIR / "data/agvision2021"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=data / "train")
    p.add_argument("--val-dir", type=Path, default=data / "val")
    p.add_argument("--classes", type=Path, default=ML_DIR / "configs/classes.yaml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help="pretrained encoder lr = lr * this (e.g. 0.1); the decoder keeps lr",
    )
    p.add_argument(
        "--colour-jitter", type=float, default=0.0, help="brightness/contrast/saturation +-j"
    )
    p.add_argument("--model", choices=MODELS, default="unet")
    p.add_argument("--base-channels", type=int, default=32, help="own unet only")
    p.add_argument("--dice-weight", type=float, default=1.0, help="0 = cross-entropy only")
    p.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    p.add_argument("--class-weight-power", type=float, default=0.5, help="weight = 1/share**power")
    p.add_argument("--stats-images", type=int, default=2000, help="train images for mean/std")
    p.add_argument("--max-train-images", type=int, default=None, help="for short trials")
    p.add_argument("--val-images", type=int, default=2000, help="val images scored each epoch")
    p.add_argument("--skip-final-val", action="store_true", help="skip scoring on ALL val")
    p.add_argument("--workers", type=int, default=4, help="data loading processes (train)")
    p.add_argument("--val-workers", type=int, default=2, help="data loading processes (val)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-grad", type=float, default=5.0, help="max gradient norm per step")
    p.add_argument("--max-skipped", type=int, default=100, help="stop after this many NaN steps")
    p.add_argument("--log-every", type=int, default=200, help="print progress every N steps")
    p.add_argument("--checkpoint-dir", type=Path, default=ML_DIR / "checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", default="cropwatch-full-dataset")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"
    torch.backends.cudnn.benchmark = True  # same input size every step: let cuDNN pick fast kernels
    class_ids = load_classes(args.classes)
    names = [n for n, _ in sorted(class_ids.items(), key=lambda kv: kv[1])]
    num_classes = len(names)

    train_keys = list_keys(args.train_dir)
    all_val_keys = list_keys(args.val_dir)
    if not train_keys or not all_val_keys:
        raise SystemExit(f"no images found in {args.train_dir} or {args.val_dir}")
    rng = random.Random(args.seed)
    if args.max_train_images:
        train_keys = sorted(rng.sample(train_keys, min(args.max_train_images, len(train_keys))))
    val_keys = sorted(rng.sample(all_val_keys, min(args.val_images, len(all_val_keys))))
    stats_keys = rng.sample(train_keys, min(args.stats_images, len(train_keys)))

    t0 = time.time()
    mean, std, freq = sample_stats(args.train_dir, stats_keys, class_ids)
    if args.class_weights == "auto":
        weights = class_weights_from(freq, args.class_weight_power)
    else:
        weights = [1.0] * num_classes
    print(f"stats from {len(stats_keys)} train images in {time.time() - t0:.0f}s")
    print(f"  mean {[round(m, 4) for m in mean]} std {[round(s, 4) for s in std]}")
    for n, f, w in zip(names, freq, weights, strict=True):
        print(f"  {n:22s} {100 * f:6.2f}% of valid px, loss weight {w:.2f}")

    common = dict(classes_path=args.classes, mean=mean, std=std)
    train_ds = AgriVisionDataset(
        args.train_dir,
        train_keys,
        mode="train",
        augment=True,
        colour_jitter=args.colour_jitter,
        **common,
    )
    val_ds = AgriVisionDataset(args.val_dir, val_keys, mode="val", **common)

    # On Windows every worker process loads PyTorch's CUDA libraries again and reserves about
    # 1.4 GB of memory commit. Too many workers at once hit the Windows commit limit and a new
    # worker silently fails to start (the run hangs). Keep the total small: train + val workers.
    def loader(ds, workers: int, persistent: bool, **kw) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=workers,
            pin_memory=device == "cuda",
            persistent_workers=persistent and workers > 0,
            **kw,
        )

    train_loader = loader(train_ds, args.workers, True, shuffle=True, drop_last=True)
    val_loader = loader(val_ds, args.val_workers, True)
    steps_per_epoch = len(train_loader)
    print(
        f"device {device} | amp {amp} | train {len(train_keys)} images ({steps_per_epoch} steps"
        f"/epoch) | eval each epoch on {len(val_keys)} val images ({len(val_ds)} tiles)",
        flush=True,
    )

    model = build_model(args.model, num_classes, args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = CEDiceLoss(torch.tensor(weights, device=device), args.dice_weight)
    # Pretrained encoder: smaller steps, so its ImageNet features are tuned, not overwritten.
    # Group 0 (decoder and head) keeps the full lr; that is the lr we log.
    encoder = getattr(model, "encoder", None)
    if args.encoder_lr_scale != 1.0 and encoder is None:
        raise SystemExit(f"--encoder-lr-scale needs a model with an encoder, not {args.model}")
    enc_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    groups = [{"params": [p for p in model.parameters() if id(p) not in enc_ids]}]
    if enc_ids:
        groups.append({"params": list(encoder.parameters()), "lr": args.lr * args.encoder_lr_scale})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    # Cosine over every step (not every epoch), so the lr falls smoothly even with few epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        ckpt_dir = args.checkpoint_dir / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params(
            {
                **{k: v for k, v in vars(args).items() if not isinstance(v, Path)},
                "train_images": len(train_keys),
                "val_images_each_epoch": len(val_keys),
                "val_images_total": len(all_val_keys),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "torch": torch.__version__,
                "params": n_params,
                "tile_size": train_ds.tile,
                "input_channels": 3,
                "augment": "flips+rot90"
                + (f"+colour{args.colour_jitter}" if args.colour_jitter else ""),
                "loss": f"ce(weighted={args.class_weights}, power={args.class_weight_power})"
                f"+{args.dice_weight}*dice",
                "norm_mean": json.dumps([round(m, 4) for m in mean]),
                "norm_std": json.dumps([round(s, 4) for s in std]),
                "class_weights": json.dumps([round(w, 3) for w in weights]),
                "train_class_px_share": json.dumps([round(f, 5) for f in freq]),
            }
        )
        mlflow.log_artifact(str(args.classes))
        print(f"mlflow run {run_id} | params {n_params:,} | checkpoints {ckpt_dir}", flush=True)

        start = time.time()
        best = {"miou_no_bg": -1.0, "epoch": 0, "cm": None}
        step = 0
        skipped = 0
        for epoch in range(1, args.epochs + 1):
            total, count, t_epoch = 0.0, 0, time.time()
            norm_sum, norm_max, norm_n = 0.0, 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                step += 1
                grad_norm = torch.tensor(float("nan"))
                if torch.isfinite(loss):
                    loss.backward()
                    # Cap the size of one update: a single extreme batch cannot wreck the weights.
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    opt.step()
                    total += loss.item() * len(x)
                    count += len(x)
                    norm_sum += grad_norm.item()
                    norm_max = max(norm_max, grad_norm.item())
                    norm_n += 1
                else:
                    # Skip this update instead of letting NaN/inf into the weights; say why.
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    print(
                        f"  SKIPPED step {step}: loss {loss.item()} grad_norm {grad_norm.item()}"
                        f" | valid px {(y != IGNORE_INDEX).sum().item()} | input finite "
                        f"{torch.isfinite(x).all().item()} | logits finite "
                        f"{torch.isfinite(logits).all().item()} max |logit| "
                        f"{logits.float().abs().max().item():.1f}",
                        flush=True,
                    )
                    if skipped > args.max_skipped:
                        raise SystemExit(f"stopping: {skipped} skipped steps (see lines above)")
                scheduler.step()
                if step % args.log_every == 0:
                    rate = (count + skipped * len(x)) / (time.time() - t_epoch)
                    eta = (steps_per_epoch * args.batch_size - count) / rate
                    print(
                        f"  epoch {epoch} step {step % steps_per_epoch or steps_per_epoch}"
                        f"/{steps_per_epoch} | loss {total / max(count, 1):.4f} | grad norm "
                        f"mean {norm_sum / max(norm_n, 1):.2f} max {norm_max:.2f} | "
                        f"{rate:.0f} img/s | epoch ends in ~{eta / 60:.0f} min",
                        flush=True,
                    )
                    norm_sum, norm_max, norm_n = 0.0, 0.0, 0
            train_secs = time.time() - t_epoch
            cm = evaluate(model, val_loader, num_classes, device, amp)
            metrics = scores(cm, names)
            metrics.update(
                {
                    "train_loss": total / max(count, 1),
                    "skipped_steps_total": skipped,
                    "lr": opt.param_groups[0]["lr"],
                    "epoch_train_seconds": train_secs,
                }
            )
            mlflow.log_metrics(metrics, step=epoch)
            if epoch == 1:
                mlflow.log_metrics(scores(baseline_cm(cm), names, prefix="baseline_"))
            torch.save(model.state_dict(), ckpt_dir / "last.pt")
            if metrics["miou_no_bg"] > best["miou_no_bg"]:
                best = {"miou_no_bg": metrics["miou_no_bg"], "epoch": epoch, "cm": cm}
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(
                f"epoch {epoch:3d} | train loss {total / max(count, 1):.4f} | val mIoU no bg "
                f"{metrics['miou_no_bg']:.3f} with bg {metrics['miou_with_bg']:.3f} | best "
                f"{best['miou_no_bg']:.3f} @{best['epoch']} | epoch {train_secs / 60:.1f} min | "
                f"total {(time.time() - start) / 60:.0f} min",
                flush=True,
            )

        elapsed = time.time() - start
        mlflow.log_metrics({"train_seconds": elapsed, "best_epoch": best["epoch"]})
        if device == "cuda":
            mlflow.log_metric("peak_gpu_mem_mb", torch.cuda.max_memory_allocated() / 1024**2)
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        # The honest numbers: best epoch scored on every val image.
        final_cm, scored_on = best["cm"], f"{len(val_keys)} sampled val images"
        # Shut down the training/val worker processes before starting new ones (memory, above).
        del train_loader, val_loader
        gc.collect()
        if not args.skip_final_val:
            model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
            full_val = AgriVisionDataset(args.val_dir, all_val_keys, mode="val", **common)
            print(f"\nscoring best epoch on all {len(all_val_keys)} val images...", flush=True)
            final_loader = loader(full_val, args.workers, False)
            final_cm = evaluate(model, final_loader, num_classes, device, amp)
            scored_on = f"all {len(all_val_keys)} val images ({len(full_val)} tiles)"
            mlflow.log_metrics(scores(final_cm, names, prefix="final_"))
            mlflow.log_metrics(scores(baseline_cm(final_cm), names, prefix="final_baseline_"))

        summary = {
            "run_id": run_id,
            "best_epoch": best["epoch"],
            "scored_on": scored_on,
            "iou": dict(zip(names, iou_per_class(final_cm), strict=True)),
            "baseline_iou": dict(zip(names, iou_per_class(baseline_cm(final_cm)), strict=True)),
            "model": args.model,
            "preprocessing": {"mean": mean, "std": std, "tile_size": train_ds.tile},
            "class_weights": dict(zip(names, weights, strict=True)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "summary.json").write_text(json.dumps(summary, indent=1))
            (tmp / "confusion_matrix_best.json").write_text(
                json.dumps({"rows_true_cols_pred": names, "counts": final_cm.tolist()})
            )
            mlflow.log_artifacts(str(tmp))
        mlflow.log_artifact(str(ckpt_dir / "best.pt"))

        print(f"\ndone in {elapsed / 60:.0f} min | run {run_id} | best epoch {best['epoch']}")
        print(f"scored on {scored_on}:\n{iou_table(final_cm, names)}")
        print(f"\n{cm_table(final_cm, names)}")
        print(f"\ntracking: {os.environ.get('MLFLOW_TRACKING_URI', mlflow.get_tracking_uri())}")


if __name__ == "__main__":
    main()
