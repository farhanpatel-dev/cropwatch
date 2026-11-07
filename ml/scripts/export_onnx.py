"""Export a trained U-Net checkpoint to ONNX for the inference service, then verify it.

Writes three files to --out-dir (default services/inference/models/):
  model.onnx       input "image" (N x 3 x 256 x 256, float32, normalised),
                   output "logits" (N x C x 256 x 256)
  model_info.json  name, version, MLflow run id, tile size, classes, preprocessing (mean/std)
  classes.yaml     copied unchanged from ml/configs/classes.yaml

Verification: the same real val tiles go through PyTorch (CPU, float32) and ONNX Runtime (CPU);
we report the largest logit difference and how many pixels get the same predicted class.

Usage:
  python ml/scripts/export_onnx.py --run-id e9ed5ab1b49941c488489c8e8ee9de16 --version 1
"""

import torch  # first: torch before mlflow (import-order rule)

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import yaml

from cropwatch_ml.datasets import list_keys, read_rgb
from cropwatch_ml.models.unet import UNet

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True, help="MLflow run that produced the checkpoint")
    p.add_argument("--checkpoint", type=Path, help="default: ml/checkpoints/<run-id>/best.pt")
    p.add_argument("--version", default="1", help="model version reported by /model-info")
    p.add_argument("--out-dir", type=Path, default=ROOT / "services/inference/models")
    p.add_argument("--classes", type=Path, default=ROOT / "ml/configs/classes.yaml")
    p.add_argument("--val-dir", type=Path, default=ROOT / "ml/data/agvision2021/val")
    p.add_argument(
        "--verify-images", type=int, default=32, help="val images used to compare outputs"
    )
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    return p.parse_args()


def run_summary(run_id: str, tracking_uri: str) -> tuple[dict, dict]:
    """summary.json (mean/std, tile size, scores) and params of the training run."""
    mlflow.set_tracking_uri(tracking_uri)
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="summary.json")
    summary = json.loads(Path(path).read_text())
    params = mlflow.get_run(run_id).data.params
    return summary, params


def val_tiles(val_dir: Path, n: int, tile: int, mean: list[float], std: list[float]) -> np.ndarray:
    """Top-left tile of n val images, normalised exactly as in training. N x 3 x tile x tile."""
    keys = list_keys(val_dir)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, n).astype(int)]  # spread over the split
    m = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    tiles = []
    for key in keys:
        rgb = read_rgb(val_dir, key)[:tile, :tile].astype(np.float32) / 255.0  # H x W x 3
        tiles.append((rgb.transpose(2, 0, 1) - m) / s)
    return np.stack(tiles).astype(np.float32)


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint or ROOT / "ml/checkpoints" / args.run_id / "best.pt"
    summary, params = run_summary(args.run_id, args.tracking_uri)
    prep = summary["preprocessing"]
    tile = int(prep["tile_size"])
    classes = yaml.safe_load(args.classes.read_text())["classes"]

    model = UNet(in_channels=3, num_classes=len(classes), base=int(params["base_channels"]))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()  # batch norm uses its running statistics, no dropout

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "model.onnx"
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        model,
        (torch.zeros(2, 3, tile, tile),),
        onnx_path,
        dynamo=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch}},  # any batch size; tile size stays fixed
        opset_version=args.opset,
        external_data=False,  # one self-contained file (weights are ~31 MB)
    )

    # Same tiles through both engines.
    x = val_tiles(args.val_dir, args.verify_images, tile, prep["mean"], prep["std"])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"image": x})[0]
    max_diff = float(np.abs(ref - out).max())
    same = ref.argmax(1) == out.argmax(1)
    agree = float(same.mean())
    print(
        f"verify on {len(x)} val tiles: max |logit diff| {max_diff:.2e}, "
        f"different class on {int((~same).sum())} of {same.size} pixels ({agree:.7f} agree)"
    )
    if agree < 0.999 or max_diff > 1e-2:
        raise SystemExit("ONNX output does not match PyTorch, not writing model_info.json")

    info = {
        "name": "unet",
        "version": str(args.version),
        "mlflow_run_id": args.run_id,
        "checkpoint_epoch": summary["best_epoch"],
        "input_channels": 3,
        "tile_size": tile,
        "classes": [{"id": c["id"], "name": c["name"]} for c in classes],
        "input": {"name": "image", "shape": ["N", 3, tile, tile], "dtype": "float32"},
        "output": {
            "name": "logits",
            "shape": ["N", len(classes), tile, tile],
            "note": "argmax over axis 1",
        },
        "preprocessing": {
            "steps": "rgb uint8 -> float / 255 -> (x - mean) / std per channel, HWC -> CHW",
            "mean": prep["mean"],
            "std": prep["std"],
        },
        "opset": args.opset,
        "verified": {
            "val_tiles": len(x),
            "max_abs_logit_diff": max_diff,
            "argmax_agreement": agree,
        },
    }
    (args.out_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    shutil.copyfile(args.classes, args.out_dir / "classes.yaml")
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path} ({size_mb:.1f} MB), model_info.json, classes.yaml to {args.out_dir}")


if __name__ == "__main__":
    main()
