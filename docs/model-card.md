# Model card: CropWatch crop-stress segmentation, version 3

| | |
| --- | --- |
| Name / version | `unet-resnet34`, version 3 (served by the inference service, `GET /model-info`) |
| MLflow run | `6d9eee7ff25843c68a0d40682deaadd1` (experiment `cropwatch-full-dataset`), best epoch 4 of 20 |
| Date | 2026-11-08 |
| Owner | Farhan (ML) |
| Files | `model.onnx` (98 MB, opset 18), `model_info.json`, `classes.yaml` in `services/inference/models/`, written by `ml/scripts/export_onnx.py`. Committed in Git (98 MB) |

## What it does
Semantic segmentation of RGB aerial farm images: every pixel gets one of four classes. The inference service turns the result into polygons with an area and a confidence, shown on a map.

| id | class | meaning |
| --- | --- | --- |
| 0 | background | everything else, including healthy crop |
| 1 | nutrient_deficiency | crop showing nutrient stress |
| 2 | drydown | crop drying down |
| 3 | weed_cluster | patches of weeds |

Intended use: a demo and learning project that shows where a field may need a closer look. It is not validated for agronomic decisions; a person should check flagged areas on the ground.

Not intended for: commercial use (see Data licence), images that are not top-down aerial RGB farm images, or automatic actions such as spraying without human review.

## Data
- Agriculture-Vision 2021 supervised dataset (Chiu et al., "Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis", CVPR 2020; 2021 challenge release). 512 x 512 images.
- Train 56,944 images, validation 18,334 images. The two splits come from different fields, so validation measures performance on unseen fields.
- Only the RGB channels are used; the dataset's NIR channel is not.
- The dataset has 9 stress labels. We keep 3; the other 6 (double_plant, endrow, planter_skip, storm_damage, water, waterway) are counted as background.
- Pixels outside the field mask or on the field boundary are ignored (not trained on, not scored).
- When a pixel has more than one of our labels, the rarer class wins: weed_cluster > nutrient_deficiency > drydown.
- Pixel share in train (valid pixels): background 72.9%, nutrient_deficiency 7.9%, drydown 12.4%, weed_cluster 6.8%.

### Data licence
Agriculture-Vision is released for non-commercial use. This model is trained on it, so the model and its outputs carry the same restriction. Check the dataset's own terms before any other use.

## Model and training
- U-Net with a ResNet34 encoder pretrained on ImageNet (segmentation_models_pytorch 0.6.0), 24.4 M parameters.
- Input 3 x 256 x 256. Training uses random 256 crops of the 512 images.
- Augmentation: random flips, 90-degree rotations, and colour jitter 0.2 (brightness, contrast, saturation).
- Loss: weighted cross-entropy + Dice. Class weights 1 / (pixel share)^0.25: background 0.655, nutrient_deficiency 1.141, drydown 1.019, weed_cluster 1.184.
- AdamW, learning rate 3e-4 (encoder 3e-5), weight decay 0.01, cosine schedule per step, gradient clipping 5, bf16 mixed precision, batch 16.
- 20 epochs, 3.1 h on an RTX 5050 Laptop GPU (8 GB; peak use 1.35 GB). The best epoch by validation mIoU on 2,000 sampled validation images was epoch 4.

### Input normalisation (must match exactly)
uint8 RGB -> divide by 255 -> per channel (x - mean) / std -> channels first. Values are stored in `model_info.json` under `preprocessing`; inference reads them from there.
- mean (R, G, B): 0.4365, 0.4444, 0.4423
- std (R, G, B): 0.1645, 0.1524, 0.1566

## Results
Scored on all 18,334 validation images (unseen fields), 4 fixed 256 x 256 tiles per image, ignored pixels excluded. IoU (intersection over union) is the overlap between predicted and true area, from 0 to 1. The baseline predicts background everywhere.

| | v3 (this model) | Baseline (all background) |
| --- | --- | --- |
| **mIoU without background** | **0.371** | 0.000 |
| **mIoU with background** | **0.440** | 0.181 |
| background | 0.645 | 0.722 |
| nutrient_deficiency | 0.315 | 0 |
| drydown | 0.558 | 0 |
| weed_cluster | 0.241 | 0 |

Share of each true class predicted as each class (rows sum to 100%):

| true \ predicted | background | nutrient_deficiency | drydown | weed_cluster |
| --- | --- | --- | --- | --- |
| background | 70.5% | 11.4% | 10.7% | 7.4% |
| nutrient_deficiency | 33.8% | **64.6%** | 1.4% | 0.3% |
| drydown | 13.5% | 0.4% | **84.3%** | 1.9% |
| weed_cluster | 44.8% | 0.4% | 0.4% | **54.3%** |

### Compared with earlier versions (same validation set)
| | v1 own U-Net (`e9ed5ab1`) | v2 ResNet34 (`0bf70274`) | v3 (`6d9eee7f`) |
| --- | --- | --- | --- |
| mIoU without background | 0.327 | 0.336 | **0.371** |
| mIoU with background | 0.404 | 0.421 | **0.440** |
| nutrient_deficiency | 0.251 | 0.247 | **0.315** |
| drydown | 0.519 | 0.539 | **0.558** |
| weed_cluster | 0.212 | 0.222 | **0.241** |
| background | 0.637 | **0.676** | 0.645 |
| Background predicted as stress | 27.9% | **23.0%** | 29.5% |

v3 is best on every stress class. The cost is more false alarms on healthy areas than v2.

### Checks of the deployed model
- ONNX vs PyTorch on 32 validation tiles: largest logit difference 2.8e-5; predicted class identical on all 2,097,152 pixels.
- Full inference pipeline (whole 512 images, overlapping tiles, stitched) on 300 random validation images: mIoU without background 0.382, with background 0.443 (baseline 0.000 / 0.179). This agrees with the training-time score, so normalisation and stitching are correct.
- Speed on CPU (16 logical CPUs on the dev laptop, ONNX Runtime, in Docker): 512 x 512 image 0.8 s per request; 6000 x 6000 image 40-41 s. The service therefore accepts at most 6000 x 6000 pixels (gateway timeout 60 s). Fewer CPU cores means slower requests.

## Limitations and known weaknesses
- **Misses stress.** About a third of nutrient_deficiency pixels and almost half of weed_cluster pixels are predicted as background. An empty result does not mean a healthy field.
- **False alarms.** 29.5% of background pixels are labelled as some stress class. Some of these are likely the 6 merged labels (for example double_plant, planter_skip) that look like stress but count as background; this is not yet measured.
- **Results vary between fields.** During training the validation score moved by about +-0.03 from epoch to epoch, and each version was trained once (seed 0). Differences between versions smaller than about 0.03 may be noise.
- **RGB only.** NIR, the usual signal for plant health, is not used. This is the most likely way to improve results (planned after the demo).
- **Domain.** Trained on Agriculture-Vision imagery only. Other cameras, resolutions, seasons, crops or regions are untested and may do worse. Only 8-bit images are accepted.
- **Overfitting.** The best epoch was 4 of 20: the model fits the training fields faster than it learns patterns that carry over to new fields.
- **Labels.** Hand-drawn polygons with soft edges limit the IoU any model can reach on this data.

## How to reproduce
```
# train (ml/.venv, MLflow running on localhost:5000)
python ml/scripts/train_full.py --model unet-resnet34 --epochs 20 --lr 3e-4 --weight-decay 0.01 --encoder-lr-scale 0.1 --colour-jitter 0.2 --class-weight-power 0.25
# export and check against PyTorch
python ml/scripts/export_onnx.py --run-id 6d9eee7ff25843c68a0d40682deaadd1 --version 3
```
All other settings are the script defaults; the exact values are logged as parameters of the MLflow run.
