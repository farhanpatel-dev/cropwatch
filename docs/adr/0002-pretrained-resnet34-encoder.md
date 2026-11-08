# ADR 0002: Use a U-Net with a pretrained ResNet34 encoder, regularised against overfitting

- Status: Accepted
- Date: 2026-11-08
- Owner: Farhan (ML)

## Context
The first full-dataset model (v1) was our own U-Net trained from scratch (7.8 M parameters). We wanted to know whether a bigger encoder pretrained on ImageNet would do better on unseen fields.

All results below are on the full Agriculture-Vision 2021 validation set (18,334 images from fields not used in training). mIoU is shown without / with background; the all-background baseline is 0.000 / 0.181.

| | v1 own U-Net | v2 ResNet34 | v3 ResNet34 + regularisation |
| --- | --- | --- | --- |
| MLflow run | e9ed5ab1 | 0bf70274 | 6d9eee7f |
| Parameters | 7.8 M | 24.4 M | 24.4 M |
| mIoU | 0.327 / 0.404 | 0.336 / 0.421 | **0.371 / 0.440** |
| nutrient_deficiency / drydown / weed_cluster IoU | 0.251 / 0.519 / 0.212 | 0.247 / 0.539 / 0.222 | **0.315 / 0.558 / 0.241** |
| Background predicted as stress | 27.9% | **23.0%** | 29.5% |
| Best epoch / epochs run | 27 / 32 | 10 / 32 | 4 / 20 |
| Training time | 5.9 h | 4.5 h | 3.1 h |
| Checkpoint size | 31 MB | 98 MB | 98 MB |

What the runs showed:
- v2 alone (bigger pretrained encoder) gained only +0.009 mIoU over v1. Validation mIoU moves by about +-0.03 between epochs, so this gain is within noise.
- v2 peaked at epoch 10 while training loss kept falling: it fitted the training fields rather than patterns that carry over to new fields.
- v3 kept the ResNet34 encoder and added regularisation: colour jitter 0.2, weight decay 0.01, and a 10x smaller learning rate for the pretrained encoder. It improved every stress class.

## Decision
Ship v3: U-Net with an ImageNet-pretrained ResNet34 encoder (segmentation_models_pytorch 0.6.0), trained with colour jitter 0.2, weight decay 0.01, encoder learning rate x0.1 and class weight power 0.25. Stop model tuning for the MVP.

## Alternatives considered
- **Keep v1 (own U-Net).** Smaller (31 MB) and nearly as good as v2, but clearly behind v3 on all stress classes.
- **Keep v2.** Fewest false alarms (23.0%), but misses more stress than v3. For a tool meant to point people at problem areas, missing stress is the worse error.
- **Larger models (DeepLabV3+, SegFormer) or more epochs.** Three different set-ups all levelled off at 0.32-0.37, so the limit appears to be the data (RGB only, label quality, field-to-field differences), not model size. Not worth the hours before the demo.

## Consequences
- Good: best stress detection so far (mIoU without background 0.371).
- Cost: more false alarms on healthy ground than v2 (29.5% vs 23.0% of background pixels).
- Cost: 3x larger model file (98 MB) and slower CPU inference than v1.
- Open: each version was trained once (seed 0), so the size of seed-to-seed variation is unknown.
- Next improvement after the demo: add the NIR channel (4 inputs). This changes `input_channels` and the inference contract, so it needs its own ADR.
