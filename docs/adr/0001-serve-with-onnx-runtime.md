# ADR 0001: Serve the model with ONNX Runtime on CPU, not PyTorch

- Status: Accepted
- Date: 2026-11-08
- Owner: Farhan (ML)

## Context
The model is trained with PyTorch on a local NVIDIA GPU. The inference service must run in Docker on the laptop, on kind (Kubernetes), and possibly on a free host, none of which has a GPU. The project has no budget, and services should stay small and simple (docs/architecture.md, "Serving format").

PyTorch with CUDA is several GB to install and is only needed for training. The service only needs a forward pass of a fixed network.

## Decision
Export the trained network to ONNX (`ml/scripts/export_onnx.py`, opset 18, dynamic batch size) and serve it with ONNX Runtime on CPU. PyTorch stays in `ml/` only and is never installed in a service image.

The export writes three files that always change together: `model.onnx`, `model_info.json` (name, version, MLflow run id, tile size, normalisation mean/std) and `classes.yaml`. Every export is checked against PyTorch on real validation tiles before it is used.

## Alternatives considered
- **PyTorch in the service (TorchScript or plain).** One less export step, but a much larger image and a second copy of the training dependencies to keep in sync.
- **TorchServe or NVIDIA Triton.** Built for GPU fleets, batching and many models; too heavy for one small model on CPU.
- **GPU inference.** Faster, but costs money and none of our targets has a GPU.

## Consequences
- Good: the image has no PyTorch (807 MB in total, including the 98 MB model, rasterio/GDAL and FastAPI). Runs on any CPU host.
- Good: the export is verified. For v3, the largest logit difference to PyTorch was 2.8e-5 and the predicted class matched on all 2,097,152 pixels of 32 validation tiles. Scoring the full service pipeline on 300 validation images gave the same mIoU as training (0.382 vs 0.371 on the full validation set).
- Cost: CPU is slow for big images: 0.8 s for 512 x 512, 40-41 s for 6000 x 6000 (16 logical CPUs). This forces the image size limit in ADR 0003.
- Cost: an extra export step after every training run, and the normalisation must be copied exactly (it is stored in `model_info.json` and read by the service, never hard-coded).
- Pins to watch: onnx 1.22.0 needs protobuf 5.x, because onnx 1.23 needs protobuf 6 and breaks mlflow 2.17.2.
