import torch

from cropwatch_ml.models.unet import UNet

MODELS = ("unet", "unet-resnet34")


def build_model(
    name: str, num_classes: int, base_channels: int = 32, pretrained: bool = True
) -> torch.nn.Module:
    """One place that turns a model name into a network, for training and for export.

    unet           our own U-Net, trained from scratch.
    unet-resnet34  U-Net whose encoder is a ResNet34 pretrained on ImageNet (via
                   segmentation_models_pytorch). pretrained=False builds the same shape with
                   random weights, which is what export does before loading a checkpoint.
    """
    if name == "unet":
        return UNet(in_channels=3, num_classes=num_classes, base=base_channels)
    if name == "unet-resnet34":
        import segmentation_models_pytorch as smp  # only needed for this model

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"unknown model {name!r}, choose from {MODELS}")


__all__ = ["MODELS", "UNet", "build_model"]
