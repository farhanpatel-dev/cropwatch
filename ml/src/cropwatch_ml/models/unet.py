"""A small U-Net, written out by hand to learn the idea.

Encoder: each level halves the image size and doubles the channels (learns "what").
Decoder: each level doubles the size back and joins the encoder output of the same size
through a skip connection (brings back "where", so edges stay sharp).
Output: one score per class per pixel (logits), shape N x num_classes x H x W.
"""

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch norm and ReLU. Keeps H and W."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]  # e.g. 32, 64, 128, 256, 512
        self.depth = depth
        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(DoubleConv(chs[i], chs[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.up_convs = nn.ModuleList(
            DoubleConv(chs[i] * 2, chs[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2**self.depth or x.shape[-2] % 2**self.depth:
            raise ValueError(f"input H and W must be multiples of {2**self.depth}")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        x = skips.pop()
        for up, conv in zip(self.ups, self.up_convs, strict=True):
            x = conv(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)
