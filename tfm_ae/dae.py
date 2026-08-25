"""Denoising Autoencoder for brain anomaly detection.

Based on Kascenas et al. (MIDL 2022):
  U-Net style AE with skip connections, no bottleneck.
  Train: add coarse noise → reconstruct clean image.
  Score: reconstruction error → anomaly map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DAE(nn.Module):
    """U-Net Denoising Autoencoder (no bottleneck)."""

    def __init__(self, in_channels: int = 1, base_ch: int = 64) -> None:
        super().__init__()
        c = base_ch

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 8)

        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 4)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 2)

        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c)

        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))
