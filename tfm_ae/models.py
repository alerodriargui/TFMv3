"""Convolutional autoencoder used for anomaly detection."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Small autoencoder for grayscale radiographs (size multiple of 8)."""

    def __init__(self, bottleneck_channels: int = 32) -> None:
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, bottleneck_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


def per_image_scores(
    model: ConvAutoencoder, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reconstruction MAE and reconstructed images."""
    reconstructed = model(images)
    scores = torch.mean(torch.abs(reconstructed - images), dim=(1, 2, 3))
    return scores, reconstructed


def build_model(bottleneck_channels: int = 32) -> ConvAutoencoder:
    """Build the single autoencoder used for anomaly detection."""
    return ConvAutoencoder(bottleneck_channels=bottleneck_channels)
