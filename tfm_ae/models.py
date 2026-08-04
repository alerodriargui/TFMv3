"""Convolutional autoencoder for full-resolution radiographs."""

from __future__ import annotations

import torch
from torch import nn

ARCHITECTURE_VERSION = "ae_1024_v2"


class ConvAutoencoder(nn.Module):
    """Autoencoder for 1024 × 1024 grayscale radiographs.

    ``latent_vector`` returns a fixed-size descriptor (mean and max pooling of
    the bottleneck) used later to score anomalies by distance to the normal
    training distribution.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 3, 2, 1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def latent_vector(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        mean_pool = features.mean(dim=(2, 3))
        max_pool = features.amax(dim=(2, 3))
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))
