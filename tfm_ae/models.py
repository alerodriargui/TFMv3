"""Convolutional autoencoder used for anomaly detection."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoencoder(nn.Module):
    """Small autoencoder for grayscale radiographs (size multiple of 8)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
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


def _gaussian_window(
    window_size: int, sigma: float, reference: torch.Tensor
) -> torch.Tensor:
    """Return a normalized 2D Gaussian kernel as a conv2d filter."""
    coords = (
        torch.arange(window_size, dtype=reference.dtype, device=reference.device)
        - window_size // 2
    )
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)
    return kernel.view(1, 1, window_size, window_size)


def ssim_map(
    images: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01,
    c2: float = 0.03,
) -> torch.Tensor:
    """Per-pixel structural similarity map (1 = identical, 0 = unrelated)."""
    c1 = (c1 * 1.0) ** 2
    c2 = (c2 * 1.0) ** 2
    window = _gaussian_window(window_size, sigma, images)
    pad = window_size // 2
    images_padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    reconstructed_padded = F.pad(reconstructed, (pad, pad, pad, pad), mode="replicate")
    mu_images = F.conv2d(images_padded, window)
    mu_reconstructed = F.conv2d(reconstructed_padded, window)
    mu_images_sq = mu_images**2
    mu_reconstructed_sq = mu_reconstructed**2
    mu_products = mu_images * mu_reconstructed
    sigma_images_sq = F.conv2d(images_padded**2, window) - mu_images_sq
    sigma_reconstructed_sq = (
        F.conv2d(reconstructed_padded**2, window) - mu_reconstructed_sq
    )
    sigma_products = (
        F.conv2d(images_padded * reconstructed_padded, window) - mu_products
    )
    numerator = (2 * mu_products + c1) * (2 * sigma_products + c2)
    denominator = (mu_images_sq + mu_reconstructed_sq + c1) * (
        sigma_images_sq + sigma_reconstructed_sq + c2
    )
    return numerator / (denominator + 1e-8)


def per_image_signals(
    model: nn.Module, images: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return raw anomaly signals and the reconstructed images.

    Higher is more anomalous. ``mae`` is the mean absolute reconstruction
    error; ``ssim`` is ``1 - SSIM`` averaged over pixels.
    """
    reconstructed = model(images)
    mae = torch.mean(torch.abs(reconstructed - images), dim=(1, 2, 3))
    ssim = torch.mean(ssim_map(images, reconstructed), dim=(1, 2, 3))
    return {"mae": mae, "ssim": 1.0 - ssim}, reconstructed


class UNetAutoencoder(nn.Module):
    """U-Net autoencoder with skip connections for sharper reconstructions.

    Requires image sizes that are multiples of 16 (four stride-2 stages).
    """

    def __init__(self) -> None:
        super().__init__()
        self.e1 = self._down(1, 16)
        self.e2 = self._down(16, 32)
        self.e3 = self._down(32, 64)
        self.e4 = self._down(64, 128)

        self.d1_up = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.d1_conv = self._conv1(128, 64)
        self.d2_up = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.d2_conv = self._conv1(64, 32)
        self.d3_up = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.d3_conv = self._conv1(32, 16)
        self.d4_up = nn.ConvTranspose2d(16, 1, 4, 2, 1)

    @staticmethod
    def _down(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _conv1(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def encode(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        e1 = self.e1(images)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        latent = self.e4(e3)
        return latent, (e1, e2, e3)

    def decode(
        self,
        latent: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        e1, e2, e3 = skips
        d = torch.relu(self.d1_up(latent))
        d = self.d1_conv(torch.cat([d, e3], dim=1))
        d = torch.relu(self.d2_up(d))
        d = self.d2_conv(torch.cat([d, e2], dim=1))
        d = torch.relu(self.d3_up(d))
        d = self.d3_conv(torch.cat([d, e1], dim=1))
        return torch.sigmoid(self.d4_up(d))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(images))


def build_model(model_name: str) -> nn.Module:
    """Factory that returns the autoencoder named ``model_name``."""
    if model_name == "ae":
        return ConvAutoencoder()
    if model_name == "unet":
        return UNetAutoencoder()
    raise ValueError(f"Modelo desconocido: {model_name}")
