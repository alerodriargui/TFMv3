"""Anomaly scores shared by training and the single-image demo."""

from __future__ import annotations

import torch

from .models import ConvAutoencoder


def reconstruction_scores(
    model: ConvAutoencoder, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image reconstruction MAE and reconstructed images."""
    reconstructed = model(images)
    scores = torch.mean(torch.abs(reconstructed - images), dim=(1, 2, 3))
    return scores, reconstructed


def center_border_difference(images: torch.Tensor) -> torch.Tensor:
    """Compare the central half with the outer eighth, at any image size."""
    pixels = images.squeeze(1)
    height, width = pixels.shape[-2:]
    if height < 8 or width < 8:
        raise ValueError("Las imágenes deben medir al menos 8 × 8 píxeles")

    center_y, center_x = height // 4, width // 4
    center = pixels[
        :, center_y : height - center_y, center_x : width - center_x
    ].mean(dim=(1, 2))

    margin_y, margin_x = height // 8, width // 8
    inner = pixels[
        :, margin_y : height - margin_y, margin_x : width - margin_x
    ]
    border_sum = pixels.sum(dim=(1, 2)) - inner.sum(dim=(1, 2))
    border_pixels = height * width - inner.shape[-2] * inner.shape[-1]
    return center - border_sum / border_pixels
