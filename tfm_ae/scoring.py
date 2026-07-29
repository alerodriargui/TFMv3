"""Anomaly scores shared by training, evaluation and the demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .models import ConvAutoencoder


@dataclass(frozen=True)
class AnomalyComponents:
    reconstructed: torch.Tensor
    absolute_error: torch.Tensor
    reconstruction_mae: torch.Tensor
    center_border: torch.Tensor


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


def anomaly_components(
    model: ConvAutoencoder, images: torch.Tensor
) -> AnomalyComponents:
    """Calculate every raw anomaly component in one model pass."""
    reconstructed = model(images)
    absolute_error = torch.abs(reconstructed - images)
    return AnomalyComponents(
        reconstructed=reconstructed,
        absolute_error=absolute_error,
        reconstruction_mae=absolute_error.mean(dim=(1, 2, 3)),
        center_border=center_border_difference(images),
    )


def apply_calibration(
    reconstruction_mae: np.ndarray | float,
    center_border: np.ndarray | float,
    calibration: Mapping[str, float | str],
) -> np.ndarray | float:
    """Apply the frozen calibration to one score or an array of scores."""
    reconstruction = np.asarray(reconstruction_mae, dtype=np.float64)
    center = np.asarray(center_border, dtype=np.float64)
    calibrated_reconstruction = (
        reconstruction * float(calibration["ae_sign"])
        - float(calibration["ae_location"])
    ) / float(calibration["ae_scale"])
    calibrated_center = (
        center * float(calibration["center_sign"])
        - float(calibration["center_location"])
    ) / float(calibration["center_scale"])
    combined = 0.5 * calibrated_reconstruction + 0.5 * calibrated_center
    return float(combined) if combined.ndim == 0 else combined
