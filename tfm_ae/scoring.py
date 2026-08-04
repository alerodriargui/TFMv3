"""Reconstruction error shared by training, evaluation and the demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ConvAutoencoder

SCORE_NAME = "reconstruction_error_q99"
DEFAULT_ERROR_QUANTILE = 0.99


@dataclass(frozen=True)
class ReconstructionScore:
    reconstructed: torch.Tensor
    absolute_error: torch.Tensor
    reconstruction_mae: torch.Tensor
    anomaly_score: torch.Tensor


def reconstruction_score(
    model: ConvAutoencoder,
    images: torch.Tensor,
    error_quantile: float = DEFAULT_ERROR_QUANTILE,
) -> ReconstructionScore:
    """Reconstruction MAE plus a localized percentile score in one pass.

    The MAE over a whole 1024x1024 image dilutes small anomalies across one
    million pixels. ``anomaly_score`` uses a high percentile of the per-pixel
    error so a localized lesion raises the score even if it covers a tiny
    fraction of the image. Both remain pure reconstruction error.
    """
    if not 0.0 < error_quantile < 1.0:
        raise ValueError("error_quantile debe estar entre 0 y 1")

    reconstructed = model(images)
    absolute_error = torch.abs(reconstructed - images)
    reconstruction_mae = absolute_error.mean(dim=(1, 2, 3))
    anomaly_score = absolute_error.flatten(1).float().quantile(
        error_quantile, dim=1
    )
    return ReconstructionScore(
        reconstructed=reconstructed,
        absolute_error=absolute_error,
        reconstruction_mae=reconstruction_mae,
        anomaly_score=anomaly_score,
    )
