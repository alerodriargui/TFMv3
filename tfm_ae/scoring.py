"""Reconstruction error shared by training, evaluation and the demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ConvAutoencoder

SCORE_NAME = "reconstruction_mae"


@dataclass(frozen=True)
class ReconstructionScore:
    reconstructed: torch.Tensor
    absolute_error: torch.Tensor
    reconstruction_mae: torch.Tensor


def reconstruction_score(
    model: ConvAutoencoder, images: torch.Tensor
) -> ReconstructionScore:
    """Calculate the per-image reconstruction MAE in one model pass."""
    reconstructed = model(images)
    absolute_error = torch.abs(reconstructed - images)
    return ReconstructionScore(
        reconstructed=reconstructed,
        absolute_error=absolute_error,
        reconstruction_mae=absolute_error.mean(dim=(1, 2, 3)),
    )
