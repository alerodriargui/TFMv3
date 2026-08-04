"""Anomaly scores shared by training, evaluation and the demo.

The reconstruction part is a high percentile of the per-pixel absolute error
(``reconstruction_error_q99``). The latent part is the Mahalanobis distance of
the encoder descriptor against the normal training distribution. Both are pure
autoencoder signals: no label is ever used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .models import ConvAutoencoder

SCORE_NAME = "recon_q99_latent_maha"
DEFAULT_ERROR_QUANTILE = 0.99
DEFAULT_MAHALANOBIS_RIDGE = 1e-3


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
    """Reconstruction MAE plus a localized percentile score in one pass."""
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


@torch.no_grad()
def latent_vectors(
    model: ConvAutoencoder, images: torch.Tensor
) -> torch.Tensor:
    """Encoder descriptors ``[N, 2 * latent_channels]`` in float32."""
    return model.latent_vector(images).float()


def fit_normal_distribution(
    features: torch.Tensor, ridge: float = DEFAULT_MAHALANOBIS_RIDGE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularized Gaussian fit on normal features: ``(mean, inverse_cov)``."""
    if len(features) < 2:
        raise ValueError("Se necesitan al menos dos vectores para ajustar el gaussiano")
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.t() @ centered / max(len(features) - 1, 1)
    scale = torch.diag(covariance).mean().clamp(min=torch.finfo(covariance.dtype).eps)
    covariance = covariance + ridge * scale * torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    return mean, torch.linalg.inv(covariance)


def mahalanobis(
    features: torch.Tensor,
    normal_mean: torch.Tensor,
    normal_inv_cov: torch.Tensor,
) -> torch.Tensor:
    """Squared Mahalanobis distance of each feature to the normal centroid."""
    diff = features - normal_mean
    return torch.einsum("nd,de,ne->n", diff, normal_inv_cov, diff)


def standardize(
    values: torch.Tensor | float, location: float, scale: float
) -> torch.Tensor | float:
    return (values - location) / scale if scale else (values - location)


def combine_components(
    recon_scores: torch.Tensor | float,
    latent_scores: torch.Tensor | float,
    calibration: Mapping[str, float],
) -> torch.Tensor | float:
    """Average of the z-scaled reconstruction and latent components."""
    recon = standardize(recon_scores, calibration["recon_location"], calibration["recon_scale"])
    latent = standardize(latent_scores, calibration["latent_location"], calibration["latent_scale"])
    return 0.5 * recon + 0.5 * latent
