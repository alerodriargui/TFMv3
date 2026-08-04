"""Losses for autoencoder training: L1 plus structural similarity."""

from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_SSIM_WEIGHT = 0.5


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    coordinates = (
        torch.arange(window_size, dtype=torch.float32, device=device)
        - window_size // 2
    )
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return (
        kernel_2d.view(1, 1, window_size, window_size)
        .expand(channels, 1, window_size, window_size)
        .contiguous()
    )


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean structural similarity loss ``1 - SSIM`` over the batch."""
    channels = prediction.shape[1]
    kernel = _gaussian_window(window_size, sigma, channels, prediction.device)
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = (
        F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels)
        - mu1_sq
    ).clamp(min=0)
    sigma2_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels)
        - mu2_sq
    ).clamp(min=0)
    sigma12 = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1 - ssim_map.mean()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = DEFAULT_SSIM_WEIGHT,
) -> torch.Tensor:
    """Training objective: L1 plus SSIM to keep reconstructions sharp."""
    return F.l1_loss(prediction, target) + ssim_weight * ssim_loss(
        prediction, target
    )
