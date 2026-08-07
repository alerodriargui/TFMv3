"""Global image statistics used as anomaly-detection signals."""

from __future__ import annotations

import numpy as np


def connected_bright(img: np.ndarray, level: float) -> tuple[float, float]:
    """Return (largest, second-largest) connected bright-region sizes (fraction)."""
    mask = img > level
    h, w = mask.shape
    if not mask.any():
        return 0.0, 0.0
    visited = np.zeros_like(mask)
    sizes = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                size = 0
                while stack:
                    ci, cj = stack.pop()
                    size += 1
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = ci + di, cj + dj
                        if (
                            0 <= ni < h
                            and 0 <= nj < w
                            and mask[ni, nj]
                            and not visited[ni, nj]
                        ):
                            visited[ni, nj] = True
                            stack.append((ni, nj))
                sizes.append(size)
    sizes.sort(reverse=True)
    largest = sizes[0]
    second = sizes[1] if len(sizes) > 1 else 0.0
    return largest / (h * w), second / (h * w)


def image_features(img: np.ndarray) -> dict[str, float]:
    """Statistics of a single grayscale image (values in [0, 1])."""
    gy, gx = np.gradient(img)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    bins = np.histogram(img, bins=32, range=(0, 1), density=True)[0]
    bins = bins[bins > 0]
    h, w = img.shape
    left, right = img[:, : w // 2], img[:, w // 2 :]
    grad_left, grad_right = grad[:, : w // 2], grad[:, w // 2 :]
    bright_left = (img[:, : w // 2] > np.percentile(img, 90)).astype(float)
    bright_right = (img[:, w // 2 :] > np.percentile(img, 90)).astype(float)
    mean = float(img.mean())
    std = float(img.std())
    level = max(float(np.percentile(img, 90)), 0.9)
    mask = img > level
    largest, second = connected_bright(img, level)
    return {
        "mean": mean,
        "std": std,
        "skew": float(((img - mean) ** 3).mean() / (std + 1e-8) ** 3),
        "kurt": float(((img - mean) ** 4).mean() / (std + 1e-8) ** 4),
        "asym_mean": float(abs(left.mean() - right.mean())),
        "asym_std": float(abs(left.std() - right.std())),
        "asym_grad": float(abs(grad_left.mean() - grad_right.mean())),
        "asym_hi": float(abs(bright_left.mean() - bright_right.mean())),
        "flip_mae": float(np.mean(np.abs(img - np.fliplr(img)))),
        "grad_mean": float(grad.mean()),
        "grad_max": float(grad.max()),
        "entropy": float(-(bins * np.log(bins)).sum()),
        "max": float(img.max()),
        "p95": float(np.percentile(img, 95)),
        "frac98": float(np.mean(img > level)),
        "bright_asym": float(abs(mask[:, : w // 2].mean() - mask[:, w // 2 :].mean())),
        "cc_largest": largest,
        "cc_second": second,
    }


GLOBAL_SIGNALS = ("kurt", "cc_largest", "grad_mean", "entropy")


def feature_matrix(images: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the feature bank for a batch of images (N, H, W)."""
    keys = list(image_features(images[0]).keys())
    out = {k: np.empty(len(images), dtype=float) for k in keys}
    for index, img in enumerate(images):
        for k, v in image_features(img).items():
            out[k][index] = v
    return out
