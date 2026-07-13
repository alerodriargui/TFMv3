"""Deterministic, dependency-light loading for Chest-RSNA."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NORMAL_NAMES = ("good", "normal")
ANOMALY_NAMES = ("Ungood", "ungood", "abnormal", "anomalous")


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dir(split_root: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = split_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No se encontró {tuple(names)} bajo {split_root}")


def deterministic_subset(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    if limit is None or limit >= len(paths):
        return paths
    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    return sorted(paths[index] for index in indices[:limit])


class RadiographDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Load grayscale images in [0, 1], retaining label and source path."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        image_size: int = 64,
    ) -> None:
        if len(paths) != len(labels) or not paths:
            raise ValueError("paths y labels deben tener la misma longitud no vacía")
        self.paths = paths
        self.labels = labels
        self.image_size = image_size

    @classmethod
    def normal_only(
        cls,
        split_root: Path,
        image_size: int = 64,
        limit: int | None = None,
        seed: int = 42,
    ) -> "RadiographDataset":
        paths = deterministic_subset(
            find_images(find_class_dir(split_root, NORMAL_NAMES)), limit, seed
        )
        return cls(paths, [0] * len(paths), image_size)

    @classmethod
    def labeled(
        cls,
        split_root: Path,
        image_size: int = 64,
        limit_per_class: int | None = None,
        seed: int = 42,
    ) -> "RadiographDataset":
        normal = deterministic_subset(
            find_images(find_class_dir(split_root, NORMAL_NAMES)),
            limit_per_class,
            seed,
        )
        anomalous = deterministic_subset(
            find_images(find_class_dir(split_root, ANOMALY_NAMES)),
            limit_per_class,
            seed + 1,
        )
        return cls(normal + anomalous, [0] * len(normal) + [1] * len(anomalous), image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(pixels).unsqueeze(0), self.labels[index], str(path)


class CachedRadiographDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Serve preprocessed uint8 tensors while preserving source paths."""

    def __init__(
        self,
        tensors: torch.Tensor,
        labels: list[int],
        paths: list[str],
        indices: list[int] | None = None,
    ) -> None:
        if tensors.dtype != torch.uint8 or tensors.ndim != 4:
            raise ValueError("La caché debe tener forma NCHW y dtype uint8")
        if len(tensors) != len(labels) or len(labels) != len(paths):
            raise ValueError("Tensores, etiquetas y rutas no coinciden")
        self.tensors = tensors
        self.labels = labels
        self.paths = paths
        self.indices = indices if indices is not None else list(range(len(labels)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        source = self.indices[index]
        return self.tensors[source].float().div(255), self.labels[source], self.paths[source]


def cache_file(cache_root: Path, split: str, label: int, image_size: int) -> Path:
    class_name = "normal" if label == 0 else "anomalous"
    return cache_root / f"{split}_{class_name}_{image_size}.pt"


def _load_cache(path: Path, label: int) -> tuple[torch.Tensor, list[int], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors = payload["images"]
    paths = payload["paths"]
    return tensors, [label] * len(tensors), paths


def _limited_indices(length: int, limit: int | None, seed: int) -> list[int]:
    indices = list(range(length))
    if limit is not None and limit < length:
        random.Random(seed).shuffle(indices)
        indices = sorted(indices[:limit])
    return indices


def cached_normal_only(
    cache_root: Path,
    split: str,
    image_size: int,
    limit: int | None,
    seed: int,
) -> CachedRadiographDataset:
    tensors, labels, paths = _load_cache(
        cache_file(cache_root, split, 0, image_size), 0
    )
    return CachedRadiographDataset(
        tensors, labels, paths, _limited_indices(len(tensors), limit, seed)
    )


def cached_labeled(
    cache_root: Path,
    split: str,
    image_size: int,
    limit_per_class: int | None,
    seed: int,
) -> CachedRadiographDataset:
    normal_tensors, normal_labels, normal_paths = _load_cache(
        cache_file(cache_root, split, 0, image_size), 0
    )
    anomaly_tensors, anomaly_labels, anomaly_paths = _load_cache(
        cache_file(cache_root, split, 1, image_size), 1
    )
    tensors = torch.cat((normal_tensors, anomaly_tensors), dim=0)
    labels = normal_labels + anomaly_labels
    paths = normal_paths + anomaly_paths
    normal_indices = _limited_indices(
        len(normal_tensors), limit_per_class, seed
    )
    anomaly_indices = [
        len(normal_tensors) + index
        for index in _limited_indices(
            len(anomaly_tensors), limit_per_class, seed + 1
        )
    ]
    return CachedRadiographDataset(
        tensors, labels, paths, normal_indices + anomaly_indices
    )


def cache_is_complete(cache_root: Path, image_size: int) -> bool:
    required = [
        cache_file(cache_root, "train", 0, image_size),
        cache_file(cache_root, "val", 0, image_size),
        cache_file(cache_root, "val", 1, image_size),
        cache_file(cache_root, "test", 0, image_size),
        cache_file(cache_root, "test", 1, image_size),
    ]
    return all(path.is_file() for path in required)
