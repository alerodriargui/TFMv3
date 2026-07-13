"""Deterministic, dependency-light loading for Chest-RSNA."""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NORMAL_NAMES = ("good", "normal")
ANOMALY_NAMES = ("Ungood", "ungood", "abnormal", "anomalous")


def resolve_data_root(explicit: Path | None = None) -> Path:
    """Find Chest-RSNA from an argument, environment variable or known paths."""
    candidates = [explicit] if explicit else []
    if value := os.environ.get("TFM_DATA_ROOT"):
        candidates.append(Path(value))
    candidates.extend(
        (
            Path("data/raw/rsna_bmad/Chest-RSNA"),
            Path("../TFMv2/data/raw/rsna_bmad/Chest-RSNA"),
        )
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "train" / "good").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("No se encontró Chest-RSNA. Usa --data-root.")


def split_dir(root: Path, split: str) -> Path:
    if split != "val":
        return root / split
    for name in ("val", "valid", "validation"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No se encontró validación bajo {root}")


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
        return cls(
            normal + anomalous, [0] * len(normal) + [1] * len(anomalous), image_size
        )

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


def supervised_development_split(
    root: Path,
    image_size: int = 64,
    seed: int = 42,
    validation_fraction: float = 0.2,
    limit_per_class: int | None = None,
) -> tuple[RadiographDataset, RadiographDataset]:
    """Build a balanced supervised split while leaving the official test untouched.

    Anomalies come from the labelled BMAD validation split. The same number of
    normal images is sampled from the normal-only training split. Both classes
    are then divided deterministically into development train and validation.
    """
    normal = find_images(find_class_dir(split_dir(root, "train"), NORMAL_NAMES))
    anomalous = find_images(find_class_dir(split_dir(root, "val"), ANOMALY_NAMES))
    per_class = min(len(normal), len(anomalous))
    if limit_per_class is not None:
        per_class = min(per_class, limit_per_class)
    if per_class < 2:
        raise ValueError("Se necesitan al menos dos imágenes por clase")

    normal = deterministic_subset(normal, per_class, seed)
    anomalous = deterministic_subset(anomalous, per_class, seed + 1)
    normal_indices = list(range(per_class))
    anomalous_indices = list(range(per_class))
    random.Random(seed + 2).shuffle(normal_indices)
    random.Random(seed + 3).shuffle(anomalous_indices)
    validation_count = max(1, round(per_class * validation_fraction))
    validation_count = min(validation_count, per_class - 1)

    def select(paths: list[Path], indices: list[int], validation: bool) -> list[Path]:
        chosen = (
            indices[:validation_count] if validation else indices[validation_count:]
        )
        return [paths[index] for index in chosen]

    train_normal = select(normal, normal_indices, False)
    train_anomalous = select(anomalous, anomalous_indices, False)
    val_normal = select(normal, normal_indices, True)
    val_anomalous = select(anomalous, anomalous_indices, True)
    return (
        RadiographDataset(
            train_normal + train_anomalous,
            [0] * len(train_normal) + [1] * len(train_anomalous),
            image_size,
        ),
        RadiographDataset(
            val_normal + val_anomalous,
            [0] * len(val_normal) + [1] * len(val_anomalous),
            image_size,
        ),
    )
