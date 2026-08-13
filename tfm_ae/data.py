"""Deterministic, dependency-light loading for BraTS2021."""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import PROJECT_ROOT

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NORMAL_NAMES = ("good", "normal")
ANOMALY_NAMES = ("Ungood", "ungood", "abnormal", "anomalous")
NON_IMAGE_DIRS = ("label", "labels", "anomaly_mask", "mask", "masks")


def resolve_data_root(explicit: Path | None = None) -> Path:
    """Find BraTS2021 from an argument, env variable or known path."""
    candidates = [explicit] if explicit else []
    if value := os.environ.get("TFM_DATA_ROOT"):
        candidates.append(Path(value))
    candidates.extend(
        (
            PROJECT_ROOT / "data/raw/rsna_bmad/BraTS2021_slice",
            PROJECT_ROOT.parent / "TFMv2/data/raw/rsna_bmad/BraTS2021_slice",
        )
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "train" / "good").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("No se encontró BraTS2021. Usa --data-root.")


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
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and all(part not in NON_IMAGE_DIRS for part in path.parts)
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


class RandomFlipRotate:
    """Random horizontal/vertical flip + 90-degree rotation for augmentation."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        angle = self.rng.choice([0, 90, 180, 270])
        if angle:
            image = image.rotate(angle)
        return image


class RadiographDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Load grayscale images in [0, 1], retaining label and source path."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        image_size: int = 64,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        if len(paths) != len(labels) or not paths:
            raise ValueError("paths y labels deben tener la misma longitud no vacía")
        self.paths = paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform

    @classmethod
    def normal_only(
        cls,
        split_root: Path,
        image_size: int = 64,
        limit: int | None = None,
        seed: int = 42,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> "RadiographDataset":
        paths = deterministic_subset(
            find_images(find_class_dir(split_root, NORMAL_NAMES)), limit, seed
        )
        return cls(paths, [0] * len(paths), image_size, transform=transform)

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
            if self.transform is not None:
                image = self.transform(image)
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(pixels).unsqueeze(0), self.labels[index], str(path)