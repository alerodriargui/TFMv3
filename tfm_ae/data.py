"""Carga determinista y con pocas dependencias de BraTS2021."""

from __future__ import annotations

import random  # RNG para submuestreo determinista y aumentación
from pathlib import Path  # Rutas de archivos multiplataforma
from typing import Callable  # Tipado de callables

import numpy as np  # Conversión de píxeles a arrays
import torch  # Tensores de salida del Dataset
from PIL import Image  # Apertura/redimensionado de imágenes
from torch.utils.data import Dataset  # Clase base de PyTorch para datasets

from . import PROJECT_ROOT  # Raíz del proyecto definida en __init__.py

NORMAL_DIR = "good"
ANOMALY_DIR = "Ungood"


def resolve_data_root(explicit: Path | None = None) -> Path:
    """Devuelve la raíz de BraTS2021: la ruta explícita si se pasa, si no la única del repositorio."""
    if explicit is not None:
        return explicit
    return PROJECT_ROOT / "data/raw/rsna_bmad/BraTS2021_slice"


def split_dir(root: Path, split: str) -> Path:
    """Devuelve el directorio del split pedido (en los datos la validación se llama 'valid')."""
    return root / ("valid" if split == "val" else split)


def find_images(root: Path) -> list[Path]:
    """Lista (ordenada) de imágenes PNG bajo 'root', excluyendo la carpeta de máscaras 'label'."""
    return sorted(path for path in root.rglob("*.png") if "label" not in path.parts)


def deterministic_subset(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    """Subconjunto de tamaño 'limit' reproducible (misma semilla -> mismo resultado), ordenado."""
    if limit is None or limit >= len(paths):
        return paths
    return sorted(random.Random(seed).sample(paths, limit))


class RandomFlipRotate:
    """Aumentación: volteo horizontal/vertical aleatorio + rotación de 90 grados."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)  # RNG propio para que la aumentación sea reproducible

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.rng.random() < 0.5:  # Volteo horizontal (50%)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:  # Volteo vertical (50%)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image.rotate(self.rng.choice((0, 90, 180, 270)))  # Rotación de 90 en 90


class RadiographDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Carga imágenes en escala de grises normalizadas a [0, 1], guardando etiqueta y ruta."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        image_size: int = 64,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        if len(paths) != len(labels) or not paths:
            raise ValueError("paths y labels deben tener la misma longitud no vacía")
        self.paths = paths  # Rutas de las imágenes
        self.labels = labels  # Etiqueta por imagen (0 = normal, 1 = anómalo)
        self.image_size = image_size  # Tamaño al que se redimensionan las imágenes
        self.transform = transform  # Aumentación opcional (p.ej. RandomFlipRotate)

    @classmethod
    def _from_class(
        cls,
        class_dir: Path,
        label: int,
        image_size: int,
        limit: int | None,
        seed: int,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> "RadiographDataset":
        paths = deterministic_subset(find_images(class_dir), limit, seed)
        return cls(paths, [label] * len(paths), image_size, transform=transform)

    @classmethod
    def normal_only(
        cls,
        split_root: Path,
        image_size: int = 64,
        limit: int | None = None,
        seed: int = 42,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> "RadiographDataset":
        """Crea un dataset solo con imágenes normales (etiqueta 0). Útil para entrenar AE."""
        return cls._from_class(
            split_root / NORMAL_DIR, 0, image_size, limit, seed, transform
        )

    @classmethod
    def labeled(
        cls,
        split_root: Path,
        image_size: int = 64,
        limit_per_class: int | None = None,
        seed: int = 42,
    ) -> "RadiographDataset":
        """Crea un dataset balanceado con normales (0) y anómalos (1) para evaluar."""
        normal = cls._from_class(
            split_root / NORMAL_DIR, 0, image_size, limit_per_class, seed
        )
        anomalous = cls._from_class(
            split_root / ANOMALY_DIR, 1, image_size, limit_per_class, seed + 1
        )
        return cls(
            normal.paths + anomalous.paths,
            normal.labels + anomalous.labels,
            image_size,
        )

    def __len__(self) -> int:
        return len(self.paths)  # Número de muestras del dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Devuelve (imagen [1, H, W] en [0,1], etiqueta, ruta) para el índice dado."""
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L").resize(  # Escala de grises y redimensiona
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            if self.transform is not None:  # Aumentación opcional
                image = self.transform(image)
            pixels = np.asarray(image, dtype=np.float32) / 255.0  # Píxeles a [0, 1]
        return torch.from_numpy(pixels).unsqueeze(0), self.labels[index], str(path)
