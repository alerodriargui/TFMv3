"""Carga de datos BraTS2021."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import PROJECT_ROOT

NORMAL_DIR = "good"
ANOMALY_DIR = "Ungood"


# Localizacion del dataset
def resolve_data_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return PROJECT_ROOT / "data/raw/rsna_bmad/BraTS2021_slice"

# Dividir el directorio raíz en subdirectorios de entrenamiento, validación y prueba.
def split_dir(root: Path, split: str) -> Path:
    return root / ("valid" if split == "val" else split)


# Encontrar imagenes PNG en el directorio raíz, excluyendo cualquier ruta que contenga "label" en sus partes.
def find_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.png") if "label" not in path.parts)

# Selección determinista de un subconjunto de rutas dado un límite y una semilla.
def deterministic_subset(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    if limit is None or limit >= len(paths):
        return paths
    return sorted(random.Random(seed).sample(paths, limit))

# Clase de dataset para imágenes radiográficas en escala de grises normalizadas a [0, 1] con etiqueta y ruta.
class RadiographDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Imágenes en escala de grises normalizadas a [0, 1] con etiqueta y ruta."""

    # Inicializa el dataset con rutas de imágenes, etiquetas y tamaño de imagen.
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        image_size: int = 128,
    ) -> None:
        if len(paths) != len(labels) or not paths:
            raise ValueError("paths y labels deben tener la misma longitud no vacía")
        self.paths = paths
        self.labels = labels
        self.image_size = image_size

    # Clase de método para crear un dataset a partir de un subdirectorio de clase específico, asignando una etiqueta y limitando el número de imágenes si es necesario.
    @classmethod
    def _from_class(
        cls,
        class_dir: Path,
        label: int,
        image_size: int,
        limit: int | None,
        seed: int,
    ) -> "RadiographDataset":
        paths = deterministic_subset(find_images(class_dir), limit, seed)
        return cls(paths, [label] * len(paths), image_size)

    # Clase de método para crear un dataset que solo contiene imágenes normales.
    @classmethod
    def normal_only(
        cls,
        split_root: Path,
        image_size: int = 128,
        limit: int | None = None,
        seed: int = 42,
    ) -> "RadiographDataset":
        return cls._from_class(split_root / NORMAL_DIR, 0, image_size, limit, seed)

    # Clase de método para crear un dataset etiquetado combinando imágenes normales y anómalas. 
    @classmethod
    def labeled(
        cls,
        split_root: Path,
        image_size: int = 128,
        limit_per_class: int | None = None,
        seed: int = 42,
    ) -> "RadiographDataset":
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
        return len(self.paths)

    # Obtiene un elemento del dataset: carga la imagen, la convierte a escala de grises, la redimensiona, normaliza los píxeles y devuelve un tensor junto con su etiqueta y ruta.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).unsqueeze(0), self.labels[index], str(path)
