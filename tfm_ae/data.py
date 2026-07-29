"""Carga y preparación de las radiografías del dataset Chest-RSNA."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import PROJECT_ROOT


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NORMAL_FOLDERS = ("good", "normal")
ANOMALY_FOLDERS = ("Ungood", "ungood", "abnormal", "anomalous")
DEFAULT_IMAGE_SIZE = 1024


def load_radiograph(path: Path, image_size: int) -> torch.Tensor:
    """Carga una radiografía y la devuelve como tensor entre 0 y 1."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")

    with Image.open(path) as image:
        image = image.convert("L")

        if image.size != (image_size, image_size):
            image = image.resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )

        pixels = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(pixels)
    return tensor.unsqueeze(0)


def resolve_data_root(explicit_path: Path = None) -> Path:
    """Busca la carpeta principal del dataset."""
    candidates = []

    if explicit_path is not None:
        candidates.append(explicit_path)

    environment_path = os.environ.get("TFM_DATA_ROOT")
    if environment_path:
        candidates.append(Path(environment_path))

    candidates.append(PROJECT_ROOT / "data/raw/rsna_bmad/Chest-RSNA")
    candidates.append(
        PROJECT_ROOT.parent / "TFMv2/data/raw/rsna_bmad/Chest-RSNA"
    )

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        train_folder = root / "train" / "good"
        test_folder = root / "test"

        if train_folder.is_dir() and test_folder.is_dir():
            return root

    raise FileNotFoundError(
        "No se encontró Chest-RSNA. Indica su ruta con --data-root."
    )


def split_dir(root: Path, split: str) -> Path:
    """Devuelve la carpeta de entrenamiento, validación o test."""
    if split == "train":
        return root / "train"

    if split == "test":
        return root / "test"

    if split == "val":
        validation_names = ("val", "valid", "validation")

        for name in validation_names:
            candidate = root / name
            if candidate.is_dir():
                return candidate

        raise FileNotFoundError(
            f"No se encontró la carpeta de validación dentro de {root}"
        )

    raise ValueError(f"Split desconocido: {split}")


def find_images(root: Path) -> list[Path]:
    """Busca todas las imágenes dentro de una carpeta."""
    images = []

    for path in root.rglob("*"):
        is_valid_image = (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if is_valid_image:
            images.append(path)

    return sorted(images)


def find_class_dir(split_root: Path, possible_names) -> Path:
    """Busca la carpeta correspondiente a una clase."""
    for name in possible_names:
        candidate = split_root / name

        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"No se encontró ninguna carpeta {possible_names} en {split_root}"
    )


def deterministic_subset(
    paths: list[Path],
    limit: int,
    seed: int,
) -> list[Path]:
    """Selecciona siempre el mismo subconjunto para una semilla."""
    if limit is None or limit >= len(paths):
        return paths

    shuffled_paths = paths.copy()
    generator = random.Random(seed)
    generator.shuffle(shuffled_paths)

    selected_paths = shuffled_paths[:limit]
    return sorted(selected_paths)


class RadiographDataset(Dataset):
    """Dataset de PyTorch para cargar radiografías bajo demanda."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        if not paths:
            raise ValueError("El dataset no contiene imágenes")

        if len(paths) != len(labels):
            raise ValueError(
                "El número de imágenes y etiquetas debe ser el mismo"
            )

        self.paths = paths
        self.labels = labels
        self.image_size = image_size

    @classmethod
    def normal_only(
        cls,
        split_root: Path,
        image_size: int = DEFAULT_IMAGE_SIZE,
        limit: int = None,
        seed: int = 42,
    ):
        """Crea un dataset formado únicamente por imágenes normales."""
        normal_folder = find_class_dir(split_root, NORMAL_FOLDERS)
        normal_paths = find_images(normal_folder)
        normal_paths = deterministic_subset(normal_paths, limit, seed)
        normal_labels = [0] * len(normal_paths)

        return cls(normal_paths, normal_labels, image_size)

    @classmethod
    def labeled(
        cls,
        split_root: Path,
        image_size: int = DEFAULT_IMAGE_SIZE,
        limit_per_class: int = None,
        seed: int = 42,
    ):
        """Crea un dataset con imágenes normales y anómalas."""
        normal_folder = find_class_dir(split_root, NORMAL_FOLDERS)
        anomaly_folder = find_class_dir(split_root, ANOMALY_FOLDERS)

        normal_paths = find_images(normal_folder)
        anomaly_paths = find_images(anomaly_folder)

        normal_paths = deterministic_subset(
            normal_paths,
            limit_per_class,
            seed,
        )
        anomaly_paths = deterministic_subset(
            anomaly_paths,
            limit_per_class,
            seed + 1,
        )

        paths = normal_paths + anomaly_paths
        labels = [0] * len(normal_paths) + [1] * len(anomaly_paths)

        return cls(paths, labels, image_size)

    def __len__(self):
        """Devuelve el número de imágenes."""
        return len(self.paths)

    def __getitem__(self, index):
        """Carga una imagen y devuelve imagen, etiqueta y ruta."""
        path = self.paths[index]
        image = load_radiograph(path, self.image_size)
        label = self.labels[index]

        return image, label, str(path)
