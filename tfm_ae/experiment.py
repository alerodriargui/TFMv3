"""Entrenamiento y evaluacion del DAE para deteccion de anomalias."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from . import PROJECT_ROOT
from .data import RadiographDataset, split_dir


# Configuracion del experimento
@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 128
    learning_rate: float = 1e-4
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None
    dae_base_ch: int = 64
    noise_sigma: float = 0.2
    noise_resolution: int = 16

# Fijar semilla para reproducibilidad
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DataLoader
def _loader(dataset: RadiographDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

# Añade ruido gaussiano grueso a las radiografías
def _coarse_noise(images: torch.Tensor, sigma: float, resolution: int) -> torch.Tensor:
    # Obtiene las dimensiones del lote de imágenes
    b, c, h, w = images.shape
    # Genera ruido gaussiano de baja resolución
    noise_small = torch.randn(b, c, resolution, resolution, device=images.device)
    # Interpola el ruido a la resolución de la imagen
    noise = F.interpolate(noise_small, size=(h, w), mode="bilinear", align_corners=False)
    # Desplaza el ruido aleatoriamente para evitar patrones repetitivos
    noise = torch.roll(
        noise, shifts=(random.randrange(h), random.randrange(w)), dims=(-2, -1)
    )
    # Aplica el ruido solo a los píxeles de interés (no fondo negro)
    mask = (images > 0.01).float()
    # Devuelve la imagen con ruido
    return images + sigma * noise * mask

# Función de pérdida MSE
def _foreground_mse(reconstructed: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # La máscara evita los píxeles de fondo (negros)
    mask = (images > 0.01).float()
    # Calcula el error cuadrático en los píxeles de interés
    squared_error = (reconstructed - images).square() * mask
    # Devuelve el error cuadrático medio en los píxeles de interés
    return squared_error.sum() / mask.sum().clamp_min(1)

# Entrenamiento de un lote
def _train_batch_dae(
    model: torch.nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    sigma: float,
    noise_resolution: int,
) -> float:
    # Prepara el optimizador para el entrenamiento
    optimizer.zero_grad(set_to_none=True)
    # Aplica ruido a las imágenes
    noisy = _coarse_noise(images, sigma, noise_resolution)
    # Reconstruye las imágenes ruidosas
    reconstructed = model(noisy)
    # Calcula la pérdida
    loss = _foreground_mse(reconstructed, images)
    # Realiza el backward pass
    loss.backward()   
    optimizer.step()
    return float(loss.detach())

# Pérdida de validación
@torch.no_grad()
def _validation_loss_dae(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sigma: float,
    noise_resolution: int,
) -> float:
    model.eval()
    total = 0.0
    pixel_count = 0
    for images, _labels, _paths in loader:
        images = images.to(device)
        noisy = _coarse_noise(images, sigma, noise_resolution)
        reconstructed = model(noisy)
        mask = images > 0.01
        total += float(((reconstructed - images).square() * mask).sum())
        pixel_count += int(mask.sum())
    return total / max(pixel_count, 1)

# Entrenamiento completo del DAE
def train_dae(
    config: ExperimentConfig, device: torch.device
) -> tuple[torch.nn.Module, list[dict], int]:
    from .dae import DAE

    # Preparación de los conjuntos de entrenamiento y validación
    train_set = RadiographDataset.normal_only(
        split_dir(config.data_root, "train"),
        config.image_size,
        config.max_train_images,
        config.seed,
    )
    validation_set = RadiographDataset.normal_only(
        split_dir(config.data_root, "val"),
        config.image_size,
        None,
        config.seed,
    )
    
    train_loader = _loader(train_set, config.batch_size, True)
    validation_loader = _loader(validation_set, config.batch_size, False)

    model = DAE(in_channels=1, base_ch=config.dae_base_ch).to(device)

    trainable = list(model.parameters())
    optimizer = torch.optim.Adam(
        trainable, lr=config.learning_rate, amsgrad=True, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    best_value = float("inf")
    history: list[dict] = []
    best_state: dict | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for images, _labels, _paths in train_loader:
            images = images.to(device)
            loss = _train_batch_dae(
                model, images, optimizer, config.noise_sigma, config.noise_resolution,
            )
            scheduler.step()
            total_loss += loss * len(images)
            seen += len(images)

        validation_value = _validation_loss_dae(
            model, validation_loader, device, config.noise_sigma, config.noise_resolution,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_value": validation_value,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"DAE epoch={epoch}/{config.epochs} "
            f"train={record['train_loss']:.6f} val={validation_value:.6f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        if validation_value < best_value:
            best_value = validation_value
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "dae",
            "image_size": config.image_size,
            "dae_base_ch": config.dae_base_ch,
            "model_state": best_state["model"],
            "selected_epoch": best_state["epoch"],
        },
        config.output_dir / "model.pt",
    )
    return model, history, int(best_state["epoch"])

# Cálculo de puntuaciones de anomalía
@torch.no_grad()
def score_dataset_dae(
    model: torch.nn.Module,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        reconstructed = model(device_images)
        mask = F.avg_pool2d((device_images > 0.01).float(), 5, stride=1, padding=2)
        error = (reconstructed - device_images).abs() * (mask > 0.95)
        patches = F.pad(error, (2, 2, 2, 2), mode="reflect")
        patches = patches.unfold(2, 5, 1).unfold(3, 5, 1)
        filtered = patches.contiguous().view(*error.shape, 25).median(-1).values
        spatial = filtered.amax(dim=1)
        batch_scores = spatial.amax(dim=(1, 2))
        scores.append(batch_scores.cpu().numpy())
        labels.append(batch_labels.numpy())
        paths.extend(batch_paths)
        if samples is None:
            samples = (images[:8], reconstructed.cpu()[:8])
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"score progress={batch_index}/{len(loader)} "
                f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    assert samples is not None
    return np.concatenate(labels), np.concatenate(scores), paths, samples

# Guardado de las puntuaciones en un .csv
def _save_scores(
    path: Path, labels: np.ndarray, scores: np.ndarray, paths: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "score", "path"])
        writer.writerows(zip(labels.tolist(), scores.tolist(), paths))

# Genera un archivo de imagen con reconstrucciones de ejemplo
def _save_reconstructions(
    path: Path, originals: torch.Tensor, reconstructed: torch.Tensor,
) -> None:
    count = min(8, len(originals))
    size = originals.shape[-1]
    canvas = Image.new("L", (count * size, 2 * size + 20), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), "DAE: original (arriba) / reconstruccion (abajo)", fill=0)
    for index in range(count):
        orig = originals[index, 0]
        recon = reconstructed[index, 0] * (orig > 0.01).float()
        for row, tensor in enumerate((orig, recon)):
            array = (tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            canvas.paste(
                Image.fromarray(array, mode="L"), (index * size, 20 + row * size)
            )
    canvas.save(path)

# Selección del umbral (Youden J statistic) para maximizar la diferencia entre TPR y FPR
def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])

# Calcular métricas de evaluación (AUROC y Balanced Accuracy) dado un umbral
def _evaluate(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }

# Función principal para ejecutar el experimento
def run(config: ExperimentConfig) -> dict:
    if config.image_size % 8 != 0:
        raise ValueError(
            f"El DAE requiere --image-size multiplo de 8 (recibido {config.image_size})"
        )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model, history, selected_epoch = train_dae(config, device)

    validation = RadiographDataset.labeled(
        split_dir(config.data_root, "val"),
        config.image_size,
        config.max_eval_images_per_class,
        config.seed,
    )
    test = RadiographDataset.labeled(
        split_dir(config.data_root, "test"),
        config.image_size,
        config.max_eval_images_per_class,
        config.seed,
    )

    # Calcular puntuaciones de validación y test
    val_labels, val_scores, val_paths, samples = score_dataset_dae(
        model, validation, config.batch_size, device,
    )
    test_labels, test_scores, test_paths, _ = score_dataset_dae(
        model, test, config.batch_size, device,
    )

    # Selección del umbral 
    threshold = _select_threshold(val_labels, val_scores)

    # Evaluación de métricas de validación y test
    validation_metrics = _evaluate(val_labels, val_scores, threshold)
    test_metrics = _evaluate(test_labels, test_scores, threshold)

    # Guardar puntuaciones y reconstrucciones
    _save_scores(config.output_dir / "validation_scores.csv", val_labels, val_scores, val_paths)
    _save_scores(config.output_dir / "test_scores.csv", test_labels, test_scores, test_paths)
    _save_reconstructions(config.output_dir / "reconstructions.png", *samples)

    # Contar parámetros del modelo
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    report = {
        "config": {**asdict(config), "data_root": str(config.data_root), "output_dir": str(config.output_dir)},
        "device": str(device),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "history": history,
        "selected_epoch": selected_epoch,
        "threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_run": (
            config.epochs >= 3
            and config.max_train_images is None
            and config.max_eval_images_per_class is None
        ),
    }
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if report["scientific_run"] and config.seed == 42:
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_name": "dae",
                "image_size": config.image_size,
                "threshold": float(threshold),
                "validation_auroc": float(validation_metrics["auroc"]),
                "test_auroc": float(test_metrics["auroc"]),
            },
            checkpoint_dir / "modelo_autoencoder.pt",
        )
    return report
