"""Training and evaluation orchestration for the autoencoder."""

from __future__ import annotations

import contextlib
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from . import PROJECT_ROOT
from .data import RadiographDataset, split_dir
from .losses import DEFAULT_SSIM_WEIGHT, reconstruction_loss
from .metrics import evaluate
from .models import ARCHITECTURE_VERSION, ConvAutoencoder
from .scoring import (
    DEFAULT_ERROR_QUANTILE,
    DEFAULT_MAHALANOBIS_RIDGE,
    SCORE_NAME,
    combine_components,
    fit_normal_distribution,
    latent_vectors,
    mahalanobis,
    reconstruction_score,
)


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 4
    image_size: int = 1024
    learning_rate: float = 1e-3
    threshold_quantile: float = 0.95
    error_quantile: float = DEFAULT_ERROR_QUANTILE
    ssim_weight: float = DEFAULT_SSIM_WEIGHT
    denoise_noise: float = 0.05
    latent_score: bool = True
    mahalanobis_ridge: float = DEFAULT_MAHALANOBIS_RIDGE
    flip_score: bool = False
    num_workers: int = 2
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _loader(
    dataset: RadiographDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _cpu_state_dict(model: ConvAutoencoder) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _checkpoint_payload(
    model_state: dict[str, torch.Tensor],
    image_size: int,
    **metadata: object,
) -> dict:
    return {
        "model_name": "ae",
        "architecture": ARCHITECTURE_VERSION,
        "image_size": image_size,
        "score": SCORE_NAME,
        "model_state": model_state,
        **metadata,
    }


def _train_batch(
    model: ConvAutoencoder,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    autocast: contextlib.AbstractContextManager,
    ssim_weight: float,
    denoise_noise: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with autocast:
        source = images
        if denoise_noise:
            source = images + torch.randn_like(images) * denoise_noise
        loss = reconstruction_loss(model(source), images, ssim_weight)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def _normal_validation_loss(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    error_quantile: float,
    autocast: contextlib.AbstractContextManager,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for images, _labels, _paths in loader:
        images = images.to(device)
        with autocast:
            score = reconstruction_score(model, images, error_quantile)
        total += float(score.reconstruction_mae.float().sum())
        count += len(images)
    return total / max(count, 1)


@torch.inference_mode()
def collect_latent_vectors(
    model: ConvAutoencoder,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    autocast: contextlib.AbstractContextManager,
) -> torch.Tensor:
    """Encoder descriptors of every image, for the normal distribution fit."""
    model.eval()
    vectors: list[torch.Tensor] = []
    loader = _loader(dataset, batch_size, False, num_workers)
    for images, _labels, _paths in loader:
        with autocast:
            vectors.append(latent_vectors(model, images.to(device)))
    return torch.cat(vectors, dim=0)


def train(
    config: ExperimentConfig, device: torch.device
) -> tuple[ConvAutoencoder, list[dict], int]:
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
    train_loader = _loader(train_set, config.batch_size, True, config.num_workers)
    validation_loader = _loader(
        validation_set, config.batch_size, False, config.num_workers
    )
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 10
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast = _autocast(device)
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
            loss = _train_batch(
                model,
                images,
                optimizer,
                scaler,
                autocast,
                config.ssim_weight,
                config.denoise_noise,
            )
            total_loss += loss * len(images)
            seen += len(images)

        scheduler.step()
        validation_value = _normal_validation_loss(
            model, validation_loader, device, config.error_quantile, autocast
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_value": validation_value,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"AE epoch={epoch}/{config.epochs} "
            f"train={record['train_loss']:.6f} val={validation_value:.6f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        improved = validation_value < best_value
        if improved:
            best_value = validation_value
            best_state = {
                "model": _cpu_state_dict(model),
                "epoch": epoch,
            }

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            best_state["model"],
            config.image_size,
            selected_epoch=best_state["epoch"],
            error_quantile=config.error_quantile,
            ssim_weight=config.ssim_weight,
            denoise_noise=config.denoise_noise,
        ),
        config.output_dir / "model.pt",
    )
    return model, history, int(best_state["epoch"])


@torch.inference_mode()
def score_dataset(
    model: ConvAutoencoder,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
    error_quantile: float,
    num_workers: int,
    autocast: contextlib.AbstractContextManager,
    latent_mean: torch.Tensor | None = None,
    latent_inv_cov: torch.Tensor | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    tuple[torch.Tensor, torch.Tensor],
]:
    model.eval()
    recon_values: list[np.ndarray] = []
    latent_values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False, num_workers)
    use_latent = latent_mean is not None and latent_inv_cov is not None
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        with autocast:
            score = reconstruction_score(model, device_images, error_quantile)
        recon_values.append(score.anomaly_score.float().cpu().numpy())
        if use_latent:
            vectors = latent_vectors(model, device_images)
            distances = mahalanobis(vectors, latent_mean, latent_inv_cov)
            latent_values.append(distances.float().cpu().numpy())
        else:
            latent_values.append(np.zeros(len(images), dtype=np.float64))
        labels.append(batch_labels.numpy())
        paths.extend(batch_paths)
        if samples is None:
            samples = (images[:8], score.reconstructed.float().cpu()[:8])
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"score progress={batch_index}/{len(loader)} "
                f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    assert samples is not None
    return (
        np.concatenate(labels),
        np.concatenate(recon_values),
        np.concatenate(latent_values),
        paths,
        samples,
    )


def _combined(
    recon_scores: np.ndarray,
    latent_scores: np.ndarray,
    calibration: dict[str, float],
    use_latent: bool,
) -> np.ndarray:
    if not use_latent:
        return recon_scores
    return np.asarray(
        combine_components(recon_scores, latent_scores, calibration),
        dtype=np.float64,
    )


def _calibration_from_normal(
    recon_scores: np.ndarray, latent_scores: np.ndarray, use_latent: bool
) -> dict[str, float]:
    calibration = {
        "recon_location": float(recon_scores.mean()),
        "recon_scale": float(recon_scores.std()) or 1.0,
        "latent_location": float(latent_scores.mean()),
        "latent_scale": float(latent_scores.std()) or 1.0,
    }
    if not use_latent:
        calibration["latent_scale"] = 1.0
    return calibration


def _save_scores(
    path: Path, labels: np.ndarray, scores: np.ndarray, paths: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "score", "path"])
        writer.writerows(zip(labels.tolist(), scores.tolist(), paths))


def _save_reconstructions(
    path: Path, originals: torch.Tensor, reconstructed: torch.Tensor
) -> None:
    count = min(8, len(originals))
    size = originals.shape[-1]
    canvas = Image.new("L", (count * size, 2 * size + 20), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), "AE: original (arriba) / reconstrucción (abajo)", fill=0)
    for index in range(count):
        for row, tensor in enumerate((originals[index, 0], reconstructed[index, 0])):
            array = (tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            canvas.paste(
                Image.fromarray(array, mode="L"), (index * size, 20 + row * size)
            )
    canvas.save(path)


def run(config: ExperimentConfig) -> dict:
    if config.image_size < 64 or config.image_size % 64:
        raise ValueError("--image-size debe ser múltiplo de 64 y al menos 64")
    if not 0.0 < config.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile debe estar entre 0 y 1")
    if not 0.0 < config.error_quantile < 1.0:
        raise ValueError("--error-quantile debe estar entre 0 y 1")
    if not 0.0 <= config.denoise_noise < 1.0:
        raise ValueError("--denoise-noise debe estar entre 0 y 1")
    if config.ssim_weight < 0.0:
        raise ValueError("--ssim-weight no puede ser negativo")
    if config.num_workers < 0:
        raise ValueError("--num-workers no puede ser negativo")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = _autocast(device)
    started = time.perf_counter()
    model, history, selected_epoch = train(config, device)
    model.eval()

    if config.latent_score:
        train_set = RadiographDataset.normal_only(
            split_dir(config.data_root, "train"),
            config.image_size,
            config.max_train_images,
            config.seed,
        )
        train_features = collect_latent_vectors(
            model,
            train_set,
            config.batch_size,
            device,
            config.num_workers,
            autocast,
        )
        latent_mean, latent_inv_cov = fit_normal_distribution(
            train_features, config.mahalanobis_ridge
        )
    else:
        latent_mean = latent_inv_cov = None

    normal_validation = RadiographDataset.normal_only(
        split_dir(config.data_root, "val"),
        config.image_size,
        None,
        config.seed,
    )
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
    _normal_labels, normal_recon, normal_latent, _normal_paths, _normal_samples = (
        score_dataset(
            model,
            normal_validation,
            config.batch_size,
            device,
            config.error_quantile,
            config.num_workers,
            autocast,
            latent_mean,
            latent_inv_cov,
        )
    )
    calibration = _calibration_from_normal(
        normal_recon, normal_latent, config.latent_score
    )
    normal_combined = _combined(
        normal_recon, normal_latent, calibration, config.latent_score
    )
    if config.flip_score:
        normal_combined = -normal_combined
    threshold = float(np.quantile(normal_combined, config.threshold_quantile))

    val_labels, val_recon, val_latent, val_paths, samples = score_dataset(
        model,
        validation,
        config.batch_size,
        device,
        config.error_quantile,
        config.num_workers,
        autocast,
        latent_mean,
        latent_inv_cov,
    )
    test_labels, test_recon, test_latent, test_paths, _ = score_dataset(
        model,
        test,
        config.batch_size,
        device,
        config.error_quantile,
        config.num_workers,
        autocast,
        latent_mean,
        latent_inv_cov,
    )
    val_combined = _combined(val_recon, val_latent, calibration, config.latent_score)
    test_combined = _combined(
        test_recon, test_latent, calibration, config.latent_score
    )
    if config.flip_score:
        val_combined = -val_combined
        test_combined = -test_combined
    validation_metrics = evaluate(val_labels, val_combined, threshold)
    test_metrics = evaluate(test_labels, test_combined, threshold)

    _save_scores(
        config.output_dir / "validation_scores.csv", val_labels, val_combined, val_paths
    )
    _save_scores(
        config.output_dir / "test_scores.csv", test_labels, test_combined, test_paths
    )
    _save_reconstructions(config.output_dir / "reconstructions.png", *samples)
    report = {
        "config": {
            **asdict(config),
            "model": "ae",
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
        },
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "selected_epoch": selected_epoch,
        "score": SCORE_NAME,
        "threshold_method": "normal_validation_quantile",
        "calibration": calibration,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_run": (
            config.epochs >= 3
            and config.image_size == 1024
            and config.threshold_quantile == 0.95
            and config.error_quantile == DEFAULT_ERROR_QUANTILE
            and config.ssim_weight == DEFAULT_SSIM_WEIGHT
            and config.denoise_noise == 0.05
            and config.latent_score
            and config.mahalanobis_ridge == DEFAULT_MAHALANOBIS_RIDGE
            and not config.flip_score
            and config.max_train_images is None
            and config.max_eval_images_per_class is None
        ),
    }
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if report["scientific_run"] and config.seed == 42:
        torch.save(
            _checkpoint_payload(
                _cpu_state_dict(model),
                config.image_size,
                threshold=float(threshold),
                threshold_method="normal_validation_quantile",
                threshold_quantile=config.threshold_quantile,
                error_quantile=config.error_quantile,
                ssim_weight=config.ssim_weight,
                denoise_noise=config.denoise_noise,
                calibration=calibration,
                latent_mean=latent_mean.cpu() if latent_mean is not None else None,
                latent_inv_cov=latent_inv_cov.cpu()
                if latent_inv_cov is not None
                else None,
                validation_auroc=float(validation_metrics["auroc"]),
                test_auroc=float(test_metrics["auroc"]),
            ),
            PROJECT_ROOT / "checkpoints/modelo_autoencoder.pt",
        )
    return report
