"""Training and evaluation orchestration for the autoencoder."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import PROJECT_ROOT
from .data import RadiographDataset, split_dir
from .metrics import auroc, best_balanced_threshold, evaluate
from .models import build_model, per_image_signals, ssim_map


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 32
    image_size: int = 64
    model_name: str = "ae"
    learning_rate: float = 1e-3
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None
    score_type: str = "mae"
    loss_type: str = "l1"
    denoise_sigma: float = 0.0
    center_weight: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset: RadiographDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _reconstruction_loss(
    images: torch.Tensor, reconstructed: torch.Tensor, loss_type: str
) -> torch.Tensor:
    """L1, SSIM or a 0.5/0.5 combination as the reconstruction loss."""
    if loss_type == "l1":
        return F.l1_loss(reconstructed, images)
    if loss_type == "ssim":
        return 1.0 - torch.mean(ssim_map(images, reconstructed))
    if loss_type == "l1ssim":
        l1 = F.l1_loss(reconstructed, images)
        ssim = 1.0 - torch.mean(ssim_map(images, reconstructed))
        return 0.5 * l1 + 0.5 * ssim
    raise ValueError(f"Tipo de pérdida desconocido: {loss_type}")


def _train_batch(
    model: ConvAutoencoder,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_type: str,
    denoise_sigma: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    target = images
    if denoise_sigma > 0.0:
        images = images + denoise_sigma * torch.randn_like(images)
    loss = _reconstruction_loss(images, model(images), loss_type)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def _normal_validation_loss(
    model: ConvAutoencoder, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for images, _labels, _paths in loader:
        images = images.to(device)
        signals, _ = per_image_signals(model, images)
        total += float(signals["mae"].sum())
        count += len(images)
    return total / max(count, 1)


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
    train_loader = _loader(train_set, config.batch_size, True)
    validation_loader = _loader(validation_set, config.batch_size, False)
    model = build_model(config.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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
                model, images, optimizer, config.loss_type, config.denoise_sigma
            )
            total_loss += loss * len(images)
            seen += len(images)

        validation_value = _normal_validation_loss(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_value": validation_value,
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
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "epoch": epoch,
            }

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": config.model_name,
            "image_size": config.image_size,
            "model_state": best_state["model"],
            "selected_epoch": best_state["epoch"],
        },
        config.output_dir / "model.pt",
    )
    return model, history, int(best_state["epoch"])


@torch.no_grad()
def score_dataset(
    model: ConvAutoencoder,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    signals: dict[str, list[np.ndarray]] = {}
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        batch_signals, reconstructed = per_image_signals(model, device_images)
        for name, values in batch_signals.items():
            signals.setdefault(name, []).append(values.cpu().numpy())
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
    return (
        np.concatenate(labels),
        {name: np.concatenate(values) for name, values in signals.items()},
        paths,
        samples,
    )


def _center_border_masks(
    image_size: int,
) -> tuple[torch.Tensor, slice, slice]:
    """Máscaras centro (mitad central) y borde (anillo exterior) del tamaño dado."""
    center = slice(image_size // 4, 3 * image_size // 4)
    border_mask = torch.ones((image_size, image_size), dtype=torch.bool)
    border_mask[image_size // 8 : 7 * image_size // 8, image_size // 8 : 7 * image_size // 8] = False
    return border_mask, center, center


@torch.no_grad()
def center_border_scores(
    dataset: RadiographDataset, batch_size: int, image_size: int
) -> np.ndarray:
    """Return the mean intensity difference between image center and border."""
    values = []
    border_mask, center_rows, center_cols = _center_border_masks(image_size)
    for images, _labels, _paths in _loader(dataset, batch_size, False):
        pixels = images.squeeze(1)
        center = pixels[:, center_rows, center_cols].mean(dim=(1, 2))
        border = pixels[:, border_mask].mean(dim=1)
        values.append((center - border).numpy())
    return np.concatenate(values)


def _z_score(
    raw: np.ndarray, sign: float, location: float, scale: float
) -> np.ndarray:
    return (raw * sign - location) / scale


def calibrate_ae_scores(
    config: ExperimentConfig,
    validation: RadiographDataset,
    test: RadiographDataset,
    val_labels: np.ndarray,
    signals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combine calibrated reconstruction signals and the center-border signal.

    Each reconstruction signal is sign-adjusted and z-scored against the
    normal distribution of validation. The final score is
    ``(1 - center_weight) * mean(signal z-scores) + center_weight * center z``.
    """
    train = RadiographDataset.normal_only(
        split_dir(config.data_root, "train"),
        config.image_size,
        config.max_train_images,
        config.seed,
    )
    train_center = center_border_scores(train, config.batch_size, config.image_size)
    val_center = center_border_scores(validation, config.batch_size, config.image_size)
    test_center = center_border_scores(test, config.batch_size, config.image_size)

    signal_parts: dict[str, dict[str, float]] = {}
    for name, (raw_val, raw_test) in signals.items():
        sign = -1.0 if auroc(val_labels, raw_val) < 0.5 else 1.0
        normal = (raw_val * sign)[val_labels == 0]
        location = float(normal.mean())
        scale = max(float(normal.std()), 1e-8)
        signal_parts[name] = {"sign": sign, "location": location, "scale": scale}

    center_sign = -1.0 if auroc(val_labels, val_center) < 0.5 else 1.0
    center_location = float((train_center * center_sign).mean())
    center_scale = max(float(train_center.std()), 1e-8)

    recon_weight = 1.0 - config.center_weight
    signal_weight = recon_weight / len(signal_parts)
    _, raw_test_reference = next(iter(signals.values()))
    val_recon = np.zeros(len(val_labels), dtype=np.float64)
    test_recon = np.zeros(len(raw_test_reference), dtype=np.float64)
    for name, (raw_val, raw_test) in signals.items():
        part = signal_parts[name]
        val_recon += signal_weight * _z_score(
            raw_val, part["sign"], part["location"], part["scale"]
        )
        test_recon += signal_weight * _z_score(
            raw_test, part["sign"], part["location"], part["scale"]
        )
    val_final = val_recon + config.center_weight * _z_score(
        val_center, center_sign, center_location, center_scale
    )
    test_final = test_recon + config.center_weight * _z_score(
        test_center, center_sign, center_location, center_scale
    )
    names = " + ".join(
        f"{signal_weight:.3f} * z({name})" for name in signals
    ) + f" + {config.center_weight:.3f} * z(center_border)"
    primary = "mae" if "mae" in signal_parts else next(iter(signal_parts))
    return (
        val_final,
        test_final,
        {
            "score": names,
            "score_type": config.score_type,
            "center_weight": config.center_weight,
            "signals": signal_parts,
            "center_sign": center_sign,
            "center_location": center_location,
            "center_scale": center_scale,
            "ae_sign": signal_parts[primary]["sign"],
            "ae_location": signal_parts[primary]["location"],
            "ae_scale": signal_parts[primary]["scale"],
        },
    )


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
    step = 16 if config.model_name == "unet" else 8
    if config.image_size % step != 0:
        raise ValueError(
            f"El modelo {config.model_name} requiere --image-size múltiplo de {step}"
        )
    score_types = {
        "mae": ("mae",),
        "ssim": ("ssim",),
        "mae_ssim": ("mae", "ssim"),
    }
    if config.score_type not in score_types:
        raise ValueError(f"Tipo de puntuación desconocido: {config.score_type}")
    if config.loss_type not in ("l1", "ssim", "l1ssim"):
        raise ValueError(f"Tipo de pérdida desconocido: {config.loss_type}")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model, history, selected_epoch = train(config, device)

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
    val_labels, val_signals, val_paths, samples = score_dataset(
        model, validation, config.batch_size, device
    )
    test_labels, test_signals, test_paths, _ = score_dataset(
        model, test, config.batch_size, device
    )
    signals = {
        name: (val_signals[name], test_signals[name])
        for name in score_types[config.score_type]
    }
    val_scores, test_scores, calibration = calibrate_ae_scores(
        config,
        validation,
        test,
        val_labels,
        signals,
    )
    threshold, _ = best_balanced_threshold(val_labels, val_scores)
    validation_metrics = evaluate(val_labels, val_scores, threshold)
    test_metrics = evaluate(test_labels, test_scores, threshold)

    _save_scores(
        config.output_dir / "validation_scores.csv", val_labels, val_scores, val_paths
    )
    _save_scores(
        config.output_dir / "test_scores.csv", test_labels, test_scores, test_paths
    )
    _save_reconstructions(config.output_dir / "reconstructions.png", *samples)
    report = {
        "config": {
            **asdict(config),
            "model": config.model_name,
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
        },
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "selected_epoch": selected_epoch,
        "calibration": calibration,
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
        torch.save(
            {
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "model_name": config.model_name,
                "image_size": config.image_size,
                "threshold": float(threshold),
                "calibration": calibration,
                "validation_auroc": float(validation_metrics["auroc"]),
                "test_auroc": float(test_metrics["auroc"]),
            },
            PROJECT_ROOT / "checkpoints/modelo_autoencoder.pt",
        )
    return report
