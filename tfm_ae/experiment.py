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
from .data import RandomFlipRotate, RadiographDataset, split_dir
from .features import GLOBAL_SIGNALS, feature_matrix
from .metrics import auroc, best_balanced_threshold, evaluate
from .models import build_model, per_image_scores, ConvAutoencoder


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
    noise_std: float = 0.0
    bottleneck_channels: int = 32


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


def _train_batch(
    model: ConvAutoencoder,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    noise_std: float = 0.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    if noise_std > 0:
        target = images
        noisy = images + noise_std * torch.randn_like(images)
        loss = F.l1_loss(model(noisy), target)
    else:
        loss = F.l1_loss(model(images), images)
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
        scores, _ = per_image_scores(model, images)
        total += float(scores.sum())
        count += len(images)
    return total / max(count, 1)


def train(
    config: ExperimentConfig, device: torch.device
) -> tuple[ConvAutoencoder, list[dict], int]:
    augmentation = RandomFlipRotate(seed=config.seed)
    train_set = RadiographDataset.normal_only(
        split_dir(config.data_root, "train"),
        config.image_size,
        config.max_train_images,
        config.seed,
        transform=augmentation,
    )
    validation_set = RadiographDataset.normal_only(
        split_dir(config.data_root, "val"),
        config.image_size,
        None,
        config.seed,
    )
    train_loader = _loader(train_set, config.batch_size, True)
    validation_loader = _loader(validation_set, config.batch_size, False)
    model = build_model(config.bottleneck_channels).to(device)
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
            loss = _train_batch(model, images, optimizer, config.noise_std)
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
            "bottleneck_channels": config.bottleneck_channels,
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
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        batch_scores, reconstructed = per_image_scores(model, device_images)
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


def _feature_scores(
    dataset: RadiographDataset, batch_size: int
) -> dict[str, np.ndarray]:
    """Compute the global feature bank for every image in ``dataset``."""
    values = {key: [] for key in GLOBAL_SIGNALS}
    for images, _labels, _paths in _loader(dataset, batch_size, False):
        bank = feature_matrix(images.squeeze(1).numpy())
        for key in GLOBAL_SIGNALS:
            values[key].extend(bank[key].tolist())
    return {key: np.asarray(values[key]) for key in GLOBAL_SIGNALS}


def calibrate_hybrid_scores(
    config: ExperimentConfig,
    validation: RadiographDataset,
    test: RadiographDataset,
    val_labels: np.ndarray,
    raw_val_scores: np.ndarray,
    raw_test_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combine calibrated global signals and AE MAE, weight chosen on validation.

    Each signal is z-scored against its own distribution on normal training
    images; the sign is set from the AUROC direction on validation. The mixing
    weight ``w`` between the summed global signal and the MAE is selected on
    validation by a coarse grid search, then frozen for the test set.
    """
    train = RadiographDataset.normal_only(
        split_dir(config.data_root, "train"),
        config.image_size,
        config.max_train_images,
        config.seed,
    )
    train_features = _feature_scores(train, config.batch_size)
    val_features = _feature_scores(validation, config.batch_size)
    test_features = _feature_scores(test, config.batch_size)

    global_signs: dict[str, float] = {}
    for key in GLOBAL_SIGNALS:
        a = auroc(val_labels, val_features[key])
        global_signs[key] = -1.0 if a < 0.5 else 1.0

    def calibrate(features: dict[str, np.ndarray]) -> np.ndarray:
        total = None
        for key in GLOBAL_SIGNALS:
            signed = features[key] * global_signs[key]
            normal = train_features[key] * global_signs[key]
            location = float(normal.mean())
            scale = max(float(normal.std()), 1e-8)
            z = (signed - location) / scale
            total = z if total is None else total + z
        assert total is not None
        return total

    global_location = {}
    global_scale = {}
    for key in GLOBAL_SIGNALS:
        normal = train_features[key] * global_signs[key]
        global_location[key] = float(normal.mean())
        global_scale[key] = max(float(normal.std()), 1e-8)

    val_global = calibrate(val_features)
    test_global = calibrate(test_features)

    ae_sign = -1.0 if auroc(val_labels, raw_val_scores) < 0.5 else 1.0
    val_ae = raw_val_scores * ae_sign
    test_ae = raw_test_scores * ae_sign
    normal_ae = val_ae[val_labels == 0]
    ae_location = float(normal_ae.mean())
    ae_scale = max(float(normal_ae.std()), 1e-8)
    val_ae = (val_ae - ae_location) / ae_scale
    test_ae = (test_ae - ae_location) / ae_scale

    best_w, best_va = 0.0, -float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        combined = w * val_global + (1 - w) * val_ae
        a = auroc(val_labels, combined)
        if a > best_va:
            best_va, best_w = a, w

    val_combined = best_w * val_global + (1 - best_w) * val_ae
    test_combined = best_w * test_global + (1 - best_w) * test_ae
    return (
        val_combined,
        test_combined,
        {
            "score": "w * calibrated_global + (1-w) * calibrated_MAE",
            "global_signals": list(GLOBAL_SIGNALS),
            "global_signs": global_signs,
            "global_location": global_location,
            "global_scale": global_scale,
            "ae_sign": ae_sign,
            "ae_location": ae_location,
            "ae_scale": ae_scale,
            "weight": best_w,
            "validation_auroc_global_only": float(auroc(val_labels, val_global)),
            "validation_auroc_mae_only": float(auroc(val_labels, val_ae)),
            "validation_auroc_hybrid": best_va,
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
    val_labels, val_scores, val_paths, samples = score_dataset(
        model, validation, config.batch_size, device
    )
    test_labels, test_scores, test_paths, _ = score_dataset(
        model, test, config.batch_size, device
    )
    val_scores, test_scores, calibration = calibrate_hybrid_scores(
        config,
        validation,
        test,
        val_labels,
        val_scores,
        test_scores,
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


def load_calibration_from_metrics(metrics_path: Path, data_root: Path) -> dict:
    """Load calibration from a metrics.json file for inference on new images."""
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = dict(report["calibration"])
    cache = metrics_path.parent / "global_stats.json"
    if "global_location" not in calibration:
        if cache.is_file():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            calibration["global_location"] = cached["location"]
            calibration["global_scale"] = cached["scale"]
        else:
            location, scale = _train_global_stats(
                report["config"],
                data_root,
                int(report["config"]["batch_size"]),
                calibration["global_signs"],
            )
            calibration["global_location"] = location
            calibration["global_scale"] = scale
            cache.write_text(
                json.dumps({"location": location, "scale": scale}, indent=2),
                encoding="utf-8",
            )
    calibration["_threshold"] = float(report["test"]["threshold"])
    calibration["_image_size"] = int(report["config"]["image_size"])
    calibration["_model_name"] = report["config"]["model"]
    calibration["_bottleneck"] = int(
        report["config"].get("bottleneck_channels", 32)
    )
    return calibration
