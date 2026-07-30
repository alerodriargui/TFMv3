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
from .metrics import evaluate
from .models import ARCHITECTURE_VERSION, ConvAutoencoder
from .scoring import SCORE_NAME, reconstruction_score


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 2
    image_size: int = 1024
    learning_rate: float = 1e-3
    threshold_quantile: float = 0.95
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None


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
) -> float:
    optimizer.zero_grad(set_to_none=True)
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
        score = reconstruction_score(model, images)
        total += float(score.reconstruction_mae.sum())
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
    model = ConvAutoencoder().to(device)
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
            loss = _train_batch(model, images, optimizer)
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
        ),
        config.output_dir / "model.pt",
    )
    return model, history, int(best_state["epoch"])


@torch.no_grad()
def score_dataset(
    model: ConvAutoencoder,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    tuple[torch.Tensor, torch.Tensor],
]:
    model.eval()
    reconstruction_values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        score = reconstruction_score(model, device_images)
        reconstruction_values.append(score.reconstruction_mae.cpu().numpy())
        labels.append(batch_labels.numpy())
        paths.extend(batch_paths)
        if samples is None:
            samples = (images[:8], score.reconstructed.cpu()[:8])
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"score progress={batch_index}/{len(loader)} "
                f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    assert samples is not None
    return (
        np.concatenate(labels),
        np.concatenate(reconstruction_values),
        paths,
        samples,
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
    if config.image_size < 64 or config.image_size % 64:
        raise ValueError("--image-size debe ser múltiplo de 64 y al menos 64")
    if not 0.0 < config.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile debe estar entre 0 y 1")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model, history, selected_epoch = train(config, device)

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
    _normal_labels, normal_scores, _normal_paths, _normal_samples = score_dataset(
        model, normal_validation, config.batch_size, device
    )
    threshold = float(np.quantile(normal_scores, config.threshold_quantile))
    val_labels, val_scores, val_paths, samples = score_dataset(
        model, validation, config.batch_size, device
    )
    test_labels, test_scores, test_paths, _ = score_dataset(
        model, test, config.batch_size, device
    )
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
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_run": (
            config.epochs >= 3
            and config.image_size == 1024
            and config.threshold_quantile == 0.95
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
                validation_auroc=float(validation_metrics["auroc"]),
                test_auroc=float(test_metrics["auroc"]),
            ),
            PROJECT_ROOT / "checkpoints/modelo_autoencoder.pt",
        )
    return report
