"""Entrenamiento y evaluación del autoencoder."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import PROJECT_ROOT
from .data import RandomFlipRotate, RadiographDataset, split_dir
from .metrics import best_balanced_threshold, evaluate
from .models import ConvAutoencoder, build_model, per_image_scores


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 32
    image_size: int = 240
    model_name: str = "ae"
    learning_rate: float = 1e-3
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None
    bottleneck_channels: int = 32
    encoder_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    junction_dim: int = 768
    junction_n_queries: int = 784
    junction_heads: int = 8
    decoder_dim: int = 768
    decoder_depth: int = 6
    decoder_heads: int = 12
    perceptual_patch_sizes: tuple[int, ...] = (32, 56)
    perceptual_layers: tuple[int, ...] = (15, 19)


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


# ── ConvAE pipeline ──────────────────────────────────────────────


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
        scores, _ = per_image_scores(model, images)
        total += float(scores.sum())
        count += len(images)
    return total / max(count, 1)


def train_conv(
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
def score_dataset_conv(
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


# ── QFAE pipeline ────────────────────────────────────────────────


def _train_batch_qfae(
    model: torch.nn.Module,
    perceptual_loss: torch.nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    reconstructed = model(images)
    loss, _ = perceptual_loss.loss_and_maps(images, reconstructed)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def _normal_validation_loss_qfae(
    model: torch.nn.Module,
    perceptual_loss: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    perceptual_loss.eval()
    total = 0.0
    count = 0
    for images, _labels, _paths in loader:
        images = images.to(device)
        reconstructed = model(images)
        _, loss_maps = perceptual_loss.loss_and_maps(images, reconstructed)
        spatial = loss_maps.amax(dim=1)
        image_scores = spatial.amax(dim=(1, 2))
        total += float(image_scores.sum())
        count += len(images)
    return total / max(count, 1)


def train_qfae(
    config: ExperimentConfig, device: torch.device
) -> tuple[torch.nn.Module, list[dict], int]:
    from .perceptual_loss import PerceptualLoss
    from .qfae import QFAE, build_qfae_model

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

    model = build_qfae_model(
        encoder_name=config.encoder_name,
        img_size=config.image_size,
        junction_dim=config.junction_dim,
        junction_n_queries=config.junction_n_queries,
        junction_heads=config.junction_heads,
        decoder_dim=config.decoder_dim,
        decoder_depth=config.decoder_depth,
        decoder_heads=config.decoder_heads,
    ).to(device)

    model.encoder.eval()
    model.encoder.requires_grad_(False)

    perceptual = PerceptualLoss(
        layers=config.perceptual_layers,
        patch_sizes=config.perceptual_patch_sizes,
        img_size=config.image_size,
    ).to(device)
    perceptual.eval()
    perceptual.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    best_value = float("inf")
    history: list[dict] = []
    best_state: dict | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        perceptual.eval()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for images, _labels, _paths in train_loader:
            images = images.to(device)
            loss = _train_batch_qfae(model, perceptual, images, optimizer)
            total_loss += loss * len(images)
            seen += len(images)

        validation_value = _normal_validation_loss_qfae(
            model, perceptual, validation_loader, device
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_value": validation_value,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"QFAE epoch={epoch}/{config.epochs} "
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
            "model_name": config.model_name,
            "image_size": config.image_size,
            "encoder_name": config.encoder_name,
            "junction_dim": config.junction_dim,
            "junction_n_queries": config.junction_n_queries,
            "junction_heads": config.junction_heads,
            "decoder_dim": config.decoder_dim,
            "decoder_depth": config.decoder_depth,
            "decoder_heads": config.decoder_heads,
            "model_state": best_state["model"],
            "selected_epoch": best_state["epoch"],
        },
        config.output_dir / "model.pt",
    )
    return model, history, int(best_state["epoch"])


@torch.no_grad()
def score_dataset_qfae(
    model: torch.nn.Module,
    perceptual_loss: torch.nn.Module,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    perceptual_loss.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        reconstructed = model(device_images)
        _, loss_maps = perceptual_loss.loss_and_maps(device_images, reconstructed)
        spatial = loss_maps.amax(dim=1)
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


# ── Shared helpers ───────────────────────────────────────────────


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


# ── Public entry point ──────────────────────────────────────────


def train(
    config: ExperimentConfig, device: torch.device
) -> tuple[torch.nn.Module, list[dict], int]:
    if config.model_name == "qfae":
        return train_qfae(config, device)
    return train_conv(config, device)


def score_dataset(
    model: torch.nn.Module,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
    model_name: str = "ae",
    perceptual_loss: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    if model_name == "qfae":
        assert perceptual_loss is not None
        return score_dataset_qfae(model, perceptual_loss, dataset, batch_size, device)
    return score_dataset_conv(model, dataset, batch_size, device)


def run(config: ExperimentConfig) -> dict:
    if config.image_size % 8 != 0:
        raise ValueError(
            f"El modelo requiere --image-size múltiplo de 8 (recibido {config.image_size})"
        )
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

    perceptual = None
    if config.model_name == "qfae":
        from .perceptual_loss import PerceptualLoss

        perceptual = PerceptualLoss(
            layers=config.perceptual_layers,
            patch_sizes=config.perceptual_patch_sizes,
            img_size=config.image_size,
        ).to(device)
        perceptual.eval()
        perceptual.requires_grad_(False)

    val_labels, val_scores, val_paths, samples = score_dataset(
        model, validation, config.batch_size, device,
        model_name=config.model_name, perceptual_loss=perceptual,
    )
    test_labels, test_scores, test_paths, _ = score_dataset(
        model, test, config.batch_size, device,
        model_name=config.model_name, perceptual_loss=perceptual,
    )
    threshold, _ = best_balanced_threshold(val_labels, val_scores)
    validation_metrics = evaluate(val_labels, val_scores, threshold)
    test_metrics = evaluate(test_labels, test_scores, threshold)

    _save_scores(config.output_dir / "validation_scores.csv", val_labels, val_scores, val_paths)
    _save_scores(config.output_dir / "test_scores.csv", test_labels, test_scores, test_paths)
    _save_reconstructions(config.output_dir / "reconstructions.png", *samples)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    report = {
        "config": {**asdict(config), "data_root": str(config.data_root), "output_dir": str(config.output_dir)},
        "device": str(device),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "history": history,
        "selected_epoch": selected_epoch,
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
                "model_name": config.model_name,
                "image_size": config.image_size,
                "threshold": float(threshold),
                "validation_auroc": float(validation_metrics["auroc"]),
                "test_auroc": float(test_metrics["auroc"]),
            },
            checkpoint_dir / "modelo_autoencoder.pt",
        )
    return report
