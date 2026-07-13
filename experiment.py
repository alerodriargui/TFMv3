"""Training and evaluation orchestration shared by all command-line scripts."""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import RadiographDataset, split_dir
from metrics import best_balanced_threshold, evaluate
from models import ModelBundle, build_model, per_image_scores, vae_loss


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    data_root: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 64
    image_size: int = 64
    latent_dim: int = 64
    learning_rate: float = 2e-4
    seed: int = 42
    max_train_images: int | None = None
    max_eval_images_per_class: int | None = None
    vae_beta: float = 1e-4
    adversarial_weight: float = 1.0
    contextual_weight: float = 50.0
    latent_weight: float = 1.0


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


def _train_standard(
    bundle: ModelBundle,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    if bundle.name == "ae":
        loss = F.l1_loss(bundle.model(images), images)
    else:
        reconstructed, mu, logvar = bundle.model(images)
        loss = vae_loss(reconstructed, images, mu, logvar, config.vae_beta)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _train_ganomaly(
    bundle: ModelBundle,
    images: torch.Tensor,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> tuple[float, float]:
    assert bundle.discriminator is not None
    discriminator = bundle.discriminator
    bce = nn.BCEWithLogitsLoss()

    generated, latent_in, latent_out = bundle.model(images)
    discriminator_optimizer.zero_grad(set_to_none=True)
    real_logits, _ = discriminator(images)
    fake_logits, _ = discriminator(generated.detach())
    discriminator_loss = 0.5 * (
        bce(real_logits, torch.ones_like(real_logits))
        + bce(fake_logits, torch.zeros_like(fake_logits))
    )
    discriminator_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad(set_to_none=True)
    generated, latent_in, latent_out = bundle.model(images)
    with torch.no_grad():
        _, real_features = discriminator(images)
    _, fake_features = discriminator(generated)
    adversarial = F.mse_loss(fake_features, real_features)
    contextual = F.l1_loss(generated, images)
    latent = F.mse_loss(latent_out, latent_in)
    generator_loss = (
        config.adversarial_weight * adversarial
        + config.contextual_weight * contextual
        + config.latent_weight * latent
    )
    generator_loss.backward()
    generator_optimizer.step()
    return float(generator_loss.detach()), float(discriminator_loss.detach())


@torch.no_grad()
def _normal_validation_loss(
    bundle: ModelBundle, loader: DataLoader, device: torch.device, vae_beta: float
) -> float:
    bundle.model.eval()
    total = 0.0
    count = 0
    for images, _labels, _paths in loader:
        images = images.to(device)
        scores, _ = per_image_scores(bundle, images, vae_beta)
        total += float(scores.sum())
        count += len(images)
    return total / max(count, 1)


def train(
    config: ExperimentConfig, device: torch.device
) -> tuple[ModelBundle, list[dict], int]:
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
    bundle = build_model(config.model, config.latent_dim)
    bundle.model.to(device)
    if bundle.discriminator is not None:
        bundle.discriminator.to(device)

    optimizer = torch.optim.Adam(
        bundle.model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = (
        torch.optim.Adam(
            bundle.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999),
        )
        if bundle.discriminator is not None
        else None
    )
    best_loss = float("inf")
    history: list[dict] = []
    best_state: dict | None = None

    for epoch in range(1, config.epochs + 1):
        bundle.model.train()
        if bundle.discriminator is not None:
            bundle.discriminator.train()
        started = time.perf_counter()
        total_generator = 0.0
        total_discriminator = 0.0
        seen = 0
        for images, _labels, _paths in train_loader:
            images = images.to(device)
            if bundle.name == "ganomaly":
                assert discriminator_optimizer is not None
                generator_loss, discriminator_loss = _train_ganomaly(
                    bundle, images, optimizer, discriminator_optimizer, config
                )
            else:
                generator_loss = _train_standard(bundle, images, optimizer, config)
                discriminator_loss = 0.0
            total_generator += generator_loss * len(images)
            total_discriminator += discriminator_loss * len(images)
            seen += len(images)

        validation_loss = _normal_validation_loss(
            bundle, validation_loader, device, config.vae_beta
        )
        record = {
            "epoch": epoch,
            "train_loss": total_generator / seen,
            "discriminator_loss": total_discriminator / seen,
            "normal_validation_score": validation_loss,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"{config.model} epoch={epoch}/{config.epochs} "
            f"train={record['train_loss']:.6f} val={validation_loss:.6f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in bundle.model.state_dict().items()
                },
                "discriminator": (
                    {
                        key: value.detach().cpu().clone()
                        for key, value in bundle.discriminator.state_dict().items()
                    }
                    if bundle.discriminator is not None
                    else None
                ),
                "epoch": epoch,
            }

    assert best_state is not None
    bundle.model.load_state_dict(best_state["model"])
    if bundle.discriminator is not None and best_state["discriminator"] is not None:
        bundle.discriminator.load_state_dict(best_state["discriminator"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": config.model,
            "latent_dim": config.latent_dim,
            "image_size": config.image_size,
            "vae_beta": config.vae_beta,
            "model_state": best_state["model"],
            "discriminator_state": best_state["discriminator"],
            "selected_epoch": best_state["epoch"],
        },
        config.output_dir / "model.pt",
    )
    return bundle, history, int(best_state["epoch"])


@torch.no_grad()
def score_dataset(
    bundle: ModelBundle,
    dataset: RadiographDataset,
    batch_size: int,
    device: torch.device,
    vae_beta: float,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[torch.Tensor, torch.Tensor]]:
    bundle.model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    samples: tuple[torch.Tensor, torch.Tensor] | None = None
    loader = _loader(dataset, batch_size, False)
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        device_images = images.to(device)
        batch_scores, reconstructed = per_image_scores(bundle, device_images, vae_beta)
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


def _save_scores(
    path: Path, labels: np.ndarray, scores: np.ndarray, paths: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "score", "path"])
        writer.writerows(zip(labels.tolist(), scores.tolist(), paths))


def _save_reconstructions(
    path: Path, originals: torch.Tensor, reconstructed: torch.Tensor, model: str
) -> None:
    count = min(8, len(originals))
    size = originals.shape[-1]
    canvas = Image.new("L", (count * size, 2 * size + 20), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), f"{model}: original (arriba) / reconstrucción (abajo)", fill=0)
    for index in range(count):
        for row, tensor in enumerate((originals[index, 0], reconstructed[index, 0])):
            array = (tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            canvas.paste(
                Image.fromarray(array, mode="L"), (index * size, 20 + row * size)
            )
    canvas.save(path)


def run(config: ExperimentConfig) -> dict:
    if config.image_size != 64:
        raise ValueError(
            "Las arquitecturas comparables actuales requieren --image-size 64"
        )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    bundle, history, selected_epoch = train(config, device)

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
        bundle, validation, config.batch_size, device, config.vae_beta
    )
    threshold, _ = best_balanced_threshold(val_labels, val_scores)
    test_labels, test_scores, test_paths, _ = score_dataset(
        bundle, test, config.batch_size, device, config.vae_beta
    )
    validation_metrics = evaluate(val_labels, val_scores, threshold)
    test_metrics = evaluate(test_labels, test_scores, threshold)

    _save_scores(
        config.output_dir / "validation_scores.csv", val_labels, val_scores, val_paths
    )
    _save_scores(
        config.output_dir / "test_scores.csv", test_labels, test_scores, test_paths
    )
    _save_reconstructions(
        config.output_dir / "reconstructions.png", *samples, config.model
    )
    report = {
        "config": {
            **asdict(config),
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
        },
        "device": str(device),
        "parameter_count": sum(
            parameter.numel() for parameter in bundle.model.parameters()
        ),
        "discriminator_parameter_count": (
            sum(parameter.numel() for parameter in bundle.discriminator.parameters())
            if bundle.discriminator is not None
            else 0
        ),
        "history": history,
        "selected_epoch": selected_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_run": (
            config.epochs >= 20
            and config.max_train_images is None
            and config.max_eval_images_per_class is None
        ),
    }
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report
