"""Detect one brain slice with the frozen hybrid-scored autoencoder."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import PROJECT_ROOT
from .data import RadiographDataset, resolve_data_root, split_dir
from .experiment import _loader
from .features import GLOBAL_SIGNALS, feature_matrix
from .models import build_model


@dataclass(frozen=True)
class DemoResult:
    reconstructed: torch.Tensor
    absolute_error: torch.Tensor
    mae: float
    anomaly_score: float
    threshold: float
    label: str
    global_features: dict[str, float]
    global_score: float
    mae_score: float


def load_image(path: Path, image_size: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def _train_global_stats(
    config: dict,
    data_root: Path,
    batch_size: int,
    global_signs: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean/std of signed global signals over normal training images."""
    train = RadiographDataset.normal_only(
        split_dir(data_root, "train"),
        int(config["image_size"]),
        config.get("max_train_images"),
        int(config["seed"]),
    )
    values = {key: [] for key in GLOBAL_SIGNALS}
    for images, _labels, _paths in _loader(train, batch_size, False):
        bank = feature_matrix(images.squeeze(1).numpy())
        for key in GLOBAL_SIGNALS:
            values[key].extend((bank[key] * global_signs[key]).tolist())
    return {key: float(np.mean(values[key])) for key in GLOBAL_SIGNALS}, {
        key: max(float(np.std(values[key])), 1e-8) for key in GLOBAL_SIGNALS
    }


def _load_calibration(metrics_path: Path, data_root: Path) -> dict:
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


def evaluate_image(
    image: torch.Tensor,
    model: torch.nn.Module,
    calibration: dict,
) -> DemoResult:
    """Apply the frozen model and its hybrid calibration to one image."""
    with torch.inference_mode():
        reconstructed = model(image)
        absolute_error = torch.abs(reconstructed - image)
        mae = float(torch.mean(absolute_error).item())

    pixels = image.squeeze(0).squeeze(0).numpy()
    bank = feature_matrix(pixels[None])
    features = {key: float(bank[key][0]) for key in GLOBAL_SIGNALS}

    global_score = 0.0
    for key in GLOBAL_SIGNALS:
        sign = float(calibration["global_signs"][key])
        location = float(calibration["global_location"][key])
        scale = float(calibration["global_scale"][key])
        global_score += (features[key] * sign - location) / scale

    ae_sign = float(calibration["ae_sign"])
    mae_score = (
        mae * ae_sign - float(calibration["ae_location"])
    ) / float(calibration["ae_scale"])
    weight = float(calibration["weight"])
    anomaly_score = weight * global_score + (1 - weight) * mae_score
    threshold = calibration["_threshold"]
    label = "ANÓMALA" if anomaly_score >= threshold else "NORMAL"
    return DemoResult(
        reconstructed=reconstructed,
        absolute_error=absolute_error,
        mae=mae,
        anomaly_score=anomaly_score,
        threshold=threshold,
        label=label,
        global_features=features,
        global_score=global_score,
        mae_score=mae_score,
    )


def create_figure(
    image: torch.Tensor,
    result: DemoResult,
    output: Path,
    show: bool,
) -> None:
    """Render the four-panel explanation and always save it as a PNG."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    original = image.squeeze().detach().cpu().numpy()
    reconstructed = result.reconstructed.squeeze().detach().cpu().numpy()
    absolute_error = result.absolute_error.squeeze().detach().cpu().numpy()

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("1. Imagen original preprocesada")

    axes[0, 1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("2. Reconstrucción del autoencoder")

    error_image = axes[1, 0].imshow(
        absolute_error,
        cmap="inferno",
        vmin=0,
        vmax=max(float(absolute_error.max()), 1e-6),
    )
    axes[1, 0].set_title(
        f"3. Error absoluto |original - reconstrucción|\nMAE: {result.mae:.4f}"
    )
    colorbar = figure.colorbar(error_image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("Error absoluto por píxel")

    score_axis = axes[1, 1]
    score_axis.set_title("4. Puntuación híbrida frente al umbral")
    score_axis.set_yticks([])
    values = (result.anomaly_score, result.threshold)
    span = max(abs(values[0] - values[1]), abs(values[0]), abs(values[1]), 1.0)
    lower = min(values) - 0.25 * span
    upper = max(values) + 0.25 * span
    score_axis.set_xlim(lower, upper)
    score_axis.set_xlabel("Escala de puntuación calibrada")
    score_axis.axvline(
        result.threshold,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label="Umbral de validación",
    )
    score_axis.scatter(
        result.anomaly_score,
        0.5,
        s=140,
        color="tab:red" if result.label == "ANÓMALA" else "tab:green",
        zorder=3,
        label="Puntuación de anomalía",
    )
    score_axis.set_ylim(0, 1)
    score_axis.grid(axis="x", alpha=0.25)
    score_axis.legend(loc="upper center", frameon=False)
    score_axis.text(
        0.04,
        0.70,
        f"Puntuación: {result.anomaly_score:.4f}\nUmbral: {result.threshold:.4f}",
        transform=score_axis.transAxes,
        ha="left",
        va="center",
        fontsize=11,
    )
    score_axis.text(
        0.5,
        0.12,
        result.label,
        transform=score_axis.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="tab:red" if result.label == "ANÓMALA" else "tab:green",
    )

    features = "  ".join(
        f"{key}: {result.global_features[key]:.3f}" for key in GLOBAL_SIGNALS
    )
    figure.suptitle(
        "Detección de anomalías en corte cerebral (score híbrido)", fontsize=14
    )
    figure.text(
        0.5,
        0.018,
        f"Señales globales [ {features} ]\n"
        f"global={result.global_score:.3f}  MAE={result.mae_score:.3f}  "
        f"final={result.anomaly_score:.3f}\n"
        "Uso experimental: este resultado no constituye un diagnóstico.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detecta si un corte cerebral se desvía de la normalidad "
        "(score híbrido)."
    )
    parser.add_argument("image", type=Path, help="Ruta del corte cerebral")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "results/brain_hybrid_full/ae_seed42/metrics.json",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/demo_hybrid_resultado.png",
        help="Ruta del PNG generado (por defecto: results/demo_hybrid_resultado.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre la figura además de guardarla (requiere entorno gráfico)",
    )
    args = parser.parse_args()

    if not args.metrics.is_file():
        raise FileNotFoundError(f"No existe metrics.json: {args.metrics}")
    data_root = resolve_data_root(args.data_root)
    calibration = _load_calibration(args.metrics, data_root)
    model_path = args.metrics.parent / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_model(
        calibration["_model_name"], calibration["_bottleneck"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = load_image(args.image, calibration["_image_size"])
    result = evaluate_image(image, model, calibration)
    create_figure(image, result, args.output, args.show)

    print(f"Imagen: {args.image}")
    print(f"Señales globales: {result.global_features}")
    print(f"Score global: {result.global_score:.4f}")
    print(f"Score MAE: {result.mae_score:.4f}")
    print(f"Puntuación final: {result.anomaly_score:.4f}")
    print(f"Umbral de validación: {result.threshold:.4f}")
    print(f"Resultado: {result.label}")
    print(f"Figura: {args.output}")
    print("Uso experimental: este resultado no constituye un diagnóstico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
