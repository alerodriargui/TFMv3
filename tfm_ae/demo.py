"""Detect one anomalous radiograph with the frozen autoencoder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from . import PROJECT_ROOT
from .data import load_radiograph
from .models import ARCHITECTURE_VERSION, ConvAutoencoder
from .scoring import SCORE_NAME, reconstruction_score


@dataclass(frozen=True)
class DemoResult:
    reconstructed: torch.Tensor
    absolute_error: torch.Tensor
    mae: float
    anomaly_score: float
    threshold: float
    label: str


def evaluate_image(
    image: torch.Tensor,
    model: ConvAutoencoder,
    checkpoint: dict,
) -> DemoResult:
    """Apply the frozen model and reconstruction threshold to one image."""
    with torch.inference_mode():
        score = reconstruction_score(model, image)

    mae = float(score.reconstruction_mae.item())
    anomaly_score = mae
    threshold = float(checkpoint["threshold"])
    label = "ANÓMALA" if anomaly_score >= threshold else "NORMAL"
    return DemoResult(
        reconstructed=score.reconstructed,
        absolute_error=score.absolute_error,
        mae=mae,
        anomaly_score=anomaly_score,
        threshold=threshold,
        label=label,
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
    score_axis.set_title("4. Puntuación frente al umbral")
    score_axis.set_yticks([])
    values = (result.anomaly_score, result.threshold)
    span = max(abs(values[0] - values[1]), abs(values[0]), abs(values[1]), 1.0)
    lower = min(values) - 0.25 * span
    upper = max(values) + 0.25 * span
    score_axis.set_xlim(lower, upper)
    score_axis.set_xlabel("Error de reconstrucción (MAE)")
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

    figure.suptitle("Demostración visual de detección de anomalías", fontsize=15)
    figure.text(
        0.5,
        0.015,
        "Uso experimental: este resultado no constituye un diagnóstico.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detecta si una radiografía se desvía de la normalidad."
    )
    parser.add_argument("image", type=Path, help="Ruta de la radiografía")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/modelo_autoencoder.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/demo_resultado.png",
        help="Ruta del PNG generado (por defecto: results/demo_resultado.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre la figura además de guardarla (requiere entorno gráfico)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo: {args.model}. Ejecuta primero "
            "python -m tfm_ae.train."
        )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE_VERSION:
        raise ValueError(
            "El checkpoint no pertenece a la arquitectura 1024×1024 actual. "
            "Vuelve a entrenar con python -m tfm_ae.train."
        )
    if checkpoint.get("score") != SCORE_NAME:
        raise ValueError(
            "El checkpoint usa una puntuación anterior. "
            "Vuelve a entrenar con python -m tfm_ae.train."
        )
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image = load_radiograph(
        args.image, int(checkpoint["image_size"])
    ).unsqueeze(0)
    result = evaluate_image(image, model, checkpoint)
    create_figure(image, result, args.output, args.show)

    print(f"Imagen: {args.image}")
    print(f"Error de reconstrucción: {result.mae:.4f}")
    print(f"Puntuación de anomalía: {result.anomaly_score:.4f}")
    print(f"Umbral de validación: {result.threshold:.4f}")
    print(f"Resultado: {result.label}")
    print(f"Figura: {args.output}")
    print("Uso experimental: este resultado no constituye un diagnóstico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
