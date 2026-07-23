"""Generate the final autoencoder architecture diagram from ``models.py``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import torch
from torch import nn

from models import ConvAutoencoder


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts"


def architecture() -> tuple[list[tuple[int, int, int]], list[str], int]:
    """Return tensor shapes and operation labels, checking the live model."""
    model = ConvAutoencoder().eval()
    convolution_outputs: list[tuple[int, int, int]] = []
    hooks = []

    def record_shape(
        _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        convolution_outputs.append(tuple(output.shape[1:]))

    convolutions = [
        module
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    for module in convolutions:
        hooks.append(module.register_forward_hook(record_shape))

    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 64, 64))
    for hook in hooks:
        hook.remove()

    shapes = [(1, 64, 64), *convolution_outputs]
    operations = []
    for index, layer in enumerate(convolutions):
        name = "Conv2d" if isinstance(layer, nn.Conv2d) else "ConvTranspose"
        activation = "Sigmoide" if index == len(convolutions) - 1 else "ReLU"
        operations.append(
            f"{name} {layer.kernel_size[0]}×{layer.kernel_size[0]}\n"
            f"s{layer.stride[0]} · p{layer.padding[0]} + {activation}"
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert shapes == [
        (1, 64, 64),
        (8, 32, 32),
        (16, 16, 16),
        (32, 8, 8),
        (16, 16, 16),
        (8, 32, 32),
        (1, 64, 64),
    ]
    assert output.shape == (1, 1, 64, 64)
    assert parameters == 16_281
    return shapes, operations, parameters


def draw() -> None:
    shapes, operations, parameters = architecture()
    OUTPUT_DIR.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(12.5, 4.6))
    figure.patch.set_facecolor("white")
    axis.set_xlim(-0.7, 15.7)
    axis.set_ylim(-0.35, 5.25)
    axis.axis("off")

    x_positions = [0.0, 2.45, 4.7, 7.15, 9.6, 11.85, 14.3]
    heights = [2.65, 2.2, 1.78, 1.42, 1.78, 2.2, 2.65]
    colors = [
        "#F4F6F8",
        "#DCEAF5",
        "#C8DEEF",
        "#F3D59B",
        "#D9EBDD",
        "#C8E2CE",
        "#F4F6F8",
    ]
    titles = [
        "Entrada",
        "Codificador",
        "Codificador",
        "Latente",
        "Decodificador",
        "Decodificador",
        "Reconstrucción",
    ]

    axis.add_patch(
        Rectangle(
            (1.62, 0.05),
            5.95,
            4.55,
            facecolor="#EAF2F8",
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (8.33, 0.05),
            4.35,
            4.55,
            facecolor="#EDF6EF",
            edgecolor="none",
            zorder=0,
        )
    )
    axis.text(
        4.6,
        4.38,
        "CODIFICADOR",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#285F85",
    )
    axis.text(
        10.5,
        4.38,
        "DECODIFICADOR",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#357047",
    )

    block_width = 1.25
    for index, ((channels, height, width), x, block_height, color, title) in enumerate(
        zip(shapes, x_positions, heights, colors, titles)
    ):
        y = 2.22 - block_height / 2
        edge = "#9A6A14" if index == 3 else "#405261"
        axis.add_patch(
            FancyBboxPatch(
                (x - block_width / 2, y),
                block_width,
                block_height,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=2.1 if index == 3 else 1.35,
                edgecolor=edge,
                facecolor=color,
                zorder=3,
            )
        )
        axis.text(
            x, 2.45, title, ha="center", va="center", fontsize=10.2, weight="bold"
        )
        axis.text(
            x,
            2.05,
            f"{channels} × {height} × {width}",
            ha="center",
            va="center",
            fontsize=11.2,
            weight="bold",
            color="#152935",
        )
        if index in (0, 6):
            axis.text(
                x,
                1.65,
                "canal × alto × ancho",
                ha="center",
                va="center",
                fontsize=7.6,
                color="#52636D",
            )

    for index, operation in enumerate(operations):
        start = x_positions[index] + block_width / 2 + 0.07
        end = x_positions[index + 1] - block_width / 2 - 0.07
        axis.add_patch(
            FancyArrowPatch(
                (start, 2.22),
                (end, 2.22),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.35,
                color="#344955",
                zorder=5,
            )
        )
        axis.text(
            (start + end) / 2,
            3.31,
            operation,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#273841",
            linespacing=1.2,
        )

    axis.text(
        7.5,
        4.91,
        "Autoencoder convolucional final",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#173A54",
    )
    summary = (
        "Compresión espacial 64 → 32 → 16 → 8   ·   "
        "Reconstrucción 8 → 16 → 32 → 64   ·   "
        f"{parameters:,} parámetros"
    ).replace(",", ".")
    axis.text(
        7.5,
        0.0,
        summary,
        ha="center",
        va="bottom",
        fontsize=10.7,
        weight="bold",
        color="#253B48",
    )

    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    metadata = {
        "Creator": "generate_architecture_diagram.py",
        "Title": "Arquitectura del autoencoder final",
    }
    for extension in ("svg", "pdf", "png"):
        target = OUTPUT_DIR / f"autoencoder_architecture.{extension}"
        options = {"dpi": 240} if extension == "png" else {}
        figure.savefig(
            target, bbox_inches="tight", facecolor="white", metadata=metadata, **options
        )
    plt.close(figure)


if __name__ == "__main__":
    draw()
