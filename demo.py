"""Classify one radiograph with the frozen supervised reference model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import build_model


def load_image(path: Path, image_size: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clasifica una radiografía como normal o anómala."
    )
    parser.add_argument("image", type=Path, help="Ruta de la radiografía")
    parser.add_argument("--model", type=Path, default=Path("modelo_clasificador.pt"))
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo: {args.model}. Ejecuta primero run.py."
        )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    bundle = build_model("classifier")
    bundle.model.load_state_dict(checkpoint["model_state"])
    bundle.model.eval()
    image = load_image(args.image, int(checkpoint["image_size"]))

    with torch.inference_mode():
        probability = float(torch.sigmoid(bundle.model(image)).item())
    threshold = float(checkpoint["threshold"])
    label = "ANÓMALA" if probability >= threshold else "NORMAL"

    print(f"Imagen: {args.image}")
    print(f"Probabilidad de anomalía: {probability:.4f}")
    print(f"Umbral de validación: {threshold:.4f}")
    print(f"Resultado: {label}")
    print("Uso experimental: este resultado no constituye un diagnóstico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
