"""Detect one anomalous radiograph with the frozen autoencoder."""

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
        description="Detecta si una radiografía se desvía de la normalidad."
    )
    parser.add_argument("image", type=Path, help="Ruta de la radiografía")
    parser.add_argument("--model", type=Path, default=Path("modelo_autoencoder.pt"))
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo: {args.model}. Ejecuta primero run.py."
        )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    bundle = build_model("ae")
    bundle.model.load_state_dict(checkpoint["model_state"])
    bundle.model.eval()
    image = load_image(args.image, int(checkpoint["image_size"]))

    with torch.inference_mode():
        reconstructed = bundle.model(image)
        mae = float(torch.mean(torch.abs(reconstructed - image)).item())
    pixels = image.squeeze(0).squeeze(0)
    center = float(pixels[16:48, 16:48].mean())
    border_mask = torch.ones((64, 64), dtype=torch.bool)
    border_mask[8:56, 8:56] = False
    center_border = center - float(pixels[border_mask].mean())
    calibration = checkpoint["calibration"]
    ae_score = (
        mae * float(calibration["ae_sign"]) - float(calibration["ae_location"])
    ) / float(calibration["ae_scale"])
    center_score = (
        center_border * float(calibration["center_sign"])
        - float(calibration["center_location"])
    ) / float(calibration["center_scale"])
    anomaly_score = 0.5 * ae_score + 0.5 * center_score
    threshold = float(checkpoint["threshold"])
    label = "ANÓMALA" if anomaly_score >= threshold else "NORMAL"

    print(f"Imagen: {args.image}")
    print(f"Error de reconstrucción: {mae:.4f}")
    print(f"Puntuación de anomalía: {anomaly_score:.4f}")
    print(f"Umbral de validación: {threshold:.4f}")
    print(f"Resultado: {label}")
    print("Uso experimental: este resultado no constituye un diagnóstico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
