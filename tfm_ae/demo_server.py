"""Servidor local para la demo web del checkpoint DAE congelado."""

from __future__ import annotations

import argparse
import base64
import io
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import PROJECT_ROOT
from .dae import DAE

DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "dae_seed42" / "model.pt"
WEB_ROOT = PROJECT_ROOT / "web"
THRESHOLD = 0.1861136555671692
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FrozenDAE:
    """Carga una vez el checkpoint y expone inferencia sin gradientes."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.image_size = int(checkpoint.get("image_size", 128))
        self.model = DAE(base_ch=int(checkpoint.get("dae_base_ch", 64)))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.inference_mode()
    def reconstruct(self, image_bytes: bytes) -> dict[str, object]:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(source, dtype=np.float32) / 255.0

        image = torch.from_numpy(pixels.copy()).unsqueeze(0).unsqueeze(0)
        reconstruction = self.model(image)
        foreground = F.avg_pool2d((image > 0.01).float(), 5, stride=1, padding=2) > 0.95
        error = (reconstruction - image).abs() * foreground
        patches = F.pad(error, (2, 2, 2, 2), mode="reflect")
        patches = patches.unfold(2, 5, 1).unfold(3, 5, 1)
        filtered = patches.contiguous().view(*error.shape, 25).median(-1).values
        anomaly_map = filtered.amax(dim=1)[0]
        score = float(anomaly_map.amax())

        masked_reconstruction = reconstruction[0, 0] * (image[0, 0] > 0.01)
        input_png = (image[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        recon_png = (masked_reconstruction.clamp(0, 1).numpy() * 255).astype(np.uint8)
        heatmap_png = self._colorize(anomaly_map.numpy(), input_png)
        return {
            "input": _png_base64(input_png),
            "reconstruction": _png_base64(recon_png),
            "heatmap": _png_base64(heatmap_png),
            "score": score,
            "threshold": THRESHOLD,
            "is_anomaly": score >= THRESHOLD,
        }

    @staticmethod
    def _colorize(error: np.ndarray, background: np.ndarray) -> np.ndarray:
        scale = np.clip(error / 0.45, 0, 1)
        stops = np.array(
            [[0, 15, 28], [0, 184, 180], [247, 225, 74], [255, 70, 35]],
            dtype=np.float32,
        )
        position = scale * (len(stops) - 1)
        lower = np.floor(position).astype(np.int32)
        upper = np.minimum(lower + 1, len(stops) - 1)
        weight = (position - lower)[..., None]
        colors = stops[lower] * (1 - weight) + stops[upper] * weight
        gray = np.repeat(background[..., None], 3, axis=2) * 0.2
        alpha = np.clip(scale[..., None] * 1.8, 0.12, 0.95)
        composed = gray * (1 - alpha) + colors * alpha
        composed[background == 0] = 0
        return composed.clip(0, 255).astype(np.uint8)


class DemoHandler(SimpleHTTPRequestHandler):
    model: FrozenDAE

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self.path = "/demo.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/reconstruct":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("La imagen está vacía o supera 10 MB.")
            payload = self.model.reconstruct(self.rfile.read(length))
            self._json(HTTPStatus.OK, payload)
        except (ValueError, OSError, RuntimeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"Imagen no válida: {error}"})

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo web del DAE congelado")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"No se encuentra el checkpoint: {args.checkpoint}")
    DemoHandler.model = FrozenDAE(args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"DAE Lab disponible en http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
