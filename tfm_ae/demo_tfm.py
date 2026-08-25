"""Generate a polished demo figure for the TFM."""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tfm_ae.models import build_model


def load_image(path, image_size):
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def evaluate_image(image, model, threshold):
    with torch.inference_mode():
        reconstructed = model(image)
        absolute_error = torch.abs(reconstructed - image)
        mae = float(torch.mean(absolute_error).item())

    label = "ANOMALA" if mae >= threshold else "NORMAL"

    return {
        "reconstructed": reconstructed,
        "absolute_error": absolute_error,
        "mae": mae,
        "threshold": threshold,
        "label": label,
    }


def create_tfm_figure(image, result, output_path, title_suffix=""):
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#f8f9fa')
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    original = image.squeeze().numpy()
    reconstructed = result["reconstructed"].squeeze().numpy()
    error = result["absolute_error"].squeeze().numpy()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("1. Imagen original", fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("2. Reconstruccion AE", fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 1e-6))
    ax3.set_title("3. Mapa de error (MAE)", fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Error')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(original, cmap='gray', vmin=0, vmax=1)
    mask = error > np.percentile(error, 85)
    overlay = np.zeros((*error.shape, 4))
    overlay[mask, 0] = 1.0
    overlay[mask, 3] = 0.5
    ax4.imshow(overlay)
    ax4.set_title("4. Regiones anomalas", fontweight='bold')
    ax4.axis('off')

    score = result["mae"]
    threshold = result["threshold"]
    is_anomaly = score >= threshold
    result_color = '#e74c3c' if is_anomaly else '#2ecc71'
    result_text = "ANOMALA" if is_anomaly else "NORMAL"

    fig.suptitle(f"Deteccion de anomalias en corte cerebral - TFM{title_suffix}",
                fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01, f"MAE={score:.4f}  Umbral={threshold:.4f}  ->  {result_text}",
            ha='center', fontsize=11, fontweight='bold', color=result_color)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figura guardada: {output_path}")


if __name__ == "__main__":
    metrics_path = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\results\brain_v2_final\ae_seed42\metrics.json")
    data_root = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\data\raw\rsna_bmad\BraTS2021_slice")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(report["test"]["threshold"])
    image_size = int(report["config"]["image_size"])
    bottleneck = int(report["config"].get("bottleneck_channels", 32))

    model_path = metrics_path.parent / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_model(bottleneck)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\results\brain_v2_final")

    demos = [
        ("test/good/img/01221_99.png", "demo_normal"),
        ("test/Ungood/img/00006_60.png", "demo_anomaly"),
    ]

    for img_rel, name in demos:
        img_path = data_root / img_rel
        image = load_image(img_path, image_size)
        result = evaluate_image(image, model, threshold)
        label_tag = "Normal" if result['label'] == "NORMAL" else "Anomalo"
        print(f"{name}: MAE={result['mae']:.4f} -> {result['label']}")
        create_tfm_figure(image, result, output_dir / f"{name}.png", f" - {label_tag} ({img_rel.split('/')[-1]})")
