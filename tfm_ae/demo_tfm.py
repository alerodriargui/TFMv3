"""Generate a polished demo figure for the TFM."""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from tfm_ae.experiment import load_calibration_from_metrics as _load_calibration
from tfm_ae.features import GLOBAL_SIGNALS, feature_matrix
from tfm_ae.models import build_model


def load_image(path, image_size):
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def evaluate_image(image, model, calibration):
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
    mae_score = (mae * ae_sign - float(calibration["ae_location"])) / float(calibration["ae_scale"])
    weight = float(calibration["weight"])
    anomaly_score = weight * global_score + (1 - weight) * mae_score
    threshold = calibration["_threshold"]
    label = "ANOMALA" if anomaly_score >= threshold else "NORMAL"

    return {
        "reconstructed": reconstructed,
        "absolute_error": absolute_error,
        "mae": mae,
        "features": features,
        "global_score": global_score,
        "mae_score": mae_score,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "label": label,
    }


def create_tfm_figure(image, result, output_path, title_suffix=""):
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('#f8f9fa')
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    original = image.squeeze().numpy()
    reconstructed = result["reconstructed"].squeeze().numpy()
    error = result["absolute_error"].squeeze().numpy()

    # Row 1: Original | Reconstruction | Error heatmap | Difference
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
    ax3.set_title("3. Mapa de error", fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Error')

    # Overlay: original + error highlight
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(original, cmap='gray', vmin=0, vmax=1)
    mask = error > np.percentile(error, 85)
    overlay = np.zeros((*error.shape, 4))
    overlay[mask, 0] = 1.0
    overlay[mask, 3] = 0.5
    ax4.imshow(overlay)
    ax4.set_title("4. Regiones anomalas", fontweight='bold')
    ax4.axis('off')

    # Row 2: Feature bars + Score gauge
    ax5 = fig.add_subplot(gs[1, :2])
    feature_names = list(GLOBAL_SIGNALS)
    feature_values = [result["features"][k] for k in feature_names]
    colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in feature_values]
    bars = ax5.barh(feature_names, feature_values, color=colors, edgecolor='white', height=0.6)
    ax5.set_title("5. Senales globales de la imagen", fontweight='bold')
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel("Valor")

    # Score gauge
    ax6 = fig.add_subplot(gs[1, 2:])
    score = result["anomaly_score"]
    threshold = result["threshold"]
    is_anomaly = score >= threshold

    gauge_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bounds = [0, threshold * 0.7, threshold * 1.3, max(score * 1.2, threshold * 2)]
    result_color = '#e74c3c' if is_anomaly else '#2ecc71'
    result_text = "ANOMALA" if is_anomaly else "NORMAL"

    for i in range(3):
        ax6.barh(0, bounds[i+1] - bounds[i], left=bounds[i], color=gauge_colors[i], 
                height=0.4, alpha=0.7, edgecolor='white')
    ax6.scatter(score, 0, s=200, c='black', zorder=5, marker='v')
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral={threshold:.2f}')
    ax6.set_xlim(bounds[0], bounds[-1])
    ax6.set_ylim(-0.5, 0.5)
    ax6.set_title(f"6. Score: {score:.3f}  ->  {result_text}", fontweight='bold', color=result_color)
    ax6.set_yticks([])
    ax6.legend(loc='upper right', fontsize=10)

    fig.suptitle(f"Deteccion de anomalias en corte cerebral - TFM{title_suffix}", 
                fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01, "Uso experimental: este resultado no constituye un diagnostico.", 
            ha='center', fontsize=9, fontstyle='italic', color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figura guardada: {output_path}")


if __name__ == "__main__":
    import sys
    metrics_path = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\results\brain_v2_final\ae_seed42\metrics.json")
    data_root = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\data\raw\rsna_bmad\BraTS2021_slice")
    
    calibration = _load_calibration(metrics_path, data_root)
    model_path = metrics_path.parent / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_model(calibration["_bottleneck"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\results\brain_v2_final")

    demos = [
        ("test/good/img/01221_99.png", "demo_normal"),
        ("test/Ungood/img/00006_60.png", "demo_anomaly"),
    ]

    for img_rel, name in demos:
        img_path = data_root / img_rel
        image = load_image(img_path, calibration["_image_size"])
        result = evaluate_image(image, model, calibration)
        label_tag = "Normal" if result['label'] == "NORMAL" else "Anomalo"
        print(f"{name}: score={result['anomaly_score']:.3f} -> {result['label']}")
        create_tfm_figure(image, result, output_dir / f"{name}.png", f" - {label_tag} ({img_rel.split('/')[-1]})")
