"""Entrena y evalúa el autoencoder (CLI)."""

from __future__ import annotations

import argparse  # Parser de argumentos por línea de comandos
import csv  # Escribir el CSV resumen de resultados
import json  # Leer los metrics.json de cada semilla
import statistics  # Media de métricas entre semillas
from pathlib import Path

from . import PROJECT_ROOT
from .data import resolve_data_root
from .experiment import ExperimentConfig, run


def summarize(output_root: Path, report_path: Path) -> None:
    """Agrega los resultados de todas las semillas en un CSV con la media."""
    rows = []
    model = "ae"
    # Recorre un metrics.json por semilla (carpeta ae_seed<num>)
    for path in sorted(output_root.glob("ae_seed*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("scientific_run", False):  # Ignora pruebas no científicas (smoke)
            continue
        model = report["config"]["model"]
        rows.append(report["test"])  # Guarda las métricas de test de esa semilla
    if not rows:
        return
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "runs", "auroc_mean", "balanced_accuracy_mean"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": model,
                "runs": len(rows),  # Cuántas semillas entraron en la media
                "auroc_mean": statistics.mean(row["auroc"] for row in rows),
                "balanced_accuracy_mean": statistics.mean(
                    row["balanced_accuracy"] for row in rows
                ),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el autoencoder.")
    parser.add_argument("--data-root", type=Path)  # Opcional: sobreescribe la raíz de datos
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results/experiments"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--model",
        choices=("ae",),
        default="ae",
        help="Arquitectura: ae (autoencoder simple)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(13, 42, 73))  # Semillas a ejecutar
    parser.add_argument("--max-train-images", type=int)  # Límite rápido de entrenamiento
    parser.add_argument("--max-eval-images-per-class", type=int)  # Límite de evaluación
    parser.add_argument("--noise-std", type=float, default=0.0)  # Ruido añadido a la entrada
    parser.add_argument("--bottleneck", type=int, default=32)  # Canales del cuello de botella
    parser.add_argument(
        "--score-mode",
        choices=("hybrid",),
        default="hybrid",
        help="hybrid: señales globales + MAE con peso en validación",
    )
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    for seed in args.seeds:  # Un experimento completo por semilla
        output_dir = args.output_root / f"ae_seed{seed}"
        metrics_path = output_dir / "metrics.json"
        # Reanudación: si la semilla ya terminó con la misma config, se salta
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("scientific_run", False):
                previous_config = previous.get("config", {})
                same_model = previous_config.get("model") == args.model
                same_size = previous_config.get("image_size") == args.image_size
                if same_model and same_size:
                    print(f"SKIP AE seed={seed}: ya está completo")
                    continue
        config = ExperimentConfig(  # Paquete de todos los hiperparámetros
            data_root=root,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            model_name=args.model,
            learning_rate=args.learning_rate,
            seed=seed,
            max_train_images=args.max_train_images,
            max_eval_images_per_class=args.max_eval_images_per_class,
            noise_std=args.noise_std,
            bottleneck_channels=args.bottleneck,
            score_mode=args.score_mode,
        )
        report = run(config)  # Entrena, valida y evalúa; devuelve el informe
        test = report["test"]
        print(
            f"RESULT AE seed={seed}: AUROC={test['auroc']:.4f} "
            f"balanced_accuracy={test['balanced_accuracy']:.4f}",
            flush=True,  # flush=True para ver el progreso en Colab
        )
    summarize(args.output_root, PROJECT_ROOT / "results/resultados.csv")  # CSV final
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
