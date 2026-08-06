"""Train and evaluate the autoencoder."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from . import PROJECT_ROOT
from .data import resolve_data_root
from .experiment import ExperimentConfig, run


def summarize(output_root: Path, report_path: Path) -> None:
    """Write the aggregated AE result."""
    rows = []
    model = "ae"
    for path in sorted(output_root.glob("ae_seed*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("scientific_run", False):
            continue
        model = report["config"]["model"]
        rows.append(report["test"])
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
                "runs": len(rows),
                "auroc_mean": statistics.mean(row["auroc"] for row in rows),
                "balanced_accuracy_mean": statistics.mean(
                    row["balanced_accuracy"] for row in rows
                ),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el autoencoder.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results/experiments"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--model",
        choices=("ae", "unet"),
        default="ae",
        help="Arquitectura: ae (autoencoder simple) o unet (U-Net con skips)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(13, 42, 73))
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images-per-class", type=int)
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    for seed in args.seeds:
        output_dir = args.output_root / f"ae_seed{seed}"
        metrics_path = output_dir / "metrics.json"
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("scientific_run", False):
                previous_config = previous.get("config", {})
                same_model = previous_config.get("model") == args.model
                same_size = previous_config.get("image_size") == args.image_size
                if same_model and same_size:
                    print(f"SKIP AE seed={seed}: ya está completo")
                    continue
        config = ExperimentConfig(
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
        )
        report = run(config)
        test = report["test"]
        print(
            f"RESULT AE seed={seed}: AUROC={test['auroc']:.4f} "
            f"balanced_accuracy={test['balanced_accuracy']:.4f}",
            flush=True,
        )
    summarize(args.output_root, PROJECT_ROOT / "results/resultados.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
