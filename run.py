"""Train and evaluate one or more models under an identical protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from data import resolve_data_root
from experiment import ExperimentConfig, run


def summarize(output_root: Path, report_path: Path) -> None:
    """Write the individual and aggregated final result tables."""
    previous = {}
    if report_path.is_file():
        with report_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["model"]: row for row in csv.DictReader(handle)}
    rows = []
    for path in sorted(output_root.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("scientific_run", False):
            continue
        test = report["test"]
        rows.append(
            {
                "model": report["config"]["model"],
                "seed": report["config"]["seed"],
                "auroc": test["auroc"],
                "average_precision": test["average_precision"],
                "balanced_accuracy": test["balanced_accuracy"],
                "sensitivity": test["sensitivity"],
                "specificity": test["specificity"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summary = dict(previous)
    for model, model_rows in sorted(grouped.items()):
        item = {"model": model, "runs": len(model_rows)}
        for metric in ("auroc", "average_precision", "balanced_accuracy"):
            values = [float(row[metric]) for row in model_rows]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[model] = item
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        rows = [summary[model] for model in sorted(summary)]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ae", "vae", "ganomaly", "classifier"),
        default=("ae", "vae", "ganomaly", "classifier"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=(13, 42, 73))
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images-per-class", type=int)
    parser.add_argument("--vae-beta", type=float, default=1e-4)
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    for seed in args.seeds:
        for model in args.models:
            output_dir = args.output_root / f"{model}_seed{seed}"
            metrics_path = output_dir / "metrics.json"
            if metrics_path.is_file():
                previous = json.loads(metrics_path.read_text(encoding="utf-8"))
                if previous.get("scientific_run", False):
                    print(f"SKIP {model} seed={seed}: ya está completo")
                    continue
            config = ExperimentConfig(
                model=model,
                data_root=root,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                learning_rate=args.learning_rate,
                seed=seed,
                max_train_images=args.max_train_images,
                max_eval_images_per_class=args.max_eval_images_per_class,
                vae_beta=args.vae_beta,
            )
            report = run(config)
            test = report["test"]
            print(
                f"RESULT {model} seed={seed}: AUROC={test['auroc']:.4f} "
                f"balanced_accuracy={test['balanced_accuracy']:.4f}",
                flush=True,
            )
    summarize(args.output_root, Path("resultados.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
