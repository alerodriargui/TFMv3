"""CLI de entrenamiento y evaluación de autoencoders para anomalías."""

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
    rows = []
    model = "dae"
    for path in sorted(output_root.glob("*_seed*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("scientific_run", False):
            continue
        model = report["config"]["model_name"]
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
    parser = argparse.ArgumentParser(description="Entrena y evalúa autoencoders.")
    parser.add_argument("--model", choices=["dae", "qfae"], default="dae")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/experiments")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images-per-class", type=int)
    parser.add_argument("--encoder-name", type=str, default="vit_large_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--junction-dim", type=int, default=768)
    parser.add_argument("--junction-n-queries", type=int, default=784)
    parser.add_argument("--junction-heads", type=int, default=8)
    parser.add_argument("--decoder-dim", type=int, default=768)
    parser.add_argument("--decoder-depth", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=12)
    parser.add_argument("--dae-base-ch", type=int, default=64)
    parser.add_argument("--noise-sigma", type=float, default=0.4)
    parser.add_argument("--noise-resolution", type=int, default=32)
    args = parser.parse_args()

    lr = args.learning_rate
    if lr is None:
        lr = 1e-3 if args.model == "dae" else 8e-5

    root = resolve_data_root(args.data_root)
    for seed in args.seeds:
        output_dir = args.output_root / f"{args.model}_seed{seed}"
        metrics_path = output_dir / "metrics.json"
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("scientific_run", False):
                previous_config = previous.get("config", {})
                if previous_config.get("image_size") == args.image_size:
                    print(f"SKIP {args.model} seed={seed}: ya está completo")
                    continue
        config = ExperimentConfig(
            data_root=root,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            model_name=args.model,
            learning_rate=lr,
            seed=seed,
            max_train_images=args.max_train_images,
            max_eval_images_per_class=args.max_eval_images_per_class,
            encoder_name=args.encoder_name,
            junction_dim=args.junction_dim,
            junction_n_queries=args.junction_n_queries,
            junction_heads=args.junction_heads,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            dae_base_ch=args.dae_base_ch,
            noise_sigma=args.noise_sigma,
            noise_resolution=args.noise_resolution,
        )
        report = run(config)
        test = report["test"]
        print(
            f"RESULT {args.model} seed={seed}: AUROC={test['auroc']:.4f} "
            f"balanced_accuracy={test['balanced_accuracy']:.4f}",
            flush=True,
        )
    summarize(args.output_root, PROJECT_ROOT / "results/resultados.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
