"""Train and evaluate one or more models under an identical protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from tfm_anomaly.experiment import ExperimentConfig, run
from tfm_anomaly.paths import resolve_data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("ae", "vae", "ganomaly"), default=("ae", "vae", "ganomaly"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images-per-class", type=int)
    parser.add_argument("--vae-beta", type=float, default=1e-4)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    for model in args.models:
        config = ExperimentConfig(
            model=model,
            data_root=root,
            output_dir=args.output_root / f"{model}_seed{args.seed}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_train_images=args.max_train_images,
            max_eval_images_per_class=args.max_eval_images_per_class,
            vae_beta=args.vae_beta,
            cache_root=args.cache_root,
        )
        report = run(config)
        test = report["test"]
        print(
            f"RESULT {model}: AUROC={test['auroc']:.4f} "
            f"AUPRC={test['average_precision']:.4f} "
            f"balanced_accuracy={test['balanced_accuracy']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
