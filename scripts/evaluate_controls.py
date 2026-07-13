"""Evaluate simple one-class intensity controls against the learned models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from tfm_anomaly.dataset import cached_labeled, cached_normal_only
from tfm_anomaly.metrics import best_balanced_threshold, evaluate


FEATURE_NAMES = (
    "mean",
    "std",
    "p10",
    "p90",
    "p90_minus_p10",
    "center_mean",
    "border_mean",
    "center_minus_border",
    "gradient_mean",
)


def features(images: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].float().div(255).squeeze(1)
        flat = batch.flatten(1)
        quantiles = torch.quantile(flat, torch.tensor([0.1, 0.9]), dim=1).T
        center = batch[:, 16:48, 16:48].mean(dim=(1, 2))
        border_mask = torch.ones((64, 64), dtype=torch.bool)
        border_mask[8:56, 8:56] = False
        border = batch[:, border_mask].mean(dim=1)
        gradient = 0.5 * (
            torch.abs(batch[:, :, 1:] - batch[:, :, :-1]).mean(dim=(1, 2))
            + torch.abs(batch[:, 1:, :] - batch[:, :-1, :]).mean(dim=(1, 2))
        )
        rows.append(
            torch.stack(
                (
                    flat.mean(1),
                    flat.std(1),
                    quantiles[:, 0],
                    quantiles[:, 1],
                    quantiles[:, 1] - quantiles[:, 0],
                    center,
                    border,
                    center - border,
                    gradient,
                ),
                dim=1,
            ).numpy()
        )
    return np.concatenate(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/cache/chest_rsna")
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/intensity_controls.json")
    )
    args = parser.parse_args()
    train = cached_normal_only(args.cache_root, "train", args.image_size, None, 42)
    validation = cached_labeled(args.cache_root, "val", args.image_size, None, 42)
    test = cached_labeled(args.cache_root, "test", args.image_size, None, 42)
    train_features = features(train.tensors[train.indices])
    validation_features = features(validation.tensors[validation.indices])
    test_features = features(test.tensors[test.indices])
    validation_labels = np.asarray([validation.labels[index] for index in validation.indices])
    test_labels = np.asarray([test.labels[index] for index in test.indices])
    location = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    validation_z = np.abs((validation_features - location) / scale)
    test_z = np.abs((test_features - location) / scale)

    results = []
    candidates = [(name, validation_z[:, index], test_z[:, index]) for index, name in enumerate(FEATURE_NAMES)]
    candidates.append(
        (
            "diagonal_multivariate",
            np.mean(validation_z**2, axis=1),
            np.mean(test_z**2, axis=1),
        )
    )
    for name, validation_scores, test_scores in candidates:
        threshold, _ = best_balanced_threshold(validation_labels, validation_scores)
        validation_metrics = evaluate(validation_labels, validation_scores, threshold)
        test_metrics = evaluate(test_labels, test_scores, threshold)
        results.append(
            {"control": name, "validation": validation_metrics, "test": test_metrics}
        )
    selected = max(results, key=lambda item: item["validation"]["auroc"])
    report = {
        "protocol": "absolute z-score fitted on normal-only training data",
        "selection": "highest validation AUROC among predefined controls",
        "features": list(FEATURE_NAMES),
        "selected_control": selected["control"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["control", "validation_auroc", "test_auroc", "test_auprc", "test_balanced_accuracy"]
        )
        for item in results:
            writer.writerow(
                [
                    item["control"],
                    item["validation"]["auroc"],
                    item["test"]["auroc"],
                    item["test"]["average_precision"],
                    item["test"]["balanced_accuracy"],
                ]
            )
    print(f"Selected: {selected['control']}")
    for item in results:
        print(
            f"{item['control']}: val_AUROC={item['validation']['auroc']:.4f} "
            f"test_AUROC={item['test']['auroc']:.4f} "
            f"test_BA={item['test']['balanced_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
