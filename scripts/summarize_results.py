"""Create a compact Markdown/CSV table from completed scientific runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.csv"))
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not args.include_smoke and not report.get("scientific_run", False):
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
                "f1": test["f1"],
                "parameters": report["parameter_count"] + report["discriminator_parameter_count"],
                "elapsed_seconds": report["elapsed_seconds"],
                "scientific_run": report["scientific_run"],
            }
        )
    if not rows:
        print("No hay ejecuciones que cumplan el filtro.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("| Modelo | Semilla | AUROC | AUPRC | Bal. acc. | Sens. | Esp. |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['seed']} | {row['auroc']:.4f} | "
            f"{row['average_precision']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} |"
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summary_rows = []
    for model, model_rows in sorted(grouped.items()):
        summary = {"model": model, "runs": len(model_rows)}
        for metric in ("auroc", "average_precision", "balanced_accuracy", "elapsed_seconds"):
            values = [float(row[metric]) for row in model_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Resumen agregado: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
