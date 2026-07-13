"""Summarize score distributions and render representative GANomaly errors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels, scores, paths = [], [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
            paths.append(row["path"])
    return np.asarray(labels), np.asarray(scores), paths


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def render_cases(cases: list[dict], output: Path) -> None:
    width, height = 192, 216
    columns = 4
    rows = (len(cases) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, case in enumerate(cases):
        x = (index % columns) * width
        y = (index // columns) * height
        with Image.open(case["path"]) as image:
            image = image.convert("L").resize((192, 192), Image.Resampling.BILINEAR)
            canvas.paste(image.convert("RGB"), (x, y))
        draw.text(
            (x + 3, y + 194),
            f"{case['kind']} score={case['score']:.4f}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    root = Path("artifacts/experiments")
    reports = []
    best_ganomaly: tuple[float, Path, dict] | None = None
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        labels, scores, paths = read_scores(metrics_path.parent / "test_scores.csv")
        item = {
            "model": metrics["config"]["model"],
            "seed": metrics["config"]["seed"],
            "threshold": metrics["test"]["threshold"],
            "normal_scores": distribution(scores[labels == 0]),
            "anomalous_scores": distribution(scores[labels == 1]),
            "median_gap_anomalous_minus_normal": float(
                np.median(scores[labels == 1]) - np.median(scores[labels == 0])
            ),
        }
        reports.append(item)
        auroc = float(metrics["test"]["auroc"])
        if item["model"] == "ganomaly" and (
            best_ganomaly is None or auroc > best_ganomaly[0]
        ):
            best_ganomaly = (auroc, metrics_path, metrics)

    assert best_ganomaly is not None
    _, metrics_path, metrics = best_ganomaly
    labels, scores, paths = read_scores(metrics_path.parent / "test_scores.csv")
    threshold = float(metrics["test"]["threshold"])
    normal_indices = np.flatnonzero((labels == 0) & (scores >= threshold))
    anomaly_indices = np.flatnonzero((labels == 1) & (scores < threshold))
    false_positives = normal_indices[np.argsort(-scores[normal_indices])[:8]]
    false_negatives = anomaly_indices[np.argsort(scores[anomaly_indices])[:8]]
    cases = [
        {"kind": "FP normal", "score": float(scores[index]), "path": paths[index]}
        for index in false_positives
    ] + [
        {"kind": "FN anomalía", "score": float(scores[index]), "path": paths[index]}
        for index in false_negatives
    ]
    output = {
        "score_distributions": reports,
        "qualitative_run": {
            "model": metrics["config"]["model"],
            "seed": metrics["config"]["seed"],
            "auroc": metrics["test"]["auroc"],
            "threshold": threshold,
            "cases": cases,
        },
    }
    json_path = Path("reports/error_analysis.json")
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_cases(cases, Path("reports/error_analysis_ganomaly.png"))
    print(
        f"Qualitative run: ganomaly seed={metrics['config']['seed']} "
        f"AUROC={metrics['test']['auroc']:.4f}"
    )
    for item in reports:
        print(
            f"{item['model']} seed={item['seed']} "
            f"median_gap={item['median_gap_anomalous_minus_normal']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
