"""Audit Chest-RSNA structure and basic integrity without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from tfm_anomaly.dataset import ANOMALY_NAMES, NORMAL_NAMES, find_class_dir, find_images
from tfm_anomaly.paths import resolve_data_root, split_dir

EXPECTED = {
    "train": {"normal": 8000, "anomalous": 0},
    "val": {"normal": 70, "anomalous": 1420},
    "test": {"normal": 781, "anomalous": 16413},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit(root: Path, strict: bool, hash_all: bool) -> dict:
    partitions: dict[str, dict] = {}
    seen_hashes: dict[str, str] = {}
    cross_split_duplicates: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        base = split_dir(root, split)
        normal = find_images(find_class_dir(base, NORMAL_NAMES))
        try:
            anomalous = find_images(find_class_dir(base, ANOMALY_NAMES))
        except FileNotFoundError:
            anomalous = []
        all_paths = normal + anomalous
        unreadable: list[str] = []
        size_counts: dict[str, int] = {}
        inspect_paths = all_paths if strict else all_paths[:: max(1, len(all_paths) // 100)]
        for path in inspect_paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    key = f"{image.width}x{image.height}:{image.mode}"
                    size_counts[key] = size_counts.get(key, 0) + 1
            except Exception as error:  # report malformed external data
                unreadable.append(f"{path}: {error}")
        if hash_all:
            for path in all_paths:
                value = digest(path)
                if value in seen_hashes:
                    cross_split_duplicates.append(
                        {"first": seen_hashes[value], "second": str(path)}
                    )
                else:
                    seen_hashes[value] = str(path)
        partitions[split] = {
            "normal": len(normal),
            "anomalous": len(anomalous),
            "total": len(all_paths),
            "inspected": len(inspect_paths),
            "image_properties": size_counts,
            "unreadable": unreadable,
        }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "strict_readability": strict,
        "hash_all": hash_all,
        "partitions": partitions,
        "cross_split_exact_duplicates": cross_split_duplicates,
        "expected_counts_match": all(
            partitions[split][label] == count
            for split, labels in EXPECTED.items()
            for label, count in labels.items()
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--strict", action="store_true", help="Open every image")
    parser.add_argument("--hash-all", action="store_true", help="Hash every image (slow)")
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    report = audit(root, args.strict, args.hash_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["expected_counts_match"]:
        return 1
    if any(item["unreadable"] for item in report["partitions"].values()):
        return 1
    if report["cross_split_exact_duplicates"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
