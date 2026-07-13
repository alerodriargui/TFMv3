"""Build a lossless uint8 cache of the deterministic 64x64 preprocessing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from tfm_anomaly.dataset import (
    ANOMALY_NAMES,
    NORMAL_NAMES,
    cache_file,
    find_class_dir,
    find_images,
)
from tfm_anomaly.paths import resolve_data_root, split_dir


def encode(paths: list[Path], image_size: int) -> torch.Tensor:
    tensors = torch.empty((len(paths), 1, image_size, image_size), dtype=torch.uint8)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            tensors[index, 0].copy_(torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(image_size, image_size))
        if (index + 1) % 1000 == 0 or index + 1 == len(paths):
            print(f"cache progress={index + 1}/{len(paths)}", flush=True)
    return tensors


def write_cache(
    output: Path,
    paths: list[Path],
    image_size: int,
    split: str,
    label: int,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    images = encode(paths, image_size)
    payload = {
        "format_version": 1,
        "image_size": image_size,
        "split": split,
        "label": label,
        "images": images,
        "paths": [str(path) for path in paths],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {"file": str(output), "images": len(paths), "bytes": output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache/chest_rsna"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    jobs = []
    for split in ("train", "val", "test"):
        base = split_dir(root, split)
        jobs.append((split, 0, find_images(find_class_dir(base, NORMAL_NAMES))))
        if split != "train":
            jobs.append((split, 1, find_images(find_class_dir(base, ANOMALY_NAMES))))

    files = []
    for split, label, paths in jobs:
        output = cache_file(args.cache_root, split, label, args.image_size)
        if output.is_file() and not args.force:
            print(f"SKIP {output}")
            files.append({"file": str(output), "images": len(paths), "bytes": output.stat().st_size})
            continue
        print(f"BUILD split={split} label={label} images={len(paths)}")
        files.append(write_cache(output, paths, args.image_size, split, label))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "image_size": args.image_size,
        "mode": "L",
        "resampling": "PIL.Image.Resampling.BILINEAR",
        "value_storage": "uint8",
        "files": files,
    }
    manifest_path = args.cache_root / f"manifest_{args.image_size}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
