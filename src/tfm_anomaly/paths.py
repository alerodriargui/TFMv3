"""Dataset path resolution kept separate from experiment logic."""

from __future__ import annotations

import os
from pathlib import Path


def candidate_data_roots() -> list[Path]:
    """Return candidate Chest-RSNA roots in priority order."""
    candidates: list[Path] = []
    if value := os.environ.get("TFM_DATA_ROOT"):
        candidates.append(Path(value))
    candidates.extend(
        [
            Path("data/raw/rsna_bmad/Chest-RSNA"),
            Path("../TFMv2/data/raw/rsna_bmad/Chest-RSNA"),
        ]
    )
    return candidates


def resolve_data_root(explicit: Path | None = None) -> Path:
    """Resolve and validate a Chest-RSNA root."""
    candidates = [explicit] if explicit is not None else candidate_data_roots()
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.expanduser().resolve()
        if (root / "train" / "good").is_dir() and (root / "test").is_dir():
            return root
    rendered = "\n  - ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "No se encontró Chest-RSNA. Rutas comprobadas:\n  - " + rendered
    )


def split_dir(root: Path, split: str) -> Path:
    """Support BMAD's ``val`` and common validation aliases."""
    if split != "val":
        return root / split
    for name in ("val", "valid", "validation"):
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No se encontró la partición de validación en {root}")
