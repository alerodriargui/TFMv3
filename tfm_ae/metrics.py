"""Binary ranking and threshold metrics without a scikit-learn dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    predicted = scores >= threshold
    positive = y_true == 1
    return Confusion(
        tp=int(np.sum(predicted & positive)),
        fp=int(np.sum(predicted & ~positive)),
        tn=int(np.sum(~predicted & ~positive)),
        fn=int(np.sum(~predicted & positive)),
    )


def threshold_metrics(value: Confusion) -> dict[str, float | int]:
    sensitivity = _divide(value.tp, value.tp + value.fn)
    specificity = _divide(value.tn, value.tn + value.fp)
    result: dict[str, float | int] = asdict(value)
    result.update(
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2,
    )
    return result


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = y_true == 1
    n_positive = int(positive.sum())
    n_negative = len(y_true) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUROC requiere ambas clases")
    # Rango promedio de cada score (empates promedian su rango): searchsorted cuenta
    # cuántos scores son < x y cuántos <= x; el rango medio = (menores + menores_o_iguales + 1) / 2
    sorted_scores = np.sort(scores, kind="mergesort")
    less = np.searchsorted(sorted_scores, scores, side="left")
    less_equal = np.searchsorted(sorted_scores, scores, side="right")
    ranks = (less + less_equal + 1) / 2
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def best_balanced_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Find the validation threshold in O(n log n), preferring specificity."""
    positives = int(np.sum(y_true == 1))
    negatives = len(y_true) - positives
    if not positives or not negatives:
        raise ValueError("La selección de umbral requiere ambas clases")

    # Ordena de mayor a menor score y acumula TP/FP en un solo paso vectorizado
    order = np.argsort(-scores, kind="mergesort")
    labels = y_true[order]
    sorted_scores = scores[order]
    tp = np.cumsum(labels)
    fp = np.arange(1, len(labels) + 1) - tp
    sensitivity = tp / positives
    specificity = (negatives - fp) / negatives
    balanced = (sensitivity + specificity) / 2

    # Solo se evalúa al final de cada grupo de empates: el umbral incluye todos los scores == valor
    group_ends = np.flatnonzero(np.diff(sorted_scores))
    group_ends = np.append(group_ends, len(labels) - 1)
    balanced_end = balanced[group_ends]
    specificity_end = specificity[group_ends]
    best = int(balanced_end.argmax())
    ties = np.flatnonzero(balanced_end == balanced_end[best])
    best = int(ties[np.argmax(specificity_end[ties])])
    return float(sorted_scores[group_ends[best]]), float(balanced_end[best])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = threshold_metrics(confusion(y_true, scores, threshold))
    result.update(
        threshold=float(threshold),
        auroc=auroc(y_true, scores),
        samples=int(len(y_true)),
        normal=int(np.sum(y_true == 0)),
        anomalous=int(np.sum(y_true == 1)),
    )
    return result
