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
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def best_balanced_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Find the validation threshold in O(n log n), preferring specificity."""
    order = np.argsort(-scores, kind="mergesort")
    labels = y_true[order]
    sorted_scores = scores[order]
    positives = int(np.sum(labels == 1))
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("La selección de umbral requiere ambas clases")

    tp = fp = 0
    best_score = -1.0
    best_specificity = -1.0
    best_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    index = 0
    while index < len(labels):
        value = sorted_scores[index]
        end = index
        while end < len(labels) and sorted_scores[end] == value:
            if labels[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        sensitivity = tp / positives
        specificity = (negatives - fp) / negatives
        balanced = (sensitivity + specificity) / 2
        if balanced > best_score or (
            balanced == best_score and specificity > best_specificity
        ):
            best_score = balanced
            best_specificity = specificity
            best_threshold = float(value)
        index = end
    return best_threshold, best_score


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
