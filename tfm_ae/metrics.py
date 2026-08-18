"""Métricas binarias (ranking y umbral) sin depender de scikit-learn."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    """Contadores de la matriz de confusión binaria."""

    tp: int  # Predijo anómalo y era anómalo
    fp: int  # Predijo anómalo pero era normal (falsa alarma)
    tn: int  # Predijo normal y era normal
    fn: int  # Predijo normal pero era anómalo (fallo grave)


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0  # Evita división por cero


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    """Compara los scores con un umbral y cuenta cada celda de la matriz."""
    predicted = scores >= threshold  # Recomendación: 1 si supera el umbral
    positive = y_true == 1  # Es anómalo de verdad
    return Confusion(
        tp=int(np.sum(predicted & positive)),
        fp=int(np.sum(predicted & ~positive)),
        tn=int(np.sum(~predicted & ~positive)),
        fn=int(np.sum(~predicted & positive)),
    )


def threshold_metrics(value: Confusion) -> dict[str, float | int]:
    """Añade métricas derivadas a los contadores de la matriz."""
    sensitivity = _divide(value.tp, value.tp + value.fn)  # % de anómalos detectados
    specificity = _divide(value.tn, value.tn + value.fp)  # % de normales no marcados
    result: dict[str, float | int] = asdict(value)
    result.update(
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2,  # Media no sesgada por clase
    )
    return result


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Probabilidad de que un anómalo puntúe más alto que un normal (Mann-Whitney)."""
    positive = y_true == 1
    n_positive = int(positive.sum())
    n_negative = len(y_true) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUROC requiere ambas clases")
    # Rango de cada score con empates promediados: searchsorted cuenta
    # los < x y los <= x; rango = (menores + menores_o_iguales + 1) / 2
    sorted_scores = np.sort(scores, kind="mergesort")
    less = np.searchsorted(sorted_scores, scores, side="left")
    less_equal = np.searchsorted(sorted_scores, scores, side="right")
    ranks = (less + less_equal + 1) / 2
    rank_sum = float(ranks[positive].sum())  # Suma de rangos de los positivos
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def best_balanced_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Umbral de validación con mejor accuracy balanceada (desempata por specificity)."""
    positives = int(np.sum(y_true == 1))
    negatives = len(y_true) - positives
    if not positives or not negatives:
        raise ValueError("La selección de umbral requiere ambas clases")

    # Recorre scores de mayor a menor acumulando TP/FP de una vez
    order = np.argsort(-scores, kind="mergesort")
    labels = y_true[order]
    sorted_scores = scores[order]
    tp = np.cumsum(labels)
    fp = np.arange(1, len(labels) + 1) - tp
    sensitivity = tp / positives
    specificity = (negatives - fp) / negatives
    balanced = (sensitivity + specificity) / 2

    # Solo evalúa al cerrar cada grupo de empates (el umbral incluye todos los iguales)
    group_ends = np.flatnonzero(np.diff(sorted_scores))
    group_ends = np.append(group_ends, len(labels) - 1)
    balanced_end = balanced[group_ends]
    specificity_end = specificity[group_ends]
    best = int(balanced_end.argmax())
    ties = np.flatnonzero(balanced_end == balanced_end[best])  # Empates a balanced
    best = int(ties[np.argmax(specificity_end[ties])])  # Gana el de mayor specificity
    return float(sorted_scores[group_ends[best]]), float(balanced_end[best])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Métricas completas: confusión, derivadas, AUROC y recuentos."""
    result = threshold_metrics(confusion(y_true, scores, threshold))
    result.update(
        threshold=float(threshold),
        auroc=auroc(y_true, scores),
        samples=int(len(y_true)),
        normal=int(np.sum(y_true == 0)),
        anomalous=int(np.sum(y_true == 1)),
    )
    return result