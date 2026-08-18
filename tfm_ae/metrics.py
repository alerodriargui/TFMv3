"""Métricas binarias (ranking y umbral) sin depender de scikit-learn."""

from __future__ import annotations

from dataclasses import asdict, dataclass  # asdict convierte la dataclass a dict

import numpy as np


@dataclass(frozen=True)  # Estructura inmutable (no se modifica tras crearla)
class Confusion:
    """Contadores de la matriz de confusión binaria."""

    tp: int  # True Positives: anómalos (1) que el modelo marcó como anómalos
    fp: int  # False Positives: normales (0) marcados como anómalos (falsas alarmas)
    tn: int  # True Negatives: normales (0) marcados como normales
    fn: int  # False Negatives: anómalos (1) marcados como normales (fallos graves)


def _divide(numerator: float, denominator: float) -> float:
    # División segura: evita "division by zero" devolviendo 0.0 si el denominador es 0
    return numerator / denominator if denominator else 0.0


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    """Cuenta tp/fp/tn/fn comparando los scores con un umbral de decisión."""
    predicted = scores >= threshold  # Predicción: 1 si el score supera el umbral
    positive = y_true == 1  # Etiqueta real: 1 si es anómalo
    return Confusion(
        tp=int(np.sum(predicted & positive)),  # predijo 1 y es 1
        fp=int(np.sum(predicted & ~positive)),  # predijo 1 pero es 0
        tn=int(np.sum(~predicted & ~positive)),  # predijo 0 y es 0
        fn=int(np.sum(~predicted & positive)),  # predijo 0 pero es 1
    )


def threshold_metrics(value: Confusion) -> dict[str, float | int]:
    """Convierte la matriz de confusión en métricas derivadas (sensibilidad, etc.)."""
    sensitivity = _divide(value.tp, value.tp + value.fn)  # TPR: % de anómalos detectados
    specificity = _divide(value.tn, value.tn + value.fp)  # % de normales no marcados
    result: dict[str, float | int] = asdict(value)  # Copia tp/fp/tn/fn como dict
    result.update(
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2,  # Media de ambos (no sesgada por clase)
    )
    return result


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Área bajo la curva ROC: probabilidad de que un anómalo tenga mayor score que un normal."""
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
    # Suma de rangos de los positivos (estadístico U de Mann-Whitney)
    rank_sum = float(ranks[positive].sum())
    # Fórmula de Mann-Whitney para el AUROC en [0, 1]
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def best_balanced_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Busca en validación el umbral con mejor accuracy balanceada, prefiriendo specificity."""
    positives = int(np.sum(y_true == 1))
    negatives = len(y_true) - positives
    if not positives or not negatives:
        raise ValueError("La selección de umbral requiere ambas clases")

    # Ordena de mayor a menor score y acumula TP/FP en un solo paso vectorizado
    order = np.argsort(-scores, kind="mergesort")
    labels = y_true[order]  # Etiquetas en el mismo orden (descendente de score)
    sorted_scores = scores[order]
    tp = np.cumsum(labels)  # TP acumulados en cada posición
    fp = np.arange(1, len(labels) + 1) - tp  # FP = muestras vistas - TP
    sensitivity = tp / positives  # Porcentaje de anómalos ya vistos
    specificity = (negatives - fp) / negatives  # Porcentaje de normales no marcados
    balanced = (sensitivity + specificity) / 2

    # Solo se evalúa al final de cada grupo de empates: el umbral incluye todos los scores == valor
    group_ends = np.flatnonzero(np.diff(sorted_scores))  # Último índice de cada grupo
    group_ends = np.append(group_ends, len(labels) - 1)  # ... y el último elemento global
    balanced_end = balanced[group_ends]
    specificity_end = specificity[group_ends]
    best = int(balanced_end.argmax())  # Umbral con mayor accuracy balanceada
    ties = np.flatnonzero(balanced_end == balanced_end[best])
    best = int(ties[np.argmax(specificity_end[ties])])  # Desempate: mayor specificity
    return float(sorted_scores[group_ends[best]]), float(balanced_end[best])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Métricas completas de un experimento: confusión, derivadas, AUROC y recuentos."""
    result = threshold_metrics(confusion(y_true, scores, threshold))
    result.update(
        threshold=float(threshold),  # El umbral usado para decidir
        auroc=auroc(y_true, scores),  # Independiente del umbral
        samples=int(len(y_true)),  # Total de muestras evaluadas
        normal=int(np.sum(y_true == 0)),  # Cuántas eran normales
        anomalous=int(np.sum(y_true == 1)),  # Cuántas eran anómalas
    )
    return result
