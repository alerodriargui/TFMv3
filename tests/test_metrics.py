import numpy as np

from tfm_anomaly.metrics import auroc, best_balanced_threshold, evaluate


def test_perfect_ranking_and_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, balanced = best_balanced_threshold(labels, scores)
    result = evaluate(labels, scores, threshold)
    assert auroc(labels, scores) == 1.0
    assert balanced == 1.0
    assert result["balanced_accuracy"] == 1.0


def test_tied_scores_have_half_auroc():
    labels = np.array([0, 1])
    scores = np.array([0.5, 0.5])
    assert auroc(labels, scores) == 0.5
