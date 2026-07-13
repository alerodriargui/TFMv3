from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tfm_anomaly.dataset import RadiographDataset
from tfm_anomaly.metrics import auroc, best_balanced_threshold, evaluate
from tfm_anomaly.models import build_model, per_image_scores


def test_dataset_reads_both_classes(tmp_path: Path):
    for name, value in (("good", 20), ("Ungood", 220)):
        target = tmp_path / name
        target.mkdir()
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(target / "x.png")
    dataset = RadiographDataset.labeled(tmp_path, image_size=64)
    image, label, _path = dataset[1]
    assert len(dataset) == 2
    assert image.shape == (1, 64, 64)
    assert label == 1


def test_models_return_one_score_per_image():
    images = torch.rand(2, 1, 64, 64)
    for name in ("ae", "vae", "ganomaly"):
        bundle = build_model(name, latent_dim=8)
        bundle.model.eval()
        scores, reconstruction = per_image_scores(bundle, images)
        assert scores.shape == (2,)
        assert reconstruction.shape == images.shape
        assert torch.isfinite(scores).all()


def test_perfect_ranking_and_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, balanced = best_balanced_threshold(labels, scores)
    result = evaluate(labels, scores, threshold)
    assert auroc(labels, scores) == 1.0
    assert balanced == 1.0
    assert result["balanced_accuracy"] == 1.0


def test_tied_scores_have_half_auroc():
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
