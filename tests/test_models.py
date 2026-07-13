import torch

from tfm_anomaly.models import build_model, per_image_scores


def test_all_models_produce_one_score_per_image():
    images = torch.rand(2, 1, 64, 64)
    for name in ("ae", "vae", "ganomaly"):
        bundle = build_model(name, latent_dim=8)
        bundle.model.eval()
        scores, reconstructed = per_image_scores(bundle, images)
        assert scores.shape == (2,)
        assert reconstructed.shape == images.shape
        assert torch.isfinite(scores).all()
