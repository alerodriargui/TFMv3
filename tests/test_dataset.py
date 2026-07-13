from pathlib import Path

import numpy as np
from PIL import Image

from tfm_anomaly.dataset import RadiographDataset


def test_labeled_dataset_reads_both_classes(tmp_path: Path):
    for name, value in (("good", 20), ("Ungood", 220)):
        target = tmp_path / name
        target.mkdir()
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(target / "x.png")
    dataset = RadiographDataset.labeled(tmp_path, image_size=64)
    assert len(dataset) == 2
    image, label, _ = dataset[1]
    assert image.shape == (1, 64, 64)
    assert label == 1
