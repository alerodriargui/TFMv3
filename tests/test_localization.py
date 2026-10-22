import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import patch
from tfm_ae.localization import evaluate_localization
from tfm_ae.localization import load_mask, overlap


class LocalizationTests(unittest.TestCase):
    def test_opaque_alpha_is_not_lesion_and_resize_is_nearest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'label').mkdir()
            rgba = np.zeros((2, 2, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            rgba[0, 0, :3] = 255
            Image.fromarray(rgba).save(root / 'label' / 'case.png')
            mask = load_mask(root / 'img' / 'case.png', 4)
            expected = np.zeros((4, 4), dtype=bool)
            expected[:2, :2] = True
            np.testing.assert_array_equal(mask, expected)

    def test_threshold_uses_validation_only(self):
        def maps(model, root, split, size, batch, device, **kwargs):
            if split == 'valid':
                yield np.array([[.1, .8]]), np.array([[False, True]]), 1, 'valid'
            else:
                yield np.array([[.9, .2]]), np.array([[False, True]]), 1, 'test'
        with tempfile.TemporaryDirectory() as directory:
            with patch('tfm_ae.localization.iter_maps', side_effect=maps), patch('tfm_ae.localization.model_fingerprint', return_value='test'):
                report = evaluate_localization(None, Path(directory), Path(directory), 2, 1, 'cpu')
        self.assertGreater(report['pixel_threshold'], .1)
        self.assertLessEqual(report['pixel_threshold'], .8)
        self.assertEqual(report['splits']['valid']['micro_dice'], 1)
        # Test cannot choose its own ideal threshold.
        self.assertLess(report['splits']['test']['micro_dice'], 1)

    def test_overlap_and_empty_reference_convention(self):
        reference = np.array([1, 1, 0, 0], dtype=bool)
        prediction = np.array([1, 0, 1, 0], dtype=bool)
        metrics = overlap(prediction, reference)
        self.assertEqual(metrics['dice'], .5)
        self.assertAlmostEqual(metrics['iou'], 1/3)
        self.assertEqual(metrics['sensitivity'], .5)
        empty = np.zeros(4, dtype=bool)
        self.assertEqual(overlap(empty, empty)['dice'], 1)
        self.assertIsNone(overlap(prediction, empty)['sensitivity'])
        self.assertEqual(overlap(prediction, empty)['dice'], 0)


if __name__ == '__main__':
    unittest.main()
