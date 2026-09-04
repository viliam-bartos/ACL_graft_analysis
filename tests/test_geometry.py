import unittest
import numpy as np
import SimpleITK as sitk
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from anaknee.geometry.metrics import calculate_tortuosity, calculate_att, calculate_staubli_tibial
from anaknee.geometry.orientation import _reorient_to_ria


class TestGeometricMetrics(unittest.TestCase):
    def test_straight_tube_tortuosity(self):
        """A straight vertical tube should have tortuosity = 1.0."""
        mask = np.zeros((30, 30, 30), dtype=bool)
        # Straight line along Y from y=5 to y=25 at z=15, x=15
        mask[15, 5:26, 15] = True
        spacing = (1.0, 1.0, 1.0)
        femur_centroid = (15.0, 5.0, 15.0)
        tibia_centroid = (15.0, 25.0, 15.0)

        tort = calculate_tortuosity(mask, femur_centroid, tibia_centroid, spacing)
        self.assertAlmostEqual(tort, 1.0, places=2)

    def test_curved_tube_tortuosity(self):
        """A curved/wavy shape should have tortuosity > 1.0."""
        mask = np.zeros((40, 40, 40), dtype=bool)
        for y in range(5, 35):
            # Curved trajectory in Z
            z = int(20 + 5 * np.sin((y - 5) / 5.0))
            mask[z, y, 20] = True

        spacing = (1.0, 1.0, 1.0)
        femur_centroid = (20.0, 5.0, 20.0)
        tibia_centroid = (20.0, 34.0, 20.0)

        tort = calculate_tortuosity(mask, femur_centroid, tibia_centroid, spacing)
        self.assertGreater(tort, 1.05)

    def test_tortuosity_nan_handling(self):
        """NaN centroids should return np.nan."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        tort = calculate_tortuosity(mask, (np.nan, 0, 0), (0, 0, 0), (1.0, 1.0, 1.0))
        self.assertTrue(np.isnan(tort))

    def test_reorient_to_ria_shape_and_type(self):
        """Reorienting an image preserves total voxel count, foreground, and pixel type."""
        arr = np.zeros((20, 25, 30), dtype=np.uint8)
        arr[10, 12, 15] = 1
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((0.5, 0.5, 0.5))

        reoriented = _reorient_to_ria(sitk_img)
        reoriented_arr = sitk.GetArrayFromImage(reoriented)
        self.assertEqual(reoriented_arr.size, arr.size)
        self.assertEqual(np.sum(reoriented_arr), 1)
        self.assertEqual(reoriented.GetPixelID(), sitk.sitkUInt8)

        # Idempotency test: already RIA stays unchanged
        re_reoriented = _reorient_to_ria(reoriented)
        re_arr = sitk.GetArrayFromImage(re_reoriented)
        self.assertEqual(re_arr.shape, reoriented_arr.shape)


if __name__ == "__main__":
    unittest.main()
