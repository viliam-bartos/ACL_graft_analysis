import unittest
import numpy as np
import sys
from pathlib import Path

# Add Source to path
SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from anaknee.geometry.plateau import PlaneModel3D, get_tibial_plateau_plane


class TestPlaneModel3D(unittest.TestCase):
    def test_horizontal_plane_fit(self):
        """Fit a simple horizontal plane at z = 5.0."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, 10, size=50)
        y = rng.uniform(-10, 10, size=50)
        z = np.full(50, 5.0)
        points = np.column_stack([x, y, z])

        model = PlaneModel3D()
        success = model.estimate(points)
        self.assertTrue(success)
        self.assertIsNotNone(model.normal)
        self.assertAlmostEqual(abs(model.normal[2]), 1.0, places=4)
        self.assertAlmostEqual(model.normal[0], 0.0, places=4)
        self.assertAlmostEqual(model.normal[1], 0.0, places=4)

    def test_residuals(self):
        """Test residual distance calculation."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        model = PlaneModel3D()
        model.estimate(points)

        test_points = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 3.5],
        ])
        res = model.residuals(test_points)
        self.assertAlmostEqual(res[0], 0.0, places=4)
        self.assertAlmostEqual(res[1], 3.5, places=4)

    def test_collinear_points_fail_gracefully(self):
        """Collinear points should return False."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        model = PlaneModel3D()
        success = model.estimate(points)
        self.assertFalse(success)


class TestTibialPlateauPlane(unittest.TestCase):
    def test_empty_mask(self):
        """Empty tibia mask returns default fallback."""
        empty_mask = np.zeros((30, 30, 30), dtype=bool)
        normal, centroid, inliers, outliers = get_tibial_plateau_plane(
            empty_mask, spacing=(1.0, 1.0, 1.0)
        )
        self.assertTrue(np.allclose(normal, [0.0, -1.0, 0.0]))
        self.assertTrue(np.allclose(centroid, [0.0, 0.0, 0.0]))
        self.assertIsNone(inliers)
        self.assertIsNone(outliers)

    def test_synthetic_plateau_plane(self):
        """Synthetic tibia column: top plateau at y=10, normal pointing along -Y."""
        mask = np.zeros((40, 40, 40), dtype=bool)
        # Tibia cylinder from y=10 to y=35, radius 8
        z_c, y_c, x_c = 20, 20, 20
        zz, yy, xx = np.ogrid[:40, :40, :40]
        cylinder = ((zz - z_c) ** 2 + (xx - x_c) ** 2 <= 64) & (yy >= 10) & (yy <= 35)
        mask[cylinder] = True

        spacing = (0.5, 0.5, 0.5)
        proximal_femur = (20 * 0.5, 0.0, 20 * 0.5)  # Femur is above (smaller y)

        normal, centroid, inliers, outliers = get_tibial_plateau_plane(
            mask, spacing, proximal_point=proximal_femur, top_fraction=0.25
        )

        self.assertEqual(len(normal), 3)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=4)
        # Normal should point up towards proximal femur (y negative direction)
        self.assertLess(normal[1], 0.0)


if __name__ == "__main__":
    unittest.main()
