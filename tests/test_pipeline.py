import unittest
import os
import numpy as np
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from anaknee.main_acl_analysis import run_geometric_analysis_from_mask


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.ref_mask_path = Path(__file__).resolve().parent.parent / "Data" / "reference" / "vysledky_074" / "mask_right_case_074.nii.gz"

    def test_run_geometric_analysis_from_mask_reference(self):
        """End-to-end integration test of rapid geometric analysis on reference mask."""
        if not self.ref_mask_path.exists():
            self.skipTest(f"Reference mask not found at: {self.ref_mask_path}")

        results_dict, mask_arr, spacing, f_cent, t_cent, p_info, vis_data = run_geometric_analysis_from_mask(
            str(self.ref_mask_path)
        )

        # Check required fields
        required_keys = [
            "angle_to_plateau_deg",
            "sagittal_angle_deg",
            "coronal_angle_deg",
            "ATT_mm",
            "Staubli_Tibial_pct",
            "BH_Length_pct",
            "BH_Depth_pct",
            "acl_volume_mm3",
            "min_dist_to_femur_mm",
            "notch_width_mm",
        ]
        for key in required_keys:
            self.assertIn(key, results_dict, f"Missing key {key} in results_dict")

        # Verify realistic biomechanical ranges
        self.assertFalse(np.isnan(results_dict["angle_to_plateau_deg"]))
        self.assertGreater(results_dict["angle_to_plateau_deg"], 20.0)
        self.assertLess(results_dict["angle_to_plateau_deg"], 80.0)

        self.assertFalse(np.isnan(results_dict["acl_volume_mm3"]))
        self.assertGreater(results_dict["acl_volume_mm3"], 100.0)

        # Verify visualization structures
        self.assertIn("plateau_normal", vis_data)
        self.assertIn("plateau_center", vis_data)
        self.assertEqual(len(vis_data["plateau_normal"]), 3)


if __name__ == "__main__":
    unittest.main()
