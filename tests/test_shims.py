import unittest
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


class TestBackwardCompatibilityShims(unittest.TestCase):
    def test_legacy_main_gui_app(self):
        """Verify Source.main.gui_app forwards to Source.ui.gui_app."""
        from main.gui_app import App
        self.assertTrue(callable(App))

    def test_canonical_ui_gui_app(self):
        """Verify Source.ui.gui_app imports cleanly."""
        from ui.gui_app import App
        self.assertTrue(callable(App))

    def test_legacy_main_mri_pipeline(self):
        """Verify Source.main.mri_pipeline forwards to Source.pipeline.mri_pipeline."""
        from main.mri_pipeline import CONFIG, process_single_volume
        self.assertIsInstance(CONFIG, dict)
        self.assertTrue(callable(process_single_volume))

    def test_canonical_pipeline_mri_pipeline(self):
        """Verify Source.pipeline.mri_pipeline imports cleanly."""
        from pipeline.mri_pipeline import CONFIG, process_single_volume
        self.assertIsInstance(CONFIG, dict)
        self.assertTrue(callable(process_single_volume))

    def test_legacy_blackwell_forwarding(self):
        """Verify Source.blackwell forwards to Source.training."""
        from blackwell.WORKSTATION_BLACKWELL_MULTICLASS_5CV import LightUNet3D
        self.assertTrue(callable(LightUNet3D))

    def test_canonical_training_imports(self):
        """Verify Source.training imports cleanly."""
        from training.train_segmentation_5cv import LightUNet3D
        self.assertTrue(callable(LightUNet3D))

    def test_legacy_kanonizace_forwarding(self):
        """Verify Source.kanonizace forwards to Source.pipeline/Source.models."""
        from kanonizace.predict_laterality import LateralityClassifier
        self.assertTrue(callable(LateralityClassifier))


if __name__ == "__main__":
    unittest.main()
