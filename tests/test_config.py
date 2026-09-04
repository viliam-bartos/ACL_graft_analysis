import unittest
import tempfile
import os
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config.settings import PipelineSettings, DEFAULT_SETTINGS


class TestConfig(unittest.TestCase):
    def test_default_settings(self):
        """Verify default settings are populated correctly."""
        settings = PipelineSettings()
        self.assertEqual(settings.mode, "FILE")
        self.assertEqual(settings.base_filters, 64)
        self.assertEqual(settings.target_spacing, (0.5, 0.5, 0.5))
        self.assertTrue(settings.run_inference)

    def test_dict_roundtrip(self):
        """Verify serialization to/from dictionary preserves all fields."""
        settings = PipelineSettings(
            mode="FOLDER",
            base_filters=32,
            target_spacing=(1.0, 1.0, 1.0),
            patch_size=(96, 96, 96),
        )
        d = settings.to_dict()
        restored = PipelineSettings.from_dict(d)

        self.assertEqual(restored.mode, "FOLDER")
        self.assertEqual(restored.base_filters, 32)
        self.assertEqual(restored.target_spacing, (1.0, 1.0, 1.0))
        self.assertEqual(restored.patch_size, (96, 96, 96))

    def test_json_roundtrip(self):
        """Verify saving and loading from JSON file."""
        settings = PipelineSettings(
            mode="FILE",
            log_file="custom_test.log",
            ransac_residual_threshold=2.0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "config.json")
            settings.save_json(json_path)
            self.assertTrue(os.path.exists(json_path))

            loaded = PipelineSettings.load_json(json_path)
            self.assertEqual(loaded.log_file, "custom_test.log")
            self.assertEqual(loaded.ransac_residual_threshold, 2.0)

    def test_validate_paths(self):
        """Path validation returns boolean dict."""
        settings = PipelineSettings()
        checks = settings.validate_paths()
        self.assertIsInstance(checks, dict)


if __name__ == "__main__":
    unittest.main()
