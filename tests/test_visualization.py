import unittest
import numpy as np
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import pyvista as pv
# Set PyVista to off-screen mode so tests never pop up GUI windows
pv.OFF_SCREEN = True


class TestVisualizationHelpers(unittest.TestCase):
    def test_pyvista_offscreen_setup(self):
        """Verify PyVista plotter initializes headlessly for tests."""
        plotter = pv.Plotter(off_screen=True)
        sphere = pv.Sphere()
        plotter.add_mesh(sphere)
        plotter.close()

    def test_visualizer_import(self):
        """Verify anaknee visualizer module imports cleanly."""
        from anaknee.visualizator_analyzator import smart_visualize
        self.assertTrue(callable(smart_visualize))


if __name__ == "__main__":
    unittest.main()
