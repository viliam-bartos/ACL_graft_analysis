import unittest
import torch
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from models.unet3d import ResBlock, LightUNet3D


class TestModels(unittest.TestCase):
    def test_resblock_forward(self):
        """Test residual block forward pass dimensions."""
        block = ResBlock(in_c=4, out_c=8)
        x = torch.randn(1, 4, 8, 8, 8)
        out = block(x)
        self.assertEqual(out.shape, (1, 8, 8, 8, 8))

    def test_light_unet3d_forward(self):
        """Test LightUNet3D forward pass produces expected multiclass output."""
        model = LightUNet3D(in_ch=1, out_ch=4, base=8, dropout_rate=0.0)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 4, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
