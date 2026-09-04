import os
import argparse
import torch
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, 
    LoadImage,
    EnsureChannelFirst, 
    Resize, 
    ScaleIntensity,
    EnsureType
)

# Inference configuration
CONFIG = {
    "model_ckpt": r"",
    "spatial_size": (96, 96, 96)
}

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from models.laterality import LateralityClassifier
except ImportError:
    from Source.models.laterality import LateralityClassifier


def main():
    parser = argparse.ArgumentParser(description="Laterality inference. Returns Left/Right.")
    parser.add_argument("--img", type=str, required=True, help="Path to the NIfTI image.")
    args = parser.parse_args()
    
    classifier = LateralityClassifier()
    result = classifier.predict(args.img)
    print(f"Result for {os.path.basename(args.img)}: {result}")

if __name__ == "__main__":
    main()
