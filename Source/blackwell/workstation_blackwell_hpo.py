"""
Backward-compatibility shim.
This HPO module has been standardized to Source/training/hpo_segmentation.py.
"""
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from training.hpo_segmentation import *
from training.hpo_segmentation import main

if __name__ == "__main__":
    main()
