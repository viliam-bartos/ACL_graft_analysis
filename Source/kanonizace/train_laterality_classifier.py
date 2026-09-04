"""
Backward-compatibility shim.
This laterality training module has been standardized to Source/training/train_laterality.py.
"""
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from training.train_laterality import *
from training.train_laterality import main

if __name__ == "__main__":
    main()
