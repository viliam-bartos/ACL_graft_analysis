"""
Backward-compatibility shim.
This batch laterality module has been standardized to Source/pipeline/batch_laterality.py.
"""
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pipeline.batch_laterality import *
from pipeline.batch_laterality import main

if __name__ == "__main__":
    main()
