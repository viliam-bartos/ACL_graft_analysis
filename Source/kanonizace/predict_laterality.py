"""
Backward-compatibility shim.
This laterality inference module has been standardized to Source/pipeline/laterality_inference.py.
"""
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pipeline.laterality_inference import *
from pipeline.laterality_inference import main

if __name__ == "__main__":
    main()
