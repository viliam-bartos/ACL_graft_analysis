"""
Backward-compatibility shim.
The pipeline orchestrator has been organized into Source/pipeline/mri_pipeline.py.
This shim ensures that any existing shortcut or script importing
`Source/main/mri_pipeline.py` continues to work seamlessly.
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pipeline.mri_pipeline import *
from pipeline.mri_pipeline import (
    CONFIG,
    DEFAULT_SETTINGS,
    PipelineSettings,
    main,
    process_single_volume,
    run_visualization_only,
    classify_laterality,
    is_dicom_input,
    convert_dicom_to_nifti,
    _load_ensemble_models,
)

if __name__ == "__main__":
    main()
