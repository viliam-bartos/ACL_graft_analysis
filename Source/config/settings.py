"""
Centralized Configuration Management for ACL Graft Analysis Pipeline
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
from typing import Dict, Any, Tuple


@dataclass
class PipelineSettings:
    """
    Configuration settings for preprocessing, neural network inference,
    anatomical geometry analysis, and visualization.
    """
    mode: str = "FILE"  # "FILE" or "FOLDER"
    input_path: str = r"Data\reference\right_case_074.nii.gz"
    input_dir: str = ""
    output_dir: str = r"Data\reference\vysledky_074"
    log_file: str = "pipeline.log"
    anaknee_ref_mri: str = r"Data\reference\right_case_074.nii.gz"
    gt_masks_dir: str = ""
    model_ckpt: str = ""
    patch_size: Tuple[int, int, int] = (128, 128, 80)
    base_filters: int = 64
    use_ensemble: bool = True
    ensemble_dir: str = r"Data\5CV"
    ensemble_pattern: str = "best_model_fold_*.pth"
    run_inference: bool = True
    run_segmentation_analysis: bool = False
    run_anatomical_analysis: bool = True
    target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ransac_residual_threshold: float = 1.5
    ransac_max_trials: int = 500
    post_proc_classes: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        1: {"lcc": True, "hole_filling": False, "closing": False},
        2: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
        3: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        d = asdict(self)
        # Ensure int keys for post_proc_classes if needed
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineSettings":
        """Instantiate PipelineSettings from dictionary with type safety."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {}
        for k, v in data.items():
            if k in valid_fields:
                if k == "patch_size" and isinstance(v, (list, tuple)):
                    filtered[k] = tuple(int(x) for x in v)
                elif k == "target_spacing" and isinstance(v, (list, tuple)):
                    filtered[k] = tuple(float(x) for x in v)
                elif k == "post_proc_classes" and isinstance(v, dict):
                    # Convert str keys back to int
                    filtered[k] = {int(class_id): conf for class_id, conf in v.items()}
                else:
                    filtered[k] = v
        return cls(**filtered)

    def save_json(self, file_path: str) -> None:
        """Serialize configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, file_path: str) -> "PipelineSettings":
        """Load configuration from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate existence of specified filesystem paths.
        
        Returns:
            dict mapping path attribute names to whether the file/directory exists.
        """
        checks = {}
        if self.mode == "FILE" and self.input_path:
            checks["input_path"] = os.path.exists(self.input_path)
        if self.mode == "FOLDER" and self.input_dir:
            checks["input_dir"] = os.path.exists(self.input_dir)
        if self.anaknee_ref_mri:
            checks["anaknee_ref_mri"] = os.path.exists(self.anaknee_ref_mri)
        if self.model_ckpt:
            checks["model_ckpt"] = os.path.exists(self.model_ckpt)
        if self.use_ensemble and self.ensemble_dir:
            checks["ensemble_dir"] = os.path.exists(self.ensemble_dir)
        if self.run_segmentation_analysis and self.gt_masks_dir:
            checks["gt_masks_dir"] = os.path.exists(self.gt_masks_dir)
        return checks


# Default instance
DEFAULT_SETTINGS = PipelineSettings()
