# Automated ACL Segmentation and 3D Geometric Analysis: Project Documentation

This document provides a comprehensive technical overview and developer guide for the **Automated ACL Segmentation and 3D Geometric Analysis** codebase (project **2509-MRI-Knee**). The software implements a medical imaging pipeline to segment the Anterior Cruciate Ligament (ACL), Femur, and Tibia from isotropic 3D knee MRI scans and performs automated 3D knee joint geometry measurements.

---

## 1. Project Overview

The project is developed as part of a Diploma Thesis at CEITEC/CTLAB under the supervision of **Ing. Jakub Lázňovský, Ph.D.** 

The goal of this system is to automate the diagnostic pipeline for assessing ACL health, graft positioning, and potential impingement post-surgery. It combines deep learning-based semantic segmentation with classical 3D graphics algorithms (e.g., surface reconstruction, Principal Component Analysis/Singular Value Decomposition plane fitting, and ray casting) to estimate geometric clinical markers.

---

## 2. Directory Structure

```
2509-MRI-Knee/
├── Data/                             # Input/output data directories, checkpoints, reference scans
│   ├── 5CV/                          # 5-Fold Cross-Validation checkpoints
│   ├── reference/                    # Reference scans for Nyul-Udupa histogram normalization
│   └── Data_analyza/                 # Prepared datasets for validation and pipeline testing
├── Documentation/                    # Project documentation, guides, and graphics
│   ├── DevOps_guidelines.md          # Git and branching strategy guide (visual layout SVG)
│   ├── Supplement.md                 # Supplementary development and git glossary documentation
│   ├── Project_Documentation.md      # This file (main developer documentation)
│   └── requirements.txt              # Required packages for environment setup
├── Source/                           # Code base directories
│   ├── anaknee/                      # Anatomy and knee joint geometric analysis modules
│   │   ├── analysis_wrapper.py       # Wrapper for invoking the analysis module
│   │   ├── main_acl_analysis.py      # Core calculations (Footprints, Plane, ATT, Stäubli, Radiomics)
│   │   └── visualizator_analyzator.py# PyVista-based interactive 3D visualizer
│   ├── blackwell/                    # 3D U-Net model definition and training scripts
│   │   ├── WORKSTATION_BLACKWELL_MULTICLASS_5CV.py  # 5-fold cross-validation training entrypoint
│   │   └── workstation_blackwell_hpo.py            # Optuna-based Hyperparameter Optimization (HPO)
│   ├── kanonizace/                   # Lateralization classification scripts
│   │   ├── predict_laterality.py     # Inference module for predicting knee laterality
│   │   ├── predict_laterality_all.py # Batch script for predicting laterality
│   │   └── train_laterality_classifier.py  # Training script for ResNet-18 laterality classifier
│   └── main/                         # Pipeline driver
│       └── mri_pipeline.py           # Combined end-to-end processing pipeline
├── results_blackwell_cv/             # Fold metrics, configs, and training curves output folder
└── README.md                         # Project overview and quickstart links
```

---

## 3. End-to-End Pipeline Architecture (`mri_pipeline.py`)

The main entry point for processing volumes is [mri_pipeline.py](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Source/main/mri_pipeline.py). It operates in either `FILE` mode (processing a single NIfTI file) or `FOLDER` mode (batch-processing all `.nii`/`.nii.gz` files in a directory). 

The pipeline consists of the following modular steps:

```
[Input MRI Scan]
       │
       ▼
1. Resampling ─────────► Resample to 0.5 mm isotropic resolution (BSpline interpolation)
       │
       ▼
2. Orientation PIL ────► Reorient orientation matrix to PIL (Posterior-Inferior-Left)
       │
       ▼
3. Canonization ───────► Predict laterality (ResNet-18). If "Right", mirror along Axis 0 to "Left" space.
       │
       ▼
4. Model Inference ────► Segment using 3D U-Net (Single model or 5-Fold Ensemble averaging)
       │
       ▼
5. Post-processing ────► Apply morphological operations (LCC filter, hole filling, closing)
       │
       ▼
6. Inverse Transform ──► Un-mirror if needed, and resample mask back to original input spacing
       │
       ▼
7. Segmental Metrics ──► Compute Dice Similarity & HD95 metrics vs. GT (if validation labels exist)
       │
       ▼
8. Anatomical Analysis ► Execute "Anaknee" pipeline (ATT, Stäubli index, Tortuosity, Radiomics, etc.)
       │
       ▼
9. 3D Visualization ───► Launch PyVista 3D interactive viewer for verification
```

### Key Modules in `mri_pipeline.py`
*   **Resampling (`run_resampling`)**: Standardizes input voxel sizes to `(0.5, 0.5, 0.5)` mm using B-spline interpolation for intensity preservation.
*   **Reorientation (`run_orientation`)**: Ensures the volume is in `PIL` (Posterior-Inferior-Left) configuration using `nibabel.orientations`.
*   **Canonization (`run_canonization`)**: Employs a pre-trained [LateralityClassifier](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Source/kanonizace/predict_laterality.py) to detect left/right knee joint. Right joints are automatically mirrored along the first axis (`axis=0`) so the segmentation model only processes canonical "Left" structures, drastically simplifying feature learning.
*   **Inference (`run_inference`)**: Feeds the preprocessed volume to a lightweight 3D U-Net (`LightUNet3D`).
    *   *Single-model mode*: Loads checkpoint path specified in `model_ckpt`.
    *   *Ensemble mode*: Activated via `use_ensemble: True`. It loads weight checkpoints matching `best_model_fold_*.pth` in the `ensemble_dir` and averages the softmax probability channels over all folds before thresholding.
*   **Post-processing (`run_postprocessing`)**: Refines masks class-by-class using customized options (Largest Connected Component to remove isolated noise, hole filling, and morphological closing with custom kernel iteration).
*   **Inverse Transformation (`run_inverse_transform`)**: Restores the processed segmentations to the physical coordinate space and voxel spacing of the raw input MRI. Right knees are un-mirrored.

---

## 4. Anatomical and Geometric Analysis (`main_acl_analysis.py`)

The anatomical analysis pipeline is contained in [main_acl_analysis.py](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Source/anaknee/main_acl_analysis.py) (sometimes referred to as the **Anaknee** module). It extracts various geometric descriptors of the joint:

### Module 1: Histogram Matching
Normalizes intensity distributions of the input volume against a reference scan (`case_074.nii.gz` by default) using Nyul-Udupa histogram standardization via `TorchIO`. This reduces scanner-specific intensity drift before radiomics extraction.

### Module 2: Footprint Centroid Extraction
Identifies the insertion sites (footprints) of the ACL on both bones:
1. Dilates the segmented ACL mask (Label 1) slightly (2 voxels kernel).
2. Intersects this dilated mask with the Femur mask (Label 2) to extract the **Femoral Footprint**.
3. Intersects it with the Tibia mask (Label 3) to extract the **Tibial Footprint**.
4. Computes the 3D physical coordinate centroids of these contact zones.

### Module 3: Tibial Plateau Plane & ACL Orientation Angles
Estimates the orientation of the ACL vector relative to the tibia:
*   **Tibial Plateau Fitting**: Uses PCA (via Singular Value Decomposition) on the coordinates of the proximal (top) tibia voxels to fit a 3D plane. The unit normal vector $\vec{n}_{plateau}$ and its physical centroid are computed.
*   **ACL Vector**: Calculated as the vector connecting the tibial footprint centroid to the femoral footprint centroid.
*   **Angles Calculated**:
    *   *Elevation angle to plateau*: The angle of the ACL vector relative to the tibial plateau plane.
    *   *Sagittal angle*: Projection of the ACL vector onto the Anterior-Superior anatomical plane relative to the plateau normal.
    *   *Coronal angle*: Projection of the ACL vector onto the Right-Superior anatomical plane relative to the plateau normal.

### Module 4: Spatial Relations & Impingement
*   **ACL Volume**: Integrates the number of voxels belonging to the ACL mask multiplied by the isotropic voxel volume.
*   **Impingement Distance**: Computes an Euclidean Distance Transform (EDT) map of the inverted femur mask. The minimum distance value within the ACL mask boundaries corresponds to the closest distance between the ACL and the intercondylar notch wall, signaling potential impingement if it approaches zero.
*   **Notch Width**: Performs horizontal ray casting along the Left-Right axis at the level of the ACL centroid to measure the width of the intercondylar notch.

### Module 5: Radiomics Extraction
Extracts quantitative texture features using `PyRadiomics` from the intensity-standardized MRI within the ACL mask boundaries. Feature classes include:
*   **First-order statistics** (e.g., mean, variance, skewness, entropy)
*   **Gray Level Co-occurrence Matrix (GLCM)** features
*   **Gray Level Run Length Matrix (GLRLM)** features

### Module 6: Advanced Geometric Features
*   **Tortuosity Index**: Evaluates ligament bending. It computes slice-by-slice centroids along the primary axis component of the ACL. The tortuosity index is the ratio of the total curved centroid trajectory length to the straight-line distance between the bone footprints:
    $$\text{Tortuosity} = \frac{\text{Curved Centroid Path Length}}{\text{Footprint-to-Footprint Distance}}$$
*   **Anterior Tibial Translation (ATT)**: Measures sagittal subluxation. It projects the anterior direction vector onto the tibial plateau. It then locates the posterior-most physical edge points of the tibia and the lateral femoral condyle. The difference in their AP positions yields the ATT metric in millimeters.
*   **Stäubli Tibial Percentage**: Measures the AP position of the tibial footprint. It computes the projection of the tibial footprint centroid relative to the total AP width of the tibial plateau along a horizontal axis on the sagittal slice, where 0% corresponds to the anterior border and 100% to the posterior border.

---

## 5. 3D Visualization (`visualizator_analyzator.py`)

The visualization module [visualizator_analyzator.py](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Source/anaknee/visualizator_analyzator.py) uses `PyVista` to render interactive 3D scenes of the knee joint. It features:
*   Smooth marching cubes surface mesh generation for the Femur, Tibia, and ACL structures.
*   Plotting of the tibial plateau plane, ACL vector line, and Bernard & Hertel grid overlays.
*   Visualization of ATT perpendicular references and Stäubli AP measurement points.
*   **Interactive checkbox widgets** rendered directly on the screen to toggle the visibility of individual anatomical components and geometric calculation actors.

---

## 6. Model Training & Hyperparameter Optimization

### Segmentation Network (`WORKSTATION_BLACKWELL_MULTICLASS_5CV.py`)
Trains a custom 3D U-Net architecture (`LightUNet3D`) incorporating residual connections:
*   **Architecture**: Composed of encoder stages with max-pooling, a bottleneck stage with dropout, and decoder stages utilizing trilinear upsampling with 1x1 convolutions for channel reduction and skip connection concatenations.
*   **Loss Function (`WeightedDiceCELoss`)**: Sum of a multi-class `DiceLoss` (excluding background) and a weighted `CrossEntropyLoss` (strongly penalizing ACL segmentation errors).
*   **Cross-Validation**: Incorporates 5-Fold Cross-Validation, early stopping based on ACL validation Dice score, and plotting tools to save learning curves (Loss, Dice, HD95, and Learning Rate decay) for each fold.

### Hyperparameter Optimization (`workstation_blackwell_hpo.py`)
Uses `Optuna` to tune training hyperparameters:
*   Optimizes parameters such as `dropout` rate, learning rate (`lr`), and the specific loss function penalty weight for the ACL class (`acl_weight`).
*   Implements the `MedianPruner` scheduler to dynamically terminate trials that perform poorly in early epochs.
*   Caches raw data loading into RAM to prevent disk I/O bottlenecks during optimization.

### Laterality Classifier (`train_laterality_classifier.py`)
Trains a standard 3D ResNet-18 network using binary cross-entropy loss to classify NIfTI scans into `Left` (0.0) or `Right` (1.0) knees. The output model is utilized in the pipeline's canonization module.

---

## 7. Dependencies and Hardware Configuration

### Dependencies Setup
The pipeline relies on several specialized libraries. The environment can be initialized using the packages defined in [requirements.txt](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Documentation/requirements.txt):
```bash
pip install -r Documentation/requirements.txt
```
*Key packages include: `torch` (with CUDA support), `monai`, `SimpleITK`, `nibabel`, `torchio`, `pyradiomics`, `pyvista`, `scipy`, `scikit-image`, `pandas`, `optuna`.*

### Workstation Recommendations
For large-scale deep learning training, refer to [deep_learning_training_optimization_guidelines_workstations.md](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Documentation/deep_learning_training_optimization_guidelines_workstations.md):
*   **GPU target**: RTX 6000 Blackwell (96 GB VRAM).
*   **Memory target**: Pre-cache datasets under ~900 GB fully into system RAM (1 TB RAM workstations) to avoid disk read bottlenecks.
*   **Batch size scaling**: Tune batch sizes incrementally; if OOM occurs, step back. Adjust learning rate proportionally to batch size variations.

---

## 8. Configuration and Usage

Parameters are adjusted centrally within the `CONFIG` dictionary at the top of `mri_pipeline.py`.

### central Configuration Keys:
```python
CONFIG = {
    "mode": "FILE",                   # "FILE" or "FOLDER" processing modes
    "input_path": "...",              # Target NIfTI file path (FILE mode)
    "input_dir": "...",               # Input directory containing target NIfTIs (FOLDER mode)
    "output_dir": "...",              # Destination folder for segmentations & reports
    "log_file": "pipeline.log",       # Execution log filename
    
    # Pre-trained Checkpoints
    "model_ckpt": "...",              # Path to single Blackwell 3D U-Net checkpoint
    "kanonizace_ckpt": "...",         # Path to Laterality model checkpoint
    
    # Ensemble settings
    "use_ensemble": True,             # True/False switch
    "ensemble_dir": "...",            # Folder holding the 5 fold checkpoints
    "ensemble_pattern": "best_model_fold_*.pth",
    
    # Module Switches (1 to enable, 0 to disable)
    "run_resampling": 0,
    "run_orientation": 0,
    "run_canonization": 0,
    "run_inference": 0,
    "run_postprocessing": 0,
    "run_inverse_transform": 0,
    "run_segmentation_analysis": 0,
    "run_anatomical_analysis": 1,
}
```

### Running the Pipeline
To run the full end-to-end pipeline:
```bash
python Source/main/mri_pipeline.py
```
Check progress and calculation logs in the console or in the `pipeline.log` file generated in the target `output_dir`.
