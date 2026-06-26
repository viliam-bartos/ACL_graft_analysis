# Technical Project Documentation

This document provides a basic developer overview of the Automated ACL Segmentation and 3D Geometric Analysis application.

## 1. System Architecture

*   **`Source/main/gui_app.py`**: The graphical interface. It handles user inputs and runs the processing pipeline on a background thread.
*   **`Source/main/mri_pipeline.py`**: The core processing script.
*   **`Source/anaknee/`**: Contains `main_acl_analysis.py` for geometric calculations and `visualizator_analyzator.py` for the PyVista 3D viewer.
*   **`Source/blackwell/`**: Deep learning models (3D U-Net) and Hyperparameter Optimization scripts.

## 2. Processing Pipeline

The pipeline processes data in the following steps:

1. **Input Parsing**: Loads `.nii/.nii.gz` or `.dcm` files.
2. **Preprocessing**: Resamples voxel spacing to `(0.5, 0.5, 0.5)` mm and enforces PIL orientation.
3. **Laterality Classification**: Determines Left/Right knee from the filename. If ambiguous, it prompts the user via a GUI dialog. Right knees are mirrored.
4. **Inference**: Uses a 3D U-Net (`LightUNet3D`) running on CUDA.
5. **Post-processing**: Applies morphological filters (Largest Connected Component, hole filling, closing).
6. **Inverse Transform**: Restores the output mask to the original physical coordinate space and un-mirrors right knees.
7. **Anatomical Metrics**: Extracts clinical metrics and saves them to `patient_results.csv`.

## 3. Geometric & Anatomical Metrics

The `anaknee` module extracts the following metrics:

*   **Footprint Centroids**: 3D coordinates of the ACL insertion sites on the Femur and Tibia.
*   **Tibial Plateau Fitting**: Iteratively estimates the plane of the tibial plateau from the top proximal voxels of the tibia.
*   **Ligament Angles**: Calculates the sagittal, coronal, and elevation angles of the ACL vector.
*   **Impingement Distance**: Measures the minimum Euclidean distance between the ACL and the intercondylar notch wall.
*   **Tortuosity Index**: The ratio of the curved ACL path length to the straight-line distance between insertion footprints.
*   **Anterior Tibial Translation (ATT)**: Measures the AP subluxation difference between the posterior femoral condyle and the posterior tibia.
*   **Stäubli Tibial Percentage**: The relative anterior-to-posterior position of the tibial ACL footprint.

## 4. 3D Visualization

The Dashboard tab in the GUI opens an interactive PyVista viewer. It renders meshes of the bones and ligaments, and displays measurement axes.

## 5. Development & Training

Model training is handled in the `blackwell` module using a custom `WeightedDiceCELoss`. Hyperparameters are optimized using `Optuna`.
