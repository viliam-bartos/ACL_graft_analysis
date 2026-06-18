# Automated ACL Segmentation and 3D Geometric Analysis

This project, titled **2509-MRI-Knee**, is developed by **Viliam Bartoš** under the supervision of **Ing. Jakub Lázňovský, Ph.D.** as part of a Diploma Thesis at CEITEC/CTLAB.

It provides an automated medical imaging pipeline for the segmentation of the Anterior Cruciate Ligament (ACL), Femur, and Tibia from 3D isotropic MRI scans. The repository also includes scripts for automated 3D geometric measurement and analysis of the knee joint.

---

## 🚀 Quick Links & Documentation

For detailed technical references, execution steps, and code explanations, please see the primary documentation files:

*   **[Technical Project Documentation](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Documentation/Project_Documentation.md)** - Explains the architecture of the pipeline, code structure, models, training, HPO, and metrics calculations.
*   **[Workstation Training Guidelines](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Documentation/deep_learning_training_optimization_guidelines_workstations.md)** - Optimization recommendations for GPU utilization and dataset caching on PC-APOLLO and PC-ATHENA workstations.
*   **[DevOps Guidelines](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Documentation/Supplement.md)** - Explains the trunk-based git workflow, issue management, and release instructions.

---

## 🛠️ Repository Modules

The codebase is organized into modular sections:

*   **`Source/main/mri_pipeline.py`**: The end-to-end processing pipeline, coordinating volume resampling, PIL reorientation, laterality classification, U-Net inference, post-processing, and anatomical calculations.
*   **`Source/anaknee/`**:
    *   `main_acl_analysis.py`: Features footprint centroid detection, tibial plateau SVD/PCA plane fitting, ACL orientation angles, radiomics features, tortuosity index, ATT (Anterior Tibial Translation), and Stäubli AP percentage.
    *   `visualizator_analyzator.py`: Interactive 3D visualization of the segmentations and geometric elements using PyVista.
*   **`Source/blackwell/`**: 3D U-Net multiclass model definitions, 5-fold cross-validation training, and Optuna HPO tuning.
*   **`Source/kanonizace/`**: 3D ResNet-18 laterality classifier to predict if the input MRI is a left or right knee.

---

## ⚙️ Installation and Setup

1.  Clone this repository to your local machine.
2.  Install dependencies:
    ```bash
    pip install -r Documentation/requirements.txt
    ```

---

## 🚦 How to Run

1.  Open [mri_pipeline.py](file:///c:/DIPLOM_PRACE/CEITEC/2509-MRI-Knee/Source/main/mri_pipeline.py) and modify the parameters in the `CONFIG` dictionary at the top (e.g., input paths, model checkpoints, switches for resampling, inference, and anatomical analysis).
2.  Execute the pipeline:
    ```bash
    python Source/main/mri_pipeline.py
    ```
3.  Check output segmentations, CSV reports, and visualization windows as configured.
