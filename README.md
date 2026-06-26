# Automated ACL Segmentation and 3D Geometric Analysis

A medical imaging application for the automated segmentation of the Anterior Cruciate Ligament (ACL), Femur, and Tibia from isotropic 3D MRI scans. This project is developed by Viliam Bartoš under the supervision of Ing. Jakub Lázňovský, Ph.D. as part of a Master Thesis. 

---

## Features

* **GUI**: Desktop application for configuring the pipeline.
* **Format Support**: Should process both `.nii/.nii.gz` NIfTI volumes and raw `.dcm` DICOM directories.
* **Automated Pipeline**: Processing pipeline including resampling, orientation, Deep Learning inference (3D U-Net), and post-processing.
* **Geometric Analysis**: Computes clinical markers such as Anterior Tibial Translation (ATT), Stäubli percentage, ACL tortuosity, and notch width.
* **3D Visualization**: PyVista-based visualization of bones, ACL, and measurement axes.
* **Laterality Detection**: Automatically detects Left/Right knees from filenames or prompts the user via a GUI popup.

---

## Installation

1. Clone this repository to your local machine.
2. **Windows Prerequisite:** Python 3.10+ requires **Visual Studio Build Tools** (Desktop development with C++) to compile `pyradiomics`.
3. Install dependencies:
   ```bash
   pip install numpy==1.26.4 versioneer
   pip install pyradiomics --no-build-isolation
   pip install -r Documentation/requirements.txt
   ```

---

## Quickstart

Run the GUI application:
```bash
python Source/main/gui_app.py
```

1. **Processing Tab:** Select a Single File or Patient Folder. Choose the output directory and select which modules to run. Click **Run Analysis**.
2. **Dashboard Tab:** Load the resulting `patient_results.csv` to view metric trend graphs and launch the interactive 3D viewer for individual scans.
3. **Settings Tab:** Configure paths to reference MRIs, model checkpoints, and ground truth directories.

Also see [Project_Documentation.md](Documentation/Project_Documentation.md).
