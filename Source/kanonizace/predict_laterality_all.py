import os
import csv
import torch
from tqdm import tqdm
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, 
    LoadImage,
    EnsureChannelFirst, 
    Resize, 
    ScaleIntensity,
    EnsureType
)

# Batch inference configuration
CONFIG = {
    "images_dir": r"",
    "output_csv": r"",
    "model_ckpt": r"",
    "spatial_size": (96, 96, 96)
}

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from models.laterality import LateralityClassifier
except ImportError:
    from Source.models.laterality import LateralityClassifier


def main():
    images_dir = CONFIG["images_dir"]
    output_csv = CONFIG["output_csv"]
    
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory '{images_dir}' does not exist.")
        return
        
    try:
        classifier = LateralityClassifier()
    except Exception as e:
        print(e)
        return
    
    # Find all NIfTI files
    files = [f for f in os.listdir(images_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    
    if not files:
        print(f"[WARNING] No NIfTI files found in '{images_dir}'.")
        return
        
    print(f"\nStarting prediction for {len(files)} images...")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Laterality", "Probability"]) 
        
        for file in tqdm(files, desc="Processing MRI volumes", unit="volume"):
            img_path = os.path.join(images_dir, file)
            try:
                prediction, prob = classifier.predict(img_path)
                writer.writerow([file, prediction, f"{prob:.4f}"])
                csvfile.flush()
            except Exception as e:
                print(f"\n[ERROR] Prediction failed for {file}: {e}")
                writer.writerow([file, "ERROR"])
                
    print(f"\nPredictions successfully saved to:\n-> {output_csv}")

if __name__ == "__main__":
    main()
