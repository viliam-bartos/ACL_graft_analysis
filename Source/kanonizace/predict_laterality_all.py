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

class LateralityClassifier:
    def __init__(self, model_path=CONFIG["model_ckpt"], spatial_size=CONFIG["spatial_size"], device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build network (same as training)
        self.model = resnet18(
            spatial_dims=3, 
            n_input_channels=1,
            num_classes=1,
            norm=("instance", {"affine": True})
        )
        
        print(f"Loading model weights from: {model_path}")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"[ERROR] Model not found at: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # MONAI inference transforms
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=torch.float32)
        ])

    def predict(self, image_path):
        """Predicts the laterality ('Left' or 'Right') and probability for a NIfTI file."""
        input_tensor = self.transforms(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device) # Add batch dimension
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        # Class encoding: Right = 1.0, Left = 0.0
        predicted_class = "Right" if prob > 0.5 else "Left"
        return predicted_class, prob


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
