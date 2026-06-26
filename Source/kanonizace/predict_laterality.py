import os
import argparse
import torch
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, 
    LoadImage,
    EnsureChannelFirst, 
    Resize, 
    ScaleIntensity,
    EnsureType
)

# Inference configuration
CONFIG = {
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
            print("[WARNING] Model checkpoint not found.")
            
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
        if not os.path.exists(image_path):
            return f"File not found: {image_path}", 0.5
            
        # Apply transforms and move tensor to device
        input_tensor = self.transforms(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device) # Batch size 1: (1, 1, Z, Y, X)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        # Class encoding: Right = 1.0, Left = 0.0
        predicted_class = "Right" if prob > 0.5 else "Left"
        return predicted_class, prob


def main():
    parser = argparse.ArgumentParser(description="Laterality inference. Returns Left/Right.")
    parser.add_argument("--img", type=str, required=True, help="Path to the NIfTI image.")
    args = parser.parse_args()
    
    classifier = LateralityClassifier()
    result = classifier.predict(args.img)
    print(f"Result for {os.path.basename(args.img)}: {result}")

if __name__ == "__main__":
    main()
