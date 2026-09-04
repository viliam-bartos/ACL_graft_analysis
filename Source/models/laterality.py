import os
import torch
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    EnsureType,
)


class LateralityClassifier:
    """
    Knee laterality classifier (Left vs Right) using a 3D ResNet-18 architecture.
    """
    def __init__(self, model_path="", spatial_size=(96, 96, 96), device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            norm=("instance", {"affine": True})
        )

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=torch.float32),
        ])

    def predict(self, image_path):
        """
        Predicts the laterality ('Left' or 'Right') and probability for a NIfTI file.
        
        Args:
            image_path (str): Path to NIfTI volume.
            
        Returns:
            tuple: (predicted_class, probability)
        """
        if not os.path.exists(image_path):
            return f"File not found: {image_path}", 0.5

        input_tensor = self.transforms(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()

        predicted_class = "Right" if prob > 0.5 else "Left"
        return predicted_class, prob
