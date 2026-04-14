import os
import argparse
import torch
from monai.networks.nets import resnet10
from monai.transforms import (
    Compose, 
    LoadImage,
    EnsureChannelFirst, 
    Resize, 
    ScaleIntensity,
    EnsureType
)

# ==============================================================================
# CONFIG PRO INFERENCI
# ==============================================================================
CONFIG = {
    # Cesta ke kontrolnímu bodu nejlepšího modelu
    "model_ckpt": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\checkpoints\best_laterality_model.pth",
    
    # Velikost na kterou byl model trénován
    "spatial_size": (64, 64, 64)
}
# ==============================================================================

class LateralityClassifier:
    def __init__(self, model_path=CONFIG["model_ckpt"], spatial_size=CONFIG["spatial_size"], device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sestavení sítě (stejně jako při tréninku)
        self.model = resnet10(
            spatial_dims=3, 
            n_input_channels=1,
            num_classes=1
        )
        
        print(f"Načítám váhy modelu z: {model_path}")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Váhy úspěšně načteny.")
        else:
            print("[VAROVÁNÍ] Model neexistuje! Musíš ho nejdříve natrénovat.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # MONAI transformace pro inference. 
        # (Používáme nediagnostické verze transformací pracujících nad numpy array / path bez slovníku)
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=torch.float32)
        ])

    def predict(self, image_path):
        """Provede predikci pro konkrétní NIfTI soubor a vrátí text 'Left' nebo 'Right'."""
        if not os.path.exists(image_path):
            return f"Soubor nenalezen: {image_path}"
            
        # Převedení snímku sekvencí transformací na tenzor a přesun na GPU
        input_tensor = self.transforms(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device) # Batch size 1 = (1, 1, Z, Y, X)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        # Náš dataset měl Right=1.0, Left=0.0
        predicted_class = "Right" if prob > 0.5 else "Left"
        return predicted_class


def main():
    parser = argparse.ArgumentParser(description="Inference laterality. Vrátí prostý text Left/Right.")
    parser.add_argument("--img", type=str, required=True, help="Cesta k NIfTI obrazu pro určení laterality.")
    args = parser.parse_args()
    
    classifier = LateralityClassifier()
    result = classifier.predict(args.img)
    print(f"Result pro {os.path.basename(args.img)}: {result}")

if __name__ == "__main__":
    main()
