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

# ==============================================================================
# CONFIG PRO HROMADNOU INFERENCI
# ==============================================================================
CONFIG = {
    # Složka s NIfTI obrazy k predikci (uprav dle potřeby)
    "images_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train_full\images",
    
    # Výstupní CSV soubor kam se uloží predikce
    "output_csv": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\predikce_train_full_laterality.csv",

    # Cesta ke kontrolnímu bodu natrénovaného modelu
    "model_ckpt": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\checkpoints\best_laterality_model.pth",
    
    # Velikost na kterou byl model trénován
    "spatial_size": (96, 96, 96)
}
# ==============================================================================

class LateralityClassifier:
    def __init__(self, model_path=CONFIG["model_ckpt"], spatial_size=CONFIG["spatial_size"], device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sestavení sítě se stejnou architekturou jako při tréninku
        self.model = resnet18(
            spatial_dims=3, 
            n_input_channels=1,
            num_classes=1,
            norm=("instance", {"affine": True}) # Instance Norm
        )
        
        print(f"Načítám váhy modelu z: {model_path}")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Váhy úspěšně načteny.")
        else:
            raise FileNotFoundError(f"[CHYBA] Model nebyl nalezen na cestě: {model_path}. Musíš ho nejprve natrénovat!")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Transformace pro nediagnostické jednoruké načtení numpy arraye
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=torch.float32)
        ])

    def predict(self, image_path):
        """Provede predikci pro konkrétní NIfTI soubor a vrátí 'Left' nebo 'Right'."""
        input_tensor = self.transforms(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device) # Zabalení do batche
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        # Náš dataset měl Right=1.0, Left=0.0
        return "Right" if prob > 0.5 else "Left"


def main():
    images_dir = CONFIG["images_dir"]
    output_csv = CONFIG["output_csv"]
    
    if not os.path.isdir(images_dir):
        print(f"[CHYBA] Zadaná složka s obrázky '{images_dir}' neexistuje.")
        return
        
    try:
        classifier = LateralityClassifier()
    except Exception as e:
        print(e)
        return
    
    # Získání seznamu NIfTI souborů
    files = [f for f in os.listdir(images_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    
    if not files:
        print(f"[VAROVÁNÍ] Ve složce '{images_dir}' nebyly nalezeny žádné NIfTI soubory.")
        return
        
    print(f"\nZačínám predikci pro {len(files)} snímků...")
    
    # Vytvoření složky pro výstupní CSV, pokud neexistuje
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Zápis hlavičky
        writer.writerow(["ID", "Laterality"]) 
        
        # Iterace přes soubory s visualním progress barem
        for file in tqdm(files, desc="Zpracovávám MRI objemy", unit="snímek"):
            img_path = os.path.join(images_dir, file)
            try:
                prediction = classifier.predict(img_path)
                writer.writerow([file, prediction])
                csvfile.flush() # Okamžitý zápis na disk
            except Exception as e:
                print(f"\n[CHYBA] Selhala predikce pro {file}: {e}")
                writer.writerow([file, "ERROR"])
                
    print(f"\nHOTOVO! Predikce byly úspěšně uloženy do:\n-> {output_csv}")

if __name__ == "__main__":
    main()
