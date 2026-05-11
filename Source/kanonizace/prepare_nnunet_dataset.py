import os
import json
import shutil
from pathlib import Path

# Nastavení cest (upravte podle potřeby)
BASE_DIR = Path(r"C:\DIPLOM_PRACE\ACL_segment\kanonizace")
IMAGES_DIR = BASE_DIR / "images_train_full_canonical"
MASKS_DIR = BASE_DIR / "masks_train_full_canonical"

# Název nového datasetu
DATASET_NAME = "Dataset001_ACL"
OUTPUT_DIR = BASE_DIR / DATASET_NAME

IMAGES_TR = OUTPUT_DIR / "imagesTr"
LABELS_TR = OUTPUT_DIR / "labelsTr"

def main():
    print(f"Vytvářím strukturu nnU-Net datasetu v: {OUTPUT_DIR}")
    
    # Vytvoření potřebných složek
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    
    # Nalezení všech trénovacích snímků (předpokládáme formát case_*.nii.gz)
    image_files = sorted(list(IMAGES_DIR.glob("case_*.nii.gz")))
    
    num_training = 0
    
    for img_path in image_files:
        identifikator = img_path.name.replace(".nii.gz", "")
        
        # Související maska má prefix 'mask_'
        mask_name = f"mask_{identifikator}.nii.gz"
        mask_path = MASKS_DIR / mask_name
        
        if mask_path.exists():
            # Cílový název snímku vyžaduje sufix kanálu _0000 (0 = první kanál/modalita)
            target_img_path = IMAGES_TR / f"{identifikator}_0000.nii.gz"
            # Cílový název masky je bez sufixu
            target_mask_path = LABELS_TR / f"{identifikator}.nii.gz"
            
            # Kopírování
            shutil.copy2(img_path, target_img_path)
            shutil.copy2(mask_path, target_mask_path)
            num_training += 1
        else:
            print(f"Varování: Maska pro snímek {identifikator} nebyla nalezena ({mask_path}). Přeskakuji.")
            
    # Definice obsahu dataset.json
    dataset_dict = {
        "channel_names": {
            "0": "MRI" # nnU-Net V2 vyžaduje číslování kanálů od 0 (odpovídá _0000)
        },
        "labels": {
            "background": 0,
            "ACL": 1,
            "Femur": 2,
            "Tibia": 3
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    # Uložení dataset.json
    json_path = OUTPUT_DIR / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=4)
        
    print(f"\nHotovo!")
    print(f"Zkopírováno případů: {num_training}")
    print(f"Soubor dataset.json vytvořen: {json_path}")

if __name__ == "__main__":
    main()
