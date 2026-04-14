import os
import csv
import torch
import torch.nn as nn
from monai.networks.nets import resnet10
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Resized, 
    ScaleIntensityd,
    EnsureTyped
)
from monai.data import Dataset, DataLoader

# ==============================================================================
# CONFIG
# ==============================================================================
CONFIG = {
    # Cesty k datům
    "train_img_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images",
    "train_csv": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\analyza_laterality_train.csv",
    
    "val_img_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\val\images",
    "val_csv": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\analyza_laterality_val.csv",
    
    "output_dir": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\checkpoints",
    
    # Parametry tréninku
    "batch_size": 4,     
    "epochs": 20,         
    "lr": 1e-4,
    
    # Architektura
    "spatial_size": (64, 64, 64), 
    "num_workers": 4
}
# ==============================================================================

def load_data_from_csv(csv_path, img_dir):
    """Načte cesty k obrázkům a jejich labely podle CSV."""
    data_list = []
    if not os.path.exists(csv_path):
        print(f"[VAROVÁNÍ] CSV soubor chybí: {csv_path}")
        return data_list
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ID"]
            laterality = row["Laterality"]
            
            img_path = os.path.join(img_dir, img_id)
            if not os.path.exists(img_path):
                continue
                
            # Right = 1.0, Left = 0.0
            label = 1.0 if laterality.strip().lower() == "right" else 0.0
            
            data_list.append({"image": img_path, "label": label})
            
    return data_list

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    # 1. Transformace
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]), # Tvar (1, Z, Y, X)
        Resized(keys=["image"], spatial_size=CONFIG["spatial_size"], mode="trilinear"), 
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

    # 2. Načtení dat
    print("Načítám trénovací data...")
    train_data = load_data_from_csv(CONFIG["train_csv"], CONFIG["train_img_dir"])
    print(f"Nalezeno {len(train_data)} trénovacích vzorků.")

    print("Načítám validační data...")
    val_data = load_data_from_csv(CONFIG["val_csv"], CONFIG["val_img_dir"])
    print(f"Nalezeno {len(val_data)} validačních vzorků.")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("[CHYBA] Trénovací nebo validační set je prázdný. Zkontroluj CSV cesty.")
        return

    train_ds = Dataset(data=train_data, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    val_ds = Dataset(data=val_data, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 3. Model: 3D ResNet
    # Použijeme např. ResNet10 (monai) jako velmi malý model, vhodný na 64^3 obrázky.
    model = resnet10(
        spatial_dims=3, 
        n_input_channels=1,
        num_classes=1  # Jedna logitová hodnota pro binární klasifikaci (Right/Left)
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # 4. Trénovací smyčka
    best_val_acc = -1.0
    
    for epoch in range(CONFIG["epochs"]):
        print(f"--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        
        # --- TRAIN ---
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1).float() # Zajištění shape (B, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
        print(f"Train Loss: {epoch_loss/step:.4f}")
        
        # --- VAL ---
        model.eval()
        val_loss = 0
        val_step = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device).unsqueeze(1).float()
                
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item()
                val_step += 1
                
                # Výpočet přesnosti
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss = val_loss / val_step
        val_acc = correct / total
        
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} ({correct}/{total})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CONFIG["output_dir"], "best_laterality_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" => Uložen nový nejlepší model! (Acc: {best_val_acc:.4f})")

    print("Trénink dokončen.")
    
if __name__ == "__main__":
    main()
