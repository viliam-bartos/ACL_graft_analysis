import os
import csv
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Resized, 
    ScaleIntensityd,
    EnsureTyped,
    RandAffined
)
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Training configuration
CONFIG = {
    "train_img_dir": r"",
    "train_csv": r"",
    "val_split": 0.15,
    "output_dir": r"",
    "batch_size": 2,          # Physical batch size on GPU
    "accumulation_steps": 8,  # Effective batch size = 2 * 8 = 16
    "epochs": 20,         
    "lr": 1e-4,
    "spatial_size": (96, 96, 96), 
    "num_workers": 4
}

def load_data_from_csv(csv_path, img_dir):
    """Loads image paths and laterality labels from CSV."""
    data_list = []
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file missing: {csv_path}")
        return data_list
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ID"]
            laterality = row["Laterality"]
            
            img_path = os.path.join(img_dir, img_id)
            if not os.path.exists(img_path):
                continue
                
            # Class mapping: Right = 1.0, Left = 0.0
            label = 1.0 if laterality.strip().lower() == "right" else 0.0
            data_list.append({"image": img_path, "label": label})
            
    return data_list

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validation transforms (load and resize only)
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=CONFIG["spatial_size"], mode="trilinear"), 
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

    # Training transforms (with augmentation for robustness)
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(0.15, 0.15, 0.15),  # ~8-9 degrees rotation
            translate_range=(5, 5, 5),        # Max 5 voxels translation
            mode="bilinear"
        ),
        Resized(keys=["image"], spatial_size=CONFIG["spatial_size"], mode="trilinear"), 
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

    print("Loading data...")
    all_data = load_data_from_csv(CONFIG["train_csv"], CONFIG["train_img_dir"])
    print(f"Found {len(all_data)} samples.")

    if len(all_data) == 0:
        print("[ERROR] Dataset is empty.")
        return

    train_data, val_data = train_test_split(
        all_data, 
        test_size=CONFIG["val_split"], 
        random_state=42, 
        shuffle=True
    )
    
    print(f"Split into {len(train_data)} train and {len(val_data)} validation samples.")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("[ERROR] Train or validation set is empty.")
        return

    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 3D ResNet-18 model for binary classification
    model = resnet18(
        spatial_dims=3, 
        n_input_channels=1,
        num_classes=1,
        norm=("instance", {"affine": True}) 
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    best_val_acc = -1.0
    
    for epoch in range(CONFIG["epochs"]):
        print(f"--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        
        # --- TRAIN ---
        model.train()
        epoch_loss = 0
        step = 0
        optimizer.zero_grad()
        
        for i, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1).float()
            
            outputs = model(inputs)
            
            # Gradient accumulation
            loss = loss_function(outputs, labels) / CONFIG["accumulation_steps"]
            loss.backward()
            
            if ((i + 1) % CONFIG["accumulation_steps"] == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * CONFIG["accumulation_steps"]
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
                
                # Accuracy calculation
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
            print(f" => Saved new best model (Acc: {best_val_acc:.4f})")

    print("Training complete.")
    
if __name__ == "__main__":
    main()
