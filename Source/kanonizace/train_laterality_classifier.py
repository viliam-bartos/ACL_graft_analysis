import os
import csv
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import torch
import torch.nn as nn
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    RandAffined,
    MapTransform,
)
from monai.data import Dataset, DataLoader, PersistentDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # Bez GUI – funguje i na serveru
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
CONFIG = {
    # Cesty k datům
    "train_img_dir": r"C:\antigrav_ACL\images",
    "train_csv": r"C:\antigrav_ACL\labels.csv",
    
    "val_split": 0.15,
    
    "output_dir": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\checkpoints2",
    "cache_dir": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\cache",
    
    # Parametry tréninku
    "batch_size": 8,          # Fyzická velikost posílaná v jednu chvíli na GPU
    "accumulation_steps": 2,  # Efektivní velikost dávky bude 2 * 8 = 16
    "epochs": 20,         
    "lr": 1e-4,
    
    # Architektura
    "spatial_size": (96, 96, 96), 
    "num_workers": 0
}
# ==============================================================================


class ReorientToPILd(MapTransform):
    """
    Reorientuje NIfTI tensor na cílovou orientaci PIL (Posterior-Inferior-Left),
    což odpovídá logice v reorient.py. Vstupem je tensor tvaru (C, Z, Y, X)
    uložený po EnsureChannelFirstd; transformace se provede per-kanál.

    Poznámka: PIL (ne standardní RAS) je záměrný – tak, aby
    Slicer/ITK-SNAP zobrazil snímek jako ASR, stejně jako v pipeline.
    """

    TARGET_ORNT = nio.axcodes2ornt("PIL")

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            tensor = d[key]  # shape: (C, D, H, W), torch.Tensor or np.ndarray
            # Převod na numpy pro nibabel
            is_tensor = isinstance(tensor, torch.Tensor)
            arr = tensor.numpy() if is_tensor else np.asarray(tensor)

            reoriented_channels = []
            for c in range(arr.shape[0]):
                # Vytvoříme dočasný NIfTI obraz s identity affine
                vol = arr[c]  # (D, H, W)
                tmp_img = nib.Nifti1Image(vol, affine=np.eye(4))
                orig_ornt = nio.io_orientation(tmp_img.affine)
                transform = nio.ornt_transform(orig_ornt, self.TARGET_ORNT)
                reoriented = tmp_img.as_reoriented(transform).get_fdata(dtype=np.float32)
                reoriented_channels.append(reoriented)

            result = np.stack(reoriented_channels, axis=0)  # (C, D', H', W')
            d[key] = torch.from_numpy(result) if is_tensor else result
        return d

def load_data_from_csv(csv_path, img_dir):
    """Načte cesty k obrázkům a jejich labely podle CSV."""
    data_list = []
    if not os.path.exists(csv_path):
        print(f"[VAROVÁNÍ] CSV soubor chybí: {csv_path}")
        return data_list
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["filename"]
            laterality = row["laterality"]
            
            img_path = os.path.join(img_dir, img_id)
            if not os.path.exists(img_path):
                continue
                
            # Right = 1.0, Left = 0.0
            label = 1.0 if laterality.strip().lower() == "right" else 0.0
            
            data_list.append({"image": img_path, "label": label})
            
    return data_list

class SimpleLateralityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Vstup po resize: (1, 96, 96, 96)
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2) # zmenší na (8, 48, 48, 48)
        
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2) # zmenší na (16, 24, 24, 24)
        
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(4) # zmenší na (32, 6, 6, 6)
        
        self.fc1 = nn.Linear(32 * 6 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten pro plně propojenou vrstvu
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    # 1. Transformace
    # Validační transformace (čisté načtení a resize)
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),        # Tvar (1, Z, Y, X)
        ReorientToPILd(keys=["image"]),             # Reorientace → PIL (stejná logika jako reorient.py)
        Resized(keys=["image"], spatial_size=CONFIG["spatial_size"], mode="trilinear"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

    # Trénovací transformace (s augmentacemi - pootočení a posun pro robustnost)
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ReorientToPILd(keys=["image"]),             # Reorientace → PIL (stejná logika jako reorient.py)
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(0.15, 0.15, 0.15),  # Mírné pootočení (~8-9 stupňů) ve všech osách
            translate_range=(5, 5, 5),        # Posun max o 5 voxelů
            mode="bilinear"
        ),
        Resized(keys=["image"], spatial_size=CONFIG["spatial_size"], mode="trilinear"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

    # 2. Načtení dat
    print("Načítám data...")
    all_data = load_data_from_csv(CONFIG["train_csv"], CONFIG["train_img_dir"])
    print(f"Nalezeno celkem {len(all_data)} vzorků.")

    if len(all_data) == 0:
        print("[CHYBA] Dataset je prázdný. Zkontroluj CSV cesty.")
        return

    # Split na train a val – stratify zajistí stejný poměr L/R v obou setech
    labels_for_stratify = [d["label"] for d in all_data]
    train_data, val_data = train_test_split(
        all_data,
        test_size=CONFIG["val_split"],
        random_state=42,
        shuffle=True,
        stratify=labels_for_stratify,
    )

    n_left_train  = sum(1 for d in train_data if d["label"] == 0.0)
    n_right_train = sum(1 for d in train_data if d["label"] == 1.0)
    n_left_val    = sum(1 for d in val_data   if d["label"] == 0.0)
    n_right_val   = sum(1 for d in val_data   if d["label"] == 1.0)
    print(f"  Train -> Left: {n_left_train}, Right: {n_right_train}")
    print(f"  Val   -> Left: {n_left_val},  Right: {n_right_val}")
    
    print(f"Rozděleno na {len(train_data)} trénovacích a {len(val_data)} validačních vzorků.")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("[CHYBA] Trénovací nebo validační set je prázdný. Zkontroluj CSV cesty.")
        return

    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=CONFIG["cache_dir"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=CONFIG["cache_dir"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 3. Model: SimpleLateralityCNN
    model = SimpleLateralityCNN().to(device)

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
        
        optimizer.zero_grad() # Vynulování před první dávkou
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")
        for i, batch_data in train_pbar:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1).float() # Zajištění shape (B, 1)
            
            outputs = model(inputs)
            
            # Loss podělíme počtem kumulačních kroků
            loss = loss_function(outputs, labels) / CONFIG["accumulation_steps"]
            loss.backward()
            
            # Provedeme aktualizaci vah pouze když dosáhneme cesty accumulation_steps, 
            # nebo pokud jsme na úplném konci trénovací sady.
            if ((i + 1) % CONFIG["accumulation_steps"] == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * CONFIG["accumulation_steps"] # Zpětný výpočet pro hezký výpis
            step += 1
            train_pbar.set_postfix({"loss": f"{epoch_loss/step:.4f}"})
            
        print(f"Train Loss: {epoch_loss/step:.4f}")
        
        # --- VAL ---
        model.eval()
        val_loss = 0
        val_step = 0
        correct = 0
        total = 0
        all_preds  = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]")
        with torch.no_grad():
            for batch_data in val_pbar:
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

                all_preds.extend(preds.cpu().squeeze(1).int().tolist())
                all_labels.extend(labels.cpu().squeeze(1).int().tolist())
                val_pbar.set_postfix({"acc": f"{correct/total:.4f}"})
                
        val_loss = val_loss / val_step
        val_acc = correct / total
        
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} ({correct}/{total})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CONFIG["output_dir"], "best_laterality_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" => Uložen nový nejlepší model! (Acc: {best_val_acc:.4f})")

    # ==========================================================================
    # 5. Finální vyhodnocení na validačním setu (poslední epoch)
    # ==========================================================================
    print("\n" + "="*60)
    print("FINÁLNÍ VÝSLEDKY NA VALIDAČNÍM SETU (poslední epoch)")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=["Left (0)", "Right (1)"],
        digits=4
    ))

    # --- Matice záměn ---
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Left", "Right"],
        yticklabels=["Left", "Right"],
        linewidths=0.5,
        linecolor="grey",
        ax=ax,
    )
    ax.set_xlabel("Predikce", fontsize=12)
    ax.set_ylabel("Skutečnost", fontsize=12)
    ax.set_title("Matice záměn pro určení laterality kolene", fontsize=13, pad=12)
    fig.tight_layout()

    cm_pdf = os.path.join(CONFIG["output_dir"], "confusion_matrix.pdf")
    cm_png = os.path.join(CONFIG["output_dir"], "confusion_matrix.png")
    fig.savefig(cm_pdf, dpi=150)
    fig.savefig(cm_png, dpi=150)
    plt.close(fig)
    print(f"Matice záměn uložena:\n  PDF → {cm_pdf}\n  PNG → {cm_png}")

    print("Trénink dokončen.")
    
if __name__ == "__main__":
    main()
