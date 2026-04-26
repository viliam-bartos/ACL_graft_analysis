import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    NormalizeIntensityd, SpatialPadd, KeepLargestConnectedComponent
)
from monai.metrics import DiceMetric

# ----------------------------------------------------
# Globální konfigurace
# ----------------------------------------------------
TRAIN_IMG_DIR = r"A:\DATA_optimalizace\images_hpo"
TRAIN_MASK_DIR = r"A:\DATA_optimalizace\labels_hpo"
MODELS_DIR = r"results_blackwell_cv\Main_Run"
CACHE_DIR = r"C:\Users\daniel.bartos\cached_predictions"

CONFIG = {
    'patch_size': (128, 128, 80),
    'base_filters': 64,
    'dropout': 0.1
}

# ----------------------------------------------------
# Architektura (pro načtení vah)
# ----------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))) + self.skip(x))

class LightUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, base=64, dropout_rate=0.1):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base * 2)
        self.enc3 = ResBlock(base * 2, base * 4)
        self.bottleneck = ResBlock(base * 4, base * 8)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce3 = nn.Conv3d(base * 8, base * 4, kernel_size=1, bias=False)
        self.dec3 = ResBlock(base * 8, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce2 = nn.Conv3d(base * 4, base * 2, kernel_size=1, bias=False)
        self.dec2 = ResBlock(base * 4, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce1 = nn.Conv3d(base * 2, base, kernel_size=1, bias=False)
        self.dec1 = ResBlock(base * 2, base)

        self.final = nn.Conv3d(base, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bn = self.dropout(self.bottleneck(self.pool(x3)))

        d3 = self.dec3(torch.cat([self.reduce3(self.up3(bn)), x3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), x2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), x1], dim=1))

        return self.final(d1)

# ----------------------------------------------------
# Fáze 1: Precompute (Uložení surových Softmaxů)
# ----------------------------------------------------
def get_val_transforms(patch_size):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size)
    ])

def precompute_logits():
    print("\n--- FÁZE 1: PRECOMPUTING SUROVÝCH PREDICÍ ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    all_imgs = np.array(sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.nii*"))))
    all_masks = np.array(sorted(glob.glob(os.path.join(TRAIN_MASK_DIR, "*.nii*"))))
    
    if len(all_imgs) == 0:
        raise RuntimeError("Data nenalezena.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_imgs)):
        fold_id = fold_idx + 1
        model_path = os.path.join(MODELS_DIR, f"best_model_fold_{fold_id}.pth")
        
        if not os.path.exists(model_path):
            print(f"Přeskakuji Fold {fold_id}: Model {model_path} neexistuje.")
            continue

        fold_cache_dir = os.path.join(CACHE_DIR, f"fold_{fold_id}")
        os.makedirs(fold_cache_dir, exist_ok=True)

        # Načíst model pouze pokud chybí nějaké cache soubory
        val_files = [{"image": img, "label": mask} for img, mask in zip(all_imgs[val_idx], all_masks[val_idx])]
        files_to_process = []
        for v in val_files:
            basename = os.path.basename(v["image"]).replace('.nii.gz', '.pt')
            if not os.path.exists(os.path.join(fold_cache_dir, basename)):
                files_to_process.append(v)
        
        if len(files_to_process) == 0:
            print(f"Fold {fold_id} je již plně nacachovaný.")
            continue

        print(f"Počítám a cachuji {len(files_to_process)} snímků pro Fold {fold_id}...")
        
        model = LightUNet3D(in_ch=1, out_ch=4, base=CONFIG['base_filters'], dropout_rate=CONFIG['dropout'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        val_ds = CacheDataset(data=files_to_process, transform=get_val_transforms(CONFIG['patch_size']), cache_rate=1.0, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True)

        with torch.no_grad():
            from torch.amp import autocast
            for i, val_batch in enumerate(tqdm(val_loader, desc=f"Cachování Fold {fold_id}")):
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"] # Necháme na CPU
                
                with autocast('cuda', dtype=torch.bfloat16):
                    val_outputs = sliding_window_inference(val_images, roi_size=CONFIG['patch_size'], sw_batch_size=16, predictor=model, overlap=0.5, mode="gaussian")
                
                probs = torch.softmax(val_outputs, dim=1).cpu().squeeze(0) # Tvar: [4, H, W, D]
                label = val_labels.cpu().squeeze(0).squeeze(0) # Tvar: [H, W, D]

                basename = os.path.basename(files_to_process[i]["image"]).replace('.nii.gz', '.pt')
                save_path = os.path.join(fold_cache_dir, basename)
                
                # Uložíme tensor pravděpodobností a originální label
                torch.save({'probs': probs, 'label': label}, save_path)

        del model
        del val_loader
        del val_ds
        torch.cuda.empty_cache()


# ----------------------------------------------------
# Fáze 2: Post-processing a Optuna
# ----------------------------------------------------
GLOBAL_CACHED_DATA = []

def load_all_to_ram():
    print("\n--- NAČÍTÁNÍ PREDICÍ POUZE PRO FOLD 1 (BLESKOVÁ OPTIMALIZACE) ---")
    # Načteme schválně jen Fold 1 pro maximální rychlost
    cached_files = glob.glob(os.path.join(CACHE_DIR, "fold_1", "*.pt"))
    if not cached_files:
        raise RuntimeError("Žádné nacachované predikce nebyly nalezeny!")
    
    global GLOBAL_CACHED_DATA
    for f_path in tqdm(cached_files, desc="Načítání do RAM"):
        data = torch.load(f_path, weights_only=False)
        GLOBAL_CACHED_DATA.append({
            'probs': data['probs'],  # Uloženo v RAM
            'label': data['label']   # Uloženo v RAM
        })
    print(f"Úspěšně načteno {len(GLOBAL_CACHED_DATA)} pacientů do RAM. Optuna poletí bleskově!")

def apply_morphology(mask_tensor):
    # Hole filling (funguje na CPU numpy poli)
    mask_np = mask_tensor.numpy()
    mask_np = ndimage.binary_fill_holes(mask_np)
    
    # CCA - Ponecháme největší komponentu
    labeled, num_features = ndimage.label(mask_np)
    if num_features > 0:
        sizes = ndimage.sum(mask_np, labeled, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        mask_np = (labeled == max_label)
    
    return torch.from_numpy(mask_np).bool()

def evaluate_thresholds(t_acl, t_femur, t_tibia):
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    for data in GLOBAL_CACHED_DATA:
        probs = data['probs'] # [4, H, W, D]
        label = data['label'] # [H, W, D]

        # Prahování do binárních masek
        m_acl = probs[1] > t_acl
        m_fem = probs[2] > t_femur
        m_tib = probs[3] > t_tibia

        # Morfologie (CCA a Hole filling pro kosti)
        m_fem = apply_morphology(m_fem)
        m_tib = apply_morphology(m_tib)

        # Overlap Check (Rozřešení kolizí)
        final_pred = torch.zeros_like(label, dtype=torch.uint8)
        stacked_masks = torch.stack([m_acl, m_fem, m_tib], dim=0)
        relevant_probs = probs[1:4]
        masked_probs = relevant_probs * stacked_masks
        
        winners = torch.argmax(masked_probs, dim=0)
        any_mask = stacked_masks.any(dim=0)
        final_pred[any_mask] = winners[any_mask].byte() + 1 

        from monai.networks.utils import one_hot
        pred_onehot = one_hot(final_pred.unsqueeze(0).unsqueeze(0), num_classes=4)
        label_onehot = one_hot(label.unsqueeze(0).unsqueeze(0), num_classes=4)

        dice_metric(y_pred=pred_onehot, y=label_onehot)

    dice_scores = dice_metric.aggregate() 
    dice_metric.reset()
    return dice_scores[0].item(), dice_scores[1].item(), dice_scores[2].item()


def run_threshold_sweep():
    print("\n--- FÁZE 2: SYSTEMATICKÝ THRESHOLD SWEEP ---")
    thresholds = np.arange(0.05, 0.96, 0.05) # 0.05, 0.10, ..., 0.95
    
    # 1. Sweep ACL
    print("\n-> Sweep pro ACL (Kosti fixovány na 0.5)")
    acl_scores = []
    for t in tqdm(thresholds, desc="ACL Sweep"):
        d_acl, _, _ = evaluate_thresholds(t, 0.5, 0.5)
        acl_scores.append(d_acl)
        
    # 2. Sweep Femur
    print("\n-> Sweep pro Femur (ACL a Tibia fixovány na 0.5)")
    femur_scores = []
    for t in tqdm(thresholds, desc="Femur Sweep"):
        _, d_fem, _ = evaluate_thresholds(0.5, t, 0.5)
        femur_scores.append(d_fem)
        
    # 3. Sweep Tibia
    print("\n-> Sweep pro Tibii (ACL a Femur fixovány na 0.5)")
    tibia_scores = []
    for t in tqdm(thresholds, desc="Tibia Sweep"):
        _, _, d_tib = evaluate_thresholds(0.5, 0.5, t)
        tibia_scores.append(d_tib)
        
    # Vykreslení grafů
    print("\n--- Vykreslování vědeckých grafů ---")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, acl_scores, marker='o', color='red', linewidth=2)
    plt.title('Vliv prahování na Křížový vaz (ACL)')
    plt.xlabel('Práh (Threshold)')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, femur_scores, marker='o', color='blue', linewidth=2)
    plt.title('Vliv prahování na Stehenní kost (Femur)')
    plt.xlabel('Práh (Threshold)')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, tibia_scores, marker='o', color='green', linewidth=2)
    plt.title('Vliv prahování na Holenní kost (Tibia)')
    plt.xlabel('Práh (Threshold)')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('threshold_sweep_results.png', dpi=300)
    
    # Nalezení a vypsání maxim
    best_acl_idx = np.argmax(acl_scores)
    best_fem_idx = np.argmax(femur_scores)
    best_tib_idx = np.argmax(tibia_scores)
    
    print("\n=========================================")
    print("NALEZENÉ OPTIMÁLNÍ PRAHY (Thresholds):")
    print(f"Nejlepší práh pro ACL:   {thresholds[best_acl_idx]:.2f} (Max Dice: {acl_scores[best_acl_idx]:.4f})")
    print(f"Nejlepší práh pro Femur: {thresholds[best_fem_idx]:.2f} (Max Dice: {femur_scores[best_fem_idx]:.4f})")
    print(f"Nejlepší práh pro Tibii: {thresholds[best_tib_idx]:.2f} (Max Dice: {tibia_scores[best_tib_idx]:.4f})")
    print("=========================================")
    print("Graf s výsledky byl úspěšně uložen jako 'threshold_sweep_results.png'.")

# ----------------------------------------------------
# PR AUC Výpočet
# ----------------------------------------------------
def calculate_pr_auc():
    print("\n--- Počítám PR AUC z raw dat ---")
    cached_files = glob.glob(os.path.join(CACHE_DIR, "fold_*", "*.pt"))
    
    sample_files = cached_files[:10]
    
    y_true_acl, y_prob_acl = [], []
    
    for f_path in tqdm(sample_files, desc="Načítání do PR křivky"):
        data = torch.load(f_path, weights_only=False)
        probs = data['probs'][1].numpy().flatten()
        label = (data['label'].numpy().flatten() == 1).astype(int)
        
        y_true_acl.append(label[::10])
        y_prob_acl.append(probs[::10])
        
    y_true_acl = np.concatenate(y_true_acl)
    y_prob_acl = np.concatenate(y_prob_acl)
    
    precision, recall, _ = precision_recall_curve(y_true_acl, y_prob_acl)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'ACL PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (ACL)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('acl_pr_auc.png', dpi=300)
    print(f"PR AUC pro ACL = {pr_auc:.4f}. Graf uložen jako acl_pr_auc.png")


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    multiprocessing.freeze_support()
    
    # 1. Předpočítat data (Fáze 1)
    precompute_logits()
    
    # 2. Vykreslit PR AUC (Volitelně, vezme data z disku)
    calculate_pr_auc()
    
    # 3. Načtení dat do RAM (Pouze Fold 1 pro maximální rychlost Sweepu)
    load_all_to_ram()
    
    # 4. Spustit Systematický Threshold Sweep
    run_threshold_sweep()

if __name__ == "__main__":
    main()
