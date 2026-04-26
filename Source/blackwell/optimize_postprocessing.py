import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
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
            for val_batch in tqdm(val_loader, desc=f"Cachování Fold {fold_id}"):
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"] # Necháme na CPU
                
                with autocast('cuda', dtype=torch.bfloat16):
                    val_outputs = sliding_window_inference(val_images, roi_size=CONFIG['patch_size'], sw_batch_size=16, predictor=model, overlap=0.5, mode="gaussian")
                
                probs = torch.softmax(val_outputs, dim=1).cpu().squeeze(0) # Tvar: [4, H, W, D]
                label = val_labels.cpu().squeeze(0).squeeze(0) # Tvar: [H, W, D]

                basename = os.path.basename(val_batch["image_meta_dict"]["filename_or_obj"][0]).replace('.nii.gz', '.pt')
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

def objective(trial):
    # 1. Nezávislé prahy z Optuny
    t_acl = trial.suggest_float('t_acl', 0.1, 0.9)
    t_femur = trial.suggest_float('t_femur', 0.1, 0.9)
    t_tibia = trial.suggest_float('t_tibia', 0.1, 0.9)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Projdeme všechny nacachované soubory napříč foldy
    cached_files = glob.glob(os.path.join(CACHE_DIR, "fold_*", "*.pt"))
    if not cached_files:
        raise RuntimeError("Žádné nacachované predikce nebyly nalezeny!")

    for f_path in cached_files:
        data = torch.load(f_path)
        probs = data['probs'] # [4, H, W, D]
        label = data['label'] # [H, W, D] (hodnoty 0, 1, 2, 3)

        # 2. Prahování do binárních masek
        m_acl = probs[1] > t_acl
        m_fem = probs[2] > t_femur
        m_tib = probs[3] > t_tibia

        # 3. Morfologie (CCA a Hole filling pro kosti)
        m_fem = apply_morphology(m_fem)
        m_tib = apply_morphology(m_tib)

        # 4. Overlap Check (Rozřešení kolizí pomocí Surových pravděpodobností)
        final_pred = torch.zeros_like(label, dtype=torch.uint8)
        
        # Seskupíme masky
        stacked_masks = torch.stack([m_acl, m_fem, m_tib], dim=0) # [3, H, W, D]
        # Vytáhneme k nim relevantní pravděpodobnosti
        relevant_probs = probs[1:4] # [3, H, W, D]
        
        # Zamaskujeme pravděpodobnosti tak, aby "přežily" jen tam, kde pixel prošel prahem
        masked_probs = relevant_probs * stacked_masks
        
        # Zjistíme "vítěze" pro každý pixel (index 0=ACL, 1=Femur, 2=Tibia)
        winners = torch.argmax(masked_probs, dim=0) # [H, W, D]
        
        # Aplikujeme vítěze pouze na pixely, které prošly alespoň jedním prahem
        any_mask = stacked_masks.any(dim=0)
        final_pred[any_mask] = winners[any_mask].byte() + 1 # +1 protože ACL je 1, pozadí je 0

        # Převedeme do One-Hot pro DiceMetric z MONAI
        from monai.networks.utils import one_hot
        pred_onehot = one_hot(final_pred.unsqueeze(0).unsqueeze(0), num_classes=4) # [1, 4, H, W, D]
        label_onehot = one_hot(label.unsqueeze(0).unsqueeze(0), num_classes=4)     # [1, 4, H, W, D]

        dice_metric(y_pred=pred_onehot, y=label_onehot)

    # Vyhodnocení
    # Vrátíme průměrné Dice pro ACL (index 0 v mean batch), případně průměr všech tříd
    # Zde maximalizujeme průměr Dice všech tříd, aby Optuna neobětovala kosti kvůli ACL
    dice_scores = dice_metric.aggregate() # Tvar [3] (ACL, Femur, Tibia)
    mean_dice = torch.mean(dice_scores).item()
    
    # Trial reportne detailní skóre, i když optimalizuje jen průměr
    trial.set_user_attr("Dice_ACL", dice_scores[0].item())
    trial.set_user_attr("Dice_Femur", dice_scores[1].item())
    trial.set_user_attr("Dice_Tibia", dice_scores[2].item())
    
    dice_metric.reset()
    return mean_dice


# ----------------------------------------------------
# PR AUC Výpočet
# ----------------------------------------------------
def calculate_pr_auc():
    print("\n--- Počítám PR AUC z raw dat ---")
    cached_files = glob.glob(os.path.join(CACHE_DIR, "fold_*", "*.pt"))
    
    # Vezmeme např. prvních 10 pacientů, aby nám nespadla RAM (150 objemů = stovky GB floatů)
    sample_files = cached_files[:10]
    
    y_true_acl, y_prob_acl = [], []
    
    for f_path in tqdm(sample_files, desc="Načítání do PR křivky"):
        data = torch.load(f_path)
        probs = data['probs'][1].numpy().flatten() # Pravděpodobnost ACL
        label = (data['label'].numpy().flatten() == 1).astype(int) # Binární label ACL
        
        # Vezmeme jen každý 10. pixel pro úsporu paměti
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
    plt.savefig('acl_pr_auc.png')
    print(f"PR AUC pro ACL = {pr_auc:.4f}. Graf uložen jako acl_pr_auc.png")


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    multiprocessing.freeze_support()
    
    # 1. Předpočítat data
    precompute_logits()
    
    # 2. Vykreslit PR AUC z uložených dat
    calculate_pr_auc()
    
    # 3. Spustit Optunu
    print("\n--- FÁZE 2: OPTUNA POST-PROCESSING ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("\n--- HOTOVO ---")
    print(f"Nejlepší průměrné Dice: {study.best_value:.4f}")
    print("Nejlepší parametry:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    best_trial = study.best_trial
    print(f"Detaily nejlepšího pokusu:")
    print(f"  ACL Dice: {best_trial.user_attrs['Dice_ACL']:.4f}")
    print(f"  Femur Dice: {best_trial.user_attrs['Dice_Femur']:.4f}")
    print(f"  Tibia Dice: {best_trial.user_attrs['Dice_Tibia']:.4f}")

if __name__ == "__main__":
    main()
