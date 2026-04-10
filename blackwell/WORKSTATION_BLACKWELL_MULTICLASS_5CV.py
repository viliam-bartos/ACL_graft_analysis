import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import csv
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandAffined, Rand3DElasticd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, NormalizeIntensityd, SpatialPadd, AsDiscrete
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.misc import set_determinism
from torch.amp import autocast

# ----------------------------------------------------
# Globální nastavení pro Testování kódu
# ----------------------------------------------------
TEST_MODE = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    # Benchmark True zapne hledání nejrychlejších konvolucí
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------
# Architektura 3D U-Net multiclass (ch_out=4)
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
    def __init__(self, in_ch=1, out_ch=4, base=64, dropout_rate=0.2):
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
# Loss Funkce s Absolutní Penalizací ACL
# ----------------------------------------------------
class WeightedDiceCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # DiceLoss řeší překrytí tvarů objemů, Background je ignorován (false).
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, include_background=False)
       
        self.ce = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.ce(inputs, targets.squeeze(1).long())


# ----------------------------------------------------
# Augmentace a předzpracování (MONAI)
# ----------------------------------------------------
def get_transforms(mode, patch_size):
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ]

    if mode == 'train':
        augmentations = [
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=patch_size, pos=2, neg=1, num_samples=4),
            RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12), mode=("bilinear", "nearest"), padding_mode="zeros"),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.5)),
            RandBiasFieldd(keys=["image"], prob=0.2, degree=3, coeff_range=(0.3, 0.5)),
            Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(80, 100), prob=0.1, mode=("bilinear", "nearest"), padding_mode="zeros"),
        ]
        return Compose(base_transforms + augmentations)
    return Compose(base_transforms)


# ----------------------------------------------------
# Vizualizační funkce
# ----------------------------------------------------
def plot_learning_curves(csv_path, save_dir, fold_idx):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if 'Epoch' not in df.columns: return
    
    train_epochs = df['Epoch'].dropna()
    train_loss = df['Train_Loss'].dropna()

    val_df = df.dropna(subset=['Val_Loss'])
    val_epochs = val_df['Epoch']
    val_loss = val_df['Val_Loss']

    plt.figure(figsize=(15, 6))
    sns.set_theme(style="whitegrid")
    
    plt.subplot(1, 2, 1)
    sns.lineplot(x=train_epochs, y=train_loss, label="Trénovací ztráta", linewidth=2.5, color='royalblue')
    if len(val_epochs) > 0:
        sns.lineplot(x=val_epochs, y=val_loss, label="Validační ztráta", linewidth=2.5, marker="o", markersize=6, color='crimson')
    plt.title(f"Křivky učení (Ztrátová funkce) - Fold {fold_idx}", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Epocha", fontsize=14, fontweight='bold')
    plt.ylabel("Hodnota ztráty", fontsize=14, fontweight='bold')
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12, loc='upper right')
    
    plt.subplot(1, 2, 2)
    if len(val_epochs) > 0:
        if 'Val_Dice_ACL' in val_df.columns:
            sns.lineplot(x=val_epochs, y=val_df['Mean_Dice'], label="MEAN Dice", color='black', linewidth=3.0, linestyle="--")
            sns.lineplot(x=val_epochs, y=val_df['Val_Dice_ACL'], label="ACL Dice", color='forestgreen', linewidth=2.5, marker="s", markersize=6)
            sns.lineplot(x=val_epochs, y=val_df['Val_Dice_Femur'], label="Femur Dice", color='orange', linewidth=2.0)
            sns.lineplot(x=val_epochs, y=val_df['Val_Dice_Tibia'], label="Tibia Dice", color='dodgerblue', linewidth=2.0)
    
    plt.title(f"Vývoj multiclass skóre - Fold {fold_idx}", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Epocha", fontsize=14, fontweight='bold')
    plt.ylabel("Dice skóre", fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"learning_curve_fold_{fold_idx}.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"learning_curve_fold_{fold_idx}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()


def plot_global_cv_results(csv_path, save_dir):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Fold", y="Dice", hue="Struktura", palette="tab10", linewidth=1.5)
    plt.title("Křížová validace: 4-Class Testovací Dice", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Testovaný Fold", fontsize=14, fontweight='bold')
    plt.ylabel("Dice skóre", fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.tick_params(labelsize=12)
    plt.legend(title="Orgán", fontsize=10)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="Fold", y="HD95 [mm]", hue="Struktura", palette="tab10", linewidth=1.5)
    plt.title("Křížová validace: Testovací HD95 (Nižší je lepší)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Testovaný Fold", fontsize=14, fontweight='bold')
    plt.ylabel("Hausdorffova vzdálenost 95% [mm]", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.tick_params(labelsize=12)
    plt.legend(title="Orgán", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_cv_boxplot.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "global_cv_boxplot.pdf"), format='pdf', bbox_inches='tight')
    plt.close()


# ----------------------------------------------------
# Fáze Evaluace Testovacích dat po Foldu
# ----------------------------------------------------
def test_best_model_on_fold(best_model_path, config, val_files, fold_idx, device, global_csv_path):
    print(f"\n[Testovací Fáze] Vyhodnocování nejlepšího modelu foldu {fold_idx}")
    model = LightUNet3D(in_ch=1, out_ch=4, base=config['base_filters'], dropout_rate=config['dropout'])
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

    val_ds = CacheDataset(data=val_files, transform=get_transforms('val', config['patch_size']), cache_rate=1.0, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    results = []
    with torch.no_grad():
        for i, val_batch in enumerate(tqdm(val_loader, desc=f"Testing Fold {fold_idx}")):
            val_images, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)
            
            start_t = time.time()
            with autocast('cuda', dtype=torch.bfloat16):
                val_outputs = sliding_window_inference(val_images, roi_size=config['patch_size'], sw_batch_size=4, predictor=model, overlap=0.5)
            inf_time = time.time() - start_t
            
            val_outputs_converted = [post_pred(j) for j in decollate_batch(val_outputs)]
            val_labels_converted = [post_label(j) for j in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs_converted, y=val_labels_converted)
            hd95_metric(y_pred=val_outputs_converted, y=val_labels_converted)

            dice = dice_metric.get_buffer()[-1]     # tensor tvaru [3] pro tridu 1,2,3
            hd95 = hd95_metric.get_buffer()[-1]
            
            file_name = os.path.basename(val_files[i]['image'])

            for class_idx, class_name in enumerate(["ACL", "Femur", "Tibia"]):
                d_val = dice[class_idx].item() if not torch.isnan(dice[class_idx]) else 0.0
                try: h_val = hd95[class_idx].item()
                except: h_val = float('nan')

                results.append({
                    "Fold": f"Fold {fold_idx}",
                    "Soubor": file_name,
                    "Inference_Time_s": round(inf_time, 2),
                    "Struktura": class_name,
                    "Dice": d_val,
                    "HD95 [mm]": h_val
                })

    file_exists = os.path.isfile(global_csv_path)
    with open(global_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Fold", "Soubor", "Inference_Time_s", "Struktura", "Dice", "HD95 [mm]"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"Hotovo. Výsledky per-class testovací sady uloženy do {global_csv_path}.")


# ----------------------------------------------------
# Hlavní Trénovací smyčka Foldu
# ----------------------------------------------------
def train_fold(config, train_files, val_files, fold_idx, run_dir, global_cv_csv_path):
    print(f"\n=========================================")
    print(f"--- Začíná Fold {fold_idx} ---")
    print(f"=========================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers_train = 0 if TEST_MODE else 16
    workers_val = 0 if TEST_MODE else 8

    train_ds = CacheDataset(data=train_files, transform=get_transforms('train', config['patch_size']), cache_rate=1.0, num_workers=workers_val)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=workers_train, pin_memory=not TEST_MODE)

    val_ds = CacheDataset(data=val_files, transform=get_transforms('val', config['patch_size']), cache_rate=1.0, num_workers=workers_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers_val)

    model = LightUNet3D(in_ch=1, out_ch=4, base=config['base_filters'], dropout_rate=config['dropout'])

    if torch.cuda.device_count() > 1:
        print(f"Detekováno {torch.cuda.device_count()} Aktivuji DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Inicializace robustní Loss funkce s váhami (ACL je na indexu 1, proto dostane vahu 5.0)
    class_weights = torch.tensor([0.1, 5.0, 1.0, 1.0], dtype=torch.float32, device=device)
    loss_function = WeightedDiceCELoss(class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config['lr_patience'])

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_metric = -1
    patience_counter = 0

    csv_path = os.path.join(run_dir, f"log_fold_{fold_idx}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Time_s', 'Inference_Time_s_Avg', 'Train_Loss', 'Val_Loss', 'Mean_Dice', 'Val_Dice_ACL', 'Val_Dice_Femur', 'Val_Dice_Tibia', 'Val_HD95_ACL', 'Val_HD95_Femur', 'Val_HD95_Tibia', 'Learning_Rate'])

    best_model_path = os.path.join(run_dir, f"best_model_fold_{fold_idx}.pth")

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"F{fold_idx} Ep {epoch + 1}/{config['epochs']} [Train]")
        for batch in pbar:
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        train_time = round(time.time() - epoch_start_time, 2)

        val_loss, avg_inf_time = "", ""
        mean_dice, v_d_acl, v_d_fem, v_d_tib = "", "", "", ""
        v_hd_acl, v_hd_fem, v_hd_tib = "", "", ""

        if (epoch + 1) % config['val_interval'] == 0:
            model.eval()
            val_loss_sum = 0
            inf_times = []

            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    inf_start = time.time()
                    with autocast('cuda', dtype=torch.bfloat16):
                        val_outputs = sliding_window_inference(inputs=val_images, roi_size=config['patch_size'], sw_batch_size=4, predictor=model, overlap=0.5)
                        v_loss = loss_function(val_outputs, val_labels)
                        val_loss_sum += v_loss.item()
                    inf_times.append(time.time() - inf_start)

                    val_outputs_converted = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_converted = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs_converted, y=val_labels_converted)
                    hd95_metric(y_pred=val_outputs_converted, y=val_labels_converted)

            val_loss = val_loss_sum / len(val_loader)
            avg_inf_time = round(np.mean(inf_times), 3)

            # Extrakce Multi-class hodnot (Tensor delky 3 -> 0:ACL, 1:Fem, 2:Tib)
            val_metric_batch = dice_metric.aggregate()
            v_d_acl = val_metric_batch[0].item() if not torch.isnan(val_metric_batch[0]) else 0.0
            v_d_fem = val_metric_batch[1].item() if not torch.isnan(val_metric_batch[1]) else 0.0
            v_d_tib = val_metric_batch[2].item() if not torch.isnan(val_metric_batch[2]) else 0.0
            mean_dice = val_metric_batch.nanmean().item() if not torch.isnan(val_metric_batch).all() else 0.0
            
            try:
                hd_batch = hd95_metric.aggregate()
                v_hd_acl = hd_batch[0].item()
                v_hd_fem = hd_batch[1].item()
                v_hd_tib = hd_batch[2].item()
            except:
                v_hd_acl, v_hd_fem, v_hd_tib = float('nan'), float('nan'), float('nan')

            dice_metric.reset(); hd95_metric.reset()

            print(f"Ep {epoch + 1} | MEAN_DICE: {mean_dice:.4f} [ACL: {v_d_acl:.4f}, Fem: {v_d_fem:.4f}, Tib: {v_d_tib:.4f}] | Val_Loss: {val_loss:.4f}")
            
            # UKLÁDACÍ KRITÉRIUM EXKLUZIVNĚ NA ÚSPĚCH ACL
            scheduler.step(v_d_acl)

            if v_d_acl > best_metric:
                best_metric = v_d_acl
                patience_counter = 0
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_model_path)
            else:
                patience_counter += config['val_interval']

        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_time, avg_inf_time, avg_train_loss, val_loss, mean_dice, v_d_acl, v_d_fem, v_d_tib, v_hd_acl, v_hd_fem, v_hd_tib, current_lr])

        if (epoch + 1) % config['val_interval'] == 0:
            plot_learning_curves(csv_path, run_dir, fold_idx)

        if patience_counter >= config['patience']:
            print(f"--- Early stopping u foldu {fold_idx} po nedostatku zlepšení u ACL po {patience_counter} epochách. ---")
            break

    print(f"Trénink Foldu {fold_idx} dokončen s nejlepším validačním skóre ACL: {best_metric:.4f}")
    if os.path.exists(best_model_path):
        test_best_model_on_fold(best_model_path, config, val_files, fold_idx, device, global_cv_csv_path)

    return best_metric


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    multiprocessing.freeze_support()
    set_seed(42)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    train_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    train_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels"
    base_save_dir = 'results_blackwell_cv'
    os.makedirs(base_save_dir, exist_ok=True)
    global_cv_csv_path = os.path.join(base_save_dir, 'cv_individual_results.csv')

    all_imgs = np.array(sorted(glob.glob(os.path.join(train_img_dir, "*.nii*"))))
    all_masks = np.array(sorted(glob.glob(os.path.join(train_mask_dir, "*.nii*"))))

    if len(all_imgs) == 0:
        raise RuntimeError("Data nenalezena zákl. cestě.")

    # Config
    if TEST_MODE:
        all_imgs = all_imgs[:4]
        all_masks = all_masks[:4]
        config = {
            'patch_size': (128, 128, 64),
            'base_filters': 16,
            'lr': 1e-4,
            'epochs': 2,
            'val_interval': 1,
            'batch_size': 2,
            'patience': 2,
            'lr_patience': 1,
            'dropout': 0.2
        }
    else:
        config = {
            'patch_size': (224, 224, 128),   
            'base_filters': 64,              
            'lr': 1e-4,
            'epochs': 1000,
            'val_interval': 5,               # Kontrola každou 5. epochu
            'batch_size': 16,                 
            'patience': 40,                  # Early stop po: 40 kroků * 5 = 200 epoch bez zlepšení
            'lr_patience': 20,               # Snížení rychlosti učení po: 20 kroků * 5 = 100 epoch
            'dropout': 0.2                  
        }

    run_dir = os.path.join(base_save_dir, "Main_Run")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    kf = KFold(n_splits=5 if not TEST_MODE else 2, shuffle=True, random_state=42)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_imgs)):
        fold_id = fold_idx + 1
        train_files = [{"image": img, "label": mask} for img, mask in zip(all_imgs[train_idx], all_masks[train_idx])]
        val_files = [{"image": img, "label": mask} for img, mask in zip(all_imgs[val_idx], all_masks[val_idx])]

        best_dice = train_fold(config, train_files, val_files, fold_id, run_dir, global_cv_csv_path)
        fold_metrics.append(best_dice)

    avg_dice = np.mean(fold_metrics)
    print(f"\n=========================================")
    print(f"TRÉNINK DOKONČEN. Průměrné nejlepší Dice čistě pro ACL ze všech foldů: {avg_dice:.4f}")
    
    if os.path.exists(global_cv_csv_path):
        print("Vykreslování finálních globálních CV boxplotů per-class...")
        plot_global_cv_results(global_cv_csv_path, base_save_dir)
        print("Grafy uloženy")

if __name__ == "__main__":
    main()
