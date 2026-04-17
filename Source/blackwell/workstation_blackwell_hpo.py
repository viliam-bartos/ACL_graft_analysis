import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import json

from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState

from monai.data import CacheDataset, Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandAffined, Rand3DElasticd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, NormalizeIntensityd, SpatialPadd, AsDiscrete
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.misc import set_determinism
from torch.amp import autocast

# ----------------------------------------------------
# Globální nastavení
# ----------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ----------------------------------------------------
# Architektura 3D U-Net multiclass
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
# Loss Funkce
# ----------------------------------------------------
class WeightedDiceCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, include_background=False)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.ce(inputs, targets.squeeze(1).long())

# ----------------------------------------------------
# Optuna Cíl
# ----------------------------------------------------
def objective(trial, cached_train_ds, cached_val_ds):
    print(f"\n=========================================")
    print(f"--- Začíná Optuna Trial {trial.number} ---")
    print(f"=========================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------
    # Definice Hyperparametrů
    # ------------------
    config = {
        'patch_size': (128, 128, 128),
        # Architektura a kapacita
        'base_filters': trial.suggest_categorical("base_filters", [16, 32, 64]),
        'dropout': trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
        
        # Optimalizace
        'lr': trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        'acl_weight': trial.suggest_float("acl_weight", 2.0, 10.0),
        
        # Augmentace (Pravděpodobnosti)
        'prob_affine': trial.suggest_categorical("prob_affine", [0.2, 0.5, 0.8]),
        'prob_elastic': trial.suggest_categorical("prob_elastic", [0.0, 0.1, 0.2]),
        'prob_noise': trial.suggest_categorical("prob_noise", [0.1, 0.3, 0.5]),
        'prob_contrast': trial.suggest_categorical("prob_contrast", [0.1, 0.3, 0.5]),
        
        # Pevné nastavení procesu
        'batch_size': 16, 
        'epochs': 150,
        'val_interval': 5,
    }

    # Dynamické augmentace vytvořené specificky pro parametry tohoto Trialu
    train_augmentations = Compose([
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=config['patch_size'], pos=2, neg=1, num_samples=4),
        RandAffined(keys=["image", "label"], prob=config['prob_affine'], rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12), mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandGaussianNoised(keys=["image"], prob=config['prob_noise'], mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=config['prob_contrast'], gamma=(0.5, 1.5)),
        RandBiasFieldd(keys=["image"], prob=0.2, degree=3, coeff_range=(0.3, 0.5)),
        Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(80, 100), prob=config['prob_elastic'], mode=("bilinear", "nearest"), padding_mode="zeros"),
    ])

    # K zabránění zbytečného a drastického přepočítávání cache mezi experimenty 
    # propojíme base dataset s dalšími vrstvami augmentací (rychlé)
    train_ds = Dataset(data=cached_train_ds, transform=train_augmentations)
    val_ds = Dataset(data=cached_val_ds, transform=None)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    model = LightUNet3D(in_ch=1, out_ch=4, base=config['base_filters'], dropout_rate=config['dropout'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Natižení unikátní váhy pro ACL
    class_weights = torch.tensor([0.1, config['acl_weight'], 1.0, 1.0], dtype=torch.float32, device=device)
    loss_function = WeightedDiceCELoss(class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6) # Rychlé snížení LR po 30 validačních epochách beze změny

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_metric = -1.0
    
    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep {epoch + 1}/{config['epochs']} [Train]", leave=False)
        
        for batch in pbar:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validační krok
        if (epoch + 1) % config['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    with autocast('cuda', dtype=torch.bfloat16):
                        val_outputs = sliding_window_inference(inputs=val_images, roi_size=config['patch_size'], sw_batch_size=4, predictor=model, overlap=0.5)
                    
                    val_outputs_converted = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_converted = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs_converted, y=val_labels_converted)
            
            val_metric_batch = dice_metric.aggregate()
            v_d_acl = val_metric_batch[0].item() if not torch.isnan(val_metric_batch[0]) else 0.0
            v_d_fem = val_metric_batch[1].item() if len(val_metric_batch) > 1 and not torch.isnan(val_metric_batch[1]) else 0.0
            
            dice_metric.reset()

            print(f"Trial {trial.number} | Ep {epoch + 1} | Val ACL Dice: {v_d_acl:.4f} (Femur: {v_d_fem:.4f})")
            scheduler.step(v_d_acl)
            
            if v_d_acl > best_metric:
                best_metric = v_d_acl
                
            # Report metric for dynamic pruning - Use Optuna to stop unpromising trials
            step = epoch // config['val_interval']
            trial.report(v_d_acl, step)
            if trial.should_prune():
                print(f"Trial {trial.number} ukončen brzo (Pruned) kvůli nízkému výkonu na Epoše {epoch+1}")
                raise optuna.exceptions.TrialPruned()

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

    all_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.nii*")))
    all_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.nii*")))

    if len(all_imgs) == 0:
        raise RuntimeError("Data nenalezena.")

    # Tvorba 1 trénovacího/validačního splitu pro HPO (test_size = 20%)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=42)

    train_files = [{"image": img, "label": mask} for img, mask in zip(train_imgs, train_masks)]
    val_files = [{"image": img, "label": mask} for img, mask in zip(val_imgs, val_masks)]

    print(f"Dataset rozdělen pro HPO: {len(train_files)} Tréninkových, {len(val_files)} Validačních ukázek.")

    # ------------------
    # Globální Před-Cached Data
    # ------------------
    # Tvorba datasetu bez random augmentací - zajistí načtení dat do RAM/GPU fixně
    base_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
    ])

    print("Spouštím před-caching RAM")
    cached_train_ds = CacheDataset(data=train_files, transform=base_transforms, cache_rate=1.0, num_workers=8)
    cached_val_ds = CacheDataset(data=val_files, transform=base_transforms, cache_rate=1.0, num_workers=8)

    # ------------------
    # Optuna HPO Setup
    # ------------------
    study_name = "Blackwell_ACL_Optimization"
    
    # Vytvoření study v paměti
    # Přepočítáno na validační kroky: n_warmup_steps=4 (20 epoch), interval_steps=1 (každá 5. epocha)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        pruner=pruner
    )

    print(f"Začíná optimalizace... Výsledky se průběžně ukládají do 'optuna_tuning_results.csv'.")
    
    
    def save_csv_callback(study, trial):
        study.trials_dataframe().to_csv("optuna_tuning_results.csv", index=False)

    
    study.optimize(
        lambda trial: objective(trial, cached_train_ds, cached_val_ds), 
        n_trials=30,
        callbacks=[save_csv_callback]
    )

    print("="*40)
    print("NEJLEPŠÍ IDENTIFIKOVANÉ PARAMETRY:")
    print(study.best_params)
    print(f"S Validačním Dice Skórem ACL: {study.best_value:.4f}")

if __name__ == "__main__":
    main()
