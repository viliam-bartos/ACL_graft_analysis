## DÁLE JE TŘEBA PŘIDAT UKLÁDÁNÍ GRAFŮ DO SVG!!!!!

# PRO VÍCE DAT JE TŘEBA NASTAVIT JINAK NAČÍTÁNÍ DAT (TEĎ SE NAČÍTAJÍ VŠECHNY DO RAM)
# ASI BY BYLO LEPŠÍ UDĚLAT NORMALIZACI A TŘEBA I N4ITK OFFLINE? MOŽNÁ ZBYTEČNÉ


import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime

from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandAffined, Rand3DElasticd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, NormalizeIntensityd, SpatialPadd
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.cuda.amp import autocast, GradScaler


from monai.utils import set_determinism

# Nastaví náhodné generátory v PyTorch, Numpy i Monai na stejné číslo
set_determinism(seed=42)

CONFIG = {
    'train_img_dir': r"C:\DIPLOM_PRACE\ACL_segment\data_train\images",
    'train_mask_dir': r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels",
    'patch_size': (128, 128, 32),
    'batch_size': 8,
    'accum_iter': 1,
    'learning_rate': 1e-4,
    'epochs': 500,
    'val_interval': 25,
    'base_filters': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'patience': 100,
    'save_dir_base': 'results_3D_CV',  # Změna názvu pro base dir
    'k_folds': 5  # Počet foldů
}

os.makedirs(CONFIG['save_dir_base'], exist_ok=True)


def get_transforms(mode='train'):
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=CONFIG['patch_size']),
    ]

    if mode == 'train':
        augmentations = [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=CONFIG['patch_size'],
                pos=2, neg=1, num_samples=4
            ),
            RandAffined(
                keys=["image", "label"], prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                mode=("bilinear", "nearest"), padding_mode="zeros"
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.5)),
            RandBiasFieldd(keys=["image"], prob=0.2, degree=3, coeff_range=(0.3, 0.5)),
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(5, 8), magnitude_range=(80, 100),
                prob=0.1, mode=("bilinear", "nearest"), padding_mode="zeros"
            ),
        ]
        return Compose(base_transforms + augmentations)

    return Compose(base_transforms)


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
    def __init__(self, in_ch=1, out_ch=1, base=16):
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
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bn = self.dropout(self.bottleneck(self.pool(x3)))

        d3 = self.dec3(torch.cat([self.reduce3(self.up3(bn)), x3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), x2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), x1], dim=1))

        return self.final(d1)


def plot_metrics(train_losses, val_losses, dice_scores, save_dir, fold_idx):
    """Vytvoří grafy pro konkrétní fold"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Graf 1: Losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)

    val_epochs = range(CONFIG['val_interval'], len(train_losses) + 1, CONFIG['val_interval'])
    if len(val_losses) > len(val_epochs):
        val_losses = val_losses[:len(val_epochs)]

    ax1.plot(val_epochs, val_losses, 'r-', marker='o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Fold {fold_idx}: Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graf 2: Dice Score
    if dice_scores:
        ax2.plot(val_epochs[:len(dice_scores)], dice_scores, 'g-', marker='s', label='Val Dice', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title(f'Fold {fold_idx}: Dice Score', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_fold_{fold_idx}.png'), dpi=150)
    plt.close()


def main():
    print(f"Running on: {CONFIG['device']}")

    # Načtení seznamu souborů
    train_imgs = sorted(glob.glob(os.path.join(CONFIG['train_img_dir'], "*.nii*")))
    train_masks = sorted(glob.glob(os.path.join(CONFIG['train_mask_dir'], "*.nii*")))

    if not train_imgs or len(train_imgs) != 5:
        # Varování, pokud data neodpovídají očekávání
        print(f"⚠ Nalezeno {len(train_imgs)} souborů. Očekáváno 5 pro 5-fold CV.")

    if len(train_imgs) == 0:
        raise RuntimeError("Žádná data!")

    # Vytvoření seznamu slovníků
    all_files = [{"image": i, "label": m} for i, m in zip(train_imgs, train_masks)]

    # K-Fold Cross Validation Loop
    k_folds = CONFIG['k_folds']
    fold_results = []

    print(f"Spouštím {k_folds}-fold Cross Validation...")
    print("=" * 60)

    for fold in range(k_folds):
        print(f"\n🚀 FOLD {fold + 1}/{k_folds}")

        # Příprava složky pro tento fold
        fold_save_dir = os.path.join(CONFIG['save_dir_base'], f'fold_{fold}')
        os.makedirs(fold_save_dir, exist_ok=True)

        # Manuální split pro fold (1 val, zbytek train)
        # Používáme modulo nebo prosté indexování, pokud máme přesně 5 souborů
        val_files = [all_files[fold]]
        train_files = [x for i, x in enumerate(all_files) if i != fold]

        print(f"  Train soubory: {len(train_files)}")
        print(f"  Val soubor: {val_files[0]['image']}")

        # Datasets & Loaders - VŽDY vytvořit nové pro každý fold
        train_ds = CacheDataset(data=train_files, transform=get_transforms('train'), cache_rate=1.0,
                                num_workers=CONFIG['num_workers'])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True)

        val_ds = CacheDataset(data=val_files, transform=get_transforms('val'), cache_rate=1.0,
                              num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Re-iniciace modelu a optimizeru
        model = LightUNet3D(in_ch=1, out_ch=1, base=CONFIG['base_filters']).to(CONFIG['device'])
        loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        scaler = GradScaler()
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        # Tracking pro tento fold
        best_metric = -1
        patience_counter = 0
        train_losses = []
        val_losses = []
        dice_scores = []

        for epoch in range(CONFIG['epochs']):
            model.train()
            epoch_loss = 0

            # Progress bar jen občas, ať nezahltíme konzoli
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)

            for batch in pbar:
                images, labels = batch["image"].to(CONFIG['device']), batch["label"].to(CONFIG['device'])

                with autocast():
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    loss = loss / CONFIG['accum_iter']

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                current_loss = loss.item() * CONFIG['accum_iter']
                epoch_loss += current_loss

                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validace
            if (epoch + 1) % CONFIG['val_interval'] == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_images = val_batch["image"].to(CONFIG['device'])
                        val_labels = val_batch["label"].to(CONFIG['device'])

                        val_outputs = sliding_window_inference(
                            inputs=val_images, roi_size=CONFIG['patch_size'],
                            sw_batch_size=2, predictor=model, overlap=0.5
                        )

                        loss = loss_function(val_outputs, val_labels)
                        val_loss += loss.item()

                        val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                        dice_metric(y_pred=val_outputs, y=val_labels)

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                dice_scores.append(metric)
                scheduler.step(metric)

                print(
                    f"Fold {fold + 1} Epoch {epoch + 1}: T-Loss: {avg_train_loss:.4f} | V-Loss: {avg_val_loss:.4f} | Dice: {metric:.4f}")

                # Save best & Early stopping
                if metric > best_metric:
                    best_metric = metric
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(fold_save_dir, "best_model.pth"))
                else:
                    patience_counter += 1

                if patience_counter >= CONFIG['patience'] // CONFIG['val_interval']:
                    print(f"🛑 Early stopping Fold {fold + 1} at epoch {epoch + 1}")
                    break

        # End of Fold
        plot_metrics(train_losses, val_losses, dice_scores, fold_save_dir, fold)
        fold_results.append(best_metric)

        # Uložení logu pro fold
        with open(os.path.join(fold_save_dir, 'log.txt'), 'w') as f:
            f.write(f"Fold {fold} Best Dice: {best_metric:.4f}\n")
            f.write(f"Epochs trained: {len(train_losses)}\n")

        # Cleanup VRAM
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 60)
    print("CROSS VALIDATION RESULTS")
    print("-" * 30)
    for i, score in enumerate(fold_results):
        print(f"Fold {i}: Dice = {score:.4f}")
    print("-" * 30)
    print(f"MEAN DICE: {np.mean(fold_results):.4f} +- {np.std(fold_results):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()