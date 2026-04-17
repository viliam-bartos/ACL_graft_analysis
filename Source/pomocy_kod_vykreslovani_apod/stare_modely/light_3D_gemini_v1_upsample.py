import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from datetime import datetime

# --- KONFIGURACE ---
CONFIG = {
    'patch_size': (128, 128, 32),
    'batch_size': 8,
    'accum_iter': 1,
    'learning_rate': 1e-4,  # Zmírněno z 3e-4
    'epochs': 200,
    'samples_per_epoch_train': 200,
    'samples_per_epoch_val': 50,
    'base_filters': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 50,  # Early stopping
    'save_dir': 'results_3D_v1'
}

# Vytvoř složku pro výsledky
os.makedirs(CONFIG['save_dir'], exist_ok=True)
print(f"Running on: {CONFIG['device']} with patch size {CONFIG['patch_size']}")


# --- 1. DATASET & PREPROCESSING ---
class ACLDataset(Dataset):
    def __init__(self, img_paths, mask_paths, patch_size, samples_per_epoch, is_train=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        self.images = []
        self.masks = []

        mode = "TRAIN" if is_train else "VALIDATION"
        print(f"[{mode}] Loading {len(self.img_paths)} volumes into RAM...")

        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            try:
                img = nib.load(img_path).get_fdata().astype(np.float32)
                mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            except Exception as e:
                print(f"CHYBA při čtení {img_path}: {e}")
                continue

            # Normalizace
            lower = np.percentile(img, 0.5)
            upper = np.percentile(img, 99.5)
            img = np.clip(img, lower, upper)
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / (std + 1e-8)

            self.images.append(img)
            self.masks.append(mask)

        if len(self.images) == 0:
            raise RuntimeError(f"[{mode}] Nepodařilo se načíst žádná data.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        vol_idx = np.random.randint(len(self.images))
        img = self.images[vol_idx]
        mask = self.masks[vol_idx]

        # Sampling strategie
        if self.is_train and np.random.rand() < 0.66:
            foreground_indices = np.argwhere(mask > 0)
            if len(foreground_indices) > 0:
                center = foreground_indices[np.random.randint(len(foreground_indices))]
            else:
                center = np.array(img.shape) // 2
        else:
            center = np.array([np.random.randint(0, s) for s in img.shape])

        # Výpočet rohů ořezu
        start = []
        end = []
        for i, dim in enumerate(self.patch_size):
            s = max(0, min(center[i] - dim // 2, img.shape[i] - dim))
            start.append(s)
            end.append(s + dim)

        img_patch = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mask_patch = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Padding
        if img_patch.shape != self.patch_size:
            pad = [(0, self.patch_size[i] - img_patch.shape[i]) for i in range(3)]
            img_patch = np.pad(img_patch, pad, mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, pad, mode='constant', constant_values=0)

        # Augmentace POUZE pro trénink
        if self.is_train:
            # Šum - POZOR: np.random.normal vrací float64, což změní typ img_patch!
            if np.random.rand() > 0.1:
                noise = np.random.normal(0, 0.1, img_patch.shape)
                img_patch = img_patch + noise

            # Gamma
            if np.random.rand() > 0.1:
                gamma = np.random.uniform(0.7, 1.3)
                min_val = img_patch.min()
                img_patch_shifted = img_patch - min_val + 1e-5
                img_patch = (img_patch_shifted ** gamma) + min_val

            # Brightness
            if np.random.rand() > 0.1:
                shift = np.random.uniform(-0.1, 0.1)
                img_patch = img_patch + shift

            # Rotace
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                # scipy rotate může taky změnit typ, ale .float() na konci to vyřeší
                img_patch = rotate(img_patch, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
                mask_patch = rotate(mask_patch, angle, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0)


        img_tensor = torch.from_numpy(img_patch.copy()).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch.copy()).unsqueeze(0).float()

        return img_tensor, mask_tensor


# --- 2. MODEL (Lightweight Residual 3D U-Net) ---
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
        self.skip = nn.Identity()
        if in_c != out_c:
            self.skip = nn.Conv3d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


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
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3 = self.pool(x3)
        bn = self.dropout(self.bottleneck(p3))

        u3 = self.up3(bn)
        u3 = self.reduce3(u3)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = self.reduce2(u2)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = self.reduce1(u1)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        return self.final(d1)


# --- 3. LOSS FUNKCE ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        return 0.5 * bce_loss + 0.5 * dice_loss


# --- 4. PLOTTING FUNKCE ---
def plot_losses(train_losses, val_losses, save_path):
    """Vytvoří graf train a val loss"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)

    # Označení nejlepší validační hodnoty
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    plt.scatter(best_epoch, best_val, color='red', s=100, zorder=5, marker='*')
    plt.annotate(f'Best: {best_val:.4f}',
                 xy=(best_epoch, best_val),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss (3D UNet v1)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def calculate_dice_score(pred, target, threshold=0.5):
    """Vypočítá Dice score z predikcí"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()


# --- 5. MAIN LOOP ---
def main():
    train_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    train_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels"

    # 1. Získat všechny cesty
    all_img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.nii*")))
    all_mask_paths = sorted(glob.glob(os.path.join(train_mask_dir, "*.nii*")))

    if len(all_img_paths) < 2:
        print("Máš příliš málo dat (méně než 2), nelze udělat split!")
        return

    # 2. Manuální split (Poslední soubor = validace)
    train_imgs = all_img_paths[:-1]
    train_masks = all_mask_paths[:-1]
    val_imgs = all_img_paths[-1:]
    val_masks = all_mask_paths[-1:]

    print(f"Dataset split: {len(train_imgs)} Train, {len(val_imgs)} Validation")

    # 3. Vytvoření Datasetů
    train_ds = ACLDataset(train_imgs, train_masks, CONFIG['patch_size'], CONFIG['samples_per_epoch_train'],
                          is_train=True)
    val_ds = ACLDataset(val_imgs, val_masks, CONFIG['patch_size'], CONFIG['samples_per_epoch_val'], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Model & Utils
    model = LightUNet3D(in_ch=1, out_ch=1, base=CONFIG['base_filters']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    criterion = DiceBCELoss()
    scaler = GradScaler()

    # Tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dice_scores = []

    start_time = datetime.now()
    print(f"Started training at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    for epoch in range(CONFIG['epochs']):
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for i, (images, masks) in enumerate(pbar):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / CONFIG['accum_iter']

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG['accum_iter'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * CONFIG['accum_iter']
            pbar.set_postfix({'loss': f'{loss.item() * CONFIG["accum_iter"]:.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        epoch_dice_scores = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(CONFIG['device'])
                masks = masks.to(CONFIG['device'])

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()

                # Vypočítej Dice pro každý batch
                dice = calculate_dice_score(outputs, masks)
                epoch_dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = np.mean(epoch_dice_scores)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_dice)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_dice:.4f}")

        # --- SAVE BEST MODEL & EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_dice,
            }, os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f}, Dice: {avg_dice:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{CONFIG['patience']} epochs")

        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break

        # --- PLOT LOSSES (každých 10 epoch) ---
        if (epoch + 1) % 10 == 0 or patience_counter >= CONFIG['patience']:
            plot_losses(train_losses, val_losses, os.path.join(CONFIG['save_dir'], 'loss_plot.png'))

        # Save latest checkpoint
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "latest_model.pth"))

    # --- FINÁLNÍ GRAFY ---
    plot_losses(train_losses, val_losses, os.path.join(CONFIG['save_dir'], 'final_loss_plot.png'))

    # Bonus: Graf Dice score
    if val_dice_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_dice_scores) + 1), val_dice_scores, 'g-', linewidth=2, marker='o')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.title('Validation Dice Score', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['save_dir'], 'dice_plot.png'), dpi=150)
        plt.close()

    # Ulož metriky do textového souboru
    end_time = datetime.now()
    with open(os.path.join(CONFIG['save_dir'], 'training_log.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Training completed (3D UNet v1 - scipy augmentation)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Start: {start_time}\n")
        f.write(f"End: {end_time}\n")
        f.write(f"Duration: {end_time - start_time}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Patch size: {CONFIG['patch_size']}\n")
        f.write(f"  Batch size: {CONFIG['batch_size']}\n")
        f.write(f"  Learning rate: {CONFIG['learning_rate']}\n")
        f.write(f"  Base filters: {CONFIG['base_filters']}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  Best Val Dice: {max(val_dice_scores):.4f}\n")
        f.write(f"  Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  Final Val Dice: {val_dice_scores[-1]:.4f}\n")
        f.write(f"  Total Epochs: {len(train_losses)}\n")

    print("=" * 60)
    print(f"Training finished!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Dice: {max(val_dice_scores):.4f}")
    print(f"Results saved to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()