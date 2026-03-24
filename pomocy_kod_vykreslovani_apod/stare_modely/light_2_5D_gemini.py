import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# --- KONFIGURACE ---
CONFIG = {
    'img_size': (256, 256),
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 150,
    'samples_per_epoch_train': 1000,
    'samples_per_epoch_val': 200,
    'base_filters': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 50,
    'save_dir': 'results_2_5D_dice'  # Upraven název složky
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
print(f"Running 2.5D on: {CONFIG['device']}")


# --- 1. DATASET ---
class ACLDataset25D(Dataset):
    def __init__(self, img_paths, mask_paths, img_size, samples_per_epoch, is_train=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        if self.is_train:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(height=img_size[0], width=img_size[1]),
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT),
                A.CenterCrop(height=img_size[0], width=img_size[1]),
                ToTensorV2()
            ])

        self.volumes = []
        self.masks = []
        self.positive_slices_map = {}
        self.acl_ranges = {}

        mode = "TRAIN" if is_train else "VALIDATION"
        print(f"[{mode}] Loading volumes...")

        for i, (img_path, mask_path) in enumerate(zip(self.img_paths, self.mask_paths)):
            try:
                img = nib.load(img_path).get_fdata().astype(np.float32)
                mask = nib.load(mask_path).get_fdata().astype(np.uint8)

                # Normalizace
                img = np.clip(img, np.percentile(img, 0.5), np.percentile(img, 99.5))
                img = (img - np.mean(img)) / (np.std(img) + 1e-8)

                # [H, W, D] -> [D, H, W] pro snazší slicing
                img = img.transpose(2, 0, 1)
                mask = mask.transpose(2, 0, 1)

                self.volumes.append(img)
                self.masks.append(mask)

                pos_indices = np.where(np.sum(mask, axis=(1, 2)) > 0)[0]
                if len(pos_indices) > 0:
                    self.positive_slices_map[i] = pos_indices
                    self.acl_ranges[i] = (pos_indices.min(), pos_indices.max())
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        if not self.volumes:
            raise RuntimeError(f"[{mode}] Žádná data nebyla načtena!")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        vol_idx = np.random.randint(len(self.volumes))
        attempts = 0

        # Snaha najít volume, který má pozitivní řezy (ACL)
        while vol_idx not in self.positive_slices_map and attempts < 10:
            vol_idx = np.random.randint(len(self.volumes))
            attempts += 1

        if vol_idx not in self.positive_slices_map:
            min_z, max_z = 0, self.volumes[vol_idx].shape[0] - 1
        else:
            min_z, max_z = self.acl_ranges[vol_idx]

        img_vol = self.volumes[vol_idx]
        mask_vol = self.masks[vol_idx]
        depth = img_vol.shape[0]

        # Sampling strategie
        if self.is_train:
            rand_val = np.random.rand()
            if rand_val < 0.70 and vol_idx in self.positive_slices_map:
                slice_idx = np.random.choice(self.positive_slices_map[vol_idx])
            elif rand_val < 0.90:
                margin = 15
                candidates = list(range(max(0, min_z - margin), min_z)) + \
                             list(range(max_z + 1, min(depth, max_z + margin)))
                slice_idx = np.random.choice(candidates) if candidates else np.random.randint(depth)
            else:
                slice_idx = np.random.randint(depth)
        else:
            # U validace chceme také validovat na ACL řezech častěji, abychom viděli Dice
            if np.random.rand() > 0.5 and vol_idx in self.positive_slices_map:
                slice_idx = np.random.choice(self.positive_slices_map[vol_idx])
            else:
                slice_idx = np.random.randint(depth)

        # 2.5D Stack (slice-1, slice, slice+1)
        stack = []
        for offset in [-1, 0, 1]:
            z = np.clip(slice_idx + offset, 0, depth - 1)
            stack.append(img_vol[z])

        img_25d = np.stack(stack, axis=-1)
        target_mask = mask_vol[slice_idx]

        augmented = self.transform(image=img_25d, mask=target_mask)
        return augmented['image'], augmented['mask'].unsqueeze(0).float()


# --- 2. MODEL ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class UNet25D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.dropout = nn.Dropout2d(p=0.2)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce3 = nn.Conv2d(base * 8, base * 4, kernel_size=1, bias=False)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce2 = nn.Conv2d(base * 4, base * 2, kernel_size=1, bias=False)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce1 = nn.Conv2d(base * 2, base, kernel_size=1, bias=False)
        self.dec1 = ConvBlock(base * 2, base)

        self.final = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))
        b = self.dropout(b)

        d3 = self.dec3(torch.cat([self.reduce3(self.up3(b)), e3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), e2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), e1], dim=1))

        return self.final(d1)


# --- 3. LOSS & METRICS ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 0.5 * bce_loss + 0.5 * (1 - dice)


def calculate_dice_metric(outputs, targets, threshold=0.5, smooth=1e-6):
    """
    Vypočítá tvrdé Dice skóre pro monitorování (ne pro gradient).
    """
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


# --- 4. PLOTTING FUNKCE (UPRAVENO) ---
def plot_metrics(history, save_path):
    """
    Vykreslí Training vs Validation Loss a Dice Score do jednoho obrázku.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Graf - LOSS
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Graf - DICE
    ax2.plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    ax2.set_title('Dice Score Evolution')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- 5. MAIN ---
def main():
    # Zkontroluj cesty!
    train_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    train_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels"

    all_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.nii*")))
    all_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.nii*")))

    if len(all_imgs) < 2:
        print("Příliš málo dat pro split!")
        return

    # Jednoduchý split - poslední volume je validace
    train_imgs, val_imgs = all_imgs[:-1], all_imgs[-1:]
    train_masks, val_masks = all_masks[:-1], all_masks[-1:]

    train_ds = ACLDataset25D(train_imgs, train_masks, CONFIG['img_size'], CONFIG['samples_per_epoch_train'],
                             is_train=True)
    val_ds = ACLDataset25D(val_imgs, val_masks, CONFIG['img_size'], CONFIG['samples_per_epoch_val'], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = UNet25D(in_ch=3, out_ch=1, base=CONFIG['base_filters']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = DiceBCELoss()
    scaler = GradScaler()

    # Tracking historie
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': []
    }

    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience_counter = 0

    start_time = datetime.now()
    print(f"Started training at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(CONFIG['epochs']):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0
        train_dice_accum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for images, masks in pbar:
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Metriky
            train_loss_accum += loss.item()
            train_dice_accum += calculate_dice_metric(outputs, masks)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_dice = train_dice_accum / len(train_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)

        # --- VALIDATION ---
        model.eval()
        val_loss_accum = 0
        val_dice_accum = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(CONFIG['device'])
                masks = masks.to(CONFIG['device'])

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss_accum += loss.item()
                val_dice_accum += calculate_dice_metric(outputs, masks)

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_dice = val_dice_accum / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

        # --- SAVE & PLOT ---

        # Plotujeme každou epochu, ať vidíš progres hned
        plot_metrics(history, os.path.join(CONFIG['save_dir'], 'metrics_plot.png'))

        # Early Stopping podle Loss (můžeš změnit na Dice, pokud chceš)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"✓ New best model saved (Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{CONFIG['patience']} epochs")

        # Pokud se zlepší Dice, taky si to poznač (informativně)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            # Můžeš uložit i "best_dice_model.pth" pokud chceš

        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break

        # Save latest
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "latest_model.pth"))

    # --- KONEC ---
    print("=" * 60)
    print(f"Training finished!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Grafy uloženy v: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()