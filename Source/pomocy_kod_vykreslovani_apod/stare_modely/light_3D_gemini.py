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

# --- KONFIGURACE ---
CONFIG = {
    'patch_size': (128, 128, 32),  # Pokud to spadne na OOM, zmenši na (48, 48, 32)
    'batch_size': 8,  # Na 6GB VRAM nedávej víc
    'accum_iter': 1,  # Virtuální batch size = 1 * 8 = 8
    'learning_rate': 3e-4,  # Opatrný start
    'epochs': 20,  # S 10 vzorky to bude lítat rychle
    'samples_per_epoch': 200,  # Kolik patchů vytáhneme v jedné epoše
    'base_filters': 32,  # Nízký počet filtrů pro úsporu paměti
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on: {CONFIG['device']} with patch size {CONFIG['patch_size']}")


# --- 1. DATASET & PREPROCESSING ---
class ACLDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size, samples_per_epoch, is_train=True):
        # 1. Hledáme .nii i .nii.gz
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))

        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        # --- DEBUG VÝPISY ---
        print(f"Hledám obrázky v: {img_dir}")
        print(f"Nalezeno obrázků: {len(self.img_paths)}")
        print(f"Hledám masky v:   {mask_dir}")
        print(f"Nalezeno masek:   {len(self.mask_paths)}")

        if len(self.img_paths) == 0:
            raise RuntimeError(f"CHYBA: Složka {img_dir} je prázdná nebo neobsahuje .nii/.nii.gz soubory!")
        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError(f"NESHODA: Máš {len(self.img_paths)} obrázků, ale {len(self.mask_paths)} masek!")
        # --------------------

        self.images = []
        self.masks = []

        print("Loading data into RAM...")
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            # Načítání s ošetřením chyb (pro případ, že by tam byl další vadný header)
            try:
                img = nib.load(img_path).get_fdata().astype(np.float32)
                mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            except Exception as e:
                print(f"CHYBA při čtení {img_path}: {e}")
                continue  # Přeskočí vadný soubor

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
            raise RuntimeError("Nepodařilo se načíst žádná data do paměti.")

        print(f"Úspěšně načteno {len(self.images)} objemů.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Náhodný výběr pacienta
        vol_idx = np.random.randint(len(self.images))
        img = self.images[vol_idx]
        mask = self.masks[vol_idx]

        # Sampling strategie: 66% šance, že vybereme střed patche uvnitř ACL (popředí)
        # To je kritické pro class imbalance!
        if self.is_train and np.random.rand() < 0.66:
            foreground_indices = np.argwhere(mask > 0)
            if len(foreground_indices) > 0:
                center = foreground_indices[np.random.randint(len(foreground_indices))]
            else:
                # Fallback, pokud maska prázdná
                center = np.array(img.shape) // 2
        else:
            # Náhodný crop odkudkoliv
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

        # Padding, pokud jsme na kraji a patch je menší
        if img_patch.shape != self.patch_size:
            pad = [(0, self.patch_size[i] - img_patch.shape[i]) for i in range(3)]
            img_patch = np.pad(img_patch, pad, mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, pad, mode='constant', constant_values=0)


            if self.is_train:
                # 1. Geometrické transformace (Flip)
                # Rotaci o 90 stupňů raději vyhoď, je nepřirozená.
                #if np.random.rand() > 0.5:
                 #   img_patch = np.flip(img_patch, axis=0)  # Flip Horizontal
                  #  mask_patch = np.flip(mask_patch, axis=0)
                #if np.random.rand() > 0.5:
                 #   img_patch = np.flip(img_patch, axis=1)  # Flip Vertical
                  #  mask_patch = np.flip(mask_patch, axis=1)

                # 2. Intenzitní transformace (MRI nutnost!)
                # Přidání gaussovského šumu (simuluje zrnitost MRI)
                if np.random.rand() > 0.1:  # 90% šance
                    noise = np.random.normal(0, 0.1, img_patch.shape)
                    img_patch = img_patch + noise

                # Změna kontrastu/jasu (Gamma korekce)
                # Simuluje různé nastavení sekvencí
                if np.random.rand() > 0.1:
                    gamma = np.random.uniform(0.7, 1.3)
                    # Musíme ošetřit záporné hodnoty před mocninou (kvůli z-score)
                    min_val = img_patch.min()
                    img_patch_shifted = img_patch - min_val + 1e-5
                    img_patch = (img_patch_shifted ** gamma) + min_val

                # Náhodný posun intenzity (Brightness)
                if np.random.rand() > 0.1:
                    shift = np.random.uniform(-0.1, 0.1)
                    img_patch = img_patch + shift

                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)  # Rotace o max +/- 15 stupňů

                    # Obraz: order=1 (Bilinear interpolace) - hladké
                    img_patch = rotate(img_patch, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)

                    # Maska: order=0 (Nearest Neighbor) - KRITICKÉ!
                    # Nesmíš interpolovat masku, jinak ti vzniknou hodnoty jako 0.5, což není label.
                    mask_patch = rotate(mask_patch, angle, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0)

        # Přidání channel dimenze [C, D, H, W]
        img_tensor = torch.from_numpy(img_patch.copy()).unsqueeze(0)
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

        # Skip connection adjustment
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
        # Encoder
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base * 2)
        self.enc3 = ResBlock(base * 2, base * 4)
        self.bottleneck = ResBlock(base * 4, base * 8)

        self.pool = nn.MaxPool3d(2)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock(base * 8, base * 4)  # base*4 from up + base*4 from skip

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base * 2, base)

        self.final = nn.Conv3d(base, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(0.2)  # Nutnost pro small data

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        # Bottleneck
        bn = self.dropout(self.bottleneck(p3))

        # Decoder
        u3 = self.up3(bn)
        u3 = torch.cat([u3, x3], dim=1)  # Skip connection
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        return self.final(d1)


# --- 3. LOSS FUNKCE (Dice + BCE) ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # BCE
        bce_loss = self.bce(inputs, targets)

        # Dice
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice

        return 0.5 * bce_loss + 0.5 * dice_loss


# --- 4. MAIN LOOP ---
def main():
    # Cesty k datům - UPRAV SI
    train_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    train_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels"

    # Dataset check
    if not os.path.exists(train_img_dir):
        print(f"Error: Directory {train_img_dir} not found. Create folders first.")
        return

    dataset = ACLDataset(train_img_dir, train_mask_dir, CONFIG['patch_size'], CONFIG['samples_per_epoch'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0,
                            pin_memory=True)  # num_workers=0 na Windows často nutnost

    model = LightUNet3D(in_ch=1, out_ch=1, base=CONFIG['base_filters']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    criterion = DiceBCELoss()
    scaler = GradScaler()  # Pro Mixed Precision

    print("Starting training...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for i, (images, masks) in enumerate(pbar):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])

            # Mixed Precision Forward
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / CONFIG['accum_iter']  # Normalizace gradientu

            # Backward
            scaler.scale(loss).backward()

            if (i + 1) % CONFIG['accum_iter'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * CONFIG['accum_iter']
            pbar.set_postfix({'loss': loss.item() * CONFIG['accum_iter']})

        print(f"Epoch {epoch + 1} finished. Avg Loss: {epoch_loss / len(dataloader):.4f}")

        # Uložíme model PO KAŽDÉ epoše (přepíše se ten samý soubor, šetří místo)
        torch.save(model.state_dict(), "acl_model_latest.pth")

        # Volitelně: Uložíme checkpoint každých 10 epoch, kdybychom se chtěli vracet
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"acl_model_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved: acl_model_epoch_{epoch + 1}.pth")

        # Tady končí cyklus for.
        # Pro jistotu uložíme finální verzi ještě jednou pod jiným jménem
    torch.save(model.state_dict(), "acl_model_final.pth")
    print("Trénink dokončen. Model uložen jako 'acl_model_final.pth'")


if __name__ == "__main__":
    main()