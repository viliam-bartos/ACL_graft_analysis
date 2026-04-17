import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import os
from tqdm import tqdm
import warnings
import multiprocessing

warnings.filterwarnings('ignore')


# ==================== KONFIGURACE ====================
class Config:
    # --- UPRAVENÉ CESTY ---
    TRAIN_IMG_DIR = Path(r"C:\DIPLOM_PRACE\ACL_segment\data_train\images")
    TRAIN_MASK_DIR = Path(r"C:\DIPLOM_PRACE\ACL_segment\data_train\labels")

    RADIMAGENET_WEIGHTS = Path(r"C:\Users\vilia\Downloads\RadImageNet_pytorch\RadImageNet_pytorch\ResNet50.pt")
    OUTPUT_DIR = Path("outputs")

    # Hyperparametry
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Na Windows nastav max 2-4. Pokud to padá, vrať se k 0.
    # Python na Windows používá 'spawn', což je pomalé a náročné na paměť.
    NUM_WORKERS = 4

    STACK_SIZE = 3
    ENCODER = "resnet50"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()


# ==================== AUGMENTACE ====================
def get_training_augmentation():
    return A.Compose([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
        ToTensorV2()
    ])


def get_validation_augmentation():
    return A.Compose([ToTensorV2()])


# ==================== PREPROCESSING FUNKCE ====================
def load_nifti_file(filepath: Path) -> np.ndarray:
    try:
        obj = nib.load(str(filepath))
    except nib.filebasedimages.ImageFileError:
        obj = nib.Nifti1Image.from_filename(str(filepath), mmap=False)
    return obj.get_fdata().astype(np.float32)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    lower = np.percentile(volume, 0.5)
    upper = np.percentile(volume, 99.5)
    volume = np.clip(volume, lower, upper)
    std = volume.std()
    mean = volume.mean()
    if std > 1e-8:
        return (volume - mean) / std
    return volume - mean


def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    lower = np.percentile(volume, 0.5)
    upper = np.percentile(volume, 99.5)
    volume = np.clip(volume, lower, upper)

    # Převedení na rozsah 0-1
    diff = upper - lower
    if diff < 1e-8:
        return np.zeros_like(volume)

    return (volume - lower) / diff

def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)))
    h, w = image.shape
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return image[start_h:start_h + size, start_w:start_w + size]


# ==================== DATASET (2.5D Stack) ====================
class ACL_2_5D_Dataset(Dataset):
    def __init__(self, volume_list, mask_list, transform=None):
        self.samples = []
        self.volumes = volume_list
        self.masks = mask_list
        self.transform = transform

        for vol_idx, (vol, msk) in enumerate(zip(self.volumes, self.masks)):
            n_slices = vol.shape[2]
            for i in range(1, n_slices - 1):
                self.samples.append((vol_idx, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.samples[idx]
        vol = self.volumes[vol_idx]
        msk = self.masks[vol_idx]

        stack = []
        for offset in [-1, 0, 1]:
            s = vol[:, :, slice_idx + offset]
            s = center_crop(s, config.IMAGE_SIZE)
            stack.append(s)

        image = np.stack(stack, axis=-1)
        mask = msk[:, :, slice_idx]
        mask = center_crop(mask, config.IMAGE_SIZE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image.float(), mask.float()


# ==================== BALANCOVÁNÍ ====================
def get_sampler_weights(dataset):
    targets = []
    print("Počítám statistiky pro sampler (to chvíli trvá)...")
    for _, mask in tqdm(dataset):
        has_acl = mask.sum() > 0
        targets.append(1 if has_acl else 0)

    targets = np.array(targets)
    count_pos = targets.sum()
    count_neg = len(targets) - count_pos

    if count_pos == 0:
        return None

    weight_pos = 1.0 / count_pos
    weight_neg = 1.0 / count_neg
    weights = np.where(targets == 1, weight_pos, weight_neg)
    weights = np.where(targets == 1, weights * 2.0, weights)

    return torch.DoubleTensor(weights)


# ==================== MODEL LOADER ====================
def create_model(pretrained_path: Path):
    print("Vytvářím model U-Net s ResNet50...")
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    if not pretrained_path.exists():
        print(f"POZOR: Váhy {pretrained_path} nenalezeny. Trénuji od nuly.")
        return model

    print(f"Načítám RadImageNet váhy: {pretrained_path.name}")
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if "fc." not in name:
                new_state_dict[name] = v
        model.encoder.load_state_dict(new_state_dict, strict=False)
        print("Váhy úspěšně nahrány do encoderu.")
    except Exception as e:
        print(f"Chyba při načítání vah: {e}")

    return model


# ==================== LOSS & METRIKY ====================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 0.5 * bce + 0.5 * (1 - dice)


def dice_coef(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2. * inter + 1e-5) / (pred.sum() + target.sum() + 1e-5)


# ==================== MAIN ====================
def main():
    # Fix pro multiprocessing na Windows, pokud by se spouštěl jinde
    if os.name == 'nt':
        multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Zařízení: {device}")

    # 1. NAČTENÍ SOUBORŮ
    image_files = sorted(list(config.TRAIN_IMG_DIR.glob("*.nii.gz")))
    mask_files = sorted(list(config.TRAIN_MASK_DIR.glob("*.nii.gz")))

    print(f"Nalezeno {len(image_files)} obrázků a {len(mask_files)} masek.")
    if len(image_files) == 0:
        return

    # 2. SPLIT
    val_count = 1
    train_files = image_files[:-val_count]
    val_files = image_files[-val_count:]
    train_masks = mask_files[:-val_count]
    val_masks = mask_files[-val_count:]

    # Funkce pro načítání musí být v main scope nebo top-level
    train_vols = [min_max_normalize(load_nifti_file(f)) for f in tqdm(train_files, desc="Načítám Train MRI")]
    train_msks = [(load_nifti_file(f) > 0).astype(np.float32) for f in tqdm(train_masks, desc="Načítám Train Masky")]
    val_vols = [min_max_normalize(load_nifti_file(f)) for f in tqdm(val_files, desc="Načítám Val MRI")]
    val_msks = [(load_nifti_file(f) > 0).astype(np.float32) for f in tqdm(val_masks, desc="Načítám Val Masky")]

    # 3. DATASETY
    train_ds = ACL_2_5D_Dataset(train_vols, train_msks, transform=get_training_augmentation())
    val_ds = ACL_2_5D_Dataset(val_vols, val_msks, transform=get_validation_augmentation())

    train_weights = get_sampler_weights(train_ds)
    sampler = WeightedRandomSampler(train_weights, len(train_weights)) if train_weights is not None else None

    # 4. DATALOADERS - ZDE JE ZMĚNA PRO WINDOWS
    # persistent_workers=True je nutnost, jinak se procesy neustále restartují a zpomalí to trénink.
    use_persistent_workers = config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    # 5. MODEL & OPTIM
    model = create_model(config.RADIMAGENET_WEIGHTS).to(device)

    # Zmrazit začátek
    for name, param in model.encoder.named_parameters():
        # Odmrazíme vrstvy layer3 a layer4, zbytek (layer1, layer2, stem) necháme zmrazený
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optimizer musí vidět jen parametry s requires_grad=True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    criterion = DiceBCELoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    # 6. SMYČKA
    best_dice = 0.0
    print("Startuji trénink...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_d = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}", leave=True)
        for img, mask in pbar:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device.type):
                out = model(img)
                loss = criterion(out, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            d = dice_coef(out, mask).item()
            train_d += d
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{d:.4f}"})

        # Validace
        model.eval()
        val_d = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                with autocast(device.type):
                    out = model(img)
                val_d += dice_coef(out, mask).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_dice = val_d / len(val_loader)

        print(f"Ep {epoch + 1} | T_Loss: {avg_train_loss:.4f} | V_Dice: {avg_val_dice:.4f}")

        scheduler.step(avg_val_dice)

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), config.OUTPUT_DIR / "best_model.pth")
            print(">>> Saved Best!")

        torch.save(model.state_dict(), config.OUTPUT_DIR / "latest_model.pth")


if __name__ == "__main__":
    main()