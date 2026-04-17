import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings('ignore')


# ==================== KONFIGURACE ====================
class Config:
    # Cesta k natrénovanému modelu (uprav pokud se jmenuje jinak)
    MODEL_PATH = Path("outputs/late_model.pth")

    # --- ZDE JE TVŮJ KONKRÉTNÍ SOUBOR ---
    # Používám r"" pro raw string, aby Python neřešil zpětná lomítka
    INPUT_FILE = Path(
        r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_03\zdrave\pd_space_sag_p4_iso\14_pd_space_sag_p4_iso.nii.gz")

    # Kam uložit výsledek
    OUTPUT_DIR = Path("Segmentace")

    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    ENCODER = "resnet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()


# ==================== POMOCNÉ FUNKCE ====================
def load_nifti_file(filepath: Path):
    try:
        obj = nib.load(str(filepath))
    except nib.filebasedimages.ImageFileError:
        obj = nib.Nifti1Image.from_filename(str(filepath), mmap=False)
    return obj


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    lower = np.percentile(volume, 0.5)
    upper = np.percentile(volume, 99.5)
    volume = np.clip(volume, lower, upper)
    std = volume.std()
    mean = volume.mean()
    if std > 1e-8:
        return (volume - mean) / std
    return volume - mean


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


def paste_center(full_shape, crop_mask, crop_size):
    full_h, full_w = full_shape
    full_mask = np.zeros((full_h, full_w), dtype=np.float32)
    start_h = (full_h - crop_size) // 2
    start_w = (full_w - crop_size) // 2

    if start_h >= 0 and start_w >= 0:
        full_mask[start_h:start_h + crop_size, start_w:start_w + crop_size] = crop_mask
    return full_mask


# ==================== DATASET ====================
class SingleVolumeDataset(Dataset):
    def __init__(self, volume):
        self.volume = volume
        self.n_slices = volume.shape[2]
        self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return self.n_slices - 2

    def __getitem__(self, idx):
        real_slice_idx = idx + 1
        stack = []
        for offset in [-1, 0, 1]:
            s = self.volume[:, :, real_slice_idx + offset]
            s = center_crop(s, config.IMAGE_SIZE)
            stack.append(s)

        image = np.stack(stack, axis=-1).astype(np.float32)
        augmented = self.transform(image=image)["image"]
        return augmented, real_slice_idx


# ==================== POUZE TENTO SOUBOR ====================
def predict_single_file():
    if not config.INPUT_FILE.exists():
        print(f"CHYBA: Soubor neexistuje: {config.INPUT_FILE}")
        return

    # 1. Model
    print(f"Načítám model...")
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(config.DEVICE)

    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    # 2. Načtení konkrétního souboru
    print(f"Zpracovávám: {config.INPUT_FILE.name}")
    nifti_obj = load_nifti_file(config.INPUT_FILE)
    original_affine = nifti_obj.affine
    raw_volume = nifti_obj.get_fdata().astype(np.float32)

    # Preprocessing
    norm_volume = zscore_normalize(raw_volume)
    full_mask_volume = np.zeros(raw_volume.shape, dtype=np.uint8)

    ds = SingleVolumeDataset(norm_volume)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Predikce
    with torch.no_grad():
        for images, slice_indices in tqdm(loader, desc="Inference"):
            images = images.to(config.DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()

            for i in range(len(slice_indices)):
                slice_idx = slice_indices[i].item()
                pred_slice = preds[i, 0, :, :]

                original_h, original_w = raw_volume.shape[:2]
                restored_slice = paste_center((original_h, original_w), pred_slice, config.IMAGE_SIZE)

                full_mask_volume[:, :, slice_idx] = restored_slice.astype(np.uint8)

    # 4. Uložení
    output_filename = config.OUTPUT_DIR / f"{config.INPUT_FILE.stem}_segmentation.nii.gz"
    new_nifti = nib.Nifti1Image(full_mask_volume, original_affine)
    nib.save(new_nifti, output_filename)

    print(f"HOTOVO. Uloženo zde: {output_filename.absolute()}")


if __name__ == "__main__":
    predict_single_file()