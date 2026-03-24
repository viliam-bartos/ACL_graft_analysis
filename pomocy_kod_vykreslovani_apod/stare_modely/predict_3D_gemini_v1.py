import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label, binary_closing

# ==========================================
# 1. MODEL (beze změny)
# ==========================================
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
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base * 2, base)
        self.final = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3 = self.pool(x3)
        bn = self.bottleneck(p3)
        u3 = self.up3(bn)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)
        return self.final(d1)


# ==========================================
# 2. LOGIKA OŘEZU A OBNOVY
# ==========================================

def get_crop_coords(data_shape, crop_size):
    """Vypočítá souřadnice pro středový ořez."""
    x, y, z = data_shape
    cx, cy, cz = crop_size

    startx = max(0, x // 2 - (cx // 2))
    starty = max(0, y // 2 - (cy // 2))
    startz = max(0, z // 2 - (cz // 2))

    # Konec ořezu (nesmí přesáhnout rozměr dat)
    endx = min(x, startx + cx)
    endy = min(y, starty + cy)
    endz = min(z, startz + cz)

    return (startx, endx, starty, endy, startz, endz)


def crop_volume(data, coords, crop_size):
    """Vrátí oříznutá data (případně doplněná nulami, pokud je vstup malý)."""
    sx, ex, sy, ey, sz, ez = coords
    cropped = data[sx:ex, sy:ey, sz:ez]

    # Pokud je ořez menší než target (třeba na krajích), doplníme padding
    if cropped.shape != crop_size:
        pad_dims = []
        for i in range(3):
            diff = crop_size[i] - cropped.shape[i]
            pad_dims.append((0, max(0, diff)))
        cropped = np.pad(cropped, pad_dims, 'constant')

    return cropped


def postprocess_mask(mask):
    """
    1. Spojí malé mezery (Morphological Closing).
    2. Zachová pouze největší souvislý objekt (Largest Connected Component).
    """
    print("  Aplikuji morfologické čištění...")

    # 1. Morfologické uzavření (Closing)
    # Spojí kousky, které jsou u sebe blíž než kernel (např. mezera 1 voxel)
    struct = np.ones((3, 3, 3), dtype=bool)
    mask_closed = binary_closing(mask, structure=struct).astype(np.uint8)

    # 2. Hledání souvislých komponent
    labeled_array, num_features = label(mask_closed)

    if num_features == 0:
        return mask  # Prázdná maska, není co čistit

    if num_features == 1:
        return mask_closed  # Jen jeden objekt, super

    # Spočítáme velikost každé komponenty
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Ignorujeme pozadí

    # Najdeme label s největším počtem voxelů
    max_label = sizes.argmax()

    # Vytvoříme novou masku jen s tímto labelem
    cleaned_mask = (labeled_array == max_label).astype(np.uint8)

    return cleaned_mask


def restore_original_geometry(pred_mask_crop, original_shape, coords):
    """Vloží predikovaný crop zpět do velké matice nul."""
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    sx, ex, sy, ey, sz, ez = coords

    # Rozměry, kam budeme vkládat
    target_shape = (ex - sx, ey - sy, ez - sz)

    # Ořízneme predikci, pokud byla při inputu padovaná (zbavíme se paddingu)
    actual_pred = pred_mask_crop[:target_shape[0], :target_shape[1], :target_shape[2]]

    # Vlepení
    full_mask[sx:ex, sy:ey, sz:ez] = actual_pred
    return full_mask


# ==========================================
# 3. PIPELINE
# ==========================================

def process_patient(model_path, input_nifti_path, output_path, crop_size=(128, 128, 128), device='cuda'):
    print(f"Zpracovávám: {input_nifti_path}")

    # --- A. NAČTENÍ DAT ---
    try:
        nii = nib.load(input_nifti_path)
    except:
        nii = nib.Nifti1Image.from_filename(input_nifti_path, mmap=False)

    original_data = nii.get_fdata().astype(np.float32)
    original_affine = nii.affine
    original_shape = original_data.shape

    print(f"  Původní tvar: {original_shape}")

    # --- B. OŘEZ ---
    coords = get_crop_coords(original_shape, crop_size)
    crop_data = crop_volume(original_data, coords, crop_size)

    # Preprocessing cropu (Normalizace)
    lower = np.percentile(crop_data, 0.5)
    upper = np.percentile(crop_data, 99.5)
    crop_data = np.clip(crop_data, lower, upper)
    mean = np.mean(crop_data)
    std = np.std(crop_data)
    crop_data = (crop_data - mean) / (std + 1e-8)

    # --- C. INFERENCE ---
    # Inicializace modelu
    model = LightUNet3D(in_ch=1, out_ch=1, base=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Příprava tensoru
    input_tensor = torch.from_numpy(crop_data).unsqueeze(0).unsqueeze(0).to(device)

    print("  Běží inference...")
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

    # Post-processing cropu
    pred_crop = (output > 0.5).float().cpu().numpy()[0, 0, :, :, :].astype(np.uint8)

    pred_crop = postprocess_mask(pred_crop)

    # --- D. OBNOVENÍ DO PŮVODNÍ VELIKOSTI ---
    print("  Rekonstrukce do původního objemu...")
    full_mask = restore_original_geometry(pred_crop, original_shape, coords)

    # --- E. ULOŽENÍ ---
    out_nifti = nib.Nifti1Image(full_mask, original_affine)
    nib.save(out_nifti, output_path)
    print(f"  Uloženo: {output_path} (Shape: {full_mask.shape})")


# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    MODEL = r"C:\DIPLOM_PRACE\ACL_segment\outputs\light_3D_v2_4pacienti_1_validace_0_57.pth"
    INPUT = r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_05\zdrave\pd_space_sag_p4_iso\case005z.nii"
    OUTPUT = "mask_005_geminiV2.nii.gz"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(MODEL) and os.path.exists(INPUT):
        process_patient(MODEL, INPUT, OUTPUT, device=DEVICE)
    else:
        print("Chybí vstupní soubor nebo model.")