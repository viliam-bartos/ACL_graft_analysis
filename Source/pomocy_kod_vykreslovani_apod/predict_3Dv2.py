import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import label, binary_closing
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd, SpatialPadd
)
from monai.data import Dataset, DataLoader


# ==========================================
# 1. MODEL (LightUNet3D)
# ==========================================
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
    def __init__(self, in_ch=1, out_ch=4, base=32):
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


# ==========================================
# 2. LOGIKA ZPRACOVÁNÍ (S POSTPROCESSINGEM)
# ==========================================

def postprocess_mask(mask):
    """
    Closing + Largest Connected Component pro každou třídu zvlášť (1 = femur, 2 = tibie).
    """
    print("  Post-processing (Closing + LCC)...")
    
    final_mask = np.zeros_like(mask)
    struct = np.ones((3, 3, 3), dtype=bool)

    # 0: Background, 1: ACL, 2: Femur, 3: Tibia
    for class_id in [1, 2, 3]:
        class_mask = (mask == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
             continue
             
        # 1. Morfologické uzavření (Closing)
        mask_closed = binary_closing(class_mask, structure=struct).astype(np.uint8)

        # 2. Hledání souvislých komponent
        labeled_array, num_features = label(mask_closed)

        if num_features == 0:
            continue

        if num_features == 1:
            final_mask[mask_closed == 1] = class_id
            continue

        # Spočítáme velikost každé komponenty (ignorujeme pozadí 0)
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0

        # Najdeme label s největším počtem voxelů
        max_label = sizes.argmax()

        # Vytvoříme novou masku jen s tímto labelem
        cleaned_mask = (labeled_array == max_label).astype(np.uint8)
        final_mask[cleaned_mask == 1] = class_id

    return final_mask


def process_patient(model_path, input_nifti_path, output_path, device='cuda'):
    print(f"Zpracovávám: {input_nifti_path}")

    # --- A. DEFINICE TRANSFORMACÍ (MONAI NATIVE) ---
    # Tohle zaručí 100% shodu s tréninkem
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        # ZÁSADNÍ BOD: nonzero=True ignoruje černé pozadí při výpočtu průměru a směrodatné odchylky
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 32)),  # Patch size z configu
    ])

    # --- B. NAČTENÍ DAT PŘES DATASET ---
    data_dict = [{"image": input_nifti_path}]
    ds = Dataset(data=data_dict, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # --- C. MODEL ---
    # Musí přesně odpovídat parametrům v trénovacím skriptu (4 kanály, base=32)
    model = LightUNet3D(in_ch=1, out_ch=4, base=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("  Běží inference (Sliding Window)...")

    # --- D. INFERENCE ---
    # Iterujeme přes loader (je tam jen 1 obrázek)
    for batch in loader:
        input_tensor = batch["image"].to(device)

        # Získání původní orientace pro uložení (MetaTensor v novém MONAI)
        try:
            original_affine = batch["image"].affine[0].numpy()
        except:
            # Fallback
            original_affine = np.eye(4)

        with torch.no_grad():
            output_tensor = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(128, 128, 32),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode='constant',  # Pro inferenci lepší než constant, pokud chceš hezčí přechody
                device=device
            )
            output_mask = torch.argmax(output_tensor, dim=1)

        # Zpět na numpy [X, Y, Z] (odstraníme batch dimenzi, channel zmizel díky argmaxu)
        pred_array = output_mask.cpu().numpy()[0, :, :, :].astype(np.uint8)

        # --- E. POST-PROCESSING ---
        final_mask = postprocess_mask(pred_array)

        # --- F. ULOŽENÍ ---
        out_nifti = nib.Nifti1Image(final_mask, original_affine)
        nib.save(out_nifti, output_path)
        print(f"  Uloženo: {output_path}")


# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    MODEL = r"C:\DIPLOM_PRACE\ACL_segment\results_3D_v3\best_model.pth"
    INPUT = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images\case_007.nii.gz"
    OUTPUT = "mask_case_007.nii.gz"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(MODEL) and os.path.exists(INPUT):
        process_patient(MODEL, INPUT, OUTPUT, device=DEVICE)
    else:
        print(f"Chyba: Nenalezen model ({MODEL}) nebo vstupní data ({INPUT})")