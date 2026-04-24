import torch
import torch.nn as nn
import os
import numpy as np
import nibabel as nib  # Potřeba pro uložení
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd, SpatialPadd
)
from monai.data import Dataset, DataLoader

# ==========================================
# 1. MANUÁLNÍ NASTAVENÍ CEST
# ==========================================
# Vstupní data
IMG_PATH = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\images_train_full_canonical\case_001.nii.gz"
MASK_PATH = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\masks_train_full_canonical\mask_case_001.nii.gz"

# Kam uložit výsledek
OUTPUT_PATH = "case001_RAW_optuna.nii.gz"

# Cesta k modelu
MODEL_PATH = r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\Optuna_best_model_150ep\best_model_trial_1.pth"

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patch_size': (128, 128, 64),
}


# ==========================================
# 2. MODEL
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
    def __init__(self, in_ch=1, out_ch=4, base=64):
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

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bn = self.bottleneck(self.pool(x3))
        d3 = self.dec3(torch.cat([self.reduce3(self.up3(bn)), x3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), x2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), x1], dim=1))
        return self.final(d1)


# ==========================================
# 3. EXEKUCE
# ==========================================
def main():
    if not os.path.exists(IMG_PATH) or not os.path.exists(MASK_PATH):
        print("CHYBA: Soubory neexistují.")
        return

    print(f"Validuji soubor: {os.path.basename(IMG_PATH)}")

    data_dict = [{"image": IMG_PATH, "label": MASK_PATH}]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=CONFIG['patch_size']),
    ])

    ds = Dataset(data=data_dict, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = LightUNet3D(in_ch=1, out_ch=4, base=64).to(CONFIG['device'])
    checkpoint = torch.load(MODEL_PATH, map_location=CONFIG['device'])
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(CONFIG['device'])
            label_gt = batch["label"].to(CONFIG['device'])

            output = sliding_window_inference(
                inputs=img,
                roi_size=CONFIG['patch_size'],
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode='gaussian'
            )

            pred = torch.argmax(output, dim=1)

            print("=" * 30)
            # DICE pro všechny třídy
            class_names = {1: "ACL", 2: "Femur", 3: "Tibie"}
            for class_id, class_name in class_names.items():
                pred_c = (pred == class_id).float()
                label_c = (label_gt.squeeze(1) == class_id).float()
                
                inter = (pred_c * label_c).sum()
                union = pred_c.sum() + label_c.sum()
                dice = 2.0 * inter / union if union > 0 else torch.tensor(1.0)
                print(f"DICE SKÓRE ({class_name}): {dice.item():.6f}")
            print("=" * 30)

            print(f"Ukládám masku do: {OUTPUT_PATH}")

            pred_np = pred.cpu().numpy()[0].astype(np.uint8)

            # --- OPRAVA ZDE ---
            # V novém MONAI je 'affine' přímo vlastností MetaTensoru (img), ne ve slovníku
            try:
                # Zkusíme moderní přístup (MetaTensor)
                original_affine = batch["image"].affine[0].numpy()
            except AttributeError:
                # Fallback pro starší verze nebo pokud se metadata ztratila (méně pravděpodobné s LoadImaged)
                print("Varování: MetaTensor affine nenalezen, používám identitu.")
                original_affine = np.eye(4)

            out_nifti = nib.Nifti1Image(pred_np, original_affine)
            nib.save(out_nifti, OUTPUT_PATH)
            print("Hotovo.")


if __name__ == "__main__":
    main()