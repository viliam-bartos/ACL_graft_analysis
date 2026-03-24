import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
from pathlib import Path
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
    Closing + Largest Connected Component pro každou třídu zvlášť.
    """
    final_mask = np.zeros_like(mask)
    struct = np.ones((3, 3, 3), dtype=bool)

    # 0: Background, 1: ACL, 2: Femur, 3: Tibia
    for class_id in [1, 2, 3]:
        class_mask = (mask == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
             continue
             
        mask_closed = binary_closing(class_mask, structure=struct).astype(np.uint8)
        labeled_array, num_features = label(mask_closed)

        if num_features == 0:
            continue
        if num_features == 1:
            final_mask[mask_closed == 1] = class_id
            continue

        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        max_label = sizes.argmax()
        cleaned_mask = (labeled_array == max_label).astype(np.uint8)
        final_mask[cleaned_mask == 1] = class_id

    return final_mask


def process_patient(model, input_path, output_path, transforms, device):
    print(f"Zpracovávám: {input_path.name}")

    data_dict = [{"image": str(input_path)}]
    ds = Dataset(data=data_dict, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    for batch in loader:
        input_tensor = batch["image"].to(device)

        try:
            original_affine = batch["image"].affine[0].numpy()
        except:
            original_affine = np.eye(4)

        with torch.no_grad():
            output_tensor = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(128, 128, 32),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode='constant',
                device=device
            )
            output_mask = torch.argmax(output_tensor, dim=1)

        pred_array = output_mask.cpu().numpy()[0, :, :, :].astype(np.uint8)

        # Pojistka pro navrácení původních rozměrů z důvodu SpatialPadd
        orig_nii = nib.load(str(input_path))
        orig_shape = orig_nii.shape
        pred_array = pred_array[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

        final_mask = postprocess_mask(pred_array)

        out_nifti = nib.Nifti1Image(final_mask, original_affine)
        nib.save(out_nifti, str(output_path))
        print(f"  Uloženo: {output_path.name}")


# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = r"C:\DIPLOM_PRACE\ACL_segment\results_3D\best_model.pth"
    INPUT_DIR = Path(r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\predict")
    OUTPUT_DIR = Path(r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\base32")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(MODEL_PATH):
        print(f"Chyba: Nenalezen model ({MODEL_PATH})")
        exit()

    print("Načítám model...")
    # 4 kanály a 32 base filters dle nové architektury
    model = LightUNet3D(in_ch=1, out_ch=4, base=32).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 32)),
    ])

    files = [f for f in INPUT_DIR.rglob("*") if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))]

    if not files:
        print(f"Ve složce {INPUT_DIR} nejsou žádné NIfTI soubory k predikci.")
    else:
        print(f"Nalezeno {len(files)} souborů. Spouštím hromadnou inferenci...")
        for file_path in files:
            out_path = OUTPUT_DIR / f"mask_{file_path.name}"
            process_patient(model, file_path, out_path, transforms, DEVICE)

    print("Hromadná inference dokončena.")