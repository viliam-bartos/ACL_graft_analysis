import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

# --- NASTAVENÍ ---
CONFIG = {
    'model_path': r'C:\DIPLOM_PRACE\ACL_segment\results_2_5D_dice\best_model.pth',
    'input_nii': r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_05\zdrave\pd_space_sag_p4_iso\case005z.nii",
    'output_nii': 'prediction_pacient_05.nii.gz',
    'img_size': (256, 256),  # Musí sedět s tréninkem
    'base_filters': 16,  # Musí sedět s tréninkem (v CONFIGu tréninku bylo 16)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'threshold': 0.5
}


# --- 1. DEFINICE MODELU (Identická s tréninkem) ---
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
        # Encoder (3 blocks)
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.dropout = nn.Dropout2d(p=0.2)  # Dropout při inferenci nevadí (bude vypnutý přes model.eval())

        # Decoder
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


# --- 2. PREPROCESSING A REKONSTRUKCE ---
def get_inference_transforms(img_size):
    """Stejná transformace jako VALIDATION v tréninku."""
    return A.Compose([
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=img_size[0], width=img_size[1]),
        ToTensorV2()
    ])


def paste_center(prediction_256, original_shape):
    """
    Vrátí predikci 256x256 zpět do středu původního rozlišení (např. 320x320).
    Inverzní operace k CenterCrop.
    """
    orig_h, orig_w = original_shape
    pred_h, pred_w = prediction_256.shape

    # Vytvoř prázdné plátno o původní velikosti
    full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

    # Spočítej souřadnice středu
    start_y = (orig_h - pred_h) // 2
    start_x = (orig_w - pred_w) // 2

    # Ošetření, pokud je predikce větší než originál (nepravděpodobné u MRI, ale pro jistotu)
    if start_y < 0 or start_x < 0:
        # Tady by se muselo ořezávat, ale předpokládáme orig > 256
        return cv2.resize(prediction_256, (orig_w, orig_h))

    full_mask[start_y:start_y + pred_h, start_x:start_x + pred_w] = prediction_256
    return full_mask


# --- 3. INFERENCE ---
def run_inference():
    print(f"Device: {CONFIG['device']}")

    # 1. Načtení modelu
    model = UNet25D(in_ch=3, out_ch=1, base=CONFIG['base_filters']).to(CONFIG['device'])

    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])

        # Ošetření různých formátů uložení
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Detected checkpoint dict, loading 'model_state_dict'...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Detected raw state_dict...")
            model.load_state_dict(checkpoint)

    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        print("Zkontroluj cestu k modelu a shodu architektury (base_filters).")
        return

    model.eval()

    # 2. Načtení dat
    print(f"Reading MRI: {CONFIG['input_nii']}")
    nii = nib.load(CONFIG['input_nii'])
    volume = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    header = nii.header

    # 3. Globální normalizace (Stejná jako v ACLDataset25D)
    print("Normalizing volume...")
    volume = np.clip(volume, np.percentile(volume, 0.5), np.percentile(volume, 99.5))
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

    # Shape: (H, W, D) -> (D, H, W) pro iteraci
    volume = volume.transpose(2, 0, 1)
    depth, orig_h, orig_w = volume.shape

    # Výstupní pole v PŮVODNÍM rozlišení
    final_mask_volume = np.zeros((depth, orig_h, orig_w), dtype=np.uint8)

    transform = get_inference_transforms(CONFIG['img_size'])

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(depth)):
            # 2.5D Stack
            stack = []
            for offset in [-1, 0, 1]:
                z = np.clip(i + offset, 0, depth - 1)
                stack.append(volume[z])

            # (256, 256, 3) -> transformace to chce channels last
            img_25d = np.stack(stack, axis=-1)

            # Albumentations transform
            # Dummy maska, protože transform vyžaduje 'mask' argument, když je v pipeline
            dummy_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            augmented = transform(image=img_25d, mask=dummy_mask)

            img_tensor = augmented['image'].unsqueeze(0).to(CONFIG['device'])  # (1, 3, 256, 256)

            # Predikce
            output = model(img_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()  # (256, 256)

            # Rekonstrukce do původního rozměru
            # Pokud jsi trénoval s CenterCrop, musíme výsledek vložit doprostřed černé masky
            full_slice_prob = paste_center(prob, (orig_h, orig_w))

            # Binarizace
            mask_slice = (full_slice_prob > CONFIG['threshold']).astype(np.uint8)
            final_mask_volume[i] = mask_slice

    # 4. Uložení
    # Zpět na (H, W, D)
    final_mask_volume = final_mask_volume.transpose(1, 2, 0)

    print(f"Saving prediction to: {CONFIG['output_nii']}")
    new_img = nib.Nifti1Image(final_mask_volume, affine, header)
    nib.save(new_img, CONFIG['output_nii'])
    print("Done.")


if __name__ == "__main__":
    run_inference()