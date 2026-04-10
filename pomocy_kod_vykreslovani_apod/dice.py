import nibabel as nib
import numpy as np


def calculate_dice(path_gt, path_pred):
    # Načtení dat
    gt = nib.load(path_gt).get_fdata().flatten()
    pred = nib.load(path_pred).get_fdata().flatten()

    # Binarizace (jistota, že máme 0 a 1)
    gt = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)

    # Výpočet Dice: 2 * průnik / součet
    intersection = np.sum(gt * pred)
    total = np.sum(gt) + np.sum(pred)

    if total == 0:
        return 1.0  # Obě masky prázdné = shoda

    return (2.0 * intersection) / total


if __name__ == "__main__":
    # --- UPRAV CESTY ZDE ---
    cesta1_moje = r"C:\DIPLOM_PRACE\ACL_segment\mask_case002_puv.nii.gz"
    cesta2_druha = r"C:\DIPLOM_PRACE\ACL_segment\mask_case001_puv.nii.gz"
    # -----------------------

    dice = calculate_dice(cesta1_moje, cesta2_druha)
    print(dice)