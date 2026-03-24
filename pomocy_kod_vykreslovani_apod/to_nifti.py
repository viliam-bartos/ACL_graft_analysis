import os
import dicom2nifti
import nibabel as nib
import numpy as np
import glob

# === CESTY ===
work_dir = r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_05\zdrave\pd_space_sag_p4_iso"
mask_filename = "Pacient_05_zdrave.nii"



# 2. KONVERZE BEZ REORIENTACE
print(f"Konvertuji DICOM (reorient=False)...")
try:
    # Tady je to kouzlo: reorient=False
    dicom2nifti.convert_directory(work_dir, work_dir, compression=True, reorient=False)
except Exception as e:
    print(f"Chyba: {e}")

# Najdeme ten nový soubor
generated_files = glob.glob(os.path.join(work_dir, "*.nii.gz"))
image_files = [f for f in generated_files if "mask" not in f.lower()]

if not image_files:
    print("Něco se pokazilo, soubor nevznikl.")
    exit()

img_path = image_files[0]
print(f"Nový soubor: {os.path.basename(img_path)}")

# 3. KONTROLA ROZMĚRŮ
mask_path = os.path.join(work_dir, mask_filename)
img_nii = nib.load(img_path)
mask_nii = nib.load(mask_path)

img_data = img_nii.get_fdata()
mask_data = mask_nii.get_fdata()

print("------------------------------------------------")
print(f"Obraz shape (reorient=False): {img_data.shape}")
print(f"Maska shape:                  {mask_data.shape}")
print("------------------------------------------------")

if img_data.shape == mask_data.shape:
    print("BINGO! Rozměry sedí 1:1.")
    print("Nemusíš nic transponovat ani flipovat.")
    print("Tento nový soubor použij pro síť.")
else:
    print("Sakra. Stále to nesedí. ITK-SNAP asi při ukládání masky taky něco přeorientoval.")