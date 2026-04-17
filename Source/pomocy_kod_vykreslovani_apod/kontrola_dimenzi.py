import nibabel as nib
import glob

# Zkontroluj prvních pár masek
masks = glob.glob(r"C:\DIPLOM_PRACE\ACL_segment\data_train\images_pcl\*.nii*")
for m in masks:
    shape = nib.load(m).shape
    print(f"{m.split('/')[-1]}: {shape}")

