import os
import numpy as np
import SimpleITK as sitk

def main():
    # Tady si nastavíš cesty natvrdo
    input_path = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train_full\images\case_030.nii.gz"
    output_path = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\case_030_pretoceny.nii.gz"
    
    mask_path = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train_full\labels\mask_case_030.nii.gz"
    output_mask_path = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\mask_case_030_pretocena.nii.gz"

    print(f"Načítám MRI objem z: {input_path}")
    img_sitk = sitk.ReadImage(input_path)
    
    print(f"Načítám segmentační masku z: {mask_path}")
    mask_sitk = sitk.ReadImage(mask_path)
    
    # Získání dat v numpy poli.
    # Osa 0 (Z): Pravá - Levá podle main_acl_analysis.py
    # Osa 1 (Y): Horní - Dolní (S-I)
    # Osa 2 (X): Přední - Zadní (A-P)
    img_array = sitk.GetArrayFromImage(img_sitk)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    
    # Zrcadlení v koronálním pohledu -> převrácení pravo-levé osy (Axis 0).
    print("Provádím zrcadlení zleva doprava (překlopení osy 0)...")
    flipped_img_array = np.flip(img_array, axis=0)
    flipped_mask_array = np.flip(mask_array, axis=0)
    
    # Vytvoření nového SimpleITK image ze zrcadlených dat.
    # Uložíme zkopírované hlavičky pro vizualizační software.
    flipped_img_sitk = sitk.GetImageFromArray(flipped_img_array)
    flipped_img_sitk.CopyInformation(img_sitk)
    
    flipped_mask_sitk = sitk.GetImageFromArray(flipped_mask_array)
    flipped_mask_sitk.CopyInformation(mask_sitk)
    
    # Uložení kanonizovaného objemu a masky.
    print(f"Ukládám kanonizovaný objem do: {output_path}")
    sitk.WriteImage(flipped_img_sitk, output_path)
    
    print(f"Ukládám kanonizovanou masku do: {output_mask_path}")
    sitk.WriteImage(flipped_mask_sitk, output_mask_path)
    print("Hotovo!")

if __name__ == "__main__":
    main()
