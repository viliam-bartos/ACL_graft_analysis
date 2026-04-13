import os
import glob
import csv
import argparse
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def get_laterality(mask_array):
    """
    Určí lateralitu (Levé/Pravé) na základě polohy ACL úponů.
    Předpokládáme, že maska má labely: 1 (ACL), 2 (Femur), 3 (Tibia).
    Úpon na femuru je vždy LATERÁLNÍ, úpon na tibii je MEDIÁLNÍ.
    """
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    
    f_dim0 = None
    t_dim0 = None
    
    if np.sum(femur_mask) > 0 and np.sum(tibia_mask) > 0 and np.sum(acl_mask) > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        acl_dilated = ndimage.binary_dilation(acl_mask, structure=struct, iterations=2)
        femoral_contact = acl_dilated & femur_mask
        tibial_contact = acl_dilated & tibia_mask
        
        if np.sum(femoral_contact) > 0 and np.sum(tibial_contact) > 0:
            f_z, f_y, f_x = ndimage.center_of_mass(femoral_contact)
            t_z, t_y, t_x = ndimage.center_of_mass(tibial_contact)
            f_dim0 = f_z
            t_dim0 = t_z
            
    if f_dim0 is None or t_dim0 is None:
        # Fallback jen na ACL, pokud masky kostí chybí
        acl_coords = np.argwhere(acl_mask)
        if len(acl_coords) == 0:
            return "Unknown"
        s_mid = acl_coords[:, 1].mean() # osa 1 je kranio-kaudální (menší = superior)
        top_half = acl_coords[acl_coords[:, 1] < s_mid]
        bottom_half = acl_coords[acl_coords[:, 1] >= s_mid]
        
        if len(top_half) == 0 or len(bottom_half) == 0:
            return "Unknown"
            
        f_dim0 = top_half[:, 0].mean() # Femoral (lateral)
        t_dim0 = bottom_half[:, 0].mean() # Tibial (medial)

    # f_dim0 = laterální, t_dim0 = mediální. (Osa 0 je pravolevá)
    # V radiologické konvenci: pravá polovina obrazovky (vyšší index) je LEVÁ strana pacienta.
    # Tedy pokud t_dim0 > f_dim0 (mediální je na vyšším indexu než laterální): 
    #   znamená to, že laterální je vlevo. Koleno s laterální stranou na obrázku vlevo 
    #   je v radiologickém pohledu PRAVÉ koleno.
    if t_dim0 > f_dim0:
        return "Right"
    else:
        return "Left"


def process_dataset(img_dir, mask_dir, out_img_dir, out_mask_dir, csv_path):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(img_dir, "*.nii.gz"))
    
    results = []
    
    for img_path in img_files:
        basename = os.path.basename(img_path)
        # Ošetření možného jiného názvu masky podle `zkouska.py` (např. 'case_002.nii.gz' vs 'mask_case_002.nii.gz')
        # Zkusíme obě varianty
        mask_path = os.path.join(mask_dir, basename)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"mask_{basename}")
            
        if not os.path.exists(mask_path):
            print(f"[VAROVÁNÍ] Nenašel jsem masku pro {basename}, přeskakuji...")
            continue
            
        print(f"Zpracovávám: {basename}")
        
        img_sitk = sitk.ReadImage(img_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        img_array = sitk.GetArrayFromImage(img_sitk)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        laterality = get_laterality(mask_array)
        results.append({"ID": basename, "Laterality": laterality})
        
        print(f" -> Detekováno: {laterality} koleno")
        
        # Kanonizace: Chceme VŠECHNA kolena LEVÁ
        if laterality == "Right":
            print(" -> Překlápím na LEVÉ...")
            out_img_array = np.flip(img_array, axis=0)
            out_mask_array = np.flip(mask_array, axis=0)
        else:
            print(" -> Ponechávám (už je levé nebo neznámé)...")
            out_img_array = img_array
            out_mask_array = mask_array
            
        # Vytvoření a uložení
        out_img_sitk = sitk.GetImageFromArray(out_img_array)
        out_img_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(out_img_sitk, os.path.join(out_img_dir, basename))
        
        out_mask_sitk = sitk.GetImageFromArray(out_mask_array)
        out_mask_sitk.CopyInformation(mask_sitk)
        
        # Uložíme masku se stejným prefixem, jako měla vstupní
        out_mask_name = os.path.basename(mask_path)
        sitk.WriteImage(out_mask_sitk, os.path.join(out_mask_dir, out_mask_name))

    # Uložení do CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Laterality"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nHotovo! Výsledky zapsány do {csv_path}")


def main():
    # Nastav si vstupní složky
    img_dir = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images"
    mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\LABELS_TRAIN"
    
    # Výstupní složky (vytvoří se samy, pokud neexistují)
    out_img_dir = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\leva_kolena_images"
    out_mask_dir = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\leva_kolena_masks"
    
    # Kde se uloží report s detekovanými lateracemi
    csv_path = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\analyza_laterality.csv"
    
    process_dataset(img_dir, mask_dir, out_img_dir, out_mask_dir, csv_path)

if __name__ == "__main__":
    main()
