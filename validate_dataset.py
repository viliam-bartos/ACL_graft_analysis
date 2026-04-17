import os
import glob
import numpy as np
import nibabel as nib

def check_dataset(image_dir, mask_dir):
    print(f"Kontroluji dataset:\n - Obrazy: {image_dir}\n - Masky: {mask_dir}\n")
    
    # Získání seznamu souborů
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))
    
    # Vytvoření slovníků pro snadnější spárování (odstranění přípon a prefixu map_ u masek)
    def get_id(filepath, is_mask=False):
        name = os.path.basename(filepath)
        if name.endswith('.nii.gz'):
            name = name[:-7]
        elif name.endswith('.nii'):
            name = name[:-4]
            
        if is_mask and name.startswith('mask_'):
            name = name[5:]
        return name

    images_dict = {get_id(p, is_mask=False): p for p in image_paths}
    masks_dict = {get_id(p, is_mask=True): p for p in mask_paths}
    
    print(f"Nalezeno {len(images_dict)} snímků a {len(masks_dict)} masek.\n")
    
    missing_masks = []
    
    # 1. Kontrola, zda má každý nifti obraz svou masku
    for base_name in images_dict:
        if base_name not in masks_dict:
            missing_masks.append(base_name)
    
    if missing_masks:
        print(f"[CHYBA] Chybí masky pro následující snímky ({len(missing_masks)}):")
        for m in missing_masks[:10]: # Vypíšeme prvních 10 abychom nespamovali
            print(f"  - {m}")
        if len(missing_masks) > 10: print("  - ... a další ...")
    else:
        print("[OK] Všechny snímky mají svou masku.")
        
    print("\n--- Kontrola jednotlivých párů (Obraz + Maska) ---")
    
    valid_pairs = 0
    issues_found = 0
    
    for base_name, img_path in images_dict.items():
        if base_name not in masks_dict:
            continue
            
        mask_path = masks_dict[base_name]
        
        try:
            img = nib.load(img_path)
            mask = nib.load(mask_path)
            
            img_data = img.get_fdata()
            # Maska by se měla načítat jako dataobj (příp. integer), nikoliv jako float přes get_fdata()
            mask_data = np.asanyarray(mask.dataobj)
            
            issues = []
            
            # a) Kontrola, zda maska sedí na obraz (rozměry)
            if img.shape != mask.shape:
                issues.append(f"Rozměry se neshodují! Obraz: {img.shape}, Maska: {mask.shape}")
                
            # b) Kontrola afinní matice (aby maska nebyla posunutá v prostoru)
            if not np.allclose(img.affine, mask.affine, atol=1e-3):
                issues.append("Afinní matice (orientace/spacing) obrazu a masky se neshodují!")
                
            # c) Kontrola 4 tříd (pozadí=0, acl=1, femur=2, tibia=3)
            unique_classes = np.unique(mask_data)
            expected_classes = {0, 1, 2, 3}
            
            if not set(unique_classes).issubset(expected_classes):
                issues.append(f"Maska obsahuje nepovolené třídy/hodnoty! Nalezeno: {list(unique_classes)}")
            
            # (volitelné varování) Kontrola chybějících tříd (co když algoritmus/člověk zapomněl nakreslit ACL?)
            if len(unique_classes) < min(4, len(expected_classes)):
                missing_classes = expected_classes.difference(set(unique_classes))
                issues.append(f"Upozornění: V masce chybí některé třídy: {list(missing_classes)}")
                
            # d) Kontrola poškozených/neplatných hodnot v obrazu
            if np.isnan(img_data).any() or np.isinf(img_data).any():
                issues.append("Obraz obsahuje neplatné NaN nebo Inf hodnoty!")
                
            # e) Kontrola 3D dimenze (aby tam náhodou nebyl přidán 4. time/color kanál)
            if len(img.shape) != 3:
                issues.append(f"Tenzor nemá 3 dimenze, ale {len(img.shape)} (shape: {img.shape}).")
                
            if issues:
                issues_found += 1
                print(f"\n[!] Problémy u snímku '{base_name}':")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                valid_pairs += 1
            
        except Exception as e:
            issues_found += 1
            print(f"\n[CHYBA] Nelze načíst nebo zpracovat '{base_name}': {e}")
            
    print(f"\n--- Shrnutí ---")
    print(f"Zkontrolováno párů:    {len(images_dict) - len(missing_masks)}")
    print(f"Párů bez problémů:     {valid_pairs}")
    print(f"Párů s problémy:       {issues_found}")
    print(f"Snímků bez masky:      {len(missing_masks)}")

if __name__ == "__main__":
    # --- ZDE ZADEJ SVÉ CESTY ---
    IMG_DIR = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\images_train_full_canonical"
    MASK_DIR = r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\masks_train_full_canonical"
    # ---------------------------
    
    check_dataset(IMG_DIR, MASK_DIR)
