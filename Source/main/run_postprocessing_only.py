import os
import glob
import logging
import SimpleITK as sitk
import numpy as np

# Importujeme existující funkce z vaší pipeliny
from mri_pipeline import postprocess_mask, perform_segmentation_analysis, setup_logging

def main():
    # 1. NASTAVENÍ CEST
    # Složka, kde máte aktuální masky z 5CV (bez postprocessingu)
    input_masks_dir = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\labels_eval_5cv_bez_postprocessingu"
    
    # Složka, kam se uloží vyčištěné masky a výsledky nové segmentační analýzy
    output_masks_dir = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\labels_eval_5cv_S_postprocessingem"
    
    # Ground Truth masky pro vyhodnocení
    gt_masks_dir = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\labels_optuna"
    
    # 2. NASTAVENÍ POSTPROCESSINGU
    post_proc_classes = {
        1: {"lcc": True, "hole_filling": False, "closing": False},
        2: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
        3: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2}
    }
    
    # ---------------------------------------------------------
    setup_logging(output_masks_dir, "postprocessing_only.log")
    os.makedirs(output_masks_dir, exist_ok=True)
    
    mask_files = glob.glob(os.path.join(input_masks_dir, "mask_*.nii.gz"))
    if not mask_files:
        logging.error(f"Ve složce {input_masks_dir} nebyly nalezeny žádné masky!")
        return
        
    logging.info(f"Nalezeno {len(mask_files)} masek k post-processingu.")
    
    for mask_path in mask_files:
        basename = os.path.basename(mask_path)
        logging.info(f"Čistím masku: {basename}")
        
        # Načtení hotové masky
        sitk_img = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(sitk_img)
        
        # Aplikace stejného postprocessingu jako v hlavní pipeline
        processed_arr = postprocess_mask(mask_arr, post_proc_classes)
        
        # Uložení s přesným zachováním původních metadat (origin, spacing, direction)
        out_sitk = sitk.GetImageFromArray(processed_arr)
        out_sitk.CopyInformation(sitk_img)
        
        out_path = os.path.join(output_masks_dir, basename)
        sitk.WriteImage(out_sitk, out_path)
        
    # 3. NOVÁ SEGMENTAČNÍ ANALÝZA
    logging.info("--- Post-processing dokončen. Spouštím novou evaluaci ---")
    perform_segmentation_analysis(output_masks_dir, gt_masks_dir)
    logging.info("Kompletně hotovo! Výsledky najdete v nové složce.")

if __name__ == "__main__":
    main()
