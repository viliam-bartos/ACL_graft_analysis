import os
import pandas as pd
from main_acl_analysis import run_analysis
from visualizator_analyzator import visualize_results

def run_pipeline(mri_path, mask_path):
    # Dle požadavku, explicitní referenční snímek pro histogram matching:
    ref_path = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images\case_074.nii.gz"
    
    # 1. Spustí matematiku a radiomiku (vrací slovník geometrických/radiomických parametrů, masku, velikost voxelů a parametry rovin)
    results_dict, mask_array, spacing, f_centroid, t_centroid, plane_info = run_analysis(mri_path, ref_path, mask_path)
    
    # 2. Uloží výsledky do tabulky (header se zapíše pouze u první položky)
    df = pd.DataFrame([results_dict])
    header = not os.path.exists("acl_results.csv")
    df.to_csv("acl_results.csv", mode='a', header=header, index=False)
    
    vis_data = {
        'femoral_centroid': f_centroid,
        'tibial_centroid': t_centroid,
        'plateau_normal': plane_info['normal'],
        'plateau_center': plane_info['center'],
        'bh_grid_info': plane_info.get('bh_grid_info', {})
    }
    
    # 3. Předá data do PyVisty (zde se script pozastaví a vykreslí přesný matematický model, dokud okno nezavřeš)
    visualize_results(mask_array, spacing, vis_data)

if __name__ == "__main__":
    # Ukázka volání (změňte dle potřeby)
    run_pipeline(r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images\case_003.nii.gz", r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\LABELS_TRAIN\mask_case_003.nii.gz")