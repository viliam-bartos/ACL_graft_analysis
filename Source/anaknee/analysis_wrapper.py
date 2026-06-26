import os
import pandas as pd
from main_acl_analysis import run_analysis
from visualizator_analyzator import visualize_results

def run_pipeline(mri_path, mask_path):
    
    ref_path = r""
    
    # Run analysis
    results_dict, mask_array, spacing, f_centroid, t_centroid, plane_info = run_analysis(mri_path, ref_path, mask_path)
    
    # Save results to CSV (header on first write)
    df = pd.DataFrame([results_dict])
    header = not os.path.exists("acl_results.csv")
    df.to_csv("acl_results.csv", mode='a', header=header, index=False)
    
    vis_data = {
        'femoral_centroid': f_centroid,
        'tibial_centroid': t_centroid,
        'plateau_normal': plane_info['normal'],
        'plateau_center': plane_info['center'],
        'bh_grid_info': plane_info.get('bh_grid_info', {}),
        'att_info': plane_info.get('att_info', {}),
        'staubli_info': plane_info.get('staubli_info', {})
    }
    
    
    visualize_results(mask_array, spacing, vis_data)

if __name__ == "__main__":
    
    run_pipeline(r"", r"")