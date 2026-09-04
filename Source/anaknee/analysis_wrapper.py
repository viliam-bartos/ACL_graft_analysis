import os
import sys
from pathlib import Path
import pandas as pd

# Add directories to path
CURR = Path(__file__).resolve().parent
ROOT = CURR.parent.parent
if str(CURR) not in sys.path:
    sys.path.insert(0, str(CURR))
if str(CURR.parent) not in sys.path:
    sys.path.insert(0, str(CURR.parent))

from main_acl_analysis import run_analysis, run_geometric_analysis_from_mask
from visualizator_analyzator import visualize_results, smart_visualize

def run_pipeline(mri_path=None, mask_path=None, ref_path=None, fast_mode=True):
    """
    Run ACL geometric analysis and open interactive 3D viewer.
    If paths are not given, falls back to reference datasets in Data/.
    """
    # Fallback to reference data if empty
    default_mri = os.path.join(ROOT, "Data", "reference", "right_case_074.nii.gz")
    default_mask = os.path.join(ROOT, "Data", "reference", "vysledky_074", "mask_right_case_074.nii.gz")
    
    if not mri_path or not os.path.exists(mri_path):
        if os.path.exists(default_mri):
            mri_path = default_mri
            print(f"[INFO] Using default reference MRI: {mri_path}")
        else:
            mri_path = None

    if not mask_path or not os.path.exists(mask_path):
        if os.path.exists(default_mask):
            mask_path = default_mask
            print(f"[INFO] Using default reference mask: {mask_path}")
        else:
            print("[ERROR] Please provide a valid mask_path or ensure reference data is in Data/.")
            return

    # Fast mode: compute geometry directly from mask in ~1.5s
    if fast_mode or not mri_path:
        print("[INFO] Running fast geometric analysis directly from mask...")
        results_dict, mask_array, spacing, f_centroid, t_centroid, plane_info, vis_data = run_geometric_analysis_from_mask(mask_path)
    else:
        print("[INFO] Running comprehensive analysis with MRI context...")
        ref_path = ref_path if ref_path and os.path.exists(ref_path) else mri_path
        results_dict, mask_array, spacing, f_centroid, t_centroid, plane_info = run_analysis(
            mri_path, ref_path, mask_path, compute_radiomics=False
        )
        vis_data = {
            'femoral_centroid': f_centroid,
            'tibial_centroid': t_centroid,
            'plateau_normal': plane_info['normal'],
            'plateau_center': plane_info['center'],
            'bh_grid_info': plane_info.get('bh_grid_info', {}),
            'att_info': plane_info.get('att_info', {}),
            'staubli_info': plane_info.get('staubli_info', {}),
            'plateau_inliers': plane_info.get('plateau_inliers'),
            'plateau_outliers': plane_info.get('plateau_outliers'),
            'results_dict': results_dict,
        }

    # Save results to CSV (header on first write)
    df = pd.DataFrame([results_dict])
    header = not os.path.exists("acl_results.csv")
    df.to_csv("acl_results.csv", mode='a', header=header, index=False)
    print(f"[INFO] Results saved to acl_results.csv (ATT: {results_dict.get('ATT_mm', 0):.2f} mm)")

    # Launch PyVista
    visualize_results(mask_array, spacing, vis_data)

if __name__ == "__main__":
    # If arguments given via CLI, use them, otherwise use reference data
    mri = sys.argv[1] if len(sys.argv) > 1 else None
    mask = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(mri, mask)