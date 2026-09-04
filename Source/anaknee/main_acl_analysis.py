"""
Anaknee - ACL Graft Analysis Pipeline Orchestrator

This module coordinates geometric and radiomic analysis workflows.
All underlying mathematical and anatomical models are modularized into
anaknee.geometry and anaknee.radiomics, but re-exported here for 100%
backward compatibility.
"""

import os
import argparse
import logging
import warnings
import numpy as np
import SimpleITK as sitk

# Suppress warnings that might clutter medical analysis output
warnings.filterwarnings("ignore")

# Import modularized components with fallback for diverse execution contexts
try:
    from .geometry import (
        _reorient_to_ria,
        PlaneModel3D,
        get_tibial_plateau_plane,
        get_bernard_hertel_grid,
        extract_footprints,
        analyze_acl_orientation,
        analyze_spatial_relations,
        calculate_tortuosity,
        calculate_att,
        calculate_staubli_tibial,
    )
    from .radiomics import match_histograms, extract_radiomics
except (ImportError, ValueError):
    try:
        from anaknee.geometry import (
            _reorient_to_ria,
            PlaneModel3D,
            get_tibial_plateau_plane,
            get_bernard_hertel_grid,
            extract_footprints,
            analyze_acl_orientation,
            analyze_spatial_relations,
            calculate_tortuosity,
            calculate_att,
            calculate_staubli_tibial,
        )
        from anaknee.radiomics import match_histograms, extract_radiomics
    except (ImportError, ValueError):
        from geometry import (  # type: ignore
            _reorient_to_ria,
            PlaneModel3D,
            get_tibial_plateau_plane,
            get_bernard_hertel_grid,
            extract_footprints,
            analyze_acl_orientation,
            analyze_spatial_relations,
            calculate_tortuosity,
            calculate_att,
            calculate_staubli_tibial,
        )
        from radiomics import match_histograms, extract_radiomics  # type: ignore

__all__ = [
    "_reorient_to_ria",
    "PlaneModel3D",
    "get_tibial_plateau_plane",
    "get_bernard_hertel_grid",
    "extract_footprints",
    "analyze_acl_orientation",
    "analyze_spatial_relations",
    "calculate_tortuosity",
    "calculate_att",
    "calculate_staubli_tibial",
    "match_histograms",
    "extract_radiomics",
    "run_analysis",
    "run_geometric_analysis_from_mask",
    "main",
]


def run_analysis(img_path, ref_path, mask_path, compute_radiomics=True):
    """
    Executes the analytical pipeline and returns structures needed for reporting and visualization.
    
    Both the image and mask are reoriented to RIA canonical orientation at the start,
    ensuring all downstream axis assumptions (dim0=R-L, dim1=S-I, dim2=A-P) are correct
    regardless of the input file's original orientation.
    """
    logging.info(f"Loading MRI sequence: {img_path}")
    logging.info(f"Loading Reference MRI: {ref_path}")
    logging.info(f"Loading Segmentation Mask: {mask_path}")
    
    img_sitk_raw = sitk.ReadImage(img_path)
    mask_sitk_raw = sitk.ReadImage(mask_path)
    
    # Canonical reorientation to RIA (dim0=R-L, dim1=S-I, dim2=A-P)
    img_sitk = _reorient_to_ria(img_sitk_raw)
    mask_sitk = _reorient_to_ria(mask_sitk_raw)
        
    spacing = img_sitk.GetSpacing()
    sz, sy, sx = spacing[2], spacing[1], spacing[0]
    spacing_zyx = (sz, sy, sx)
    
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    
    # Module 1 (only if radiomics requested)
    if compute_radiomics and ref_path and os.path.exists(ref_path):
        std_img_sitk = match_histograms(img_sitk, ref_path, mask_sitk)
    else:
        std_img_sitk = None
    
    # Module 2
    f_centroid, t_centroid, bh_grid_info = extract_footprints(mask_array, spacing_zyx)
    
    # Module 3
    orientation_metrics = analyze_acl_orientation(f_centroid, t_centroid, mask_array, spacing_zyx)
    
    # Module 4
    spatial_metrics = analyze_spatial_relations(mask_array, spacing_zyx)
    
    # Module 5
    if compute_radiomics and std_img_sitk is not None:
        radiomics_features = extract_radiomics(std_img_sitk, mask_sitk)
    else:
        radiomics_features = {}
    
    plane_info = {
        "normal": orientation_metrics.get("plateau_normal", np.array([0.0, 1.0, 0.0])),
        "center": orientation_metrics.get("plateau_center", np.array([0.0, 0.0, 0.0])),
        "bh_grid_info": bh_grid_info,
        "plateau_inliers": orientation_metrics.get("plateau_inliers"),
        "plateau_outliers": orientation_metrics.get("plateau_outliers"),
    }
    
    # Module 6: Advanced Geometric Features
    acl_mask = (mask_array == 1)
    tortuosity_idx = calculate_tortuosity(acl_mask, f_centroid, t_centroid, spacing_zyx)
    
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    att_mm, att_debug_info = calculate_att(femur_mask, tibia_mask, spacing_zyx, plane_info, f_centroid, t_centroid)
    
    staubli_pct, staubli_debug_info = calculate_staubli_tibial(tibia_mask, t_centroid, f_centroid, spacing_zyx, plane_info)
    
    plane_info['att_info'] = att_debug_info
    plane_info['staubli_info'] = staubli_debug_info
    
    # Extract grid percentages
    bh_len_pct = bh_grid_info.get('bh_length_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan
    bh_dep_pct = bh_grid_info.get('bh_depth_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan

    results_dict = {
        "Staubli_Tibial_pct": staubli_pct,
        "Tortuosity_Index": tortuosity_idx,
        "ATT_mm": att_mm,
        "BH_Length_pct": bh_len_pct,
        "BH_Depth_pct": bh_dep_pct,
        "angle_to_plateau_deg": orientation_metrics.get("angle_to_plateau_deg", np.nan),
        "sagittal_angle_deg": orientation_metrics.get("sagittal_angle_deg", np.nan),
        "coronal_angle_deg": orientation_metrics.get("coronal_angle_deg", np.nan),
        **spatial_metrics,
        **radiomics_features
    }
    
    return results_dict, mask_array, spacing_zyx, f_centroid, t_centroid, plane_info


def run_geometric_analysis_from_mask(mask_input, spacing=None, progress_callback=None):
    """
    Rapid, standalone geometric analysis directly from a segmentation mask.
    Executes in ~0.5-2 seconds (skips MRI histogram matching and radiomics).
    
    Args:
        mask_input (str, sitk.Image, or np.ndarray): Mask path, SimpleITK image, or 3D numpy array.
        spacing (tuple, optional): Voxel spacing (sz, sy, sx) if mask_input is a numpy array.
        progress_callback (callable, optional): Callback taking a status string.
        
    Returns:
        tuple: (results_dict, mask_array, spacing_zyx, f_centroid, t_centroid, plane_info, vis_data)
    """
    def _notify(step_msg):
        if progress_callback:
            try:
                progress_callback(step_msg)
            except Exception:
                pass

    _notify("Načítání 3D masky z disku...")
    if isinstance(mask_input, str):
        mask_sitk_raw = sitk.ReadImage(mask_input)
        _notify("Reorientace masky do RIA prostoru...")
        mask_sitk = _reorient_to_ria(mask_sitk_raw)
        sp = mask_sitk.GetSpacing()
        spacing_zyx = (sp[2], sp[1], sp[0])
        mask_array = sitk.GetArrayFromImage(mask_sitk)
    elif isinstance(mask_input, sitk.Image):
        _notify("Reorientace masky do RIA prostoru...")
        mask_sitk = _reorient_to_ria(mask_input)
        sp = mask_sitk.GetSpacing()
        spacing_zyx = (sp[2], sp[1], sp[0])
        mask_array = sitk.GetArrayFromImage(mask_sitk)
    elif isinstance(mask_input, np.ndarray):
        mask_array = mask_input
        spacing_zyx = spacing if spacing is not None else (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unsupported mask input type: {type(mask_input)}")

    # Module 2: Footprints
    _notify("Detekce úponů vazu a Bernard-Hertel mřížky...")
    f_centroid, t_centroid, bh_grid_info = extract_footprints(mask_array, spacing_zyx)
    
    # Module 3: Orientation & RANSAC plateau
    _notify("Rychlé RANSAC fitování tibiálního plata a úhlů...")
    orientation_metrics = analyze_acl_orientation(f_centroid, t_centroid, mask_array, spacing_zyx)
    
    # Module 4: Spatial relations
    _notify("Analýza impingementu a interkondylární fossy...")
    spatial_metrics = analyze_spatial_relations(mask_array, spacing_zyx)
    
    plane_info = {
        "normal": orientation_metrics.get("plateau_normal", np.array([0.0, 1.0, 0.0])),
        "center": orientation_metrics.get("plateau_center", np.array([0.0, 0.0, 0.0])),
        "bh_grid_info": bh_grid_info,
        "plateau_inliers": orientation_metrics.get("plateau_inliers"),
        "plateau_outliers": orientation_metrics.get("plateau_outliers"),
    }
    
    # Module 6: Advanced Geometric Features
    _notify("Výpočet biomechaniky (ATT translace, Stäubli, tortuozita)...")
    acl_mask = (mask_array == 1)
    tortuosity_idx = calculate_tortuosity(acl_mask, f_centroid, t_centroid, spacing_zyx)
    
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    att_mm, att_debug_info = calculate_att(femur_mask, tibia_mask, spacing_zyx, plane_info, f_centroid, t_centroid)
    staubli_pct, staubli_debug_info = calculate_staubli_tibial(tibia_mask, t_centroid, f_centroid, spacing_zyx, plane_info)
    
    plane_info['att_info'] = att_debug_info
    plane_info['staubli_info'] = staubli_debug_info
    
    bh_len_pct = bh_grid_info.get('bh_length_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan
    bh_dep_pct = bh_grid_info.get('bh_depth_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan

    results_dict = {
        "Staubli_Tibial_pct": staubli_pct,
        "Tortuosity_Index": tortuosity_idx,
        "ATT_mm": att_mm,
        "BH_Length_pct": bh_len_pct,
        "BH_Depth_pct": bh_dep_pct,
        "angle_to_plateau_deg": orientation_metrics.get("angle_to_plateau_deg", np.nan),
        "sagittal_angle_deg": orientation_metrics.get("sagittal_angle_deg", np.nan),
        "coronal_angle_deg": orientation_metrics.get("coronal_angle_deg", np.nan),
        **spatial_metrics,
    }

    vis_data = {
        "femoral_centroid": f_centroid,
        "tibial_centroid": t_centroid,
        "plateau_normal": plane_info["normal"],
        "plateau_center": plane_info["center"],
        "bh_grid_info": bh_grid_info,
        "att_info": att_debug_info,
        "staubli_info": staubli_debug_info,
        "plateau_inliers": plane_info.get("plateau_inliers"),
        "plateau_outliers": plane_info.get("plateau_outliers"),
        "results_dict": results_dict,
    }
    
    return results_dict, mask_array, spacing_zyx, f_centroid, t_centroid, plane_info, vis_data


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Comprehensive 3D Isotropic MRI ACL Analysis")
    parser.add_argument("--img", type=str, required=True, help="Path to original input MRI (.nii.gz)")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference MRI for intensity norm (.nii.gz)")
    parser.add_argument("--mask", type=str, required=True, help="Path to segmentation mask (.nii.gz)")
    args = parser.parse_args()
    
    logging.info(f"Loading MRI sequence: {args.img}")
    logging.info(f"Loading Reference MRI: {args.ref}")
    logging.info(f"Loading Segmentation Mask: {args.mask}")
    
    try:
        img_sitk_raw = sitk.ReadImage(args.img)
        mask_sitk_raw = sitk.ReadImage(args.mask)
    except Exception as e:
        logging.error(f"Failed to load NIfTI images: {e}")
        return
    
    # Canonical reorientation to RIA
    img_sitk = _reorient_to_ria(img_sitk_raw)
    mask_sitk = _reorient_to_ria(mask_sitk_raw)
        
    spacing = img_sitk.GetSpacing()
    sz, sy, sx = spacing[2], spacing[1], spacing[0]
    spacing_zyx = (sz, sy, sx)
    logging.info(f"Image scaling (Z, Y, X): {spacing_zyx}")
    
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    
    # --- Execute Modules ---
    
    # Module 1
    std_img_sitk = match_histograms(img_sitk, args.ref, mask_sitk)
    logging.info("Histogram matching completed.")
    
    # Module 2
    f_centroid, t_centroid, _ = extract_footprints(mask_array, spacing_zyx)
    logging.info(f"Femoral Centroid (Phys): {f_centroid}")
    logging.info(f"Tibial Centroid (Phys): {t_centroid}")
    
    # Module 3
    orientation_metrics = analyze_acl_orientation(f_centroid, t_centroid, mask_array, spacing_zyx)
    logging.info(f"ACL Orientation Mechanics: {orientation_metrics}")
    
    # Module 4
    spatial_metrics = analyze_spatial_relations(mask_array, spacing_zyx)
    logging.info(f"Spatial Relations & Impingement Assessment: {spatial_metrics}")
    
    # Module 5
    radiomics_features = extract_radiomics(std_img_sitk, mask_sitk)
    logging.info(f"Extracted {len(radiomics_features)} radiomics features.")
    
    # Print brief summary of GLCM
    glcm_features = {k: v for k, v in radiomics_features.items() if 'glcm' in k.lower()}
    if glcm_features:
        logging.info("Sample GLCM features:")
        for k, v in list(glcm_features.items())[:3]:
            logging.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
