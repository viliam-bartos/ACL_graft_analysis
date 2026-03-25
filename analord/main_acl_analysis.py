import os
import argparse
import logging
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from scipy import ndimage
from scipy.stats import linregress
from skimage.measure import ransac, LineModelND
import radiomics
from radiomics import featureextractor
import warnings

# Suppress warnings that might clutter the medical image analysis output
warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

# =============================================================================
# Module 1: Histogram Matching (Intensity Standardization)
# =============================================================================
def match_histograms(img_sitk, ref_path, mask_sitk=None):
    """
    Match the intensity distribution of the input MRI to the reference MRI
    using Nyul-Udupa histogram standardization via TorchIO.
    
    Parameters:
        img_sitk (SimpleITK.Image): Input 3D MRI.
        ref_path (str): Path to Reference 3D MRI.
        mask_sitk (SimpleITK.Image, optional): Mask to define non-zero background regions.
    
    Returns:
        SimpleITK.Image: The standardized MRI.
    """
    logging.info("Starting Module 1: Histogram Matching (Nyul-Udupa via TorchIO).")
    
    img_array = sitk.GetArrayFromImage(img_sitk)
    
    # Convert arrays to TorchIO ScalarImages, adding a channel dimension (C, D, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
    
    img_tio = tio.ScalarImage(tensor=img_tensor)
    
    # Train landmarks on the reference image path directly
    # The masking function ensures we apply the learned distribution to the foreground.
    landmarks = tio.HistogramStandardization.train(
        [str(ref_path)],
        masking_function=lambda x: x > 0
    )
    
    # Create the transform with the learned landmarks
    transform = tio.HistogramStandardization({'mri': landmarks})
    
    # Apply to the input subject
    subject = tio.Subject(mri=img_tio)
    
    # Apply mask if provided (to maintain structural strictness if necessary)
    if mask_sitk is not None:
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        subject.add_image(tio.LabelMap(tensor=mask_tensor), 'mask')
        # We could use the mask to exclude background during application,
        # but HistogramStandardization applies to the whole image.
        
    standardized_subject = transform(subject)
    
    # Extract standardized tensor and convert back to SimpleITK
    standardized_tensor = standardized_subject['mri'].data.squeeze(0).numpy()
    standardized_sitk = sitk.GetImageFromArray(standardized_tensor)
    standardized_sitk.CopyInformation(img_sitk)
    
    return standardized_sitk

def get_tibial_plateau_plane(tibia_mask, spacing):
    sR, sS, sA = spacing
    coords = np.argwhere(tibia_mask)
    if len(coords) == 0:
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])
        
    phys_coords = coords * np.array([sR, sS, sA])
    
    # Sort by the S-axis (Superior-Inferior, column 1). S=0 is Superior.
    sorted_indices = np.argsort(phys_coords[:, 1])
    sorted_phys_coords = phys_coords[sorted_indices]
    
    # Take the top 15% of voxels to calculate the orientation
    n_top = int(len(sorted_phys_coords) * 0.15)
    if n_top < 3:
        n_top = min(3, len(sorted_phys_coords))
        
    top_points = sorted_phys_coords[:n_top]
    
    if len(top_points) >= 3:
        centroid = top_points.mean(axis=0)
        centered = top_points - centroid
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :] 
    else:
        centroid = phys_coords.mean(axis=0)
        normal = np.array([0.0, -1.0, 0.0])
        
    if normal[1] > 0:
        normal = -normal
        
    return normal, centroid

def get_bernard_hertel_grid(femur_mask, footprint_phys, spacing_zyx):
    """
    Korekce os: Numpy maska má tvar (R-L, S-I, A-P).
    Osa 0 = X = R-L
    Osa 1 = Y = S-I
    Osa 2 = Z = A-P
    PyVista mapuje tyto rozměry přímo na (X, Y, Z). Takže Sagitální rovina odpovídá fixnímu X.
    """
    sz, sy, sx = spacing_zyx 
    # V numpy po getArrayFromImage: sz = R-L, sy = S-I, sx = A-P
    
    # 1. PŘEVOD NA VOXELY
    # footprint_phys je (R-L, S-I, A-P)
    x_vox = int(np.round(footprint_phys[0] / sz)) # Osa X (R-L)
    x_vox = np.clip(x_vox, 0, femur_mask.shape[0] - 1)
    
    # 2. Extrakce sagitálního řezu
    # Maska je (R-L, S-I, A-P), takže řežeme na indexu 0 pro konstantní R-L
    sagittal_slice = femur_mask[x_vox, :, :]
    coords = np.argwhere(sagittal_slice)
    
    if len(coords) == 0:
        logging.error("Na ose X v tomto místě není žádná kost!")
        return []

    # Z coords máme sloupce: coords[:, 0] je S-I (Y), coords[:, 1] je A-P (Z)
    
    # 3. Nalezení střechy (Blumensaat) - hledáme v zadní polovině (Z)
    z_min, z_max = coords[:, 1].min(), coords[:, 1].max()
    posterior_threshold = z_min + (z_max - z_min) * 0.5
    posterior_coords = coords[coords[:, 1] >= posterior_threshold]
    
    boundary_pts = []
    # Pro každý A-P (Z) bod najdeme nejspodnější S-I (Y) bod
    for z_val in np.unique(posterior_coords[:, 1]):
        y_bottom = np.max(posterior_coords[posterior_coords[:, 1] == z_val, 0])
        boundary_pts.append([z_val, y_bottom])
        
    boundary_pts = np.array(boundary_pts)
    
    # 4. Výpočet DLOUHÉ strany (Rovnoběžka se střechou)
    blum_line = None
    if len(boundary_pts) > 3:
        # linregress(Z, Y) -> jak se mění výška(Y) při posunu zepředu dozadu(Z)
        slope, intercept, _, _, _ = linregress(boundary_pts[:, 0], boundary_pts[:, 1])
        
        # --- ZMĚNA 1 PRO OTOČENÍ ÚHLU ---
        # Původně: long_vec = np.array([1.0, slope])
        # Změnou znaménka u slope otočíme úhel mřížky naopak.
        long_vec = np.array([1.0, -slope]) 
        
        # Výpočet Blumensaatovy linie pro vizualizaci přesně podle regrese
        x_phys = footprint_phys[0]
        z_start = coords[:, 1].min()
        z_end = coords[:, 1].max()
        y_start = slope * z_start + intercept
        y_end = slope * z_end + intercept
        
        p1_blum = (x_phys, y_start * sy, z_start * sx)
        p2_blum = (x_phys, y_end * sy, z_end * sx)
        blum_line = (p1_blum, p2_blum)
    else:
        long_vec = np.array([1.0, 0.0])

    # Zabezpečení orientace - dlouhý vektor musí ukazovat zepředu dozadu (kladný směr Z)
    if long_vec[0] < 0:
        long_vec = -long_vec
        
    # Výpočet KRÁTKÉ strany (Kolmice na střechu - výška kondylu)
    # Kolmice k [dZ, dY] je [-dY, dZ]
    short_vec = np.array([-long_vec[1], long_vec[0]])
    
    # Krátký vektor musí ukazovat shora dolů (kladný směr Y)
    if short_vec[1] < 0:
        # --- ZMĚNA 2 (OPRAVA CHYBY Z MINULA) ---
        # Tady chybělo mínus, teď už vektor opravdu otočíme.
        short_vec = -short_vec
        
     # 5. Promítnutí kondylu pro získání Bounding Boxu

    # coords = [Y_val, Z_val]

    z_coords = coords[:, 1]

    y_coords = coords[:, 0]

   

    # Dot product ručně pro jistotu, že se nám to neprohodí

    long_projs = z_coords * long_vec[0] + y_coords * long_vec[1]

    short_projs = z_coords * short_vec[0] + y_coords * short_vec[1]

   

    # Limity dlouhé strany (Délka kondylu)

    l_min = np.min(long_projs)

    l_max = np.max(long_projs)

   

    # Limity krátké strany (Výška kondylu)

    s_min = np.min(short_projs) # Strop zářezu

    s_max = np.max(short_projs) # Spodek kondylu

    # 6. Tvorba mřížky pro PyVistu
    grid_lines = []
    ref_edge = None
    
    # Fixní souřadnice R-L (X v PyVistě)
    x_phys = footprint_phys[0]
    
    # --- Čáry rovnoběžné s DLOUHOU stranou (řežou výšku kondylu) ---
    for i in range(5):
        s_val = s_min + (s_max - s_min) * (i / 4.0)
        p1_2d = l_min * long_vec + s_val * short_vec
        p2_2d = l_max * long_vec + s_val * short_vec
        
        # p1_2d = [Z_index_val, Y_index_val]
        # Převod do PyVista XYZ = (R-L_fixed, S-I_var, A-P_var) -> (x_phys, Y_index * sy, Z_index * sx)
        p1_phys = (x_phys, p1_2d[1] * sy, p1_2d[0] * sx)
        p2_phys = (x_phys, p2_2d[1] * sy, p2_2d[0] * sx)
        
        grid_lines.append((p1_phys, p2_phys))
        
    # --- Čáry rovnoběžné s KRÁTKOU stranou (řežou délku kondylu) ---
    for j in range(5):
        l_val = l_min + (l_max - l_min) * (j / 4.0)
        p1_2d = l_val * long_vec + s_min * short_vec
        p2_2d = l_val * long_vec + s_max * short_vec
        
        p1_phys = (x_phys, p1_2d[1] * sy, p1_2d[0] * sx)
        p2_phys = (x_phys, p2_2d[1] * sy, p2_2d[0] * sx)
        
        if j == 0:
            ref_edge = (p1_phys, p2_phys)
        else:
            grid_lines.append((p1_phys, p2_phys))
        
    return {'lines': grid_lines, 'ref_edge': ref_edge, 'blum_line': blum_line}

# =============================================================================
# Module 2: Automated Footprint Extraction & Standardization
# =============================================================================
def extract_footprints(mask_array, spacing):
    """
    Find contact areas between the ACL mask (1) and Femur (2), and ACL (1) and Tibia (3).
    Calculates 3D centroids of these regions.
    Proposes a heuristic for Bernard & Hertel 4x4 grid projection.
    
    Parameters:
        mask_array (np.ndarray): 3D numpy array of the segmentation mask.
        spacing (tuple): Voxel spacing (z, y, x).
        
    Returns:
        tuple: femur_centroid_phys, tibia_centroid_phys
    """
    logging.info("Starting Module 2: Footprint Extraction.")
    
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    
    # Binary dilation of the ACL to capture the boundary contact points
    struct = ndimage.generate_binary_structure(3, 1) # 1-connectivity cross
    acl_dilated = ndimage.binary_dilation(acl_mask, structure=struct, iterations=2)
    
    # Locate exact footprint contact areas
    femoral_contact = acl_dilated & femur_mask
    tibial_contact = acl_dilated & tibia_mask
    
    if np.sum(femoral_contact) == 0 or np.sum(tibial_contact) == 0:
        logging.warning("Footprint contact area not found. Check mask labels or dilation radius.")
    
    # Calculate 3D centroids (in voxel coordinates)
    # Using scipy's center_of_mass: returns (z, y, x)
    fem_z, fem_y, fem_x = ndimage.center_of_mass(femoral_contact)
    tib_z, tib_y, tib_x = ndimage.center_of_mass(tibial_contact)
    
    # Calculate ACL vector (from Tibia to Femur) in voxel coordinates
    acl_vector_voxels = np.array([fem_z - tib_z, fem_y - tib_y, fem_x - tib_x])
    
    # We must calculate femur_centroid_phys before B&H grid
    sz, sy, sx = spacing
    femur_centroid_phys = (fem_z * sz, fem_y * sy, fem_x * sx)
    
    # Generate Bernard & Hertel 4x4 Grid
    bh_grid_info = get_bernard_hertel_grid(femur_mask, femur_centroid_phys, spacing)
    if bh_grid_info and bh_grid_info.get('lines'):
        logging.info(f"Generated {len(bh_grid_info['lines'])} B&H grid lines and 1 reference edge.")
    
    # Convert centroids to physical space depending on spacing (sR, sS, sA)
    sz, sy, sx = spacing
    femur_centroid_phys = (fem_z * sz, fem_y * sy, fem_x * sx)
    tibia_centroid_phys = (tib_z * sz, tib_y * sy, tib_x * sx)
    
    return femur_centroid_phys, tibia_centroid_phys, bh_grid_info

# =============================================================================
# Module 3: ACL Vector and Orientation Analysis
# =============================================================================
def analyze_acl_orientation(femur_centroid, tibia_centroid, mask_array, spacing):
    """
    Define the ACL vector. Approximate the tibial plateau (superior boundary of Tibia),
    fit a 3D plane, and calculate the sagittal and coronal angles.
    
    Parameters:
        femur_centroid (tuple): Physical coordinates of femur footprint.
        tibia_centroid (tuple): Physical coordinates of tibia footprint.
        mask_array (np.ndarray): 3D numpy array of the segmentation mask.
        spacing (tuple): Voxel spacing.
        
    Returns:
        dict: Angles in degrees (angle_to_plateau, sagittal_angle, coronal_angle).
    """
    logging.info("Starting Module 3: ACL Vector & Orientation Analysis.")
    
    if any(np.isnan(c) for c in femur_centroid) or any(np.isnan(c) for c in tibia_centroid):
        return {"angle_to_plateau": np.nan, "sagittal_angle": np.nan, "coronal_angle": np.nan}
        
    p_f = np.array(femur_centroid)
    p_t = np.array(tibia_centroid)
    
    # ACL vector (from Tibia to Femur)
    acl_vector = p_f - p_t 
    acl_vector_norm = acl_vector / np.linalg.norm(acl_vector)
    
    tibia_mask = (mask_array == 3)
    plane_normal, centroid = get_tibial_plateau_plane(tibia_mask, spacing)
    
    # Calculate elevation angle relative to the plane
    angle_to_normal_rad = np.arccos(np.clip(np.abs(np.dot(acl_vector_norm, plane_normal)), -1.0, 1.0))
    angle_to_plateau_deg = 90.0 - np.degrees(angle_to_normal_rad)
    
    # Sagittal plane (Projection onto A-S plane: Axis 2=A, Axis 1=S, constant Axis 0=R=0)
    sagittal_vec = np.array([0.0, acl_vector_norm[1], acl_vector_norm[2]])
    if np.linalg.norm(sagittal_vec) > 0:
        sagittal_vec = sagittal_vec / np.linalg.norm(sagittal_vec)
        # Angle relative to Vertical S-axis (Axis 1 = [0, -1, 0] since smaller is UP)
        sag_angle_rad = np.arccos(np.clip(np.abs(np.dot(sagittal_vec, np.array([0.0, -1.0, 0.0]))), -1.0, 1.0))
        sagittal_angle = np.degrees(sag_angle_rad)
    else:
        sagittal_angle = np.nan
        
    # Coronal plane (Projection onto R-S plane: Axis 0=R, Axis 1=S, constant Axis 2=A=0)
    coronal_vec = np.array([acl_vector_norm[0], acl_vector_norm[1], 0.0])
    if np.linalg.norm(coronal_vec) > 0:
        coronal_vec = coronal_vec / np.linalg.norm(coronal_vec)
        # Angle relative to Vertical S-axis
        cor_angle_rad = np.arccos(np.clip(np.abs(np.dot(coronal_vec, np.array([0.0, -1.0, 0.0]))), -1.0, 1.0))
        coronal_angle = np.degrees(cor_angle_rad)
    else:
        coronal_angle = np.nan
        
    return {
        "angle_to_plateau_deg": angle_to_plateau_deg,
        "sagittal_angle_deg": sagittal_angle,
        "coronal_angle_deg": coronal_angle,
        "plateau_normal": plane_normal,
        "plateau_center": centroid
    }

# =============================================================================
# Module 4: Spatial Relations, Impingement & Notch Width Index
# =============================================================================
def analyze_spatial_relations(mask_array, spacing):
    """
    Volume of ACL, Minimal Distance for Impingement, Notch Width heuristic.
    """
    logging.info("Starting Module 4: Spatial Relations & Impingement.")
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    
    # 1. Volume
    acl_volume_mm3 = np.sum(acl_mask) * voxel_vol
    
    # 2. Impingement Assessment (Minimal distance ACL to Femur notch roof)
    # Calculate distance transform from the femur surface
    inv_femur = ~femur_mask
    dist_map = ndimage.distance_transform_edt(inv_femur, sampling=spacing)
    
    # Since they are contiguous, the minimum distance is zero at the contact footprint.
    # A reliable impingement distance should evaluate the upper third of the ACL (excluding footprint).
    # For a general minimum distance of non-footprint ACL to femur:
    acl_distances = dist_map[acl_mask]
    min_dist_to_femur = acl_distances.min() if acl_distances.size > 0 else np.nan
    
    # 3. Intercondylar Notch Width Heuristic
    # Using the Z-level of the ACL centroid as the trajectory level
    acl_centroid = ndimage.center_of_mass(acl_mask)
    if not np.isnan(acl_centroid[0]):
        z_level = int(np.round(acl_centroid[0]))
        femur_slice = femur_mask[z_level, :, :]
        
        coords = np.argwhere(femur_slice)
        if coords.size > 0:
            # Extreme heuristic: measure bounding box width and assume notch is internal void
            x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
            total_width = (x_max - x_min) * spacing[2]
            notch_width_heuristic = total_width * 0.3 # Assumes notch ~30% width
        else:
            notch_width_heuristic = np.nan
    else:
        notch_width_heuristic = np.nan
        
    logging.info("[HEURISTIC NOTE] Reliable notch width calculation requires morphological tracing of condylar boundaries.")
        
    return {
        "acl_volume_mm3": acl_volume_mm3,
        "min_dist_to_femur_mm": min_dist_to_femur,
        "notch_width_heuristic_mm": notch_width_heuristic
    }

# =============================================================================
# Module 5: Radiomics Extraction
# =============================================================================
def extract_radiomics(standardized_img_sitk, mask_sitk):
    """
    Extract first-order, GLCM, and GLRLM radiomics features from the ACL (label 1).
    """
    logging.info("Starting Module 5: Radiomics Extraction.")
    # Ensure mask contains only label=1 for feature extractor
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    acl_only_array = (mask_array == 1).astype(np.uint8)
    acl_only_sitk = sitk.GetImageFromArray(acl_only_array)
    acl_only_sitk.CopyInformation(mask_sitk)
    
    # Check if ACL mask is empty
    if np.sum(acl_only_array) == 0:
        logging.warning("ACL mask is empty. Skipping radiomics extraction.")
        return {}

    # PyRadiomics settings tailored for 3D isotropic 0.5mm MRI
    settings = {
        'binWidth': 25, 
        'resampledPixelSpacing': None, # Assuming data is already isotropic 0.5mm
        'interpolator': sitk.sitkBSpline,
        'geometryTolerance': 1e-4
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    
    try:
        features = extractor.execute(standardized_img_sitk, acl_only_sitk)
        extracted = {k: float(v) for k, v in features.items() if not k.startswith('diagnostics')}
        return extracted
    except Exception as e:
        logging.error(f"PyRadiomics extraction failed: {e}")
        return {}

# =============================================================================
# Main Entry Point
# =============================================================================
def run_analysis(img_path, ref_path, mask_path):
    """
    Executes the analytical pipeline and returns structures needed for reporting and visualization.
    """
    logging.info(f"Loading MRI sequence: {img_path}")
    logging.info(f"Loading Reference MRI: {ref_path}")
    logging.info(f"Loading Segmentation Mask: {mask_path}")
    
    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)
        
    spacing = img_sitk.GetSpacing()
    sz, sy, sx = spacing[2], spacing[1], spacing[0]
    spacing_zyx = (sz, sy, sx)
    
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    
    # Module 1
    std_img_sitk = match_histograms(img_sitk, ref_path, mask_sitk)
    
    # Module 2
    f_centroid, t_centroid, bh_grid_info = extract_footprints(mask_array, spacing_zyx)
    
    # Module 3
    orientation_metrics = analyze_acl_orientation(f_centroid, t_centroid, mask_array, spacing_zyx)
    
    # Module 4
    spatial_metrics = analyze_spatial_relations(mask_array, spacing_zyx)
    
    # Module 5
    radiomics_features = extract_radiomics(std_img_sitk, mask_sitk)
    
    results_dict = {
        "angle_to_plateau_deg": orientation_metrics.get("angle_to_plateau_deg", np.nan),
        "sagittal_angle_deg": orientation_metrics.get("sagittal_angle_deg", np.nan),
        "coronal_angle_deg": orientation_metrics.get("coronal_angle_deg", np.nan),
        **spatial_metrics,
        **radiomics_features
    }
    
    plane_info = {
        "normal": orientation_metrics.get("plateau_normal", np.array([0.0, 1.0, 0.0])),
        "center": orientation_metrics.get("plateau_center", np.array([0.0, 0.0, 0.0])),
        "bh_grid_info": bh_grid_info
    }
    
    return results_dict, mask_array, spacing_zyx, f_centroid, t_centroid, plane_info

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
        img_sitk = sitk.ReadImage(args.img)
        mask_sitk = sitk.ReadImage(args.mask)
    except Exception as e:
        logging.error(f"Failed to load NIfTI images: {e}")
        return
        
    spacing = img_sitk.GetSpacing()
    # Note: SimpleITK spacing is (x, y, z), but numpy shape is (z, y, x).
    # We pass it as (sz, sy, sx) to match np array indexing.
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
        for k, v in list(glcm_features.items())[:3]: # First 3
            logging.info(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
