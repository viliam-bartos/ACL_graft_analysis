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

def get_bernard_hertel_grid(femur_mask, fem_vox, tib_vox, spacing_zyx, acl_center_dim0=None):
    sz, sy, sx = spacing_zyx
    f_dim0, f_dim1, f_dim2 = fem_vox
    t_dim0, t_dim1, t_dim2 = tib_vox
    
    # 1. Dynamický posun do středu fossy (podle těžiště ACL)
    if acl_center_dim0 is not None and not np.isnan(acl_center_dim0):
        slice_dim0 = int(np.round(acl_center_dim0))
    else:
        # Fallback, pokud by těžiště ACL z nějakého důvodu selhalo
        direction = np.sign(t_dim0 - f_dim0)
        if direction == 0: direction = 1
        slice_dim0 = int(np.round(f_dim0 + 10 * direction))
        
    slice_dim0 = np.clip(slice_dim0, 0, femur_mask.shape[0] - 1)
    
    # 2. Extrakce sagitálního řezu uprostřed fossy
    sag_slice = femur_mask[slice_dim0, :, :]
    
    y_c = int(f_dim1) 
    z_c = int(f_dim2) 
    
    boundary_pts = []
    
    # 3. Ray casting "doleva" (k menším indexům Z)
    y_min_ray = max(0, y_c - 20)
    y_max_ray = min(sag_slice.shape[0], y_c + 20)
    
    for y in range(y_min_ray, y_max_ray):
        ray = sag_slice[y, :z_c]
        hit_indices = np.argwhere(ray[::-1])
        
        if len(hit_indices) > 0:
            first_hit_reversed = hit_indices[0][0]
            hit_z = z_c - 1 - first_hit_reversed
            boundary_pts.append([hit_z, y])
            
    boundary_pts = np.array(boundary_pts)
    
    if len(boundary_pts) < 2:
        return {} 
        
    # ZDE CHYBĚLA DEFINICE (Extrémy pro vykreslení úsečky)
    d2_min, d2_max = boundary_pts[:, 0].min(), boundary_pts[:, 0].max()
    
    # 4. Lineární regrese
    slope, intercept, _, _, _ = linregress(boundary_pts[:, 0], boundary_pts[:, 1])
    
    phys_dim0 = slice_dim0 * sz
    
    d1_start = slope * d2_min + intercept
    d1_end = slope * d2_max + intercept
    
    p1_blum = (phys_dim0, d1_start * sy, d2_min * sx)
    p2_blum = (phys_dim0, d1_end * sy, d2_max * sx)
    blum_line = (p1_blum, p2_blum)
    
    # 5. Vektory a plný 2D Bounding Box přes laterální kondyl
    v_long = np.array([0.0, (d1_end - d1_start) * sy, (d2_max - d2_min) * sx])
    blum_length = np.linalg.norm(v_long)
    
    if blum_length == 0:
        return {}
        
    v_long = v_long / blum_length
    v_short = np.array([0.0, -v_long[2], v_long[1]])
    
    # Vektor musí ukazovat dolů ke kondylu
    if v_short[1] < 0:
        v_short = -v_short

    lateral_dir = -np.sign(t_dim0 - f_dim0)
    if lateral_dir == 0: lateral_dir = -1
    
    if lateral_dir > 0:
        lateral_slab = femur_mask[slice_dim0:, :, :]
    else:
        lateral_slab = femur_mask[:slice_dim0+1, :, :]
        
    bone_coords_3d = np.argwhere(lateral_slab)
    
    if len(bone_coords_3d) > 0:
        vec_y = bone_coords_3d[:, 1] * sy - p1_blum[1]
        vec_z = bone_coords_3d[:, 2] * sx - p1_blum[2]
        
        proj_long = vec_y * v_long[1] + vec_z * v_long[2]
        proj_short = vec_y * v_short[1] + vec_z * v_short[2]
        
        # TRIK: Odřízneme tělo femuru. Bereme jen voxely, které leží pod úrovní BL 
        # (nebo max 5 fyzických milimetrů nad ní, abychom chytili přední okraj chrupavky).
        condyle_voxels = proj_short > -5.0 
        
        valid_proj_long = proj_long[condyle_voxels]
        valid_proj_short = proj_short[condyle_voxels]
        
        if len(valid_proj_long) > 0:
            min_long = np.min(valid_proj_long) # Přední/zadní hrana
            max_long = np.max(valid_proj_long) # Přední/zadní hrana
            max_short = np.max(valid_proj_short) # Spodní hrana
        else:
            min_long = 0
            max_long = blum_length
            max_short = blum_length
    else:
        min_long = 0
        max_long = blum_length
        max_short = blum_length
        
    if max_short <= 0: max_short = blum_length
    
    # Posun počátku mřížky na nový nalezený okraj kosti (na ose v_long)
    grid_origin = np.array(p1_blum) + min_long * v_long
    grid_length = max_long - min_long
    grid_depth = max_short
    
    grid_lines = []
    ref_edge = None
    
    for i in range(5):
        t = i / 4.0
        
        start_pt = grid_origin + t * grid_depth * v_short
        end_pt = start_pt + grid_length * v_long
        grid_lines.append((tuple(start_pt), tuple(end_pt)))
        
        start_pt2 = grid_origin + t * grid_length * v_long
        end_pt2 = start_pt2 + grid_depth * v_short
        grid_lines.append((tuple(start_pt2), tuple(end_pt2)))
        
        if i == 0: 
            ref_edge = (tuple(grid_origin), tuple(grid_origin + grid_length * v_long))
            
    return {
        'lines': grid_lines, 
        'ref_edge': ref_edge, 
        'blum_line': blum_line,
        'grid_origin': grid_origin,
        'v_long': v_long,
        'v_short': v_short,
        'grid_length': grid_length,
        'grid_depth': grid_depth
    }


def extract_footprints(mask_array, spacing):
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    
    struct = ndimage.generate_binary_structure(3, 1)
    acl_dilated = ndimage.binary_dilation(acl_mask, structure=struct, iterations=2)
    
    femoral_contact = acl_dilated & femur_mask
    tibial_contact = acl_dilated & tibia_mask
    
    # Centroidy úponů ve voxelových souřadnicích
    fem_z, fem_y, fem_x = ndimage.center_of_mass(femoral_contact)
    tib_z, tib_y, tib_x = ndimage.center_of_mass(tibial_contact)
    
    # Těžiště samotného vazu (pro nalezení středu fossy)
    acl_z, acl_y, acl_x = ndimage.center_of_mass(acl_mask)
    
    # Generování mřížky s dynamickým řezem uprostřed vazu
    bh_grid_info = get_bernard_hertel_grid(
        femur_mask, 
        (fem_z, fem_y, fem_x), 
        (tib_z, tib_y, tib_x), 
        spacing,
        acl_center_dim0=acl_z
    )
    
    sz, sy, sx = spacing
    femur_centroid_phys = (fem_z * sz, fem_y * sy, fem_x * sx)
    tibia_centroid_phys = (tib_z * sz, tib_y * sy, tib_x * sx)
    
    # Výpočet B&H procent
    bh_grid_info['bh_length_pct'] = np.nan
    bh_grid_info['bh_depth_pct'] = np.nan
    
    if bh_grid_info and 'grid_origin' in bh_grid_info:
        g_orig = bh_grid_info['grid_origin']
        v_l = bh_grid_info['v_long']
        v_s = bh_grid_info['v_short']
        g_len = bh_grid_info['grid_length']
        g_dep = bh_grid_info['grid_depth']
        
        vec_to_cent = np.array(femur_centroid_phys) - g_orig
        
        proj_l = np.dot(vec_to_cent, v_l)
        proj_s = np.dot(vec_to_cent, v_s)
        
        if g_len > 0:
            raw_length_pct = (proj_l / g_len) * 100.0
            
            # Anatomický kompas: Vektor úponů ukazuje z femuru na tibii (směřuje anteriorně)
            acl_vec = np.array(tibia_centroid_phys) - np.array(femur_centroid_phys)
            
            # Skalární součin zjistí, zda v_long směřuje také dopředu
            if np.dot(v_l, acl_vec) > 0:
                # Mřížka začíná vzadu a jde dopředu. Výpočet z počátku je správný.
                bh_grid_info['bh_length_pct'] = raw_length_pct
            else:
                # Mřížka začíná vpředu a jde dozadu. Procenta odečteme od 100.
                bh_grid_info['bh_length_pct'] = 100.0 - raw_length_pct
                
        if g_dep > 0:
            bh_grid_info['bh_depth_pct'] = (proj_s / g_dep) * 100.0
            
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
    Volume of ACL, Minimal Distance for Impingement, Exact Notch Width at ACL centroid.
    """
    logging.info("Starting Module 4: Spatial Relations & Impingement.")
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    
    # 1. Objem
    acl_volume_mm3 = np.sum(acl_mask) * voxel_vol
    
    # 2. Impingement Assessment (Vzdálenost ACL od femuru)
    inv_femur = ~femur_mask
    dist_map = ndimage.distance_transform_edt(inv_femur, sampling=spacing)
    acl_distances = dist_map[acl_mask]
    min_dist_to_femur = acl_distances.min() if acl_distances.size > 0 else np.nan
    
    # 3. Exaktní šířka interkondylární fossy (Ray Casting)
    acl_centroid = ndimage.center_of_mass(acl_mask)
    notch_width_mm = np.nan
    
    if not np.isnan(acl_centroid[0]):
        dim0_c = int(np.round(acl_centroid[0])) # Pravo-levá osa (R-L)
        dim1_c = int(np.round(acl_centroid[1])) # Předozadní osa (A-P) / Superior-Inferior
        dim2_c = int(np.round(acl_centroid[2]))
        
        # Vyřízneme 1D paprsek napříč kolenem (R-L osa) přesně v úrovni těžiště ACL
        # Fixujeme zbylé dvě osy, abychom stříleli rovně do stran
        try:
            rl_ray = femur_mask[:, dim1_c, dim2_c]
            
            # Hledáme hranice kosti směrem doleva a doprava od vazu
            left_side = rl_ray[:dim0_c]
            right_side = rl_ray[dim0_c:]
            
            left_hits = np.argwhere(left_side)
            right_hits = np.argwhere(right_side)
            
            if len(left_hits) > 0 and len(right_hits) > 0:
                # Nejbližší stěna zleva (poslední bod před vazem)
                left_edge = left_hits[-1][0]
                
                # Nejbližší stěna zprava (první bod za vazem)
                right_edge = right_hits[0][0] + dim0_c
                
                # Výpočet fyzické vzdálenosti v milimetrech (dim0 odpovídá spacing[0])
                notch_width_mm = (right_edge - left_edge) * spacing[0]
                
        except IndexError:
            logging.warning("Těžiště ACL je mimo masku femuru, nelze změřit šířku fossy.")
            
    return {
        "acl_volume_mm3": acl_volume_mm3,
        "min_dist_to_femur_mm": min_dist_to_femur,
        "notch_width_mm": notch_width_mm
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
# Module 6: Advanced Geometric Features (Tortuosity & ATT)
# =============================================================================
def calculate_tortuosity(acl_mask, femur_centroid, tibia_centroid, spacing):
    """
    Calculate the Tortuosity index of the ACL.
    Returns ratio of curved path length to straight-line footprint distance.
    """
    logging.info("Starting Module 6: Advanced Geometric Features.")
    logging.info("Calculating Tortuosity Index.")
    if any(np.isnan(c) for c in femur_centroid) or any(np.isnan(c) for c in tibia_centroid):
        return np.nan
        
    p_f = np.array(femur_centroid)
    p_t = np.array(tibia_centroid)
    straight_length = np.linalg.norm(p_f - p_t)
    
    if straight_length == 0:
        return np.nan
        
    centroids_3d = []
    # Find active slices along the primary axis component (usually dimension 1, Y-axis)
    slice_indices = np.where(np.any(acl_mask, axis=(0, 2)))[0]
    
    for y in slice_indices:
        slice_mask = acl_mask[:, y, :]
        if np.sum(slice_mask) > 0:
            coords = np.argwhere(slice_mask)
            cz, cx = coords.mean(axis=0)
            phys_z = cz * spacing[0]
            phys_y = y * spacing[1]
            phys_x = cx * spacing[2]
            centroids_3d.append([phys_z, phys_y, phys_x])
            
    if len(centroids_3d) < 2:
        return 1.0
        
    centroids_3d = np.array(centroids_3d)
    centroids_3d = centroids_3d[np.argsort(centroids_3d[:, 1])]
    
    diffs = np.diff(centroids_3d, axis=0)
    curved_length = np.sum(np.linalg.norm(diffs, axis=1))
    
    tortuosity = curved_length / straight_length
    return max(1.0, float(tortuosity))

def calculate_att(femur_mask, tibia_mask, spacing, plane_info, f_centroid, t_centroid):
    """
    Calculate Anterior Tibial Translation (ATT) in millimeters.
    """
    logging.info("Calculating Anterior Tibial Translation (ATT).")
    if any(np.isnan(c) for c in f_centroid) or any(np.isnan(c) for c in t_centroid):
        return np.nan, {}
        
    plane_normal = plane_info.get("normal", np.array([0.0, 1.0, 0.0]))
    
    # Estimate Anterior direction globally from footprints (Tibia is anterior to Femur)
    acl_vec = np.array(t_centroid) - np.array(f_centroid)
    
    # Z-axis (dim 2) is historically A-P in these volumes
    ap_global = np.array([0.0, 0.0, 1.0])
    
    # Align ap_global with the actual anterior direction 
    if np.dot(ap_global, acl_vec) < 0:
        ap_global = -ap_global
        
    # Project anterior vector onto the tibial plateau
    dot_prod = np.dot(ap_global, plane_normal)
    v_anterior = ap_global - dot_prod * plane_normal
    
    norm_v_ap = np.linalg.norm(v_anterior)
    if norm_v_ap == 0:
        v_anterior = ap_global 
    else:
        v_anterior = v_anterior / norm_v_ap
        
    sz, sy, sx = spacing
    
    # 1. Tibial posterior edge
    tib_coords = np.argwhere(tibia_mask)
    if len(tib_coords) == 0:
        return np.nan, {}
    tib_phys = tib_coords * np.array([sz, sy, sx])
    tib_proj = np.dot(tib_phys, v_anterior)
    tibia_idx = np.argmin(tib_proj)
    tibia_posterior_edge = tib_proj[tibia_idx] # Minimum of anterior projection is posterior
    tibia_pt = tib_phys[tibia_idx]
    
    # 2. Femoral posterior edge (lateral condyle only)
    f_dim0 = int(np.round(f_centroid[0] / sz)) if sz > 0 else 0
    t_dim0 = int(np.round(t_centroid[0] / sz)) if sz > 0 else 0
    
    lateral_dir = -np.sign(t_dim0 - f_dim0)
    if lateral_dir == 0: lateral_dir = -1
    
    f_dim0 = np.clip(f_dim0, 0, femur_mask.shape[0] - 1)
    
    if lateral_dir > 0:
        lateral_slab = femur_mask[f_dim0:, :, :]
        coords_offset = np.array([f_dim0, 0, 0])
    else:
        lateral_slab = femur_mask[:f_dim0+1, :, :]
        coords_offset = np.array([0, 0, 0])
        
    fem_coords = np.argwhere(lateral_slab)
    if len(fem_coords) == 0:
        return np.nan, {}
        
    fem_phys = (fem_coords + coords_offset) * np.array([sz, sy, sx])
    fem_proj = np.dot(fem_phys, v_anterior)
    femur_idx = np.argmin(fem_proj)
    femur_posterior_edge = fem_proj[femur_idx]
    femur_pt = fem_phys[femur_idx]
    
    # 3. Translation: Distance between tibial 'wall' and femoral 'wall'
    # Positive ATT means tibia is more anterior than femur
    att_mm = float(tibia_posterior_edge - femur_posterior_edge)
    
    debug_info = {
        "tibia_pt": tibia_pt,
        "femur_pt": femur_pt,
        "v_anterior": v_anterior,
        "plane_normal": plane_normal
    }
    
    return att_mm, debug_info

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
    
    plane_info = {
        "normal": orientation_metrics.get("plateau_normal", np.array([0.0, 1.0, 0.0])),
        "center": orientation_metrics.get("plateau_center", np.array([0.0, 0.0, 0.0])),
        "bh_grid_info": bh_grid_info
    }
    
    # Module 6: Advanced Geometric Features
    acl_mask = (mask_array == 1)
    tortuosity_idx = calculate_tortuosity(acl_mask, f_centroid, t_centroid, spacing_zyx)
    
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)
    att_mm, att_debug_info = calculate_att(femur_mask, tibia_mask, spacing_zyx, plane_info, f_centroid, t_centroid)
    
    plane_info['att_info'] = att_debug_info
    
    # Vytáhnutí procent z mřížky
    bh_len_pct = bh_grid_info.get('bh_length_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan
    bh_dep_pct = bh_grid_info.get('bh_depth_pct', np.nan) if isinstance(bh_grid_info, dict) else np.nan

    results_dict = {
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
