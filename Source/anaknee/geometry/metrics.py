import logging
import numpy as np
from scipy import ndimage
from .plateau import get_tibial_plateau_plane


def analyze_acl_orientation(femur_centroid, tibia_centroid, mask_array, spacing):
    """
    Define the ACL vector. Approximate the tibial plateau (superior boundary of Tibia),
    fit a 3D plane using RANSAC, and calculate the sagittal and coronal angles.

    Parameters:
        femur_centroid (tuple): Physical coordinates of femur footprint (z, y, x).
        tibia_centroid (tuple): Physical coordinates of tibia footprint (z, y, x).
        mask_array (np.ndarray): 3D numpy array of the segmentation mask.
        spacing (tuple): Voxel spacing (sz, sy, sx).

    Returns:
        dict: Angles in degrees, plateau normal, center, inliers, and outliers.
    """
    logging.info("Starting Module 3: ACL Vector & Orientation Analysis.")

    if any(np.isnan(c) for c in femur_centroid) or any(np.isnan(c) for c in tibia_centroid):
        return {
            "angle_to_plateau_deg": np.nan,
            "sagittal_angle_deg": np.nan,
            "coronal_angle_deg": np.nan,
            "plateau_normal": np.array([0.0, 1.0, 0.0]),
            "plateau_center": np.array([0.0, 0.0, 0.0]),
            "plateau_inliers": None,
            "plateau_outliers": None,
        }

    p_f = np.array(femur_centroid)
    p_t = np.array(tibia_centroid)

    # ACL vector (from Tibia to Femur)
    acl_vector = p_f - p_t
    acl_vector_norm = acl_vector / np.linalg.norm(acl_vector)

    tibia_mask = (mask_array == 3)
    # Use femur centroid (p_f) as proximal reference so 'top' is defined anatomically
    plane_normal, centroid, plateau_inliers, plateau_outliers = get_tibial_plateau_plane(
        tibia_mask, spacing, proximal_point=p_f
    )

    # Calculate elevation angle relative to the plane
    angle_to_normal_rad = np.arccos(np.clip(np.abs(np.dot(acl_vector_norm, plane_normal)), -1.0, 1.0))
    angle_to_plateau_deg = 90.0 - np.degrees(angle_to_normal_rad)

    # Sagittal plane (Projection onto A-S plane: Axis 2=A, Axis 1=S, constant Axis 0=R=0)
    sagittal_vec = np.array([0.0, acl_vector_norm[1], acl_vector_norm[2]])
    sagittal_normal = np.array([0.0, plane_normal[1], plane_normal[2]])
    if np.linalg.norm(sagittal_vec) > 0 and np.linalg.norm(sagittal_normal) > 0:
        sagittal_vec = sagittal_vec / np.linalg.norm(sagittal_vec)
        sagittal_normal = sagittal_normal / np.linalg.norm(sagittal_normal)
        sag_angle_rad = np.arccos(np.clip(np.abs(np.dot(sagittal_vec, sagittal_normal)), -1.0, 1.0))
        sagittal_angle = 90.0 - np.degrees(sag_angle_rad)
    else:
        sagittal_angle = np.nan

    # Coronal plane (Projection onto R-S plane: Axis 0=R, Axis 1=S, constant Axis 2=A=0)
    coronal_vec = np.array([acl_vector_norm[0], acl_vector_norm[1], 0.0])
    coronal_normal = np.array([plane_normal[0], plane_normal[1], 0.0])
    if np.linalg.norm(coronal_vec) > 0 and np.linalg.norm(coronal_normal) > 0:
        coronal_vec = coronal_vec / np.linalg.norm(coronal_vec)
        coronal_normal = coronal_normal / np.linalg.norm(coronal_normal)
        cor_angle_rad = np.arccos(np.clip(np.abs(np.dot(coronal_vec, coronal_normal)), -1.0, 1.0))
        coronal_angle = 90.0 - np.degrees(cor_angle_rad)
    else:
        coronal_angle = np.nan

    return {
        "angle_to_plateau_deg": angle_to_plateau_deg,
        "sagittal_angle_deg": sagittal_angle,
        "coronal_angle_deg": coronal_angle,
        "plateau_normal": plane_normal,
        "plateau_center": centroid,
        "plateau_inliers": plateau_inliers,
        "plateau_outliers": plateau_outliers,
    }


def analyze_spatial_relations(mask_array, spacing):
    """
    Volume of ACL, Minimal Distance for Impingement, Exact Notch Width at ACL centroid.
    Uses bounded sub-volume EDT for speed.
    """
    logging.info("Starting Module 4: Spatial Relations & Impingement.")
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)

    # 1. Volume
    acl_volume_mm3 = np.sum(acl_mask) * voxel_vol

    # 2. Impingement assessment (ACL-femur distance via bounded EDT)
    acl_coords = np.argwhere(acl_mask)
    if len(acl_coords) > 0:
        margin = 25
        z_min = max(0, acl_coords[:, 0].min() - margin)
        z_max = min(mask_array.shape[0], acl_coords[:, 0].max() + margin)
        y_min = max(0, acl_coords[:, 1].min() - margin)
        y_max = min(mask_array.shape[1], acl_coords[:, 1].max() + margin)
        x_min = max(0, acl_coords[:, 2].min() - margin)
        x_max = min(mask_array.shape[2], acl_coords[:, 2].max() + margin)

        sub_femur = femur_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        sub_acl = acl_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        dist_map = ndimage.distance_transform_edt(~sub_femur, sampling=spacing)
        acl_distances = dist_map[sub_acl]
        min_dist_to_femur = acl_distances.min() if acl_distances.size > 0 else np.nan
    else:
        min_dist_to_femur = np.nan

    # 3. Intercondylar notch width (ray casting)
    acl_centroid = ndimage.center_of_mass(acl_mask)
    notch_width_mm = np.nan

    if not np.isnan(acl_centroid[0]):
        dim0_c = int(np.round(acl_centroid[0]))  # R-L axis
        dim1_c = int(np.round(acl_centroid[1]))
        dim2_c = int(np.round(acl_centroid[2]))

        try:
            rl_ray = femur_mask[:, dim1_c, dim2_c]

            left_side = rl_ray[:dim0_c]
            right_side = rl_ray[dim0_c:]

            left_hits = np.argwhere(left_side)
            right_hits = np.argwhere(right_side)

            if len(left_hits) > 0 and len(right_hits) > 0:
                left_edge = left_hits[-1][0]
                right_edge = right_hits[0][0] + dim0_c
                notch_width_mm = (right_edge - left_edge) * spacing[0]

        except IndexError:
            logging.warning("ACL centroid is outside femur mask range; cannot measure notch width.")

    return {
        "acl_volume_mm3": acl_volume_mm3,
        "min_dist_to_femur_mm": min_dist_to_femur,
        "notch_width_mm": notch_width_mm,
    }


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

    # Z-axis (dim 2) is A-P in RIA volumes
    ap_global = np.array([0.0, 0.0, 1.0])

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
    tibia_posterior_edge = tib_proj[tibia_idx]
    tibia_pt = tib_phys[tibia_idx]

    # 2. Femoral posterior edge (lateral condyle only)
    f_dim0 = int(np.round(f_centroid[0] / sz)) if sz > 0 else 0
    t_dim0 = int(np.round(t_centroid[0] / sz)) if sz > 0 else 0

    lateral_dir = -np.sign(t_dim0 - f_dim0)
    if lateral_dir == 0:
        lateral_dir = -1

    f_dim0 = np.clip(f_dim0, 0, femur_mask.shape[0] - 1)

    if lateral_dir > 0:
        lateral_slab = femur_mask[f_dim0:, :, :]
        coords_offset = np.array([f_dim0, 0, 0])
    else:
        lateral_slab = femur_mask[:f_dim0 + 1, :, :]
        coords_offset = np.array([0, 0, 0])

    fem_coords = np.argwhere(lateral_slab)
    if len(fem_coords) == 0:
        return np.nan, {}

    fem_phys = (fem_coords + coords_offset) * np.array([sz, sy, sx])
    fem_proj = np.dot(fem_phys, v_anterior)
    femur_idx = np.argmin(fem_proj)
    femur_posterior_edge = fem_proj[femur_idx]
    femur_pt = fem_phys[femur_idx]

    # 3. Translation: Distance between tibial wall and femoral wall
    att_mm = float(tibia_posterior_edge - femur_posterior_edge)

    debug_info = {
        "tibia_pt": tibia_pt,
        "femur_pt": femur_pt,
        "v_anterior": v_anterior,
        "plane_normal": plane_normal,
    }

    return att_mm, debug_info


def calculate_staubli_tibial(tibia_mask, t_centroid, f_centroid, spacing, plane_info):
    """
    Calculate Stäubli percentage for the tibial footprint.
    Percentage represents the distance from the anterior tibial margin to the ACL centroid
    divided by the total AP dimension on the sagittal slice.
    """
    logging.info("Calculating Stäubli percentage for the tibial footprint.")
    if any(np.isnan(c) for c in t_centroid) or any(np.isnan(c) for c in f_centroid):
        return np.nan, {}

    sz, sy, sx = spacing

    tib_z_idx = int(np.round(t_centroid[0] / sz)) if sz > 0 else 0
    tib_z_idx = np.clip(tib_z_idx, 0, tibia_mask.shape[0] - 1)

    sag_slice = tibia_mask[tib_z_idx, :, :]
    coords_2d = np.argwhere(sag_slice)

    if len(coords_2d) == 0:
        return np.nan, {}

    phys_dim0 = tib_z_idx * sz
    y_phys = coords_2d[:, 0] * sy
    x_phys = coords_2d[:, 1] * sx

    points_3d = np.column_stack((np.full_like(y_phys, phys_dim0), y_phys, x_phys))

    plane_normal = plane_info.get("normal", np.array([0.0, -1.0, 0.0]))
    slice_normal = np.array([1.0, 0.0, 0.0])

    # Vector lying in sagittal slice and parallel to plateau
    v_horizontal = np.cross(plane_normal, slice_normal)

    if np.linalg.norm(v_horizontal) > 0:
        v_horizontal = v_horizontal / np.linalg.norm(v_horizontal)
    else:
        v_horizontal = np.array([0.0, 0.0, 1.0])

    # Orient to anterior direction
    acl_vec = np.array(t_centroid) - np.array(f_centroid)
    ap_global = np.array([0.0, 0.0, 1.0])
    if np.dot(ap_global, acl_vec) < 0:
        ap_global = -ap_global

    if np.dot(v_horizontal, ap_global) < 0:
        v_horizontal = -v_horizontal

    # Limit evaluated points to top 20mm to avoid posterior cortex slope
    d_up = np.dot(points_3d, plane_normal)
    max_d = np.max(d_up)

    plateau_slab_mask = d_up >= (max_d - 20.0)
    plateau_points = points_3d[plateau_slab_mask]

    projections = np.dot(plateau_points, v_horizontal)

    ant_edge = np.max(projections)
    post_edge = np.min(projections)

    t_cent_slice = np.array([phys_dim0, t_centroid[1], t_centroid[2]])
    cent_proj = np.dot(t_cent_slice, v_horizontal)

    total_ap_length = ant_edge - post_edge

    if total_ap_length <= 0:
        return np.nan, {}

    staubli_pct = ((ant_edge - cent_proj) / total_ap_length) * 100.0
    staubli_pct = np.clip(staubli_pct, 0.0, 100.0)

    # Calculate line endpoints on the plateau plane intersection
    plane_center = plane_info.get("center", np.array([0.0, 0.0, 0.0]))
    v_x_plane = np.array([1.0, 0.0, 0.0]) - plane_normal[0] * plane_normal

    if abs(v_x_plane[0]) > 1e-6:
        t_intersect = (phys_dim0 - plane_center[0]) / v_x_plane[0]
        base_pt_on_intersection = plane_center + t_intersect * v_x_plane
    else:
        base_pt_on_intersection = np.array([phys_dim0, plane_center[1], plane_center[2]])

    base_proj = np.dot(base_pt_on_intersection, v_horizontal)
    line_ortho_plateau = base_pt_on_intersection - (base_proj * v_horizontal)

    horizontal_ant_pt = line_ortho_plateau + (ant_edge * v_horizontal)
    horizontal_post_pt = line_ortho_plateau + (post_edge * v_horizontal)

    debug_info = {
        "anterior_pt": horizontal_ant_pt,
        "posterior_pt": horizontal_post_pt,
        "v_anterior_sag": v_horizontal,
    }

    return float(staubli_pct), debug_info
