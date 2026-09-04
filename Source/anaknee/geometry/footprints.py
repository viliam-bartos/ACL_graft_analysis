import numpy as np
from scipy import ndimage
from scipy.stats import linregress


def get_bernard_hertel_grid(femur_mask, fem_vox, tib_vox, spacing_zyx, acl_center_dim0=None):
    """
    Construct the Bernard-Hertel grid on the lateral femoral condyle.

    Args:
        femur_mask (np.ndarray): 3D boolean mask of the femur.
        fem_vox (tuple): (z, y, x) voxel coordinates of the femoral footprint.
        tib_vox (tuple): (z, y, x) voxel coordinates of the tibial footprint.
        spacing_zyx (tuple): Voxel spacing (sz, sy, sx) in mm.
        acl_center_dim0 (float, optional): ACL centroid along dim0 to center the fossa slice.

    Returns:
        dict: Grid geometry dictionary including lines, reference edge, and vectors.
    """
    sz, sy, sx = spacing_zyx
    f_dim0, f_dim1, f_dim2 = fem_vox
    t_dim0, t_dim1, t_dim2 = tib_vox

    # 1. Shift to fossa center (by ACL centroid)
    if acl_center_dim0 is not None and not np.isnan(acl_center_dim0):
        slice_dim0 = int(np.round(acl_center_dim0))
    else:
        direction = np.sign(t_dim0 - f_dim0)
        if direction == 0:
            direction = 1
        slice_dim0 = int(np.round(f_dim0 + 10 * direction))

    slice_dim0 = np.clip(slice_dim0, 0, femur_mask.shape[0] - 1)

    # 2. Extract sagittal slice at fossa center
    sag_slice = femur_mask[slice_dim0, :, :]

    y_c = int(f_dim1)
    z_c = int(f_dim2)

    boundary_pts = []

    # 3. Ray casting left (to smaller Z indices)
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

    # Extrema for line plotting
    d2_min, d2_max = boundary_pts[:, 0].min(), boundary_pts[:, 0].max()

    # 4. Linear regression
    slope, intercept, _, _, _ = linregress(boundary_pts[:, 0], boundary_pts[:, 1])

    phys_dim0 = slice_dim0 * sz

    d1_start = slope * d2_min + intercept
    d1_end = slope * d2_max + intercept

    p1_blum = (phys_dim0, d1_start * sy, d2_min * sx)
    p2_blum = (phys_dim0, d1_end * sy, d2_max * sx)
    blum_line = (p1_blum, p2_blum)

    # 5. Vectors and 2D bounding box over lateral condyle
    v_long = np.array([0.0, (d1_end - d1_start) * sy, (d2_max - d2_min) * sx])
    blum_length = np.linalg.norm(v_long)

    if blum_length == 0:
        return {}

    v_long = v_long / blum_length
    v_short = np.array([0.0, -v_long[2], v_long[1]])

    # Vector must point down to condyle
    if v_short[1] < 0:
        v_short = -v_short

    lateral_dir = -np.sign(t_dim0 - f_dim0)
    if lateral_dir == 0:
        lateral_dir = -1

    if lateral_dir > 0:
        lateral_slab = femur_mask[slice_dim0:, :, :]
    else:
        lateral_slab = femur_mask[:slice_dim0 + 1, :, :]

    bone_coords_3d = np.argwhere(lateral_slab)

    if len(bone_coords_3d) > 0:
        vec_y = bone_coords_3d[:, 1] * sy - p1_blum[1]
        vec_z = bone_coords_3d[:, 2] * sx - p1_blum[2]

        proj_long = vec_y * v_long[1] + vec_z * v_long[2]
        proj_short = vec_y * v_short[1] + vec_z * v_short[2]

        # Keep voxels under BL level (or max 5mm above to capture anterior cartilage margin)
        condyle_voxels = proj_short > -5.0

        valid_proj_long = proj_long[condyle_voxels]
        valid_proj_short = proj_short[condyle_voxels]

        if len(valid_proj_long) > 0:
            min_long = np.min(valid_proj_long)  # Anterior/posterior edge
            max_long = np.max(valid_proj_long)  # Anterior/posterior edge
            max_short = np.max(valid_proj_short)  # Inferior edge
        else:
            min_long = 0
            max_long = blum_length
            max_short = blum_length
    else:
        min_long = 0
        max_long = blum_length
        max_short = blum_length

    if max_short <= 0:
        max_short = blum_length

    # Shift grid origin to detected bone edge (along v_long axis)
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
    """
    Extract ACL femoral and tibial footprint centroids and Bernard-Hertel grid.

    Args:
        mask_array (np.ndarray): 3D segmentation mask (1=ACL, 2=Femur, 3=Tibia).
        spacing (tuple): Voxel spacing (sz, sy, sx) in mm.

    Returns:
        tuple: (femur_centroid_phys, tibia_centroid_phys, bh_grid_info)
    """
    acl_mask = (mask_array == 1)
    femur_mask = (mask_array == 2)
    tibia_mask = (mask_array == 3)

    struct = ndimage.generate_binary_structure(3, 1)
    acl_dilated = ndimage.binary_dilation(acl_mask, structure=struct, iterations=2)

    femoral_contact = acl_dilated & femur_mask
    tibial_contact = acl_dilated & tibia_mask

    # Footprint centroids in voxel coordinates
    fem_z, fem_y, fem_x = ndimage.center_of_mass(femoral_contact)
    tib_z, tib_y, tib_x = ndimage.center_of_mass(tibial_contact)

    # ACL centroid (to find fossa center)
    acl_z, acl_y, acl_x = ndimage.center_of_mass(acl_mask)

    # Grid generation at ACL center slice
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

    # Calculate B&H percentages
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

            # Footprint vector points from femur to tibia (anteriorly)
            acl_vec = np.array(tibia_centroid_phys) - np.array(femur_centroid_phys)

            # Check if v_long also points forward
            if np.dot(v_l, acl_vec) > 0:
                bh_grid_info['bh_length_pct'] = raw_length_pct
            else:
                bh_grid_info['bh_length_pct'] = 100.0 - raw_length_pct

        if g_dep > 0:
            bh_grid_info['bh_depth_pct'] = (proj_s / g_dep) * 100.0

    return femur_centroid_phys, tibia_centroid_phys, bh_grid_info
