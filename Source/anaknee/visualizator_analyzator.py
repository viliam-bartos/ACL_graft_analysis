import numpy as np
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from scipy.ndimage import binary_dilation


# ═══════════════════════════════════════════════════════════════════════
# Color Palette – Medical Imaging Dark Theme
# ═══════════════════════════════════════════════════════════════════════
_PALETTE = {
    # Background
    "bg_top":       "#0d1117",
    "bg_bottom":    "#161b22",
    # Anatomy
    "femur":        "#e8dcc8",     # warm ivory
    "tibia":        "#d4c5a9",     # warm beige
    "acl":          "#ff8c42",     # vivid orange
    "footprint":    "#ff4d6a",     # coral red
    # Centroids
    "fem_centroid": "#5cacee",     # steel blue
    "tib_centroid": "#5cee8c",     # emerald green
    "acl_vector":   "#c084fc",     # soft purple
    # Plateau
    "plateau":      "#22d3ee",     # cyan
    "inlier":       "#34d399",     # emerald
    "outlier":      "#f87171",     # rose red
    # B&H Grid
    "bh_grid":      "#94a3b8",     # slate
    "bh_ref":       "#fbbf24",     # amber
    "blumensaat":   "#4ade80",     # green
    # ATT
    "att_tibia":    "#ef4444",     # red
    "att_femur":    "#3b82f6",     # blue
    "att_measure":  "#facc15",     # yellow
    # Stäubli
    "staubli":      "#06b6d4",     # teal
    "staubli_pt":   "#67e8f9",     # light teal
    # Text
    "text":         "#e2e8f0",     # light slate
    "text_dim":     "#94a3b8",     # dim slate
    "text_label":   "#cbd5e1",     # label slate
}


# ═══════════════════════════════════════════════════════════════════════
# Mesh Generation
# ═══════════════════════════════════════════════════════════════════════
def create_surface_mesh(binary_mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), smooth=True):
    """Converts a 3D numpy boolean/int array into a PyVista surface mesh."""
    padded_mask = np.pad(binary_mask, 1, mode='constant', constant_values=0)
    
    grid = pv.ImageData()
    grid.dimensions = padded_mask.shape
    grid.spacing = spacing
    grid.origin = tuple(o - s for o, s in zip(origin, spacing))
    grid.point_data['values'] = padded_mask.flatten(order='F')
    
    mesh = grid.contour([0.5])
    
    if smooth and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.1)
        
    return mesh


# ═══════════════════════════════════════════════════════════════════════
# Metric Formatting
# ═══════════════════════════════════════════════════════════════════════
def _format_metric(label, value, unit="", precision=1):
    """Format a metric value as a display string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{label}: N/A"
    return f"{label}: {value:.{precision}f}{unit}"


def _build_metrics_text(vis_data):
    """Build a multi-line metrics string for the overlay panel."""
    rd = vis_data.get("results_dict", {})
    
    lines = []
    lines.append("─── Geometric Metrics ───")
    lines.append(_format_metric("B&H Length", rd.get("BH_Length_pct"), " %"))
    lines.append(_format_metric("B&H Depth", rd.get("BH_Depth_pct"), " %"))
    lines.append(_format_metric("Stäubli", rd.get("Staubli_Tibial_pct"), " %"))
    lines.append(_format_metric("ATT", rd.get("ATT_mm"), " mm"))
    lines.append(_format_metric("Tortuosity", rd.get("Tortuosity_Index"), "", 2))
    lines.append("")
    lines.append("─── Orientation ───")
    lines.append(_format_metric("Elevation", rd.get("angle_to_plateau_deg"), "°"))
    lines.append(_format_metric("Sagittal", rd.get("sagittal_angle_deg"), "°"))
    lines.append(_format_metric("Coronal", rd.get("coronal_angle_deg"), "°"))
    lines.append("")
    lines.append("─── Spatial ───")
    lines.append(_format_metric("ACL Vol.", rd.get("acl_volume_mm3"), " mm³", 0))
    lines.append(_format_metric("Notch W.", rd.get("notch_width_mm"), " mm"))
    lines.append(_format_metric("Impinge.", rd.get("min_dist_to_femur_mm"), " mm"))
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main Visualization
# ═══════════════════════════════════════════════════════════════════════
def visualize_results(mask_data, spacing, vis_data):
    """
    Premium 3D visualization of ACL geometric analysis results.
    
    Features:
    - Dark gradient background with specular bone rendering
    - Grouped checkbox visibility toggles
    - RANSAC inlier/outlier plateau point clouds
    - Metric annotation panel (upper right)
    
    Args:
        mask_data (np.ndarray): 3D segmentation mask (0=BG, 1=ACL, 2=Femur, 3=Tibia)
        spacing (tuple): Voxel spacing (sz, sy, sx) in mm.
        vis_data (dict): Geometric analysis results from run_analysis.
    """
    
    # ── Data Extraction ─────────────────────────────────────────────
    p = _PALETTE
    
    fem_centroid = np.asarray(vis_data.get('femoral_centroid', [50, 60, 70]), dtype=float)
    tib_centroid = np.asarray(vis_data.get('tibial_centroid', [45, 55, 30]), dtype=float)
    
    plateau_normal = np.asarray(vis_data.get('plateau_normal', [0.05, 0.05, 1.0]), dtype=float)
    if np.linalg.norm(plateau_normal) > 0:
        plateau_normal = plateau_normal / np.linalg.norm(plateau_normal)
    plateau_center = np.asarray(vis_data.get('plateau_center', [45, 55, 32]), dtype=float)
    
    plateau_inliers = vis_data.get('plateau_inliers')
    plateau_outliers = vis_data.get('plateau_outliers')
    
    bh_grid_info = vis_data.get('bh_grid_info', {})
    att_info = vis_data.get('att_info', {})
    staubli_info = vis_data.get('staubli_info', {})
    
    origin = (0.0, 0.0, 0.0)

    # ── Anatomical Meshes ───────────────────────────────────────────
    mask_acl = mask_data == 1
    mask_femur = mask_data == 2
    mask_tibia = mask_data == 3
    
    print("Generating 3D anatomical meshes...")
    dilated_acl = binary_dilation(mask_acl, iterations=2)
    footprint_femur_mask = dilated_acl & mask_femur
    footprint_tibia_mask = dilated_acl & mask_tibia
    
    mesh_acl = create_surface_mesh(mask_acl, spacing=spacing, origin=origin)
    mesh_femur = create_surface_mesh(mask_femur, spacing=spacing, origin=origin)
    mesh_tibia = create_surface_mesh(mask_tibia, spacing=spacing, origin=origin)
    mesh_fp_femur = create_surface_mesh(footprint_femur_mask, spacing=spacing, origin=origin, smooth=False)
    mesh_fp_tibia = create_surface_mesh(footprint_tibia_mask, spacing=spacing, origin=origin, smooth=False)

    # ── Geometric Primitives ────────────────────────────────────────
    sphere_fem = pv.Sphere(radius=2.0, center=fem_centroid)
    sphere_tib = pv.Sphere(radius=2.0, center=tib_centroid)
    acl_line = pv.Line(fem_centroid, tib_centroid)
    
    # Tibial Plateau Plane
    temp_i = np.array([1.0, 0.0, 0.0])
    temp_i_proj = temp_i - np.dot(temp_i, plateau_normal) * plateau_normal
    if np.linalg.norm(temp_i_proj) > 0:
        i_dir = temp_i_proj / np.linalg.norm(temp_i_proj)
    else:
        i_dir = np.array([0.0, 0.0, 1.0])
    j_dir = np.cross(plateau_normal, i_dir)
    j_dir = j_dir / (np.linalg.norm(j_dir) + 1e-12)

    base_plane = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=60, j_size=60,
                          i_resolution=1, j_resolution=1)
    tm = np.eye(4)
    tm[0:3, 0] = i_dir
    tm[0:3, 1] = j_dir
    tm[0:3, 2] = plateau_normal
    tm[0:3, 3] = plateau_center
    plateau_plane = base_plane.transform(tm, inplace=False)
    
    # RANSAC point clouds
    if plateau_inliers is not None and len(plateau_inliers) > 0:
        pc_inliers = pv.PolyData(plateau_inliers)
    else:
        pc_inliers = pv.PolyData()
    
    if plateau_outliers is not None and len(plateau_outliers) > 0:
        pc_outliers = pv.PolyData(plateau_outliers)
    else:
        pc_outliers = pv.PolyData()

    # B&H Grid
    bh_lines = []
    bh_grid_data = bh_grid_info.get('lines', []) if isinstance(bh_grid_info, dict) else []
    ref_edge = bh_grid_info.get('ref_edge') if isinstance(bh_grid_info, dict) else None
    blum_line = bh_grid_info.get('blum_line') if isinstance(bh_grid_info, dict) else None
    
    for start_pt, end_pt in bh_grid_data:
        bh_lines.append(pv.Line(start_pt, end_pt))
    
    bh_grid_mb = pv.MultiBlock(bh_lines) if bh_lines else pv.PolyData()

    # ATT geometries
    has_att = att_info and 'tibia_pt' in att_info
    if has_att:
        t_pt = att_info['tibia_pt']
        f_pt = att_info['femur_pt']
        v_ant = att_info['v_anterior']
        n_p = att_info['plane_normal']
        
        att_t_line = pv.Line(t_pt - 40*n_p, t_pt + 40*n_p)
        att_f_line = pv.Line(f_pt - 40*n_p, f_pt + 40*n_p)
        att_t_sphere = pv.Sphere(radius=2.0, center=t_pt)
        att_f_sphere = pv.Sphere(radius=2.0, center=f_pt)
        
        dist = np.dot(t_pt - f_pt, v_ant)
        att_measure_line = pv.Line(f_pt, f_pt + dist * v_ant)
    
    # Stäubli geometries
    has_staubli = staubli_info and 'anterior_pt' in staubli_info
    if has_staubli:
        ant_pt = staubli_info['anterior_pt']
        post_pt = staubli_info['posterior_pt']
        v_ant_sag = staubli_info['v_anterior_sag']
        
        staubli_line = pv.Line(ant_pt, post_pt)
        staubli_ant_sphere = pv.Sphere(radius=2.0, center=ant_pt)
        staubli_post_sphere = pv.Sphere(radius=2.0, center=post_pt)
        
        cent_proj = np.dot(tib_centroid - ant_pt, v_ant_sag)
        cent_on_line = ant_pt + cent_proj * v_ant_sag
        staubli_cent_sphere = pv.Sphere(radius=2.0, center=cent_on_line)
    
    # Blumensaat geometries
    has_blum = blum_line is not None
    if has_blum:
        blum_pv = pv.Line(blum_line[0], blum_line[1])
        blum_s1 = pv.Sphere(radius=2.0, center=blum_line[0])
        blum_s2 = pv.Sphere(radius=2.0, center=blum_line[1])

    # ── Plotter Setup ───────────────────────────────────────────────
    plotter = pv.Plotter(title="ACL 3D Geometric Analysis", window_size=(1600, 1000))
    plotter.set_background(p["bg_bottom"], top=p["bg_top"])
    
    # ── Add Meshes ──────────────────────────────────────────────────
    # Group 1: Anatomy
    a_femur = plotter.add_mesh(mesh_femur, color=p["femur"], opacity=0.25,
                               specular=0.5, specular_power=30, smooth_shading=True)
    a_tibia = plotter.add_mesh(mesh_tibia, color=p["tibia"], opacity=0.25,
                               specular=0.5, specular_power=30, smooth_shading=True)
    a_acl = plotter.add_mesh(mesh_acl, color=p["acl"], opacity=0.85,
                             specular=0.3, specular_power=20, smooth_shading=True)
    a_fp_fem = plotter.add_mesh(mesh_fp_femur, color=p["footprint"], opacity=1.0,
                                specular=0.4, smooth_shading=True)
    a_fp_tib = plotter.add_mesh(mesh_fp_tibia, color=p["footprint"], opacity=1.0,
                                specular=0.4, smooth_shading=True)
    
    # Group 2: Centroids & ACL Vector
    a_cent_fem = plotter.add_mesh(sphere_fem, color=p["fem_centroid"],
                                  specular=1.0, specular_power=50)
    a_cent_tib = plotter.add_mesh(sphere_tib, color=p["tib_centroid"],
                                  specular=1.0, specular_power=50)
    a_acl_vec = plotter.add_mesh(acl_line, color=p["acl_vector"], line_width=4)
    
    # Group 3: Tibial Plateau
    a_plateau = plotter.add_mesh(plateau_plane, color=p["plateau"], opacity=0.35)
    a_inliers = plotter.add_mesh(pc_inliers, color=p["inlier"],
                                 point_size=5, render_points_as_spheres=True)
    a_outliers = plotter.add_mesh(pc_outliers, color=p["outlier"],
                                  point_size=5, render_points_as_spheres=True)
    
    # Group 4: B&H Grid
    a_bh = plotter.add_mesh(bh_grid_mb, color=p["bh_grid"], line_width=2)
    
    a_bh_ref = a_bh_ref1 = a_bh_ref2 = None
    if ref_edge:
        a_bh_ref = plotter.add_mesh(pv.Line(ref_edge[0], ref_edge[1]),
                                     color=p["bh_ref"], line_width=4)
        a_bh_ref1 = plotter.add_mesh(pv.Sphere(radius=1.5, center=ref_edge[0]),
                                      color=p["bh_ref"])
        a_bh_ref2 = plotter.add_mesh(pv.Sphere(radius=1.5, center=ref_edge[1]),
                                      color=p["bh_ref"])
    
    a_blum = a_blum1 = a_blum2 = None
    if has_blum:
        a_blum = plotter.add_mesh(blum_pv, color=p["blumensaat"], line_width=5)
        a_blum1 = plotter.add_mesh(blum_s1, color=p["blumensaat"])
        a_blum2 = plotter.add_mesh(blum_s2, color=p["blumensaat"])
    
    # Group 5: ATT
    a_att_tl = a_att_fl = a_att_m = a_att_ts = a_att_fs = None
    if has_att:
        a_att_tl = plotter.add_mesh(att_t_line, color=p["att_tibia"], line_width=3)
        a_att_fl = plotter.add_mesh(att_f_line, color=p["att_femur"], line_width=3)
        a_att_m  = plotter.add_mesh(att_measure_line, color=p["att_measure"], line_width=5)
        a_att_ts = plotter.add_mesh(att_t_sphere, color=p["att_tibia"])
        a_att_fs = plotter.add_mesh(att_f_sphere, color=p["att_femur"])
    
    # Group 6: Stäubli
    a_st_l = a_st_a = a_st_p = a_st_c = None
    if has_staubli:
        a_st_l = plotter.add_mesh(staubli_line, color=p["staubli"], line_width=5)
        a_st_a = plotter.add_mesh(staubli_ant_sphere, color=p["staubli_pt"])
        a_st_p = plotter.add_mesh(staubli_post_sphere, color=p["staubli_pt"])
        a_st_c = plotter.add_mesh(staubli_cent_sphere, color=p["footprint"])

    # ── Axes ────────────────────────────────────────────────────────
    plotter.add_axes(
        color=p["text_dim"],
        line_width=2,
        labels_off=False,
        xlabel="R-L (dim0)",
        ylabel="S-I (dim1)",
        zlabel="P-A (dim2)",
    )

    # ── Metrics Panel ───────────────────────────────────────────────
    metrics_text = _build_metrics_text(vis_data)
    plotter.add_text(
        metrics_text,
        position="upper_right",
        font_size=9,
        color=p["text"],
        font="courier",
        shadow=True,
    )

    # ── Title ───────────────────────────────────────────────────────
    plotter.add_text(
        "ACL 3D Geometric Analysis",
        position="upper_left",
        font_size=14,
        color=p["text"],
        font="arial",
        shadow=True,
    )

    # ── Grouped Checkbox Toggles ────────────────────────────────────
    def _toggle(flag, actors):
        for a in actors:
            if a is not None:
                a.SetVisibility(flag)

    x_cb = 10
    y_pos = 30
    cb_size = 25
    step = 35
    
    # Helper: add one checkbox row
    def _add_row(label, actors, color, y):
        plotter.add_checkbox_button_widget(
            lambda state, act=actors: _toggle(state, act if isinstance(act, (list, tuple)) else [act]),
            value=True, position=(x_cb, y), size=cb_size,
            color_on=color, color_off="#374151"
        )
        plotter.add_text(label, position=(x_cb + 35, y + 4), font_size=10,
                         color=p["text_label"], font="arial")

    # Group header helper
    def _add_header(label, y):
        plotter.add_text(label, position=(x_cb, y + 6), font_size=10,
                         color=p["text_dim"], font="arial")

    # ── ANATOMY ──
    _add_header("▸ ANATOMY", y_pos); y_pos += step
    _add_row("Femur", [a_femur], p["femur"], y_pos); y_pos += step
    _add_row("Tibia", [a_tibia], p["tibia"], y_pos); y_pos += step
    _add_row("ACL / Graft", [a_acl], p["acl"], y_pos); y_pos += step
    _add_row("Footprints", [a_fp_fem, a_fp_tib], p["footprint"], y_pos); y_pos += step

    # ── CENTROIDS & VECTOR ──
    _add_header("▸ CENTROIDS", y_pos); y_pos += step
    _add_row("Centroids + Vector", [a_cent_fem, a_cent_tib, a_acl_vec], p["acl_vector"], y_pos)
    y_pos += step

    # ── TIBIAL PLATEAU ──
    _add_header("▸ TIBIAL PLATEAU", y_pos); y_pos += step
    _add_row("Plateau Plane", [a_plateau], p["plateau"], y_pos); y_pos += step
    _add_row("RANSAC In/Out", [a_inliers, a_outliers], p["inlier"], y_pos); y_pos += step

    # ── B&H GRID ──
    bh_actors = [a_bh, a_bh_ref, a_bh_ref1, a_bh_ref2, a_blum, a_blum1, a_blum2]
    _add_header("▸ BERNARD-HERTEL", y_pos); y_pos += step
    _add_row("B&H Grid + Blumensaat", bh_actors, p["blumensaat"], y_pos); y_pos += step

    # ── ATT ──
    if has_att:
        att_actors = [a_att_tl, a_att_fl, a_att_m, a_att_ts, a_att_fs]
        _add_header("▸ ATT", y_pos); y_pos += step
        _add_row("ATT Measurement", att_actors, p["att_measure"], y_pos); y_pos += step

    # ── STÄUBLI ──
    if has_staubli:
        st_actors = [a_st_l, a_st_a, a_st_p, a_st_c]
        _add_header("▸ STÄUBLI", y_pos); y_pos += step
        _add_row("Stäubli AP Line", st_actors, p["staubli"], y_pos); y_pos += step

    # ── Launch ──────────────────────────────────────────────────────
    print("\n[INFO] Starting interactive 3D viewer. Use the checkboxes to toggle element groups.")
    plotter.show()


# ═══════════════════════════════════════════════════════════════════════
# MRI Volume Viewer (Grayscale Orthoslices & Volume Rendering)
# ═══════════════════════════════════════════════════════════════════════
def visualize_mri_volume(image_input, spacing=None, title="MRI 3D Volume Viewer"):
    """
    Interactive PyVista viewer for raw MRI / grayscale 3D volumes.
    Displays orthogonal slices with medical window/level, bounding box, axes, and metadata.
    
    Args:
        image_input (str, sitk.Image, or np.ndarray): File path, SimpleITK image, or 3D numpy array.
        spacing (tuple, optional): Voxel spacing (sz, sy, sx) if numpy array.
        title (str): Window title.
    """
    import SimpleITK as sitk

    if isinstance(image_input, str):
        sitk_img = sitk.ReadImage(image_input)
        sp = sitk_img.GetSpacing()
        spacing = (sp[0], sp[1], sp[2])
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        if title == "MRI 3D Volume Viewer":
            import os
            title = f"MRI Volume: {os.path.basename(image_input)}"
    elif isinstance(image_input, sitk.Image):
        sp = image_input.GetSpacing()
        spacing = (sp[0], sp[1], sp[2])
        arr = sitk.GetArrayFromImage(image_input).astype(np.float32)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    p = _PALETTE
    grid = pv.ImageData()
    grid.dimensions = arr.shape
    grid.spacing = spacing
    grid.point_data["values"] = arr.flatten(order="F")

    plotter = pv.Plotter(title=title, window_size=(1400, 900))
    plotter.set_background(p["bg_bottom"], top=p["bg_top"])

    # Orthogonal slices
    slices = grid.slice_orthogonal()
    plotter.add_mesh(
        slices, cmap="bone", show_scalar_bar=True,
        scalar_bar_args={"title": "Intensity", "color": p["text"], "width": 0.25}
    )
    plotter.add_bounding_box(color=p["text_dim"])

    plotter.add_axes(
        color=p["text_dim"],
        line_width=2,
        xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)"
    )

    plotter.add_text(
        title,
        position="upper_left",
        font_size=13,
        color=p["text"],
        font="arial",
        shadow=True,
    )

    # Volume metadata overlay
    v_min, v_max = float(arr.min()), float(arr.max())
    info_text = (
        f"Dimensions: {arr.shape[2]} x {arr.shape[1]} x {arr.shape[0]}\n"
        f"Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm\n"
        f"Intensity: [{v_min:.1f}, {v_max:.1f}]"
    )
    plotter.add_text(
        info_text,
        position="upper_right",
        font_size=10,
        color=p["text"],
        font="courier",
        shadow=True,
    )

    print(f"\n[INFO] Starting MRI Volume Viewer: {title}")
    plotter.show()


# ═══════════════════════════════════════════════════════════════════════
# Smart Universal Volume Visualizer
# ═══════════════════════════════════════════════════════════════════════
def smart_visualize(primary_path, secondary_path=None):
    """
    Intelligently inspects and visualizes ANY medical volume in PyVista:
    - If primary is a segmentation mask -> runs fast geometric analysis and opens anatomical 3D viewer.
    - If primary is an MRI scan and secondary is a mask -> runs anatomical 3D viewer with scan context.
    - If primary is an MRI scan -> opens interactive orthogonal slice volume viewer.
    
    Args:
        primary_path (str): Path to volume (.nii, .nii.gz, .dcm)
        secondary_path (str, optional): Secondary mask or image path.
    """
    import os
    import SimpleITK as sitk
    from anaknee.main_acl_analysis import run_geometric_analysis_from_mask

    if not os.path.exists(primary_path):
        raise FileNotFoundError(f"Primary volume file not found: {primary_path}")

    # Read image header / small sample to detect mask vs grayscale
    sitk_img = sitk.ReadImage(primary_path)
    arr_sample = sitk.GetArrayViewFromImage(sitk_img)
    u_vals = np.unique(arr_sample[:min(10, arr_sample.shape[0])])
    is_mask = (len(u_vals) <= 10 and arr_sample.max() <= 10) or os.path.basename(primary_path).startswith("mask_")

    if is_mask:
        print(f"[INFO] Detected segmentation mask in {os.path.basename(primary_path)}. Running fast 3D geometric analysis...")
        results_dict, mask_array, spacing_zyx, f_cent, t_cent, p_info, vis_data = run_geometric_analysis_from_mask(sitk_img)
        visualize_results(mask_array, spacing_zyx, vis_data)
    elif secondary_path and os.path.exists(secondary_path):
        # Secondary file might be mask
        print(f"[INFO] Using secondary mask: {os.path.basename(secondary_path)}...")
        results_dict, mask_array, spacing_zyx, f_cent, t_cent, p_info, vis_data = run_geometric_analysis_from_mask(secondary_path)
        visualize_results(mask_array, spacing_zyx, vis_data)
    else:
        print(f"[INFO] Detected grayscale MRI volume in {os.path.basename(primary_path)}. Opening 3D Orthoslice Viewer...")
        visualize_mri_volume(sitk_img, title=f"MRI Volume: {os.path.basename(primary_path)}")

