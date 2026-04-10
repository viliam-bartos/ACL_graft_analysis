import numpy as np
import pyvista as pv
import nibabel as nib
from scipy.ndimage import binary_dilation

def create_surface_mesh(binary_mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), smooth=True):
    """Converts a 3D numpy boolean/int array into a PyVista surface mesh."""
    # Pad mask to ensure closed surfaces at the volume boundaries
    padded_mask = np.pad(binary_mask, 1, mode='constant', constant_values=0)
    
    # Create PyVista Grid
    grid = pv.ImageData()
    grid.dimensions = padded_mask.shape
    grid.spacing = spacing
    grid.origin = tuple(o - s for o, s in zip(origin, spacing))
    
    # Add data, PyVista expects Fortran order for ImageData
    grid.point_data['values'] = padded_mask.flatten(order='F')
    
    # Run marching cubes for isosurface
    mesh = grid.contour([0.5])
    
    # Optional Laplacian smoothing for cleaner anatomical meshes
    if smooth and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=50, relaxation_factor=0.05)
        
    return mesh

def visualize_results(mask_data, spacing, vis_data):
    # =========================================================================
    # 1. Configuration & Data Extraction
    # =========================================================================
    
    # Extract calculated coordinates from the analysis pipeline
    dummy_femoral_centroid = vis_data.get('femoral_centroid', np.array([50.0, 60.0, 70.0]))
    dummy_tibial_centroid = vis_data.get('tibial_centroid', np.array([45.0, 55.0, 30.0]))
    
    dummy_plateau_normal = vis_data.get('plateau_normal', np.array([0.05, 0.05, 1.0]))
    # Ensure normal is normalized
    if np.linalg.norm(dummy_plateau_normal) > 0:
        dummy_plateau_normal = dummy_plateau_normal / np.linalg.norm(dummy_plateau_normal)
        
    dummy_plateau_center = vis_data.get('plateau_center', np.array([45.0, 55.0, 32.0]))
    
    # Bernard & Hertel Grid (Lateral Femoral Condyle Box) - keeping heuristic based on femoral centroid
    # In PyVista based on Numpy (R-L, S-I, A-P), dimensions are X=R-L, Y=S-I, Z=A-P.
    # A sagittal slice has a fixed R-L (X coordinate) and extends along Y and Z.
    bh_sagittal_x = dummy_femoral_centroid[0] if not np.isnan(dummy_femoral_centroid[0]) else 50.0
    bh_y_min, bh_y_max = 40.0, 80.0
    bh_z_min, bh_z_max = 60.0, 100.0

    origin = (0.0, 0.0, 0.0)

    # Label extraction (1=ACL, 2=Femur, 3=Tibia)
    mask_acl = mask_data == 1
    mask_femur = mask_data == 2
    mask_tibia = mask_data == 3
    
    # Footprint calculation: Intersection of slightly dilated ACL with Bone masks
    print("Isolating footprints...")
    dilated_acl = binary_dilation(mask_acl, iterations=2)
    footprint_femur_mask = dilated_acl & mask_femur
    footprint_tibia_mask = dilated_acl & mask_tibia

    # =========================================================================
    # 2. Generating PyVista Meshes
    # =========================================================================
    print("Generating 3D anatomical meshes (this may take a moment)...")
    mesh_acl = create_surface_mesh(mask_acl, spacing=spacing, origin=origin)
    mesh_femur = create_surface_mesh(mask_femur, spacing=spacing, origin=origin)
    mesh_tibia = create_surface_mesh(mask_tibia, spacing=spacing, origin=origin)
    
    # Footprint meshes (no smoothing so exact intersection voxels are seen)
    mesh_fp_femur = create_surface_mesh(footprint_femur_mask, spacing=spacing, origin=origin, smooth=False)
    mesh_fp_tibia = create_surface_mesh(footprint_tibia_mask, spacing=spacing, origin=origin, smooth=False)

    # =========================================================================
    # 3. Geometric Overlays Creation
    # =========================================================================
    # A. Centroids (Solid Spheres)
    sphere_femoral = pv.Sphere(radius=2.0, center=dummy_femoral_centroid)
    sphere_tibial = pv.Sphere(radius=2.0, center=dummy_tibial_centroid)
    
    # B. ACL Vector (Connecting Line)
    vector_line = pv.Line(dummy_femoral_centroid, dummy_tibial_centroid)
    
    # C. Tibial Plateau Plane
    # To prevent PyVista from arbitrarily rotating the plane edges (creating a 'rhombus'), 
    # we enforce the i_direction to be roughly the Left-Right axis (1, 0, 0), projected onto the plane.
    temp_i = np.array([1.0, 0.0, 0.0])
    temp_i_proj = temp_i - np.dot(temp_i, dummy_plateau_normal) * dummy_plateau_normal
    if np.linalg.norm(temp_i_proj) > 0:
        i_dir = temp_i_proj / np.linalg.norm(temp_i_proj)
    else:
        i_dir = np.array([0.0, 0.0, 1.0])
        
    j_dir = np.cross(dummy_plateau_normal, i_dir)
    j_dir = j_dir / np.linalg.norm(j_dir)

    plateau_plane = pv.Plane(center=dummy_plateau_center, direction=dummy_plateau_normal,
                             i_size=60, j_size=60, 
                             i_resolution=1, j_resolution=1)
    
    # In older PyVista versions, pv.Plane doesn't reliably accept i_direction/j_direction in kwargs.
    # The safest way to orient the plane is to generate it at origin with Z normal, then transform it.
    base_plane = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=60, j_size=60, i_resolution=1, j_resolution=1)
    # Transformation matrix: Columns are i_dir, j_dir, normal, and the last column is the translation.
    trans_matrix = np.eye(4)
    trans_matrix[0:3, 0] = i_dir
    trans_matrix[0:3, 1] = j_dir
    trans_matrix[0:3, 2] = dummy_plateau_normal
    trans_matrix[0:3, 3] = dummy_plateau_center
    plateau_plane = base_plane.transform(trans_matrix)
    # D. Bernard & Hertel Grid
    bh_lines = []
    bh_grid_info = vis_data.get('bh_grid_info', {})
    bh_grid_data = bh_grid_info.get('lines', []) if isinstance(bh_grid_info, dict) else bh_grid_info
    ref_edge = bh_grid_info.get('ref_edge') if isinstance(bh_grid_info, dict) else None
    blum_line = bh_grid_info.get('blum_line') if isinstance(bh_grid_info, dict) else None
    
    if bh_grid_data:
        for start_pt, end_pt in bh_grid_data:
            bh_lines.append(pv.Line(start_pt, end_pt))
    else:
        # Fallback dummy logic
        for i in range(5):
            y = bh_y_min + i * (bh_y_max - bh_y_min) / 4.0
            bh_lines.append(pv.Line([bh_sagittal_x, y, bh_z_min], [bh_sagittal_x, y, bh_z_max]))
        for i in range(5):
            z = bh_z_min + i * (bh_z_max - bh_z_min) / 4.0
            bh_lines.append(pv.Line([bh_sagittal_x, bh_y_min, z], [bh_sagittal_x, bh_y_max, z]))
            
    bh_grid_multiblock = pv.MultiBlock(bh_lines)

    # E. Reference edge and orientation points
    if ref_edge:
        ref_line = pv.Line(ref_edge[0], ref_edge[1])
        ref_point_1 = pv.Sphere(radius=2.0, center=ref_edge[0])
        ref_point_2 = pv.Sphere(radius=2.0, center=ref_edge[1])
    else:
        ref_line = pv.PolyData()
        ref_point_1 = pv.PolyData()
        ref_point_2 = pv.PolyData()
        
    # F. Blumensaat line (True Regression Line)
    if blum_line:
        actor_blum_line = pv.Line(blum_line[0], blum_line[1])
        actor_blum_pt1 = pv.Sphere(radius=2.5, center=blum_line[0])
        actor_blum_pt2 = pv.Sphere(radius=2.5, center=blum_line[1])
    else:
        actor_blum_line = pv.PolyData()
        actor_blum_pt1 = pv.PolyData()
        actor_blum_pt2 = pv.PolyData()

    # G. ATT Lines (Kolmice)
    att_info = vis_data.get('att_info', {})
    if att_info and 'tibia_pt' in att_info:
        t_pt = att_info['tibia_pt']
        f_pt = att_info['femur_pt']
        v_ant = att_info['v_anterior']
        n_p = att_info['plane_normal']
        
        # Osa kolmice k platu (nahoru/dolu podel normaly)
        t_line_start = t_pt - 40 * n_p
        t_line_end = t_pt + 40 * n_p
        f_line_start = f_pt - 40 * n_p
        f_line_end = f_pt + 40 * n_p
        
        actor_att_t_line = pv.Line(t_line_start, t_line_end)
        actor_att_f_line = pv.Line(f_line_start, f_line_end)
        actor_att_t_pt = pv.Sphere(radius=2.5, center=t_pt)
        actor_att_f_pt = pv.Sphere(radius=2.5, center=f_pt)
        
        # Linie měření (vzdálenost mezi stěnami) v predozadní projektované ose v_ant
        dist = np.dot((t_pt - f_pt), v_ant)
        measure_end = f_pt + dist * v_ant
        actor_att_measure = pv.Line(f_pt, measure_end)
    else:
        actor_att_t_line = pv.PolyData()
        actor_att_f_line = pv.PolyData()
        actor_att_t_pt = pv.PolyData()
        actor_att_f_pt = pv.PolyData()
        actor_att_measure = pv.PolyData()

    # =========================================================================
    # 4. PyVista Plotter Setup and Rendering
    # =========================================================================
    plotter = pv.Plotter(title="ACL 3D Geometric Analysis Verification")
    plotter.set_background("white")
    
    # Add main anatomical actors
    actor_femur = plotter.add_mesh(mesh_femur, color="ivory", opacity=0.3, label="Femur")
    actor_tibia = plotter.add_mesh(mesh_tibia, color="beige", opacity=0.3, label="Tibia")
    actor_acl = plotter.add_mesh(mesh_acl, color="orange", opacity=0.6, label="ACL")
    
    # Add exact contact areas (Footprints)
    actor_fp_femur = plotter.add_mesh(mesh_fp_femur, color="red", label="Femoral Footprint")
    actor_fp_tibia = plotter.add_mesh(mesh_fp_tibia, color="red", label="Tibial Footprint")
    
    # Add geometric calculation actors
    actor_cent_femur = plotter.add_mesh(sphere_femoral, color="blue", label="Femoral Centroid")
    actor_cent_tibia = plotter.add_mesh(sphere_tibial, color="green", label="Tibial Centroid")
    actor_vector = plotter.add_mesh(vector_line, color="purple", line_width=5, label="ACL Vector")
    actor_plane = plotter.add_mesh(plateau_plane, color="cyan", opacity=0.4, label="Tibial Plateau")
    actor_bh_grid = plotter.add_mesh(bh_grid_multiblock, color="black", line_width=3, label="Bernard & Hertel Grid")
    
    if ref_edge:
        actor_ref_edge = plotter.add_mesh(ref_line, color="red", line_width=4, label="B&H Reference Edge")
        actor_ref_pt1 = plotter.add_mesh(ref_point_1, color="yellow", label="B&H Ref Point 1")
        actor_ref_pt2 = plotter.add_mesh(ref_point_2, color="magenta", label="B&H Ref Point 2")
    else:
        actor_ref_edge = actor_ref_pt1 = actor_ref_pt2 = None
        
    if blum_line:
        actor_bl = plotter.add_mesh(actor_blum_line, color="green", line_width=6, label="Regrese: Blumensaatova linie")
        actor_bl_p1 = plotter.add_mesh(actor_blum_pt1, color="lime", label="Blumensaat Pt1")
        actor_bl_p2 = plotter.add_mesh(actor_blum_pt2, color="lime", label="Blumensaat Pt2")
    else:
        actor_bl = actor_bl_p1 = actor_bl_p2 = None

    if att_info and 'tibia_pt' in att_info:
        actor_att_t = plotter.add_mesh(actor_att_t_line, color="red", line_width=4, label="ATT Kolmice: Tibia")
        actor_att_f = plotter.add_mesh(actor_att_f_line, color="blue", line_width=4, label="ATT Kolmice: Femur")
        actor_att_m = plotter.add_mesh(actor_att_measure, color="yellow", line_width=5, label="ATT Vzdálenost")
        plotter.add_mesh(actor_att_t_pt, color="red")
        plotter.add_mesh(actor_att_f_pt, color="blue")
    else:
        actor_att_t = actor_att_f = actor_att_m = None

    # Add default interaction tools
    plotter.add_axes()
    plotter.add_legend(bcolor=(1, 1, 1), face='rectangle')

    # Add Interactive Checkbox Toggles for Visibility
    def toggle_vis(flag, actor):
        actor.SetVisibility(flag)
    
    start_y = 30
    step = 40
    size = 30
    
    elements = [
        ("Femur Geometry", actor_femur, "ivory"),
        ("Tibia Geometry", actor_tibia, "beige"),
        ("ACL Geometry", actor_acl, "orange"),
        ("Femoral Footprint", actor_fp_femur, "red"),
        ("Tibial Footprint", actor_fp_tibia, "red"),
        ("Tibial Plateau Plane", actor_plane, "cyan"),
        ("Bernard & Hertel Grid", actor_bh_grid, "black"),
        ("Femoral Centroid", actor_cent_femur, "blue"),
        ("Tibial Centroid", actor_cent_tibia, "green"),
        ("ACL Vector", actor_vector, "purple")
    ]
    
    if ref_edge:
        elements.extend([
            ("B&H Reference Edge", actor_ref_edge, "red"),
            ("B&H Ref Point 1", actor_ref_pt1, "yellow"),
            ("B&H Ref Point 2", actor_ref_pt2, "magenta")
        ])
        
    if blum_line:
        elements.extend([
            ("Regrese: Blumensaatova Linie", actor_bl, "green"),
            ("Blum. Začátek", actor_bl_p1, "lime"),
            ("Blum. Konec", actor_bl_p2, "lime")
        ])
        
    if att_info and 'tibia_pt' in att_info:
        elements.extend([
            ("ATT: Kolmice Tibie (R)", actor_att_t, "red"),
            ("ATT: Kolmice Femuru (B)", actor_att_f, "blue"),
            ("ATT: Vzdálenost (Y)", actor_att_m, "yellow")
        ])
    
    for i, (name, actor, color_code) in enumerate(elements):
        y_pos = start_y + i * step
        plotter.add_checkbox_button_widget(
            lambda state, act=actor: toggle_vis(state, act), 
            value=True, 
            position=(10, y_pos), 
            size=size, 
            color_on=color_code, 
            color_off="grey"
        )
        plotter.add_text(name, position=(50, y_pos + 5), font_size=11, color="black", font="arial")

    print("\n[INFO] Starting interactive 3D viewer. Use the checkboxes to toggle elements.")
    plotter.show()
