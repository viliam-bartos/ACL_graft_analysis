import logging
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import nibabel.orientations as nio


def _reorient_to_ria(sitk_img):
    """
    Reorient a SimpleITK image to RIA orientation using nibabel.
    
    RIA means numpy array dimensions increase as:
        dim 0: L -> R (Right)      = R-L axis
        dim 1: S -> I (Inferior)   = S-I axis
        dim 2: P -> A (Anterior)   = A-P axis
    
    This matches the hardcoded axis assumptions in the analysis code:
        - Sagittal slice = femur_mask[dim0, :, :]  (dim0 = R-L)
        - ap_global = [0, 0, 1]                    (dim2 = A-P)
        - v_short[1] > 0 = downward                (dim1 = S-I)
    
    Args:
        sitk_img: SimpleITK.Image to reorient
        
    Returns:
        SimpleITK.Image in RIA orientation
    """
    arr = sitk.GetArrayFromImage(sitk_img)
    spacing_xyz = sitk_img.GetSpacing()       # (sx, sy, sz)
    origin_xyz = sitk_img.GetOrigin()          # (ox, oy, oz)
    direction = sitk_img.GetDirection()        # 9-element flat tuple
    
    dir_mat = np.array(direction).reshape(3, 3)
    affine = np.eye(4)
    for i in range(3):
        for j in range(3):
            affine[i, j] = dir_mat[i, j] * spacing_xyz[j]
    affine[:3, 3] = origin_xyz
    
    # nibabel expects (i,j,k) -> (x,y,z), but sitk.GetArrayFromImage returns (k,j,i) numpy order.
    affine_nib = np.eye(4)
    affine_nib[:3, 0] = affine[:3, 2]  # numpy dim0 = sitk k -> physical
    affine_nib[:3, 1] = affine[:3, 1]  # numpy dim1 = sitk j -> physical
    affine_nib[:3, 2] = affine[:3, 0]  # numpy dim2 = sitk i -> physical
    affine_nib[:3, 3] = affine[:3, 3]  # origin
    
    nib_img = nib.Nifti1Image(arr, affine_nib)
    
    current_ornt = nio.io_orientation(nib_img.affine)
    target_ornt = nio.axcodes2ornt("RIA")
    transform = nio.ornt_transform(current_ornt, target_ornt)
    
    identity = np.array([[0, 1], [1, 1], [2, 1]])
    if np.array_equal(transform, identity):
        logging.info("  -> Anaknee: Data is already in RIA orientation.")
        return sitk_img
    
    current_codes = nio.ornt2axcodes(current_ornt)
    logging.info(f"  -> Anaknee: Reorienting from {''.join(current_codes)} to RIA.")
    
    reoriented_nib = nib_img.as_reoriented(transform)
    # Use native dataobj without full float64 cast to preserve memory and speed
    reoriented_arr = np.ascontiguousarray(np.asanyarray(reoriented_nib.dataobj))
    
    new_affine = reoriented_nib.affine
    new_spacing_zyx = (
        np.linalg.norm(new_affine[:3, 0]),
        np.linalg.norm(new_affine[:3, 1]),
        np.linalg.norm(new_affine[:3, 2]),
    )
    new_spacing_sitk = (new_spacing_zyx[2], new_spacing_zyx[1], new_spacing_zyx[0])
    
    dir_cols = []
    for c in [2, 1, 0]:
        col = new_affine[:3, c]
        norm = np.linalg.norm(col)
        if norm > 0:
            col = col / norm
        dir_cols.append(col)
    new_direction = tuple(np.array(dir_cols).T.flatten())
    new_origin = tuple(new_affine[:3, 3])
    
    standardized_sitk = sitk.GetImageFromArray(reoriented_arr)
    standardized_sitk.SetSpacing(new_spacing_sitk)
    standardized_sitk.SetDirection(new_direction)
    standardized_sitk.SetOrigin(new_origin)
    
    return standardized_sitk
