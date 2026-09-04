import logging
import numpy as np
from skimage.measure import ransac


class PlaneModel3D:
    """
    3D plane model compatible with skimage.measure.ransac.
    
    A plane is defined by a point on the plane (centroid) and a unit normal vector.
    The model can be estimated from >= 3 non-collinear points using SVD.
    """
    
    def __init__(self):
        self.normal = None
        self.centroid = None

    @classmethod
    def from_estimate(cls, data):
        """Construct model from data estimation (skimage >= 0.26 standard)."""
        model = cls()
        success = model.estimate(data)
        return model if success else None
    
    def estimate(self, data):
        """Estimate plane parameters from a set of 3D points (N x 3).
        
        Returns True if estimation was successful, False otherwise.
        """
        if data.shape[0] < 3:
            return False
        
        self.centroid = data.mean(axis=0)
        centered = data - self.centroid
        
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            # Check for degenerate case: collinear points have rank < 2,
            # meaning the second singular value is near zero.
            # (Three coplanar points are NOT degenerate — they define a valid plane.)
            if s[1] / (s[0] + 1e-12) < 1e-10:
                return False
            self.normal = vh[-1, :]
            norm = np.linalg.norm(self.normal)
            if norm > 0:
                self.normal = self.normal / norm
            return True
        except np.linalg.LinAlgError:
            return False
    
    def residuals(self, data):
        """Compute signed perpendicular distances from points to the plane."""
        return np.abs((data - self.centroid).dot(self.normal))


def get_tibial_plateau_plane(tibia_mask, spacing, proximal_point=None, top_fraction=0.25):
    """
    Robust estimation of the tibial plateau plane using RANSAC.

    Strategy:
    1. Iteratively selects the proximal fraction of tibia voxels (plateau region).
    2. Applies fast RANSAC with subsampling to fit a plane robust to anatomical outliers
       (tibial spine, osteophytes, posterior slope irregularities).
    3. Falls back to standard SVD fit if RANSAC fails.
      
    Args:
        tibia_mask (np.ndarray): boolean mask of tibia voxels (in voxel coords)
        spacing (tuple): voxel spacing (sz, sy, sx) in mm
        proximal_point (iterable, optional): physical coordinate (z,y,x) pointing towards femur
        top_fraction (float): fraction of voxels to take from the proximal end

    Returns:
        normal (np.ndarray): unit normal vector of the fitted plateau plane (physical coords)
        centroid (np.ndarray): physical centroid of the selected plateau points
        inlier_points (np.ndarray): physical coords of RANSAC inliers (N x 3), or None
        outlier_points (np.ndarray): physical coords of RANSAC outliers (M x 3), or None
    """
    s0, s1, s2 = spacing
    coords = np.argwhere(tibia_mask)
    if len(coords) == 0:
        return np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0]), None, None

    phys_coords = coords * np.array([s0, s1, s2])
    mean_all = phys_coords.mean(axis=0)

    # 1. Initialize the upward axis
    if proximal_point is not None:
        current_axis = np.array(proximal_point) - mean_all
    else:
        current_axis = np.array([0.0, -1.0, 0.0])
        
    nrm = np.linalg.norm(current_axis)
    if nrm > 0:
        current_axis = current_axis / nrm
    else:
        current_axis = np.array([0.0, -1.0, 0.0])

    # 2. Iterative proximal voxel selection (3 iterations to converge)
    n_top = max(3, int(len(phys_coords) * float(top_fraction)))
    centroid = mean_all
    top_points = phys_coords
    
    for _ in range(3):
        projections = phys_coords.dot(current_axis)
        top_idx = np.argsort(projections)[-n_top:]
        top_points = phys_coords[top_idx]
        
        centroid = top_points.mean(axis=0)
        centered = top_points - centroid
        
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            new_normal = vh[-1, :]
        except Exception:
            new_normal = current_axis
            
        if np.dot(new_normal, current_axis) < 0:
            new_normal = -new_normal
            
        nrm = np.linalg.norm(new_normal)
        if nrm > 0:
            current_axis = new_normal / nrm

    # 3. RANSAC robust plane fitting on the selected plateau points
    inlier_points = None
    outlier_points = None
    
    if len(top_points) >= 10:
        try:
            # For large point clouds (e.g. 100k+ voxels), subsample for fast robust fitting
            if len(top_points) > 3000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(top_points), size=3000, replace=False)
                fit_points = top_points[sample_idx]
                trials = 200
            else:
                fit_points = top_points
                trials = 500

            model, _ = ransac(
                fit_points,
                PlaneModel3D,
                min_samples=3,
                residual_threshold=1.5,  # mm — typical cortical surface tolerance
                max_trials=trials,
            )
            
            if model is not None and model.normal is not None:
                all_res = model.residuals(top_points)
                inlier_mask = all_res < 1.5
                n_inliers = np.sum(inlier_mask)
                inlier_ratio = n_inliers / len(top_points)
                
                if inlier_ratio > 0.15:
                    normal = model.normal.copy()
                    centroid = model.centroid.copy()
                    inlier_points = top_points[inlier_mask]
                    outlier_points = top_points[~inlier_mask]
                    
                    logging.info(
                        f"  -> RANSAC plateau fit: {n_inliers}/{len(top_points)} inliers "
                        f"({inlier_ratio:.1%})"
                    )
                else:
                    logging.warning(
                        f"  -> RANSAC inlier ratio too low ({inlier_ratio:.1%}), "
                        f"falling back to SVD."
                    )
                    normal = current_axis
            else:
                logging.warning("  -> RANSAC model fit returned None, falling back to SVD.")
                normal = current_axis
        except Exception as e:
            logging.warning(f"  -> RANSAC failed ({e}), falling back to SVD.")
            normal = current_axis
    else:
        logging.info("  -> Too few plateau points for RANSAC, using SVD fit.")
        normal = current_axis

    # Orient normal strictly towards proximal_point (femur) if available
    if proximal_point is not None:
        if np.dot(normal, np.array(proximal_point) - centroid) < 0:
            normal = -normal

    return normal, centroid, inlier_points, outlier_points
