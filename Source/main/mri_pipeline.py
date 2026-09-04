import os
import sys
import glob
import logging
import traceback
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import nibabel.orientations as nio
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import (
    DiceMetric,
    compute_hausdorff_distance,
)
from monai.transforms import AsDiscrete


CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent

for path in [
    SOURCE_DIR,
    SOURCE_DIR / "kanonizace",
    SOURCE_DIR / "blackwell",
    SOURCE_DIR / "anaknee",
]:
    if str(path) not in sys.path:
        sys.path.append(str(path))


try:
    from models.unet3d import LightUNet3D
except ImportError:
    try:
        from Source.models.unet3d import LightUNet3D
    except ImportError:
        from WORKSTATION_BLACKWELL_MULTICLASS_5CV import LightUNet3D  # noqa: E402
from main_acl_analysis import run_analysis  # noqa: E402
from visualizator_analyzator import visualize_results  # noqa: E402


_RIGHT_PATTERN = re.compile(
    r"(?:^|[_\-\s\.])(right|dexter|dext|dx|rt|prav[áaéeý]?)(?:$|[_\-\s\.])",
    re.IGNORECASE,
)
_LEFT_PATTERN = re.compile(
    r"(?:^|[_\-\s\.])(left|sinister|sinist|sin|lt|lev[áaéeý]?)(?:$|[_\-\s\.])",
    re.IGNORECASE,
)


def classify_laterality(file_path: str):
    """Returns 'Left', 'Right', or None based on filename."""
    basename = os.path.basename(file_path)
    name_without_ext = basename.split(".")[0]

    has_right = _RIGHT_PATTERN.search(name_without_ext)
    has_left = _LEFT_PATTERN.search(name_without_ext)

    if has_right and not has_left:
        return "Right"
    elif has_left and not has_right:
        return "Left"
    else:
        logging.info(f"Laterality not detected in filename: '{basename}'")
        return None


def is_dicom_input(path):
    """True if path is a DICOM file or folder with DICOM series."""
    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return path.lower().endswith(".dcm")
    if os.path.isdir(path):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
            return len(series_ids) > 0
        except Exception:
            return False
    return False


def convert_dicom_to_nifti(dicom_path, output_dir):
    """Convert DICOM series to NIfTI via SimpleITK. Returns output paths."""
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(dicom_path):
        dicom_dir = os.path.dirname(dicom_path)
    else:
        dicom_dir = dicom_path

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    nifti_paths = []

    for idx, series_id in enumerate(series_ids):
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dicom_dir, series_id
        )
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        if len(series_ids) == 1:
            folder_name = os.path.basename(os.path.normpath(dicom_dir))
            output_name = f"{folder_name}.nii.gz"
        else:
            output_name = f"series_{idx}.nii.gz"

        output_path = os.path.join(output_dir, output_name)
        sitk.WriteImage(image, output_path)
        logging.info(f"  -> DICOM series converted: {output_path}")
        nifti_paths.append(output_path)

    return nifti_paths


try:
    from config import PipelineSettings, DEFAULT_SETTINGS
    CONFIG = DEFAULT_SETTINGS.to_dict()
except ImportError:
    try:
        from Source.config import PipelineSettings, DEFAULT_SETTINGS
        CONFIG = DEFAULT_SETTINGS.to_dict()
    except ImportError:
        CONFIG = {
            "mode": "FILE",  # "FILE" or "FOLDER"
            "input_path": r"Data\reference\right_case_074.nii.gz",
            "input_dir": r"",
            "output_dir": r"Data\reference\vysledky_074",
            "log_file": "pipeline.log",
            "anaknee_ref_mri": r"Data\reference\right_case_074.nii.gz",
            "gt_masks_dir": r"",
            "model_ckpt": r"",
            "patch_size": (128, 128, 80),
            "base_filters": 64,
            "use_ensemble": True,
            "ensemble_dir": r"Data\5CV",
            "ensemble_pattern": "best_model_fold_*.pth",
            "run_inference": 1,
            "run_segmentation_analysis": 0,
            "run_anatomical_analysis": 1,
            "post_proc_classes": {
                1: {"lcc": True, "hole_filling": False, "closing": False},
                2: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
                3: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
            },
        }


def setup_logging(output_dir, log_file_name):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file_name)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def resample_image_sitk(sitk_img, target_spacing=(0.5, 0.5, 0.5)):
    """Resamples to target spacing using B-Spline interpolation."""
    original_spacing = sitk_img.GetSpacing()
    if np.allclose(original_spacing, target_spacing, atol=1e-3):
        logging.info("  -> Spacing is already correct, skipping resample.")
        return sitk_img

    logging.info(f"  -> Resampling from {original_spacing} to {target_spacing}")
    orig_size = np.array(sitk_img.GetSize(), dtype=int)
    new_size = np.round(
        orig_size * (np.array(original_spacing) / np.array(target_spacing))
    ).astype(int)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_img.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(sitk_img)


def force_reorient_pil(nifti_path):
    """Enforces PIL (ASR) orientation via nibabel."""
    img = nib.load(nifti_path)
    target_ornt = nio.axcodes2ornt("PIL")
    orig_ornt = nio.io_orientation(img.affine)
    transform = nio.ornt_transform(orig_ornt, target_ornt)

    if not np.array_equal(transform, [[0, 1], [1, 1], [2, 1]]):
        logging.info("  -> Orientation is not PIL, applying reorientation.")
        new_img = img.as_reoriented(transform)
        nib.save(new_img, nifti_path)
    else:
        logging.info("  -> Orientation is already PIL.")


def postprocess_mask(mask_array, config_classes):
    """Applies LCC, hole filling, closing per label."""
    output_mask = np.zeros_like(mask_array)
    unique_labels = np.unique(mask_array)

    for lbl in unique_labels:
        if lbl == 0:
            continue

        lbl_mask = mask_array == lbl

        if lbl in config_classes:
            cfg = config_classes[lbl]

            if cfg.get("hole_filling", False):
                lbl_mask = ndimage.binary_fill_holes(lbl_mask)

            if cfg.get("closing", False):
                k_size = cfg.get("closing_kernel", 2)
                struct = ndimage.generate_binary_structure(3, 1)
                if k_size > 1:
                    struct = ndimage.iterate_structure(struct, k_size)
                lbl_mask = ndimage.binary_closing(lbl_mask, structure=struct)

            if cfg.get("lcc", False):
                labeled, num_features = ndimage.label(lbl_mask)
                if num_features > 0:
                    sizes = ndimage.sum(lbl_mask, labeled, range(1, num_features + 1))
                    largest_idx = np.argmax(sizes) + 1
                    lbl_mask = labeled == largest_idx

        output_mask[lbl_mask] = lbl

    return output_mask


def _preprocess_image(img_path):
    """Normalizes NIfTI volume intensities for inference."""
    sitk_img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

    img_array = np.transpose(img_array, (2, 1, 0))
    p05 = np.percentile(img_array, 0.5)
    p995 = np.percentile(img_array, 99.5)
    img_array = np.clip(img_array, p05, p995)
    img_array = img_array - p05

    non_zero = img_array > 0
    if np.any(non_zero):
        img_array[non_zero] = (img_array[non_zero] - img_array[non_zero].mean()) / (
            img_array[non_zero].std() + 1e-8
        )

    return img_array


def _apply_thresholds(probs):
    """Applies per-class thresholds to probability map."""
    pred_argmax = torch.argmax(probs, dim=0)  # [X, Y, Z]
    pred = torch.zeros_like(pred_argmax)
    pred[(pred_argmax == 1) & (probs[1] >= 0.45)] = 1  # ACL
    pred[(pred_argmax == 2) & (probs[2] >= 0.90)] = 2  # Femur
    pred[(pred_argmax == 3) & (probs[3] >= 0.80)] = 3  # Tibia
    return pred


def infer_model(img_path, model, device, config):
    """Single-model sliding window inference."""
    img_array = _preprocess_image(img_path)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = sliding_window_inference(
                inputs=tensor,
                roi_size=config["patch_size"],
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred = _apply_thresholds(probs)

    pred_np = pred.cpu().numpy()
    pred_np = np.transpose(pred_np, (2, 1, 0)) # Transpose back to (Z, Y, X)
    return pred_np


def infer_ensemble(img_path, ensemble_models, device, config):
    """Ensemble inference averaging probabilities across folds."""
    logging.info(f"  -> Ensemble inference with {len(ensemble_models)} models.")
    img_array = _preprocess_image(img_path)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    accumulated_probs = None

    for fold_idx, model in enumerate(ensemble_models):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = sliding_window_inference(
                    inputs=tensor,
                    roi_size=config["patch_size"],
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
            fold_probs = torch.softmax(outputs, dim=1).squeeze(0).cpu()

            del outputs
            torch.cuda.empty_cache()

            logging.info(f"     Fold {fold_idx + 1}/{len(ensemble_models)} complete.")

            if accumulated_probs is None:
                accumulated_probs = fold_probs.clone()
            else:
                accumulated_probs += fold_probs

    avg_probs = accumulated_probs / len(ensemble_models)
    pred = _apply_thresholds(avg_probs)

    pred_np = pred.cpu().numpy()
    pred_np = np.transpose(pred_np, (2, 1, 0)) # Transpose back to (Z, Y, X)
    return pred_np


def perform_segmentation_analysis(output_dir, gt_dir):
    """Computes Dice and HD95 vs ground truth."""
    logging.info("--- Starting Segmentation Metrics Analysis ---")

    results = []
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_func = AsDiscrete(to_onehot=4)

    pred_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not pred_files:
        logging.warning("No output masks found. Skipping metrics analysis.")
        return

    for p_path in pred_files:
        basename = os.path.basename(p_path)

        match = re.search(r"(\d+)", basename)
        if not match:
            logging.warning(f"Could not extract ID from prediction filename: {basename}")
            continue

        file_id = match.group(1)

        gt_search_pattern = os.path.join(gt_dir, f"*{file_id}*.nii.gz")
        gt_matches = glob.glob(gt_search_pattern)

        if not gt_matches:
            logging.warning(f"GT mask not found for ID: {file_id}")
            continue

        gt_path = gt_matches[0]
        if len(gt_matches) > 1:
            logging.warning(f"Multiple GT matches found for ID {file_id}. Using: {os.path.basename(gt_path)}")

        logging.info(f"Comparing GT vs Pred for ID {file_id}")

        p_sitk = sitk.ReadImage(p_path)
        g_sitk = sitk.ReadImage(gt_path)

        p_arr = sitk.GetArrayFromImage(p_sitk)
        g_arr = sitk.GetArrayFromImage(g_sitk)

        try:
            p_t = post_func(torch.from_numpy(p_arr).unsqueeze(0))
            g_t = post_func(torch.from_numpy(g_arr).unsqueeze(0))
            
            dx, dy, dz = p_sitk.GetSpacing()
            spacing_zyx = [dz, dy, dx]

            dice_metric(y_pred=[p_t], y=[g_t])
            dice = dice_metric.get_buffer()[-1]

            hd95_tensor = compute_hausdorff_distance(
                y_pred=p_t.unsqueeze(0),
                y=g_t.unsqueeze(0),
                include_background=False,
                percentile=95,
                spacing=spacing_zyx,
            )
            hd95 = hd95_tensor[0]

            for class_idx, class_name in enumerate(["ACL", "Femur", "Tibia"]):
                d_val = (
                    dice[class_idx].item() if not torch.isnan(dice[class_idx]) else 0.0
                )
                try:
                    h_val = hd95[class_idx].item()
                except Exception:
                    h_val = float("nan")

                results.append(
                    {
                        "Soubor": basename,
                        "Struktura": class_name,
                        "Dice": d_val,
                        "HD95 [mm]": h_val,
                    }
                )
        except Exception as e:
            logging.error(f"Error during verification of {basename}: {e}")

    if not results:
        return

    df = pd.DataFrame(results)
    stats_dir = os.path.join(output_dir, "Segmentation_Reports")
    os.makedirs(stats_dir, exist_ok=True)

    csv_path = os.path.join(stats_dir, "segmentation_metrics.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Metrics saved to: {csv_path}")

    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Struktura", y="Dice", palette="tab10")
    plt.title("Dice Score", fontweight="bold")
    plt.ylim(0, 1.05)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="Struktura", y="HD95 [mm]", palette="tab10")
    plt.title("Hausdorff 95%", fontweight="bold")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "metrics_boxplots.png"), dpi=200)
    plt.savefig(os.path.join(stats_dir, "metrics_boxplots.pdf"))
    plt.close()


def process_single_volume(
    file_path, model, device, run_viz_at_end=False, ensemble_models=None,
    laterality_callback=None,
):
    """Main pipeline for a single MRI volume."""
    logging.info(f"====== START PROCESSING: {os.path.basename(file_path)} ======")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file missing: {file_path}")

    orig_sitk = sitk.ReadImage(file_path)

    temp_nifti_path = os.path.join(
        CONFIG["output_dir"], f"process_raw_{os.path.basename(file_path)}"
    )
    final_basename = os.path.basename(file_path)
    if not final_basename.startswith("mask_"):
        final_basename = f"mask_{final_basename}"
    final_mask_path = os.path.join(CONFIG["output_dir"], final_basename)

    working_sitk = orig_sitk

    # 1. Resampling
    working_sitk = resample_image_sitk(working_sitk, target_spacing=(0.5, 0.5, 0.5))
    sitk.WriteImage(working_sitk, temp_nifti_path)

    # 2. Reorientation (Enforce PIL)
    force_reorient_pil(temp_nifti_path)
    working_sitk = sitk.ReadImage(temp_nifti_path)

    # 3. Laterality classification and mirror flip if right knee
    is_flipped = False
    laterality = classify_laterality(file_path)

    if laterality is None:
        if laterality_callback:
            laterality = laterality_callback(os.path.basename(file_path))
        else:
            laterality = "Left"
            logging.warning(
                f"  -> Laterality unknown for '{os.path.basename(file_path)}', defaulting to Left."
            )

    logging.info(f"  -> Laterality: {laterality}")

    if laterality == "Right":
        logging.info("  -> Flipping right knee to match Left space (axis=0).")
        is_flipped = True
        arr = sitk.GetArrayFromImage(working_sitk)
        arr = np.flip(arr, axis=0)
        working_sitk = sitk.GetImageFromArray(arr)
        working_sitk.CopyInformation(sitk.ReadImage(temp_nifti_path))
        sitk.WriteImage(working_sitk, temp_nifti_path)

    # 4. Inference
    mask_arr = None
    if CONFIG["run_inference"]:
        if CONFIG.get("use_ensemble") and ensemble_models:
            logging.info("  -> Running ensemble inference (5-Fold).")
            mask_arr = infer_ensemble(temp_nifti_path, ensemble_models, device, CONFIG)
        elif model:
            logging.info("  -> Running single-model inference.")
            mask_arr = infer_model(temp_nifti_path, model, device, CONFIG)
        else:
            logging.warning("  -> Inference active but no model available. Skipping.")
    else:
        logging.info("  -> Inference module deactivated.")

    # 5. Post-processing and Inverse transforms
    if mask_arr is not None:
        logging.info("  -> Applying post-processing.")
        mask_arr = postprocess_mask(mask_arr, CONFIG["post_proc_classes"])

        if is_flipped:
            logging.info("  -> Flipping mask back to Right knee space.")
            mask_arr = np.flip(mask_arr, axis=0)

        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        meta_sitk = sitk.ReadImage(temp_nifti_path)
        mask_sitk.CopyInformation(meta_sitk)

        # Resample mask back to original space
        logging.info("  -> Resampling mask back to original input space.")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(orig_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        final_mask_sitk = resampler.Execute(mask_sitk)
        sitk.WriteImage(final_mask_sitk, final_mask_path)
        logging.info(f"  -> Final mask saved to: {final_mask_path}")

    # 6. Anatomical Analysis (Anaknee)
    if CONFIG["run_anatomical_analysis"]:
        logging.info("  -> Running Anaknee pipeline.")
        try:
            ref_path = CONFIG["anaknee_ref_mri"]
            res_dict, mask_array_ana, spacing_zyx, f_cent, t_cent, p_info = (
                run_analysis(file_path, ref_path, final_mask_path)
            )

            # The result is returned and saved into patient_results.csv by the caller.

            if run_viz_at_end:
                vis_data = {
                    "femoral_centroid": f_cent,
                    "tibial_centroid": t_cent,
                    "plateau_normal": p_info["normal"],
                    "plateau_center": p_info["center"],
                    "bh_grid_info": p_info.get("bh_grid_info", {}),
                    "att_info": p_info.get("att_info", {}),
                    "staubli_info": p_info.get("staubli_info", {}),
                    "plateau_inliers": p_info.get("plateau_inliers"),
                    "plateau_outliers": p_info.get("plateau_outliers"),
                    "results_dict": res_dict,
                }
                logging.info("  -> Launching PyVista visualization.")
                visualize_results(mask_array_ana, spacing_zyx, vis_data)

        except Exception as e:
            logging.error(f"Anaknee analysis failed: {e}")
            traceback.print_exc()

    # 7. Cleanup temp files
    if os.path.exists(temp_nifti_path):
        try:
            os.remove(temp_nifti_path)
            logging.info(f"  -> Removed temp file: {os.path.basename(temp_nifti_path)}")
        except Exception as e:
            logging.warning(f"  -> Could not remove temp file {temp_nifti_path}: {e}")
            
    if "res_dict" in locals():
        res_dict["Filename"] = os.path.basename(file_path)
        return res_dict
    return None


def _load_ensemble_models(config, device):
    """Loads fold checkpoints for ensemble inference."""
    fold_weight_paths = sorted(
        glob.glob(os.path.join(config["ensemble_dir"], config["ensemble_pattern"]))
    )

    if not fold_weight_paths:
        logging.error(f"Ensemble: No checkpoints found in '{config['ensemble_dir']}'.")
        return []

    logging.info(f"Ensemble: Found {len(fold_weight_paths)} models.")
    loaded_models = []
    for weight_path in fold_weight_paths:
        try:
            m = LightUNet3D(in_ch=1, out_ch=4, base=config["base_filters"])
            state = torch.load(weight_path, map_location=device)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            loaded_models.append(m)
            logging.info(f"  -> Loaded fold checkpoint: {os.path.basename(weight_path)}")
        except Exception as e:
            logging.error(f"  -> Could not load checkpoint {weight_path}: {e}")

    return loaded_models


def run_visualization_only(img_path=None, ref_path=None, mask_path=None):
    """
    Fast PyVista 3D viewer (no CSV logging, no slow radiomics).
    - If mask_path is provided: runs fast geometric reconstruction (bones, ACL, plateau, B&H, ATT).
    - If only img_path is provided: opens 3D MRI volume viewer (orthogonal slices).
    """
    from anaknee.visualizator_analyzator import smart_visualize, visualize_results, visualize_mri_volume
    try:
        if mask_path and os.path.exists(mask_path):
            logging.info(f"Preparing fast 3D anatomical visualization for: {os.path.basename(mask_path)}")
            from anaknee.main_acl_analysis import run_geometric_analysis_from_mask
            res_dict, mask_array_ana, spacing_zyx, f_cent, t_cent, p_info, vis_data = run_geometric_analysis_from_mask(
                mask_path
            )
            visualize_results(mask_array_ana, spacing_zyx, vis_data)
        elif img_path and os.path.exists(img_path):
            logging.info(f"Opening 3D volume viewer for MRI: {os.path.basename(img_path)}")
            visualize_mri_volume(img_path)
        else:
            logging.error("run_visualization_only: No valid mask or image path provided.")
    except Exception as e:
        logging.error(f"Could not open 3D visualization: {e}")
        traceback.print_exc()


def main():
    setup_logging(CONFIG["output_dir"], CONFIG["log_file"])
    logging.info("==== START PIPELINE RUN ====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = None
    ensemble_models = []

    if CONFIG["run_inference"]:
        if CONFIG.get("use_ensemble", False):
            logging.info("Ensemble mode: loading checkpoints...")
            ensemble_models = _load_ensemble_models(CONFIG, device)
            if not ensemble_models:
                logging.error("Ensemble load failed. Falling back to single model.")
                CONFIG["use_ensemble"] = False

        if not CONFIG.get("use_ensemble", False):
            try:
                model = LightUNet3D(in_ch=1, out_ch=4, base=CONFIG["base_filters"])
                if os.path.exists(CONFIG["model_ckpt"]):
                    model.load_state_dict(
                        torch.load(CONFIG["model_ckpt"], map_location=device)
                    )
                    model.to(device)
                    logging.info(f"Loaded single model checkpoint: {CONFIG['model_ckpt']}")
                else:
                    logging.warning(f"Checkpoint not found at: {CONFIG['model_ckpt']}. Skipping inference.")
                    model = None
            except Exception as e:
                logging.error(f"Could not load single model: {e}")
                model = None

    if CONFIG["mode"] == "FILE":
        file_path = CONFIG["input_path"]
        try:
            process_single_volume(
                file_path,
                model,
                device,
                run_viz_at_end=True,
                ensemble_models=ensemble_models,
            )
        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            traceback.print_exc()

    elif CONFIG["mode"] == "FOLDER":
        search_path = os.path.join(CONFIG["input_dir"], "*.nii*")
        files = glob.glob(search_path)
        logging.info(f"Folder mode found {len(files)} files.")

        for f in files:
            try:
                process_single_volume(
                    f,
                    model,
                    device,
                    run_viz_at_end=False,
                    ensemble_models=ensemble_models,
                )
            except Exception as e:
                logging.error(f"Pipeline error on file {f}: {e}")
                traceback.print_exc()

        if CONFIG["run_segmentation_analysis"]:
            perform_segmentation_analysis(CONFIG["output_dir"], CONFIG["gt_masks_dir"])


if __name__ == "__main__":
    main()
