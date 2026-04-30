import os
import sys
import glob
import logging
import traceback
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
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# -------------------------------------------------------------------------
# OŠETŘENÍ IMPORTŮ MEZI SLOŽKAMI (přidání do sys.path)
# -------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))
    sys.path.append(str(SOURCE_DIR / "kanonizace"))
    sys.path.append(str(SOURCE_DIR / "blackwell"))
    sys.path.append(str(SOURCE_DIR / "anaknee"))

from predict_laterality import LateralityClassifier
from WORKSTATION_BLACKWELL_MULTICLASS_5CV import LightUNet3D
from main_acl_analysis import run_analysis
from visualizator_analyzator import visualize_results

# -------------------------------------------------------------------------
# CENTRÁLNÍ KONFIGURACE (VŠECHNY PARAMETRY ZDE)
# -------------------------------------------------------------------------
CONFIG = {
    # ZÁKLADNÍ MÓD A CESTY
    "mode": "FOLDER",  # "FILE" nebo "FOLDER"
    "input_path": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\images_ASR", # Použije se pro FILE
    "input_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\images_ASR",                      # Použije se pro FOLDER
    "output_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\labels_optuna",
    "log_file": "pipeline.log", # Vytvoří se uvnitř output_dir
    
    # ANAKNEE
    "anaknee_ref_mri": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images\case_074.nii.gz",
    
    # GROUND TRUTH A ANALÝZA METRIK
    "gt_masks_dir": r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\test\labels",
    
    # MODEL SÍŤ (Blackwell a Kanonizace)
    "model_ckpt": r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\Optuna_best_model_150ep\best_model_trial_1.pth", 
    "kanonizace_ckpt": r"C:\DIPLOM_PRACE\ACL_segment\kanonizace\checkpoints\best_laterality_model.pth",
    "patch_size": (128, 128, 80),
    "base_filters": 64,
    
    # ENSEMBLE INFERENCE (5-Fold Cross-Validation)
    # Pokud je True, načtou se váhy ze všech foldů a pravděpodobnosti se průměrují.
    # Pokud je False, použije se pouze model_ckpt (single model).
    "use_ensemble": True,
    "ensemble_dir": r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\5CV",
    "ensemble_pattern": "best_model_fold_*.pth",  # glob vzor pro vyhledávání vah foldů
    
    # PŘEPÍNAČE MODULŮ (Zapínat/Vypínat dle potřeby)
    "run_resampling": True,
    "run_orientation": True,
    "run_canonization": True,
    "run_inference": True,
    "run_postprocessing": True,
    "run_inverse_transform": True,
    "run_segmentation_analysis": False,
    "run_anatomical_analysis": False,
    
    # POST-PROCESSING TŘÍDY
    # 1: ACL, 2: Femur, 3: Tibia
    "post_proc_classes": {
        1: {"lcc": True, "hole_filling": False, "closing": False},
        2: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2},
        3: {"lcc": True, "hole_filling": True, "closing": True, "closing_kernel": 2}
    }
}

# -------------------------------------------------------------------------
# FUNKCE MODULŮ
# -------------------------------------------------------------------------
def setup_logging(output_dir, log_file_name):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file_name)
    
    # Odebereme staré handlery při opakovaném spuštění
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

def resample_image_sitk(sitk_img, target_spacing=(0.5, 0.5, 0.5)):
    """ Resampling do fyzického rozlišení """
    original_spacing = sitk_img.GetSpacing()
    if np.allclose(original_spacing, target_spacing, atol=1e-3):
        logging.info("  -> Spacing je správný, vynechávám resample.")
        return sitk_img
    
    logging.info(f"  -> Provádím resample z {original_spacing} na {target_spacing}")
    orig_size = np.array(sitk_img.GetSize(), dtype=int)
    new_size = np.round(orig_size * (np.array(original_spacing) / np.array(target_spacing))).astype(int)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_img.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline) # plynulý b-spline pro raw data

    return resample.Execute(sitk_img)

def force_reorient_pil(nifti_path):
    """ ASR kontrola a tranformace přes nibabel.orientations """
    img = nib.load(nifti_path)
    target_ornt = nio.axcodes2ornt("PIL")
    orig_ornt = nio.io_orientation(img.affine)
    transform = nio.ornt_transform(orig_ornt, target_ornt)
    
    if not np.array_equal(transform, [[0, 1], [1, 1], [2, 1]]):
        logging.info("  -> Orientace není ASR (PIL), aplikuji transformaci.")
        new_img = img.as_reoriented(transform)
        nib.save(new_img, nifti_path)
    else:
        logging.info("  -> Orientace je správně ASR, bez měnění.")

def postprocess_mask(mask_array, config_classes):
    """ Modulární post-processing v rámci tříd na zadaném numpy poli """
    output_mask = np.zeros_like(mask_array)
    unique_labels = np.unique(mask_array)
    
    for lbl in unique_labels:
        if lbl == 0:
            continue
        
        lbl_mask = (mask_array == lbl)
        
        if lbl in config_classes:
            cfg = config_classes[lbl]
            
            # Hole filling
            if cfg.get("hole_filling", False):
                lbl_mask = ndimage.binary_fill_holes(lbl_mask)
                
            # Closing
            if cfg.get("closing", False):
                k_size = cfg.get("closing_kernel", 2)
                struct = ndimage.generate_binary_structure(3, 1)
                if k_size > 1:
                    struct = ndimage.iterate_structure(struct, k_size)
                lbl_mask = ndimage.binary_closing(lbl_mask, structure=struct)
                
            # Largest Connected Component
            if cfg.get("lcc", False):
                labeled, num_features = ndimage.label(lbl_mask)
                if num_features > 0:
                    sizes = ndimage.sum(lbl_mask, labeled, range(1, num_features + 1))
                    largest_idx = np.argmax(sizes) + 1
                    lbl_mask = (labeled == largest_idx)
                    
        output_mask[lbl_mask] = lbl
        
    return output_mask

def _preprocess_image(img_path):
    """ Společné předzpracování obrazu pro single i ensemble inference. """
    sitk_img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    
    # Transpozice z (Z, Y, X) do (X, Y, Z) pro model
    img_array = np.transpose(img_array, (2, 1, 0))
    
    # Skálování intenzit (MONAI ScaleIntensityRangePercentilesd logika)
    p05 = np.percentile(img_array, 0.5)
    p995 = np.percentile(img_array, 99.5)
    img_array = np.clip(img_array, p05, p995)
    img_array = (img_array - p05) / (p995 - p05 + 1e-8)
    
    # Standardize (NormalizeIntensityd channel wise na non-zero)
    non_zero = img_array > 0
    if np.any(non_zero):
        img_array[non_zero] = (img_array[non_zero] - img_array[non_zero].mean()) / (img_array[non_zero].std() + 1e-8)
    
    return img_array


def _apply_thresholds(probs):
    """
    Aplikace uživatelských prahů na tenzor pravděpodobností.
    probs: torch.Tensor tvaru [4, X, Y, Z] (softmax pravděpodobnosti)
    Vrací: torch.Tensor tvaru [X, Y, Z] s hodnotami 0–3.
    """
    pred_argmax = torch.argmax(probs, dim=0)  # [X, Y, Z]
    pred = torch.zeros_like(pred_argmax)
    pred[(pred_argmax == 1) & (probs[1] >= 0.45)] = 1  # ACL
    pred[(pred_argmax == 2) & (probs[2] >= 0.90)] = 2  # Femur
    pred[(pred_argmax == 3) & (probs[3] >= 0.80)] = 3  # Tibia
    return pred


def infer_model(img_path, model, device, config):
    """ Single-model inference sekce s monai sliding window. """
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
                mode='gaussian'
            )
        probs = torch.softmax(outputs, dim=1).squeeze(0)  # [4, X, Y, Z]
        pred = _apply_thresholds(probs)
        
    pred_np = pred.cpu().numpy()
    # Transpozice zpět z (X, Y, Z) do (Z, Y, X) pro uložení přes SimpleITK
    pred_np = np.transpose(pred_np, (2, 1, 0))
    return pred_np


def infer_ensemble(img_path, ensemble_models, device, config):
    """
    Ensemble inference: průměrování softmax pravděpodobností ze všech fold modelů.
    
    Každý model zpracuje celý objem přes sliding window inference.
    Výsledné pravděpodobnostní mapy se průměrují (simple average ensemble).
    Thresholdy se aplikují jednou na průměrné pravděpodobnosti.
    
    Args:
        img_path: cesta k vstupnímu NIfTI souboru
        ensemble_models: list načtených PyTorch modelů (jeden per fold)
        device: torch.device
        config: CONFIG slovník s patch_size a thresholdovými hodnotami
    
    Returns:
        numpy array tvaru (Z, Y, X) s predikovanou maskou tříd 0–3
    """
    logging.info(f"  -> Ensemble inference s {len(ensemble_models)} fold modely.")
    img_array = _preprocess_image(img_path)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    accumulated_probs = None  # torch.Tensor [4, X, Y, Z]
    
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
                    mode='gaussian'
                )
            fold_probs = torch.softmax(outputs, dim=1).squeeze(0)  # [4, X, Y, Z]
            logging.info(f"     Fold {fold_idx + 1}/{len(ensemble_models)} hotov.")
            
            if accumulated_probs is None:
                accumulated_probs = fold_probs.clone()
            else:
                accumulated_probs += fold_probs
    
    # Průměrné pravděpodobnosti ze všech foldů
    avg_probs = accumulated_probs / len(ensemble_models)  # [4, X, Y, Z]
    
    # Aplikace thresholdů na průměrné pravděpodobnosti
    pred = _apply_thresholds(avg_probs)
    
    pred_np = pred.cpu().numpy()
    # Transpozice zpět z (X, Y, Z) do (Z, Y, X) pro uložení přes SimpleITK
    pred_np = np.transpose(pred_np, (2, 1, 0))
    return pred_np

def perform_segmentation_analysis(output_dir, gt_dir):
    """ Výpočet Dice/HD95 pomocí monai """
    logging.info("--- Spouštím Segmentační Analýzu ---")
    
    results = []
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    post_func = AsDiscrete(to_onehot=4)
    
    pred_files = glob.glob(os.path.join(output_dir, "mask_*.nii.gz"))
    if not pred_files:
        logging.warning("Neočekávaně nenalezeny žádné výstupní masky. Segmentační analýza přeskočena.")
        return
        
    for p_path in pred_files:
        basename = os.path.basename(p_path)
        # GT masky mají prefix 'mask_'
        gt_path = os.path.join(gt_dir, basename)
        
        if not os.path.exists(gt_path):
            logging.warning(f"GT maska nenalezena pro: {basename}")
            continue
            
        logging.info(f"Srovnávám GT test vs. Pred pro: {basename}")
        
        p_sitk = sitk.ReadImage(p_path)
        g_sitk = sitk.ReadImage(gt_path)
        
        p_arr = sitk.GetArrayFromImage(p_sitk)
        g_arr = sitk.GetArrayFromImage(g_sitk)
        
        try:
            # Převedení do OHE a přidání dimenzí Batche a Channelu
            p_t = post_func(torch.from_numpy(p_arr).unsqueeze(0))
            g_t = post_func(torch.from_numpy(g_arr).unsqueeze(0))
            
            # Tyto metriky vyžadují tvar: Množství Batch tensors jako List
            dice_metric(y_pred=[p_t], y=[g_t])
            hd95_metric(y_pred=[p_t], y=[g_t])
            
            dice = dice_metric.get_buffer()[-1]     
            hd95 = hd95_metric.get_buffer()[-1]
            
            for class_idx, class_name in enumerate(["ACL", "Femur", "Tibia"]):
                d_val = dice[class_idx].item() if not torch.isnan(dice[class_idx]) else 0.0
                try: h_val = hd95[class_idx].item()
                except: h_val = float('nan')
                
                results.append({
                    "Soubor": basename,
                    "Struktura": class_name,
                    "Dice": d_val,
                    "HD95 [mm]": h_val
                })
        except Exception as e:
            logging.error(f"Chyba při verifikaci {basename}: {e}")
            
    if not results:
        return
        
    # Uložení reportu
    df = pd.DataFrame(results)
    stats_dir = os.path.join(output_dir, "Segmentation_Reports")
    os.makedirs(stats_dir, exist_ok=True)
    
    csv_path = os.path.join(stats_dir, "segmentation_metrics.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Metriky uloženy do: {csv_path}")
    
    # Grafy
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Struktura", y="Dice", palette="tab10")
    plt.title("Dice Skóre", fontweight="bold")
    plt.ylim(0, 1.05)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="Struktura", y="HD95 [mm]", palette="tab10")
    plt.title("Hausdorff 95%", fontweight="bold")
    plt.yscale("log")
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "metrics_boxplots.png"), dpi=200)
    plt.savefig(os.path.join(stats_dir, "metrics_boxplots.pdf"))
    plt.close()


# -------------------------------------------------------------------------
# HLAVNÍ ENGINE
# -------------------------------------------------------------------------
def process_single_volume(file_path, lat_classifier, model, device, run_viz_at_end=False, ensemble_models=None):
    logging.info(f"====== ZAČÁTEK ZPRACOVÁNÍ: {os.path.basename(file_path)} ======")
    
    # Kontrola existence a načtení
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vstupní soubor chybí: {file_path}")
        
    orig_sitk = sitk.ReadImage(file_path)
    
    # Založíme si dočasný NIfTI pro mezikroky na disku (vyžadováno např. anaknee)
    temp_nifti_path = os.path.join(CONFIG["output_dir"], f"process_raw_{os.path.basename(file_path)}")
    # GT test filename matching expects 'mask_case_123.nii.gz' where case is original file name. 
    # Example input file target looks like `case_123.nii.gz` -> `mask_case_123.nii.gz`.
    # Wait, the user said GT masks have prefix mask_ -> mask_case_ID. If output is already mask_case_ID...
    final_basename = os.path.basename(file_path)
    if not final_basename.startswith("mask_"):
        final_basename = f"mask_{final_basename}"
    final_mask_path = os.path.join(CONFIG["output_dir"], final_basename)
    
    working_sitk = orig_sitk
    
    # 1. RESAMPLING
    if CONFIG["run_resampling"]:
        working_sitk = resample_image_sitk(working_sitk, target_spacing=(0.5, 0.5, 0.5))
        
    sitk.WriteImage(working_sitk, temp_nifti_path)
    
    # 2. ORIENTACE (in-place přes zápis)
    if CONFIG["run_orientation"]:
        force_reorient_pil(temp_nifti_path)
        # Nyní aktualizujeme wokring_sitk s čerstvými orientacemi z disku
        working_sitk = sitk.ReadImage(temp_nifti_path)
        
    # 3. KANONIZACE / LATERALITA
    is_flipped = False
    if CONFIG["run_canonization"] and lat_classifier:
        laterality = lat_classifier.predict(temp_nifti_path)
        logging.info(f"  -> Laterality predikována jako: {laterality}")
        
        if laterality == "Right":
            logging.info("  -> Provádím zrcadlení pro Inference do LEVÉHO uspořádání (axis=0).")
            is_flipped = True
            arr = sitk.GetArrayFromImage(working_sitk)
            arr = np.flip(arr, axis=0) # Přesný mechanizmus ze skriptu extrakce
            working_sitk = sitk.GetImageFromArray(arr)
            working_sitk.CopyInformation(sitk.ReadImage(temp_nifti_path)) # Zachová orig meta
            sitk.WriteImage(working_sitk, temp_nifti_path)
            
    # 4. INFERENCE (single model nebo ensemble)
    mask_arr = None
    if CONFIG["run_inference"]:
        if CONFIG.get("use_ensemble") and ensemble_models:
            logging.info("  -> Spouštím ENSEMBLE multiclass inference (Blackwell 5-Fold).")
            mask_arr = infer_ensemble(temp_nifti_path, ensemble_models, device, CONFIG)
        elif model:
            logging.info("  -> Spouštím single-model multiclass inference (Blackwell).")
            mask_arr = infer_model(temp_nifti_path, model, device, CONFIG)
        else:
            logging.warning("  -> Inference je zapnuta, ale není k dispozici žádný model ani ensemble. Přeskakuji.")
    else:
        logging.info("  -> Modul inference deaktivován, maska bude ignorována.")
        
    # 5/6. POSTPROCESSING A INVERZNÍ TRANSFORMACE
    if mask_arr is not None:
        if CONFIG["run_postprocessing"]:
            logging.info("  -> Vykonávám zadaný post-processing.")
            mask_arr = postprocess_mask(mask_arr, CONFIG["post_proc_classes"])
            
        if is_flipped and CONFIG["run_inverse_transform"]:
             logging.info("  -> Inverzní transformace zrcadlení masky zpět na Pravou.")
             mask_arr = np.flip(mask_arr, axis=0)
             
        # Uložení masky (do spacingu a orientace po převedení)
        # Načteme fyzicky uložený originální reorientovaný file PŘED canonizací (pokud se zrcadlilo, použijeme re-written spacing. Zrcadlení přes SimpleITK nezměnilo metadata).
        # Takže do pracovního stavu natlačíme hrubou masku
        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        
        # Orientace a Spacing pochází z NIfTI souboru těsně před "Inverzní kanonizací".
        # Ale pokud chceme resamplovat do ORIGINÁLNÍHO OBJEMU (před resamplováním a před kanonizací), musíme resamplovat k `orig_sitk`
        if CONFIG["run_inverse_transform"]:
            logging.info("  -> Zpětný resampling masky do prostoru a rozlišení původních vstupních dat.")
            
            # Nejdřív namapovat fyzické informace zpracovaného obrazu (Z bodu 2/3 bez FLIPU)
            # Trik: Pokud se dělal flip u raw dat, temp_nifti_path má teď flip. Ale metadata to neovlivnilo orientačně,
            # SimpleITK má direction a origin stejný i po np.flip(., axis=0). Zpětný flip v array stačí.  
            
            meta_sitk = sitk.ReadImage(temp_nifti_path)
            mask_sitk.CopyInformation(meta_sitk)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(orig_sitk)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # maska = NN
            resampler.SetDefaultPixelValue(0)
            
            final_mask_sitk = resampler.Execute(mask_sitk)
            sitk.WriteImage(final_mask_sitk, final_mask_path)
        else:
            meta_sitk = sitk.ReadImage(temp_nifti_path)
            mask_sitk.CopyInformation(meta_sitk)
            sitk.WriteImage(mask_sitk, final_mask_path)
            
        logging.info(f"  -> Výstupní maska uložena do: {final_mask_path}")    
        
    # 7. ANATOMICKÁ ANALÝZA (Anaknee)
    if CONFIG["run_anatomical_analysis"]:
        logging.info("  -> Spouštím Anaknee pipeline.")
        
        try:
            # Original raw (without resampling!) + corresponding original space mask
            ref_path = CONFIG["anaknee_ref_mri"]
            res_dict, mask_array_ana, spacing_zyx, f_cent, t_cent, p_info = run_analysis(
                file_path, ref_path, final_mask_path
            )
            
            # CSV appending
            df = pd.DataFrame([res_dict])
            csv_path = os.path.join(CONFIG["output_dir"], "anaknee_results.csv")
            header = not os.path.exists(csv_path)
            df.to_csv(csv_path, mode='a', header=header, index=False)
            
            # Pokud voláme jeden soubor, nebo je vynuceno, pustíme pyvistu (pozor, blokující prvek)
            if run_viz_at_end:
                 vis_data = {
                    'femoral_centroid': f_cent,
                    'tibial_centroid': t_cent,
                    'plateau_normal': p_info['normal'],
                    'plateau_center': p_info['center'],
                    'bh_grid_info': p_info.get('bh_grid_info', {}),
                    'att_info': p_info.get('att_info', {}),
                    'staubli_info': p_info.get('staubli_info', {})
                 }
                 logging.info("  -> Vyvolávám grafické okno PyVista.")
                 visualize_results(mask_array_ana, spacing_zyx, vis_data)
                 
        except Exception as e:
             logging.error(f"Při Anaknee analýze došlo k problému: {e}")
             traceback.print_exc()

    # 8. ÚKLID DOČASNÝCH SOUBORŮ
    if os.path.exists(temp_nifti_path):
        try:
            os.remove(temp_nifti_path)
            logging.info(f"  -> Smazán dočasný soubor: {os.path.basename(temp_nifti_path)}")
        except Exception as e:
            logging.warning(f"  -> Nepodařilo se smazat dočasný soubor {temp_nifti_path}: {e}")

def _load_ensemble_models(config, device):
    """
    Načte všechny fold modely ze složky ensemble_dir podle vzoru ensemble_pattern.
    Vrací list modelů připravených k inference (eval mode, na device).
    """
    fold_weight_paths = sorted(glob.glob(
        os.path.join(config["ensemble_dir"], config["ensemble_pattern"])
    ))
    
    if not fold_weight_paths:
        logging.error(f"Ensemble: nenalezeny žádné váhy ve složce '{config['ensemble_dir']}' "
                      f"se vzorem '{config['ensemble_pattern']}'.")
        return []
    
    logging.info(f"Ensemble: nalezeno {len(fold_weight_paths)} foldů:")
    for p in fold_weight_paths:
        logging.info(f"  - {os.path.basename(p)}")
    
    loaded_models = []
    for weight_path in fold_weight_paths:
        try:
            m = LightUNet3D(in_ch=1, out_ch=4, base=config["base_filters"])
            state = torch.load(weight_path, map_location=device)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            loaded_models.append(m)
            logging.info(f"  -> Načten fold model: {os.path.basename(weight_path)}")
        except Exception as e:
            logging.error(f"  -> Nelze načíst fold model {weight_path}: {e}")
    
    return loaded_models


def main():
    setup_logging(CONFIG["output_dir"], CONFIG["log_file"])
    logging.info("==== SPOUŠTÍM PIPELINE VYSOCE AUTOMATIZOVANÉHO ZPRACOVÁNÍ ====")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Používám zařízení: {device}")
    
    lat_classifier = None
    if CONFIG["run_canonization"]:
        try:
            lat_classifier = LateralityClassifier(model_path=CONFIG["kanonizace_ckpt"], device=device)
        except Exception as e:
            logging.error(f"LateralityClassifier nelze inicializovat: {e}")
    
    # --- Inicializace modelů ---
    model = None           # Single model (fallback)
    ensemble_models = []   # Ensemble modelů (5 foldů)
    
    if CONFIG["run_inference"]:
        if CONFIG.get("use_ensemble", False):
            logging.info("Ensemble mód: načítám váhy ze všech foldů...")
            ensemble_models = _load_ensemble_models(CONFIG, device)
            if not ensemble_models:
                logging.error("Ensemble selhal – žádné modely nebyly načteny. Zkouším single model jako zálohu.")
                CONFIG["use_ensemble"] = False  # Automatický fallback
        
        # Single model jako záloha (nebo primář, pokud ensemble je vypnut)
        if not CONFIG.get("use_ensemble", False):
            try:
                model = LightUNet3D(in_ch=1, out_ch=4, base=CONFIG["base_filters"])
                if os.path.exists(CONFIG["model_ckpt"]):
                    model.load_state_dict(torch.load(CONFIG["model_ckpt"], map_location=device))
                    model.to(device)
                    logging.info(f"Načten single model Blackwell z: {CONFIG['model_ckpt']}")
                else:
                    logging.warning(f"Váhy nenalezeny v {CONFIG['model_ckpt']} – inference bude přeskočena.")
                    model = None
            except Exception as e:
                logging.error(f"Single model Blackwell nelze inicializovat/načíst: {e}")
                model = None

    if CONFIG["mode"] == "FILE":
        file_path = CONFIG["input_path"]
        try:
            process_single_volume(
                file_path, lat_classifier, model, device,
                run_viz_at_end=True,
                ensemble_models=ensemble_models
            )
        except Exception as e:
            logging.error(f"Fatální chyba při zpracování: {e}")
            traceback.print_exc()
            
    elif CONFIG["mode"] == "FOLDER":
        search_path = os.path.join(CONFIG["input_dir"], "*.nii*")
        files = glob.glob(search_path)
        logging.info(f"Mód Složky nalezen s počtem souborů: {len(files)}")
        
        for f in files:
            try:
                process_single_volume(
                    f, lat_classifier, model, device,
                    run_viz_at_end=False,
                    ensemble_models=ensemble_models
                )
            except Exception as e:
                logging.error(f"Chyba ve složkovém módu pro soubor {f}: {e}")
                logging.info("Pokračujeme k dalšímu pacientovi...")
                traceback.print_exc()
        
        # Segmentační analýza se pro složkové řešení pouští nakonec pro celou batch
        if CONFIG["run_segmentation_analysis"]:
            perform_segmentation_analysis(CONFIG["output_dir"], CONFIG["gt_masks_dir"])

if __name__ == "__main__":
    main()
