import os
import glob
import logging
from tqdm import tqdm
import optuna
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

# -------------------------------------------------------------------------
# KONFIGURACE (Upravitelné pro uživatele)
# -------------------------------------------------------------------------
CONFIG = {
    # Složka obsahující surové inference masky (bez post-processingu)
    "raw_masks_dir": r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\Raw_Masks",
    
    # Složka obsahující skutečné anotace (Ground Truth)
    "gt_masks_dir": r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\GT_masks",
    
    # KAM UKLÁDAT VÝSLEDKY
    "output_dir": r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\Optuna_Results",
    
    # TYP OPTIMALIZACE: 
    # "DICE" (hledá max Dice), 
    # "HD95" (hledá min HD95 vzdálenost), 
    # "MULTI" (hledá kompromis obojího, vrací Pareto frontu)
    "OPTIMIZATION_TARGET": "MULTI", 
    
    # POČET POKUSŮ
    "n_trials": 50,
}

# -------------------------------------------------------------------------
# POST-PROCESSING LOGIKA (Identická s mri_pipeline.py)
# -------------------------------------------------------------------------
def apply_postprocessing(mask_array, pp_config):
    """ Post-processing na zadaném numpy poli řízený zkušebními parametry Optuny """
    output_mask = np.zeros_like(mask_array)
    unique_labels = np.unique(mask_array)
    
    for lbl in unique_labels:
        if lbl == 0:
            continue
            
        lbl_mask = (mask_array == lbl)
        
        if lbl in pp_config:
            cfg = pp_config[lbl]
            
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
                    lbl_mask = (labeled == largest_idx)
                    
        output_mask[lbl_mask] = lbl
        
    return output_mask

# -------------------------------------------------------------------------
# OPTIMALIZAČNÍ LOGIKA
# -------------------------------------------------------------------------
class Objective:
    def __init__(self, raw_files, gt_dir, target_mode):
        self.raw_files = raw_files
        self.gt_dir = gt_dir
        self.target_mode = target_mode
        self.post_func = AsDiscrete(to_onehot=4) # 3 třídy + pozadí
        
    def __call__(self, trial):
        # 1. NALOSOVÁNÍ PARAMETRŮ PRO POST-PROCESSING
        # ACL
        acl_lcc = trial.suggest_categorical("acl_lcc", [True, False])
        
        # FEMUR
        femur_fill = trial.suggest_categorical("femur_hole_filling", [True, False])
        femur_close = trial.suggest_categorical("femur_closing", [True, False])
        femur_k = trial.suggest_int("femur_kernel", 1, 4) if femur_close else 0
        femur_lcc = trial.suggest_categorical("femur_lcc", [True, False])
        
        # TIBIA
        tibia_fill = trial.suggest_categorical("tibia_hole_filling", [True, False])
        tibia_close = trial.suggest_categorical("tibia_closing", [True, False])
        tibia_k = trial.suggest_int("tibia_kernel", 1, 4) if tibia_close else 0
        tibia_lcc = trial.suggest_categorical("tibia_lcc", [True, False])
        
        # Slovník pro post-processing logiku
        pp_config = {
            1: {"lcc": acl_lcc, "hole_filling": False, "closing": False},
            2: {"lcc": femur_lcc, "hole_filling": femur_fill, "closing": femur_close, "closing_kernel": femur_k},
            3: {"lcc": tibia_lcc, "hole_filling": tibia_fill, "closing": tibia_close, "closing_kernel": tibia_k}
        }
        
        # 2. DEFINICE METRIK PRO VÝPOČET
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
        
        # 3. ZPRACOVACÍ SMYČKA PRO VŠECHNY MASKY PACIENTŮ (LAZY LOAD DO RAM)
        for raw_path in self.raw_files:
            basename = os.path.basename(raw_path)
            gt_path = os.path.join(self.gt_dir, basename)
            
            if not os.path.exists(gt_path):
                continue
                
            # Načítání a vytažení pole
            r_img = sitk.ReadImage(raw_path)
            g_img = sitk.ReadImage(gt_path)
            
            r_arr = sitk.GetArrayFromImage(r_img)
            g_arr = sitk.GetArrayFromImage(g_img)
            
            # Aplikace zkoušeného post-processingu
            p_arr = apply_postprocessing(r_arr, pp_config)
            
            # Odeslání do MONAI (OHE)
            p_t = self.post_func(torch.from_numpy(p_arr).unsqueeze(0))
            g_t = self.post_func(torch.from_numpy(g_arr).unsqueeze(0))
            
            dice_metric(y_pred=[p_t], y=[g_t])
            hd95_metric(y_pred=[p_t], y=[g_t])
            
        # 4. AGREGOVÁNÍ VÝSLEDKŮ (Průměr a Nan Záchrana)
        dice_agg = dice_metric.aggregate()
        hd95_agg = hd95_metric.aggregate()
        
        mean_dice = dice_agg.nanmean().item() if not torch.isnan(dice_agg).all() else 0.0
        
        mean_hd95 = hd95_agg.nanmean().item() if not torch.isnan(hd95_agg).all() else 100.0 # Trestní velká vzdálenost v mm pro selhání
        
        logging.info(f"Trial {trial.number}: MEAN_DICE: {mean_dice:.4f} | MEAN_HD95: {mean_hd95:.2f} mm")
        
        # Vrati hodnotu dle zvyklostí požadovaného targetu
        if self.target_mode == "DICE":
            return mean_dice
        elif self.target_mode == "HD95":
            return mean_hd95
        else: # "MULTI"
            return mean_dice, mean_hd95

# -------------------------------------------------------------------------
# HLAVNÍ SPOUŠTĚČ
# -------------------------------------------------------------------------
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # 1. Hledání masek
    search_path = os.path.join(CONFIG["raw_masks_dir"], "mask_*.nii.gz")
    raw_files = glob.glob(search_path)
    
    if not raw_files:
         logging.error(f"Nepodařilo se nalézt žádné raw masky na cestě: {search_path}")
         return
         
    num_gt = len(glob.glob(os.path.join(CONFIG["gt_masks_dir"], "mask_*.nii.gz")))
    logging.info(f"Nalezeno {len(raw_files)} surových masek, a {num_gt} Ground Truth masek.")
    
    # 2. Režimy Studie a směr hledání
    mode = CONFIG["OPTIMIZATION_TARGET"].upper()
    study_name = f"Postprocessing_Tuning_{mode}"
    
    if mode == "DICE":
         directions = ["maximize"]
    elif mode == "HD95":
         directions = ["minimize"]
    elif mode == "MULTI":
         directions = ["maximize", "minimize"] 
    else:
         logging.error("Neznámý parametr optimaizace v CONFIG!")
         return
         
    # 3. Založení studie Optuna
    logging.info(f"== ZAKLÁDÁM OPTUNA STUDII ({mode} režim, snaha o {directions}) ==")
    
    # Vytvoření studie. TpeSampler je inteligentní Bayesian sampler v Optuně (Pro MULTI režim použije MOTPE/NSGA-II)
    if mode == "MULTI":
        sampler = optuna.samplers.NSGAIISampler()
    else:
        sampler = optuna.samplers.TPESampler()
        
    study = optuna.create_study(
        study_name=study_name, 
        directions=directions, 
        sampler=sampler
    )
    
    # Odeslat rovnou k řešení
    objective = Objective(raw_files, CONFIG["gt_masks_dir"], mode)
    study.optimize(objective, n_trials=CONFIG["n_trials"], show_progress_bar=True)
    
    # 4. Reportování Výsledků
    logging.info(" === OPTIMALIZACE DOKONČENA ===")
    
    if mode == "MULTI":
        # Pro vícekriteriální optimalizaci vrací Pareto Frontu (těch pár nejlepších nedominovaných trialů)
        best_trials = study.best_trials
        logging.info(f"Nalezeno {len(best_trials)} nejlepších bodů tvořících Paretovu frontu:")
        res_list = []
        for t in best_trials:
            logging.info(f" -> Pokus #{t.number}: [DICE={t.values[0]:.4f}, HD95={t.values[1]:.2f} mm]")
            logging.info(f"    Parametry: {t.params}")
            d = t.params.copy()
            d["Trial"] = t.number
            d["O_DICE"] = t.values[0]
            d["O_HD95"] = t.values[1]
            res_list.append(d)
        
        df = pd.DataFrame(res_list)
        out_csv = os.path.join(CONFIG["output_dir"], f"best_paerto_front_{mode}.csv")
        df.to_csv(out_csv, index=False)
        logging.info(f"Zapsána Paretova tabulka do {out_csv}")
        
    else:
        best_trial = study.best_trial
        logging.info(f"Nejlepší hodnota: {best_trial.value}")
        logging.info(f"Nejlepší parametry:")
        for key, value in best_trial.params.items():
            logging.info(f"    {key}: {value}")
            
        df = study.trials_dataframe()
        out_csv = os.path.join(CONFIG["output_dir"], f"all_trials_{mode}.csv")
        df.to_csv(out_csv, index=False)
        logging.info(f"Všechny výsledky uloženy do {out_csv}")

if __name__ == "__main__":
    main()
