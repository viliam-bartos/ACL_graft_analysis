import os
import SimpleITK as sitk
import logging
import glob

# --- 1. ZMĚŇ TYTO CESTY ---

# VSTUP: Cesta k tvým původním datům (nad složkou 'pacient_01')
SOURCE_BASE_DIR = r"C:\DIPLOM_PRACE\ACL_segment\Data"

# VÝSTUP: Cesta k nnUnet 'raw' složce (musí odpovídat tvé proměnné prostředí)
NNUNET_RAW_DIR = r"C:\nnUNet_v2\nnUNet_raw"
DATASET_NAME = "Dataset001_ACL"  # Název datasetu pro nnUnet

# Hledaný popis DICOM série (z tvého skriptu, můžeme použít k ověření)
# Pokud je název složky vždy 'pd_tse_fs_sag_DRB', není to ani nutné,
# ale ponechme to pro robustnost.
TARGET_SERIES_DESCRIPTION = "pd_tse_fs_sag_DRB"

# --- Konec nastavení ---

# Nastavení logování
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cesty pro nnUnet
dataset_dir = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)
imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
labelsTr_dir = os.path.join(dataset_dir, "labelsTr")


def create_nnunet_dirs():
    """Vytvoří základní adresářovou strukturu pro nnUnet."""
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    logging.info(f"Vytvořeny/ověřeny adresáře v: {dataset_dir}")


def reorient_to_lps(image: sitk.Image) -> sitk.Image:
    """Reorientuje obraz na LPS (standard pro nnUnet)."""
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPS")
    return orient_filter.Execute(image)


def convert_dicom_series_to_nifti(dicom_dir: str) -> sitk.Image | None:
    """Načte DICOM sérii ze složky, reorientuje a vrátí jako sitk.Image."""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            logging.warning(f"Ve složce {dicom_dir} nebyly nalezeny žádné DICOM soubory.")
            return None

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Ošetření 4D dat (pokud by DICOM série byla 4D)
        if image.GetDimension() == 4:
            logging.warning(f"Obraz v {dicom_dir} je 4D, extrahuji první 3D volume.")
            size = list(image.GetSize());
            size[3] = 0;
            index = [0, 0, 0, 0]
            image = sitk.Extract(image, size, index)

        image_reoriented = reorient_to_lps(image)
        return image_reoriented

    except Exception as e:
        logging.error(f"Chyba při konverzi DICOM série z {dicom_dir}: {e}")
        return None


def find_and_resample_mask(dicom_dir: str, reference_image: sitk.Image) -> sitk.Image | None:
    """
    Najde masku (.nii.gz) ve STEJNÉ složce jako DICOMy a resampluje ji.
    """
    try:
        # Najdeme jakýkoliv .nii.gz soubor v daném adresáři
        mask_paths = glob.glob(os.path.join(dicom_dir, "*.nii.gz"))

        if not mask_paths:
            logging.warning(f"Nenalezena žádná .nii.gz maska ve složce: {dicom_dir}")
            return None

        if len(mask_paths) > 1:
            logging.warning(f"Nalezeno více .nii.gz masek v {dicom_dir}, používám první: {mask_paths[0]}")

        mask_path = mask_paths[0]
        logging.info(f"  Nalezena maska: {os.path.basename(mask_path)}")

        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)  # Načíst jako integer
        mask_reoriented = reorient_to_lps(mask)

        # Nastavení resampleru
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)  # Zarovnat na NIfTI obraz
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nutné pro masky!

        mask_resampled = resampler.Execute(mask_reoriented)
        return mask_resampled

    except Exception as e:
        logging.error(f"Chyba při resamplování masky {mask_path}: {e}")
        return None


# --- Hlavní skript ---
if __name__ == "__main__":
    create_nnunet_dirs()
    logging.info(f"Prohledávám zdrojová data v: {SOURCE_BASE_DIR}")

    patient_folders = [d for d in os.listdir(SOURCE_BASE_DIR)
                       if os.path.isdir(os.path.join(SOURCE_BASE_DIR, d)) and d.startswith("pacient_")]

    if not patient_folders:
        logging.warning("Nenalezeny žádné složky 'pacient_XX' ve zdrojovém adresáři.")

    total_cases = 0
    processed_cases = 0
    training_file_count = 0  # Počítadlo pro dataset.json

    for patient_folder in sorted(patient_folders):  # pacient_01, pacient_02...
        patient_path = os.path.join(SOURCE_BASE_DIR, patient_folder)

        condition_folders = [d for d in os.listdir(patient_path)
                             if os.path.isdir(os.path.join(patient_path, d))]  # zdrave, po_rekonstrukci

        for condition in condition_folders:
            condition_path = os.path.join(patient_path, condition)
            series_folders = [d for d in os.listdir(condition_path)
                              if os.path.isdir(os.path.join(condition_path, d))]

            for series_folder in series_folders:
                # Ověření, zda název složky odpovídá hledané sérii
                # Můžeš to zakomentovat, pokud je tam vždy jen jedna relevantní složka
                if TARGET_SERIES_DESCRIPTION not in series_folder:
                    logging.debug(f"Přeskakuji složku (neodpovídá popisu): {series_folder}")
                    continue

                dicom_dir_path = os.path.join(condition_path, series_folder)

                # 1. Identifikátor případu
                case_id = f"{patient_folder}_{condition}"  # např. "pacient_01_zdrave"
                total_cases += 1
                logging.info(f"--- Zpracovávám případ: {case_id} ---")
                logging.info(f"  Zdroj DICOM: {dicom_dir_path}")

                # 2. Zpracování obrazu (DICOM -> NIfTI)
                reference_image = convert_dicom_series_to_nifti(dicom_dir_path)

                if reference_image is None:
                    logging.error(f"Přeskakuji případ {case_id} kvůli chybě při konverzi obrazu.")
                    continue

                # 3. Zpracování masky (Najít -> Resamplovat)
                mask_image = find_and_resample_mask(dicom_dir_path, reference_image)

                if mask_image is None:
                    logging.error(f"Přeskakuji případ {case_id}, nebyla nalezena nebo zpracována maska.")
                    continue

                # 4. Uložení obou souborů
                try:
                    image_output_path = os.path.join(imagesTr_dir, f"{case_id}_0000.nii.gz")
                    sitk.WriteImage(reference_image, image_output_path)

                    label_output_path = os.path.join(labelsTr_dir, f"{case_id}.nii.gz")
                    sitk.WriteImage(mask_image, label_output_path)

                    logging.info(f"Úspěšně uloženo: {case_id}")
                    processed_cases += 1
                    training_file_count += 1
                except Exception as e:
                    logging.error(f"Chyba při ukládání souborů pro {case_id}: {e}")

    logging.info("--- Dokončeno ---")
    logging.info(f"Celkem nalezeno pokusů o zpracování: {total_cases}")
    logging.info(f"Úspěšně zpracováno (obraz + maska): {processed_cases}")
    logging.info(f"Celkový počet souborů v 'imagesTr' bude: {training_file_count}")

    if total_cases != processed_cases:
        logging.warning("Některé případy selhaly. Zkontrolujte log výše.")

    print(f"\nNezapomeň vytvořit 'dataset.json' s hodnotou \"numTraining\": {training_file_count}")