import os
import shutil
import pydicom
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
import logging

# --- Konfigurace ---
SOURCE_BASE_DIR = r"C:\DIPLOM_PRACE\ACL_segment\nova_data"
DEST_BASE_DIR = r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data" # Cílový adresář pro uspořádaná data
TARGET_SERIES_DESCRIPTION = "pd_space_sag_p4_iso" # Hledaný popis DICOM série

# Nastavení logování pro lepší přehled
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_dicom_series_paths(directory: str) -> dict:
    """
    Najde všechny DICOM série v daném adresáři a vrátí slovník.
    Klíč: Series Instance UID
    Hodnota: Seznam cest k souborům dané série
    """
    series_paths = {}
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            logging.warning(f"Nenalezeny žádné DICOM série v: {directory}")
            return {}

        reader = sitk.ImageSeriesReader()
        for series_id in series_ids:
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(directory, series_id)
                if dicom_names:
                    series_paths[series_id] = dicom_names
            except Exception as e:
                logging.error(f"Chyba při získávání souborů pro sérii {series_id} v {directory}: {e}")
    except RuntimeError as e:
        logging.error(f"Nelze přečíst série z adresáře (možná neobsahuje DICOM): {directory}. Chyba: {e}")
    except Exception as e:
         logging.error(f"Neočekávaná chyba při hledání sérií v {directory}: {e}")
    return series_paths

def get_series_description(dicom_file_path: str) -> str | None:
    """Přečte popis série z hlavičky DICOM souboru."""
    try:
        dicom_header = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
        return dicom_header.get("SeriesDescription", None)
    except InvalidDicomError:
        logging.warning(f"Soubor není validní DICOM: {dicom_file_path}")
        return None
    except Exception as e:
        logging.error(f"Chyba při čtení DICOM hlavičky {dicom_file_path}: {e}")
        return None

def process_patient_folder(patient_dir: str, dest_base: str, target_desc: str):
    """Zpracuje složku jednoho pacienta."""
    patient_name_parts = os.path.basename(patient_dir).split()
    if len(patient_name_parts) != 2 or not patient_name_parts[1].isdigit():
        logging.warning(f"Přeskakuji složku (neočekávaný formát názvu): {patient_dir}")
        return
    patient_id_str = f"pacient_{int(patient_name_parts[1]):02d}" # Např. pacient_01
    logging.info(f"Zpracovávám pacienta: {patient_id_str}")

    data_dir = os.path.join(patient_dir, "DATA")
    if not os.path.isdir(data_dir):
        logging.warning(f"Adresář DATA nenalezen pro pacienta: {patient_id_str}")
        return

    # Najdeme numerické složky uvnitř DATA
    numeric_subdirs = []
    try:
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            # Ověření, zda je to složka a zda název obsahuje tečky a čísla (typické pro DICOM UID)
            if os.path.isdir(item_path) and '.' in item and any(char.isdigit() for char in item):
                 numeric_subdirs.append(item_path)
    except OSError as e:
        logging.error(f"Chyba při čtení obsahu adresáře {data_dir}: {e}")
        return

    # Předpoklad: první je zdravé, druhá po rekonstrukci
    # POZNÁMKA: Pokud toto pořadí není konzistentní, je potřeba jiný způsob identifikace!
    conditions = {}
    if len(numeric_subdirs) >= 1:
        conditions['zdrave'] = numeric_subdirs[0]
        logging.info(f"  Nalezena složka pro 'zdrave': {os.path.basename(numeric_subdirs[0])}")
    if len(numeric_subdirs) >= 2:
        conditions['po_rekonstrukci'] = numeric_subdirs[1]
        logging.info(f"  Nalezena složka pro 'po_rekonstrukci': {os.path.basename(numeric_subdirs[1])}")
    if len(numeric_subdirs) > 2:
         logging.warning(f"  Nalezeno více než 2 numerické složky v {data_dir}. Používám první dvě.")
    if not conditions:
        logging.warning(f"  Nenalezeny žádné numerické složky (série?) v {data_dir}")
        return


    # Projdeme složky 'zdrave' a 'po_rekonstrukci'
    for condition_name, source_series_dir in conditions.items():
        logging.info(f"  Prohledávám '{condition_name}' v {os.path.basename(source_series_dir)}...")
        series_in_dir = find_dicom_series_paths(source_series_dir)

        found_target = False
        for series_id, file_paths in series_in_dir.items():
            if not file_paths:
                continue

            # Získáme popis série z prvního souboru
            description = get_series_description(file_paths[0])
            logging.debug(f"    Nalezena série: '{description}' (ID: {series_id})")

            if description and description.strip() == target_desc.strip():
                logging.info(f"    Nalezena cílová série: '{description}'")
                found_target = True

                # Vytvoření cílové cesty
                dest_dir_final = os.path.join(dest_base, patient_id_str, condition_name, target_desc)
                try:
                    os.makedirs(dest_dir_final, exist_ok=True)
                    logging.info(f"    Vytvářím/ověřuji cílový adresář: {dest_dir_final}")

                    # Kopírování souborů série
                    copied_count = 0
                    for file_path in file_paths:
                         dest_file_path = os.path.join(dest_dir_final, os.path.basename(file_path))
                         try:
                             # Zkontrolujeme, zda soubor již neexistuje, abychom zbytečně nekopírovali
                             if not os.path.exists(dest_file_path):
                                 shutil.copy2(file_path, dest_file_path) # copy2 zachovává metadata
                                 copied_count += 1
                             else:
                                 # Soubor již existuje, můžeme přeskočit nebo logovat
                                 logging.debug(f"    Soubor již existuje, přeskakuji: {dest_file_path}")
                                 pass # Explicitně nic neděláme
                         except OSError as copy_err:
                              logging.error(f"      Chyba při kopírování souboru {file_path} do {dest_dir_final}: {copy_err}")

                    if copied_count > 0:
                        logging.info(f"    Úspěšně zkopírováno {copied_count}/{len(file_paths)} souborů série '{description}' do {dest_dir_final}")
                    else:
                        logging.info(f"    Všechny soubory pro sérii '{description}' již existují v {dest_dir_final}")


                except OSError as e:
                    logging.error(f"    Nelze vytvořit cílový adresář {dest_dir_final}: {e}")
                break # Našli jsme hledanou sérii, můžeme přejít k další podmínce (zdrave/po_rekonstrukci)

        if not found_target:
            logging.warning(f"  Cílová série '{target_desc}' nebyla nalezena ve složce '{condition_name}' ({os.path.basename(source_series_dir)})")

# --- Hlavní část skriptu ---
if __name__ == "__main__":
    if not os.path.isdir(SOURCE_BASE_DIR):
        logging.error(f"Zdrojový adresář neexistuje: {SOURCE_BASE_DIR}")
    else:
        logging.info(f"Prohledávám zdrojový adresář: {SOURCE_BASE_DIR}")
        logging.info(f"Cílový adresář pro výstup: {DEST_BASE_DIR}")
        logging.info(f"Hledám sérii s popisem: '{TARGET_SERIES_DESCRIPTION}'")

        patient_folders = []
        try:
             patient_folders = [os.path.join(SOURCE_BASE_DIR, d) for d in os.listdir(SOURCE_BASE_DIR)
                               if os.path.isdir(os.path.join(SOURCE_BASE_DIR, d)) and d.lower().startswith("pacient")]
        except OSError as e:
            logging.error(f"Chyba při čtení obsahu zdrojového adresáře {SOURCE_BASE_DIR}: {e}")

        if not patient_folders:
             logging.warning("Nebyly nalezeny žádné složky pacientů (s názvem začínajícím 'Pacient').")
        else:
             logging.info(f"Nalezeno {len(patient_folders)} složek pacientů.")
             for patient_folder_path in sorted(patient_folders): # Seřadíme pro konzistentní pořadí zpracování
                 process_patient_folder(patient_folder_path, DEST_BASE_DIR, TARGET_SERIES_DESCRIPTION)

        logging.info("Zpracování dokončeno.")