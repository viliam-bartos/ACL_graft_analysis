import os
import shutil
import csv
import numpy as np
from pathlib import Path
import re

try:
    import scipy.io
except ImportError:
    print("Nenalezen modul scipy. Nainstaluj: pip install scipy")
try:
    import h5py
except ImportError:
    print("Nenalezen modul h5py. Nainstaluj: pip install h5py")


def parse_dicom_age(age_val):
    if age_val is None or (isinstance(age_val, float) and np.isnan(age_val)):
        return np.nan
    age_str = str(age_val).strip()
    if not age_str or age_str.lower() == 'nan':
        return np.nan
    try:
        return float(age_str)
    except ValueError:
        pass

    match = re.match(r'^0*(\d+)([YMWDymwd])$', age_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'Y':
            return val
        elif unit == 'M':
            return round(val / 12, 2)
        elif unit == 'W':
            return round(val / 52.14, 2)
        elif unit == 'D':
            return round(val / 365.25, 2)
    return np.nan


def parse_weight(weight_val):
    if weight_val is None or (isinstance(weight_val, float) and np.isnan(weight_val)):
        return np.nan
    try:
        clean_str = re.sub(r'[^\d.]', '', str(weight_val))
        return float(clean_str) if clean_str else np.nan
    except:
        return np.nan


def find_in_struct(data, target_name):
    """Rekurzivně propátrá celou strukturu a najde hodnotu bez ohledu na zanoření."""
    if isinstance(data, dict):
        if target_name in data:
            return data[target_name]
        for v in data.values():
            res = find_in_struct(v, target_name)
            if res is not None:
                return res

    elif isinstance(data, np.ndarray):
        # Ošetření strukturovaných polí (typický výstup scipy ze Siemens dat)
        if data.dtype.names:
            if target_name in data.dtype.names:
                return data[target_name]
            for name in data.dtype.names:
                res = find_in_struct(data[name], target_name)
                if res is not None:
                    return res
        # Ošetření zanořených polí s jedním prvkem
        elif data.size == 1 and hasattr(data, 'item'):
            try:
                res = find_in_struct(data.item(), target_name)
                if res is not None:
                    return res
            except Exception:
                pass

    # Pro h5py (v7.3 mat soubory)
    elif hasattr(data, 'keys') and callable(getattr(data, 'keys')):
        if target_name in data.keys():
            return data[target_name][()]
        for k in data.keys():
            res = find_in_struct(data[k], target_name)
            if res is not None:
                return res

    return None


def extract_mat_data(mat_path):
    data = {'PatientAge': np.nan, 'PatientSex': np.nan, 'UsedPatientWeight': np.nan}

    if not mat_path.exists():
        print(f"Varování: Chybí .mat soubor: {mat_path.name}")
        return data

    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
        for k in data.keys():
            val = find_in_struct(mat, k)
            if val is not None:
                data[k] = val

    except NotImplementedError:
        try:
            with h5py.File(mat_path, 'r') as f:
                for k in data.keys():
                    val = find_in_struct(f, k)
                    if val is not None:
                        # h5py čte stringy často jako binární pole
                        if isinstance(val, np.ndarray) and val.dtype.kind in 'SUaO':
                            try:
                                data[k] = ''.join(chr(c[0]) for c in val)
                            except:
                                data[k] = val.item() if val.size == 1 else val
                        else:
                            data[k] = val.item() if isinstance(val, np.ndarray) and val.size == 1 else val
        except Exception as e:
            print(f"Varování: Nelze přečíst v7.3 soubor {mat_path.name}. Chyba: {e}")
    except Exception as e:
        print(f"Varování: Neočekávaná chyba při čtení {mat_path.name}. Chyba: {e}")

    # Agresivní čištění
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.size == 1:
                data[k] = v.item()
            elif v.size == 0:
                data[k] = np.nan
            else:
                try:
                    data[k] = "".join(v)
                except:
                    data[k] = str(v)

        if isinstance(data[k], str):
            data[k] = data[k].strip()

    data['PatientAge'] = parse_dicom_age(data.get('PatientAge'))
    data['UsedPatientWeight'] = parse_weight(data.get('UsedPatientWeight'))

    return data


def organize_data(src_dir, target_dir):
    src = Path(src_dir)
    target = Path(target_dir)

    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = target / "metadata.csv"
    mapping_path = target / "mapping.csv"

    allowed_names = ["pd_spc_rst_sag_p2_iso", "pd_space_sag_p4_iso"]

    all_nifti_files = [f for f in src.rglob("*") if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))]
    print(f"Počet všech nalezených nifti souborů: {len(all_nifti_files)}")

    valid_files = [f for f in all_nifti_files if any(name in f.name for name in allowed_names)]
    print(f"Odpovídá podmínkám a bude zpracováno: {len(valid_files)}\n")

    if not valid_files:
        print("Nemám co zpracovávat. Konec.")
        return

    metadata_records = []
    mapping_records = []

    for i, nii_path in enumerate(valid_files, start=1):
        case_id = f"case_{i:03d}"

        ext = ".nii.gz" if nii_path.name.endswith(".nii.gz") else ".nii"
        base_name = nii_path.name.replace(ext, "")

        mat_candidates = list(nii_path.parent.glob(f"*{base_name}*.mat"))
        if mat_candidates:
            mat_path = mat_candidates[0]
        else:
            mat_path = nii_path.parent / f"_{base_name}_dicom_header.mat"

        new_nii_path = images_dir / f"{case_id}{ext}"
        shutil.copy2(nii_path, new_nii_path)

        mat_data = extract_mat_data(mat_path)

        metadata_records.append({
            "case_id": case_id,
            "age": mat_data['PatientAge'],
            "sex": mat_data['PatientSex'],
            "weight": mat_data['UsedPatientWeight']
        })

        mapping_records.append({
            "case_id": case_id,
            "original_path": str(nii_path.resolve())
        })

    with open(metadata_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "age", "sex", "weight"])
        writer.writeheader()
        writer.writerows(metadata_records)

    with open(mapping_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "original_path"])
        writer.writeheader()
        writer.writerows(mapping_records)

    print(f"Hotovo. Data uložena do: {target}")


if __name__ == "__main__":
    SOURCE = r"C:\DIPLOM_PRACE\ACL_segment\puvodni_mr_data\VUT"
    TARGET = r"C:\DIPLOM_PRACE\ACL_segment\data_train"
    organize_data(SOURCE, TARGET)