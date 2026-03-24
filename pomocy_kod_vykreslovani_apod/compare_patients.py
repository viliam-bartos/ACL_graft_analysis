import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def reorient_to_standard(image: sitk.Image, desired_orientation: str = "LPS") -> sitk.Image:
    """
    Reorient image to a specified standard orientation using DICOMOrientImageFilter.
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(desired_orientation)
    return orient_filter.Execute(image)


def crop_to_content(image_array: np.ndarray, threshold_factor: float = 0.05) -> np.ndarray:
    """
    Crops a 3D numpy array to remove empty (black) borders.
    """
    threshold = np.max(image_array) * threshold_factor
    coords = np.argwhere(image_array > threshold)
    if coords.size == 0:
        return image_array
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    return image_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]


def find_and_load_series(directory: str, series_description_to_find: str) -> sitk.Image | None:
    """
    Finds a series by its description and loads it.
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        for series_id in series_ids:
            dicom_names = reader.GetGDCMSeriesFileNames(directory, series_id)
            if not dicom_names: continue
            header = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
            if header.get("SeriesDescription") == series_description_to_find:
                print(
                    f"  Nalezena série '{series_description_to_find}' v adresáři '{os.path.basename(directory)}'. Načítám...")
                reader.SetFileNames(dicom_names)
                return reader.Execute()
    except Exception:
        # Tichá chyba, pokud adresář neobsahuje DICOM soubory
        pass
    return None


if __name__ == '__main__':
    BASE_DIRECTORY = r"C:\DIPLOM_PRACE\ACL_segment\MRI"
    patients = [f"Pacient 0{i}" for i in range(1, 4)]
    target_series = [
        "pd_tse_fs_sag_DRB",
        "pd_space_sag_p4_iso"
    ]
    excluded_dirs = ['7z', 'DW']

    print("Spouštím porovnání pacientů...")

    for patient in patients:
        patient_path = os.path.join(BASE_DIRECTORY, patient)
        data_root = os.path.join(patient_path, 'DATA')

        if not os.path.exists(data_root):
            print(f"Adresář DATA pro pacienta {patient} nenalezen.")
            continue

        # Prohledá všechny podadresáře v DATA, s výjimkou specifikovaných
        subdirs = [os.path.join(data_root, d) for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d)) and d not in excluded_dirs]

        for dicom_dir in subdirs:
            print(f"\nProhledávám složku: {os.path.basename(dicom_dir)} pro pacienta {patient}")
            for series_name in target_series:
                image = find_and_load_series(dicom_dir, series_name)

                if image is None:
                    continue

                # --- Zpracování dat ---
                if image.GetDimension() == 4:
                    size = list(image.GetSize())
                    size[3] = 0
                    index = [0, 0, 0, 0]
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize(size)
                    extractor.SetIndex(index)
                    image = extractor.Execute(image)

                oriented_image = reorient_to_standard(image)
                spacing = oriented_image.GetSpacing()
                img_array = sitk.GetArrayFromImage(oriented_image)
                cropped_array = crop_to_content(img_array)

                if cropped_array.size == 0:
                    continue

                aspect_ratio = spacing[2] / spacing[1]

                # --- Vykreslení ---
                fig = plt.figure(figsize=(7, 9))
                ax = fig.add_subplot(1, 1, 1)

                middle_slice_idx = cropped_array.shape[2] // 2
                # Correct orientation for anatomical view (Superior at top)
                slice_data = np.flipud(cropped_array[:, :, middle_slice_idx])

                ax.imshow(slice_data, cmap='gray', aspect=aspect_ratio)
                ax.set_title(f"{patient} - {os.path.basename(dicom_dir)}\n{series_name}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()

    print("\nVšechna data zpracována. Zobrazuji výsledky...")
    plt.show()

