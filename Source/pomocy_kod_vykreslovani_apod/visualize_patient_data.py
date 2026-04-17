import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from matplotlib.colors import ListedColormap


# --- Funkce pro zpracování obrazu ---

def reorient_to_standard(image: sitk.Image, desired_orientation: str = "LPS") -> sitk.Image:
    """Reorient image to a specified standard orientation."""
    print(f"Provádím reorientaci do standardní orientace ({desired_orientation})...")
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(desired_orientation)
    return orient_filter.Execute(image)


def crop_to_content(image_array: np.ndarray, threshold_factor: float = 0.05) -> tuple[np.ndarray, list[slice]]:
    """Ořízne 3D numpy pole a vrátí oříznuté pole a použité řezy."""
    print("Provádím ořez na obsah obrazu...")
    threshold = np.max(image_array) * threshold_factor
    coords = np.argwhere(image_array > threshold)
    if coords.size == 0:
        print("VAROVÁNÍ: Nenalezen žádný obsah k oříznutí, vracím původní obraz.")
        return image_array, [slice(None)] * 3
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    crop_slices = [slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1)]
    print(f"Ořez dokončen. Původní tvar: {image_array.shape}, Nový tvar: {image_array[tuple(crop_slices)].shape}")
    return image_array[tuple(crop_slices)], crop_slices


def select_and_load_dicom_series(directory: str):
    """Identifikuje všechny DICOM série, nechá uživatele vybrat a načte sérii."""
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(directory)
    except RuntimeError:
        series_ids = []

    if not series_ids:
        raise FileNotFoundError(f"V adresáři '{directory}' nebyly nalezeny žádné DICOM série.")

    print(f"Nalezeno {len(series_ids)} sérií. Vyberte jednu pro zobrazení:")
    series_details = []
    for i, series_id in enumerate(series_ids):
        dicom_names = reader.GetGDCMSeriesFileNames(directory, series_id)
        description = "N/A"
        if dicom_names:
            try:
                header = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
                description = header.get("SeriesDescription", "N/A")
            except Exception:
                description = "Nelze přečíst popis"
        series_details.append({"id": series_id, "description": description, "files": len(dicom_names)})
        print(f"  [{i + 1}] {description} (Počet souborů: {len(dicom_names)})")

    while True:
        try:
            choice = int(input(f"\nZadejte číslo série (1-{len(series_details)}): ")) - 1
            if 0 <= choice < len(series_details):
                break
            else:
                print("Neplatná volba.")
        except ValueError:
            print("Prosím zadejte číslo.")

    selected = series_details[choice]
    print(f"\n---> Načítám sérii: '{selected['description']}'...")
    dicom_names = reader.GetGDCMSeriesFileNames(directory, selected['id'])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    print("Načítání dokončeno.")
    return image, selected['description']


# --- Interaktivní prohlížeč ---

class InteractiveViewer:
    """Interaktivní prohlížeč, který dokáže překrýt segmentační masku."""

    def __init__(self, volume, mask_volume=None, plane='sagittal', aspect=1.0):
        self.volume = volume
        self.mask_volume = mask_volume
        self.plane = plane
        self.aspect = aspect
        self.mask_cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.4)])

        # Osa pro řezání se nemění, ale zobrazení ano
        if self.plane == 'sagittal':
            self.axis_idx = 2  # Řezy podél osy X
        elif self.plane == 'coronal':
            self.axis_idx = 1  # Řezy podél osy Y
        else:
            self.axis_idx = 0  # Řezy podél osy Z (axial)

        self.slices = volume.shape[self.axis_idx]
        self.ind = self.slices // 2

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 10))
        self.fig.suptitle(f'Segmentace - {self.plane.capitalize()} rovina\n(Scrolujte kolečkem myši)', fontsize=16)

        # 'origin=lower' zajišťuje, že se obraz zobrazí anatomicky správně
        self.im_display = self.ax.imshow(self.get_slice_data(self.volume), cmap='gray', aspect=self.aspect,
                                         origin='lower')
        if self.mask_volume is not None:
            self.mask_display = self.ax.imshow(self.get_slice_data(self.mask_volume), cmap=self.mask_cmap,
                                               aspect=self.aspect, interpolation='none', origin='lower')

        self.update_title()

    def get_slice_data(self, vol):
        # NumPy pole je (Z, Y, X)
        if self.plane == 'sagittal':
            # Vykreslujeme řez (Z, Y)
            return vol[:, :, self.ind]
        elif self.plane == 'coronal':
            # Vykreslujeme řez (Z, X)
            return vol[:, self.ind, :]
        else:  # axial
            # Vykreslujeme řez (Y, X)
            return vol[self.ind, :, :]

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1 + self.slices) % self.slices
        self.update_slice()

    def update_title(self):
        self.ax.set_title(f'Řez: {self.ind + 1}/{self.slices}')

    def update_slice(self):
        self.im_display.set_data(self.get_slice_data(self.volume))
        if self.mask_volume is not None:
            self.mask_display.set_data(self.get_slice_data(self.mask_volume))
        self.update_title()
        self.fig.canvas.draw_idle()

    def show(self):
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()


# --- Hlavní skript ---

if __name__ == "__main__":
    # --- ZMĚŇTE TYTO CESTY ---
    DICOM_DIR_PARENT = r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_01\zdrave\pd_space_sag_p4_iso"
    SEGMENTATION_PATH = r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_01\zdrave\pd_space_sag_p4_iso\Pacient_01_zdrave_mask.nii"
    # --- ----------------- ---

    try:
        # 1. Výběr a načtení DICOM série
        image, series_name = select_and_load_dicom_series(DICOM_DIR_PARENT)

        if image.GetDimension() == 4:
            size = list(image.GetSize());
            size[3] = 0;
            index = [0, 0, 0, 0]
            image = sitk.Extract(image, size, index)

        image_reoriented = reorient_to_standard(image)

        # 2. Načtení a zpracování segmentace
        print("Načítám segmentační masku...")
        mask = sitk.ReadImage(SEGMENTATION_PATH, sitk.sitkUInt8)
        mask_reoriented = reorient_to_standard(mask)

        # 3. KROK ZAROVNÁNÍ: Převzorkování masky na mřížku obrazu
        print("Zarovnávám masku na obraz (resampling)...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_reoriented)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nutné pro masky!
        mask_resampled = resampler.Execute(mask_reoriented)

        # 4. Převedení na NumPy pole
        image_array = sitk.GetArrayFromImage(image_reoriented)
        mask_array = sitk.GetArrayFromImage(mask_resampled)

        # 5. Ořezání obou polí podle obsahu obrazu
        image_cropped, crop_slices = crop_to_content(image_array)
        mask_cropped = mask_array[tuple(crop_slices)]

        print(f"Finální tvar obrazu: {image_cropped.shape}")
        print(f"Finální tvar masky: {mask_cropped.shape}")

        if image_cropped.shape != mask_cropped.shape:
            print("VAROVÁNÍ: Rozměry obrazu a masky se po zpracování neshodují! Chyba v logice.")

        # 6. Výpočet poměru stran a zobrazení
        spacing = image_reoriented.GetSpacing()  # Formát: (X, Y, Z)
        viewing_plane = 'sagittal'
        aspect_ratio = 1.0

        if 'sag' in series_name.lower():
            viewing_plane = 'sagittal'
            # Zobrazujeme řez (Z, Y). Poměr stran = (rozteč ve výšce) / (rozteč v šířce)
            # Výška = osa Z, Šířka = osa Y
            aspect_ratio = spacing[2] / spacing[1]
        elif 'cor' in series_name.lower():
            viewing_plane = 'coronal'
            # Zobrazujeme řez (Z, X).
            # Výška = osa Z, Šířka = osa X
            aspect_ratio = spacing[2] / spacing[0]
        elif 'tra' in series_name.lower() or 'ax' in series_name.lower():
            viewing_plane = 'axial'
            # Zobrazujeme řez (Y, X).
            # Výška = osa Y, Šířka = osa X
            aspect_ratio = spacing[1] / spacing[0]

        viewer = InteractiveViewer(image_cropped, mask_volume=mask_cropped, plane=viewing_plane, aspect=aspect_ratio)
        viewer.show()

    except Exception as e:
        print(f"Vyskytla se chyba: {e}")

