import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import napari

import SimpleITK as sitk
import pydicom
import nibabel as nib
from scipy.ndimage import label, binary_closing, binary_erosion, gaussian_filter
from skimage.morphology import skeletonize

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd, SpatialPadd
)
from monai.data import Dataset, DataLoader


# ==========================================
# 1. ARCHITEKTURA MODELU (LightUNet3D)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))) + self.skip(x))


class LightUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base * 2)
        self.enc3 = ResBlock(base * 2, base * 4)
        self.bottleneck = ResBlock(base * 4, base * 8)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce3 = nn.Conv3d(base * 8, base * 4, kernel_size=1, bias=False)
        self.dec3 = ResBlock(base * 8, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce2 = nn.Conv3d(base * 4, base * 2, kernel_size=1, bias=False)
        self.dec2 = ResBlock(base * 4, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.reduce1 = nn.Conv3d(base * 2, base, kernel_size=1, bias=False)
        self.dec1 = ResBlock(base * 2, base)

        self.final = nn.Conv3d(base, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bn = self.dropout(self.bottleneck(self.pool(x3)))

        d3 = self.dec3(torch.cat([self.reduce3(self.up3(bn)), x3], dim=1))
        d2 = self.dec2(torch.cat([self.reduce2(self.up2(d3)), x2], dim=1))
        d1 = self.dec1(torch.cat([self.reduce1(self.up1(d2)), x1], dim=1))

        return self.final(d1)


# ==========================================
# 2. NAČÍTÁNÍ A INFERENCE
# ==========================================
def load_mri_data(path, target_sequence="pd_space_sag_p4_iso"):
    image = None

    if os.path.isfile(path) and path.endswith(('.nii', '.nii.gz')):
        image = sitk.ReadImage(path)

    elif os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        try:
            series_ids = reader.GetGDCMSeriesIDs(path)
        except RuntimeError:
            series_ids = []

        if not series_ids:
            raise FileNotFoundError(f"V adresáři '{path}' nebyly nalezeny žádné DICOM série.")

        series_details = []
        for series_id in series_ids:
            dicom_names = reader.GetGDCMSeriesFileNames(path, series_id)
            if not dicom_names:
                continue

            try:
                header = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
                description = header.get("SeriesDescription", "N/A")
            except Exception:
                description = "Nelze přečíst popis"

            if target_sequence.lower() in description.lower():
                print(f"  Automaticky nalezena sekvence: '{description}'")
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                break

            series_details.append({
                "id": series_id,
                "description": description,
                "files": dicom_names
            })

        if image is None:
            print(f"\nSekvence '{target_sequence}' nebyla automaticky nalezena.")
            print(f"Nalezeno {len(series_details)} sérií. Vyber jednu pro zpracování:")
            for i, detail in enumerate(series_details):
                print(f"  [{i + 1}] {detail['description']} (Počet souborů: {len(detail['files'])})")

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
            print(f"  Načítám sérii: '{selected['description']}'...")
            reader.SetFileNames(selected['files'])
            image = reader.Execute()

    if image is None:
        raise ValueError(f"Neplatná cesta k datům: {path}")

    # FIX: Odstranění prázdné 4. dimenze
    if image.GetDimension() == 4:
        image = image[:, :, :, 0]

    return image


def get_scan_date(path):
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        if series_ids:
            dicom_names = reader.GetGDCMSeriesFileNames(path, series_ids[0])
            if dicom_names:
                dcm = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
                date_str = getattr(dcm, 'AcquisitionDate', getattr(dcm, 'StudyDate', ''))
                if date_str:
                    try:
                        return datetime.strptime(date_str, '%Y%m%d')
                    except ValueError:
                        pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def run_inference(image_sitk, model_path, device):
    """Provede inferenci přes MONAI a vrátí SimpleITK masku."""
    # Uložení do temp filu pro MONAI pipeline
    temp_dir = tempfile.mkdtemp()
    temp_in_path = os.path.join(temp_dir, "temp_img.nii.gz")
    temp_out_path = os.path.join(temp_dir, "temp_mask.nii.gz")
    sitk.WriteImage(image_sitk, temp_in_path)

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 32)),
    ])

    ds = Dataset(data=[{"image": temp_in_path}], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = LightUNet3D(in_ch=1, out_ch=1, base=16).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    for batch in loader:
        input_tensor = batch["image"].to(device)
        original_affine = batch["image"].affine[0].numpy() if hasattr(batch["image"], 'affine') else np.eye(4)

        with torch.no_grad():
            output_tensor = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(128, 128, 32),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode='constant',
                device=device
            )
            output_mask = (torch.sigmoid(output_tensor) > 0.5).float()

        pred_array = output_mask.cpu().numpy()[0, 0, :, :, :].astype(np.uint8)

        # Ořezání paddingu (SpatialPadd přidává na konec, musíme oříznout zpět na původní rozměr z hlavičky)
        orig_shape = nib.load(temp_in_path).shape
        pred_array = pred_array[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

        out_nifti = nib.Nifti1Image(pred_array, original_affine)
        nib.save(out_nifti, temp_out_path)

    # Zpět do SimpleITK
    result_sitk = sitk.ReadImage(temp_out_path)

    # Úklid
    os.remove(temp_in_path)
    os.remove(temp_out_path)
    os.rmdir(temp_dir)

    return result_sitk


# ==========================================
# 3. ZPRACOVÁNÍ MASEK A EXTRAKCE
# ==========================================
def postprocess_mask_general(mask_sitk):
    mask_arr = sitk.GetArrayFromImage(mask_sitk)

    # 1. Nejdřív odstraníme šum (LCC)
    labeled_array, num_features = label(mask_arr)
    if num_features > 1:
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0  # Ignorujeme pozadí
        max_label = sizes.argmax()
        mask_arr = (labeled_array == max_label).astype(np.uint8)

    # 2. Až teď aplikujeme Closing na ten jeden vyčištěný vaz
    struct = np.ones((3, 3, 3), dtype=bool)
    closed_mask = binary_closing(mask_arr, structure=struct).astype(np.uint8)

    result_itk = sitk.GetImageFromArray(closed_mask)
    result_itk.CopyInformation(mask_sitk)
    return result_itk


def extract_pcl_median(clean_pcl_mask, image):
    mask_arr = sitk.GetArrayFromImage(clean_pcl_mask)
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # 1. Agresivnější eroze (odstraní okraje, které by mohly přesahovat do výpotku)
    eroded_mask = binary_erosion(mask_arr, iterations=2)
    pcl_voxels = img_arr[eroded_mask == 1]

    if len(pcl_voxels) == 0:
        # Fallback, pokud byl PCL tak tenký, že zmizel - použijeme původní masku
        pcl_voxels = img_arr[mask_arr == 1]

    # 2. Bereme jen ty nejtmavší voxely (5. až 30. percentil)
    # Tím spolehlivě odřízneme jakýkoliv šum z okolní tekutiny
    p5, p30 = np.percentile(pcl_voxels, [5, 30])
    dark_core_voxels = pcl_voxels[(pcl_voxels >= p5) & (pcl_voxels <= p30)]

    return np.median(dark_core_voxels)


def get_principal_axis(mask_sitk, trim_percent=0.45):
    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(bool)

    # 1. Vytvoření 1D středové křivky
    skeleton = skeletonize(mask_arr)
    coords = np.argwhere(skeleton).astype(np.float64)

    if len(coords) < 10:
        return None

    # 2. Převod na fyzické jednotky
    spacing = mask_sitk.GetSpacing()
    coords[:, 0] *= spacing[2]
    coords[:, 1] *= spacing[1]
    coords[:, 2] *= spacing[0]

    # 3. Odříznutí zahnutých konců (úponů)
    # Seřadíme voxely podle osy Z (v numpy coords[:, 0]) a vezmeme jen střední část
    coords = coords[coords[:, 0].argsort()]
    trim_len = int(len(coords) * trim_percent)

    if trim_len > 0 and len(coords) > 2 * trim_len:
        coords = coords[trim_len:-trim_len]

    if len(coords) < 5:
        return None

    # 4. Výpočet osy jen ze střední, relativně rovné části
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords, rowvar=False)
    _, evecs = np.linalg.eigh(cov)

    return evecs[:, -1]


def calculate_acl_pcl_angle(acl_mask, pcl_mask):
    v_acl = get_principal_axis(acl_mask)
    v_pcl = get_principal_axis(pcl_mask)

    if v_acl is None or v_pcl is None:
        return np.nan

    cos_theta = np.dot(v_acl, v_pcl) / (np.linalg.norm(v_acl) * np.linalg.norm(v_pcl))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_theta))

    if angle < 90:
        angle = 180 - angle

    return angle


def get_tibial_plateau_vector(acl_mask_sitk, pcl_mask_sitk, bottom_percent=0.15):
    """Vypočítá směrový vektor virtuálního tibiálního plata ze spodních úponů vazů."""
    acl_arr = sitk.GetArrayFromImage(acl_mask_sitk)
    pcl_arr = sitk.GetArrayFromImage(pcl_mask_sitk)

    acl_coords = np.argwhere(acl_arr == 1).astype(np.float64)
    pcl_coords = np.argwhere(pcl_arr == 1).astype(np.float64)

    if len(acl_coords) < 10 or len(pcl_coords) < 10:
        return None

    spacing = acl_mask_sitk.GetSpacing()
    for coords in [acl_coords, pcl_coords]:
        coords[:, 0] *= spacing[2]
        coords[:, 1] *= spacing[1]
        coords[:, 2] *= spacing[0]

    # Osa Y je index 1. Nejvyšší hodnoty Y představují nejspodnější část obrazu (úpon na tibii).
    acl_y_thresh = np.percentile(acl_coords[:, 1], 100 - (bottom_percent * 100))
    pcl_y_thresh = np.percentile(pcl_coords[:, 1], 100 - (bottom_percent * 100))

    acl_bottom = acl_coords[acl_coords[:, 1] >= acl_y_thresh]
    pcl_bottom = pcl_coords[pcl_coords[:, 1] >= pcl_y_thresh]

    if len(acl_bottom) == 0 or len(pcl_bottom) == 0:
        return None

    centroid_acl = np.mean(acl_bottom, axis=0)
    centroid_pcl = np.mean(pcl_bottom, axis=0)

    # Vektor směruje od PCL (vzadu) k ACL (vpředu)
    plateau_vec = centroid_acl - centroid_pcl
    norm = np.linalg.norm(plateau_vec)
    if norm == 0:
        return None

    return plateau_vec / norm


def calculate_acl_tibial_angle(acl_mask, pcl_mask):
    """Vypočítá úhel mezi skeletem ACL a virtuálním tibiálním platem."""
    # Použijeme tvou stávající skeletonizační funkci pro osu ACL
    v_acl = get_principal_axis(acl_mask, trim_percent=0.2)
    v_plateau = get_tibial_plateau_vector(acl_mask, pcl_mask)

    if v_acl is None or v_plateau is None:
        return np.nan

    cos_theta = np.dot(v_acl, v_plateau) / (np.linalg.norm(v_acl) * np.linalg.norm(v_plateau))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    # Anatomický úhel vazu vůči kosti je ostrý, většinou někde kolem 45-55 stupňů.
    if angle > 90:
        angle = 180 - angle

    return angle


# Přidán parametr pcl_mask
def extract_features(image, acl_mask, pcl_mask, pcl_median_intensity):
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(acl_mask)

    acl_voxels = img_arr[mask_arr == 1]
    features = {}

    if len(acl_voxels) == 0:
        return features

    spacing = image.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    features['volume_mm3'] = len(acl_voxels) * voxel_vol

    acl_median = np.median(acl_voxels)
    features['acl_median_intensity'] = acl_median

    if pcl_median_intensity and not np.isnan(pcl_median_intensity) and pcl_median_intensity > 0:
        features['siq_acl_pcl'] = acl_median / pcl_median_intensity
    else:
        features['siq_acl_pcl'] = np.nan

    counts, _ = np.histogram(acl_voxels, bins=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    features['entropy'] = -np.sum(probs * np.log2(probs))

    # Nový výpočet úhlu
    features['acl_pcl_angle'] = calculate_acl_pcl_angle(acl_mask, pcl_mask)

    features['acl_tibial_angle'] = calculate_acl_tibial_angle(acl_mask, pcl_mask)

    return features



# ==========================================
# 4. VIZUALIZACE A MAIN
# ==========================================
def plot_heatmap(image, acl_mask, pcl_median, date_str):
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(acl_mask)

    z_slice = np.argmax(np.sum(mask_arr, axis=(1, 2)))
    img_slice = img_arr[z_slice]
    mask_slice = mask_arr[z_slice]

    heatmap = np.zeros_like(img_slice, dtype=np.float32)
    if pcl_median and pcl_median > 0:
        heatmap[mask_slice == 1] = img_slice[mask_slice == 1] / pcl_median
    else:
        heatmap[mask_slice == 1] = img_slice[mask_slice == 1]

    heatmap_masked = np.ma.masked_where(mask_slice == 0, heatmap)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(heatmap_masked, cmap='coolwarm', alpha=0.6, vmin=0.5, vmax=2.5)
    plt.colorbar(label='SIQ (Signal Intensity Quotient)')
    plt.title(f'ACL Heatmapa - {date_str}')
    plt.axis('off')
    plt.show()


def get_pca_line(mask_sitk, length_voxels=120, trim_percent=0.35):
    # 1. Získáme správný vektor ze skeletonizace
    v_phys = get_principal_axis(mask_sitk, trim_percent=trim_percent)

    if v_phys is None:
        return None

    # 2. Najdeme střed vazu, abychom měli čáru kam ukotvit
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    coords = np.argwhere(mask_arr == 1).astype(np.float64)
    if len(coords) < 1:
        return None
    centroid = np.mean(coords, axis=0)

    # 3. Převod vektoru zpět do voxelů pro zobrazení v Napari
    spacing = mask_sitk.GetSpacing()
    v_voxel = np.zeros(3)
    v_voxel[0] = v_phys[0] / spacing[2]
    v_voxel[1] = v_phys[1] / spacing[1]
    v_voxel[2] = v_phys[2] / spacing[0]

    v_voxel = v_voxel / np.linalg.norm(v_voxel)

    p1 = centroid - v_voxel * (length_voxels / 2)
    p2 = centroid + v_voxel * (length_voxels / 2)

    return [np.array([p1, p2])]


def get_tibial_plateau_line(acl_mask_sitk, pcl_mask_sitk, length_voxels=150, bottom_percent=0.15):
    acl_arr = sitk.GetArrayFromImage(acl_mask_sitk)
    pcl_arr = sitk.GetArrayFromImage(pcl_mask_sitk)

    acl_coords = np.argwhere(acl_arr == 1).astype(np.float64)
    pcl_coords = np.argwhere(pcl_arr == 1).astype(np.float64)

    if len(acl_coords) < 10 or len(pcl_coords) < 10:
        return None

    spacing = acl_mask_sitk.GetSpacing()
    acl_phys = acl_coords.copy()
    pcl_phys = pcl_coords.copy()

    for coords in [acl_phys, pcl_phys]:
        coords[:, 0] *= spacing[2]
        coords[:, 1] *= spacing[1]
        coords[:, 2] *= spacing[0]

    acl_y_thresh = np.percentile(acl_phys[:, 1], 100 - (bottom_percent * 100))
    pcl_y_thresh = np.percentile(pcl_phys[:, 1], 100 - (bottom_percent * 100))

    acl_bottom = acl_phys[acl_phys[:, 1] >= acl_y_thresh]
    pcl_bottom = pcl_phys[pcl_phys[:, 1] >= pcl_y_thresh]

    if len(acl_bottom) == 0 or len(pcl_bottom) == 0:
        return None

    centroid_acl = np.mean(acl_bottom, axis=0)
    centroid_pcl = np.mean(pcl_bottom, axis=0)

    # Kotevní bod přímky přesně mezi úpony
    anchor_phys = (centroid_acl + centroid_pcl) / 2.0

    plateau_vec = centroid_acl - centroid_pcl
    norm = np.linalg.norm(plateau_vec)
    if norm == 0:
        return None
    v_phys = plateau_vec / norm

    # Převod zpět do voxelů pro zobrazení
    anchor_voxel = np.zeros(3)
    anchor_voxel[0] = anchor_phys[0] / spacing[2]
    anchor_voxel[1] = anchor_phys[1] / spacing[1]
    anchor_voxel[2] = anchor_phys[2] / spacing[0]

    v_voxel = np.zeros(3)
    v_voxel[0] = v_phys[0] / spacing[2]
    v_voxel[1] = v_phys[1] / spacing[1]
    v_voxel[2] = v_phys[2] / spacing[0]

    v_voxel = v_voxel / np.linalg.norm(v_voxel)

    p1 = anchor_voxel - v_voxel * (length_voxels / 2)
    p2 = anchor_voxel + v_voxel * (length_voxels / 2)

    return [np.array([p1, p2])]


def show_in_napari(image_sitk, acl_mask_sitk, pcl_mask_sitk, pcl_median, smooth_sigma=1.0):
    img_arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    acl_arr = sitk.GetArrayFromImage(acl_mask_sitk).astype(np.float32)
    pcl_arr = sitk.GetArrayFromImage(pcl_mask_sitk).astype(np.float32)

    if pcl_median and pcl_median > 0:
        siq_map = img_arr / pcl_median
    else:
        siq_map = img_arr.copy()

    if smooth_sigma > 0:
        siq_map = gaussian_filter(siq_map, sigma=smooth_sigma)

    heatmap_3d = np.full_like(img_arr, np.nan, dtype=np.float32)
    heatmap_3d[acl_arr == 1] = siq_map[acl_arr == 1]

    # Výpočet úseček pro PCA osy
    acl_line = get_pca_line(acl_mask_sitk)
    pcl_line = get_pca_line(pcl_mask_sitk)

    viewer = napari.Viewer()

    # Vynutí 3D zobrazení hned po spuštění, aby byly vidět osy
    viewer.dims.ndisplay = 3


    viewer.add_image(img_arr, name='MRI', colormap='gray')
    viewer.add_image(pcl_arr, name='PCL Maska', colormap='blue', blending='additive', opacity=0.4)
    viewer.add_image(acl_arr, name='ACL Maska', colormap='green', blending='additive', opacity=0.4)

    viewer.add_image(
        heatmap_3d,
        name='ACL Heatmapa (vyhlazená)',
        colormap='coolwarm',
        contrast_limits=[0.5, 2.5],
        opacity=0.8,
        blending='translucent',
        interpolation3d='linear',
        visible = False
    )

    # Rozdělené vrstvy pro jistotu zobrazení
    acl_line = get_pca_line(acl_mask_sitk)
    pcl_line = get_pca_line(pcl_mask_sitk)

    if acl_line is not None:
        viewer.add_shapes(
            acl_line, shape_type='line', edge_color='magenta',
            edge_width=3, name='Osa ACL'
        )
    if pcl_line is not None:
        viewer.add_shapes(
            pcl_line, shape_type='line', edge_color='cyan',
            edge_width=3, name='Osa PCL'
        )

    tibial_line = get_tibial_plateau_line(acl_mask_sitk, pcl_mask_sitk)

    if tibial_line is not None:
        viewer.add_shapes(
            tibial_line, shape_type='line', edge_color='red',
            edge_width=3, name='Virtuální Tibiální Plato'
        )

    napari.run()


def main():
    ACL_MODEL_PATH = r"C:\DIPLOM_PRACE\ACL_segment\vysledky_modelu\results_3D_CV\fold_0\best_model.pth"
    PCL_MODEL_PATH = r"C:\DIPLOM_PRACE\ACL_segment\vysledky_modelu\results_3D_pcl_funkcni\best_model.pth"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # PŘEPÍNAČ: True = ukáže 3D prohlížeč (zastaví skript), False = projede to potichu
    SHOW_NAPARI = True

    patient_data_paths = [
        r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_01\zdrave\pd_space_sag_p4_iso",
        r"C:\DIPLOM_PRACE\ACL_segment\Organized_Data\pacient_01\po_rekonstrukci\pd_space_sag_p4_iso",
    ]

    if not patient_data_paths:
        print("Doplň cesty k datům do 'patient_data_paths'.")
        return

    results = []

    for path in patient_data_paths:
        try:
            date_obj = get_scan_date(path)
            print(f"\n--- Zpracovávám: {path} (Datum: {date_obj.strftime('%Y-%m-%d')}) ---")

            img = load_mri_data(path)

            print("  Inference ACL...")
            acl_pred = run_inference(img, ACL_MODEL_PATH, DEVICE)
            print("  Inference PCL...")
            pcl_pred = run_inference(img, PCL_MODEL_PATH, DEVICE)

            print("  Postprocessing obou masek...")
            acl_post = postprocess_mask_general(acl_pred)
            pcl_post = postprocess_mask_general(pcl_pred)

            print("  Extrakce příznaků...")
            pcl_median = extract_pcl_median(pcl_post, img)

            feats = extract_features(img, acl_post, pcl_post, pcl_median)

            feats['date'] = date_obj
            feats['path'] = path
            results.append(feats)

            # PODMÍNKA PRO ZOBRAZENÍ
            if SHOW_NAPARI:
                print("  Spouštím 3D vizualizaci. Pro pokračování skriptu zavři okno Napari...")
                show_in_napari(img, acl_post, pcl_post, pcl_median, smooth_sigma=1.0)

        except Exception as e:
            print(f"Chyba při zpracování {path}: {e}")

    if results:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        df = pd.DataFrame(results).sort_values(by='date')
        print("\nVýsledky:")
        print(df[['date', 'volume_mm3', 'acl_median_intensity', 'siq_acl_pcl', 'entropy', 'acl_pcl_angle', 'acl_tibial_angle']])

        if len(df) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(df['date'], df['siq_acl_pcl'], marker='o', color='b', label='SIQ')
            plt.xlabel('Datum vyšetření')
            plt.ylabel('SIQ (ACL / PCL)')
            plt.title('Vývoj ligamentizace štěpu v čase')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()