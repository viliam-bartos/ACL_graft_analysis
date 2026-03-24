import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    RandBiasFieldd
)

INPUT_FILE = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images\case001.nii.gz"

# Zvýšíme sílu, aby to bylo vidět (v tréninku nech nižší)
BIAS_CONFIG = {
    'prob': 1.0,
    'degree': 3,  # Složitost polynomu (3 je klasika pro MRI)
    'coeff_range': (0.3, 0.5)  # Rozsah síly (default je mnohem menší cca 0.1)
}


def visualize_bias_field():
    if not os.path.exists(INPUT_FILE):
        return

    # 1. Pipeline
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Normalizace je nutná, jinak se efekt ztratí v obrovských číslech
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),

        RandBiasFieldd(
            keys=["image"],
            prob=BIAS_CONFIG['prob'],
            degree=BIAS_CONFIG['degree'],
            coeff_range=BIAS_CONFIG['coeff_range']
        )
    ])

    # 2. Načtení dat
    # A) Čistý
    loader = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
    ])
    orig_data = loader({"image": INPUT_FILE})

    # B) S Bias Fieldem
    # Zafixujeme seed, abychom viděli konzistentní výsledek
    biased_data = transforms({"image": INPUT_FILE})

    img_orig = orig_data["image"][0]
    img_bias = biased_data["image"][0]

    # 3. Výpočet samotného pole (rozdíl/podíl)
    # Bias field je multiplikativní efekt: Image_new = Image_old * Field
    # Takže Field = Image_new / (Image_old + epsilon)
    # Pro vizualizaci stačí prostý rozdíl, abychom viděli "stín"
    diff_map = img_bias - img_orig

    # 4. Vizualizace
    slice_idx = img_orig.shape[2] // 2

    def fix(arr):
        return np.rot90(np.rot90(np.fliplr(arr[:, :, slice_idx].T)))

    plt.figure(figsize=(24, 8))

    # Originál
    plt.subplot(1, 3, 1)
    plt.imshow(fix(img_orig), cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.title("Originál", fontsize=30)
    plt.axis('off')

    # S Bias Fieldem
    plt.subplot(1, 3, 2)
    plt.imshow(fix(img_bias), cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.title(f"Bias Field (coeff={BIAS_CONFIG['coeff_range']})", fontsize=25)
    plt.axis('off')

    # Rozdílová mapa (ukáže jen ten přidaný stín)
    plt.subplot(1, 3, 3)
    # Použijeme colomapu 'bwr' (blue-white-red), kde bílá je 0, modrá -, červená +
    # Tím uvidíš, kterou část obrazu to zesvětlilo a kterou ztmavilo
    plt.imshow(fix(diff_map), cmap="bwr", origin="lower", vmin=-0.5, vmax=0.5)
    plt.title("Vizualizace pole (Rozdíl)", fontsize=30)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_bias_field()