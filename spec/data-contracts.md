# Datové kontrakty (Data Contracts)

Tento dokument je **závazným technickým předpisem** pro datové typy, paměťové tvary polí, kódování segmentačních štítků a strukturu výstupních souborů v projektu.

---

## 1. Kódování segmentačních tříd (Label Encoding)

Všechny segmentační masky (`mask_*.nii.gz`) a odvozená NumPy pole **musí** striktně dodržovat následující přiřazení celočíselných štítků:

| Label ID | Anatomická struktura | Anglický název | Datový typ masky | Povolené hodnoty |
| :---: | :--- | :--- | :---: | :---: |
| `0` | Pozadí / ostatní tkáně | Background | `uint8` | `0` nebo `1` (binárně) |
| `1` | Přední zkřížený vaz / štěp | Anterior Cruciate Ligament (ACL / Graft) | `uint8` | `0` nebo `1` (binárně) |
| `2` | Kost stehenní | Femur | `uint8` | `0` nebo `1` (binárně) |
| `3` | Kost holenní | Tibia | `uint8` | `0` nebo `1` (binárně) |

> [!WARNING]
> Žádný modul nesmí renumberovat nebo slučovat tyto štítky bez explicitní konverzní vrstvy. Hodnoty vyšší než 3 jsou v masce neplatné.

---

## 2. Tvary polí a konvence dimenzí (Array Shapes & Dtypes)

Projekt propojuje knihovny **SimpleITK**, **Nibabel**, **NumPy** a **PyTorch**. Přechod mezi nimi se řídí následujícími striktními pravidly:

### 2.1 SimpleITK $\leftrightarrow$ NumPy konvence
* **SimpleITK Image**:
  * Velikost: `(SizeX, SizeY, SizeZ)`
  * Spacing: `(SpacingX, SpacingY, SpacingZ)` v milimetrech
  * Pixel Type pro MRI: `sitk.sitkFloat32` (nebo `sitkInt16`)
  * Pixel Type pro masky: `sitk.sitkUInt8`
* **NumPy Array (`sitk.GetArrayFromImage(img)`)**:
  * Tvar pole: **`(DimZ, DimY, DimX)`** (obrácené pořadí dimenzí!)
  * Indexace: `arr[z, y, x]`
  * Dtype pro intenzity: `np.float32`
  * Dtype pro masky: `np.uint8` nebo `bool`

### 2.2 PyTorch Inference konvence
* **Vstup do modelu (`sliding_window_inference`)**:
  * Před vstupem do PyTorch se NumPy pole transponuje:
    ```python
    # z (DimZ, DimY, DimX) na (DimX, DimY, DimZ)
    img_array = np.transpose(img_array, (2, 1, 0))
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, X, Y, Z)
    ```
  * Velikost ROI (patch): `(128, 128, 80)` typu `int`
  * Dtype výpočtu: `torch.bfloat16` (autocast na CUDA)
* **Výstup pravděpodobnostní mapy**:
  * Tvar před transpozicí: `probs` o rozměru `(4, X, Y, Z)`
  * Výsledná predikce se po aplikaci prahů transponuje zpět:
    ```python
    pred_np = np.transpose(pred.cpu().numpy(), (2, 1, 0))  # Výsledný tvar: (Z, Y, X)
    ```

---

## 3. Prahovací vektor pravděpodobností (Thresholding Contract)

Rozhodovací pravidlo pro přiřazení třídy z výstupního tenzoru pravděpodobností $\mathbf{P} \in [0, 1]^{4 \times X \times Y \times Z}$:

```python
pred_argmax = torch.argmax(probs, dim=0)
pred = torch.zeros_like(pred_argmax, dtype=torch.uint8)

pred[(pred_argmax == 1) & (probs[1] >= 0.45)] = 1  # ACL / Graft
pred[(pred_argmax == 2) & (probs[2] >= 0.90)] = 2  # Femur
pred[(pred_argmax == 3) & (probs[3] >= 0.80)] = 3  # Tibia
```

| Třída | Podmínka argmax | Minimální pravděpodobnost | Důvod |
| :--- | :---: | :---: | :--- |
| **ACL** | `1` | $\ge 0.45$ | Prevence falešné diskontinuity tenkého svazku vazu |
| **Femur** | `2` | $\ge 0.90$ | Potlačení falešných detekcí v kloubní chrupavce |
| **Tibia** | `3` | $\ge 0.80$ | Přísná hranice na tibiálním platu |

---

## 4. Spacing a izotropní mřížka

* **Požadovaný spacing**: Cílový voxel spacing je **přesně** `(0.5, 0.5, 0.5)` mm.
* **Tolerance**: Pokud `np.allclose(orig_spacing, (0.5, 0.5, 0.5), atol=1e-3)`, resamplování se přeskakuje.
* **Interpolátor pro MRI**: `sitk.sitkBSpline`
* **Interpolátor pro masky**: `sitk.sitkNearestNeighbor`

---

## 5. Schéma souboru `patient_results.csv`

Soubor `patient_results.csv` je hlavním strukturovaným výstupem analýzy pacientů. Každý řádek odpovídá jednomu zpracovanému vyšetření.

| Název sloupce | Typ | Jednotka | Povolené hodnoty / Rozsah | Popis |
| :--- | :---: | :---: | :---: | :--- |
| `Filename` | `str` | - | platný název `.nii.gz` souboru | Primární klíč záznamu |
| `Staubli_Tibial_pct` | `float` | % | `0.0` až `100.0` (nebo `NaN`) | Relativní AP poloha tibiálního úponu |
| `Tortuosity_Index` | `float` | - | $\ge 1.0$ (nebo `NaN`) | Index zakřivení/zvlnění vazu ($\ge 1.0$) |
| `ATT_mm` | `float` | mm | obvykle $-15.0$ až $+25.0$ | Přední translace tibie vůči femuru |
| `BH_Length_pct` | `float` | % | `0.0` až `100.0` (nebo `NaN`) | Bernard-Hertel délkové procento úponu |
| `BH_Depth_pct` | `float` | % | `0.0` až `100.0` (nebo `NaN`) | Bernard-Hertel hloubkové procento úponu |
| `angle_to_plateau_deg` | `float` | stupně (°) | `0.0` až `90.0` (nebo `NaN`) | Úhel elevace ACL k rovině plata |
| `sagittal_angle_deg` | `float` | stupně (°) | `0.0` až `90.0` (nebo `NaN`) | Úhel ACL v sagitální rovině |
| `coronal_angle_deg` | `float` | stupně (°) | `0.0` až `90.0` (nebo `NaN`) | Úhel ACL v koronální rovině |
| `acl_volume_mm3` | `float` | $\text{mm}^3$ | $> 0.0$ | Fyzikální objem vazu |
| `min_dist_to_femur_mm` | `float` | mm | $\ge 0.0$ | Minimální vzdálenost od stěny femuru |
| `notch_width_mm` | `float` | mm | $> 0.0$ (nebo `NaN`) | Šířka interkondylární fossy |
| `original_firstorder_*` | `float` | různé | libovolné reálné číslo | First-order radiomické parametry |
| `original_glcm_*` | `float` | různé | libovolné reálné číslo | Texturální parametry GLCM |
| `original_glrlm_*` | `float` | různé | libovolné reálné číslo | Směrové texturální parametry GLRLM |

> [!NOTE]
> Pokud se pro daného pacienta nepodaří některou metriku spočítat (např. prázdná maska vazu nebo chybějící femur), hodnota **musí** být uložena jako `np.nan` (`float`), nikoliv jako řetězec nebo nula.
