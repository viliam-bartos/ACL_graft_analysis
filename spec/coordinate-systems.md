# Souřadnicové systémy a prostorové transformace

Tento dokument je **závazným technickým předpisem** pro koordinátní prostory, orientace os, přepočty mezi voxelovými a fyzikálními jednotkami a pravidla zrcadlení končetin.

---

## 1. Fyzikální vs Voxelové souřadnice

Každý bod v kolenním kloubu může být vyjádřen ve dvou systémech:

### 1.1 Voxelové indexy (Discrete Grid Coordinates)
* Celočíselné nebo sub-voxelové plovoucí indexy v poli: $(i_0, i_1, i_2)$.
* V NumPy poli odpovídají indexům `arr[i_0, i_1, i_2]`.

### 1.2 Fyzikální souřadnice (Physical Coordinates v mm)
* Přepočet z voxelových indexů na fyzikální metrické souřadnice:
  $$\mathbf{p}_{\text{phys}} = (i_0 \cdot s_0, \; i_1 \cdot s_1, \; i_2 \cdot s_2)$$
  kde $(s_0, s_1, s_2)$ je `spacing_zyx = (sz, sy, sx)`.

> [!IMPORTANT]
> Všechny klinické výstupy (vzdálenosti, úhly, posuny ATT v mm, šířka fossy v mm a objemy v $\text{mm}^3$) **musí** být počítány výhradně ve fyzikálních souřadnicích s korekcí o spacing, nikoliv v počtech voxelů!

---

## 2. Kanonické orientační kódy (Orientation Codes)

V projektu se striktně rozlišují dvě fáze orientace:

### 2.1 Fáze A: PIL orientace (Inference Pipeline)
V souboru `Source/main/mri_pipeline.py` se NIfTI objem převádí funkcí `force_reorient_pil` do orientace **PIL**:
* **P** – Posterior (osa roste od přední strany k zadní)
* **I** – Inferior (osa roste od hlavy k nohám)
* **L** – Left (osa roste zprava doleva pacienta)

Tato orientace odpovídá geometrické definici natrénovaného modelu `LightUNet3D`.

### 2.2 Fáze B: RIA orientace (Anaknee Geometrie)
Před spuštěním analytických algoritmů v `Source/anaknee/main_acl_analysis.py` se obraz i maska kanonizují funkcí `_reorient_to_ria` do orientace **RIA**:
* **Dim 0 (R)**: Right (osa roste zleva doprava: $L \to R$) $\to$ **Latero-mediální osa**
* **Dim 1 (I)**: Inferior (osa roste shora dolů: $S \to I$) $\to$ **Kraniokaudální osa**
* **Dim 2 (A)**: Anterior (osa roste zezadu dopředu: $P \to A$) $\to$ **Předozadní (AP) osa**

#### Důsledky pro kód:
1. **Sagitální řez** leží v rovině kolmé na latero-mediální osu:
   ```python
   sag_slice = femur_mask[dim0, :, :]  # dim0 odpovídá latero-mediální poloze řezu
   ```
2. **Globální AP směr** odpovídá bázovému vektoru:
   $$\mathbf{v}_{\text{AP\_global}} = [0.0, 0.0, 1.0]$$
3. **Vertikální směr dolů (inferior)**:
   $$\mathbf{v}_{\text{down}} = [0.0, 1.0, 0.0] \quad \implies \quad \mathbf{v}_{\text{up}} = [0.0, -1.0, 0.0]$$

---

## 3. Pravidlo zrcadlení pravého kolene (Laterality Mirroring)

Pro zaručení invariance vůči straně těla se pravá kolena zrcadlí do prostoru levého kolene.

### 3.1 Transformační matice
Zrcadlení se provádí podél nulté osy NumPy pole (odpovídá prostorové ose zrcadlení):
$$\mathbf{A}_{\text{left\_space}} = \text{np.flip}(\mathbf{A}_{\text{right\_native}}, \text{axis}=0)$$

Formálně odpovídá afinní transformaci:
$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

### 3.2 Zpětné odzrcadlení predikované masky
Po dokončení inference a morfologického postprocessingu **musí** být maska neprodleně vrácena do nativního prostoru pacienta:
$$\mathbf{M}_{\text{right\_native}} = \text{np.flip}(\mathbf{M}_{\text{left\_space}}, \text{axis}=0)$$

### 3.3 Zachování metadat SimpleITK
Při manipulaci s polem přes `sitk.GetImageFromArray` a `np.flip` dochází ke ztrátě hlavičky. Kód **musí** provést obnovení metadat:
```python
working_sitk = sitk.GetImageFromArray(arr)
working_sitk.CopyInformation(ref_meta_sitk)
```
Následně se maska převzorkuje na referenční nativní sken pomocí `sitk.ResampleImageFilter` s interpolací `sitkNearestNeighbor`.
