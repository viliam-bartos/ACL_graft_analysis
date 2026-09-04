# Rozhraní modulů a signatury funkcí (Interfaces)

Tento dokument je **závazným technickým předpisem** pro rozhraní mezi moduly v projektu `ACL_graft_analysis`. Definuje typové signatury, povinné argumenty, návratové hodnoty a vláknový model.

---

## 1. Modul `Source/main/mri_pipeline.py`

Hlavní orchestrátor celého výpočetního toku.

### 1.1 `process_single_volume`
Zpracuje jeden MRI sken (resampling, reorientace, laterality check, inference, postprocessing, anaknee výpočet).

```python
def process_single_volume(
    file_path: str,
    model: Optional[torch.nn.Module],
    device: torch.device,
    run_viz_at_end: bool = False,
    ensemble_models: Optional[List[torch.nn.Module]] = None,
    laterality_callback: Optional[Callable[[str], str]] = None,
) -> Optional[Dict[str, Any]]:
```

* **Parametry**:
  * `file_path`: Cesta k vstupnímu souboru (`.nii`, `.nii.gz` nebo `.dcm`).
  * `model`: Instance natrénovaného modelu `LightUNet3D` pro single-model režim (nebo `None`).
  * `device`: Výpočetní zařízení (`torch.device("cuda")` nebo `"cpu"`).
  * `run_viz_at_end`: Pokud je `True`, na konci se otevře interaktivní PyVista 3D okno.
  * `ensemble_models`: Seznam instancí modelů pro 5-Fold ansámblovou inferenci.
  * `laterality_callback`: Funkce vyvolaná v případě nejednoznačné laterality (např. GUI dialog), vracející `"Left"` nebo `"Right"`.
* **Návratová hodnota**:
  * Slovník `res_dict` obsahující všechny extrahované geometrické metriky a klíč `"Filename"`, nebo `None` při chybě.

---

## 2. Modul `Source/anaknee/main_acl_analysis.py`

Výpočetní jádro geometrických, topologických a radiomických parametrů.

### 2.1 `run_analysis`
Hlavní vstupní bod anatomické analýzy.

```python
def run_analysis(
    img_path: str, 
    ref_path: str, 
    mask_path: str
) -> Tuple[
    Dict[str, Any],                      # results_dict
    np.ndarray,                          # mask_array (RIA orientace)
    Tuple[float, float, float],          # spacing_zyx (sz, sy, sx)
    Tuple[float, float, float],          # f_centroid (femorální footprint v mm)
    Tuple[float, float, float],          # t_centroid (tibiální footprint v mm)
    Dict[str, Any]                       # plane_info (rovina plata, mřížky, ATT info)
]:
```

* **Struktura `results_dict`**:
  * Obsahuje všechny sloupce specifikované v [`spec/data-contracts.md`](file:///c:/ACL_analysis/ACL_graft_analysis/spec/data-contracts.md).
* **Struktura `plane_info`**:
  * `"normal"`: Jednotkový normálový vektor roviny tibiálního plata (`np.ndarray` délky 3) po robustním RANSAC fitu.
  * `"center"`: Fyzikální těžiště plata v mm (`np.ndarray` délky 3).
  * `"plateau_inliers"`: Fyzikální souřadnice bodů plata zařazených RANSACem jako inliery (`np.ndarray` tvaru $N \times 3$ nebo `None`).
  * `"plateau_outliers"`: Fyzikální souřadnice bodů plata označených jako outliery (`np.ndarray` tvaru $M \times 3$ nebo `None`).
  * `"bh_grid_info"`: Slovník pro vykreslení Bernard-Hertelovy mřížky (`lines`, `ref_edge`, `grid_origin`, `v_long`, `v_short`, `grid_length`, `grid_depth`).
  * `"att_info"`: Slovník bodů a vektorů pro zobrazení přední translace tibie.
  * `"staubli_info"`: Slovník bodů pro zobrazení Stäubliho úsečky na platu.

### 2.2 `run_geometric_analysis_from_mask`
Blesková geometrická analýza přímo ze segmentační masky (vynechává zdlouhavé histogramové párování a radiomiku). Doba běhu cca 1–2 s.

```python
def run_geometric_analysis_from_mask(
    mask_input: Union[str, sitk.Image, np.ndarray],
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[
    Dict[str, Any],                      # results_dict
    np.ndarray,                          # mask_array (RIA orientace)
    Tuple[float, float, float],          # spacing_zyx (sz, sy, sx)
    Tuple[float, float, float],          # f_centroid (femorální footprint v mm)
    Tuple[float, float, float],          # t_centroid (tibiální footprint v mm)
    Dict[str, Any],                      # plane_info
    Dict[str, Any]                       # vis_data připravená pro visualize_results
]:
```

---

## 3. Modul `Source/anaknee/visualizator_analyzator.py`

3D vizualizační jádro postavené na PyVista s prémiovým medicínským dark theme, seskupenými checkboxy a informačním panelem metrik.

### 3.1 `visualize_results`
Otevře interaktivní 3D okno s polyedrickými sítěmi kostí, štěpu, RANSAC mračny bodů plata a biomedicínskými osami.

```python
def visualize_results(
    mask_array: np.ndarray,
    spacing: Tuple[float, float, float],
    vis_data: Dict[str, Any]
) -> None:
```

### 3.2 `visualize_mri_volume`
Interaktivní 3D prohlížeč surových MRI dat / šedotónových objemů. Zobrazuje ortogonální řezy (axiální, sagitální, koronální), denzitní škálu, ohraničující box a fyzikální rozměry.

```python
def visualize_mri_volume(
    image_input: Union[str, sitk.Image, np.ndarray],
    spacing: Optional[Tuple[float, float, float]] = None,
    title: str = "MRI 3D Volume Viewer"
) -> None:
```

### 3.3 `smart_visualize`
Univerzální funkce schopná automaticky rozpoznat typ vstupního objemu (maska vs. MRI sken) a spustit odpovídající 3D zobrazení.

```python
def smart_visualize(
    primary_path: str,
    secondary_path: Optional[str] = None
) -> None:
```

---

## 4. Modul `Source/main/gui_app.py`

Uživatelské rozhraní postavené na CustomTkinter.

### 4.1 Architektura záložek
1. **`🚀 3D Prohlížeč (PyVista)`**:
   - Primární vstup pro výběr libovolného NIfTI/DICOM souboru s automatickou detekcí typu objemu.
   - Rychlé předvolby referenčních dat (1-klik načtení).
   - Tlačítko **`▶ OTEVŘÍT V PYVISTA 3D`** spouštějící vizualizaci na pozadí (`threading.Thread(daemon=True)`).
   - Rychlý panel naměřených hodnot (ATT, Stäubli, Úhel plata, B&H).
2. **`⚡ Dávková Analýza`**:
   - Zpracování jednotlivého souboru nebo celé složky pacienta.
   - Volba modulů (AI segmentace, geometrie, volitelná radiomika).
   - Průběhový ukazatel a rozbalitelná textová konzole.
3. **`📊 Výsledky & Případy`**:
   - Seznam zpracovaných případů s přímým tlačítkem **`👁 3D`** u každého řádku pro okamžité otevření v PyVista.
   - Graf trendu vybraného biomarkeru.
4. **`⚙ Nastavení`**:
   - Konfigurace cest k modelům a referenčním datům s kontrolou existence souborů.

---

## 5. Modul `Source/blackwell/WORKSTATION_BLACKWELL_MULTICLASS_5CV.py`

Architektura neuronové sítě.

### 4.1 `LightUNet3D`
```python
class LightUNet3D(torch.nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 4, base: int = 64) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

---

## 5. Vláknový model a GUI komunikace (`gui_app.py`)

1. **Izolace hlavního vlákna**: Veškerý GUI kód běží na hlavním vlákně Tkinter / CustomTkinter.
2. **Výpočetní vlákno**: Výpočetní pipeline `process_single_volume` a dávkové zpracování **musí** vždy běžet v odděleném vlákně (`threading.Thread(daemon=True)`), aby nedocházelo k zamrzání grafického rozhraní.
3. **Předávání událostí zpět do UI**: Jakákoliv aktualizace prvků GUI (progress bar, text v logu, otevření dialogu, zobrazení výsledků v Dashboardu) **musí** být předána přes `self.after(0, callback)`.
