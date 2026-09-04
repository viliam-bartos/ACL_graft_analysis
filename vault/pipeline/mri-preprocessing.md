# Předzpracování MRI a kanonizace

Tento dokument vysvětluje principy předzpracování obrazových dat, prostorové kanonizace a normalizace intenzit v pipeline `Source/main/mri_pipeline.py` a `Source/anaknee/main_acl_analysis.py`.

---

## 1. Zpracování vstupních formátů

Pipeline podporuje dva vstupní medicínské formáty:
* **DICOM (`.dcm` nebo složka se sérií)**:
  * Detekce pomocí `sitk.ImageSeriesReader.GetGDCMSeriesIDs`.
  * Pokud složka obsahuje DICOM řezy, dojde k jejich seřazení podle prostorové polohy (`ImagePositionPatient`) a bezeztrátovému převodu do NIfTI formátu (`.nii.gz`).
* **NIfTI (`.nii`, `.nii.gz`)**:
  * Standardní formát pro neurozobrazování a muskuloskeletální analýzu uchovávající prostorovou afinní matici (počátek, směr, spacing).

---

## 2. Izotropní resamplování na $0.5 \times 0.5 \times 0.5\,\text{mm}$

### Důvod
Klinické MRI sekvence kolene (např. 3D FSE, CUBE, SPACE nebo VIBE) mívají nepatrně odlišné rozlišení v závislosti na výrobci skeneru (např. $0.45 \times 0.45 \times 0.6\,\text{mm}$). Segmentační konvoluční síť a geometrické algoritmy (např. výpočet eukleidovských vzdáleností a zakřivení) však předpokládají přesně izotropní mřížku s krokem $0.5\,\text{mm}$.

### Algoritmus
* Pro zobrazení MRI intenzit se používá **B-spline interpolace 3. řádu** (`sitk.sitkBSpline`), která zachovává spojitost gradientů a zabraňuje vzniku schodovitých artefaktů na hranicích chrupavky a vazu.
* Nový rozměr mřížky se spočte jako:
  $$N_{\text{new}} = \text{round}\left(N_{\text{orig}} \cdot \frac{s_{\text{orig}}}{s_{\text{target}}}\right)$$
* Pro segmentační masky (při zpětné transformaci) se zásadně používá interpolace **nejbližšího souseda** (`sitk.sitkNearestNeighbor`), aby nedošlo ke vzniku neexistujících mezilehlých tříd.

---

## 3. Orientace a koordinátní prostory

### 3.1 Vynucení PIL orientace (nibabel)
V primární pipeline se vynucuje orientace **PIL** (Posterior, Inferior, Left):
* Osa 0: Anterior $\to$ Posterior
* Osa 1: Superior $\to$ Inferior
* Osa 2: Right $\to$ Left

### 3.2 Kanonická reorientace RIA v Anaknee
Při detailní geometrické analýze v modulu `anaknee` se obraz převádí do orientace **RIA** (Right, Inferior, Anterior):
* `dim 0`: Left $\to$ Right (latero-mediální osa kolene)
* `dim 1`: Superior $\to$ Inferior (kraniokaudální osa končetiny)
* `dim 2`: Posterior $\to$ Anterior (předozadní osa kolene)

Tato orientace zajišťuje pevné ukotvení anatomických předpokladů v kódu (např. sagitální řez je řezem podél `dim 0`).

---

## 4. Detekce laterality a zrcadlení končetiny

### 4.1 Proč zrcadlíme pravá kolena?
Trénovací data a anatomické atlasy bývají unifikovány na jednu stranu (zde prostor **levého kolene**). Pokud bychom trénovali síť na směsi levých a pravých kolen bez rozlišení, vaz by směřoval v koronální rovině pod opačným úhlem, což by zbytečně mátlo konvoluční filtry a zhoršovalo přesnost predikce.

### 4.2 Detekce laterality
Systém vyhodnocuje název souboru pomocí regulárních výrazů:
* **Pravé koleno (Right)**: vzory `right`, `dexter`, `dext`, `dx`, `rt`, `prav...`
* **Levé koleno (Left)**: vzory `left`, `sinister`, `sinist`, `sin`, `lt`, `lev...`
* **Fallback**: Pokud název neobsahuje jednoznačný identifikátor, GUI vyzve uživatele k manuálnímu výběru (nebo se v dávkovém běhu loguje varování).

### 4.3 Transformační cyklus
1. Pokud je koleno **Right**:
   * Provede se zrcadlení obrazu:
     $$\mathbf{A}_{\text{mirrored}} = \text{flip}(\mathbf{A}, \text{axis}=0)$$
   * Tím se z pravého kolene stane topologicky levé koleno.
2. Provede se segmentace unifikovaným modelem pro levé koleno.
3. Výsledná maska se invertuje zpět:
   $$\mathbf{M}_{\text{patient}} = \text{flip}(\mathbf{M}_{\text{pred}}, \text{axis}=0)$$
4. Maska se resampluje do původního nativního prostoru pacienta.

---

## 5. Normalizace a standardizace intenzit

Pipeline aplikuje dvoustupňovou normalizaci:

### 5.1 Z-Score s ořezáním percentilů (pro AI inferenci)
1. Intenzity se oříznou na percentily $[0.5, 99.5]$ pro eliminaci extrémních artefaktů magnetického pole.
2. Nenulové voxely se normalizují:
   $$I_{\text{norm}}(\mathbf{x}) = \frac{I(\mathbf{x}) - \mu}{\sigma + 10^{-8}}$$

### 5.2 Nyul-Udupa histogramová standardizace (pro Radiomiku)
Pro spolehlivou extrakci radiomických parametrů (GLCM, GLRLM) nelze spoléhat na prosté Z-score, protože MRI nemá fixní Hounsfieldovy jednotky jako CT. 
* Používá se knihovna `TorchIO` (`HistogramStandardization`).
* Natrénují se kotevní kvantily (landmarks) z referenčního MRI skenu (`right_case_074.nii.gz`).
* Distribuce šedotónových hodnot analyzovaného pacienta se nelineárně přemapuje na tuto referenci, což eliminuje meziskenerovou variabilitu.
