# Biomedicínské a geometrické metriky

Tento dokument detailně popisuje matematické definice, geometrická odvození a klinickou interpretaci všech metrik extrahovaných modulem `anaknee` (`Source/anaknee/main_acl_analysis.py`).

---

## 1. Bernard-Hertel mřížka (Femorální úpon)

Bernard-Hertelova mřížka (*Bernard & Hertel quadrant method*) je mezinárodně uznávaná radiologická metoda pro hodnocení polohy femorálního úponu na sagitálním rentgenu či MRI řezu.

### 1.1 Detekce Blumensaatovy linie
1. Identifikuje se sagitální řez středem interkondylární fossy (index $Z_{\text{slice}} = \text{round}(Z_{\text{ACL\_centroid}})$).
2. V oblasti interkondylární střechy se provede ray-casting podél osy $Z$ pro nalezení kortikální hrany kosti.
3. Body hrany $\{(x_i, y_i)\}$ se proloží přímkou pomocí lineární regrese:
   $$y = a \cdot x + b$$
   Tato úsečka definuje **Blumensaatovu linii** o délce $L_{\text{Blum}} = \|\mathbf{p}_2 - \mathbf{p}_1\|$.

### 1.2 Báze souřadného systému mřížky
Definují se dva ortonormální vektory:
* $\mathbf{v}_{\text{long}}$: jednotkový vektor podél Blumensaatovy linie (anteriorně/posteriorně).
* $\mathbf{v}_{\text{short}}$: jednotkový vektor kolmý na $\mathbf{v}_{\text{long}}$ směřující distálně (dolů ke kondylu):
  $$\mathbf{v}_{\text{short}} = [0, -\mathbf{v}_{\text{long}}[2], \mathbf{v}_{\text{long}}[1]]$$

### 1.3 Ohraničení kondylu a výpočet procent
Voxely laterálního femorálního kondylu se promítnou do této báze, čímž vznikne obdélník o celkové délce $L_{\text{grid}}$ a hloubce $D_{\text{grid}}$.

Pro těžiště femorálního úponu $\mathbf{c}_{\text{femur}}$ se spočte vektor relativního posunu vůči počátku mřížky $\mathbf{g}_0$:
$$\mathbf{d} = \mathbf{c}_{\text{femur}} - \mathbf{g}_0$$
$$\text{proj}_{\text{long}} = \mathbf{d} \cdot \mathbf{v}_{\text{long}}, \quad \text{proj}_{\text{short}} = \mathbf{d} \cdot \mathbf{v}_{\text{short}}$$

* **`BH_Length_pct` (%)**: Pozice podél Blumensaatovy linie (0 % = nejvíce dorzální/zadní okraj, 100 % = přední okraj):
  $$\text{BH\_Length\_pct} = \frac{\text{proj}_{\text{long}}}{L_{\text{grid}}} \times 100\%$$
* **`BH_Depth_pct` (%)**: Pozice do hloubky kondylu (0 % = na Blumensaatově linii, 100 % = nejnižší distální okraj kondylu):
  $$\text{BH\_Depth\_pct} = \frac{\text{proj}_{\text{short}}}{D_{\text{grid}}} \times 100\%$$

*Klinická reference nativního ACL*: cca 25–30 % délky a 30–35 % hloubky.

---

## 2. Stäubliho tibiální procento (Staubli Tibial Percentage)

Metodika dle Stäubli et al. hodnotí sagitální předozadní polohu tibiálního úponu ACL v poměru k celkovému AP rozměru tibiálního plata.

### Matematické odvození
1. Zvolí se sagitální řez procházející těžištěm tibiálního úponu $\mathbf{c}_{\text{tibia}}$.
2. Určí se horizontální vektor $\mathbf{v}_{\text{horiz}}$ ležící v sagitálním řezu a rovnoběžný s rovinou tibiálního plata:
   $$\mathbf{v}_{\text{horiz}} = \frac{\mathbf{n}_{\text{plateau}} \times [1, 0, 0]}{\|\mathbf{n}_{\text{plateau}} \times [1, 0, 0]\|}$$
   Vektor je orientován anteriorně.
3. Body proximální části tibie (horních $20\,\text{mm}$) se promítnou na $\mathbf{v}_{\text{horiz}}$:
   $$\text{Ant\_Edge} = \max(\mathbf{p}_i \cdot \mathbf{v}_{\text{horiz}}), \quad \text{Post\_Edge} = \min(\mathbf{p}_i \cdot \mathbf{v}_{\text{horiz}})$$
   $$\text{Total\_AP} = \text{Ant\_Edge} - \text{Post\_Edge}$$
4. Projekce těžiště úponu $\mathbf{c}_{\text{tibia}}$ dává Stäubliho procento:
   $$\text{Staubli\_Tibial\_pct} = \frac{\text{Ant\_Edge} - (\mathbf{c}_{\text{tibia}} \cdot \mathbf{v}_{\text{horiz}})}{\text{Total\_AP}} \times 100\%$$

*Klinická interpretace*:
* $0\,\%$ odpovídá přednímu kortikálnímu okraji plata.
* $100\,\%$ odpovídá zadnímu kortikálnímu okraji plata.
* *Fyziologické rozmezí nativního ACL*: přibližně $40\,\% - 44\,\%$.

---

## 3. Přední translace tibie (Anterior Tibial Translation – ATT)

ATT měří přední patologický posun tibie vůči femuru (subluxaci) vyjádřený v milimetrech.

### Algoritmus výpočtu
1. Stanoví se globální přední vektor $\mathbf{v}_{\text{ant}}$ promítnutý do roviny tibiálního plata:
   $$\mathbf{v}_{\text{ant}} = \mathbf{a}_{\text{global}} - (\mathbf{a}_{\text{global}} \cdot \mathbf{n}_{\text{plateau}})\,\mathbf{n}_{\text{plateau}}$$
2. Pro voxely laterálního kondylu femuru se nalezne nejvíce dorzální bod:
   $$E_{\text{femur}} = \min_{i}(\mathbf{p}_{\text{femur}, i} \cdot \mathbf{v}_{\text{ant}})$$
3. Pro voxely tibie se nalezne nejvíce dorzální bod:
   $$E_{\text{tibia}} = \min_{j}(\mathbf{p}_{\text{tibia}, j} \cdot \mathbf{v}_{\text{ant}})$$
4. ATT je rozdíl těchto extrémních zadních pozic:
   $$\text{ATT\_mm} = E_{\text{tibia}} - E_{\text{femur}}$$

*Interpretace*:
* $\text{ATT\_mm} > 0$: Tibia je posunuta dopředu oproti femuru (pozitivní zásuvkový fenomén).
* Rozdíl mezi operovaným a zdravým kolenem $> 3\,\text{mm}$ klinicky indikuje nestabilitu nebo selhání vazu.

---

## 4. Index tortuozity (Tortuosity Index)

Index tortuozity $\tau$ kvantifikuje zakřivení a zvlnění průběhu vazu.

### Matematická definice
Pro každý aktivní řez $y$ vazu se spočte dílčí těžiště $\mathbf{c}(y) = [z_c, y, x_c] \cdot \mathbf{s}$. Seřazením těchto středových bodů vznikne prostorová křivka $\mathcal{C} = \{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_K\}$.

$$\text{Tortuosity\_Index} = \frac{L_{\text{curved}}}{L_{\text{straight}}} = \frac{\sum_{k=1}^{K-1} \|\mathbf{c}_{k+1} - \mathbf{c}_k\|}{\|\mathbf{c}_{\text{femur}} - \mathbf{c}_{\text{tibia}}\|}$$

*Hodnoty*:
* $\tau = 1.0$: Dokonale přímý vaz bez jakéhokoliv zakřivení.
* $\tau \in [1.05, 1.15]$: Fyziologicky napnutý vaz.
* $\tau > 1.25$: Výrazné zvlnění vazu – typické pro laxní štěp, parciální rupturu nebo ztrátu biomechanického tahu.

---

## 5. Rovina tibiálního plata (RANSAC Plateau Plane Fitting)

Rovina tibiálního plata je primární geometrickou referencí pro:
* výpočet elevačního úhlu vazu (`angle_to_plateau_deg`),
* stanovení sagitální horizontály pro Stäubliho procento,
* definici tečné roviny pro měření ATT.

### 5.1 Anatomická úskalí a motivace pro RANSAC
Tibiální plato není dokonale ploché. Obsahuje výrazné anatomické vyvýšeniny a patologie:
1. **Eminentia intercondylaris (tibial spines)**: Centrální kostní hroty sahající vysoko nad úroveň kloubní plochy.
2. **Osteofyty a kostní apozice**: U osteoartrózy či chronické léze ACL lemují okraje plata.
3. **Konkávnost/konvexnost kondylů**: Mediální plato je lehce konkávní, laterální konvexní se zadním sklonem (posterior tibial slope).

Pouhé fitování metodou nejmenších čtverců (SVD/PCA) je na tyto outliery extrémně citlivé, což vede k nepřirozenému naklopení normály $\mathbf{n}_{\text{plateau}}$.

### 5.2 Algoritmus RANSAC (`PlaneModel3D`)
1. **Výběr proximálních voxelů**: Iterativně se vybere horních $25\,\%$ voxelů tibie ve směru k femuru.
2. **Model roviny**: Rovina je parametrizována bodem $\mathbf{c}$ (centroid vzorku) a jednotkovou normálou $\mathbf{n}$.
   $$(\mathbf{x} - \mathbf{c}) \cdot \mathbf{n} = 0$$
3. **RANSAC proces (`skimage.measure.ransac`)**:
   * Minimální počet bodů pro vzorek: `min_samples = 3` (s kontrolou kolinearity přes singulární čísla $s_1 / s_0 > 10^{-10}$).
   * Prahová vzdálenost reziduí: `residual_threshold = 1.5 mm` (odpovídá tloušťce kortikální kosti plata).
   * Maximální počet iterací: `max_trials = 1000`.
4. **Kritéria úspěšnosti a SVD fallback**:
   * Model je akceptován, pokud podíl inlierů překročí $30\,\%$ z proximálních bodů.
   * V opačném případě (nebo při méně než 10 bodech) algoritmus bezpečně přechází na robustní SVD fallback.
5. **Orientace normály**: Vektor $\mathbf{n}_{\text{plateau}}$ je orientován tak, aby skalární součin se spojnicí k femuru byl kladný (směr kraniálně/proximálně).

---

## 6. Prostorová orientace a úhly vazu

Centrální vektor vazu je definován od tibiálního k femorálnímu úponu:
$$\mathbf{v}_{\text{ACL}} = \frac{\mathbf{c}_{\text{femur}} - \mathbf{c}_{\text{tibia}}}{\|\mathbf{c}_{\text{femur}} - \mathbf{c}_{\text{tibia}}\|}$$

### 6.1 Úhel k rovině plata (`angle_to_plateau_deg`)
Měří elevaci vazu vůči rovině tibiálního plata $\mathbf{n}_{\text{plateau}}$:
$$\theta_{\text{normal}} = \arccos(|\mathbf{v}_{\text{ACL}} \cdot \mathbf{n}_{\text{plateau}}|)$$
$$\text{angle\_to\_plateau\_deg} = 90^\circ - \theta_{\text{normal}}$$

### 6.2 Sagitální úhel (`sagittal_angle_deg`)
Projekce do sagitální roviny (A-S rovina, $X = 0$).

### 6.3 Koronální úhel (`coronal_angle_deg`)
Projekce do frontální/koronální roviny (R-S rovina, $Z = 0$).

---

## 7. Impingement a šířka interkondylární fossy

### 7.1 Minimální vzdálenost od femuru (`min_dist_to_femur_mm`)
Počítá se pomocí eukleidovské vzdálenostní transformace inverzní masky femuru:
$$\text{EDT}(\mathbf{x}) = \min_{\mathbf{y} \in \text{Femur}} \|\mathbf{x} - \mathbf{y}\|_2$$
$$\text{min\_dist\_to\_femur\_mm} = \min_{\mathbf{x} \in \text{ACL}} \text{EDT}(\mathbf{x})$$
Vzdálenost $0.0\,\text{mm}$ indikuje přímý kontakt nebo zaškrcení vazu v zářezu.

### 7.2 Šířka fossy (`notch_width_mm`)
Měřena paprskem podél latero-mediální osy procházejícím středem ACL tělesa:
$$\text{notch\_width\_mm} = (x_{\text{right\_wall}} - x_{\text{left\_wall}}) \cdot s_x$$

---

## 8. Radiomické příznaky (PyRadiomics)

Příznaky jsou počítány výhradně na masce vazu (Label 1) po histogramové standardizaci intenzit:
1. **First Order Statistics**: Průměr, rozptyl, šikmost (skewness), špičatost (kurtosis), energie a entropie distribuce intenzit voxelů.
2. **GLCM (Gray Level Co-occurrence Matrix)**: Kontrast, korelace, homogenita, disimilarita – popisují jemnou texturu a vnitřní homogenitu kolagenních vláken štěpu.
3. **GLRLM (Gray Level Run Length Matrix)**: Délky sekvencí stejných úrovní šedi – popisují směrovou soudržnost vláken.
