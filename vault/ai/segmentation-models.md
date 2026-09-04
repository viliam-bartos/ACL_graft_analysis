# Segmentační modely a hluboké učení

Tento dokument popisuje architekturu hluboké neuronové sítě, trénovací strategii, princip ansámblové inference a morfologické post-processingové operace.

---

## 1. Architektura `LightUNet3D`

Pro segmentaci 3D objemů kolenního kloubu se využívá plně trojrozměrná konvoluční síť `LightUNet3D` (implementovaná v modulu `blackwell`).

### Klíčové parametry
* **Vstup**: Jednokanálový 3D tenzor $(B, 1, D, H, W)$ normalizovaných intenzit.
* **Výstup**: 4-třídový logitový tenzor $(B, 4, D, H, W)$:
  * Třída 0: Pozadí (Background)
  * Třída 1: Přední zkřížený vaz (ACL / Graft)
  * Třída 2: Kost stehenní (Femur)
  * Třída 3: Kost holenní (Tibia)
* **Základní počet filtrů**: `base_filters = 64`
* **Reziduální spoje**: Urychlují konvergenci a brání vymizení gradientu v hlubokých vrstvách.

---

## 2. Sliding Window Inference (MONAI)

Vzhledem k vysokému rozlišení izotropního 3D skenu nelze celý objem najednou umístit do paměti GPU. Proto se využívá technika posuvného okna (`sliding_window_inference`):
* **Velikost patchů (ROI)**: $(128, 128, 80)$ voxelů.
* **Překryv (Overlap)**: $50\,\%$ (`overlap = 0.5`).
* **Váhové prolnutí (Blending mode)**: `gaussian` – predikce na okrajích okna mají nižší váhu než uprostřed, což eliminuje švy a diskontinuity na hranicích patchů.
* **Akcelerace**: `torch.amp.autocast("cuda", dtype=torch.bfloat16)` pro úsporu VRAM a maximální propustnost na moderních GPU architekturách (Nvidia Blackwell / Ada / Ampere).

---

## 3. 5-Fold Cross-Validation Ansámbl

Pro dosažení maximální robustnosti a generalizace napříč různými patologiemi a tvary kolen pipeline preferenčně využívá ansámbl 5 modelů:
$$\mathbf{P}_{\text{ensemble}}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} \text{Softmax}(\mathbf{z}_k(\mathbf{x}))$$
kde $K = 5$ a $\mathbf{z}_k$ jsou logity jednotlivých modelů z 5-násobné křížové validace (`best_model_fold_*.pth`).

Ansámblování radikálně snižuje rozptyl predikce (variance reduction) a eliminuje falešně pozitivní izolované shluky voxelů.

---

## 4. Asymetrické per-class prahování

Standardní `argmax` rozhodování není pro anatomické struktury s dramaticky odlišným objemem optimální. Model proto aplikuje diferencované prahy pravděpodobností:

```python
pred[(pred_argmax == 1) & (probs[1] >= 0.45)] = 1  # ACL (Graft)
pred[(pred_argmax == 2) & (probs[2] >= 0.90)] = 2  # Femur
pred[(pred_argmax == 3) & (probs[3] >= 0.80)] = 3  # Tibia
```

### Racionální důvod
* **ACL (práh 0.45)**: Tenký, šikmo orientovaný vaz s objemem pouze cca $1000 - 2500\,\text{mm}^3$. Nižší práh brání vzniku přetržených "děr" ve vazu při částečném objemovém efektu (partial volume effect).
* **Kosti (prahy 0.90 a 0.80)**: Velké kompaktní struktury s jasným kontrastem vůči chrupavce a měkkým tkáním. Vysoký práh zabraňuje přetečení kosti do kloubní štěrbiny.

---

## 5. Morfologický post-processing

Po získání diskrétní masky se aplikuje specializované morfologické čištění pro každou třídu zvlášť:

### ACL (Třída 1)
* **LCC (Largest Connected Component)**: Ponechá pouze největší spojitou komponentu. Odstraňuje drobné falešně pozitivní artefakty ve svalech či meniscích.
* **Hole Filling**: Vypnuto (`False`), aby nedošlo k umělému slití vazu se zadním zkříženým vazem (PCL).
* **Closing**: Vypnuto (`False`), aby se zachovaly jemné obrysy inserčních stop.

### Femur a Tibia (Třídy 2 a 3)
* **LCC**: Ponechá hlavní tělo kosti.
* **Hole Filling**: Zapnuto (`True`). Vyplňuje vnitřek kosti (kostní dřeň / spongiózu), která může mít na T1/T2 zobrazení odlišný signál než kortikalis.
* **Morphological Closing**: Zapnuto (`True`, jádro 2). Vyhlazuje drobné nerovnosti kortikálního povrchu před proložením rovin a výpočtem geometrických os.
