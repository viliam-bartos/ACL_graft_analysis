# Vault – Kontextová paměť projektu ACL Graft Analysis

Vítejte ve **Vaultu** projektu *Automated ACL Segmentation and 3D Geometric Analysis*. 

Adresář `vault/` slouží jako dlouhodobá kontextová paměť pro vývojáře, výzkumníky a klinické partnery. Obsahuje teoretické základy, anatomická pravidla, matematické rovnice, vysvětlení biomedicínských biomarkerů a vědeckou metodiku, na které je software vystavěn.

> [!NOTE]
> Zatímco `vault/` odpovídá na otázku **PROČ a CO** (vědecký a doménový kontext), adresář [`spec/`](file:///c:/ACL_analysis/ACL_graft_analysis/spec/README.md) odpovídá na otázku **JAK PŘESNĚ** (závazné technické předpisy, datové kontrakty a rozhraní).

---

## Struktura a navigace

### 1. Anatomie a biomechanika
* [`anatomie/acl.md`](file:///c:/ACL_analysis/ACL_graft_analysis/vault/anatomie/acl.md): Anatomie předního zkříženého vazu (*Ligamentum cruciatum anterius*), femuru, tibie, inserčních stop (footprints) a specifika pooperačního štěpu (graftu).
* [`anatomie/biomedical-metrics.md`](file:///c:/ACL_analysis/ACL_graft_analysis/vault/anatomie/biomedical-metrics.md): Podrobný rozbor a odvození klinických biomarkerů:
  * Bernard-Hertel mřížka (femorální pozice štěpu)
  * Stäubli procento (tibiální pozice úponu)
  * Anterior Tibial Translation (ATT – přední subluxace tibie)
  * Tortuosity Index (zvlnění a laxita vazu)
  * Úhly elevace, sagitální a koronální sklony
  * Šířka interkondylární fossy a impingement vzdálenost
  * Radiomické příznaky (first-order, GLCM, GLRLM)

### 2. Zpracování obrazu a kanonizace
* [`pipeline/mri-preprocessing.md`](file:///c:/ACL_analysis/ACL_graft_analysis/vault/pipeline/mri-preprocessing.md): Metodika předzpracování izotropních 3D MRI skenů kolenního kloubu:
  * DICOM $\to$ NIfTI standardizace
  * Izotropní resamplování na $0.5 \times 0.5 \times 0.5\,\text{mm}$ (B-spline interpolace)
  * Orientace prostoru (PIL vs RIA)
  * Detekce laterality a zrcadlení pravého kolene do prostoru levého kolene
  * Standardizace intenzit (Nyul-Udupa histogram matching)

### 3. Hluboké učení a AI
* [`ai/segmentation-models.md`](file:///c:/ACL_analysis/ACL_graft_analysis/vault/ai/segmentation-models.md): Principy segmentačního modelu:
  * 3D konvoluční architektura `LightUNet3D`
  * 5-Fold Cross-Validation ansámbl (průměrování predikčních pravděpodobností)
  * Asymetrické per-class prahování
  * Morfologický post-processing (LCC, fill holes, closing)
