# Spec – Závazný technický předpis projektu

Adresář `spec/` představuje **závaznou technickou specifikaci** pro celou kódovou bázi projektu *Automated ACL Segmentation and 3D Geometric Analysis*.

Každá úprava existujícího kódu, refaktoring, nový modul nebo externí skript přistupující k datům či funkcím projektu **musí striktně splňovat předpisy definované v těchto dokumentech**.

---

## Přehled technických specifikací

| Specifikace | Účel a rozsah |
| :--- | :--- |
| [`spec/data-contracts.md`](file:///c:/ACL_analysis/ACL_graft_analysis/spec/data-contracts.md) | **Datové kontrakty**: Názvy a kódování tříd, tvary polí (array shapes), datové typy (dtypes), paměťové konvence (C-order vs F-order), prahovací vektory, schema `patient_results.csv`. |
| [`spec/coordinate-systems.md`](file:///c:/ACL_analysis/ACL_graft_analysis/spec/coordinate-systems.md) | **Souřadnicové systémy a prostorové transformace**: Přepočet indexů na fyzikální milimetry, SimpleITK vs NumPy pořadí os, orientační kódy (PIL vs RIA), matice laterálního zrcadlení. |
| [`spec/interfaces.md`](file:///c:/ACL_analysis/ACL_graft_analysis/spec/interfaces.md) | **Struktura rozhraní a signatury funkcí**: Typové anotace, vstupní/výstupní datové struktury pro `mri_pipeline`, `anaknee` a `visualizator_analyzator`, pravidla pro ošetření výjimek a vláknový model GUI. |

---

## Závazná pravidla pro vývojáře a AI agenty

1. **Zákaz tichých změn schématu**: Názvy výstupních sloupců v `patient_results.csv` a číselné kódování tříd v maskách nesmí být změněny bez formální aktualizace specifikace a zpětné kompatibility.
2. **Konzistence dimenzí**: Při přechodu mezi SimpleITK `(X, Y, Z)` a NumPy `(Z, Y, X)` musí být vždy explicitně dokumentována a dodržena konvence indexace.
3. **Izotropie mřížky**: Veškeré geometrické výpočty předpokládají rozlišení $(0.5, 0.5, 0.5)\,\text{mm}$. Žádný výpočetní modul nesmí běžet na neizotropním objemu.
