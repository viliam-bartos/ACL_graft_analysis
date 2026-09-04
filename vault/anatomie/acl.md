# Anatomie a biomechanika ACL a kolenního kloubu

Tento dokument shrnuje anatomická pravidla, geometrické vztahy a klinický kontext předního zkříženého vazu (*Ligamentum cruciatum anterius* – ACL), přilehlých kostí (femur, tibia) a rekonstruovaného štěpu (graftu).

---

## 1. Nativní anatomie ACL

Přední zkřížený vaz je primárním pasivním stabilizátorem kolenního kloubu. Zabraňuje:
1. **Přední translaci tibie vůči femuru** (poskytuje cca 85 % pasivního odporu proti přednímu posunu).
2. **Vnitřní rotaci tibie** při extenzi a flexi.
3. **Hyperextenzi** kolenního kloubu.

### Funkční svazky
Anatomicky se ACL skládá ze dvou hlavních funkčních svazků pojmenovaných podle jejich tibiálního úponu:
* **Anteromediální svazek (AM bundle)**:
  * Napíná se především při **flexi** kolene.
  * Zajišťuje především předozadní (AP) stabilitu.
* **Posterolaterální svazek (PL bundle)**:
  * Napíná se především při **extenzi** (vystření) kolene.
  * Zajišťuje rotační stabilitu (brání pivot-shift fenoménu).

Při běžné 3D MRI segmentaci (a zejména po rekonstrukci vazu štěpem) je vaz zpravidla segmentován jako jeden kompaktní anatomický objekt (Label 1).

---

## 2. Inserční zóny (Footprints)

Přesné umístění úponů vazu je klíčové pro izometrii a fyziologické fungování.

### 2.1 Femorální úpon (Femoral Footprint)
* **Lokalizace**: Zadní část vnitřní (mediální) plochy laterálního femorálního kondylu (*lateral femoral condyle*).
* **Anatomické orientační body**:
  * **Blumensaatova linie (Blumensaat's line)**: Kortikální linie tvořící strop interkondylární fossy viditelná na sagitálních řezech.
  * **Laterální interkondylární hřeben (*Resident's ridge*)**: Dělí laterální stěnu fossy na přední a zadní část; nativní úpon leží bezprostředně dorzálně od tohoto hřebene.
* **Klinické riziko chybné pozice štěpu**:
  * *Příliš ventrálně (anteriorně)*: Štěp se při flexi extrémně napíná, dochází k omezení hybnosti nebo selhání vazu.
  * *Příliš vertikálně*: Ztráta rotační stability kolene.

### 2.2 Tibiální úpon (Tibial Footprint)
* **Lokalizace**: Interkondylární plocha tibie (*area intercondylaris anterior*), mezi eminentia intercondylaris a předním úponem laterálního menisku.
* **Vztah k tibiálnímu platu**:
  * Leží v přední třetině až polovině sagitální AP hloubky tibie (podrobněji viz [Stäubliho procento](file:///c:/ACL_analysis/ACL_graft_analysis/vault/anatomie/biomedical-metrics.md#2-staubliho-tibiální-procento-staubli-tibial-percentage)).
* **Klinické riziko chybné pozice štěpu**:
  * *Příliš ventrálně (anteriorně)*: Způsobuje impingement (narážení) štěpu o strop interkondylární fossy (Blumensaatovu linii) při plné extenzi kolene.
  * *Příliš dorzálně (posteriorně)*: Ztráta stability v předním tahu (přední zásuvkový fenomén).

---

## 3. Kloubní geometrie a přilehlé struktury

### 3.1 Tibiální plato (Tibial Plateau)
* Plocha horního konce tibie tvořená mediální a laterální facetou.
* Sklon tibiálního plata (*tibial slope*) – fyziologicky mírný dorzální sklon (obvykle 7–10°).
* V naší analýze fitujeme rovinu tibiálního plata $\mathbf{n}_{\text{plateau}}$ pomocí SVD analýzy z proximálních voxelů tibie. Tato rovina tvoří biomechanickou referenci pro výpočet úhlů i přední translace (ATT).

### 3.2 Interkondylární fossa (Intercondylar Notch)
* Zářez mezi mediálním a laterálním kondylem femuru, kterým prochází ACL i PCL (zadní zkřížený vaz).
* **Notch Stenosis (zúžení fossy)**: Úzká fossa výrazně zvyšuje riziko ruptury jak nativního vazu, tak pooperačního štěpu v důsledku mechanického otěru (impingementu) o laterální stěnu či strop fossy.

---

## 4. Klinická analýza rekonstruovaného vazu (ACL Graft)

Po rekonstrukci ACL (např. pomocí štěpu z *m. semitendinosus / gracilis*, patelární šlachy BTB nebo šlachy kvadricepsu) prochází štěp fází vhojování a ligamentizace:
1. **Laxita a prodloužení**: Nedostatečné napětí štěpu vede ke zvětšení přední translace tibie (ATT $> 3\,\text{mm}$ oproti zdravé končetině značí patologii).
2. **Tortuozita (zvlnění)**: Zdravý, funkčně napnutý vaz má téměř lineární průběh ($\tau \approx 1.05 - 1.15$). Ochablý, částečně prasklý nebo insuficientní štěp vykazuje zvlněný průběh ($\tau > 1.25$).
3. **Orientace v prostoru**: Úhel vůči tibiálnímu platu by měl dosahovat fyziologických hodnot (cca 45°–60° v sagitální rovině). Plošší vaz ($<45^\circ$) neposkytuje dostatečnou vertikální oporu, příliš strmý vaz ($>70^\circ$) nebrání přednímu posunu tibie.
