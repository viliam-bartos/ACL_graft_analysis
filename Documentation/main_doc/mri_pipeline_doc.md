Hlavní skript, který zprostředkuje následující kroky v izolovaných a vypínatelných funkcích. Obsahuje centrální CONFIG slovník a využívá modul logging s nastavením logování do souboru .log. Iterace přes složku bude v try-except bloku, takže chyba u jednoho pacienta nezastaví celý proces.

Postup pipeline v rámci souboru:

Resampling: Pomocí SimpleITK nebo MONAI se ověří originální spacing. Pokud není přesně (0.5, 0.5, 0.5) (s drobnou tolerancí např. 0.01 mm), provede se resamplování vstupního 3D objemu s interpolací 3. řádu (B-Spline). Původní spacing si skript zapamatuje pro krok 5.
Kontrola Orientace: Načtením logiky ze skriptu reorient.py se převedou axiální kódy. Pokud objem není ASR (neboli cílový orientanční systém např. "PIL" pro Slicer), přeorientuje jej pomocí nibabel.orientations.

Kanonizace a lateralita: Importuje se třída LateralityClassifier ze složky kanonizace. Získá se text z predikce. Pokud vyjde Right, pole se přes osu 0 (axis=0) asymetricky zrcadlí, aby síť vždy pracovala se strukturou levého kolene.

Inference (Blackwell segmentace): Vytvoří a načte architekturu LightUNet3D a její váhy (cestu definuje CONFIG). Použitím sliding_window_inference z knihovny MONAI provede multiclass segmentaci (předpoklad: třídy 1=ACL, 2=Femur, 3=Tibia).

Inverzní transformace masky: Maska opouští síť. V případě předchozího zrcadlení se provede zpětný flip np.flip(..., axis=0). Následně je celá maska interpolována nultým řádem (Nearest-Neighbor) zpět do originálních rozměrů a spacingu získaného v kroku 1. Objem dat z pacienta se nezmění. Výsledná maska se uloží.

Post-processing masek: Modul provede pročištění komponent. Pro třídy typu ACL je možné zapnout izolaci největší komponenty (LCC - Largest Connected Component) přes scipy.ndimage.label.

Segmentační analýza: Modul najednou projde uložené inference a srovná je s Ground Truth maskou. Metriky budeme počítat pomocí MONAI (DiceMetric, HausdorffDistanceMetric) anebo medpy.metric.binary. Z naměřených dat skript vygeneruje grafické výstupy via seaborn a matplotlib (boxploty) do PDF i PNG, a zapíše metrics.csv.
Anatomická analýza (anaknee): Výstupy (původní raw nifti a finální maska) se odešlou do skriptu run_analysis v anaknee. Dle konfigurace (pokud běžíme dávku, generují se jen CSV a základní popisy, pro jednotlivý případ se na konci vyvolá visualize_results přes pyvista).