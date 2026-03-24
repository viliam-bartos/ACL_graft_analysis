import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def stratified_split(base_dir, target_dir, seed=42):
    base = Path(base_dir)
    target = Path(target_dir)

    try:
        meta = pd.read_csv(base / "metadata.csv")
        mapping = pd.read_csv(base / "mapping.csv")
    except FileNotFoundError as e:
        print(f"Chyba: Nenalezen soubor {e.filename}.")
        return

    # Sloučení přes case_id
    df = pd.merge(meta, mapping, on="case_id")

    # Pojistka pro prázdné hodnoty, jinak stratifikace spadne
    df['sex'] = df['sex'].fillna('Unknown')

    # První split: 70 % train, 30 % zbytek
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['sex']
    )

    # Druhý split: těch 30 % rozpůlí na 15 % val a 15 % test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df['sex']
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for split_name, split_data in splits.items():
        split_dir = target / split_name
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Uložení rozdělených metadat a mapování
        split_data[['case_id', 'age', 'sex', 'weight']].to_csv(split_dir / "metadata.csv", index=False)
        split_data[['case_id', 'original_path']].to_csv(split_dir / "mapping.csv", index=False)

        # Fyzické kopírování
        for cid in split_data['case_id']:
            src_nii_gz = base / "images" / f"{cid}.nii.gz"
            src_nii = base / "images" / f"{cid}.nii"

            if src_nii_gz.exists():
                shutil.copy2(src_nii_gz, img_dir / f"{cid}.nii.gz")
            elif src_nii.exists():
                shutil.copy2(src_nii, img_dir / f"{cid}.nii")
            else:
                print(f"Varování: Chybí zdrojový obrázek pro {cid}")

        m_count = sum(split_data['sex'] == 'M')
        f_count = sum(split_data['sex'] == 'F')
        u_count = sum(split_data['sex'] == 'Unknown')
        print(f"Split {split_name}: {len(split_data)} záznamů (M: {m_count}, F: {f_count}, Unknown: {u_count})")

if __name__ == "__main__":
    BASE = r"C:\DIPLOM_PRACE\ACL_segment\data_train"
    TARGET = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split"
    stratified_split(BASE, TARGET)