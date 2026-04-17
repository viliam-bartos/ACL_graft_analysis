import nibabel as nib
import nibabel.orientations as nio
from pathlib import Path


def reorient_nifti_to_asr(input_dir, output_dir):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in in_path.rglob("*") if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))]

    if not files:
        print("Nenašel jsem žádné NIfTI soubory.")
        return

    # Fígl: Záměrně požadujeme PIL, aby to Slicer a ITK-SNAP přečetly jako ASR
    target_ornt = nio.axcodes2ornt("PIL")

    for f in files:
        print(f"Zpracovávám: {f.name}")
        try:
            img = nib.load(f)
            orig_ornt = nio.io_orientation(img.affine)

            transform = nio.ornt_transform(orig_ornt, target_ornt)
            new_img = img.as_reoriented(transform)

            nib.save(new_img, out_path / f.name)
        except Exception as e:
            print(f"Chyba u {f.name}: {e}")


if __name__ == "__main__":
    INPUT = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images"
    OUTPUT = r"C:\DIPLOM_PRACE\ACL_segment\data_train\images_ASR"

    reorient_nifti_to_asr(INPUT, OUTPUT)