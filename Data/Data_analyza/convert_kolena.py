import os
import glob
import nibabel as nib
import nibabel.orientations as nio
import SimpleITK as sitk
from pathlib import Path

def get_patient_id(filename):
    # Extracts patient ID from filename, e.g. "A05828 dx000.dcm" -> "A05828"
    return filename.split()[0].replace("-", "").strip()

def process_dcm(dcm_path, output_dir):
    dcm_path = Path(dcm_path)
    filename = dcm_path.name
    patient_id = get_patient_id(filename)
    
    # Create patient directory
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean output filename
    # Remove "000.dcm" or ".dcm" and append ".nii.gz"
    clean_name = dcm_path.stem.replace("000", "").strip()
    # Handle the " - " case
    clean_name = clean_name.replace(" - ", "_").replace(" ", "_")
    
    out_nii_path = patient_dir / f"{clean_name}.nii.gz"
    temp_nii_path = patient_dir / f"temp_{clean_name}.nii.gz"
    
    print(f"Processing: {filename} -> {out_nii_path.name} (Patient: {patient_id})")
    
    try:
        # 1. Read DICOM with SimpleITK
        # If it's a single 3D DICOM file (enhanced), SimpleITK reads it directly
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(dcm_path))
        image = reader.Execute()
        
        # Write to temporary NIfTI
        sitk.WriteImage(image, str(temp_nii_path))
        
        # 2. Reorient to PIL using nibabel
        img = nib.load(str(temp_nii_path))
        orig_ornt = nio.io_orientation(img.affine)
        target_ornt = nio.axcodes2ornt("PIL")
        
        transform = nio.ornt_transform(orig_ornt, target_ornt)
        new_img = img.as_reoriented(transform)
        
        # Save reoriented NIfTI
        nib.save(new_img, str(out_nii_path))
        
        # Clean up temporary file
        if temp_nii_path.exists():
            temp_nii_path.unlink()
            
        print(f"  [OK] Saved to {out_nii_path}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to process {filename}: {e}")
        if temp_nii_path.exists():
            try:
                temp_nii_path.unlink()
            except:
                pass
        return False

def main():
    input_dir = Path(r"C:\DIPLOM_PRACE\CEITEC\2509-MRI-Knee\Data\Data_analyza\kolena anon\FS")
    output_dir = input_dir.parent / (input_dir.name + "_nifti_PIL")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dcm_files = glob.glob(str(input_dir / "*.dcm"))
    if not dcm_files:
        print("No DICOM files found.")
        return
        
    print(f"Found {len(dcm_files)} DICOM files. Converting to NIfTI (PIL orientation)...")
    print(f"Output directory: {output_dir}")
    
    success_count = 0
    for dcm in sorted(dcm_files):
        if process_dcm(dcm, output_dir):
            success_count += 1
            
    print(f"\nFinished processing. Successfully converted {success_count}/{len(dcm_files)} files.")

if __name__ == "__main__":
    main()
