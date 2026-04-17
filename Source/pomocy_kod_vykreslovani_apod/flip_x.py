import nibabel as nib
import numpy as np
import os

def flip_nifti_x(input_path, output_path):
    print(f"Loading {input_path}...")
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    print("Flipping along x-axis (axis 0)...")
    flipped_data = np.flip(data, axis=1)
    #flipped_data = np.flip(flipped_data, axis=1)
    
    # Create new NIfTI image with the same affine and header
    flipped_img = nib.Nifti1Image(flipped_data.astype(data.dtype), affine, header)
    
    print(f"Saving to {output_path}...")
    nib.save(flipped_img, output_path)
    print("Done!")

if __name__ == "__main__":
    input_file = r"C:\DIPLOM_PRACE\ACL_segment\dataset_split\train\images\case_153.nii.gz"
    output_file = r"c:\DIPLOM_PRACE\ACL_segment\case_153.nii.gz"
    
    if os.path.exists(input_file):
        flip_nifti_x(input_file, output_file)
    else:
        print(f"Error: {input_file} not found.")
