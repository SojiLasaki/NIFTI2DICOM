import os
from pathlib import Path
import SimpleITK as sitk

from nifti2dcm_converter import nifti2dcm
from nifti2dcmseg_converter import nifti2dcm_seg

# adjust these values to point to your input and output directories
input_dir = Path("input_nifti_files")
dicom_output_root = Path("output_dicom_series")


# Loop through each folder in the input directory and process the NIfTI files
for f in os.listdir(input_dir):
    print(f"Processing folder: {f}")
    try:
        t2_path = input_dir / f / f"{f}_t2w.nii.gz"
        gland_path = input_dir / f / f"{f}_gland.nii.gz"

        if not t2_path.exists():
            print(f"File not found: {t2_path}")
            print("----------------------------")
            continue

        if not gland_path.exists():
            print(f"File not found: {gland_path}")
            print("----------------------------")
            continue

        dicom_series_folder = dicom_output_root / f / "DICOM_SERIES"
        dicom_seg_folder = dicom_output_root / f / "DICOM_SEG"
        dicom_seg_folder.mkdir(parents=True, exist_ok=True)

        t2_img = sitk.ReadImage(str(t2_path))
        t2_img = sitk.Cast(t2_img, sitk.sitkInt16)

        nifti2dcm(t2_img, str(dicom_series_folder))
        print("DICOM series conversion complete.")
        print("----------------------------")

        out_seg_path = dicom_seg_folder / f"{f}_gland.dcm"

        nifti2dcm_seg(str(dicom_series_folder), str(gland_path), str(out_seg_path))
        print("DICOM SEG conversion complete.")
        print("----------------------------")
        print("Finished processing folder:", f)
        print("----------------------------")

    except Exception as e:
        print(f"Error processing folder {f}: {e}")
        print("----------------------------")
    print("")
