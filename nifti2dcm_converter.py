import os
import SimpleITK as sitk
from matplotlib.path import Path
import numpy as np
from pydicom.uid import generate_uid
import pydicom
from nifti2dcmseg_converter import nifti2dcm_seg

study_uid = generate_uid()
series_uid = generate_uid()
frame_uid = generate_uid()

# dicom coonversion can sometimes produce slices with inconsistent metadata which causes them to load separately in viewers. This function fixes that by ensuring all slices share the same Study, Series, and Frame UIDs, while still giving each slice a unique SOPInstanceUID and InstanceNumber. It also ensures the Modality is set to MR for all slices. Run this after the initial conversion to ensure a consistent DICOM series.

def adjust_metadata_consistency(output_dir):
    print("Adjusting metadata consistency across DICOM slices...")

    files = sorted([f for f in os.listdir(output_dir) if f.endswith(".dcm")])

    for i, f in enumerate(files, start=1):
        ds = pydicom.dcmread(os.path.join(output_dir, f))

        # shared across the whole series
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid

        # unique per slice
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = i

        # keep it as MR
        ds.Modality = "MR"
        ds.SeriesNumber = 1

        ds.save_as(os.path.join(output_dir, f), write_like_original=False)

    print("Fixed DICOM series metadata.")


# function to convert NIfTI to DICOM series
def nifti2dcm(t2_img, output_folder):
    print("Converting NIfTI to DICOM series...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    size = t2_img.GetSize()
    print("Image size (x, y, z):", size)
    print("---------------------------")
    # shared across the whole series

    writer = sitk.ImageFileWriter()

    for z in range(size[2]):
        slice_img = t2_img[:, :, z]

        # shared UIDs for every slice
        slice_img.SetMetaData("0020|000d", study_uid)   # StudyInstanceUID
        slice_img.SetMetaData("0020|000e", series_uid)  # SeriesInstanceUID
        slice_img.SetMetaData("0020|0052", frame_uid)   # FrameOfReferenceUID

        # unique per slice
        slice_img.SetMetaData("0008|0018", generate_uid())  # SOPInstanceUID
        slice_img.SetMetaData("0020|0013", str(z + 1))       # InstanceNumber

        # MRI-related tags
        slice_img.SetMetaData("0008|0060", "MR")  # Modality
        slice_img.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.4")  # MR Image Storage

        # geometry
        origin = t2_img.TransformIndexToPhysicalPoint((0, 0, z))
        slice_img.SetMetaData("0020|0032", f"{origin[0]}\\{origin[1]}\\{origin[2]}")
        slice_img.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")

        spacing = t2_img.GetSpacing()  # (x, y, z)

        slice_img.SetMetaData("0018|0050", str(float(spacing[2])))     # SliceThickness
        slice_img.SetMetaData("0028|0030", f"{float(spacing[1])}\\{float(spacing[0])}")  # PixelSpacing

        filename = os.path.join(output_folder, f"IM_{z:04d}.dcm")
        writer.SetFileName(filename)
        writer.Execute(slice_img)

    print(f"Saved {size[2]} DICOM slices to {output_folder}")
    print("---------------------------")

    # call the metadata consistency function after conversion to ensure all slices are consistent
    adjust_metadata_consistency(output_folder)

