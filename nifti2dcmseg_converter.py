from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
import os
import highdicom as hd
from pydicom.sr.codedict import codes

# this is a function to convert a NIfTI segmentation mask into a DICOM SEG object. It takes the path to the DICOM series, the path to the NIfTI segmentation mask, and the output path for the DICOM SEG file. It reads the source DICOM images, resamples the NIfTI mask to match the geometry of the DICOM series, and creates a DICOM SEG object with appropriate metadata. The resulting DICOM SEG file is saved to the specified output path.  

def nifti2dcm_seg(dicom_dir, gland_path, dicom_seg_output):
    print("Converting NIfTI to DICOM series...")
    dicom_dir = Path(dicom_dir)
    dicom_seg_output = Path(dicom_seg_output)
    gland_path = Path(gland_path)

    # Source DICOM for highdicom
    source_images = [pydicom.dcmread(str(p)) for p in sorted(dicom_dir.glob("*.dcm"))]

    # Read DICOM series as a SimpleITK reference image
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_files)
    ref_img = reader.Execute()

    # Read gland mask NIfTI
    mask_img = sitk.ReadImage(str(gland_path))
    mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

    # Resample mask to the DICOM geometry
    mask_img = sitk.Resample(
        mask_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    # Convert to (slices, rows, cols)
    mask = sitk.GetArrayFromImage(mask_img) > 0

    # print("mask shape:", mask.shape)
    # print("expected:", (len(source_images), source_images[0].Rows, source_images[0].Columns))

    # set algorithm identification for the segmentation
    alg_id = hd.AlgorithmIdentificationSequence(
        name="AiMed Segmentation Algorithm",
        version="1.0",
        family=codes.cid7162.ArtificialIntelligence,
    )

    # adjust data as needed - copied from ChatGPT. 
    segment_desc = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="gland",
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=alg_id,
        tracking_uid=hd.UID(),
        tracking_id="gland segmentation"
    )

    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=mask.astype(bool),
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[segment_desc],
        series_instance_uid=hd.UID(),
        series_number=300,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="MyLab",
        manufacturer_model_name="AiMed Segmentation Model",
        software_versions="1.0",
        device_serial_number="NA",
        series_description="Gland SEG"
    )

    seg.save_as(str(dicom_seg_output))
    print(f"Saved {dicom_seg_output}")

    return