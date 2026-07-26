from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
from crop import crop_images

import torch

case_id = "11278"

full_size_t2w_path = f"../NIFTI2DICOM/input_nifti_files/{case_id}/{case_id}_t2w.nii.gz"
full_size_ADC_path = f"../NIFTI2DICOM/input_nifti_files/{case_id}/{case_id}_adc.nii.gz"
GLAND_MASK_PATH = f"../NIFTI2DICOM/input_nifti_files/{case_id}/{case_id}_gland.nii.gz"

output_dir = f"cropped_nifti/imagesTr/{case_id}"
os.makedirs(output_dir, exist_ok=True)

def tumor_seg():
    crop_images(full_size_t2w_path, full_size_ADC_path, GLAND_MASK_PATH, output_dir)
    tumor_seg_output_folder = f"output_nifti_files/{case_id}"

    input_files = [
        f"cropped_nifti/imagesTr/{case_id}/{case_id}_0000.nii.gz",
        f"cropped_nifti/imagesTr/{case_id}/{case_id}_0001.nii.gz",
    ]

    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
    if not os.path.exists(tumor_seg_output_folder):
        os.makedirs(tumor_seg_output_folder)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cpu"), # use cuda if gpu is available
        verbose=False,
    )

    predictor.initialize_from_trained_model_folder(
        "nnUNet_results/Dataset006_C3_PICAI_P158/nnUNetTrainer_FullPatchC3__nnUNetPlans__3d_fullres/",
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth"
    )

    predictor.predict_from_files(
        [input_files],
        tumor_seg_output_folder,
        save_probabilities=False
    )


if __name__ == "__main__":
    tumor_seg()

