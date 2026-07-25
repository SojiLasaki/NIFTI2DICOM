from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
from crop import crop_images

import torch

full_size_t2w_path = "../NIFTI2DICOM/input_nifti_files/10005/10005_t2w.nii.gz"
full_size_ADC_path = "../NIFTI2DICOM/input_nifti_files/10005/10005_adc.nii.gz"
GLAND_MASK_PATH = "../NIFTI2DICOM/input_nifti_files/10005/10005_gland.nii.gz"

output_dir = "cropped_nifti/imagesTr/{}".format(os.path.basename(full_size_t2w_path).split('_')[0])
os.makedirs(output_dir, exist_ok=True)

def tumor_seg():
    crop_images(full_size_t2w_path, full_size_ADC_path, GLAND_MASK_PATH, output_dir)
    tumor_seg_output_folder = "output_nifti_files/10005"

    input_files = [
        "cropped_nifti/imagesTr/10005/10005_0000.nii.gz",
        "cropped_nifti/imagesTr/10005/10005_0001.nii.gz",
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

