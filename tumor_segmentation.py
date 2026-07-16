from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os

import torch

def main():
    output_folder = "output_nifti_files/10005"

    input_files = [
        "imagesTr/10005/10005_0000.nii.gz",
        "imagesTr/10005/10005_0001.nii.gz",
    ]

    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
        checkpoint_name="checkpoint_best.pth"
    )

    predictor.predict_from_files(
        [input_files],
        output_folder,
        save_probabilities=False
    )


if __name__ == "__main__":
    main()