to run inference

nnUNetv2_predict \
  -i output_nifti_files/imagesTr \
  -o output_nifti_files \
  -d Dataset006_C3_PICAI_P158 \
  -c 3d_fullres \
  -f 0 1 2 \
  -tr nnUNetTrainer_FinalSOTAKillVersion \
  --disable_tta


use this

nnUNetv2_predict \
  -i output_nifti_files/imagesTr \
  -o output_nifti_files \
  -d Dataset006_C3_PICAI_P158 \
  -c 3d_fullres \
  -f 0 1 2 \
  -tr nnUNetTrainer_FullPatchC3 \
  -device cpu \
  --disable_tta


export nnUNet_raw="/Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_raw"
export nnUNet_preprocessed="/Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_preprocessed"
export nnUNet_results="/Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_results"


mkdir -p /Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_raw
mkdir -p /Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_preprocessed
mkdir -p /Users/oluwasojilasaki/Downloads/code/NIFTI2DICOM/nnUNet_results



issues with background workers dying. due to channel mismatch. 

check to see what is required
cat nnUNet_results/Dataset006_C3_PICAI_P158/nnUNetTrainer_FullPatchC3__nnUNetPlans__3d_fullres/dataset.json


