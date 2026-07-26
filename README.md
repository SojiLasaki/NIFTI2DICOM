Heee is the workflow to have this work

File structure for model
- nnUNet_results /
    - Dataset006_C3_PICAI_P158 /
      - nnUNetTrainer_FullPatchC3__nnUNetPlans__3d_fullres /
        - Fold /
          -  checkpoint.final.pth
        - dataset_fingerprint.json
        - dataset.json
        - plans.json

correct the location of the input files, once done, run the tumor_seg.py script. The script contains a function that first calls a crop function. This script makes sure the t2w file is cropped down to the region of interest being on ly the gland as the model was trained on only cropped files. 

The model takes in 2 files
- t2ww (case_id_0000.nii.gz)
- adc (case_id_0001.nii.gz)

Do not rename the files in the file, if your file names are not formatted, the crop script would rename it regardless. 

Once the file is cropped, the files would then be passed into the predictor. The file result would be placed in output_nifti_files. 

Review the predictor driver and change to cuda if GPU is avaialble, if not, leave as cpu.