import os
from Prostate_cancer_lesion_detection.customized_trainer import nnUNetTrainer_FinalSOTAKillVersion
from Prostate_cancer_lesion_detection.gland_crop_bbox import box_mask_crop

ckpt_dir = "nnUNet_results/Dataset006_C3_PICAI_P158/nnUNetTrainer_FullPatchC3__nnUNetPlans__3d_fullres/",  
print(f"Checkpoint directory: {ckpt_dir}")
