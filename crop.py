import os
import numpy as np
import SimpleITK as sitk

#config
'''
This script crops the T2W and ADC images based on the gland mask. It finds the bounding box of the gland in the mask and crops both images accordingly. The cropped images are saved in a specified output directory.
Configurable parameters:
- t2w_path: Path to the T2W image.
- ADC_path: Path to the ADC image.
- GLAND_MASK_PATH: Path to the gland mask image.
- output_dir: Directory where the cropped images will be saved.

Note that the value for the gland in the mask is assumed to be non-zero. The script checks for geometry consistency between the images and prints relevant information. It also sets the value of the gland mask voxels to the range of 0.5 - 1 and the background to 0 - 0.49 ;for consistency - make sure to adjust as needed to fit training file.
'''




#padding around the gland in VOXELS if needed
# PADDING = 10 
def crop_images(t2w_path, ADC_path, GLAND_MASK_PATH, output_dir):
    # read the images
    t2w_image = sitk.ReadImage(t2w_path)
    adc_image = sitk.ReadImage(ADC_path)
    gland_mask_image = sitk.ReadImage(GLAND_MASK_PATH)

    # print image information
    def print_image_info(name, size, spacing, origin, direction):
        print(f"{name} image information:")
        print(f"  Size: {size}")
        print(f"  Spacing: {spacing}")
        print(f"  Origin: {origin}")
        print(f"  Direction: {direction}")
        print()

    print_image_info("T2W", t2w_image.GetSize(), t2w_image.GetSpacing(), t2w_image.GetOrigin(), t2w_image.GetDirection())
    print_image_info("ADC", adc_image.GetSize(), adc_image.GetSpacing(), adc_image.GetOrigin(), adc_image.GetDirection())
    print_image_info("Gland Mask", gland_mask_image.GetSize(), gland_mask_image.GetSpacing(), gland_mask_image.GetOrigin(), gland_mask_image.GetDirection())
    print()

    def check_geometry(img1, img2, name1, name2):

        print(f"Checking geometry between {name1} and {name2}...")

        if img1.GetSize() != img2.GetSize():
            print(f"  Size mismatch: {name1} size {img1.GetSize()} vs {name2} size {img2.GetSize()}")
        if not np.allclose(img1.GetSpacing(), img2.GetSpacing()):
            print(f"  Spacing mismatch: {name1} spacing {img1.GetSpacing()} vs {name2} spacing {img2.GetSpacing()}")
        if not np.allclose(img1.GetOrigin(), img2.GetOrigin()):
            print(f"  Origin mismatch: {name1} origin {img1.GetOrigin()} vs {name2} origin {img2.GetOrigin()}")
        if not np.allclose(img1.GetDirection(), img2.GetDirection()):
            print(f"  Direction mismatch: {name1} direction {img1.GetDirection()} vs {name2} direction {img2.GetDirection()}")
        print("  Geometry check completed.\n Matches \n")

    check_geometry(t2w_image, adc_image, "T2W", "ADC")
    check_geometry(t2w_image, gland_mask_image, "T2W", "Gland Mask")

    # convert gland mask to numpy array

    gland_array = sitk.GetArrayFromImage(gland_mask_image)
    print("\n Gland mask array shape:", gland_array.shape)
    print(" Gland mask unique values:", np.unique(gland_array)) 

    # find gland voxels

    # 0 = background, 1 = gland
    gland_voxels = np.argwhere(gland_array > 0.5)  # Assuming gland is represented by non-zero values
    if gland_voxels.size == 0:
        raise ValueError("No gland voxels found in the mask.")

    print("\n Number of gland voxels:", gland_voxels.shape[0])

    # find the bounding box

    # [Z, Y , X]
    z_min, y_min, x_min = gland_voxels.min(axis=0)
    z_max, y_max, x_max = gland_voxels.max(axis=0)

    # apply padding
    # z_min = max(z_min - PADDING, 0)
    # y_min = max(y_min - PADDING, 0)
    # x_min = max(x_min - PADDING, 0)
    # z_max = min(z_max + PADDING, gland_array.shape[0] - 1)
    # y_max = min(y_max + PADDING, gland_array.shape[1] - 1)
    # x_max = min(x_max + PADDING, gland_array.shape[2] - 1)

    print("\n Bounding box:")
    print(f"  Z: {z_min} - {z_max}")
    print(f"  Y: {y_min} - {y_max}")
    print(f"  X: {x_min} - {x_max}")

    # calculate the cropping size

    crop_index = [
        int(x_min),
        int(y_min),
        int(z_min)
    ]

    crop_size = [
        int(x_max - x_min + 1),
        int(y_max - y_min + 1),
        int(z_max - z_min + 1)
    ]

    print("\n Crop index:", crop_index)
    print(" Crop size:", crop_size)

    # crop t2w

    print("\n Cropping T2W and ADC images based on the gland mask bounding box...")
    cropped_t2w = sitk.RegionOfInterest(t2w_image, crop_size, crop_index)
    cropped_t2w_path = os.path.join(output_dir, f"{os.path.basename(t2w_path).split('_')[0]}_0000.nii.gz")
    sitk.WriteImage(cropped_t2w, cropped_t2w_path)
    print(f"Cropped T2W image saved to: {cropped_t2w_path}")

    print("\n Cropping ADC image...")
    cropped_adc = sitk.RegionOfInterest(adc_image, crop_size, crop_index)
    cropped_adc_path = os.path.join(output_dir, f"{os.path.basename(ADC_path).split('_')[0]}_0001.nii.gz")
    sitk.WriteImage(cropped_adc, cropped_adc_path)
    print(f"Cropped ADC image saved to: {cropped_adc_path}")

    # test the cropped images by reading them back and printing their information

    def image_inspect(name, path):
        image = sitk.ReadImage(path)
        print(f"\n{name} image information:")
        print(f"  Size: {image.GetSize()}")
        print(f"  Spacing: {image.GetSpacing()}")
        print(f"  Origin: {image.GetOrigin()}")
        print(f"  Direction: {image.GetDirection()}")
        print(f"Pixel type: {image.GetPixelIDTypeAsString()}")

    image_inspect("Original T2W", t2w_path)
    image_inspect("Original ADC", ADC_path)
    image_inspect("Cropped T2W", cropped_t2w_path)
    image_inspect("Cropped ADC", cropped_adc_path) 
    image_inspect("Gland Mask", GLAND_MASK_PATH)

    