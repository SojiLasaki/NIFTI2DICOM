import subprocess
import os

input_dir = "input_dicom_seriew/"

if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

output_dir = "output_nifti/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def convert_dicom_to_nifti(input_dir, output_dir):
    cmd = [
        "dcm2niix",
        "-z", "y",
        "-o", output_dir,
        input_dir
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    subprocess.run(cmd, check=True)

convert_dicom_to_nifti(input_dir, output_dir)