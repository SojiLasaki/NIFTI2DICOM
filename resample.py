from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BASE_DIR = Path("/content/drive/MyDrive/JIIM_datasets")

DATA_ROOT = BASE_DIR / "Data"
GAN_T2_ROOT = BASE_DIR / "GAN_harmonized"
REF_T2_ROOT = BASE_DIR / "cropped_new_fast_cropfirst"
OUT_ROOT = BASE_DIR / "GAN_harmonized_readynew"

DATASETS = ["PICAI", "P158"]

NUM_WORKERS = min(8, os.cpu_count() or 2)

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

print("CPU:", os.cpu_count())
print("Workers:", NUM_WORKERS)

def is_nifti(p):
    n = p.name.lower()
    return p.is_file() and not p.name.startswith("._") and (
        n.endswith(".nii") or n.endswith(".nii.gz")
    )

def find_file(case_dir, include, exclude=None):
    if exclude is None:
        exclude = []

    if not case_dir.exists():
        return None

    files = [p for p in case_dir.glob("*.nii*") if is_nifti(p)]

    hits = []
    for p in files:
        name = p.name.lower()
        if any(k in name for k in include) and not any(k in name for k in exclude):
            hits.append(p)

    return sorted(hits)[0] if hits else None

def find_adc(case_dir):
    p = find_file(case_dir, ["adc_reg"])
    if p is None:
        p = find_file(case_dir, ["adc"])
    return p

def find_gland(case_dir):
    return find_file(case_dir, ["gland", "prostate"])

def find_tumor(case_dir):
    case_id = case_dir.name

    exact1 = case_dir / f"{case_id}_tumor.nii.gz"
    exact2 = case_dir / f"{case_id}_tumor.nii"

    if exact1.exists():
        return exact1
    if exact2.exists():
        return exact2

    return find_file(case_dir, ["tumor", "lesion"])

def find_t2(case_dir):
    p = find_file(case_dir, ["t2w", "t2"])
    if p is None and case_dir.exists():
        files = [x for x in case_dir.glob("*.nii*") if is_nifti(x)]
        return sorted(files)[0] if files else None
    return p

def resample_to_ref(img, ref, is_mask=False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    return r.Execute(img)

def binarize_uint8(img):
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out

def write_empty_mask(ref, out_path):
    empty = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(ref)
    sitk.WriteImage(empty, str(out_path), True)

def copy_gan_with_ref_geometry_fast(gan_path, ref_img, out_path):
    gan = sitk.ReadImage(str(gan_path))

    if gan.GetSize() != ref_img.GetSize():
        raise ValueError(f"GAN size {gan.GetSize()} != ref size {ref_img.GetSize()}")

    gan.CopyInformation(ref_img)
    sitk.WriteImage(gan, str(out_path), True)

def process_case(args):
    dataset, case_id = args

    try:
        orig_case_dir = DATA_ROOT / dataset / case_id
        gan_case_dir = GAN_T2_ROOT / dataset / case_id
        ref_case_dir = REF_T2_ROOT / dataset / case_id

        out_case_dir = OUT_ROOT / dataset / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        gan_t2_path = find_t2(gan_case_dir)
        ref_t2_path = find_t2(ref_case_dir)

        adc_path = find_adc(orig_case_dir)
        gland_path = find_gland(orig_case_dir)
        tumor_path = find_tumor(orig_case_dir)

        if gan_t2_path is None:
            return False, dataset, case_id, "Missing GAN T2W"

        if ref_t2_path is None:
            return False, dataset, case_id, "Missing reference cropped T2W"

        if adc_path is None:
            return False, dataset, case_id, "Missing ADC"

        if gland_path is None:
            return False, dataset, case_id, "Missing gland"

        ref_t2 = sitk.ReadImage(str(ref_t2_path))

        copy_gan_with_ref_geometry_fast(
            gan_t2_path,
            ref_t2,
            out_case_dir / f"{case_id}_t2w.nii.gz"
        )

        adc = sitk.ReadImage(str(adc_path))
        adc_on_t2 = resample_to_ref(adc, ref_t2, is_mask=False)
        sitk.WriteImage(
            adc_on_t2,
            str(out_case_dir / f"{case_id}_adc_reg.nii.gz"),
            True
        )

        gland = sitk.ReadImage(str(gland_path))
        gland_on_t2 = resample_to_ref(gland, ref_t2, is_mask=True)
        gland_on_t2 = binarize_uint8(gland_on_t2)
        sitk.WriteImage(
            gland_on_t2,
            str(out_case_dir / f"{case_id}_gland.nii.gz"),
            True
        )

        out_tumor_path = out_case_dir / f"{case_id}_tumor.nii.gz"

        if tumor_path is not None:
            tumor = sitk.ReadImage(str(tumor_path))
            tumor_arr = sitk.GetArrayFromImage(tumor)

            # IMPORTANT: if original tumor is empty, keep output empty
            if int((tumor_arr > 0).sum()) == 0:
                write_empty_mask(ref_t2, out_tumor_path)
            else:
                tumor_on_t2 = resample_to_ref(tumor, ref_t2, is_mask=True)
                tumor_on_t2 = binarize_uint8(tumor_on_t2)
                sitk.WriteImage(tumor_on_t2, str(out_tumor_path), True)
        else:
            write_empty_mask(ref_t2, out_tumor_path)

        return True, dataset, case_id, "done"

    except Exception as e:
        return False, dataset, case_id, str(e)

all_jobs = []

for dataset in DATASETS:
    dataset_dir = DATA_ROOT / dataset

    if not dataset_dir.exists():
        print("Missing dataset:", dataset_dir)
        continue

    case_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    print(f"{dataset}: {len(case_dirs)} cases")

    for case_dir in case_dirs:
        all_jobs.append((dataset, case_dir.name))

print("\nTotal jobs:", len(all_jobs))

failed = []
saved = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_case, job) for job in all_jobs]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Fast preparing GAN-ready dataset"
    ):
        ok, dataset, case_id, msg = future.result()

        if ok:
            saved += 1
        else:
            failed.append((dataset, case_id, msg))

print("\nDONE")
print("Saved cases:", saved)
print("Output:", OUT_ROOT)

if failed:
    print("\nFAILED CASES:", len(failed))
    for x in failed[:100]:
        print(x)
else:
    print("No failures.")