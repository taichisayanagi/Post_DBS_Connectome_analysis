"""Generate a fake volume and browser QC report; contains no patient-derived data."""
import argparse
import nibabel as nib
import numpy as np
from dbs_connectome.masks import study_dilate, tube_mask
from dbs_connectome.provenance import new_run, record
from dbs_connectome.qc import browser_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    out = new_run(args.output_root, "synthetic_qc")
    shape = (64, 64, 64)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = -64
    voxel = np.moveaxis(np.indices(shape), 0, -1)
    xyz = nib.affines.apply_affine(affine, voxel)
    r = ((xyz / [48, 56, 52]) ** 2).sum(-1)
    rng = np.random.default_rng(42)
    data = np.where(r < 1, 80 + 110 * np.exp(-2*r) + rng.normal(0, 3, shape), 0).astype(np.float32)
    paths = [[[-12, 4, -18], [-14, 7, 36]], [[12, 4, -18], [14, 7, 36]]]
    void = tube_mask(shape, affine, paths, 3.5)
    data[void] = 0
    mask = study_dilate(void, affine, 1)
    reference, mask_path = out / "synthetic_b0.nii.gz", out / "synthetic_mask.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), reference)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
    page = out / "mask_review.html"
    browser_report(reference, mask_path, page, synthetic=True)
    record(out, "synthetic_demo", [], {"seed": 42, "patient_data": False}, [reference, mask_path, page])
    print(page)


if __name__ == "__main__":
    main()
