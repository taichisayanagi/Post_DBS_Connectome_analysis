"""Conservative, review-gated mask primitives, not automatic lead localization."""

from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage

from .provenance import digest, image3d, read_json, same_grid, write_json


def tube_mask(shape, affine, paths, radius_mm=3.5):
    """Rasterize piecewise-linear centerlines in NIfTI RAS+ millimetres.

    Exact voxel-center-to-segment distance supports anisotropic/oblique grids.
    CT points in LPS, voxel coordinates or unregistered CT space are invalid.
    """
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Mask radius must be positive and finite")
    if not paths:
        raise ValueError("At least one reviewed centerline is required")
    mask = np.zeros(shape, dtype=bool)
    for path in paths:
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Each centerline must contain at least two finite RAS+ points")
        # Process one slab at a time to bound memory use on CT-size arrays.
        count_before = int(mask.sum())
        for z in range(shape[2]):
            ij = np.indices(shape[:2]).reshape(2, -1).T
            vox = np.column_stack((ij, np.full(len(ij), z)))
            xyz = nib.affines.apply_affine(affine, vox)
            inside = np.zeros(len(xyz), dtype=bool)
            for a, b in zip(points[:-1], points[1:]):
                vector = b - a
                length2 = np.dot(vector, vector)
                if length2 < 1e-12:
                    raise ValueError("Consecutive centerline points must differ")
                t = np.clip((xyz - a) @ vector / length2, 0, 1)
                distance = np.linalg.norm(xyz - (a + t[:, None] * vector), axis=1)
                inside |= distance <= radius_mm
            mask[:, :, z] |= inside.reshape(shape[:2])
        if int(mask.sum()) == count_before:
            raise ValueError("A centerline adds no mask voxels: check coordinate convention and overlap")
    return mask


def study_dilate(mask, affine, passes=1):
    """Study-style 6-neighbour dilation on a 2-mm grid (not a Euclidean shell)."""
    if passes < 0 or int(passes) != passes:
        raise ValueError("Dilation passes must be a nonnegative integer")
    if not np.allclose(nib.affines.voxel_sizes(affine), 2.0, atol=1e-4):
        raise ValueError("Study dilation requires 2-mm isotropic voxels; resample explicitly first")
    basis = affine[:3, :3] / 2
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4):
        raise ValueError("Sheared grids are not supported by study-style dilation")
    if passes == 0:
        return np.asarray(mask, dtype=bool).copy()
    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1), int(passes))


def validate_binary(image):
    data = np.asanyarray(image.dataobj)
    if not np.isin(data, (0, 1)).all() or not np.any(data):
        raise ValueError("Mask must be nonempty and binary (0/1)")
    return data.astype(bool)


def mask_qc(mask, affine):
    _, components = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    return {"voxels": int(mask.sum()), "volume_mm3": float(mask.sum() * abs(np.linalg.det(affine[:3, :3]))),
            "components_6_connected": int(components),
            "warning": "Connected components are descriptive; no component is automatically deleted"}


def create_candidate(reference, centerlines, out, radius=3.5, dilation=1):
    ref = image3d(reference)
    spec = read_json(centerlines)
    if spec.get("coordinate_system") != "NIFTI_RAS_MM":
        raise ValueError("Explicit NIFTI_RAS_MM coordinates are required")
    if spec.get("reference_sha256") != digest(reference):
        raise ValueError("Centerlines are not bound to this reference image")
    mask = tube_mask(ref.shape, ref.affine, spec["paths"], radius)
    mask = study_dilate(mask, ref.affine, dilation)
    path = Path(out) / "mask_candidate.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine), path)
    write_json(Path(out) / "mask_qc.json", mask_qc(mask, ref.affine))
    return path


def union_registered(reference, masks, registration_record, out):
    """Union already transformed masks; never treat affine equality as registration."""
    ref = image3d(reference)
    registration = read_json(registration_record)
    if registration.get("reference_sha256") != digest(reference):
        raise ValueError("Registration QC record belongs to a different reference")
    if not registration.get("anatomical_alignment_reviewed") or not registration.get("reviewer"):
        raise ValueError("Human anatomical registration review is required")
    if len(masks) < 2:
        raise ValueError("A longitudinal union requires at least two masks")
    expected = registration.get("registered_mask_sha256", [])
    if sorted(expected) != sorted(digest(p) for p in masks):
        raise ValueError("Registered masks differ from the reviewed versions")
    merged = np.zeros(ref.shape, dtype=bool)
    for path in masks:
        image = image3d(path)
        same_grid(ref, image)
        merged |= validate_binary(image)
    path = Path(out) / "mask_union.nii.gz"
    nib.save(nib.Nifti1Image(merged.astype(np.uint8), ref.affine), path)
    return path


REVIEW_ITEMS = ("registration", "void_coverage", "full_shaft", "spurious_components", "atlas_alignment")


def approval_record(mask, reference, atlas, reviewer, checks):
    ref, mi, ai = image3d(reference), image3d(mask), image3d(atlas)
    same_grid(ref, mi)
    same_grid(ref, ai)
    validate_binary(mi)
    if not reviewer.strip() or set(checks) != set(REVIEW_ITEMS):
        raise ValueError("All five explicit visual checks and a reviewer identifier are required")
    return {"schema_version": 1, "reviewer": reviewer, "checks": sorted(checks),
            "mask_sha256": digest(mask), "reference_sha256": digest(reference),
            "atlas_sha256": digest(atlas), "research_only": True}


def require_approval(approval, mask, reference, atlas):
    spec = read_json(approval)
    expected = approval_record(mask, reference, atlas, spec.get("reviewer", ""), spec.get("checks", []))
    if any(spec.get(key) != expected[key] for key in ("mask_sha256", "reference_sha256", "atlas_sha256")):
        raise ValueError("QC approval is stale; reviewed mask, atlas or reference changed")
