"""Image-guided electrode candidates; metal, MRI void and exclusion stay distinct.

These geometric candidates are NOT contact reconstructions, electric fields, or
the study's original void-follow implementation. Every candidate needs human QC.
"""
import base64
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .anatomy import ct_centerlines
from .masks import mask_qc, study_dilate, tube_mask, validate_binary
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, image3d,
                         read_json, record, require_separate_output, same_grid, write_json)
from .qc import browser_report


def validate_paths(paths, reference):
    if not isinstance(paths, list) or not 1 <= len(paths) <= 8:
        raise ValueError("Supply 1..8 centerlines")
    inverse = np.linalg.inv(reference.affine)
    for path in paths:
        p = np.asarray(path, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not 2 <= len(p) <= 256 or not np.isfinite(p).all():
            raise ValueError("Each lead needs 2..256 finite RAS-mm points")
        vox = nib.affines.apply_affine(inverse, p)
        if np.any(vox < -.5) or np.any(vox > np.asarray(reference.shape) - .5):
            raise ValueError("Centerline point outside reference field of view")
        if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) < .01):
            raise ValueError("Consecutive points must differ by at least 0.01 mm")
    return paths


def checked_images(reference, images):
    ref = image3d(reference)
    if np.prod(ref.shape) > 32_000_000:
        raise ValueError("Image exceeds the interactive size limit; use a reviewed analysis grid")
    result = {}
    for name, path in images.items():
        if path:
            im = image3d(path)
            same_grid(ref, im)
            result[name] = im
    return ref, result


def viewer_payload(reference, images=None, centerlines=None):
    """Only reorient display arrays; all editing coordinates are world RAS mm."""
    ref, auxiliary = checked_images(reference, images or {})
    r = nib.as_closest_canonical(ref)
    payload = {"shape": list(r.shape), "affine": r.affine.tolist(),
               "inverse_affine": np.linalg.inv(r.affine).tolist(),
               "reference_sha256": digest(reference), "backgrounds": {},
               "image_hashes": {"reference": digest(reference)}, "paths": []}
    for name, im in {"Reference": ref, **auxiliary}.items():
        data = nib.as_closest_canonical(im).get_fdata()
        lo, hi = np.percentile(data, [1, 99.9])
        data = np.clip((data - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        payload["backgrounds"][name] = base64.b64encode(data.tobytes(order="F")).decode()
        if name in (images or {}):
            payload["image_hashes"][name] = digest(images[name])
    if centerlines:
        spec = read_json(centerlines)
        if spec.get("reference_sha256") != digest(reference) or spec.get("coordinate_system") != "NIFTI_RAS_MM":
            raise ValueError("Centerlines belong to a different reference or coordinate system")
        payload["paths"] = validate_paths(spec["paths"], ref)
    return payload


def _number(spec, name, default, low, high):
    value = spec.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return float(value)


def make_electrode(reference, spec, out, ct=None, mri=None, brain_mask=None):
    """CT-seeded or manually placed paths, MRI-void union, then explicit edits.

    MRI void is detected only inside the seed corridor and reviewed brain mask.
    A connected dark component must touch the centerline tube. No darkness alone
    is interpreted as metal, no lead/contact orientation is inferred for VTA.
    """
    out = Path(out).resolve()
    image_paths = {k: str(v) for k, v in {"CT": ct, "MRI": mri, "Brain": brain_mask}.items() if v}
    for p in [reference, *image_paths.values()]:
        require_separate_output(out, Path(p).resolve().parent)
    inputs = [reference, *image_paths.values()]
    before = freeze_inputs(inputs)
    ref, ims = checked_images(reference, image_paths)
    if spec.get("coordinate_system") != "NIFTI_RAS_MM" or spec.get("reference_sha256") != digest(reference):
        raise ValueError("Image changed or missing NIFTI_RAS_MM reference binding; reload editor")
    if spec.get("image_hashes") != {"reference": digest(reference), **{k: digest(v) for k, v in image_paths.items()}}:
        raise ValueError("An auxiliary image changed or was not displayed; reload editor")
    if spec.get("registration_reviewed") is not True:
        raise ValueError("Review CT/MRI/reference anatomical registration before creating candidates")
    radius = _number(spec, "radius_mm", 3.5, .5, 10)
    corridor = _number(spec, "corridor_mm", 7, radius, 15)
    fraction = _number(spec, "dark_fraction", .35, .01, .9)
    dilation = spec.get("dilation_passes", 1)
    if isinstance(dilation, bool) or not isinstance(dilation, int) or not 0 <= dilation <= 3:
        raise ValueError("Choose 0..3 dilation passes")
    mode = spec.get("mode", "manual")
    if mode not in ("manual", "ct", "virtual"):
        raise ValueError("Unknown electrode candidate mode")
    if mode == "ct":
        if not ct or not brain_mask or spec.get("ct_hu_reviewed") is not True:
            raise ValueError("CT candidates require CT, brain mask and reviewed HU scaling")
        count = spec.get("expected_leads", 2)
        if isinstance(count, bool) or count not in (1, 2):
            raise ValueError("CT localization supports one or two reviewed lead candidates")
        candidate = ct_centerlines(ct, brain_mask, reference, out,
                                  _number(spec, "ct_threshold_hu", 2000, 1000, 10000), count)
        paths = read_json(candidate)["paths"]
    else:
        paths = spec.get("paths")
    validate_paths(paths, ref)
    core = tube_mask(ref.shape, ref.affine, paths, radius)
    near = tube_mask(ref.shape, ref.affine, paths, corridor)
    metal = np.zeros(ref.shape, bool)
    if ct and mode != "virtual":
        if spec.get("ct_hu_reviewed") is not True:
            raise ValueError("Confirm CT HU scaling even with manually edited paths")
        metal = near & (ims["CT"].get_fdata() >= _number(spec, "ct_threshold_hu", 2000, 1000, 10000))
    void = np.zeros(ref.shape, bool)
    warnings = ["Candidate only: review all slices, registration, shaft and atlas before approval.",
                "MRI void is not the metal surface. CT centerline is not a contact/tip reconstruction."]
    void_threshold = None
    if spec.get("include_mri_void"):
        if mode == "virtual":
            raise ValueError("Virtual exclusion geometry must not be inferred from a real MRI void")
        if not mri or not brain_mask:
            raise ValueError("MRI void extraction requires registered MRI and a reviewed brain mask")
        brain = validate_binary(ims["Brain"])
        data = ims["MRI"].get_fdata()
        surround = ndimage.binary_dilation(near, iterations=2) & ~near & brain & (data > 0)
        if surround.sum() < 20:
            raise ValueError("Insufficient surrounding MRI signal to estimate a void threshold")
        void_threshold = float(np.median(data[surround]) * fraction)
        dark = near & brain & (data < void_threshold)
        labels, _ = ndimage.label(dark, ndimage.generate_binary_structure(3, 1))
        touched = np.unique(labels[core & dark])
        void = np.isin(labels, touched[touched > 0])
        if not void.any():
            warnings.append("No connected MRI void found; mask contains geometry/metal only.")
        if np.any(void & ~ndimage.binary_erosion(near)):
            warnings.append("MRI void reaches the search corridor boundary; review possible under-coverage.")
    mask = study_dilate(core | metal | void, ref.affine, dilation)
    edits = spec.get("edits", [])
    if not isinstance(edits, list) or len(edits) > 128:
        raise ValueError("At most 128 explicit spherical mask edits are allowed")
    # Edit spheres are rasterized in physical space. Edits are never applied to inputs.
    for edit in edits:
        position = np.asarray(edit.get("ras_mm"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid edit position")
        rad = _number(edit, "radius_mm", 2, .5, 10)
        action = edit.get("action")
        if action not in ("add", "erase") or (action == "erase" and not str(edit.get("reason", "")).strip()):
            raise ValueError("Erase edits require a recorded reason")
        sphere = np.zeros(ref.shape, bool)
        for z in range(ref.shape[2]):
            ij = np.indices(ref.shape[:2]).reshape(2, -1).T
            xyz = nib.affines.apply_affine(ref.affine, np.column_stack((ij, np.full(len(ij), z))))
            sphere[:, :, z] = (np.linalg.norm(xyz-position, axis=1) <= rad).reshape(ref.shape[:2])
        if not sphere.any():
            raise ValueError("Edit does not intersect the reference grid")
        mask = mask | sphere if action == "add" else mask & ~sphere
    if not mask.any():
        raise ValueError("Edits removed the entire mask")
    if np.any((metal | void) & ~mask):
        warnings.append("Manual erasure removed detected metal/void voxels; inspect explicitly before QC.")
    outputs = {}
    for name, values in (("mask", mask), ("metal", metal), ("void", void)):
        path = out / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(values.astype(np.uint8), ref.affine), path)
        outputs[name] = str(path)
    centerline_path = out / "reviewed_geometry.json"
    write_json(centerline_path, {"coordinate_system": "NIFTI_RAS_MM", "reference_sha256": digest(reference),
                                "paths": paths, "approved": False, "source_mode": mode})
    outputs["centerlines"] = str(centerline_path)
    write_json(out / "request.json", spec)
    browser_report(reference, outputs["mask"], out / "mask_review.html",
                   backgrounds={k: v for k, v in image_paths.items() if k != "Brain"})
    result = {"status": "needs_qc", "reference": str(Path(reference).resolve()),
              "reference_sha256": digest(reference), "outputs": outputs,
              "hashes": {p: digest(p) for p in outputs.values()}, "paths": paths,
              "algorithm": "geometry_CT_corridor_connected_MRI_void_v1", "mode": mode,
              "void_threshold_intensity": void_threshold, "warnings": warnings,
              "mask_qc": mask_qc(mask, ref.affine), "automatic_qc_approval": False,
              "study_pipeline_equivalence": "not_validated", "clinical_validity": False}
    write_json(out / "electrode_result.json", result)
    assert_inputs_unchanged(before)
    record(out, "electrode", inputs, {"spec": spec, "source_bytes_unchanged": True},
           [out / "electrode_result.json", out / "request.json", out / "mask_review.html", *outputs.values()])
    return result


def verified_electrode(run, reference):
    from .cli import checked_artifact
    path, _ = checked_artifact(run, "electrode_result.json", "electrode")
    result = read_json(path)
    if result["reference_sha256"] != digest(reference):
        raise ValueError("Electrode candidate belongs to a different reference")
    for path in result["outputs"].values():
        if digest(path) != result["hashes"].get(path):
            raise ValueError("Electrode candidate changed after creation")
    same_grid(image3d(reference), image3d(result["outputs"]["mask"]))
    return result
