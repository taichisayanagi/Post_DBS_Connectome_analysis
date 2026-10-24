"""Immutable mask corrections and bidirectional native-grid union candidates.

Neither a correction recipe nor a successful registration grants image QC.
Original masks, registrations and patient images are never overwritten.
"""
from pathlib import Path
import html

import nibabel as nib
import numpy as np

from .external import check_threads, execute
from .masks import mask_qc, study_dilate, validate_binary
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, image3d,
                         read_json, record, require_separate_output, same_grid, write_json)
from .qc import browser_report


def recipe_paths(spec):
    return [spec[k] for k in ("reference", "mask", "add", "remove", "review_evidence") if spec.get(k)]


def apply_repair(reference, mask, add=None, remove=None):
    ref, original = image3d(reference), image3d(mask)
    same_grid(ref, original)
    before = validate_binary(original)
    overlays = []
    for path in (add, remove):
        if path:
            image = image3d(path)
            same_grid(ref, image)
            values = np.asanyarray(image.dataobj)
            if not np.isin(values, (0, 1)).all():
                raise ValueError("Repair overlays must be binary")
            overlays.append(values.astype(bool))
        else:
            overlays.append(np.zeros(ref.shape, bool))
    plus, minus = overlays
    if np.any(plus & minus):
        raise ValueError("Add and remove overlays conflict")
    if np.any(minus & ~before):
        raise ValueError("Removal overlay includes voxels absent from the original mask")
    if np.any(plus & before):
        raise ValueError("Addition overlay includes already masked voxels")
    after = (before | plus) & ~minus
    if not after.any():
        raise ValueError("Repair would remove the entire mask")
    return after, {"added_voxels": int(plus.sum()), "removed_voxels": int(minus.sum())}


def repair_mask(recipe, out):
    spec = read_json(recipe)
    if not spec.get("reason") or not (spec.get("add") or spec.get("remove")):
        raise ValueError("Provide explicit correction overlays and a reason")
    paths = [recipe, *recipe_paths(spec)]
    for p in paths:
        require_separate_output(out, Path(p).parent)
    for key in ("reference", "mask", "add", "remove", "review_evidence"):
        if spec.get(key) and digest(spec[key]) != spec.get(key+"_sha256"):
            raise ValueError("Correction input changed: " + key)
    frozen = freeze_inputs(paths)
    data, stats = apply_repair(spec["reference"], spec["mask"], spec.get("add"), spec.get("remove"))
    ref = image3d(spec["reference"])
    out = Path(out)
    target = out / "mask_repaired_candidate.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.uint8), ref.affine), target)
    browser_report(spec["reference"], target, out / "mask_review.html")
    write_json(out / "mask_lineage.json", {"status": "needs_qc", "algorithm": "explicit_binary_overlay",
        "parent_mask_sha256": digest(spec["mask"]), "output_sha256": digest(target),
        "reason": spec["reason"], "changes": stats, "geometry": mask_qc(data, ref.affine),
        "review_evidence": spec.get("review_evidence"), "automatic_qc_approval": False})
    assert_inputs_unchanged(frozen)
    record(out, "mask-repair", paths, {"manual_correction_replayed": True}, sorted(out.glob('*')))


def shared_mask_plan(first_reference, second_reference, first_mask, second_mask, out, threads):
    check_threads(threads)
    refs = [image3d(p) for p in (first_reference, second_reference)]
    for ref, path in zip(refs, (first_mask, second_mask)):
        image = image3d(path)
        same_grid(ref, image)
        validate_binary(image)
        study_dilate(np.asanyarray(image.dataobj), ref.affine, 0)
    out = Path(out)
    transform = out / "post1_to_post2_0GenericAffine.mat"
    commands = [["antsRegistrationSyNQuick.sh", "-d", "3", "-f", str(second_reference),
        "-m", str(first_reference), "-o", str(out / "post1_to_post2_"), "-t", "a", "-n", str(threads)],
        ["antsApplyTransforms", "-d", "3", "-i", str(first_mask), "-r", str(second_reference),
         "-o", str(out / "first_in_second.nii.gz"), "-n", "NearestNeighbor", "-t", str(transform)],
        ["antsApplyTransforms", "-d", "3", "-i", str(second_mask), "-r", str(first_reference),
         "-o", str(out / "second_in_first.nii.gz"), "-n", "NearestNeighbor", "-t", f"[{transform},1]"]]
    return commands


def shared_masks(first_reference, second_reference, first_mask, second_mask, out, threads, execute_tools=False):
    paths = [first_reference, second_reference, first_mask, second_mask]
    for p in paths:
        require_separate_output(out, Path(p).parent)
    frozen = freeze_inputs(paths)
    out = Path(out)
    commands = shared_mask_plan(*paths, out, threads)
    write_json(out / "plan.json", commands)
    if not execute_tools:
        record(out, "shared-masks", paths, {"threads": threads}, [out / "plan.json"], "planned_not_executed")
        return
    logs = out / "registration_commands"
    logs.mkdir()
    execute(commands, logs, threads)
    outputs = []
    volumes = []
    for name, reference, own, other in (
        ("post1", first_reference, first_mask, out / "second_in_first.nii.gz"),
        ("post2", second_reference, second_mask, out / "first_in_second.nii.gz")):
        ref = image3d(reference)
        a, b = image3d(own), image3d(other)
        same_grid(ref, a); same_grid(ref, b)
        union = validate_binary(a) | validate_binary(b)
        raw = out / (name+"_union.nii.gz")
        dilated = out / (name+"_union_dil1.nii.gz")
        nib.save(nib.Nifti1Image(union.astype(np.uint8), ref.affine), raw)
        nib.save(nib.Nifti1Image(study_dilate(union,ref.affine,1).astype(np.uint8),ref.affine),dilated)
        browser_report(reference, dilated, out / (name+"_mask_review.html"))
        volumes.append(mask_qc(union, ref.affine)["volume_mm3"])
        outputs.append({"visit":name,"union":str(raw),"dilated":str(dilated),"sha256":digest(dilated)})
    panels = []
    for visit in ("post1", "post2"):
        document = (out / (visit + "_mask_review.html")).read_text(encoding="utf-8")
        panels.append('<h2>' + visit + '</h2><iframe title="' + visit +
                      ' native-grid review" style="width:100%;height:850px;border:1px solid #bbb" srcdoc="' +
                      html.escape(document, quote=True) + '"></iframe>')
    (out / "mask_review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared-mask review</title>'
        '<body style="background:white;font-family:system-ui;margin:24px">'
        '<h1>Review both native grids</h1><p>Successful registration does not grant QC approval. '
        'Review registration and electrode coverage in BOTH visits. Each atlas-bound mask requires '
        'its own approval.</p>' + ''.join(panels) + '</body></html>', encoding="utf-8")
    assert_inputs_unchanged(frozen)
    write_json(out / "shared_masks.json", {"status":"needs_qc","outputs":outputs,
        "registration":"post1 moving to post2 fixed; rigid plus affine; inverse to post1; nearest-neighbor masks",
        "dilation":"one 6-neighbour pass on 2-mm grid AFTER union",
        "volume_difference_percent":100*abs(volumes[1]-volumes[0])/np.mean(volumes),
        "automatic_qc_approval":False,"source_bytes_unchanged":True,
        "next":"Review BOTH native-grid registrations and mask coverage; approve each atlas-bound mask separately."})
    record(out,"shared-masks",paths,{"threads":threads,"dilation_passes":1},sorted(out.glob('*.*')))
