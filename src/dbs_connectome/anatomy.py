"""Locally installed anatomy tools, explicit label mapping and candidate CT masks.

The public implementation is a generalization, not a byte-identical study replay.
Neither geometric consistency nor candidate localization constitutes anatomical QC.
"""

from pathlib import Path
import csv
import os
import re

import nibabel as nib
import numpy as np
from scipy import ndimage

from .external import execute
from .gradients import load_nodes
from .masks import create_candidate
from .provenance import digest, image3d, read_json, same_grid, write_json


def label_image(path):
    image = image3d(path)
    data = np.asanyarray(image.dataobj)
    if np.any(data < 0) or not np.all(data == np.floor(data)):
        raise ValueError("Atlas input must contain nonnegative integer labels")
    return image, data.astype(np.int32)


def build_hybrid(surface, left, right, nodes_path, output):
    """Dense labels follow a supplied, cohort-frozen TSV, never voxel sorting.

    source_label is the original NextBrain ID or Schaefer ID (1..400).
    Only listed noncortical structures are included; there is no inferred GM/WM rule.
    Schaefer cortex takes precedence. Overlapping hemispheres cause refusal.
    """
    ref, seg = label_image(surface)
    li, l = label_image(left)
    ri, r = label_image(right)
    same_grid(ref, li)
    same_grid(ref, ri)
    nodes = load_nodes(nodes_path)
    if any("source_label" not in row for row in nodes):
        raise ValueError("Anatomy node TSV additionally requires source_label")
    if [int(row["label"]) for row in nodes] != list(range(1, len(nodes) + 1)):
        raise ValueError("New hybrid atlas uses dense labels 1..N in frozen TSV order")
    cortex = np.zeros(seg.shape, np.int32)
    lh = (seg >= 1001) & (seg <= 1200)
    rh = (seg >= 2001) & (seg <= 2200)
    cortex[lh], cortex[rh] = seg[lh] - 1000, seg[rh] - 1800
    result = np.zeros(seg.shape, np.int32)
    keys, missing = set(), []
    for row in nodes:
        source = int(row["source_label"])
        key = (row["hemisphere"], row["tissue"], source)
        if source < 1 or key in keys:
            raise ValueError("Invalid/duplicate source label mapping")
        keys.add(key)
        if row["tissue"] == "cortex":
            hemi = "L" if 1 <= source <= 200 else "R" if 201 <= source <= 400 else None
            if row["hemisphere"] != hemi or not row["name"].startswith("7Networks_"):
                raise ValueError("Cortical source label/name/hemisphere mismatch")
            selected = cortex == source
        else:
            if row["hemisphere"] not in ("L", "R") or 2001 <= source <= 2035:
                raise ValueError("NextBrain mapping must be bilateral and exclude DKT cortex")
            selected = ((l if row["hemisphere"] == "L" else r) == source) & (cortex == 0)
        if np.any(selected & (result != 0)):
            raise ValueError("Overlapping atlas assignments; inspect hemisphere registration")
        result[selected] = int(row["label"])
        if not selected.any():
            missing.append(row["name"])
    if missing:
        raise ValueError("Frozen node basis contains absent T1 labels: " + ", ".join(missing[:12]))
    nib.save(nib.Nifti1Image(result, ref.affine), output)
    return {"nodes": len(nodes), "cortical_voxels": int(np.count_nonzero(cortex)),
            "node_mapping_sha256": digest(nodes_path), "cortical_precedence": True}


def normalize_five_tissue(volumes, reference, output):
    ref = image3d(reference)
    values = []
    for path in volumes:
        image = image3d(path)
        same_grid(ref, image)
        values.append(np.asanyarray(image.dataobj))
    if len(values) != 5:
        raise ValueError("ACT requires exactly five tissue volumes")
    data = np.clip(np.stack(values, -1), 0, 1)
    total = data.sum(-1, keepdims=True)
    data = np.divide(data, total, out=np.zeros_like(data, dtype=float), where=total > 0)
    if not np.any(data[..., 2] > 0):
        raise ValueError("ACT white-matter compartment is empty")
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine), output)


def ct_centerlines(ct, brain_mask, reference, out, threshold_hu, expected_leads=2):
    """Conservative elongated-metal candidate detection on a registered CT grid.

    No automatic bone removal, tip identity, contact numbering or brainstem target
    inference. Ambiguous counts are rejected rather than selecting the largest two.
    QC must assess full shaft and b0 signal void; this is NOT study void-follow.
    """
    ref, scan, brain = image3d(reference), image3d(ct), image3d(brain_mask)
    same_grid(ref, scan)
    same_grid(ref, brain)
    if not np.isfinite(threshold_hu) or not 1000 <= threshold_hu <= 10000:
        raise ValueError("Review CT HU scaling and choose a metal threshold in 1000..10000 HU")
    if expected_leads not in (1, 2):
        raise ValueError("This candidate detector supports one or two leads")
    roi = ndimage.binary_dilation(np.asanyarray(brain.dataobj) > 0, iterations=2)
    labeled, count = ndimage.label((np.asanyarray(scan.dataobj) >= threshold_hu) & roi,
                                  ndimage.generate_binary_structure(3, 3))
    candidates = []
    for component in range(1, count + 1):
        vox = np.argwhere(labeled == component)
        if len(vox) < 4:
            continue
        points = nib.affines.apply_affine(ref.affine, vox)
        center = points.mean(0)
        _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
        along = (points - center) @ axes[0]
        length = float(np.ptp(along))
        ratio = float(singular[0] / max(singular[1], 1e-6))
        if length >= 10 and ratio >= 4:
            # Bin centroids preserve modest curvature without following arbitrary dark CSF.
            bins = np.floor((along - along.min()) / 4).astype(int)
            path = [points[bins == k].mean(0).tolist() for k in np.unique(bins)]
            if len(path) >= 2:
                candidates.append({"path": path, "length_mm": length, "elongation": ratio})
    write_json(Path(out) / "ct_candidates.json", {"threshold_hu": threshold_hu, "candidates": candidates,
        "expected_leads": expected_leads, "approved": False,
        "algorithm": "elongated_CT_components_NOT_study_void_follow"})
    if len(candidates) != expected_leads:
        raise ValueError(f"CT localization ambiguous: {len(candidates)} candidate shafts, expected {expected_leads}. Supply reviewed RAS centerlines instead.")
    path = Path(out) / "centerlines.json"
    write_json(path, {"coordinate_system": "NIFTI_RAS_MM", "reference_sha256": digest(reference),
                     "paths": [c["path"] for c in candidates], "approved": False})
    return path


def validate_resources(resources):
    required = ("freesurfer_home", "fsaverage", "lh_annotation", "rh_annotation", "nextbrain_atlas", "nodes")
    if not all(resources.get(k) for k in required):
        raise ValueError("Configure installed FreeSurfer, fsaverage, both Schaefer annotations, NextBrain atlas and frozen node TSV")
    paths = {k: Path(resources[k]).expanduser().resolve(strict=True) for k in required}
    for k in ("lh_annotation", "rh_annotation", "nodes"):
        if not paths[k].is_file():
            raise ValueError("Missing resource file: " + k)
    annotation_names = {}
    for hemi in ("lh", "rh"):
        _, _, names = nib.freesurfer.read_annot(paths[hemi + "_annotation"])
        parcels = [n.decode() for n in names if n.decode().startswith("7Networks_")]
        if len(parcels) != 200 or any(not n.startswith("7Networks_" + hemi.upper() + "_") for n in parcels):
            raise ValueError("Expected Schaefer400 seven-network annotation for " + hemi)
        annotation_names[hemi] = [n.decode() for n in names]
    script = paths["freesurfer_home"] / "python/packages/ERC_bayesian_segmentation/scripts/segment.py"
    if not script.is_file() or not (paths["nextbrain_atlas"] / "size.npy").is_file():
        raise ValueError("Install the full NextBrain atlas and its FreeSurfer segment.py first; no resources are downloaded")
    for row in load_nodes(paths["nodes"]):
        if "source_label" not in row:
            raise ValueError("Frozen node TSV requires a source_label column for atlas assembly")
        if row["tissue"] == "cortex":
            label = int(row["source_label"])
            hemi, index = ("lh", label) if 1 <= label <= 200 else ("rh", label - 200)
            if not 1 <= index <= 200 or annotation_names[hemi][index] != row["name"]:
                raise ValueError("Node names/source labels do not match the actual surface annotation order")
    paths["nextbrain_script"] = script
    return paths


def anatomical_plan(t1, transform, reference, resources, out, threads):
    out = Path(out).resolve()
    # Installed NextBrain invokes shell strings internally. Refuse unsupported paths
    # instead of interpolating user paths into its unquoted internal commands.
    for path in (out, resources["freesurfer_home"], resources["nextbrain_atlas"]):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
            raise ValueError("Installed NextBrain requires a processing/resource path without spaces or shell metacharacters")
    p = lambda name: str(out / name)
    fs = resources["freesurfer_home"]
    subject = "analysis_subject"
    sd = out / "subjects"
    sub = sd / subject
    mri = sub / "mri"
    # Do not run two hemispheres simultaneously with N OpenMP workers each.
    commands = [[str(fs / "bin/recon-all"), "-sd", str(sd), "-s", subject, "-i", str(t1),
                 "-all", "-openmp", str(threads)]]
    for hemi in ("lh", "rh"):
        commands.append([str(fs / "bin/mri_surf2surf"), "--srcsubject", "fsaverage", "--trgsubject", subject,
                         "--hemi", hemi, "--sval-annot", str(resources[hemi + "_annotation"]),
                         "--tval", str(sub / "label" / (hemi + ".schaefer400_7net.annot"))])
    commands.append([str(fs / "bin/mri_aparc2aseg"), "--s", subject, "--annot", "schaefer400_7net", "--o", p("surface.mgz")])
    for hemi in ("left", "right"):
        commands.append([str(fs / "bin/fspython"), str(resources["nextbrain_script"]), "--i", str(mri / "T1.mgz"),
            "--atlas_mode", "full", "--atlas_dir", str(resources["nextbrain_atlas"]), "--hemi", hemi[0],
            "--i_seg", p("nextbrain/SynthSeg.mgz"), "--i_field", p("nextbrain/MNI_registration.mgz"),
            "--o", p("nextbrain/seg_" + hemi + ".mgz"), "--o_vol", p("nextbrain/vols_" + hemi + ".csv"),
            "--threads", str(threads), "--cpu", "--gmm_mode", "1mm", "--bf_mode", "dct"])
    for name, source in (("surface", p("surface.mgz")), ("left", p("nextbrain/seg_left.mgz")), ("right", p("nextbrain/seg_right.mgz"))):
        commands.append([str(fs / "bin/mri_vol2vol"), "--mov", source, "--targ", str(t1),
                         "--o", p(name + "_in_t1.nii.gz"), "--regheader", "--nearest"])
    commands.append(["5ttgen", "freesurfer", str(mri / "aparc+aseg.mgz"), p("5tt_t1.mif"), "-nthreads", str(threads)])
    for i in range(5):
        commands += [["mrconvert", p("5tt_t1.mif"), p(f"tissue_{i}.nii.gz"), "-coord", "3", str(i), "-axes", "0,1,2"],
            ["antsApplyTransforms", "-d", "3", "-i", p(f"tissue_{i}.nii.gz"), "-r", str(reference),
             "-o", p(f"tissue_{i}_dwi.nii.gz"), "-n", "Linear", "-t", str(transform)]]
    return commands
