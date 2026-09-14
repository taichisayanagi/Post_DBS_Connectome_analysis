"""Automatic artifact handoffs with a mandatory human QC checkpoint."""

from argparse import Namespace
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from .anatomy import anatomical_plan, build_hybrid, ct_centerlines, normalize_five_tissue, validate_resources
from .external import check_threads, execute, validate_mif_grid
from .gradients import load_nodes
from .intake import verified_output
from .masks import create_candidate, require_approval
from .provenance import assert_inputs_unchanged, digest, freeze_inputs, image3d, read_json, record, require_separate_output, write_json
from .qc import browser_report
from .reconstruction import selected_artifacts


def verified_reconstruction(run):
    result = read_json(verified_output(run, "reconstruction.json", "reconstruct"))
    for key, path in result["outputs"].items():
        if Path(run).resolve() not in Path(path).resolve(strict=True).parents or result["hashes"].get(path) != digest(path):
            raise ValueError("Reconstruction artifact changed: " + key)
    return result


def prepare_session(reconstruction_run, settings, out, threads, execute_tools=False):
    check_threads(threads)
    out = Path(out).resolve()
    require_separate_output(out, reconstruction_run)
    recon = verified_reconstruction(reconstruction_run)
    if recon.get("source_root"):
        require_separate_output(out, recon["source_root"])
    resources = validate_resources(settings["resources"])
    for key in ("freesurfer_home", "fsaverage", "nextbrain_atlas"):
        require_separate_output(out, resources[key])
    if settings.get("reconstruction_reviewed") is not True:
        raise ValueError("Review DWI/eddy, b-vectors and CT/T1/DWI registration before anatomical processing")
    artifacts = selected_artifacts(recon["import_run"], {"artifacts": recon["artifacts"]})
    t1 = artifacts["t1"]["image"]
    outputs = recon["outputs"]
    mode = settings.get("mask_mode")
    if mode == "ct_candidate":
        if "ct_in_b0" not in outputs or settings.get("ct_hu_reviewed") is not True:
            raise ValueError("CT candidate localization requires registered CT and explicit HU-scaling review")
    elif mode == "centerlines":
        if not settings.get("centerlines"):
            raise ValueError("Select reviewed centerlines bound to the reconstructed b0 reference")
    else:
        raise ValueError("Select ct_candidate or reviewed centerlines; MRI-only automatic localization is not validated")
    commands = anatomical_plan(t1, outputs["t1_transform"], outputs["reference"], resources, out, threads)
    write_json(out / "plan.json", commands)
    write_json(out / "settings.json", settings)
    inputs = [Path(reconstruction_run) / "reconstruction.json", t1, *outputs.values(),
              resources["nodes"], resources["lh_annotation"], resources["rh_annotation"],
              resources["nextbrain_script"], resources["nextbrain_atlas"] / "size.npy"]
    if mode == "centerlines":
        inputs.append(settings["centerlines"])
    if not execute_tools:
        record(out, "prepare-session", inputs, {"threads": threads}, [out / "plan.json", out / "settings.json"], "planned_not_executed")
        return
    before = freeze_inputs(inputs)
    write_json(out / "input_hashes_before.json", before)
    (out / "subjects").mkdir(mode=0o700)
    # FreeSurfer reads this template; all subject writes target the private subject.
    (out / "subjects/fsaverage").symlink_to(resources["fsaverage"], target_is_directory=True)
    (out / "nextbrain").mkdir(mode=0o700)
    logs = out / "anatomy_commands"
    logs.mkdir()
    execute(commands, logs, threads, {"SUBJECTS_DIR": str(out / "subjects"), "FREESURFER_HOME": str(resources["freesurfer_home"])})
    hybrid = out / "atlas_t1.nii.gz"
    build_hybrid(out / "surface_in_t1.nii.gz", out / "left_in_t1.nii.gz", out / "right_in_t1.nii.gz", resources["nodes"], hybrid)
    atlas = out / "atlas_dwi.nii.gz"
    warp = [["antsApplyTransforms", "-d", "3", "-i", str(hybrid), "-r", outputs["reference"],
             "-o", str(atlas), "-n", "NearestNeighbor", "-t", outputs["t1_transform"]]]
    warp_logs = out / "atlas_registration"
    warp_logs.mkdir()
    execute(warp, warp_logs, threads)
    five = out / "five_tissue_dwi.nii.gz"
    normalize_five_tissue([out / f"tissue_{i}_dwi.nii.gz" for i in range(5)], outputs["reference"], five)
    centerlines = settings.get("centerlines")
    if mode == "ct_candidate":
        centerlines = ct_centerlines(outputs["ct_in_b0"], outputs["brain_mask"], outputs["reference"], out,
                                    settings.get("ct_threshold_hu", 2000), settings.get("expected_leads", 2))
    mask = create_candidate(outputs["reference"], centerlines, out,
                            settings.get("radius_mm", 3.5), settings.get("dilation_passes", 1))
    nodes = out / "nodes.tsv"
    shutil.copyfile(resources["nodes"], nodes)
    present = set(np.unique(np.asanyarray(image3d(atlas).dataobj)).astype(int))
    missing = [r["name"] for r in load_nodes(nodes) if int(r["label"]) not in present]
    backgrounds = {"T1": outputs["t1_in_b0"]}
    if "ct_in_b0" in outputs:
        backgrounds["CT"] = outputs["ct_in_b0"]
    browser_report(outputs["reference"], mask, out / "mask_review.html", atlas=atlas,
                   backgrounds=backgrounds, five_tissue=five)
    bundle = {"reference": outputs["reference"], "fod": outputs["fod"], "five_tissue": str(five),
              "atlas": str(atlas), "mask": str(mask), "nodes": str(nodes)}
    assert_inputs_unchanged(before)
    write_json(out / "source_integrity.json", {"source_bytes_unchanged": True, "input_count": len(before)})
    write_json(out / "prepared.json", {"status": "needs_qc", "outputs": bundle,
        "source_root": recon.get("source_root"),
        "hashes": {p: digest(p) for p in bundle.values()}, "missing_dwi_nodes": missing,
        "reconstruction_run": str(Path(reconstruction_run).resolve()),
        "mask_algorithm": "CT_or_reviewed_centerline_tube_NOT_study_void_follow",
        "automatic_qc_approval": False, "study_pipeline_equivalence": "not_validated",
        "next": "Inspect b0/CT/atlas/5TT and mask coverage, then approve exact files. No tractography or clinical inference has run."})
    record(out, "prepare-session", inputs, {"threads": threads, "mask_algorithm": mode, "manual_qc_required": True},
           [out / "prepared.json", out / "settings.json", out / "source_integrity.json", out / "mask_review.html"])


def prepared_paths(run):
    prepared = read_json(verified_output(run, "prepared.json", "prepare-session"))
    for key, path in prepared["outputs"].items():
        if prepared["hashes"].get(path) != digest(path):
            raise ValueError("Prepared output changed; repeat preparation/QC: " + key)
    if prepared["missing_dwi_nodes"]:
        raise ValueError("Atlas has missing nodes on the DWI grid; establish a common valid node basis across visits before proceeding")
    return prepared["outputs"]


def finish_session(prepared_run, approval, out, threads, streamlines=1000000, reference_run=None, execute_tools=False):
    """After QC: ACT -> mask exclusion -> refit SIFT2 -> connectome -> G1..G4.

    Without an independent/training reference the output is an unaligned single-
    session embedding, not a certified longitudinal comparison.
    """
    from .cli import run
    check_threads(threads)
    if isinstance(streamlines, bool) or not isinstance(streamlines, int) or not 1000 <= streamlines <= 10000000:
        raise ValueError("Choose 1000..10000000 streamlines; reduced counts are smoke tests, not study analyses")
    out = Path(out).resolve()
    require_separate_output(out, prepared_run)
    original = read_json(Path(prepared_run) / "prepared.json").get("source_root")
    if original:
        require_separate_output(out, original)
    paths = prepared_paths(prepared_run)
    require_approval(approval, paths["mask"], paths["reference"], paths["atlas"])
    source_inputs = [Path(prepared_run) / "prepared.json", approval, *paths.values()]
    before = freeze_inputs(source_inputs)
    write_json(out / "input_hashes_before.json", before)
    tracks = out / "tractography/tracks.tck"
    gmwmi = out / "tractography/gmwmi.mif"
    commands = [["5tt2gmwmi", paths["five_tissue"], str(gmwmi), "-nthreads", str(threads)],
        ["tckgen", paths["fod"], str(tracks), "-algorithm", "iFOD2", "-act", paths["five_tissue"],
         "-backtrack", "-seed_gmwmi", str(gmwmi), "-select", str(streamlines), "-cutoff", "0.06",
         "-step", "0.5", "-downsample", "1", "-nthreads", str(threads)]]
    config = {**paths, "tractogram": str(tracks), "approval": str(Path(approval).resolve())}
    write_json(out / "connectome_config.json", config)
    write_json(out / "plan.json", {"tractography": commands,
        "then": ["tckedit electrode exclusion", "SIFT2 refit", "tck2connectome", "hemisphere G1-G4"],
        "reference_run": str(reference_run) if reference_run else None,
        "warning": "No reference supplied: single-session unaligned embedding only" if not reference_run else None})
    if not execute_tools:
        record(out, "finish-session", source_inputs, {"threads": threads, "streamlines": streamlines},
               [out / "plan.json", out / "connectome_config.json"], "planned_not_executed")
        return
    for name in ("tractography", "connectome", "gradient"):
        (out / name).mkdir(mode=0o700)
    validate_mif_grid(paths["fod"], paths["reference"], out / "tractography")
    validate_mif_grid(paths["five_tissue"], paths["reference"], out / "tractography")
    execute(commands, out / "tractography", threads)
    run(Namespace(command="connectome", config=str(out / "connectome_config.json"), threads=threads, execute=True), out / "connectome")
    if reference_run:
        run(Namespace(command="embed", matrix=str(out / "connectome/connectome.npy"), nodes=paths["nodes"],
                      reference_run=str(reference_run), connectome_run=str(out / "connectome")), out / "gradient")
    else:
        run(Namespace(command="reference", matrix=str(out / "connectome/connectome.npy"), nodes=paths["nodes"]), out / "gradient")
    assert_inputs_unchanged(before)
    write_json(out / "source_integrity.json", {"source_bytes_unchanged": True, "input_count": len(before)})
    write_json(out / "session_result.json", {"status": "completed", "connectome_run": str(out / "connectome"),
        "gradient_run": str(out / "gradient"), "alignment": "reference_aligned" if reference_run else "UNALIGNED_SINGLE_SESSION",
        "nodes": paths["nodes"], "streamlines_requested": streamlines,
        "study_pipeline_equivalence": "not_validated", "clinical_validity": False})
    record(out, "finish-session", source_inputs, {"threads": threads, "streamlines": streamlines},
           [out / "session_result.json", out / "source_integrity.json", out / "connectome/provenance.json", out / "gradient/provenance.json"])
