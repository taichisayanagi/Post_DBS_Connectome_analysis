"""Acquisition-gated DWI/FOD and rigid registration adapter.

This generalization is NOT a byte-equivalent replay of the manuscript pipeline.
It stops before atlas/mask approval and never silently selects acquisition metadata.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from .environment import discover, tool_path
from .external import check_threads, execute
from .intake import verified_output
from .provenance import assert_inputs_unchanged, digest, freeze_inputs, read_json, record, require_separate_output, write_json
from .qc import browser_report


def selected_artifacts(import_run, choices):
    stage = read_json(Path(import_run) / "provenance.json")["stage"]
    if stage not in ("import-session", "import-nifti"):
        raise ValueError("Expected a completed DICOM or NIfTI import")
    converted = read_json(verified_output(import_run, "converted.json", stage))
    artifacts = {a["id"]: a for a in converted["artifacts"]}
    selected = choices.get("artifacts", {})
    if not {"t1", "dwi"}.issubset(selected) or not set(selected).issubset({"t1", "dwi", "ct", "t2", "reverse_b0"}):
        raise ValueError("Select T1 and raw DWI converted images")
    result = {}
    for role, identifier in selected.items():
        if identifier not in artifacts or artifacts[identifier]["role"] != role:
            raise ValueError("Converted image does not match its assigned role")
        artifact = artifacts[identifier]
        for path, expected in artifact["hashes"].items():
            if Path(import_run).resolve() not in Path(path).resolve(strict=True).parents or digest(path) != expected:
                raise ValueError("Converted image or metadata changed; re-import before processing")
        result[role] = artifact
    return result


def check_image_header(path, dimensions):
    image = nib.load(path)
    if len(image.shape) not in dimensions or min(image.shape[:3]) < 2:
        raise ValueError("Unexpected image dimensions: " + Path(path).name)
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if not qc and not sc or qc and sc and not np.allclose(q, s, atol=1e-3):
        raise ValueError("Missing or conflicting NIfTI spatial transforms")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid image affine")
    return image


def diffusion_summary(artifact):
    image = check_image_header(artifact["image"], (4,))
    if not all(key in artifact for key in ("bval", "bvec", "json")):
        raise ValueError("DWI requires a metadata JSON, b-values and b-vectors; ADC/FA are not valid inputs")
    bval = np.atleast_1d(np.loadtxt(artifact["bval"]))
    bvec = np.loadtxt(artifact["bvec"])
    if bval.shape != (image.shape[3],) or bvec.shape != (3, image.shape[3]):
        raise ValueError("DWI volume count differs from b-values or 3×N b-vectors")
    if not np.isfinite(bval).all() or not np.isfinite(bvec).all() or np.any(bval < 0):
        raise ValueError("Invalid diffusion gradient numbers")
    baseline = bval <= 50
    if not baseline.any() or np.count_nonzero(~baseline) < 6:
        raise ValueError("At least one b0 and six diffusion directions are required")
    norm = np.linalg.norm(bvec[:, ~baseline], axis=0)
    if np.any(np.abs(norm - 1) > .05):
        raise ValueError("Diffusion b-vectors are not unit length; do not silently normalize")
    direction = bvec[:, ~baseline].T
    design = np.column_stack([direction[:, 0]**2, direction[:, 1]**2, direction[:, 2]**2,
                              direction[:, 0]*direction[:, 1], direction[:, 0]*direction[:, 2],
                              direction[:, 1]*direction[:, 2]])
    if np.linalg.matrix_rank(design) < 6:
        raise ValueError("Diffusion directions are rank-deficient")
    clusters = []
    for value in np.sort(bval[~baseline]):
        if not clusters or abs(value - np.mean(clusters[-1])) > 100:
            clusters.append([])
        clusters[-1].append(float(value))
    return {"volumes": len(bval), "b0_volumes": int(baseline.sum()),
            "shells": [{"b": round(float(np.mean(c)), 1), "directions": len(c)} for c in clusters],
            "shell_tolerance_s_mm2": 100, "b0_threshold_s_mm2": 50,
            "orientation_verified": False}


def reconstruction_plan(import_run, choices, out, threads):
    check_threads(threads)
    if threads > 128:
        raise ValueError("Thread count exceeds the supported GUI limit")
    if choices.get("acquisition_reviewed") is not True:
        raise ValueError("Review DWI orientation, acquisition settings and intended raw magnitude series first")
    artifacts = selected_artifacts(import_run, choices)
    dwi = artifacts["dwi"]
    summary = diffusion_summary(dwi)
    meta = read_json(dwi["json"])
    image = check_image_header(dwi["image"], (4,))
    for role in ("t1", "t2", "ct"):
        if role in artifacts:
            check_image_header(artifacts[role]["image"], (3,))
    pe = choices.get("phase_encoding")
    readout = choices.get("readout_seconds")
    if pe not in ("i", "i-", "j", "j-", "k", "k-"):
        raise ValueError("Phase encoding must be a reviewed NIfTI axis i/j/k with optional minus; AP is not assumed")
    if isinstance(readout, bool) or not isinstance(readout, (int, float)) or not np.isfinite(readout) or not 0 < readout < 2:
        raise ValueError("Enter the verified TotalReadoutTime in seconds (not echo spacing)")
    override = (pe != meta.get("PhaseEncodingDirection") or
                not isinstance(meta.get("TotalReadoutTime"), (int, float)) or
                not np.isclose(readout, meta.get("TotalReadoutTime", 0), atol=1e-7))
    if override and not str(choices.get("metadata_reason", "")).strip():
        raise ValueError("Missing/overridden acquisition metadata requires a documented source or reason")
    model = choices.get("fod_model")
    if model not in ("wm_csf", "wm_gm_csf"):
        raise ValueError("Explicitly select two-tissue WM+CSF or three-tissue WM+GM+CSF CSD")
    if model == "wm_gm_csf" and len(summary["shells"]) < 2:
        raise ValueError("Three-tissue MSMT-CSD requires at least two nonzero shells; single-shell is not three-tissue")
    if any(s["directions"] < 28 for s in summary["shells"]):
        raise ValueError("This initial CSD adapter requires at least 28 directions per shell; expert low-direction protocols are not automated")
    mode = choices.get("distortion")
    if mode not in ("none", "paired"):
        raise ValueError("Choose no reverse-PE correction or a reviewed matched reverse-b0 pair")
    if mode == "none" and choices.get("uncorrected_distortion_accepted") is not True:
        raise ValueError("Without reverse phase encoding, susceptibility distortion remains; explicit acknowledgement required")
    axes = choices.get("degibbs_axes")
    if axes not in ("skip", "0,1", "0,2", "1,2"):
        raise ValueError("Review the acquisition slice plane before enabling Gibbs correction")
    if axes != "skip":
        pf = meta.get("PartialFourier")
        if pf != 1 and choices.get("partial_fourier_reviewed") is not True:
            raise ValueError("Partial Fourier is present or unknown; review Gibbs correction applicability or select Skip")
        slice_axis = str(meta.get("SliceEncodingDirection", "")).rstrip("-")
        if slice_axis in ("i", "j", "k") and str("ijk".index(slice_axis)) in axes.split(","):
            raise ValueError("Gibbs correction axes include the slice axis; check the acquisition plane")
    out = Path(out).resolve()
    def p(name):
        return str(out / name)
    n = ["-nthreads", str(threads)]
    commands = []
    def add(tool, *args):
        commands.append([tool, *map(str, args), *n])
    add("mrconvert", dwi["image"], p("raw.mif"), "-fslgrad", dwi["bvec"], dwi["bval"],
        "-json_import", dwi["json"], "-datatype", "float32")
    add("dwidenoise", p("raw.mif"), p("denoised.mif"), "-noise", p("noise.mif"))
    add("mrcalc", p("raw.mif"), p("denoised.mif"), "-subtract", p("denoise_residual.mif"))
    if axes != "skip":
        add("mrdegibbs", p("denoised.mif"), p("degibbs.mif"), "-axes", axes)
        add("mrcalc", p("denoised.mif"), p("degibbs.mif"), "-subtract", p("degibbs_residual.mif"))
    source = p("denoised.mif" if axes == "skip" else "degibbs.mif")
    add("dwi2mask", source, p("eddy_mask.mif"))
    rpe = ["-rpe_none"]
    if mode == "paired":
        if "reverse_b0" not in artifacts or choices.get("reverse_contrast_reviewed") is not True:
            raise ValueError("Select reverse-b0 and verify matching SE-EPI contrast, TE/TR/flip angle and coverage")
        reverse = artifacts["reverse_b0"]
        rev = check_image_header(reverse["image"], (3, 4))
        rev_meta = read_json(reverse["json"]) if "json" in reverse else {}
        opposite = pe[:-1] if pe.endswith("-") else pe + "-"
        if rev_meta.get("PhaseEncodingDirection") != opposite or not np.isclose(rev_meta.get("TotalReadoutTime", -1), readout, atol=1e-7):
            raise ValueError("Reverse-b0 must have opposite PE and equal TotalReadoutTime; missing metadata cannot be guessed")
        if rev.shape[:3] != image.shape[:3] or not np.allclose(rev.affine, image.affine, atol=1e-3):
            raise ValueError("Reverse-b0 has a different acquisition grid; do not resample raw data automatically")
        if "bval" in reverse and np.any(np.loadtxt(reverse["bval"]) > 50):
            raise ValueError("Reverse-b0 selection contains diffusion-weighted volumes")
        if np.atleast_1d(np.loadtxt(dwi["bval"]))[0] > 50:
            raise ValueError("This initial paired adapter requires the first DWI volume to be b0 for -align_seepi")
        add("mrconvert", source, p("forward_b0.mif"), "-coord", "3", "0", "-axes", "0,1,2")
        if len(rev.shape) == 4:
            add("mrmath", reverse["image"], "mean", p("reverse_b0.mif"), "-axis", "3")
        else:
            add("mrconvert", reverse["image"], p("reverse_b0.mif"))
        add("mrcat", p("forward_b0.mif"), p("reverse_b0.mif"), p("b0_pair.mif"), "-axis", "3")
        rpe = ["-rpe_pair", "-se_epi", p("b0_pair.mif"), "-align_seepi"]
    # In paired mode, the first SE-EPI image is exactly the first (b0) DWI volume.
    add("dwifslpreproc", source, p("eddy.mif"), *rpe, "-pe_dir", pe, "-readout_time", readout,
        "-eddy_mask", p("eddy_mask.mif"), "-eddy_options", f" --repol --data_is_shelled --cnr_maps --residuals --nthr={threads}",
        "-eddyqc_all", p("eddy_qc"), "-export_grad_fsl", p("eddy.bvec"), p("eddy.bval"))
    add("dwi2mask", p("eddy.mif"), p("prebias_mask.mif"))
    add("dwibiascorrect", "ants", p("eddy.mif"), p("native_preprocessed.mif"), "-mask", p("prebias_mask.mif"), "-bias", p("bias.mif"))
    add("dwi2mask", p("native_preprocessed.mif"), p("native_mask.mif"))
    add("mrgrid", p("native_preprocessed.mif"), "regrid", p("dwi_2mm.mif"), "-voxel", "2", "-interp", "cubic")
    add("mrgrid", p("native_mask.mif"), "regrid", p("brain_mask_2mm.nii.gz"), "-template", p("dwi_2mm.mif"), "-interp", "nearest")
    add("dwiextract", p("dwi_2mm.mif"), p("bzeros_2mm.mif"), "-bzero")
    add("mrmath", p("bzeros_2mm.mif"), "mean", p("b0_2mm.nii.gz"), "-axis", "3")
    add("mrcalc", p("b0_2mm.nii.gz"), p("brain_mask_2mm.nii.gz"), "-mult", p("b0_brain_2mm.nii.gz"))
    add("dwi2response", "dhollander", p("dwi_2mm.mif"), p("wm.txt"), p("gm.txt"), p("csf.txt"), "-mask", p("brain_mask_2mm.nii.gz"))
    tissues = ["wm", "csf"] if model == "wm_csf" else ["wm", "gm", "csf"]
    fod_args, norm_args = [], []
    for tissue in tissues:
        fod_args += [p(tissue + ".txt"), p(tissue + "_fod.mif")]
        norm_args += [p(tissue + "_fod.mif"), p(tissue + "_norm.mif")]
    add("dwi2fod", "msmt_csd", p("dwi_2mm.mif"), *fod_args, "-mask", p("brain_mask_2mm.nii.gz"))
    add("mtnormalise", *norm_args, "-mask", p("brain_mask_2mm.nii.gz"))
    commands.append(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", p("b0_2mm.nii.gz"),
                     "-m", artifacts["t1"]["image"], "-o", p("t1_to_b0_"), "-t", "r", "-n", str(threads)])
    for role in ("ct", "t2"):
        if role in artifacts:
            commands.append(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", artifacts["t1"]["image"],
                "-m", artifacts[role]["image"], "-o", p(role + "_to_t1_"), "-t", "r", "-n", str(threads)])
            commands.append(["antsApplyTransforms", "-d", "3", "-i", artifacts[role]["image"], "-r", p("b0_2mm.nii.gz"),
                "-o", p(role + "_in_b0.nii.gz"), "-n", "Linear", "-t", p("t1_to_b0_0GenericAffine.mat"),
                "-t", p(role + "_to_t1_0GenericAffine.mat")])
    outputs = {"dwi": p("dwi_2mm.mif"), "fod": p("wm_norm.mif"), "reference": p("b0_brain_2mm.nii.gz"),
               "brain_mask": p("brain_mask_2mm.nii.gz"), "t1_in_b0": p("t1_to_b0_Warped.nii.gz"),
               "t1_transform": p("t1_to_b0_0GenericAffine.mat")}
    if "ct" in artifacts:
        outputs["ct_in_b0"] = p("ct_in_b0.nii.gz")
    return artifacts, summary, commands, outputs


def reconstruct(import_run, choices, out, threads, execute_tools=False):
    require_separate_output(out, import_run)
    original = read_json(Path(import_run) / "converted.json").get("source_root")
    if original:
        require_separate_output(out, original)
    artifacts, summary, commands, outputs = reconstruction_plan(import_run, choices, out, threads)
    out = Path(out)
    write_json(out / "plan.json", commands)
    write_json(out / "acquisition.json", {"choices": choices, "diffusion": summary, "threads": threads})
    write_json(out / "environment.json", discover())
    inputs = [Path(import_run) / "converted.json", *[p for a in artifacts.values() for p in a["hashes"]]]
    status = "planned_not_executed"
    if execute_tools:
        before = freeze_inputs(inputs)
        write_json(out / "input_hashes_before.json", before)
        if not any(tool_path(t) for t in ("eddy", "eddy_cpu", "eddy_openmp", "eddy_cuda")):
            raise ValueError("No FSL eddy executable found; configure the existing FSL installation")
        if choices["distortion"] == "paired" and not tool_path("topup"):
            raise ValueError("Reverse-PE correction requires FSL topup")
        execute(commands, out, threads)
        assert_inputs_unchanged(before)
        write_json(out / "source_integrity.json", {"source_bytes_unchanged": True, "input_count": len(before)})
        for path in outputs.values():
            if not Path(path).is_file():
                raise ValueError("External tools did not produce an expected output")
        # This brain-mask view is explicitly not an electrode-mask approval.
        browser_report(outputs["reference"], outputs["brain_mask"], out / "mask_review.html")
        html = (out / "mask_review.html").read_text()
        html = html.replace("Electrode mask", "DWI brain mask").replace("electrode mask", "brain mask (not electrode exclusion)")
        # Generated report transformation, not a source-file edit.
        (out / "mask_review.html").write_text(html)
        status = "completed"
        write_json(out / "reconstruction.json", {"status": "reconstruction_qc_required", "outputs": outputs,
            "import_run": str(Path(import_run).resolve()), "source_root": original, "artifacts": choices["artifacts"],
            "hashes": {p: digest(p) for p in outputs.values()},
            "next": ["Inspect raw/denoise/eddy residuals and rotated gradients", "Review T1/CT/DWI registration",
                     "Generate and review surface/NextBrain atlas and 5TT", "Localize leads and approve signal-void exclusion mask",
                     "Generate ACT tractogram, then use the audited connectome stage"],
            "approved_for_connectome": False, "clinical_validity": False})
    saved = [p for p in out.iterdir() if p.is_file() and p.suffix == ".json"]
    record(out, "reconstruct", inputs, {"threads": threads, "acquisition": choices,
        "manual_qc_required": True, "study_pipeline_equivalence": "not_validated"}, saved, status)
