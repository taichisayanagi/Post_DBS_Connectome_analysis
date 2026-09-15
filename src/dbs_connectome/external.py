"""Explicit external-tool plans and a fail-fast, non-shell executor."""

from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import numpy as np

from .gradients import load_nodes
from .environment import command_environment
from .masks import require_approval
from .provenance import digest, image3d, read_json, same_grid, write_json


def check_threads(threads):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("Explicit positive thread count required before running MRI tools")


def conversion_plan(dicom_dir, out):
    if not Path(dicom_dir).is_dir():
        raise ValueError("Supply a locally selected DICOM series directory")
    # Numeric series names avoid copying patient/protocol names into filenames.
    # -ba y only anonymizes selected JSON fields, NOT the entire dataset.
    return [["dcm2niix", "-b", "y", "-ba", "y", "-z", "y", "-f", "series_%s",
             "-o", str(Path(out).resolve()), str(Path(dicom_dir).resolve())]]


def connectome_plan(config_path, out, threads):
    check_threads(threads)
    spec = read_json(config_path)
    keys = ("tractogram", "fod", "five_tissue", "mask", "reference", "atlas", "nodes", "approval")
    if set(spec) != set(keys):
        raise ValueError("Prepared config must contain exactly: " + ", ".join(keys))
    base = Path(config_path).resolve().parent
    paths = {key: str((base / Path(spec[key]).expanduser()).resolve()) for key in keys}
    if any(not Path(p).is_file() for p in paths.values()):
        raise ValueError("One or more prepared input files are missing")
    require_approval(paths["approval"], paths["mask"], paths["reference"], paths["atlas"])
    ref, atlas = image3d(paths["reference"]), image3d(paths["atlas"])
    same_grid(ref, atlas)
    labels = np.asanyarray(atlas.dataobj)
    if np.any(labels < 0) or not np.all(labels == np.floor(labels)):
        raise ValueError("Atlas must contain nonnegative integer labels")
    node_labels = {int(row["label"]) for row in load_nodes(paths["nodes"])}
    if not node_labels.issubset(set(np.unique(labels).astype(int))):
        raise ValueError("Requested node labels are absent from the native atlas")
    # MIF image geometry is checked by mrinfo in the executor, before tckedit.
    out = Path(out).resolve()
    common = ["-nthreads", str(threads)]
    tracks = str(out / "tracks_excluded.tck")
    weights = str(out / "sift2_weights.txt")
    commands = [
        ["tckedit", paths["tractogram"], tracks, "-exclude", paths["mask"], *common],
        ["tcksift2", tracks, paths["fod"], weights, "-act", paths["five_tissue"],
         "-out_mu", str(out / "sift2_mu.txt"), *common],
        ["tck2connectome", tracks, paths["atlas"], str(out / "connectome_raw.csv"),
         "-tck_weights_in", weights, "-symmetric", "-zero_diagonal",
         "-assignment_radial_search", "4", "-out_assignments", str(out / "assignments.txt"), *common],
    ]
    return paths, commands


def validate_mif_grid(path, reference, out):
    """Ask MRtrix for header geometry; refusal is safer than assuming co-registration."""
    report = Path(out) / (Path(path).name + ".header.json")
    subprocess.run(["mrinfo", str(path), "-json_all", str(report)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    meta = read_json(report)
    if isinstance(meta, list):
        if len(meta) != 1:
            raise ValueError("Unexpected MRtrix header result")
        meta = meta[0]
    size = tuple(meta["size"][:3])
    spacing = np.asarray(meta["spacing"][:3], dtype=float)
    transform = np.asarray(meta["transform"], dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = transform[:3, :3] * spacing[np.newaxis, :]
    affine[:3, 3] = transform[:3, 3]
    ref = image3d(reference)
    if size != ref.shape or not np.allclose(affine, ref.affine, atol=1e-3):
        raise ValueError("FOD/5TT grid mismatch; review registration, do not automatically resample")


def validate_tractogram_sampling(path, reference):
    """Reject sparse imported tracks that can jump over a mask between vertices.

    This is a sampling/FOV gate, not proof of anatomical registration or a continuous
    segment/voxel intersection algorithm. No automatic resampling changes the study input.
    """
    ref = image3d(reference)
    inv = np.linalg.inv(ref.affine)
    limit_mm = float(min(nib.affines.voxel_sizes(ref.affine)) / 2)
    loaded = nib.streamlines.load(str(path), lazy_load=True)
    count, max_segment = 0, 0.0
    for points in loaded.streamlines:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Invalid streamline geometry")
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())
        if length > limit_mm + 1e-4:
            raise ValueError("Sparse tractogram can jump over the mask; explicitly resample and revalidate upstream")
        vox = nib.affines.apply_affine(inv, points)
        if np.any(vox < -.5) or np.any(vox > np.asarray(ref.shape) - .5):
            raise ValueError("Streamline leaves the reference FOV; verify coordinate space")
        max_segment = max(max_segment, length)
        count += 1
    if count == 0:
        raise ValueError("Empty tractogram")
    return {"count": count, "max_segment_mm": max_segment, "allowed_max_segment_mm": limit_mm}


def audit_mask_exclusion(path, mask):
    """Independent closed segment/voxel-box intersection, including between vertices.

    A retained crossing fails the run before SIFT2/connectome construction.
    This validates exclusion geometry, not the adequacy of the clinical mask.
    """
    from .masks import validate_binary
    image = image3d(mask)
    binary = validate_binary(image)
    inverse = np.linalg.inv(image.affine)
    count = 0
    for track in nib.streamlines.load(str(path), lazy_load=True).streamlines:
        vox = nib.affines.apply_affine(inverse, track)
        if not np.isfinite(vox).all() or len(vox) < 2:
            raise ValueError("Invalid retained streamline")
        a, b = vox[:-1], vox[1:]
        lower = np.floor(np.minimum(a, b) + .5 - 1e-8).astype(int)
        upper = np.floor(np.maximum(a, b) + .5).astype(int)
        if np.any(upper - lower > 1):
            raise ValueError("Retained streamline is too sparse for the segment exclusion audit")
        direction = b - a
        for bits in np.ndindex(2, 2, 2):
            cells = np.where(np.asarray(bits), upper, lower)
            valid = np.all((cells >= 0) & (cells < np.asarray(binary.shape)), axis=1)
            candidates = np.flatnonzero(valid)
            candidates = candidates[binary[tuple(cells[candidates].T)]]
            if not len(candidates):
                continue
            start, delta, cell = a[candidates], direction[candidates], cells[candidates]
            zero = np.abs(delta) < 1e-12
            outside = zero & ((start < cell - .5) | (start > cell + .5))
            safe = np.where(zero, 1, delta)
            t1, t2 = (cell - .5 - start) / safe, (cell + .5 - start) / safe
            near = np.where(zero, -np.inf, np.minimum(t1, t2)).max(1)
            far = np.where(zero, np.inf, np.maximum(t1, t2)).min(1)
            hit = (~outside.any(1)) & (np.maximum(near, 0) <= np.minimum(far, 1))
            if hit.any():
                raise ValueError("Electrode exclusion audit failed: a retained streamline segment intersects the mask; no connectome accepted")
        count += 1
    if not count:
        raise ValueError("No streamlines remain after electrode exclusion")
    return {"retained_streamlines": count, "mask_intersections": 0,
            "method": "continuous_segment_closed_voxel_box", "mask_sha256": digest(mask)}


def execute(commands, out, threads=None, environment=None):
    out = Path(out).resolve(strict=True)
    if threads is not None:
        check_threads(threads)
    env = command_environment(threads or 1)
    if environment:
        if not set(environment).issubset({"SUBJECTS_DIR", "FREESURFER_HOME"}):
            raise ValueError("Unsupported environment override")
        env.update({k: str(v) for k, v in environment.items()})
        if environment.get("FREESURFER_HOME"):
            env["PATH"] = str(Path(environment["FREESURFER_HOME"]) / "bin") + ":" + env["PATH"]
    missing = sorted({cmd[0] for cmd in commands if shutil.which(cmd[0], path=env["PATH"]) is None})
    if missing:
        raise ValueError("Missing external tools: " + ", ".join(missing))
    versions = {}
    for name in sorted({cmd[0] for cmd in commands}):
        executable = shutil.which(name, path=env["PATH"])
        if Path(name).name.startswith("ants") or Path(name).name in ("fspython", "mri_surf2surf", "mri_aparc2aseg", "mri_vol2vol"):
            versions[name] = {"path": executable, "sha256": digest(executable), "version": "not_queried"}
            continue
        flag = "--version" if name == "dcm2niix" else "-version"
        version = subprocess.run([executable, flag], capture_output=True, text=True, timeout=30, env=env, cwd=out)
        versions[name] = {"path": executable, "sha256": digest(executable), "returncode": version.returncode,
                          "text": (version.stdout + version.stderr).strip()}
    write_json(Path(out) / "external_versions.json", versions)
    for i, command in enumerate(commands):
        print(f"[step {i+1}/{len(commands)}] Running {command[0]}", flush=True)
        # stdout/stderr can contain PHI. They stay in the private run directory.
        actual = [versions[command[0]]["path"], *command[1:]]
        with (Path(out) / f"command_{i:02d}.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(actual, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", env=env, cwd=out) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                process.stdout.close()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, actual)
        print(f"[step {i+1}/{len(commands)}] Completed {command[0]}", flush=True)
