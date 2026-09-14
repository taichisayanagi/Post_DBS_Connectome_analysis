"""Local-only run records; no telemetry and no upload functionality."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # Exclusive creation prevents accidental reuse/overwrite of old results.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def new_run(root, stage):
    root = Path(root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    # Source checkout must never accumulate MRI products, even when ignored.
    if (repo / "pyproject.toml").exists() and (root == repo or repo in root.parents):
        raise ValueError("Choose an output directory outside the source repository")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M")
    out = root / f"{stage}_{stamp}"
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    return out


def require_separate_output(output, input_directory):
    """Never create run products inside a selected patient/source directory."""
    output = Path(output).expanduser().resolve()
    source = Path(input_directory).expanduser().resolve(strict=True)
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the input/patient directory; original data are read-only")


def freeze_inputs(paths):
    """Record hashes before execution, without changing files, permissions or timestamps."""
    return {str(Path(p).resolve(strict=True)): digest(p) for p in paths}


def assert_inputs_unchanged(before):
    changed = [p for p, expected in before.items() if not Path(p).is_file() or digest(p) != expected]
    if changed:
        raise ValueError("An input changed during processing; results cannot be accepted. Check the private input hash manifest.")


def record(out, stage, inputs, parameters, outputs, status="completed"):
    versions = {}
    for name in ("numpy", "scipy", "nibabel", "brainspace"):
        versions[name] = importlib.metadata.version(name)
    item = {
        "schema_version": 1,
        "software_version": "0.1.0.dev0",
        "utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "parameters": parameters,
        "versions": versions,
        "inputs": {str(Path(p).resolve()): digest(p) for p in inputs},
        "outputs": {str(Path(p).resolve()): digest(p) for p in outputs},
        "privacy": "LOCAL ONLY: paths, image hashes and products may be identifying; do not publish",
    }
    write_json(Path(out) / "provenance.json", item)
    return item


def image3d(path):
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError("Expected a 3D NIfTI image")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid NIfTI affine")
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-3):
        raise ValueError("Conflicting qform/sform: resolve orientation before processing")
    if not qc and not sc:
        raise ValueError("Missing coded spatial transform")
    if not np.isfinite(np.asanyarray(image.dataobj)).all():
        raise ValueError("Non-finite image voxels")
    return image


def same_grid(a, b):
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-5):
        raise ValueError("Images are not on the same grid; explicit registration/resampling is required")
    # This checks grids only. It never establishes anatomical registration.


def voxel_fingerprint(path):
    """Detect decoded voxel duplicates despite different gzip/NIfTI headers.

    Geometry is deliberately recorded separately from intensity identity.
    This is not a de-identification operation.
    """
    image = nib.load(path)
    array = np.asarray(image.dataobj, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite image voxels")
    h = hashlib.sha256(str(array.shape).encode())
    h.update(array.tobytes(order="C"))
    return {"voxel_sha256": h.hexdigest(), "shape": list(array.shape),
            "affine": image.affine.tolist(), "file_sha256": digest(path)}
