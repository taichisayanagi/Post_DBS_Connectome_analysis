"""Non-mutating installation discovery. Detection is not validation or a license check."""

import importlib.metadata
import os
from pathlib import Path
import shutil
import sys


TOOLS = {
    "DICOM import": ["dcm2niix"],
    "DWI reconstruction": ["mrconvert", "dwidenoise", "mrdegibbs", "dwi2mask", "dwifslpreproc",
                           "dwibiascorrect", "dwiextract", "mrmath", "mrgrid", "mrcalc",
                           "dwi2response", "dwi2fod", "mtnormalise", "mrcat", "N4BiasFieldCorrection"],
    "Registration": ["antsRegistrationSyNQuick.sh", "antsApplyTransforms"],
    "FreeSurfer / surface atlas": ["recon-all", "mri_surf2surf", "mri_aparc2aseg", "mri_convert", "5ttgen"],
    "Tractography / connectome": ["tckgen", "tckedit", "tcksift2", "tck2connectome", "mrinfo"],
    "Optional MATLAB bridge": ["matlab"],
}


def tool_path(name):
    """Respect PATH first; use only explicitly configured suite roots as fallbacks."""
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    candidates = []
    for key in ("FSLDIR", "FREESURFER_HOME", "MRTRIX_HOME", "ANTSPATH"):
        if os.environ.get(key):
            root = Path(os.environ[key]).expanduser()
            candidates.extend([root / "bin" / name, root / name])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def discover():
    suites = []
    for stage, names in TOOLS.items():
        entries = [{"tool": name, "path": tool_path(name)} for name in names]
        for entry in entries:
            entry["status"] = "detected_not_tested" if entry["path"] else "not_found"
        suites.append({"stage": stage, "tools": entries})
    eddy = [name for name in ("eddy", "eddy_cpu", "eddy_openmp", "eddy_cuda") if tool_path(name)]
    packages = {}
    for name in ("numpy", "scipy", "nibabel", "brainspace", "pydicom"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_found"
    return {"python": sys.executable, "packages": packages, "stages": suites,
            "eddy_candidates": eddy, "topup": tool_path("topup"),
            "resources_not_validated": ["NextBrain model/atlas", "Schaefer surface annotations",
                                        "FreeSurfer license", "SPM12"],
            "notice": "Discovery only. No programs launched, no downloads or installation changes; licenses not read."}


def command_environment(threads):
    env = dict(os.environ)
    directories = []
    for names in TOOLS.values():
        for name in names:
            path = tool_path(name)
            if path and str(Path(path).parent) not in directories:
                directories.append(str(Path(path).parent))
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *directories])
    for name in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "MRTRIX_NTHREADS"):
        env[name] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    return env
