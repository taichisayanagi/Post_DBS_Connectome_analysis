"""Private-copy NIfTI intake. Original files and their permissions are never changed."""

from pathlib import Path
import re
import shutil

import nibabel as nib

from .intake import ROLES
from .provenance import assert_inputs_unchanged, digest, freeze_inputs, read_json, record, require_separate_output, write_json


def validate_selection(selection, out):
    root = Path(selection["source_root"]).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("NIfTI source root must be a directory")
    require_separate_output(out, root)
    for key in ("subject", "visit"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", str(selection.get(key, ""))):
            raise ValueError("Use a pseudonym for " + key)
    if selection.get("identity_reviewed") is not True or selection.get("raw_dwi_reviewed") is not True:
        raise ValueError("Confirm patient/visit identity and that DWI is raw magnitude data with matching gradients")
    roles = selection.get("roles", {})
    if not {"t1", "dwi"}.issubset(roles) or not set(roles).issubset(ROLES):
        raise ValueError("NIfTI intake requires T1 and raw DWI")
    checked = {}
    for role, files in roles.items():
        if not isinstance(files, dict) or "image" not in files or not set(files).issubset({"image", "json", "bval", "bvec"}):
            raise ValueError("Each role needs an image and optional JSON/bval/bvec")
        checked[role] = {}
        for key, value in files.items():
            path = (root / Path(value).expanduser()).resolve(strict=True)
            if not path.is_file() or root not in path.parents:
                raise ValueError("Selected NIfTI/sidecar escaped the source root")
            checked[role][key] = str(path)
        if not str(files["image"]).endswith((".nii", ".nii.gz")):
            raise ValueError("Input images must be NIfTI .nii or .nii.gz")
    if not {"bval", "bvec"}.issubset(checked["dwi"]):
        raise ValueError("DWI requires its matching bval and bvec files")
    if len({files["image"] for files in checked.values()}) != len(checked):
        raise ValueError("One image cannot be assigned to multiple roles")
    return checked


def import_nifti(selection, out):
    from .reconstruction import check_image_header, diffusion_summary
    out = Path(out)
    roles = validate_selection(selection, out)
    before = freeze_inputs([p for files in roles.values() for p in files.values()])
    write_json(out / "source_hashes_before.json", before)
    artifacts = []
    for role, files in roles.items():
        directory = out / role
        directory.mkdir(mode=0o700)
        artifact = {"id": role, "role": role}
        for key, source in files.items():
            suffix = (".nii.gz" if source.endswith(".gz") else ".nii") if key == "image" else "." + key
            target = directory / (role + suffix)
            shutil.copyfile(source, target)
            if digest(target) != before[source]:
                raise ValueError("Source changed while copying NIfTI input")
            artifact[key] = str(target)
        if "json" not in artifact:
            target = directory / (role + ".json")
            write_json(target, {})
            artifact["json"] = str(target)
            artifact["metadata_source"] = "missing_not_inferred"
        image = check_image_header(artifact["image"], (4,) if role == "dwi" else ((3, 4) if role == "reverse_b0" else (3,)))
        metadata = read_json(artifact["json"])
        artifact.update(shape=list(image.shape), phase_encoding=metadata.get("PhaseEncodingDirection"),
                        readout_seconds=metadata.get("TotalReadoutTime"), partial_fourier=metadata.get("PartialFourier"),
                        slice_encoding=metadata.get("SliceEncodingDirection"))
        if role == "dwi":
            artifact["diffusion"] = diffusion_summary(artifact)
        artifact["hashes"] = {artifact[k]: digest(artifact[k]) for k in ("image", "json", "bval", "bvec") if k in artifact}
        artifacts.append(artifact)
    assert_inputs_unchanged(before)
    result = {"schema_version": 1, "source_format": "NIfTI", "source_root": str(Path(selection["source_root"]).resolve()), "subject": selection["subject"],
              "visit": selection["visit"], "artifacts": artifacts, "status": "input_review_required",
              "notice": "NIfTI cannot establish patient identity or DICOM provenance. Human identity/visit and raw-DWI attestations are recorded. Missing acquisition metadata is not inferred."}
    write_json(out / "converted.json", result)
    write_json(out / "selection.json", selection)
    write_json(out / "source_integrity.json", {"source_bytes_unchanged": True, "input_count": len(before)})
    record(out, "import-nifti", list(before), {"deidentified": False, "identity": "human_attestation_only"},
           [out / p for p in ("converted.json", "selection.json", "source_hashes_before.json", "source_integrity.json")])
    return result
