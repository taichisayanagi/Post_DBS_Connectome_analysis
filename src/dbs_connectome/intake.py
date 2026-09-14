"""Private DICOM inventory, explicit series adjudication and traceable conversion.

Names are never used as filenames. This module is NOT a DICOM anonymizer.
Header-only inventory is followed by byte hashes when selected series are copied.
"""

import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import shutil

import nibabel as nib
import pydicom
from pydicom.errors import InvalidDicomError

from .external import execute
from .provenance import assert_inputs_unchanged, digest, read_json, record, require_separate_output, write_json


TAGS = ["SeriesInstanceUID", "StudyInstanceUID", "SOPInstanceUID", "PatientID", "IssuerOfPatientID",
        "PatientBirthDate", "Modality", "SeriesDescription", "ProtocolName", "SeriesNumber",
        "StudyDate", "ImageType", "Rows", "Columns", "NumberOfFrames", "DiffusionBValue"]
ROLES = ("t1", "dwi", "ct", "t2", "reverse_b0")


def role_hints(modality, description, image_type):
    value = (description + " " + image_type).lower()
    if modality == "CT":
        return ["ct"]
    if modality != "MR":
        return []
    if any(token in value for token in ("adc", "tracew", "fractional", "derived")):
        return ["derived_image_not_raw_dwi"]
    hints = []
    for role, terms in (("dwi", ("dwi", "diff", "dti")), ("t1", ("t1", "mprage", "spgr")),
                        ("t2", ("t2", "space", "cube")), ("reverse_b0", ("b0", "topup", "rev"))):
        if any(term in value for term in terms):
            hints.append(role)
    return hints


def inventory(dicom_dir, out):
    root, out = Path(dicom_dir).resolve(strict=True), Path(out)
    if not root.is_dir():
        raise ValueError("DICOM source must be a directory")
    require_separate_output(out, root)
    salt = secrets.token_bytes(32)
    def code(value):
        return hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()[:20]
    groups, rejected, seen_sops = {}, [], {}
    n_files = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink() and not d.startswith("."))
        for name in sorted(files):
            source = Path(directory) / name
            if source.is_symlink() or name.startswith("."):
                continue
            n_files += 1
            try:
                ds = pydicom.dcmread(source, stop_before_pixels=True, specific_tags=TAGS)
            except (InvalidDicomError, OSError, ValueError) as exc:
                rejected.append({"path": str(source), "reason": type(exc).__name__})
                continue
            uid, study, sop = (str(getattr(ds, key, "")) for key in
                               ("SeriesInstanceUID", "StudyInstanceUID", "SOPInstanceUID"))
            if not uid or not study or not sop or not getattr(ds, "Rows", 0) or not getattr(ds, "Columns", 0):
                rejected.append({"path": str(source), "reason": "Missing image/UID fields"})
                continue
            patient_id = str(getattr(ds, "PatientID", "")).strip()
            # Same IDs issued by different organizations are not silently treated as one patient.
            identity = "|".join(str(getattr(ds, key, "")) for key in ("IssuerOfPatientID", "PatientID", "PatientBirthDate"))
            patient = code(identity) if patient_id else None
            key = code(uid)
            desc = str(getattr(ds, "SeriesDescription", ""))
            typ = "\\".join(str(v) for v in getattr(ds, "ImageType", []))
            series = groups.setdefault(key, {"id": key, "patient_key": patient, "study_key": code(study),
                "series_uid": uid, "study_uid": study, "modality": str(getattr(ds, "Modality", "")),
                "description": desc, "series_number": str(getattr(ds, "SeriesNumber", "")),
                "study_date": str(getattr(ds, "StudyDate", "")), "image_type": typ,
                "hints": role_hints(str(getattr(ds, "Modality", "")), desc, typ),
                "files": [], "frames": 0, "errors": []})
            if series["patient_key"] != patient or series["study_uid"] != study:
                series["errors"].append("Series UID reused across patients/studies")
            if sop in seen_sops:
                series["errors"].append("Duplicate SOPInstanceUID; remove/resolve duplicate exports first")
                groups[seen_sops[sop]]["errors"].append("Duplicate SOPInstanceUID; remove/resolve duplicate exports first")
            seen_sops[sop] = key
            series["files"].append({"path": str(source), "sop_uid": sop,
                                    "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns})
            series["frames"] += int(getattr(ds, "NumberOfFrames", 1))
    result = {"schema_version": 1, "source_root": str(root), "files_scanned": n_files,
              "series": list(groups.values()), "rejected": rejected,
              "notice": "Private header inventory, NOT deidentified. Hints are not assignments. Enhanced/mosaic volumes are resolved by dcm2niix."}
    if not result["series"]:
        raise ValueError("No supported DICOM images found; verify that this is an uncompressed DICOM export")
    path = out / "inventory.json"
    write_json(path, result)
    record(out, "inventory", [], {"header_only": True, "source_root": str(root)}, [path])
    return result


def verified_output(run, name, stage):
    run = Path(run).resolve(strict=True)
    meta = read_json(run / "provenance.json")
    path = run / name
    if meta["stage"] != stage or meta["status"] != "completed" or meta["outputs"].get(str(path)) != digest(path):
        raise ValueError("Upstream run is incomplete or changed: " + stage)
    return path


def validate_selection(inventory_run, selection):
    report = read_json(verified_output(inventory_run, "inventory.json", "inventory"))
    for name in ("subject", "visit"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", str(selection.get(name, ""))):
            raise ValueError("Use a pseudonym, letters/numbers/hyphens only: " + name)
    if selection.get("identity_reviewed") is not True:
        raise ValueError("Confirm that selected series belong to this patient and intended visit")
    roles = selection.get("roles", {})
    if not isinstance(roles, dict) or not {"t1", "dwi"}.issubset(roles) or not set(roles).issubset(ROLES):
        raise ValueError("Choose T1 and DWI; optional roles are CT, T2 and reverse_b0")
    if len(set(roles.values())) != len(roles):
        raise ValueError("One DICOM series cannot occupy two roles")
    indexed = {s["id"]: s for s in report["series"]}
    if not set(roles.values()).issubset(indexed):
        raise ValueError("Selected series does not belong to this inventory")
    selected = {role: indexed[key] for role, key in roles.items()}
    if any(s["errors"] for s in selected.values()):
        raise ValueError("Selected series has unresolved inventory errors")
    identities = {s["patient_key"] for s in selected.values()}
    if None in identities or len(identities) != 1:
        raise ValueError("Patient identifiers are missing or do not match; do not merge these series")
    for role, series in selected.items():
        if series["modality"] != ("CT" if role == "ct" else "MR"):
            raise ValueError("DICOM modality is incompatible with selected role: " + role)
        if role == "dwi" and "DERIVED" in series["image_type"].upper():
            raise ValueError("Derived DWI/ADC/FA images are not raw diffusion input")
    if len({s["study_key"] for s in selected.values()}) > 1 and selection.get("cross_study_reviewed") is not True:
        raise ValueError("CT/MRI come from different studies; confirm the intended cross-study pairing")
    source_root = Path(report["source_root"]).resolve(strict=True)
    for series in selected.values():
        for item in series["files"]:
            path = Path(item["path"])
            if path.is_symlink() or source_root not in path.resolve(strict=True).parents:
                raise ValueError("DICOM file escaped its inventory root")
            stat = path.stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("DICOM changed since inventory; scan it again")
    return selected


def convert_selection(inventory_run, selection, out, execute_tools=False):
    out = Path(out).resolve()
    selected = validate_selection(inventory_run, selection)
    require_separate_output(out, read_json(Path(inventory_run) / "inventory.json")["source_root"])
    plan = []
    for role in selected:
        plan.append(["dcm2niix", "-b", "y", "-ba", "y", "-z", "y", "-f", role,
                     "-o", str(out / role), str(out / "dicom_staging" / role)])
    write_json(out / "selection.json", selection)
    write_json(out / "plan.json", plan)
    if not execute_tools:
        record(out, "import-session", [Path(inventory_run) / "inventory.json"],
               {"selection": selection, "deidentified": False}, [out / "plan.json", out / "selection.json"],
               "planned_not_executed")
        return None
    sources = []
    for role, series in selected.items():
        staging = out / "dicom_staging" / role
        staging.mkdir(parents=True, mode=0o700)
        (out / role).mkdir(mode=0o700)
        for i, item in enumerate(series["files"]):
            target = staging / f"{i:07d}.dcm"
            before = digest(item["path"])
            shutil.copyfile(item["path"], target)
            if before != digest(target) or before != digest(item["path"]):
                raise ValueError("DICOM changed while making a private conversion copy")
            sources.append({"role": role, "source": item["path"], "sha256": before})
    write_json(out / "dicom_source_hashes.json", sources)
    execute(plan, out)
    assert_inputs_unchanged({item["source"]: item["sha256"] for item in sources})
    write_json(out / "source_integrity.json", {"source_bytes_unchanged": True, "source_count": len(sources)})
    artifacts = []
    for role in selected:
        for path in sorted((out / role).glob("*.nii*")):
            image = nib.load(path)
            stem = str(path).removesuffix(".gz").removesuffix(".nii")
            related = {key: stem + ext for key, ext in (("json", ".json"), ("bval", ".bval"), ("bvec", ".bvec"))
                       if Path(stem + ext).is_file()}
            metadata = read_json(related["json"]) if "json" in related else {}
            artifact = {"id": f"{role}/{path.name}", "role": role, "image": str(path), **related,
                "shape": list(image.shape), "phase_encoding": metadata.get("PhaseEncodingDirection"),
                "readout_seconds": metadata.get("TotalReadoutTime"), "partial_fourier": metadata.get("PartialFourier"),
                "slice_encoding": metadata.get("SliceEncodingDirection"),
                "hashes": {str(p): digest(p) for p in [str(path), *related.values()]}}
            if role == "dwi":
                from .reconstruction import diffusion_summary
                try:
                    artifact["diffusion"] = diffusion_summary(artifact)
                except (ValueError, OSError) as exc:
                    artifact["input_error"] = str(exc)
            artifacts.append(artifact)
        if not any(a["role"] == role for a in artifacts):
            raise ValueError("dcm2niix produced no NIfTI for " + role)
    result = {"schema_version": 1, "source_root": read_json(Path(inventory_run) / "inventory.json")["source_root"], "subject": selection["subject"], "visit": selection["visit"],
              "artifacts": artifacts, "status": "input_review_required",
              "notice": "Select the intended magnitude DWI and anatomical output; multi-echo/derived outputs are never chosen silently."}
    write_json(out / "converted.json", result)
    outputs = [out / "converted.json", out / "dicom_source_hashes.json", out / "selection.json", out / "plan.json", out / "source_integrity.json"]
    record(out, "import-session", [Path(inventory_run) / "inventory.json"],
           {"deidentified": False, "patient_pairing": "identifier_check_plus_human_review"}, outputs)
    return result
