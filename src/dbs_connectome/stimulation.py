"""Optional OSS-DBS subprocess adapter, separate from connectome exclusion.

The first adapter supports one lead, a grounded outer boundary and a single-
frequency quasi-static field. Its thresholded field is a VTA-like approximation,
NOT an axon activation simulation, treatment recommendation or pulse-width model.
No OSS-DBS/NGSolve code or electrode geometry is bundled.
"""
import os
import base64
import json
from itertools import product
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np

from .electrodes import _number
from .external import check_threads, execute
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, image3d,
                         record, require_separate_output, write_json)


def material_from_five_tissue(reference, five_tissue, out):
    """Candidate isotropic tissue labels, not a conductivity/segmentation validation."""
    out = Path(out).resolve()
    for p in (reference, five_tissue):
        require_separate_output(out, Path(p).resolve().parent)
    before = freeze_inputs([reference, five_tissue])
    ref = image3d(reference)
    im = nib.load(five_tissue)
    if im.shape != ref.shape + (5,) or not np.allclose(im.affine, ref.affine, atol=1e-5):
        raise ValueError("5TT and reference grids differ")
    data = im.get_fdata()
    if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
        raise ValueError("Invalid tissue fractions")
    winner = data.argmax(-1)
    labels = np.asarray([3, 3, 2, 1, 0], np.uint8)[winner]
    labels[data.sum(-1) < .5] = 0
    path = out / "tissues_candidate.nii.gz"
    nib.save(nib.Nifti1Image(labels, ref.affine), path)
    result = {"segmentation": str(path), "reference": str(Path(reference).resolve()),
              "labels": {"0": "Unknown/outside/pathology", "1": "CSF", "2": "White matter", "3": "Gray matter"},
              "status": "needs_qc", "warning": "Argmax 5TT conversion; pathology maps to unknown, not blood. "
              "Inspect tissue boundaries near contacts. A coarse DWI grid may be unsuitable for field modeling.",
              "sha256": digest(path), "source_bytes_unchanged": True}
    write_json(out / "material_result.json", result)
    assert_inputs_unchanged(before)
    record(out, "material", [reference, five_tissue], {"clinical_validity": False}, [path, out / "material_result.json"])
    return result

# Public model identifiers, checked against the upstream electrode registry.
MODELS = {"Medtronic3387": 4, "Medtronic3389": 4, "Medtronic3391": 4,
          "MedtronicSenSightB33005": 8, "MedtronicSenSightB33015": 8,
          "BostonScientificVercise": 8, "BostonScientificVerciseDirected": 8,
          "AbbottStJudeDirected6172": 8, "AbbottStJudeDirected6173": 8}


def solver_python():
    """Operator-configured interpreter only; browser requests cannot select code."""
    value = os.environ.get("OSSDBS_PYTHON")
    if not value:
        raise ValueError("OSS-DBS is optional. Set OSSDBS_PYTHON to its separately installed environment's Python, then restart the GUI")
    # Do not resolve the executable symlink: that would bypass a venv's pyvenv.cfg.
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("OSSDBS_PYTHON is not executable")
    return str(path)


def _position(value, name):
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError(name + " requires three finite RAS-mm values")
    return p


def _xyz(p):
    return dict(zip(("x[mm]", "y[mm]", "z[mm]"), map(float, p)))


def configuration(segmentation, spec, out):
    image = image3d(segmentation)
    labels = np.unique(np.asanyarray(image.dataobj))
    if not np.isin(labels, (0, 1, 2, 3, 4)).all() or not any(v in labels for v in (2, 3)):
        raise ValueError("Reviewed segmentation must use 0 unknown, 1 CSF, 2 WM, 3 GM, 4 blood; it is not a raw MRI or an atlas")
    required = ("segmentation_reviewed", "physical_tip_reviewed", "model_contact_numbering_reviewed", "approximation_accepted")
    if not all(spec.get(k) is True for k in required):
        raise ValueError("Review tissue labels, physical tip, model/contact numbering and field/VTA approximation")
    model = spec.get("model")
    if model not in MODELS:
        raise ValueError("Select a supported explicit electrode model; IPG name does not identify the lead")
    if ("Directed" in model or "SenSight" in model) and spec.get("rotation_reviewed") is not True:
        raise ValueError("Directional leads require independently reviewed axial rotation; a shaft centerline cannot determine it")
    tip = _position(spec.get("tip_ras_mm"), "Physical tip")
    shaft = _position(spec.get("shaft_ras_mm"), "Shaft point")
    direction = shaft - tip
    length = np.linalg.norm(direction)
    if length < 5:
        raise ValueError("Shaft point must be at least 5 mm from the reviewed physical tip")
    direction /= length
    voxel_tip = nib.affines.apply_affine(np.linalg.inv(image.affine), tip)
    if np.any(voxel_tip < 0) or np.any(voxel_tip > np.asarray(image.shape)-1):
        raise ValueError("Physical tip is outside the segmentation field of view")
    rotation = _number(spec, "rotation_deg", 0, -360, 360)
    threshold = _number(spec, "threshold_v_per_mm", .2, .001, 10)
    frequency = _number(spec, "field_frequency_hz", 1000, 1, 100000)
    domain = _number(spec, "domain_mm", 60, 30, 150)
    spacing = _number(spec, "lattice_spacing_mm", .5, .1, 2)
    lattice = _number(spec, "lattice_extent_mm", 20, 5, 60)
    if lattice >= domain / np.sqrt(3):
        raise ValueError("Field lattice corners must lie inside the spherical model domain")
    size = int(np.ceil(lattice / spacing)) + 1
    if size ** 3 > 1_000_000:
        raise ValueError("Limit the field lattice to one million sample points")
    mesh = spec.get("mesh", "Moderate")
    if mesh not in ("Coarse", "Moderate", "Fine"):
        raise ValueError("Choose Coarse, Moderate or Fine mesh")
    control = spec.get("control", "voltage")
    if control not in ("voltage", "current"):
        raise ValueError("Choose voltage (V) or current (mA) control")
    amplitudes = spec.get("contact_amplitudes")
    if not isinstance(amplitudes, list) or len(amplitudes) != MODELS[model]:
        raise ValueError("Provide one amplitude per model contact, using OSS-DBS 1-based contact order")
    amps = np.asarray(amplitudes, dtype=float)
    if not np.isfinite(amps).all() or np.any(np.abs(amps) > 20) or not np.any(amps):
        raise ValueError("Amplitudes must be finite, within +/-20 V or mA, and not all zero (software bounds, not clinical recommendations)")
    contacts = []
    for i, amplitude in enumerate(amps):
        current = float(amplitude * .001) if control == "current" else 0.
        # The outer boundary is the sole grounded active surface in current mode.
        contacts.append({"Contact_ID": i+1, "Active": bool(amplitude) and control == "voltage",
                         "Floating": bool(control == "current" or amplitude == 0),
                         "Current[A]": current, "Voltage[V]": float(amplitude) if control == "voltage" else 0.,
                         "MaxMeshSizeEdge": .25})
    center = tip + direction * 6
    # Conservatively require the entire domain bounding box inside the tissue FOV.
    corners = np.asarray(list(product(*[(v-domain/2, v+domain/2) for v in center])))
    corners_vox = nib.affines.apply_affine(np.linalg.inv(image.affine), corners)
    if np.any(corners_vox < 0) or np.any(corners_vox > np.asarray(image.shape)-1):
        raise ValueError("Spherical model domain extends outside tissue image FOV; provide a sufficiently large registered tissue map")
    config = {
        "BrainRegion": {"Center": _xyz(center), "Dimension": _xyz([domain]*3), "Shape": "Sphere"},
        "Electrodes": [{"Name": model, "Rotation[Degrees]": rotation, "TipPosition": _xyz(tip),
                        "Direction": _xyz(direction), "Contacts": contacts,
                        "EncapsulationLayer": {"Thickness[mm]": 0.0}}],
        "Surfaces": [{"Name": "BrainSurface", "Active": True, "Floating": False, "Voltage[V]": 0.0,
                      "Current[A]": float(-amps.sum() * .001) if control == "current" else 0.0}],
        "MaterialDistribution": {"MRIPath": str(Path(out) / "inputs/tissues.nii.gz"),
            "MRIMapping": {"Unknown": 0, "CSF": 1, "White matter": 2, "Gray matter": 3, "Blood": 4},
            "DiffusionTensorActive": False, "DTIPath": ""},
        "DielectricModel": {"Type": "ColeCole4"}, "EQSMode": False, "FEMOrder": 2,
        "Mesh": {"LoadMesh": False, "SaveMesh": False, "MeshElementOrder": 2,
                 "MeshingHypothesis": {"Type": mesh}},
        "Solver": {"Type": "CG", "Preconditioner": "local", "MaximumSteps": 10000, "RelativeTolerance": 1e-8},
        "StimulationSignal": {"Type": "Multisine", "ListOfFrequencies": [frequency], "CurrentControlled": control == "current"},
        "PointModel": {"Lattice": {"Active": True, "Center": _xyz(center),
            "Shape": dict(zip(("x", "y", "z"), [size]*3)), "Direction": _xyz([0, 0, 1]),
            "PointDistance[mm]": spacing, "CollapseVTA": False, "ExportField": True},
            "Pathway": {"Active": False}},
        "OutputPath": str(Path(out) / "solver_results"), "StimulationFolder": str(out),
        "ExportVTK": True, "ExportElectrode": True,
        "ComputeImpedance": control == "voltage" and int(np.count_nonzero(amps)) == 1,
        "ExportFrequency": frequency, "ActivationThresholdVTA[V-per-m]": threshold * 1000,
        "CalcAxonActivation": False,
    }
    return config


def simulate(segmentation, spec, out, threads, execute_tools=False):
    check_threads(threads)
    out = Path(out).resolve()
    require_separate_output(out, Path(segmentation).resolve().parent)
    before = freeze_inputs([segmentation])
    config = configuration(segmentation, spec, out)
    write_json(out / "settings.json", spec)
    write_json(out / "ossdbs_config.json", config)
    warning = ("Single-frequency isotropic quasi-static field; thresholded VTA-like volume, not axon activation. "
               "No pulse-width dependence, encapsulation or treatment optimization. Verify mesh/domain convergence. "
               "Never use the artifact-exclusion mask as VTA or tissue segmentation.")
    write_json(out / "plan.json", {"engine": "external OSS-DBSv2", "warning": warning,
               "threads": threads, "engine_installation": "separate, not bundled", "execute_requested": execute_tools})
    inputs = [segmentation]
    outputs = [out / "settings.json", out / "ossdbs_config.json", out / "plan.json"]
    if not execute_tools:
        record(out, "simulate", inputs, {"clinical_validity": False}, outputs, "planned_not_executed")
        return
    python = solver_python()
    probe = subprocess.run([python, "-c", "import importlib.metadata; print(importlib.metadata.version('ossdbs'))"],
                           check=True, capture_output=True, text=True, timeout=30)
    version = probe.stdout.strip()
    (out / "inputs").mkdir(mode=0o700)
    # Re-encode only the private copy; no converter/solver sees original filenames.
    image = image3d(segmentation)
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj).astype(np.uint8), image.affine), out / "inputs/tissues.nii.gz")
    logs = out / "commands"
    logs.mkdir()
    runner = Path(__file__).with_name("ossdbs_runner.py")
    execute([[python, str(runner), str(out / "ossdbs_config.json"), str(threads)]], logs, threads)
    products = list((out / "solver_results").rglob("*.nii")) + list((out / "solver_results").rglob("*.nii.gz"))
    vtas = [p for p in products if "vta" in p.name.lower()]
    if not vtas:
        raise ValueError("Solver exited without a VTA NIfTI; inspect its logs/version. Simulation is NOT accepted as complete")
    descriptions = []
    for path in vtas:
        im = image3d(path)
        values = np.asanyarray(im.dataobj)
        if not np.isin(values, (0, 1)).all():
            raise ValueError("Unexpected nonbinary VTA output")
        descriptions.append({"path": str(path), "volume_mm3": float(np.count_nonzero(values) * abs(np.linalg.det(im.affine[:3, :3]))),
                             "voxel_count": int(np.count_nonzero(values)), "sha256": digest(path)})
    fields = [p for p in products if "e_field_solution" in p.name.lower()]
    if len(fields) != 1 or len(vtas) != 1:
        raise ValueError("Expected exactly one field and one VTA-like NIfTI for this single-lattice adapter")
    report = field_report(fields[0], vtas[0], segmentation, out, spec.get("threshold_v_per_mm", .2))
    assert_inputs_unchanged(before)
    write_json(out / "simulation_result.json", {"status": "completed", "engine": "OSS-DBS", "version": version,
        "source_bytes_unchanged": True, "clinical_validity": False, "warning": warning,
        "threshold_v_per_mm": spec.get("threshold_v_per_mm", .2), "vta_like": descriptions,
        "nifti_outputs": [str(p) for p in products], "mesh_convergence_validated": False, "report": str(report)})
    outputs += [out / "simulation_result.json", *products, out / "inputs/tissues.nii.gz", report]
    record(out, "simulate", inputs, {"engine_version": version, "threads": threads, "clinical_validity": False}, outputs)


def field_report(field_path, vta_path, segmentation, out, threshold):
    from nibabel.processing import resample_from_to
    from .provenance import same_grid
    field, vta = image3d(field_path), image3d(vta_path)
    same_grid(field, vta)
    if np.prod(field.shape) > 1_000_000:
        raise ValueError("Field viewer sample limit exceeded")
    field, vta = nib.as_closest_canonical(field), nib.as_closest_canonical(vta)
    # Display-only resampling of the tissue map to the solver output lattice.
    tissues = resample_from_to(image3d(segmentation), field, order=0)
    values = field.get_fdata() / 1000.  # OSS-DBS NIfTI magnitude is V/m, viewer uses V/mm.
    if np.any(values < 0):
        raise ValueError("Unexpected negative field magnitude")
    payload = {"shape": list(field.shape), "affine": field.affine.tolist(), "threshold": threshold,
        "field": base64.b64encode(values.astype('<f4').tobytes(order='F')).decode(),
        "vta": base64.b64encode(np.asarray(vta.dataobj, np.uint8).tobytes(order='F')).decode(),
        "tissues": base64.b64encode(tissues.get_fdata().astype(np.uint8).tobytes(order='F')).decode(),
        "max_v_per_mm": float(values.max()), "units": "V/mm", "clinical_validity": False}
    path = Path(out) / "simulation_review.html"
    html = (Path(__file__).parent / "static/field.html").read_text().replace("__DATA__", json.dumps(payload, allow_nan=False))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(html)
    return path
