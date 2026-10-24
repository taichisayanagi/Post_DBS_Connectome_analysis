"""Synthetic editor and optional real OSS-DBS checks, never patient data.

PYTHONPATH=src python examples/verify_gui_release.py --output-root /private/work
Add --simulate only with OSSDBS_PYTHON pointing to a separate solver environment.
The FEM check uses one thread and a small homogeneous phantom, not clinical data.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from dbs_connectome.electrodes import make_electrode
from dbs_connectome.provenance import digest, new_run, write_json
from dbs_connectome.stimulation import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--current", action="store_true", help="Current-controlled FEM smoke test (otherwise voltage)")
    args = parser.parse_args()
    root = new_run(args.output_root, "synthetic_gui_release")
    source = root / "synthetic_inputs"; source.mkdir()
    electrode = root / "electrode"; electrode.mkdir()
    affine = np.diag([2., 2., 2., 1.]); affine[:3, 3] = -48
    shape = (48, 48, 48)
    values = np.indices(shape)
    brain = ((values - 24)**2).sum(0) < 23**2
    mri = np.where(brain, 100 + values[0] + values[1] / 2, 0).astype(np.float32)
    ct = np.where(brain, 35, 0).astype(np.float32)
    for x in (18, 30):
        ct[x, 24, 12:38] = 3000
        mri[x-1:x+2, 23:26, 12:38] = 5
    def save(name, data):
        path = source / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(data, np.float32), affine), path)
        return str(path)
    reference = save("b0", mri)
    ct_path = save("ct", ct)
    brain_path = save("brain", brain)
    # Large homogeneous WM region for a numerical interoperability test.
    segmentation = save("tissues", np.full(shape, 2))
    spec = {"coordinate_system": "NIFTI_RAS_MM", "reference_sha256": digest(reference),
            "image_hashes": {"reference": digest(reference), "CT": digest(ct_path), "MRI": digest(reference), "Brain": digest(brain_path)},
            "mode": "ct", "expected_leads": 2, "registration_reviewed": True,
            "ct_hu_reviewed": True, "include_mri_void": True}
    result = make_electrode(reference, spec, electrode, ct_path, reference, brain_path)
    write_json(root / "editor_context.json", {"reference": reference, "CT": ct_path, "MRI": reference,
                                               "Brain": brain_path, "centerlines": result["outputs"]["centerlines"]})
    if args.simulate:
        out = root / "stimulation"; out.mkdir()
        settings = {"model": "Medtronic3389", "tip_ras_mm": [0, 0, -10], "shaft_ras_mm": [0, 0, 15],
                    "control": "current" if args.current else "voltage", "contact_amplitudes": [-1, 0, 0, 0],
                    "segmentation_reviewed": True, "physical_tip_reviewed": True,
                    "model_contact_numbering_reviewed": True, "approximation_accepted": True,
                    "domain_mm": 40, "lattice_extent_mm": 10, "lattice_spacing_mm": 1,
                    "mesh": "Coarse", "field_frequency_hz": 1000, "threshold_v_per_mm": .2}
        simulate(segmentation, settings, out, 1, True)
    print(root)


if __name__ == "__main__":
    main()
