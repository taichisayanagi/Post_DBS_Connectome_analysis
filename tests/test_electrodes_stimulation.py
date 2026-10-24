"""Synthetic-only safety, geometry, immutable handoff and stimulation contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from dbs_connectome.electrodes import make_electrode, viewer_payload, validate_paths, verified_electrode
from dbs_connectome.masks import approval_record, require_approval, REVIEW_ITEMS
from dbs_connectome.provenance import digest, read_json, record, write_json
from dbs_connectome.stimulation import configuration, material_from_five_tissue, simulate, solver_python
from dbs_connectome.workflow import adopt_electrode, prepared_paths


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.out = self.root / "result"
        self.out.mkdir()
        self.affine = np.diag([2., 2., 2., 1.])
        self.ref = self.save("ref", np.full((24, 24, 24), 100.))
        self.brain = self.save("brain", np.ones((24, 24, 24)))
        ct = np.zeros((24, 24, 24)); ct[7, 10, 3:20] = ct[17, 10, 3:20] = 3000
        self.ct = self.save("ct", ct)
        mri = np.full((24, 24, 24), 100.)
        mri[6:9, 9:12, 3:20] = mri[16:19, 9:12, 3:20] = 5
        mri[0:2, 0:2, :] = 0  # unrelated darkness must never enter the void mask
        self.mri = self.save("mri", mri)
        self.paths = [[[14, 20, 6], [14, 20, 38]], [[34, 20, 6], [34, 20, 38]]]
        self.spec = {"coordinate_system": "NIFTI_RAS_MM", "reference_sha256": digest(self.ref),
                     "image_hashes": {"reference": digest(self.ref), "CT": digest(self.ct), "MRI": digest(self.mri), "Brain": digest(self.brain)},
                     "paths": self.paths, "registration_reviewed": True, "ct_hu_reviewed": True,
                     "include_mri_void": True, "mode": "manual"}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, data):
        path = self.source / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(data, np.float32), self.affine), path)
        return str(path)

    def generate(self, spec=None):
        return make_electrode(self.ref, spec or self.spec, self.out, self.ct, self.mri, self.brain)

    def test_ct_mri_fusion_preserves_original_bytes_mtime_and_modes(self):
        before = {p: (digest(p), p.stat().st_mtime_ns, p.stat().st_mode) for p in self.source.iterdir()}
        spec = {**self.spec, "mode": "ct"}
        result = self.generate(spec)
        self.assertEqual(len(result["paths"]), 2)
        self.assertFalse(result["automatic_qc_approval"])
        metal, void, mask = [nib.load(result["outputs"][k]).get_fdata().astype(bool) for k in ("metal", "void", "mask")]
        self.assertTrue(metal.any() and void.any())
        self.assertTrue(np.all(mask[metal | void]))
        self.assertFalse(void[:2, :2].any())
        self.assertEqual(before, {p: (digest(p), p.stat().st_mtime_ns, p.stat().st_mode) for p in self.source.iterdir()})

    def test_missing_registration_and_stale_image_refused(self):
        for change, message in (({"registration_reviewed": False}, "registration"), ({"reference_sha256": "bad"}, "binding"),
                                ({"image_hashes": {}}, "auxiliary")):
            with self.assertRaisesRegex(ValueError, message):
                self.generate({**self.spec, **change})

    def test_mri_only_manual_seed_route(self):
        spec = {**self.spec, "image_hashes": {k: v for k, v in self.spec["image_hashes"].items() if k != "CT"}}
        result = make_electrode(self.ref, spec, self.out, mri=self.mri, brain_mask=self.brain)
        self.assertTrue(nib.load(result["outputs"]["void"]).get_fdata().any())
        self.assertFalse(nib.load(result["outputs"]["metal"]).get_fdata().any())

    def test_virtual_does_not_infer_real_void(self):
        with self.assertRaisesRegex(ValueError, "Virtual"):
            self.generate({**self.spec, "mode": "virtual"})

    def test_out_of_bounds_and_duplicate_points_refused(self):
        for paths in ([[[-100, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]):
            with self.assertRaises(ValueError):
                validate_paths(paths, nib.load(self.ref))

    def test_anisotropic_or_mismatched_inputs_refused(self):
        shifted = nib.Nifti1Image(np.ones((24, 24, 24)), np.diag([1., 1., 1., 1.]))
        nib.save(shifted, self.mri)
        with self.assertRaisesRegex(ValueError, "same grid"):
            self.generate()

    def test_manual_erasure_has_audit_reason(self):
        spec = {**self.spec, "edits": [{"action": "erase", "ras_mm": [14, 20, 20], "radius_mm": 2}]}
        with self.assertRaisesRegex(ValueError, "reason"):
            self.generate(spec)
        spec["edits"][0]["reason"] = "Synthetic correction test"
        result = self.generate(spec)
        self.assertEqual(nib.load(result["outputs"]["mask"]).get_fdata()[7, 10, 10], 0)
        self.assertIn("Manual erasure", " ".join(result["warnings"]))

    def test_output_inside_source_refused_before_writing(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            make_electrode(self.ref, self.spec, self.source / "bad", self.ct, self.mri, self.brain)
        self.assertFalse((self.source / "bad").exists())

    def test_viewer_canonical_coordinates_roundtrip_oblique(self):
        affine = np.array([[-2, 0, 0, 44], [0, 0, -2, 44], [0, 2, 0, 0], [0, 0, 0, 1.]])
        nib.save(nib.Nifti1Image(np.ones((24, 24, 24)), affine), self.ref)
        payload = viewer_payload(self.ref)
        point = np.array([3, 11, 17, 1.])
        np.testing.assert_allclose(np.array(payload["inverse_affine"]) @ (np.array(payload["affine"]) @ point), point)

    def test_candidate_modified_after_creation_refused(self):
        result = self.generate()
        with open(result["outputs"]["centerlines"], "a") as f:
            f.write(" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            verified_electrode(self.out, self.ref)

    def test_immutable_mask_adoption_invalidates_old_qc(self):
        candidate = self.generate()
        prepared = self.root / "prepared"; prepared.mkdir()
        atlas = self.save("atlas", np.ones((24, 24, 24)))
        old_mask = self.save("old_mask", np.ones((24, 24, 24)))
        five = np.zeros((24, 24, 24, 5)); five[..., 2] = 1
        five = self.save("five", five)
        nodes = self.source / "nodes.tsv"; nodes.write_text("synthetic")
        paths = {"reference": self.ref, "mask": old_mask, "atlas": atlas, "five_tissue": five, "fod": self.ref, "nodes": str(nodes)}
        write_json(prepared / "prepared.json", {"outputs": paths, "hashes": {p: digest(p) for p in paths.values()}, "missing_dwi_nodes": [], "source_root": str(self.source)})
        record(prepared, "prepare-session", [], {}, [prepared / "prepared.json"])
        approval = self.root / "old_approval.json"
        write_json(approval, approval_record(old_mask, self.ref, atlas, "Synthetic reviewer", REVIEW_ITEMS))
        revised = self.root / "revised"; revised.mkdir()
        before = digest(old_mask), digest(prepared / "prepared.json")
        adopt_electrode(prepared, self.out, revised)
        self.assertEqual(before, (digest(old_mask), digest(prepared / "prepared.json")))
        current = prepared_paths(revised)
        self.assertEqual(digest(current["mask"]), digest(candidate["outputs"]["mask"]))
        with self.assertRaisesRegex(ValueError, "stale"):
            require_approval(approval, current["mask"], self.ref, atlas)


class StimulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"; self.source.mkdir()
        self.out = self.root / "out"; self.out.mkdir()
        self.seg = self.source / "tissues.nii.gz"
        nib.save(nib.Nifti1Image(np.full((40, 40, 40), 2, np.uint8), np.diag([2., 2., 2., 1.])), self.seg)
        self.spec = {"model": "Medtronic3389", "tip_ras_mm": [40, 40, 30], "shaft_ras_mm": [40, 40, 50],
                     "contact_amplitudes": [-2, 0, 0, 0], "segmentation_reviewed": True,
                     "physical_tip_reviewed": True, "model_contact_numbering_reviewed": True,
                     "approximation_accepted": True}

    def tearDown(self):
        self.tmp.cleanup()

    def test_units_threshold_and_nonclinical_approximation(self):
        config = configuration(self.seg, {**self.spec, "control": "current", "threshold_v_per_mm": .25}, self.out)
        self.assertEqual(config["Electrodes"][0]["Contacts"][0]["Current[A]"], -.002)
        self.assertEqual(config["Surfaces"][0]["Current[A]"], .002)
        self.assertEqual(config["ActivationThresholdVTA[V-per-m]"], 250)
        self.assertEqual(config["StimulationSignal"]["Type"], "Multisine")
        self.assertFalse(config["CalcAxonActivation"])

    def test_bad_units_models_contacts_and_coordinates_refused(self):
        for update in ({"model": "Percept RC"}, {"contact_amplitudes": [-2]}, {"contact_amplitudes": [0]*4},
                       {"contact_amplitudes": [float("nan"), 0, 0, 0]}, {"tip_ras_mm": [-999, 0, 0]},
                       {"physical_tip_reviewed": False}, {"lattice_extent_mm": 60}):
            with self.assertRaises(ValueError):
                configuration(self.seg, {**self.spec, **update}, self.out)

    def test_directional_rotation_cannot_be_guessed(self):
        with self.assertRaisesRegex(ValueError, "rotation"):
            configuration(self.seg, {**self.spec, "model": "MedtronicSenSightB33005", "contact_amplitudes": [-2]+[0]*7}, self.out)

    def test_domain_must_fit_tissue_image_and_venv_link_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "domain extends"):
            configuration(self.seg, {**self.spec, "domain_mm": 150}, self.out)
        import sys
        link = self.root / "python"
        link.symlink_to(sys.executable)
        with patch.dict("os.environ", {"OSSDBS_PYTHON": str(link)}):
            self.assertEqual(solver_python(), str(link))

    def test_raw_mri_not_accepted_as_segmentation(self):
        nib.save(nib.Nifti1Image(np.full((40, 40, 40), 150, np.uint8), np.diag([2., 2., 2., 1.])), self.seg)
        with self.assertRaisesRegex(ValueError, "not a raw MRI"):
            configuration(self.seg, self.spec, self.out)

    def test_plan_never_launches_solver(self):
        with patch("dbs_connectome.stimulation.execute") as launch:
            simulate(self.seg, self.spec, self.out, 1, False)
            launch.assert_not_called()
        self.assertEqual(read_json(self.out / "provenance.json")["status"], "planned_not_executed")
        self.assertFalse((self.out / "simulation_result.json").exists())

    def test_cli_refuses_source_output_before_creating_run(self):
        from dbs_connectome.cli import main
        import contextlib
        import io
        settings = self.root / "settings.json"
        write_json(settings, self.spec)
        forbidden = self.source / "must_not_exist"
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["simulate", "--segmentation", str(self.seg), "--settings", str(settings),
                         "--threads", "1", "--output-root", str(forbidden)])
        self.assertEqual(code, 1)
        self.assertFalse(forbidden.exists())

    def test_five_tissue_pathology_is_unknown_not_blood(self):
        five = np.zeros((40, 40, 40, 5)); five[..., 2] = 1; five[0, 0, 0, :] = [0, 0, 0, 0, 1]
        path = self.source / "five.nii.gz"
        nib.save(nib.Nifti1Image(five, nib.load(self.seg).affine), path)
        result = material_from_five_tissue(self.seg, path, self.out)
        self.assertEqual(nib.load(result["segmentation"]).get_fdata()[0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
