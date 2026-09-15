"""Synthetic DICOM only. These tests do not establish real scanner compatibility."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dbs_connectome.intake import convert_selection, inventory, validate_selection
from dbs_connectome.provenance import assert_inputs_unchanged, digest, freeze_inputs, read_json, record, require_separate_output, write_json
from dbs_connectome.reconstruction import diffusion_summary, reconstruction_plan
from dbs_connectome.environment import discover


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "dicom"
        self.data.mkdir()
        self.inv = self.root / "inventory"
        self.inv.mkdir()
        self.study = generate_uid()

    def tearDown(self):
        self.tmp.cleanup()

    def dicom(self, name, patient="TEST-A", series=None, study=None, modality="MR", description="T1"):
        path = self.data / name
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient
        ds.PatientName = "SYNTHETIC^PRIVATE_SENTINEL"
        ds.PatientBirthDate = "19000101"
        ds.StudyInstanceUID = study or self.study
        ds.SeriesInstanceUID = series or generate_uid()
        ds.Modality = modality
        ds.SeriesDescription = description
        ds.Rows, ds.Columns = 8, 8
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
        ds.save_as(path, enforce_file_format=True)
        return ds

    def selection(self, report):
        return {"subject": "sub-001", "visit": "post1", "identity_reviewed": True,
                "roles": {"t1": report["series"][0]["id"], "dwi": report["series"][1]["id"]}}

    def test_header_inventory_does_not_export_name_or_patient_id(self):
        self.dicom("a.dcm")
        self.dicom("b.dcm", description="DWI")
        report = inventory(self.data, self.inv)
        text = (self.inv / "inventory.json").read_text()
        self.assertNotIn("SYNTHETIC^PRIVATE_SENTINEL", text)
        self.assertNotIn("TEST-A", text)
        self.assertEqual(len(report["series"]), 2)
        self.assertEqual(len({s["patient_key"] for s in report["series"]}), 1)
        self.assertEqual(set(validate_selection(self.inv, self.selection(report))), {"t1", "dwi"})

    def test_cross_patient_blocked(self):
        self.dicom("a.dcm")
        self.dicom("b.dcm", patient="TEST-B")
        report = inventory(self.data, self.inv)
        with self.assertRaisesRegex(ValueError, "identifiers"):
            validate_selection(self.inv, self.selection(report))

    def test_missing_patient_id_blocked(self):
        self.dicom("a.dcm", patient="")
        self.dicom("b.dcm", patient="")
        report = inventory(self.data, self.inv)
        with self.assertRaises(ValueError):
            validate_selection(self.inv, self.selection(report))

    def test_different_studies_need_explicit_review(self):
        self.dicom("a.dcm")
        self.dicom("b.dcm", study=generate_uid())
        report = inventory(self.data, self.inv)
        selection = self.selection(report)
        with self.assertRaisesRegex(ValueError, "different studies"):
            validate_selection(self.inv, selection)
        selection["cross_study_reviewed"] = True
        validate_selection(self.inv, selection)

    def test_duplicate_and_changed_files_blocked(self):
        ds = self.dicom("a.dcm")
        ds.save_as(self.data / "duplicate.dcm", enforce_file_format=True)
        self.dicom("b.dcm")
        report = inventory(self.data, self.inv)
        self.assertTrue(any(s["errors"] for s in report["series"]))
        with self.assertRaises(ValueError):
            validate_selection(self.inv, self.selection(report))

    def test_inventory_hash_and_source_stat_checked(self):
        self.dicom("a.dcm")
        self.dicom("b.dcm")
        report = inventory(self.data, self.inv)
        with (self.data / "a.dcm").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            validate_selection(self.inv, self.selection(report))

    def test_symlink_escape_not_scanned(self):
        self.dicom("a.dcm")
        outside = self.root / "outside"
        outside.mkdir()
        (self.data / "linked").symlink_to(outside)
        report = inventory(self.data, self.inv)
        self.assertEqual(report["files_scanned"], 1)

    def test_import_dry_run_does_not_copy_or_execute(self):
        self.dicom("a.dcm")
        self.dicom("b.dcm")
        report = inventory(self.data, self.inv)
        out = self.root / "planned"
        out.mkdir()
        with patch("dbs_connectome.intake.execute") as run:
            convert_selection(self.inv, self.selection(report), out)
            run.assert_not_called()
        self.assertFalse((out / "dicom_staging").exists())
        self.assertEqual(read_json(out / "provenance.json")["status"], "planned_not_executed")

    def test_readonly_environment_does_not_launch_programs(self):
        with patch("subprocess.run") as run, patch("subprocess.Popen") as popen:
            report = discover()
            run.assert_not_called()
            popen.assert_not_called()
        self.assertIn("stages", report)

    def test_output_overlap_and_symlink_overlap_rejected(self):
        linked = self.root / "alias"
        linked.symlink_to(self.data)
        for output in (self.data, self.data / "results", linked / "results"):
            with self.assertRaisesRegex(ValueError, "read-only"):
                require_separate_output(output, self.data)

    def test_source_hash_change_detected(self):
        self.dicom("a.dcm")
        path = self.data / "a.dcm"
        before = freeze_inputs([path])
        assert_inputs_unchanged(before)
        with path.open("ab") as f:
            f.write(b"synthetic mutation")
        with self.assertRaisesRegex(ValueError, "input changed"):
            assert_inputs_unchanged(before)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.imported = self.root / "imported"
        self.imported.mkdir()
        rng = np.random.default_rng(13)
        vectors = rng.normal(size=(3, 30))
        vectors /= np.linalg.norm(vectors, axis=0)
        self.bvec = np.column_stack([np.zeros(3), vectors])
        self.bval = np.r_[0, np.full(30, 1000)]
        self.meta = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": .08,
                     "PartialFourier": 1, "SliceEncodingDirection": "k"}
        self.dwi = self.artifact("dwi", (6, 6, 6, 31))
        self.t1 = self.artifact("t1", (6, 6, 6))
        self.reverse = self.artifact("reverse_b0", (6, 6, 6))
        write_json(self.imported / "converted.json", {"artifacts": [self.dwi, self.t1, self.reverse]})
        record(self.imported, "import-session", [], {}, [self.imported / "converted.json"])
        self.choices = {"artifacts": {"dwi": self.dwi["id"], "t1": self.t1["id"]},
            "phase_encoding": "j-", "readout_seconds": .08, "fod_model": "wm_csf",
            "distortion": "none", "degibbs_axes": "0,1", "acquisition_reviewed": True,
            "uncorrected_distortion_accepted": True}

    def tearDown(self):
        self.tmp.cleanup()

    def artifact(self, role, shape):
        path = self.imported / (role + ".nii.gz")
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.diag([2., 2., 2., 1.])), path)
        meta = dict(self.meta)
        if role == "reverse_b0":
            meta["PhaseEncodingDirection"] = "j"
        sidecar = self.imported / (role + ".json")
        write_json(sidecar, meta)
        a = {"id": role, "role": role, "image": str(path), "json": str(sidecar)}
        if role == "dwi":
            for key, data in (("bval", self.bval), ("bvec", self.bvec)):
                target = self.imported / (role + "." + key)
                np.savetxt(target, data)
                a[key] = str(target)
        a["hashes"] = {v: digest(v) for k, v in a.items() if k not in ("id", "role")}
        return a

    def plan(self):
        return reconstruction_plan(self.imported, self.choices, self.root / "output", 12)

    def test_single_shell_two_tissue_plan_and_native_first(self):
        _, summary, commands, _ = self.plan()
        self.assertEqual(summary["shells"], [{"b": 1000., "directions": 30}])
        names = [c[0] for c in commands]
        self.assertLess(names.index("dwifslpreproc"), names.index("mrgrid"))
        csd = next(c for c in commands if c[0] == "dwi2fod")
        self.assertNotIn(str(self.root / "output/gm.txt"), csd)
        self.assertNotIn("tckgen", names)
        self.assertNotIn("tck2connectome", names)
        eddy = next(c for c in commands if c[0] == "dwifslpreproc")
        self.assertIn("-rpe_none", eddy)
        self.assertIn("j-", eddy)
        self.assertIn("0.08", eddy)

    def test_three_tissue_single_shell_refused(self):
        self.choices["fod_model"] = "wm_gm_csf"
        with self.assertRaisesRegex(ValueError, "two nonzero shells"):
            self.plan()

    def test_no_topup_acknowledgement_required(self):
        self.choices["uncorrected_distortion_accepted"] = False
        with self.assertRaisesRegex(ValueError, "susceptibility distortion"):
            self.plan()

    def test_metadata_override_needs_reason(self):
        self.choices["phase_encoding"] = "j"
        with self.assertRaisesRegex(ValueError, "documented source"):
            self.plan()
        self.choices["metadata_reason"] = "Synthetic protocol correction for test"
        self.plan()

    def test_invalid_axes_refused(self):
        self.choices["degibbs_axes"] = "1,2"
        with self.assertRaisesRegex(ValueError, "slice axis"):
            self.plan()

    def test_changed_converted_data_blocked(self):
        with Path(self.dwi["bval"]).open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.plan()

    def test_paired_reverse_requires_matching_metadata(self):
        self.choices.update(distortion="paired", reverse_contrast_reviewed=True)
        self.choices["artifacts"]["reverse_b0"] = "reverse_b0"
        _, _, commands, _ = self.plan()
        eddy = next(c for c in commands if c[0] == "dwifslpreproc")
        self.assertIn("-rpe_pair", eddy)
        self.assertIn("-align_seepi", eddy)
        self.assertEqual(eddy[eddy.index("-topup_options") + 1], " --nthr=12")
        self.assertTrue(any("-coord" in c and c[c.index("-coord")+1:c.index("-coord")+3] == ["3", "0"] for c in commands))

    def test_bad_gradient_table_is_not_repaired_silently(self):
        np.savetxt(self.dwi["bvec"], np.zeros((3, 31)))
        with self.assertRaisesRegex(ValueError, "unit length"):
            diffusion_summary(self.dwi)

    def reverse_series(self, values):
        # Replace synthetic fixtures only; production records remain write-once.
        Path(self.reverse["json"]).unlink()
        (self.imported / "converted.json").unlink()
        (self.imported / "provenance.json").unlink()
        self.reverse = self.artifact("reverse_b0", (6, 6, 6, 3))
        if values is not None:
            path = self.imported / "reverse_b0.bval"
            np.savetxt(path, values)
            self.reverse["bval"] = str(path)
            self.reverse["hashes"][str(path)] = digest(path)
        write_json(self.imported / "converted.json", {"artifacts": [self.dwi, self.t1, self.reverse]})
        record(self.imported, "import-session", [], {}, [self.imported / "converted.json"])
        self.choices.update(distortion="paired", reverse_contrast_reviewed=True)
        self.choices["artifacts"]["reverse_b0"] = "reverse_b0"

    def test_reverse_diffusion_volumes_are_excluded_before_averaging(self):
        self.reverse_series([0, 1000, 0])
        _, summary, commands, _ = self.plan()
        self.assertEqual(summary["reverse_pe"]["selected_b0_indices"], [0, 2])
        extraction = next(c for c in commands if "reverse_selected_b0.mif" in " ".join(c) and c[0] == "mrconvert")
        self.assertEqual(extraction[extraction.index("-coord") + 1:extraction.index("-coord") + 3], ["3", "0,2"])
        average = next(c for c in commands if c[0] == "mrmath" and c[3].endswith("reverse_b0.mif"))
        self.assertTrue(average[1].endswith("reverse_selected_b0.mif"))

    def test_reverse_missing_invalid_or_no_baseline_bvalues_fail(self):
        for values, message in [(None, "requires b-values"), ([0, 1000], "volume count"),
                                ([1000, 1000, 1000], "no b0"), ([0, float("nan"), 0], "invalid")]:
            with self.subTest(values=values):
                self.reverse_series(values)
                with self.assertRaisesRegex(ValueError, message):
                    self.plan()

    def test_volume_gradient_count_mismatch(self):
        np.savetxt(self.dwi["bval"], np.zeros(30))
        with self.assertRaisesRegex(ValueError, "volume count"):
            diffusion_summary(self.dwi)


if __name__ == "__main__":
    unittest.main()
