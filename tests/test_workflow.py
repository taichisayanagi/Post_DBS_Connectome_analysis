"""Synthetic integration and refusal cases. External-tool mocks are explicit."""

from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from dbs_connectome.anatomy import build_hybrid, ct_centerlines, normalize_five_tissue, anatomical_plan
from dbs_connectome.external import audit_mask_exclusion, filter_segment_crossings, _segment_intersects_mask, validate_sift2_weights
from dbs_connectome.nifti import import_nifti, validate_selection
from dbs_connectome.provenance import digest, read_json, write_json, record
from dbs_connectome.reconstruction import selected_artifacts, reconstruction_plan
from dbs_connectome.workflow import finish_session, prepared_paths, verified_reconstruction
from dbs_connectome.masks import approval_record, REVIEW_ITEMS
from dbs_connectome.qc import browser_report


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.out = self.root / "processed"
        self.out.mkdir()
        self.affine = np.diag([2., 2., 2., 1.])

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, data, directory=None):
        path = (directory or self.source) / name
        nib.save(nib.Nifti1Image(np.asarray(data, np.float32), self.affine), path)
        return str(path)

    def nifti_selection(self):
        t1 = self.image("t1.nii.gz", np.ones((8, 8, 8)))
        dwi = self.image("dwi.nii.gz", np.ones((8, 8, 8, 31)))
        vectors = np.random.default_rng(4).normal(size=(3, 30))
        vectors /= np.linalg.norm(vectors, axis=0)
        np.savetxt(self.source / "dwi.bval", np.r_[0, np.full(30, 1000)])
        np.savetxt(self.source / "dwi.bvec", np.column_stack([np.zeros(3), vectors]))
        write_json(self.source / "dwi.json", {"PhaseEncodingDirection": "j-", "TotalReadoutTime": .08})
        return {"source_root": str(self.source), "subject": "synthetic", "visit": "post1",
                "identity_reviewed": True, "raw_dwi_reviewed": True,
                "roles": {"t1": {"image": t1}, "dwi": {"image": dwi,
                    **{k: str(self.source / ("dwi." + k)) for k in ("bval", "bvec", "json")}}}}

    def test_nifti_copy_preserves_bytes_stat_and_modes_and_feeds_reconstruction(self):
        selection = self.nifti_selection()
        before = {p: (digest(p), p.stat().st_mtime_ns, p.stat().st_mode) for p in self.source.iterdir()}
        result = import_nifti(selection, self.out)
        self.assertEqual(result["source_format"], "NIfTI")
        for path, expected in before.items():
            self.assertEqual((digest(path), path.stat().st_mtime_ns, path.stat().st_mode), expected)
        choices = {"artifacts": {"t1": "t1", "dwi": "dwi"}, "acquisition_reviewed": True,
                   "phase_encoding": "j-", "readout_seconds": .08, "fod_model": "wm_csf",
                   "distortion": "none", "uncorrected_distortion_accepted": True, "degibbs_axes": "skip"}
        _, summary, commands, outputs = reconstruction_plan(self.out, choices, self.root / "recon", 12)
        self.assertEqual(summary["volumes"], 31)
        self.assertIn("t1_transform", outputs)
        self.assertTrue(all(str(self.source) not in " ".join(c) for c in commands))

    def test_nifti_identity_confirmation_required(self):
        selection = self.nifti_selection()
        selection["identity_reviewed"] = False
        with self.assertRaisesRegex(ValueError, "Confirm"):
            import_nifti(selection, self.out)

    def test_nifti_output_inside_original_rejected(self):
        selection = self.nifti_selection()
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate_selection(selection, self.source / "results")

    def test_nifti_missing_sidecar_not_invented(self):
        selection = self.nifti_selection()
        del selection["roles"]["dwi"]["json"]
        result = import_nifti(selection, self.out)
        dwi = next(a for a in result["artifacts"] if a["role"] == "dwi")
        self.assertIsNone(dwi["phase_encoding"])
        self.assertEqual(read_json(dwi["json"]), {})

    def test_nifti_changed_private_copy_refused(self):
        selection = self.nifti_selection()
        import_nifti(selection, self.out)
        with (self.out / "dwi/dwi.bval").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            selected_artifacts(self.out, {"artifacts": {"t1": "t1", "dwi": "dwi"}})

    def nodes(self, rows):
        path = self.source / "nodes.tsv"
        fields = ["index", "label", "name", "hemisphere", "tissue", "network", "source_label"]
        with path.open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(fields)
            writer.writerows(rows)
        return str(path)

    def test_hybrid_dense_mapping_and_cortical_precedence(self):
        s, l, r = [np.zeros((8, 8, 8)) for _ in range(3)]
        s[1, 1, 1] = 1001
        s[6, 1, 1] = 2001
        l[1, 1, 1] = l[2, 2, 2] = 42
        r[5, 2, 2] = 42
        nodes = self.nodes([[0, 1, "7Networks_LH_Vis_1", "L", "cortex", "Vis", 1],
                            [1, 2, "7Networks_RH_Vis_1", "R", "cortex", "Vis", 201],
                            [2, 3, "synthetic L", "L", "subcortex", "Subcortical", 42],
                            [3, 4, "synthetic R", "R", "subcortex", "Subcortical", 42]])
        paths = [self.image(name + ".nii.gz", v) for name, v in zip(("s", "l", "r"), (s, l, r))]
        output = self.out / "hybrid.nii.gz"
        build_hybrid(*paths, nodes, output)
        actual = np.asanyarray(nib.load(output).dataobj)
        self.assertEqual(actual[1, 1, 1], 1)
        self.assertEqual(actual[2, 2, 2], 3)
        self.assertEqual(actual[5, 2, 2], 4)

    def test_five_tissue_normalization(self):
        ref = self.image("ref.nii.gz", np.ones((8, 8, 8)))
        volumes = [self.image(f"t{i}.nii.gz", np.full((8, 8, 8), .4)) for i in range(5)]
        output = self.out / "5tt.nii.gz"
        normalize_five_tissue(volumes, ref, output)
        self.assertTrue(np.allclose(nib.load(output).get_fdata().sum(-1), 1))

    def ct_fixture(self):
        ref = self.image("ref.nii.gz", np.ones((20, 20, 20)))
        brain = self.image("brain.nii.gz", np.ones((20, 20, 20)))
        data = np.zeros((20, 20, 20))
        data[5, 7, 2:18] = data[14, 7, 2:18] = 3000
        return ref, brain, self.image("ct.nii.gz", data)

    def test_ct_candidates_are_unapproved_and_reference_bound(self):
        ref, brain, ct = self.ct_fixture()
        path = ct_centerlines(ct, brain, ref, self.out, 2000, 2)
        result = read_json(path)
        self.assertEqual(len(result["paths"]), 2)
        self.assertFalse(result["approved"])
        self.assertEqual(result["reference_sha256"], digest(ref))

    def test_ambiguous_ct_candidates_fail_instead_of_cherry_picking(self):
        ref, brain, ct = self.ct_fixture()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ct_centerlines(ct, brain, ref, self.out, 2000, 1)

    def track(self, name, points):
        path = self.source / (name + ".tck")
        t = nib.streamlines.Tractogram([np.asarray(points, np.float32)], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(t, str(path))
        return path

    def test_continuous_audit_detects_crossing_between_vertices(self):
        m = np.zeros((8, 8, 8)); m[3, 3, 3] = 1
        mask = self.image("mask.nii.gz", m)
        # Clips a voxel corner, while both endpoints fall outside the masked voxel.
        track = self.track("corner", [[4.8, 5.2, 6], [5.2, 4.8, 6]])
        with self.assertRaisesRegex(ValueError, "intersects"):
            audit_mask_exclusion(track, mask)

    def test_continuous_audit_accepts_nonintersecting_track(self):
        m = np.zeros((8, 8, 8)); m[3, 3, 3] = 1
        mask = self.image("mask.nii.gz", m)
        track = self.track("clear", [[2, 2, 2], [2.3, 2, 2]])
        self.assertEqual(audit_mask_exclusion(track, mask)["mask_intersections"], 0)

    def test_strict_filter_removes_corner_without_changing_retained_vertices(self):
        m = np.zeros((8, 8, 8)); m[3, 3, 3] = 1
        mask = self.image("mask.nii.gz", m)
        clear = np.array([[2, 2, 2], [2.3, 2, 2]], np.float32)
        corner = np.array([[4.8, 5.2, 6], [5.2, 4.8, 6]], np.float32)
        source = self.source / "tracks.tck"
        nib.streamlines.save(nib.streamlines.Tractogram([corner, clear], affine_to_rasmm=np.eye(4)), str(source))
        before = digest(source), digest(mask)
        target = self.out / "strict.tck"
        counts = filter_segment_crossings(source, mask, target)
        self.assertEqual(counts["additional_removed"], 1)
        self.assertEqual(counts["retained_streamlines"], 1)
        actual = nib.streamlines.load(str(target)).streamlines
        np.testing.assert_array_equal(actual[0], clear)
        self.assertEqual(audit_mask_exclusion(target, mask)["mask_intersections"], 0)
        self.assertEqual(before, (digest(source), digest(mask)))
        with self.assertRaisesRegex(ValueError, "new output"):
            filter_segment_crossings(source, mask, target)

    def test_strict_filter_rejects_empty_retained_set(self):
        m = np.zeros((8, 8, 8)); m[3, 3, 3] = 1
        mask = self.image("mask.nii.gz", m)
        source = self.track("inside", [[6, 6, 6], [6.2, 6, 6]])
        with self.assertRaisesRegex(ValueError, "No streamlines remain"):
            filter_segment_crossings(source, mask, self.out / "empty.tck")

    def test_segment_predicate_matches_bruteforce_boxes_and_oblique_space(self):
        rng = np.random.default_rng(482)
        mask = rng.random((6, 6, 6)) < .15
        cells = np.argwhere(mask)
        angle = .37
        affine = np.array([[2*np.cos(angle), -2*np.sin(angle), 0, 10],
                           [2*np.sin(angle), 2*np.cos(angle), 0, -13], [0, 0, 2, 6], [0, 0, 0, 1]])
        def oracle(a, b):
            for cell in cells:
                enter, leave = 0., 1.
                for axis in range(3):
                    delta = b[axis] - a[axis]
                    if abs(delta) < 1e-12:
                        if not cell[axis] - .5 <= a[axis] <= cell[axis] + .5:
                            leave = -1.; break
                    else:
                        t0, t1 = sorted(((cell[axis] - .5 - a[axis])/delta, (cell[axis] + .5 - a[axis])/delta))
                        enter, leave = max(enter, t0), min(leave, t1)
                if enter <= leave:
                    return True
            return False
        for _ in range(350):
            a = rng.uniform(.5, 4.5, 3)
            b = a + rng.uniform(-.35, .35, 3)
            track = np.array([a, b])
            expected = oracle(a, b)
            self.assertEqual(_segment_intersects_mask(track, np.eye(4), mask), expected)
            world = nib.affines.apply_affine(affine, track)
            self.assertEqual(_segment_intersects_mask(world, np.linalg.inv(affine), mask), expected)

    def test_sift2_weights_are_bound_to_retained_track_count(self):
        w, m = self.out / "weights.txt", self.out / "mu.txt"
        np.savetxt(w, [0, 1.2, 3.4]); np.savetxt(m, [.4])
        self.assertEqual(validate_sift2_weights(w, m, 3)["weight_count"], 3)
        with self.assertRaisesRegex(ValueError, "count differs"):
            validate_sift2_weights(w, m, 4)
        for bad in ([0, -1, 2], [0, float('nan'), 2], [0, 0, 0]):
            np.savetxt(w, bad)
            with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
                validate_sift2_weights(w, m, 3)
        np.savetxt(w, [1])
        np.savetxt(m, [-1])
        with self.assertRaisesRegex(ValueError, "mu must"):
            validate_sift2_weights(w, m, 1)

    def test_qc_contains_atlas_and_additional_backgrounds(self):
        ref, brain, ct = self.ct_fixture()
        output = self.out / "review.html"
        browser_report(ref, brain, output, atlas=brain, backgrounds={"CT": ct})
        text = output.read_text()
        self.assertIn('"CT":', text)
        self.assertIn('"atlas_boundary":', text)
        self.assertIn("No approval is recorded", text)

    def test_anatomical_plan_uses_private_subjects_and_explicit_interpolation(self):
        resources = {"freesurfer_home": Path("/installed/fs"), "nextbrain_script": Path("/installed/segment.py"),
                     "nextbrain_atlas": Path("/installed/atlas"), "lh_annotation": Path("/atlas/lh.annot"), "rh_annotation": Path("/atlas/rh.annot")}
        commands = anatomical_plan("/private/t1.nii.gz", "/private/t1_to_b0.mat", "/private/b0.nii.gz", resources, "/private/work", 12)
        self.assertIn("/private/work/subjects", commands[0])
        self.assertTrue(any("-openmp" in c and "12" in c for c in commands))
        self.assertEqual(sum(c[0] == "antsApplyTransforms" for c in commands), 5)
        self.assertTrue(all("Linear" in c for c in commands if c[0] == "antsApplyTransforms"))
        self.assertTrue(all("--nearest" in c for c in commands if c[0].endswith("mri_vol2vol")))

    def test_nextbrain_unsafe_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "without spaces"):
            anatomical_plan("t1", "t", "r", {"freesurfer_home": Path("/fs"), "nextbrain_atlas": Path("/atlas")}, "/private/a path", 12)

    def prepared_fixture(self):
        nodes = self.nodes([[i, i+1, f"synthetic_{i}", "L" if i < 14 else "R", "cortex", "Vis", i+1] for i in range(28)])
        ref = self.image("ref.nii.gz", np.ones((8, 8, 8)))
        mask_data = np.zeros((8, 8, 8)); mask_data[6, 6, 6] = 1
        mask = self.image("mask.nii.gz", mask_data)
        atlas_data = np.zeros((8, 8, 8)); atlas_data.flat[:28] = np.arange(1, 29)
        atlas = self.image("atlas.nii.gz", atlas_data)
        fod = self.image("fod.nii.gz", np.ones((8, 8, 8, 6)))
        five = self.image("five.nii.gz", np.full((8, 8, 8, 5), .2))
        paths = dict(reference=ref, mask=mask, atlas=atlas, fod=fod, five_tissue=five, nodes=nodes)
        prepared = self.root / "prepared"; prepared.mkdir()
        write_json(prepared / "prepared.json", {"outputs": paths, "hashes": {p: digest(p) for p in paths.values()}, "missing_dwi_nodes": []})
        record(prepared, "prepare-session", [], {}, [prepared / "prepared.json"])
        approval = self.source / "approval.json"
        write_json(approval, approval_record(mask, ref, atlas, "SYNTHETIC_TEST", REVIEW_ITEMS))
        return prepared, approval, paths

    def test_finish_dry_run_never_launches_tools(self):
        prepared, approval, _ = self.prepared_fixture()
        with patch("dbs_connectome.workflow.execute") as execute:
            finish_session(prepared, approval, self.out, 12)
            execute.assert_not_called()
        self.assertEqual(read_json(self.out / "provenance.json")["status"], "planned_not_executed")
        self.assertFalse((self.out / "tractography").exists())

    def test_changed_prepared_mask_cannot_reuse_approval(self):
        prepared, approval, paths = self.prepared_fixture()
        with Path(paths["mask"]).open("ab") as f:
            f.write(b"synthetic corruption")
        with self.assertRaisesRegex(ValueError, "changed"):
            finish_session(prepared, approval, self.out, 12)

    def test_mocked_external_end_to_end_handoff_and_gradient_lineage(self):
        # Mocked MRI tools test orchestration only; exclusion audit and gradients run for real.
        prepared, approval, paths = self.prepared_fixture()
        before = {p: digest(p) for p in paths.values()}
        rng = np.random.default_rng(61)
        matrix = rng.uniform(1, 5, (28, 28)); matrix += matrix.T; np.fill_diagonal(matrix, 0)
        calls = []
        def fake_execute(commands, out, *args, **kwargs):
            for c in commands:
                calls.append(c[0])
                if c[0] in ("tckgen", "tckedit"):
                    target = c[2]
                    tr = nib.streamlines.Tractogram([np.array([[2, 2, 2], [2.3, 2, 2]], np.float32)], affine_to_rasmm=np.eye(4))
                    nib.streamlines.save(tr, target)
                if c[0] == "tck2connectome":
                    np.savetxt(c[3], matrix, delimiter=",")
                if c[0] == "tcksift2":
                    np.savetxt(c[3], [1.])
                    np.savetxt(c[c.index("-out_mu") + 1], [1.])
        with patch("dbs_connectome.workflow.execute", side_effect=fake_execute), patch("dbs_connectome.cli.execute", side_effect=fake_execute), patch("dbs_connectome.workflow.validate_mif_grid"), patch("dbs_connectome.cli.validate_mif_grid"):
            finish_session(prepared, approval, self.out, 12, streamlines=1000, execute_tools=True)
        self.assertEqual(calls, ["5tt2gmwmi", "tckgen", "tckedit", "tcksift2", "tck2connectome"])
        self.assertEqual(read_json(self.out / "connectome/exclusion_audit.json")["mask_intersections"], 0)
        self.assertEqual(read_json(self.out / "connectome/sift2_audit.json")["weight_count"], 1)
        self.assertEqual(read_json(self.out / "session_result.json")["alignment"], "UNALIGNED_SINGLE_SESSION")
        self.assertTrue((self.out / "gradient/gradients.npz").exists())
        for p, expected in before.items():
            self.assertEqual(digest(p), expected)


if __name__ == "__main__":
    unittest.main()
