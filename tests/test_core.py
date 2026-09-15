"""Synthetic data only; these tests do not validate performance on implanted brains."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import nibabel as nib
import numpy as np

from dbs_connectome.cli import main
from dbs_connectome.external import connectome_plan, conversion_plan, check_threads, execute, validate_mif_grid, validate_tractogram_sampling
from dbs_connectome.gradients import Settings, compact_raw, displacement, embed, geometry, load_nodes, validate_matrix
from dbs_connectome.masks import REVIEW_ITEMS, approval_record, require_approval, study_dilate, tube_mask, union_registered
from dbs_connectome.provenance import digest, image3d, new_run, same_grid, voxel_fingerprint, write_json
from dbs_connectome.qc import browser_report


class SyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.affine = np.diag([2., 2., 2., 1.])

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name, data=None, affine=None):
        path = self.root / name
        if data is None:
            data = np.ones((12, 12, 12), np.float32)
        nib.save(nib.Nifti1Image(data, self.affine if affine is None else affine), path)
        return path

    def matrix_nodes(self, count=32):
        rng = np.random.default_rng(23)
        x = rng.uniform(.1, 4, size=(count, count))
        x = (x + x.T) / 2
        np.fill_diagonal(x, 0)
        matrix = self.root / "matrix.npy"
        np.save(matrix, x)
        nodes = self.root / "nodes.tsv"
        text = "index\tlabel\tname\themisphere\ttissue\tnetwork\n"
        for i in range(count):
            text += f"{i}\t{i+1}\tparcel{i}\t{'L' if i < count//2 else 'R'}\tcortex\tSynthetic\n"
        nodes.write_text(text)
        return matrix, nodes

    def call(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)


class MaskTests(SyntheticFixture):
    def test_exact_segment_mask(self):
        mask = tube_mask((12, 12, 12), self.affine, [[[10, 10, 4], [10, 10, 18]]], 2)
        self.assertTrue(mask[5, 5, 6])
        self.assertTrue(mask[6, 5, 6])
        self.assertFalse(mask[7, 5, 6])

    def test_anisotropic_mask(self):
        affine = np.diag([1., 2., 4., 1.])
        mask = tube_mask((12, 12, 12), affine, [[[5, 10, 4], [5, 10, 32]]], 2)
        self.assertTrue(mask[7, 5, 5])
        self.assertFalse(mask[8, 5, 5])
        self.assertTrue(mask[5, 6, 5])

    def test_oblique_mask_equivariance(self):
        angle = .43
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        points = np.array([[10., 10., 4.], [10., 10., 18.]])
        expected = tube_mask((12, 12, 12), self.affine, [points], 2.1)
        affine = self.affine.copy()
        affine[:3, :3] = rotation @ affine[:3, :3]
        affine[:3, 3] = [23., -7., 11.]
        actual = tube_mask((12, 12, 12), affine, [points @ rotation.T + affine[:3, 3]], 2.1)
        np.testing.assert_array_equal(actual, expected)

    def test_bad_coordinates_rejected(self):
        for points in ([[[1000, 1000, 0], [1000, 1000, 50]]], [[[0, 0, 0], [0, 0, 0]]], []):
            with self.assertRaises(ValueError):
                tube_mask((12, 12, 12), self.affine, points)

    def test_dilation_geometry(self):
        mask = np.zeros((5, 5, 5), bool)
        mask[2, 2, 2] = True
        self.assertEqual(int(study_dilate(mask, self.affine, 1).sum()), 7)
        self.assertEqual(int(study_dilate(mask, self.affine, 0).sum()), 1)
        with self.assertRaises(ValueError):
            study_dilate(mask, np.diag([1., 1., 1., 1.]))

    def test_review_staleness(self):
        reference = self.image("b0.nii.gz")
        atlas = self.image("atlas.nii.gz")
        mask = self.image("mask.nii.gz", np.ones((12, 12, 12), np.uint8))
        approval = self.root / "approval.json"
        write_json(approval, approval_record(mask, reference, atlas, "reviewer-test", REVIEW_ITEMS))
        require_approval(approval, mask, reference, atlas)
        self.image("mask.nii.gz", np.pad(np.ones((10, 10, 10), np.uint8), 1))
        with self.assertRaises(ValueError):
            require_approval(approval, mask, reference, atlas)

    def test_incomplete_review(self):
        p = self.image("test.nii.gz")
        with self.assertRaises(ValueError):
            approval_record(p, p, p, "reviewer-test", ["registration"])

    def test_union_needs_registration_and_matching_hashes(self):
        ref = self.image("b0.nii.gz")
        a = np.zeros((12, 12, 12), np.uint8)
        a[4, 4, 4] = 1
        b = a.copy()
        b[5, 5, 5] = 1
        p1, p2 = self.image("a.nii.gz", a), self.image("b.nii.gz", b)
        rec = self.root / "reg.json"
        spec = {"reference_sha256": digest(ref), "anatomical_alignment_reviewed": True,
                "reviewer": "test", "registered_mask_sha256": [digest(p1), digest(p2)]}
        write_json(rec, spec)
        out = self.root / "union"
        out.mkdir()
        result = union_registered(ref, [p1, p2], rec, out)
        self.assertEqual(nib.load(result).get_fdata().sum(), 2)
        self.image("b.nii.gz", np.ones((12, 12, 12), np.uint8))
        with self.assertRaises(ValueError):
            union_registered(ref, [p1, p2], rec, out)


class MatrixTests(SyntheticFixture):
    def test_mapping_no_off_by_one(self):
        m = np.array([[0., 2., 7.], [2., 0., 5.], [7., 5., 0.]])
        np.testing.assert_array_equal(compact_raw(m, [3, 1]), [[0, 7], [7, 0]])
        for bad in ([0, 1], [1, 1], [1, 4], [1., 2.]):
            with self.assertRaises(ValueError):
                compact_raw(m, bad)

    def test_bad_matrices_rejected(self):
        for bad in (np.ones((2, 3)), np.eye(2), [[0, 2], [3, 0]], [[0, -1], [-1, 0]], [[0, np.nan], [np.nan, 0]]):
            with self.assertRaises(ValueError):
                validate_matrix(bad)

    def test_network_label_mismatch_rejected(self):
        path = self.root / "bad.tsv"
        path.write_text("index\tlabel\tname\themisphere\ttissue\tnetwork\n0\t1\t7Networks_LH_Vis_1\tL\tcortex\tSomMot\n")
        with self.assertRaises(ValueError):
            load_nodes(path)

    def test_embed_deterministic(self):
        path, _ = self.matrix_nodes()
        m = np.load(path)
        g, lam = embed(m, np.arange(16))
        second, _ = embed(m, np.arange(16))
        np.testing.assert_array_equal(g, second)
        self.assertEqual(g.shape, (16, 10))
        self.assertEqual(lam.shape, (10,))
        aligned, _ = embed(m, np.arange(16), reference=g)
        np.testing.assert_allclose(aligned, g, atol=1e-10)

    def test_isolated_nodes_rejected(self):
        with self.assertRaises(ValueError):
            embed(np.zeros((16, 16)), np.arange(16))

    def test_geometry_invariance_for_same_retained_subspace(self):
        rng = np.random.default_rng(21)
        a, b = rng.normal(size=(15, 4)), rng.normal(size=(15, 4))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        center = a.mean(0)
        np.testing.assert_allclose(geometry(a, center)["eccentricity"],
                                   geometry(a @ rotation, center @ rotation)["eccentricity"])
        np.testing.assert_allclose(displacement(a, b), displacement(a @ rotation, b @ rotation))
        self.assertFalse(np.allclose(np.ptp(a, axis=0), np.ptp(a @ rotation, axis=0)))


class ProvenanceTests(SyntheticFixture):
    def test_browser_report_self_contained_private(self):
        reference = self.image("reference.nii.gz")
        mask = self.image("mask.nii.gz", np.ones((12, 12, 12), np.uint8))
        page = self.root / "qc.html"
        browser_report(reference, mask, page)
        text = page.read_text()
        self.assertIn("PRIVATE ANATOMICAL DATA", text)
        self.assertIn("connect-src 'none'", text)
        self.assertNotIn("<script src", text)
        self.assertNotIn(str(self.root), text)
        self.assertIn(digest(mask), text)

    def test_browser_qc_does_not_approve(self):
        reference = self.image("reference.nii.gz")
        mask = self.image("mask.nii.gz", np.ones((12, 12, 12), np.uint8))
        self.assertEqual(self.call(["qc", "--reference", str(reference), "--mask", str(mask),
                                   "--output-root", str(self.root / "out")]), 0)
        run = next((self.root / "out").iterdir())
        self.assertFalse(json.loads((run / "provenance.json").read_text())["parameters"]["approved"])
        self.assertFalse((run / "approval.json").exists())

    def test_browser_qc_rejects_grid_mismatch(self):
        reference = self.image("reference.nii.gz")
        mask = self.image("mask.nii.gz", np.ones((12, 12, 12), np.uint8), affine=np.eye(4))
        with self.assertRaises(ValueError):
            browser_report(reference, mask, self.root / "qc.html")

    def test_decoded_duplicate_detection(self):
        a = self.image("a.nii.gz")
        changed = self.affine.copy()
        changed[0, 3] = 20
        b = self.image("b.nii.gz", affine=changed)
        self.assertNotEqual(digest(a), digest(b))
        self.assertEqual(voxel_fingerprint(a)["voxel_sha256"], voxel_fingerprint(b)["voxel_sha256"])

    def test_conflicting_orientation_rejected(self):
        path = self.image("image.nii.gz")
        image = nib.load(path)
        different = self.affine.copy()
        different[0, 3] = 100
        image.set_qform(different, code=1)
        nib.save(image, path)
        with self.assertRaises(ValueError):
            image3d(path)

    def test_grid_mismatch_rejected(self):
        a = image3d(self.image("a.nii.gz"))
        b = image3d(self.image("b.nii.gz", affine=np.eye(4)))
        with self.assertRaises(ValueError):
            same_grid(a, b)

    def test_no_overwrite(self):
        new_run(self.root, "test")
        with self.assertRaises(FileExistsError):
            new_run(self.root, "test")

    def test_conversion_is_argument_list_no_shell(self):
        commands = conversion_plan(self.root, self.root / "converted")
        self.assertIn("series_%s", commands[0])
        self.assertIn("-ba", commands[0])

    def test_threads_required(self):
        for bad in (0, -1, None, 1.5, True):
            with self.assertRaises(ValueError):
                check_threads(bad)
        check_threads(8)

    def test_private_output_permissions(self):
        out = new_run(self.root, "private")
        self.assertEqual(out.stat().st_mode & 0o077, 0)

    def test_prepared_pipeline_plan(self):
        reference = self.image("b0.nii.gz")
        mask = self.image("mask.nii.gz", np.ones((12, 12, 12), np.uint8))
        atlas = self.image("atlas.nii.gz", np.ones((12, 12, 12), np.uint8))
        nodes = self.root / "nodes.tsv"
        nodes.write_text("index\tlabel\tname\themisphere\ttissue\tnetwork\n0\t1\tparcel\tL\tcortex\tSynthetic\n")
        approval = self.root / "approval.json"
        write_json(approval, approval_record(mask, reference, atlas, "test", REVIEW_ITEMS))
        tracks = self.root / "dummy.tck"
        tracks.write_bytes(b"synthetic plan placeholder; not a tractogram")
        spec = {"reference": str(reference), "mask": str(mask), "atlas": str(atlas), "nodes": str(nodes),
                "approval": str(approval), "tractogram": str(tracks), "fod": str(reference), "five_tissue": str(reference)}
        config = self.root / "config.json"
        write_json(config, spec)
        paths, commands = connectome_plan(config, self.root / "out", 8)
        self.assertEqual([x[0] for x in commands], ["tckedit", "tcksift2", "tck2connectome"])
        self.assertIn("-exclude", commands[0])
        self.assertEqual(commands[1][1], commands[0][2])
        self.assertEqual(commands[2][1], commands[0][2])
        self.assertEqual(commands[2][commands[2].index("-tck_weights_in") + 1], commands[1][3])
        self.assertNotIn("-keep_unassigned", commands[2])
        self.assertTrue(all(x[-2:] == ["-nthreads", "8"] for x in commands))

    def test_fail_fast_external_executor(self):
        with patch("dbs_connectome.external.shutil.which", return_value="/fake"), \
             patch("dbs_connectome.external.digest", return_value="synthetic-hash"), \
             patch("dbs_connectome.external.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "fake 1.0", "")), \
             patch("dbs_connectome.external.subprocess.Popen") as launch:
            process = launch.return_value.__enter__.return_value
            process.stdout.__iter__.return_value = iter(["synthetic failure\n"])
            process.wait.return_value = 2
            process.returncode = 2
            with self.assertRaises(subprocess.CalledProcessError):
                execute([["stage1", "input"], ["stage2", "output"]], self.root, 8)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[0], ["/fake", "input"])
            self.assertEqual(launch.call_args.kwargs["cwd"], self.root.resolve())

    def test_mif_header_validation(self):
        reference = self.image("reference.nii.gz")
        def fake(command, **kwargs):
            self.assertIn("-json_all", command)
            write_json(command[-1], {"size": [12, 12, 12, 5], "spacing": [2, 2, 2, 1], "transform": np.eye(4).tolist()})
            return subprocess.CompletedProcess(command, 0)
        with patch("dbs_connectome.external.subprocess.run", side_effect=fake):
            validate_mif_grid("synthetic.mif", reference, self.root)

    def test_sparse_tractogram_rejected(self):
        reference = self.image("reference.nii.gz")
        path = self.root / "sparse.tck"
        tg = nib.streamlines.Tractogram([np.array([[0., 10., 10.], [20., 10., 10.]])], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tg, str(path))
        with self.assertRaisesRegex(ValueError, "Sparse"):
            validate_tractogram_sampling(path, reference)

    def test_dense_tractogram_accepted(self):
        reference = self.image("reference.nii.gz")
        path = self.root / "dense.tck"
        points = np.column_stack((np.linspace(0, 20, 21), np.full(21, 10.), np.full(21, 10.)))
        tg = nib.streamlines.Tractogram([points], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tg, str(path))
        self.assertEqual(validate_tractogram_sampling(path, reference)["count"], 1)

    def test_reference_hash_mismatch_blocks_comparison(self):
        matrix, nodes = self.matrix_nodes()
        common = ["--matrix", str(matrix), "--nodes", str(nodes)]
        reference_root = self.root / "reference"
        self.assertEqual(self.call(["reference", *common, "--output-root", str(reference_root)]), 0)
        reference_run = next(reference_root.iterdir())
        nodes.write_text(nodes.read_text().replace("Synthetic", "Changed"))
        self.assertEqual(self.call(["embed", *common, "--reference-run", str(reference_run),
                                   "--output-root", str(self.root / "embed")]), 1)

    def test_gradient_cli_end_to_end_synthetic(self):
        matrix, nodes = self.matrix_nodes()
        common = ["--matrix", str(matrix), "--nodes", str(nodes)]
        reference_root = self.root / "reference"
        self.assertEqual(self.call(["reference", *common, "--output-root", str(reference_root)]), 0)
        reference_run = next(reference_root.iterdir())
        for tag in ("first", "second"):
            self.assertEqual(self.call(["embed", *common, "--reference-run", str(reference_run),
                                       "--output-root", str(self.root / tag)]), 0)
        first, second = next((self.root / "first").iterdir()), next((self.root / "second").iterdir())
        self.assertEqual(self.call(["change", "--first-run", str(first), "--second-run", str(second),
                                   "--nodes", str(nodes), "--output-root", str(self.root / "change")]), 0)
        saved = next((self.root / "change").glob("*/parcel_change.json"))
        self.assertTrue(all(row["displacement_4d"] == 0 for row in json.loads(saved.read_text())))
        # Tamper with an upstream output: downstream must refuse it.
        with (first / "gradients.npz").open("ab") as stream:
            stream.write(b"tampered")
        self.assertEqual(self.call(["change", "--first-run", str(first), "--second-run", str(second),
                                   "--nodes", str(nodes), "--output-root", str(self.root / "badchange")]), 1)


if __name__ == "__main__":
    unittest.main()
