"""Numerical report/paired-visit contracts with explicitly synthetic provenance."""
from argparse import Namespace
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dbs_connectome.cli import run
from dbs_connectome.longitudinal import cohort_reference, paired_change
from dbs_connectome.provenance import digest, read_json, record
from dbs_connectome.reports import gradient_report


class LongitudinalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nodes = self.root / "nodes.tsv"
        text = "index\tlabel\tname\themisphere\ttissue\tnetwork\n"
        for i in range(32):
            text += f"{i}\t{i+1}\tparcel{i}\t{'L' if i < 16 else 'R'}\tcortex\tSynthetic\n"
        self.nodes.write_text(text)
        self.runs = []
        rng = np.random.default_rng(57)
        for i in range(3):
            directory = self.root / f"connectome{i}"; directory.mkdir()
            m = rng.uniform(.1, 4, (32, 32)); m = (m+m.T)/2; np.fill_diagonal(m, 0)
            np.save(directory / "connectome.npy", m)
            record(directory, "connectome", [self.nodes], {"nodes_sha256": digest(self.nodes), "synthetic_test_only": True},
                   [directory / "connectome.npy"])
            self.runs.append(str(directory))

    def tearDown(self):
        self.tmp.cleanup()

    def target(self, name):
        p = self.root / name; p.mkdir(); return p

    def reference(self):
        target = self.target("cohort")
        cohort_reference(self.runs[:2], self.nodes, target, "none", True)
        return target / "reference"

    def test_mean_and_paired_csv_exact_signed_change(self):
        reference = self.reference()
        expected = sum(np.load(Path(p)/"connectome.npy") for p in self.runs[:2])/2
        np.testing.assert_allclose(np.load(reference.parent/"mean_connectome.npy"), expected)
        out = self.target("paired")
        paired_change(self.runs[0], self.runs[2], reference, self.nodes, out, True)
        with (out/"paired_gradient_change.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 32)
        a, b = [np.load(out / visit / "gradients.npz") for visit in ("first", "second")]
        np.testing.assert_allclose([float(rows[0][f"delta_G{k}"]) for k in range(1,5)],
                                   b["L_coordinates"][0,:4]-a["L_coordinates"][0,:4])
        self.assertAlmostEqual(float(rows[0]["displacement_4d"]), np.linalg.norm(b["L_coordinates"][0,:4]-a["L_coordinates"][0,:4]))
        self.assertTrue((out/"longitudinal_review.html").is_file())
        self.assertTrue((out/"second_report/results_review.html").is_file())
        self.assertFalse(read_json(out/"longitudinal_result.json")["clinical_inference"])
        a.close(); b.close()

    def test_reference_requires_review_unique_bytes_and_raw_scale(self):
        for runs, mode, review in ((self.runs[:2], "none", False), ([self.runs[0]]*2, "none", True),
                                   (self.runs[:2], "total_weight", True)):
            with self.assertRaises(ValueError):
                cohort_reference(runs, self.nodes, self.root, mode, review)

    def test_pairing_and_identical_visits_refused(self):
        reference = self.reference()
        for second, review in ((self.runs[1], False), (self.runs[0], True)):
            with self.assertRaises(ValueError):
                paired_change(self.runs[0], second, reference, self.nodes, self.root, review)

    def test_report_rejects_wrong_matrix_and_escapes_names(self):
        reference = self.reference()
        out = self.target("report")
        with self.assertRaisesRegex(ValueError, "not used"):
            gradient_report(reference, self.nodes, out, self.runs[2])
        self.nodes.write_text(self.nodes.read_text().replace("parcel0", "</script><script>alert(1)</script>"))
        ref2 = self.target("newref")
        run(Namespace(command="reference", matrix=str(Path(self.runs[0])/"connectome.npy"), nodes=self.nodes), ref2)
        gradient_report(ref2, self.nodes, out)
        html = (out/"results_review.html").read_text()
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertIn("\\u003c/script>", html)

    def test_report_mismatched_node_table_refused(self):
        reference = self.reference()
        self.nodes.write_text(self.nodes.read_text().replace("parcel0", "changed"))
        with self.assertRaisesRegex(ValueError, "node table"):
            gradient_report(reference, self.nodes, self.target("report"))


if __name__ == "__main__":
    unittest.main()
