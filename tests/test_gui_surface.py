"""Static interface contracts; these checks do not replace visual/browser testing."""
from html.parser import HTMLParser
from importlib.resources import files
import re
import unittest

from dbs_connectome.gui import FIELDS
from dbs_connectome.qc import TEMPLATE


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.identifiers = []
        self.labels = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.identifiers.append(attrs["id"])
        if tag == "label" and "for" in attrs:
            self.labels.append(attrs["for"])


class GuiSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.html = files("dbs_connectome").joinpath("static/index.html").read_text()

    def test_unique_ids_and_label_targets(self):
        parsed = Controls()
        parsed.feed(self.html)
        self.assertEqual(len(parsed.identifiers), len(set(parsed.identifiers)))
        self.assertTrue(set(parsed.labels).issubset(parsed.identifiers))

    def test_every_backend_stage_and_input_is_exposed(self):
        for stage, required in FIELDS.items():
            entry = re.search(r"^'?" + re.escape(stage) + r"'?:\{[^\n]+", self.html, re.MULTILINE)
            self.assertIsNotNone(entry, stage)
            for name in required:
                self.assertIn("'" + name + "'", entry.group(0))
        self.assertIn("connectome_run", self.html)
        self.assertIn("approve_compute:", self.html)
        self.assertIn("threads:Number(", self.html)

    def test_long_resource_paths_do_not_force_grid_width(self):
        self.assertIn("grid-template-columns:repeat(2,minmax(0,1fr))", self.html)
        self.assertIn(".workflow-grid>div{min-width:0}", self.html)
        self.assertIn(".workflow-status{overflow-wrap:anywhere}", self.html)

    def test_private_qc_and_development_limits_remain_visible(self):
        self.assertIn("Research use only", self.html)
        self.assertIn("not independently validated end to end", self.html)
        self.assertIn("QC is never approved automatically", self.html)
        self.assertIn("End-to-end DICOM reconstruction is not yet validated", self.html)
        self.assertIn("No approval is recorded here", TEMPLATE)
        self.assertIn("__NOTICE__", TEMPLATE)
        self.assertIn("connect-src 'none'", TEMPLATE)


if __name__ == "__main__":
    unittest.main()
