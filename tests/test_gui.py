"""Local GUI boundary checks using synthetic files, never clinical data."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np

from dbs_connectome.gui import JobManager, ThreadingHTTPServer, make_handler


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.manager = JobManager(self.data, self.root / "outputs")

    def tearDown(self):
        self.manager.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def image(self, name):
        path = self.data / name
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), np.diag([2., 2., 2., 1.])), path)
        return str(path)

    def test_path_escape_and_symlink_rejected(self):
        secret = self.root / "outside.txt"
        secret.write_text("synthetic private sentinel")
        with self.assertRaises(ValueError):
            self.manager.resolve_input(secret)
        link = self.data / "link.txt"
        link.symlink_to(secret)
        with self.assertRaises(ValueError):
            self.manager.resolve_input(link)

    def test_unknown_command_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.arguments({"stage": "shell", "inputs": {}}, self.root)

    def test_optional_mask_lineage_is_forwarded(self):
        matrix = self.data / "matrix.npy"
        nodes = self.data / "nodes.tsv"
        matrix.touch()
        nodes.touch()
        run = self.data / "prepared_run"
        run.mkdir()
        args = self.manager.arguments({"stage": "embed", "inputs": {
            "matrix": str(matrix), "nodes": str(nodes), "reference_run": str(run),
            "connectome_run": str(run)}}, self.root)
        self.assertEqual(args[-2:], ["--connectome-run", str(run.resolve())])

    def test_queued_cancellation_persists(self):
        gate = threading.Event()
        self.manager.pool.submit(gate.wait)
        try:
            identifier = self.manager.submit({"stage": "fingerprint", "inputs": {
                "image": self.image("pending.nii.gz")}})
            self.manager.cancel(identifier)
            job = self.manager.jobs[identifier]
            saved = json.loads((Path(job["output_root"]) / "job_result.json").read_text())
            self.assertEqual(saved["status"], "cancelled")
            self.assertEqual(saved["current_step"], "cancelled")
        finally:
            gate.set()

    def test_heavy_execution_requires_consent(self):
        with self.assertRaises(ValueError):
            self.manager.arguments({"stage": "convert", "inputs": {"dicom_dir": str(self.data)},
                                    "execute": True, "approve_compute": False}, self.root)

    def test_qc_background_job_progress_and_result(self):
        identifier = self.manager.submit({"stage": "qc", "inputs": {"reference": self.image("ref.nii.gz"),
                                                                       "mask": self.image("mask.nii.gz")}})
        deadline = time.monotonic() + 15
        while self.manager.jobs[identifier]["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(.02)
        job = self.manager.state()["jobs"][0]
        self.assertEqual(job["status"], "completed", job["logs"])
        self.assertTrue(Path(job["report"]).exists())
        self.assertGreaterEqual(job["elapsed_seconds"], 0)

    def test_http_authentication_and_origin(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.manager, "synthetic-token"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/state", timeout=2)
            self.assertEqual(error.exception.code, 403)
            request = Request(base + "/api/state", headers={"Authorization": "Bearer synthetic-token"})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(json.load(response)["version"], "0.1.0.dev0")
            request = Request(base + "/api/state", headers={"Authorization": "Bearer synthetic-token", "Origin": "https://untrusted.example"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
