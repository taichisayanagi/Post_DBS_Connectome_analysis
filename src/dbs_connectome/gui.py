"""Loopback-only research GUI with allowlisted jobs, explicit resource consent and logs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

from .masks import REVIEW_ITEMS
from .provenance import new_run, read_json, write_json

FIELDS = {
    "convert": ["dicom_dir"], "mask": ["reference", "centerlines"], "qc": ["reference", "mask"],
    "approve": ["reference", "mask", "atlas"], "connectome": ["config"],
    "reference": ["matrix", "nodes"], "embed": ["matrix", "nodes", "reference_run"],
    "change": ["first_run", "second_run", "nodes"], "fingerprint": ["image"],
}


class JobManager:
    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root).expanduser().resolve(strict=True)
        if not self.data_root.is_dir():
            raise ValueError("Data root must be a directory")
        self.root = new_run(output_root, "gui_session")
        self.jobs = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def resolve_input(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        path = path.resolve(strict=True)
        roots = (self.data_root, self.root.parent)
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError("Input path lies outside the configured data/output roots")
        return path

    def arguments(self, body, job_root):
        stage = body.get("stage")
        if stage not in FIELDS:
            raise ValueError("Unsupported stage; registration/preprocessing are not implemented yet")
        supplied = body.get("inputs", {})
        if not isinstance(supplied, dict):
            raise ValueError("Inputs must be an object")
        args = [stage, "--output-root", str(job_root)]
        for field in FIELDS[stage]:
            value = supplied.get(field, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Missing input: " + field)
            path = self.resolve_input(value)
            is_directory = field in ("dicom_dir", "reference_run", "first_run", "second_run")
            if path.is_dir() != is_directory:
                raise ValueError("Wrong file/directory type: " + field)
            args += ["--" + field.replace("_", "-"), str(path)]
        if stage == "embed" and supplied.get("connectome_run", "").strip():
            lineage = self.resolve_input(supplied["connectome_run"])
            if not lineage.is_dir():
                raise ValueError("Connectome run must be a directory")
            args += ["--connectome-run", str(lineage)]
        if stage == "connectome":
            config_path = self.resolve_input(supplied["config"])
            config = read_json(config_path)
            for name, value in config.items():
                if not isinstance(value, str):
                    raise ValueError("Prepared config paths must be strings")
                self.resolve_input(config_path.parent / value)
            threads = body.get("threads")
            if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 128:
                raise ValueError("Select an explicit thread count between 1 and 128")
            args += ["--threads", str(threads)]
        if stage in ("convert", "connectome") and body.get("execute"):
            if body.get("approve_compute") is not True:
                raise ValueError("Explicit confirmation is required to execute external image-processing tools")
            args.append("--execute")
        if stage == "approve":
            checks = body.get("checks", [])
            reviewer = body.get("reviewer", "")
            if set(checks) != set(REVIEW_ITEMS) or not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError("Reviewer and all five visual checks are required")
            args += ["--reviewer", reviewer, "--checked", *checks]
        return args

    def submit(self, body):
        with self.lock:
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 10:
                raise ValueError("Queue is full")
            identifier = uuid.uuid4().hex
            job_root = self.root / identifier
            arguments = self.arguments(body, job_root)
            job_root.mkdir(mode=0o700)
            job = {"id": identifier, "stage": body["stage"], "status": "queued", "logs": [],
                   "created": time.time(), "started": None, "ended": None, "process": None,
                   "output_root": str(job_root), "report": None, "current_step": "Waiting for worker"}
            self.jobs[identifier] = job
            write_json(job_root / "request.json", {"arguments": arguments, "utc": datetime.now(timezone.utc).isoformat()})
            self.pool.submit(self._run, identifier, arguments)
            return identifier

    def _run(self, identifier, arguments):
        job = self.jobs[identifier]
        with self.lock:
            if job["status"] == "cancelled":
                write_json(Path(job["output_root"]) / "job_result.json", self.snapshot(job))
                return
            job.update(status="running", started=time.time(), current_step="Validating inputs / executing selected stage")
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            # One GUI job at a time; MRtrix's requested -nthreads governs its heavy work.
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            with self.lock:
                if job["status"] == "cancelled":
                    return
                process = subprocess.Popen([sys.executable, "-m", "dbs_connectome.cli", *arguments],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           env=env, start_new_session=True)
                job["process"] = process
            with process.stdout, (Path(job["output_root"]) / "job.log").open("x", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        job["logs"] = (job["logs"] + [line.rstrip()])[-100:]
                        if line.startswith("[step "):
                            job["current_step"] = line.strip()
            code = process.wait()
            with self.lock:
                if job["status"] != "cancelled":
                    job["status"] = "completed" if code == 0 else "failed"
                if job["status"] == "completed":
                    provs = list(Path(job["output_root"]).glob("*/provenance.json"))
                    if provs and read_json(provs[0])["status"] == "planned_not_executed":
                        job["status"] = "planned"
                    reports = list(Path(job["output_root"]).glob("*/mask_review.html"))
                    if reports:
                        job["report"] = str(reports[0])
                job["returncode"] = code
        except Exception as exc:
            with self.lock:
                job["status"] = "failed"
                job["logs"].append(f"{type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                job["ended"] = time.time()
                job["process"] = None
                job["current_step"] = job["status"]
                write_json(Path(job["output_root"]) / "job_result.json", self.snapshot(job))

    def snapshot(self, job):
        public = {k: v for k, v in job.items() if k != "process"}
        public["elapsed_seconds"] = round((job["ended"] or time.time()) - (job["started"] or job["created"]), 1)
        return public

    def state(self):
        with self.lock:
            return {"jobs": [self.snapshot(j) for j in reversed(list(self.jobs.values()))],
                    "data_root": str(self.data_root), "output_root": str(self.root), "version": "0.1.0.dev0"}

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None or job["status"] not in ("queued", "running"):
                raise ValueError("Job cannot be cancelled")
            job["status"] = "cancelled"
            job["ended"] = time.time()
            job["current_step"] = "cancelled"
            if job["process"] is not None:
                try:
                    os.killpg(job["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            write_json(Path(job["output_root"]) / "job_result.json", self.snapshot(job))


def make_handler(manager, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            # Never print the token-bearing URL or patient paths to the console.
            pass

        def authorized(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host:
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + expected_host:
                return False
            query_token = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            header_token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            return hmac.compare_digest(token, header_token or query_token)

        def send(self, status, data, content_type="application/json"):
            if isinstance(data, dict):
                data = json.dumps(data, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.authorized():
                return self.send(403, {"error": "Local access token required"})
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self.send(200, (Path(__file__).parent / "static/index.html").read_bytes(), "text/html")
                if path == "/api/state":
                    return self.send(200, manager.state())
                if path == "/api/browse":
                    relative = parse_qs(urlparse(self.path).query).get("path", [str(manager.data_root)])[0]
                    directory = manager.resolve_input(relative)
                    if not directory.is_dir():
                        raise ValueError("Choose a directory")
                    entries = []
                    for p in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                        if p.name.startswith("."):
                            continue
                        try:
                            safe = manager.resolve_input(p)
                        except (ValueError, FileNotFoundError):
                            continue
                        entries.append({"name": p.name, "path": str(safe), "directory": safe.is_dir()})
                    return self.send(200, {"path": str(directory), "entries": entries[:1000]})
                if path == "/report":
                    identifier = parse_qs(urlparse(self.path).query).get("job", [""])[0]
                    job = manager.jobs.get(identifier)
                    if not job or not job.get("report"):
                        raise ValueError("No QC report is available for this job")
                    return self.send(200, Path(job["report"]).read_bytes(), "text/html")
                return self.send(404, {"error": "Not found"})
            except (ValueError, OSError) as exc:
                return self.send(400, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {"error": "Local access token required"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected a small JSON request")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if urlparse(self.path).path == "/api/jobs":
                    return self.send(202, {"id": manager.submit(body)})
                if urlparse(self.path).path == "/api/cancel":
                    manager.cancel(body.get("id"))
                    return self.send(200, {"cancelled": True})
                return self.send(404, {"error": "Not found"})
            except (ValueError, OSError) as exc:
                return self.send(400, {"error": str(exc)})
    return Handler


def serve(data_root, output_root, port=8765):
    if not 0 <= port <= 65535:
        raise ValueError("Invalid port")
    manager = JobManager(data_root, output_root)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(manager, token))
    print(f"Open locally: http://127.0.0.1:{server.server_port}/?token={token}", flush=True)
    print("Research prototype. One job at a time. Ctrl-C stops the server and cancels active work.", flush=True)
    try:
        server.serve_forever()
    finally:
        for identifier, job in list(manager.jobs.items()):
            if job["status"] in ("queued", "running"):
                manager.cancel(identifier)
        manager.pool.shutdown(wait=False, cancel_futures=True)
        server.server_close()
        server.server_close()
