"""Local, inspectable G1-G4 report and numerical export; no symptom-axis labels."""
import base64
import csv
import json
from pathlib import Path

import numpy as np

from .gradients import load_nodes, validate_matrix
from .provenance import digest, read_json, record, require_separate_output, write_json


def gradient_report(gradient_run, nodes_path, out, connectome_run=None):
    from .cli import checked_artifact
    out = Path(out).resolve()
    require_separate_output(out, gradient_run)
    stage = read_json(Path(gradient_run) / "provenance.json")["stage"]
    if stage not in ("reference", "embed"):
        raise ValueError("Choose a completed reference or aligned embedding run")
    gradients, provenance = checked_artifact(gradient_run, "gradients.npz", stage)
    nodes = load_nodes(nodes_path)
    if provenance["parameters"]["nodes_sha256"] != digest(nodes_path):
        raise ValueError("Report node table differs from embedding node identities")
    matrix = None
    inputs = [gradients, nodes_path]
    if connectome_run:
        matrix_path, mp = checked_artifact(connectome_run, "connectome.npy", "connectome")
        if mp["parameters"]["nodes_sha256"] != digest(nodes_path):
            raise ValueError("Connectome and embedding node identities differ")
        if digest(matrix_path) not in provenance["inputs"].values():
            raise ValueError("This matrix was not used to compute the selected embedding")
        matrix = validate_matrix(np.load(matrix_path, allow_pickle=False))
        if len(matrix) != len(nodes) or len(matrix) > 2000:
            raise ValueError("Matrix/node size mismatch or interactive size limit exceeded")
        inputs.append(matrix_path)
        synthetic = mp["parameters"].get("synthetic_test_only", False)
    else:
        synthetic = False
    rows = []
    spectra = {}
    with np.load(gradients, allow_pickle=False) as saved:
        for hemi in ("L", "R"):
            g = saved[f"{hemi}_coordinates"]
            ids = saved[f"{hemi}_indices"]
            spectra[hemi] = saved[f"{hemi}_eigenvalues"].tolist()
            for j, i in enumerate(ids):
                if not 0 <= i < len(nodes) or not np.isfinite(g[j, :4]).all():
                    raise ValueError("Invalid gradient coordinates or node indices")
                rows.append({"index": int(i), "name": nodes[i]["name"], "network": nodes[i]["network"],
                             "hemisphere": hemi, **{f"G{k+1}": float(g[j, k]) for k in range(4)}})
    data = {"rows": rows, "spectra": spectra, "nodes": nodes,
            "alignment": "REFERENCE_ALIGNED" if stage == "embed" else "UNALIGNED_REFERENCE",
            "matrix_mask_lineage": "SYNTHETIC_NUMERICAL_FIXTURE_NO_MRI" if synthetic else
                ("verified_masked_connectome" if connectome_run else provenance["parameters"].get("matrix_mask_lineage", "unverified_matrix_import")),
            "matrix": None if matrix is None else base64.b64encode(matrix.astype("<f4").tobytes()).decode(),
            "n": len(nodes), "clinical_inference": False}
    csv_path = out / "gradient_coordinates.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "name", "network", "hemisphere", "G1", "G2", "G3", "G4"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "report_data.json", data)
    # Escape '<' so node names cannot terminate the JSON script element.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = (Path(__file__).parent / "static/results.html").read_text().replace("__DATA__", payload)
    path = out / "results_review.html"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(html)
    record(out, "results", inputs, {"clinical_inference": False}, [path, csv_path, out / "report_data.json"])
    return path
