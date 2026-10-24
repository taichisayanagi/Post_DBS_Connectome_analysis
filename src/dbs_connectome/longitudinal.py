"""Explicit training-reference construction and paired aligned-visit analysis."""
from argparse import Namespace
import csv
import json
from pathlib import Path

import numpy as np

from .gradients import load_nodes, validate_matrix
from .provenance import digest, read_json, record, require_separate_output, write_json


def cohort_reference(runs, nodes, out, normalization, training_reviewed):
    from .cli import checked_artifact, run
    out = Path(out).resolve()
    if training_reviewed is not True:
        raise ValueError("Review training/independent reference membership and possible data leakage")
    if not isinstance(runs, list) or not 2 <= len(runs) <= 500:
        raise ValueError("Choose 2..500 completed connectome runs")
    runs = [str(Path(p).resolve(strict=True)) for p in runs]
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicated reference run")
    if normalization != "none":
        raise ValueError("This release uses raw-weight arithmetic means only; scaling targets/reference together is not implemented")
    nodes_hash = digest(nodes)
    n = len(load_nodes(nodes))
    mean = np.zeros((n, n), dtype=float)
    inputs, hashes, scales = [nodes], [], []
    for path in runs:
        require_separate_output(out, path)
        artifact, prov = checked_artifact(path, "connectome.npy", "connectome")
        if prov["parameters"]["nodes_sha256"] != nodes_hash:
            raise ValueError("Cohort references require identical frozen node identities; no implicit intersection/padding")
        m = validate_matrix(np.load(artifact, allow_pickle=False))
        if m.shape != (n, n) or m.sum() <= 0:
            raise ValueError("Empty or incompatible reference connectome")
        sha = digest(artifact)
        if sha in hashes:
            raise ValueError("Duplicate connectome bytes in reference membership; review scan identities")
        hashes.append(sha)
        scale = 1.
        scales.append(scale)
        mean += m / scale / len(runs)
        inputs.append(artifact)
    matrix = out / "mean_connectome.npy"
    np.save(matrix, mean)
    target = out / "reference"; target.mkdir()
    run(Namespace(command="reference", matrix=str(matrix), nodes=nodes), target)
    result = {"reference_run": str(target), "members": runs, "normalization": normalization,
              "scales": scales, "member_hashes": hashes, "training_reviewed": True,
              "warning": "Membership is operator-defined; automatic LOSO/leakage detection is not performed. "
              "Use the same matrix normalization for all embeddings compared with this reference."}
    write_json(out / "cohort_result.json", result)
    write_json(target / "cohort_membership.json", result)
    record(out, "cohort-reference", inputs, {"normalization": normalization},
           [matrix, out / "cohort_result.json", target / "provenance.json", target / "cohort_membership.json"])


def paired_change(first, second, reference, nodes, out, pairing_reviewed):
    from .cli import checked_artifact, run
    from .reports import gradient_report
    out = Path(out).resolve()
    if pairing_reviewed is not True:
        raise ValueError("Confirm same patient, ordered visits, same exclusion strategy and clinical timepoints")
    for path in (first, second, reference):
        require_separate_output(out, path)
    reference_artifact, _ = checked_artifact(reference, "gradients.npz", "reference")
    membership = Path(reference) / "cohort_membership.json"
    if membership.exists() and read_json(membership).get("normalization") != "none":
        raise ValueError("This paired raw-connectome workflow requires an unscaled reference")
    a, _ = checked_artifact(first, "connectome.npy", "connectome")
    b, _ = checked_artifact(second, "connectome.npy", "connectome")
    if digest(a) == digest(b):
        raise ValueError("Identical connectome bytes at both visits; review input pairing")
    for name, matrix, conn in (("first", a, first), ("second", b, second)):
        target = out / name; target.mkdir()
        run(Namespace(command="embed", matrix=str(matrix), nodes=nodes, reference_run=reference,
                      connectome_run=conn), target)
        report = out / (name + "_report"); report.mkdir()
        gradient_report(target, nodes, report, conn)
    target = out / "change"; target.mkdir()
    run(Namespace(command="change", first_run=str(out / "first"), second_run=str(out / "second"), nodes=nodes), target)
    rows = read_json(target / "parcel_change.json")
    node_rows = load_nodes(nodes)
    csv_path = out / "paired_gradient_change.csv"
    fields = ["node_index", "name", "hemisphere", "network", "displacement_4d", "eccentricity_reference_center_change", "delta_G1", "delta_G2", "delta_G3", "delta_G4"]
    report_rows = []
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            values = {k: v for k, v in row.items() if k != "axis_change_G1_G4"}
            values["name"] = node_rows[row["node_index"]]["name"]
            values.update({f"delta_G{i+1}": v for i, v in enumerate(row["axis_change_G1_G4"])})
            writer.writerow(values)
            report_rows.append(values)
    report_path = out / "longitudinal_review.html"
    payload = json.dumps({"rows": report_rows, "fields": fields}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = (Path(__file__).parent / "static/longitudinal.html").read_text().replace("__DATA__", payload)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(html)
    result = {"first_embed": str(out / "first"), "second_embed": str(out / "second"), "change_run": str(target),
              "csv": str(csv_path), "report": str(report_path), "pairing_reviewed": True, "clinical_inference": False,
              "warning": "Signed high-order axis changes require independent axis-stability assessment. "
              "Matching alignment does not prove test-retest reliability or remove residual electrode distortion."}
    write_json(out / "longitudinal_result.json", result)
    record(out, "longitudinal", [a, b, reference_artifact, nodes], {"pairing_reviewed": True},
           [csv_path, report_path, out / "longitudinal_result.json", target / "provenance.json",
            out / "first_report/provenance.json", out / "second_report/provenance.json"])
