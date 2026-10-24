"""Portable, manifest-driven replay from reviewed connectome matrices.

No patient names, paths, data or result constants are embedded in this module.
Passing this replay does not validate the upstream DICOM reconstruction.
"""
import csv
import json
from pathlib import Path
import re

import numpy as np

from .gradients import Settings, embed, load_nodes, validate_matrix
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, read_json,
                         record, require_separate_output, write_json)
from .study_metrics import exact_family, identification, interval_metrics, session_metrics


def manifest_paths(spec):
    """Enumerate every user-supplied file for GUI path confinement and hashing."""
    paths = [spec["nodes"]]
    for subject in spec["subjects"]:
        for visit in subject["visits"]:
            paths.append(visit["matrix"])
            if visit.get("exclusion_evidence"):
                paths.append(visit["exclusion_evidence"])
    for check in spec.get("expected", []) + spec.get("graph_expected", []):
        paths.append(check["path"])
    if spec.get("preoperative_graph"):
        pre = spec["preoperative_graph"]
        paths.append(pre["nodes"])
        paths.extend(item["matrix"] for item in pre["subjects"])
    if spec.get("sensitivity_reference"):
        paths.append(spec["sensitivity_reference"]["path"])
    for case in spec.get("control_sessions", {}).values():
        for item in case["subjects"]:
            paths.append(item["matrix"])
            if item.get("exclusion_evidence"):
                paths.append(item["exclusion_evidence"])
    if spec.get("healthy_reference"):
        paths.append(spec["healthy_reference"]["path"])
    if spec.get("geometry_controls"):
        geometry = spec['geometry_controls']
        for subject in geometry['subjects']:
            paths.extend(subject[k]['path'] for k in ('atlas', 'distance_mask', 'overlap_mask'))
            if subject.get('unmasked_preoperative'):
                paths.append(subject['unmasked_preoperative']['matrix'])
        for pair in geometry.get('registered_pairs', []):
            paths.append(pair['mask']['path'])
        for arm in geometry.get('swaps', {}).values():
            for subject in arm['subjects']:
                paths.extend([subject['matrix'], subject['exclusion_evidence']])
    if spec.get('comparative'):
        cfg = spec['comparative']
        links = list(cfg.get('calibres', {}).values())
        if cfg.get('external'): links.append(cfg['external'])
        if cfg.get('transfer_reference'): links.append(cfg['transfer_reference'])
        for item in links:
            linked = read_json(item['manifest'])
            if linked.get('comparative'):
                raise ValueError('Nested comparative studies are forbidden')
            paths.extend([item['manifest'], *manifest_paths(linked)])
    return paths


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty results table")
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def load_manifest(path, *, allow_session_masks=False, allow_unmasked=False):
    spec = read_json(path)
    if spec.get("schema_version") != 1 or spec.get("pairing_reviewed") is not True:
        raise ValueError("Reviewed subject/visit/mask pairing and schema_version=1 are required")
    nodes = load_nodes(spec["nodes"])
    if digest(spec["nodes"]) != spec.get("nodes_sha256"):
        raise ValueError("Node identities changed since manifest review")
    subjects = spec.get("subjects", [])
    if not 3 <= len(subjects) <= 8:
        raise ValueError("Exact clinical replay currently supports 3..8 subjects per cohort")
    ids = [s["id"] for s in subjects]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", s) for s in ids):
        raise ValueError("Use unique coded subject identifiers")
    matrices, seen = [], set()
    for subject in subjects:
        visits = subject["visits"]
        if [v.get("timepoint") for v in visits] != ["pre", "post1", "post2"]:
            raise ValueError("Exactly pre/post1/post2 in chronological order are required")
        current = []
        for v in visits:
            actual = digest(v["matrix"])
            if actual != v.get("sha256") or actual in seen:
                raise ValueError("Changed or duplicated session matrix; review input identity")
            seen.add(actual)
            allowed = ("virtual", "shared_postoperative", "none") + (("session_postoperative",) if allow_session_masks else ())
            if v.get("mask_scheme") not in allowed:
                raise ValueError("State the matrix mask scheme")
            postoperative = ("shared_postoperative", "session_postoperative") if allow_session_masks else ("shared_postoperative",)
            if allow_unmasked:
                postoperative += ("none",)
            if v["timepoint"] != "pre" and v["mask_scheme"] not in postoperative:
                raise ValueError("Primary postoperative replay requires the shared-mask matrix variant")
            if v["mask_scheme"] != "none":
                if not v.get("exclusion_evidence") or digest(v["exclusion_evidence"]) != v.get("exclusion_evidence_sha256"):
                    raise ValueError("Masked matrices require unchanged upstream exclusion evidence")
            p = Path(v["matrix"])
            m = np.load(p, allow_pickle=False) if p.suffix == ".npy" else np.loadtxt(p, delimiter=",")
            m = validate_matrix(m)
            indices = v.get("matrix_indices")
            if indices is not None:
                ix = np.asarray(indices)
                if ix.dtype.kind not in "iu" or ix.ndim != 1 or len(ix) != len(nodes) or len(set(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(m)):
                    raise ValueError("Explicit matrix row mapping is invalid")
                m = m[np.ix_(ix, ix)]
            if m.shape != (len(nodes), len(nodes)):
                raise ValueError("Matrix shape and frozen node identities differ")
            current.append(m)
        matrices.append(current)
    endpoints = spec.get("clinical", {})
    if not endpoints:
        raise ValueError("Supply explicitly named clinical endpoints and their definitions")
    for name, endpoint in endpoints.items():
        if not name or not endpoint.get("definition") or set(endpoint["values"]) != set(ids):
            raise ValueError("Clinical endpoint requires definition and exactly matching subject IDs")
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)) for v in endpoint["values"].values()):
            raise ValueError("Clinical values must be finite numbers or explicit null; never impute")
    return spec, nodes, np.asarray(matrices)


def cohort_embeddings(matrices, nodes, reference_mode, legacy_zero_profiles=False, settings=None, retain_all=False):
    n, visits = matrices.shape[:2]
    indices = {h: np.array([i for i, r in enumerate(nodes) if r["tissue"] == "cortex" and r["hemisphere"] == h]) for h in ("L", "R")}
    order = np.concatenate(list(indices.values()))
    settings = settings or Settings()
    keep = settings.n_components if retain_all else settings.n_retained
    gradients = np.empty((n, visits, len(order), keep))
    eigenvalues = np.empty((n, visits, 2, settings.n_components))
    references, memberships = {}, {}
    common = matrices[:, 0].mean(0)
    for s in range(n):
        membership = [i for i in range(n) if reference_mode == "common" or i != s]
        mean = common if reference_mode == "common" else matrices[membership, 0].mean(0)
        key = "common" if reference_mode == "common" else str(s)
        if key not in references:
            references[key] = {h: embed(mean, ix, settings, legacy_zero_profiles=legacy_zero_profiles)[0] for h, ix in indices.items()}
            memberships[key] = membership
        for t in range(visits):
            start = 0
            for hi, (h, ix) in enumerate(indices.items()):
                g, lam = embed(matrices[s, t], ix, settings, references[key][h], legacy_zero_profiles=legacy_zero_profiles)
                gradients[s, t, start:start+len(ix)] = g[:, :keep]
                eigenvalues[s, t, hi] = lam
                start += len(ix)
        print(f"[step {s+1}/{n}] {reference_mode} reference embedding", flush=True)
    return gradients, eigenvalues, order, memberships


def isolated_profiles(matrices,nodes,subjects,visits):
    rows=[]
    for si,subject in enumerate(subjects):
        for ti,visit in enumerate(visits):
            for hemi in ('L','R'):
                ix=np.array([i for i,r in enumerate(nodes) if r['tissue']=='cortex' and r['hemisphere']==hemi])
                zero=ix[matrices[si,ti][np.ix_(ix,ix)].sum(1)==0]
                if len(zero):rows.append({'subject':subject,'visit':visit,'hemisphere':hemi,'node_indices':zero.tolist()})
    return rows


def expected_checks(spec, arrays, rows, correlations=()):
    """Independent frozen outputs are read only; never substitute them for recomputation."""
    results = []
    for check in spec.get("expected", []):
        if digest(check["path"]) != check["sha256"]:
            raise ValueError("Frozen expected output changed")
        atol, rtol = check.get("atol", 1e-10), check.get("rtol", 1e-8)
        if not 0 <= atol <= 1e-8 or not 0 <= rtol <= 1e-6:
            raise ValueError("Replay tolerances exceed permitted numerical comparison limits")
        if check["kind"] == "array":
            actual = arrays[check["actual"]]
            with np.load(check["path"], allow_pickle=False) as z:
                expected = z[check["key"]]
            same = actual.shape == expected.shape
            ok = same and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
            error = float(np.max(np.abs(actual-expected))) if same else None
            count = int(actual.size)
        elif check["kind"] == "numbers_json":
            if not check.get('paths'):
                raise ValueError('Frozen numerical check must compare at least one explicit value')
            expected = read_json(check["path"])
            actual = arrays[check["actual"]]
            differences = []
            ok = True
            for keys in check["paths"]:
                a,b = actual,expected
                actual_keys=keys['actual'] if isinstance(keys,dict) else keys
                expected_keys=keys['expected'] if isinstance(keys,dict) else keys
                for key in actual_keys:a=a[key]
                for key in expected_keys:b=b[key]
                a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
                same = a.shape == b.shape
                ok = ok and same and bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False))
                if same:
                    differences.extend(np.abs(a-b).ravel().tolist())
            error = max(differences,default=None)
            count = len(differences)
        elif check["kind"] in ("metrics_csv", "correlations_csv", "table_csv"):
            with Path(check["path"]).open(newline="", encoding="utf-8") as f:
                expected = list(csv.DictReader(f))
            is_correlation = check["kind"] == "correlations_csv"
            actual_rows = arrays[check["actual"]] if check["kind"] == "table_csv" else (correlations if is_correlation else rows)
            keys = check.get("keys") or (("network", "metric", "outcome") if is_correlation else ("subject", "network", "metric"))
            column_map = check.get("column_map", {})
            key_map = check.get("key_map", {})
            keyed = {tuple(r[key_map.get(k,k)] for k in keys): r for r in expected}
            if len(keyed) != len(expected):
                raise ValueError("Duplicate expected metric rows")
            differences = []
            ok = len(actual_rows) == len(expected)
            for row in actual_rows:
                old = keyed.get(tuple(row[k] for k in keys))
                if old is None:
                    ok = False
                    continue
                for column in check.get("columns", ["delta"]):
                    a = row[column]
                    raw = old[column_map.get(column,column)]
                    b = float(raw) if raw.strip() else float("nan")
                    if a is None and not np.isfinite(b):
                        continue
                    if a is None or not np.isfinite(b):
                        ok = False
                        continue
                    differences.append(abs(a-b))
                    ok = ok and bool(np.isclose(a, b, atol=atol, rtol=rtol))
            error = max(differences, default=None)
            count = len(differences)
        else:
            raise ValueError("Unknown expected-output check")
        if count == 0:
            raise ValueError('Frozen comparison contained no comparable finite values')
        results.append({"check": check["actual"], "passed": bool(ok), "compared_values": count,
                        "max_absolute_error": error, "atol": atol, "rtol": rtol})
    return results


def run_study(manifest, out):
    out = Path(out)
    initial = read_json(manifest)
    paths = [manifest, *manifest_paths(initial)]
    for path in paths:
        require_separate_output(out, Path(path).parent)
    frozen = freeze_inputs(paths)
    spec, nodes, matrices = load_manifest(manifest)
    subjects = [s["id"] for s in spec["subjects"]]
    legacy = spec.get("legacy_zero_profile_policy", "reject")
    if legacy not in ("reject", "brainspace_normalized_angle"):
        raise ValueError("Unknown zero-profile policy")
    if legacy != "reject" and not spec.get("expected"):
        raise ValueError("Zero-profile compatibility is only permitted for explicit frozen-output replay")
    isolated = []
    for si, subject in enumerate(subjects):
        for ti, visit in enumerate(("pre", "post1", "post2")):
            for hemi in ("L", "R"):
                ix = np.array([i for i, r in enumerate(nodes) if r["tissue"] == "cortex" and r["hemisphere"] == hemi])
                zero = ix[matrices[si, ti][np.ix_(ix, ix)].sum(1) == 0]
                if len(zero):isolated.append({"subject": subject, "visit": visit, "hemisphere": hemi, "node_indices": zero.tolist()})
    g, lam, order, membership = cohort_embeddings(matrices, nodes, "loso", legacy != "reject")
    common, common_lam, _, common_membership = cohort_embeddings(matrices, nodes, "common", legacy != "reject")
    np.savez_compressed(out / "study_gradients.npz", loso=g, common=common, lambdas=lam,
                        common_lambdas=common_lam, cortical_indices=order, subjects=np.array(subjects))
    nets = np.array([nodes[i]["network"] for i in order])
    hemis = np.array([nodes[i]["hemisphere"] for i in order])
    rows = []
    for i, subject in enumerate(subjects):
        a, b = session_metrics(g[i, 1], nets, hemis), session_metrics(g[i, 2], nets, hemis)
        between = interval_metrics(g[i, 1], g[i, 2], nets, hemis)
        for (net, metric), value in sorted(a.items()):
            rows.append({"subject": subject, "network": net, "metric": metric,
                         "post1": value, "post2": b[net, metric], "delta": b[net, metric]-value})
        for (net, metric), value in sorted(between.items()):
            rows.append({"subject": subject, "network": net, "metric": metric,
                         "post1": None, "post2": None, "delta": value})
    write_csv(out / "network_metrics.csv", rows)
    features = sorted({(r["network"], r["metric"]) for r in rows})
    lookup = {(r["subject"], r["network"], r["metric"]): r["delta"] for r in rows}
    x = np.array([[lookup[s, *key] for key in features] for s in subjects])
    correlations, exclusions, plot_values = [], [], []
    for endpoint, clinical in spec["clinical"].items():
        valid = np.array([clinical["values"][s] is not None for s in subjects])
        y = np.array([clinical["values"][s] for s in subjects if clinical["values"][s] is not None], float)
        if len(y) < 3 or np.ptp(y) == 0:
            exclusions.append({"outcome": endpoint, "reason": "fewer than three observations or constant outcome"})
            continue
        keep = np.ptp(x[valid], axis=0) > 0
        if not keep.any():
            exclusions.append({"outcome": endpoint, "reason": "all imaging measures constant"})
            continue
        stats = exact_family(x[valid][:, keep], y)
        selected = [s for s, v in zip(subjects, valid) if v]
        for j, (net, metric) in enumerate(np.array(features)[keep]):
            correlations.append({"network": net, "metric": metric, "outcome": endpoint,
                                 "n": len(y), "family_size": int(keep.sum()),
                                 **{k: float(stats[k][j]) for k in ("rho", "p_exact", "p_greater", "p_less", "q_bh", "p_maxT")},
                                 "loo_min": float(np.nanmin(stats["loo_rho"][:, j])),
                                 "loo_max": float(np.nanmax(stats["loo_rho"][:, j])),
                                 "subjects": ";".join(selected)})
            for subject, xv, yv in zip(selected, x[valid][:, keep][:, j], y):
                plot_values.append({"subject": subject, "network": net, "metric": metric,
                                    "outcome": endpoint, "x": float(xv), "y": float(yv)})
        for key in np.array(features)[~keep]:
            exclusions.append({"outcome": endpoint, "reason": "constant imaging variable: " + "/".join(key)})
    if correlations:
        write_csv(out / "clinical_correlations.csv", correlations)
    identification_result, sim = identification(common.reshape(len(subjects)*3, -1), len(subjects),
                                                spec.get("identification_samples", 100000), spec.get("seed", 42))
    np.save(out / "identification_similarity.npy", sim)
    checks = expected_checks(spec, {"loso": g, "common": common, "lambdas": lam}, rows, correlations)
    summary = {"scope": "matrix_to_primary_study_analysis_only", "subjects": subjects,
               "n_nodes": len(nodes), "n_cortical": len(order), "reference_membership": membership,
               "zero_profile_policy": legacy, "isolated_cortical_profiles": isolated,
               "common_reference_membership": common_membership,
               "identification": identification_result, "expected_checks": checks,
               "replay_status": "passed" if checks and all(c["passed"] for c in checks) else "failed" if checks else "not_compared",
               "clinical_exclusions": exclusions, "clinical_definitions": {k: v["definition"] for k, v in spec["clinical"].items()},
               "plot_values": plot_values,
               "upstream_validation": "not_reexecuted; supplied historical exclusion evidence is recorded, not independently proved",
               "warning": "This is not whole-paper or DICOM-to-result validation. Clinical screening is exploratory. G3/G4 identity is not established by alignment."}
    write_json(out / "study_result.json", clean(summary))
    from .study_report import report
    report(out / "study_review.html", summary, rows, correlations)
    assert_inputs_unchanged(frozen)
    outputs = sorted(p for p in out.iterdir() if p.is_file())
    record(out, "study", paths, {"schema": 1, "settings": vars(Settings()), "n_subjects": len(subjects)}, outputs,
           "completed" if summary["replay_status"] != "failed" else "numerical_validation_failed")
    if summary["replay_status"] == "failed":
        raise ValueError("Frozen-output replay differs; see study_result.json. Do not accept or publish validation claims.")
