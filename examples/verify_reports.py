"""Create wholly synthetic connectome/reference/paired reports for GUI inspection.

These numerical fixtures are not MRI data or evidence of electrode exclusion.
Synthetic provenance is deliberately labelled and is only for testing software contracts.
"""
import argparse
from pathlib import Path

import numpy as np

from dbs_connectome.longitudinal import cohort_reference, paired_change
from dbs_connectome.provenance import new_run, digest, record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = new_run(args.output_root, "synthetic_reports")
    nodes = root / "nodes.tsv"
    text = "index\tlabel\tname\themisphere\ttissue\tnetwork\n"
    for i in range(32):
        text += f"{i}\t{i+1}\tSynthetic_parcel_{i}\t{'L' if i < 16 else 'R'}\tcortex\tSynthetic_{i%3}\n"
    with nodes.open("x") as stream:
        stream.write(text)
    rng = np.random.default_rng(54)
    runs = []
    for i in range(4):
        out = root / f"synthetic_connectome_{i}"; out.mkdir()
        m = rng.uniform(.1, 4, (32,32)); m = (m+m.T)/2; np.fill_diagonal(m, 0)
        np.save(out/"connectome.npy", m)
        record(out, "connectome", [nodes], {"nodes_sha256": digest(nodes), "synthetic_test_only": True}, [out/"connectome.npy"])
        runs.append(str(out))
    cohort = root / "cohort"; cohort.mkdir()
    cohort_reference(runs[:2], nodes, cohort, "none", True)
    paired = root / "paired"; paired.mkdir()
    paired_change(runs[2], runs[3], cohort/"reference", nodes, paired, True)
    write_json(root/"gui_paths.json", {"synthetic_only": True, "nodes": str(nodes), "reference_run": str(cohort/"reference"),
               "first_connectome_run": runs[2], "second_connectome_run": runs[3], "first_embed": str(paired/"first")})
    print(root)


if __name__ == "__main__":
    main()
