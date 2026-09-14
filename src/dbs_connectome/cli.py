"""Small, explicit stages; no automatic publication or clinical decision output."""

import argparse
from pathlib import Path
import sys

import numpy as np

from .external import audit_mask_exclusion, connectome_plan, conversion_plan, execute, validate_mif_grid, validate_tractogram_sampling
from .gradients import compact_raw, displacement, geometry, hemisphere_embeddings, load_nodes, settings_dict
from .masks import REVIEW_ITEMS, approval_record, create_candidate, union_registered
from .provenance import assert_inputs_unchanged, digest, freeze_inputs, new_run, read_json, record, require_separate_output, voxel_fingerprint, write_json
from .qc import browser_report


def parser():
    p = argparse.ArgumentParser(description="DBS Native Connectome: unvalidated research prototype")
    p.add_argument("--version", action="version", version="0.1.0.dev0")
    sub = p.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Start the loopback-only browser application")
    gui.add_argument("--data-root", required=True)
    gui.add_argument("--output-root", required=True)
    gui.add_argument("--port", type=int, default=8765)
    for name in ("environment", "inventory", "import-session", "import-nifti", "reconstruct", "prepare-session", "finish-session", "convert", "mask", "union", "approve", "connectome", "reference", "embed", "change", "fingerprint", "qc"):
        s = sub.add_parser(name)
        s.add_argument("--output-root", required=True, help="Private result root outside source checkout")
        if name == "environment":
            pass
        elif name == "inventory":
            s.add_argument("--dicom-dir", required=True)
        elif name == "import-session":
            s.add_argument("--inventory-run", required=True)
            s.add_argument("--selection", required=True)
            s.add_argument("--execute", action="store_true")
        elif name == "import-nifti":
            s.add_argument("--selection", required=True)
        elif name == "prepare-session":
            s.add_argument("--reconstruction-run", required=True)
            s.add_argument("--settings", required=True)
            s.add_argument("--threads", type=int, required=True)
            s.add_argument("--execute", action="store_true")
        elif name == "finish-session":
            s.add_argument("--prepared-run", required=True)
            s.add_argument("--approval", required=True)
            s.add_argument("--threads", type=int, required=True)
            s.add_argument("--streamlines", type=int, default=1000000)
            s.add_argument("--reference-run")
            s.add_argument("--execute", action="store_true")
        elif name == "reconstruct":
            s.add_argument("--import-run", required=True)
            s.add_argument("--choices", required=True)
            s.add_argument("--threads", type=int, required=True)
            s.add_argument("--execute", action="store_true")
        elif name == "qc":
            s.add_argument("--reference", required=True)
            s.add_argument("--mask", required=True)
        elif name == "convert":
            s.add_argument("--dicom-dir", required=True)
            s.add_argument("--execute", action="store_true", help="Default only writes a command plan")
        elif name == "mask":
            s.add_argument("--reference", required=True)
            s.add_argument("--centerlines", required=True)
            s.add_argument("--radius-mm", type=float, default=3.5)
            s.add_argument("--dilation-passes", type=int, default=1)
        elif name == "union":
            s.add_argument("--reference", required=True)
            s.add_argument("--masks", nargs="+", required=True)
            s.add_argument("--registration-record", required=True)
        elif name == "approve":
            for k in ("mask", "reference", "atlas", "reviewer"):
                s.add_argument("--" + k, required=True)
            s.add_argument("--checked", nargs="+", choices=REVIEW_ITEMS, required=True)
        elif name == "connectome":
            s.add_argument("--config", required=True)
            s.add_argument("--threads", required=True, type=int)
            s.add_argument("--execute", action="store_true")
        elif name in ("reference", "embed"):
            s.add_argument("--matrix", required=True, help="Compact .npy in exact TSV node order")
            s.add_argument("--nodes", required=True)
            if name == "embed":
                s.add_argument("--reference-run", required=True)
                s.add_argument("--connectome-run", help="Optional hash-verified masked-connectome provenance")
        elif name == "change":
            s.add_argument("--first-run", required=True)
            s.add_argument("--second-run", required=True)
            s.add_argument("--nodes", required=True)
        else:
            s.add_argument("--image", required=True)
    return p


def checked_artifact(run, filename, stage):
    run = Path(run).resolve()
    provenance = read_json(run / "provenance.json")
    path = run / filename
    if provenance["stage"] != stage or provenance["status"] != "completed":
        raise ValueError("Wrong or incomplete upstream stage")
    if provenance["outputs"].get(str(path)) != digest(path):
        raise ValueError("Upstream output changed after its provenance was recorded")
    return path, provenance


def run(args, out):
    command = args.command
    inputs, outputs, parameters, status = [], [], {}, "completed"
    if command == "inventory":
        from .intake import inventory
        inventory(args.dicom_dir, out)
        return
    elif command == "import-session":
        from .intake import convert_selection
        convert_selection(args.inventory_run, read_json(args.selection), out, args.execute)
        return
    elif command == "import-nifti":
        from .nifti import import_nifti
        import_nifti(read_json(args.selection), out)
        return
    elif command == "prepare-session":
        from .workflow import prepare_session
        prepare_session(args.reconstruction_run, read_json(args.settings), out, args.threads, args.execute)
        return
    elif command == "finish-session":
        from .workflow import finish_session
        finish_session(args.prepared_run, args.approval, out, args.threads, args.streamlines, args.reference_run, args.execute)
        return
    elif command == "reconstruct":
        from .reconstruction import reconstruct
        reconstruct(args.import_run, read_json(args.choices), out, args.threads, args.execute)
        return
    elif command == "environment":
        from .environment import discover
        path = out / "environment.json"
        write_json(path, discover())
        outputs = [path]
    elif command == "qc":
        path = out / "mask_review.html"
        qc = browser_report(args.reference, args.mask, path)
        inputs, outputs = [args.reference, args.mask], [path]
        parameters = {"qc": qc, "approved": False, "private_anatomical_data": True}
    elif command == "fingerprint":
        path = out / "fingerprint.json"
        write_json(path, voxel_fingerprint(args.image))
        inputs, outputs = [args.image], [path]
    elif command == "convert":
        commands = conversion_plan(args.dicom_dir, out)
        write_json(out / "plan.json", commands)
        # Conversion does not select anatomy/DWI, verify DWI gradients, or de-identify images.
        parameters = {"input_directory": str(Path(args.dicom_dir).resolve()), "deidentified": False}
        if args.execute:
            execute(commands, out)
        else:
            status = "planned_not_executed"
        outputs = [p for p in out.iterdir() if p.is_file()]
    elif command == "mask":
        path = create_candidate(args.reference, args.centerlines, out, args.radius_mm, args.dilation_passes)
        inputs, outputs = [args.reference, args.centerlines], [path, out / "mask_qc.json"]
        parameters = {"radius_mm": args.radius_mm, "dilation_passes_6_connected_2mm": args.dilation_passes,
                      "algorithm": "reviewed_centerline_tube_NOT_study_void_follow", "approved": False}
    elif command == "union":
        path = union_registered(args.reference, args.masks, args.registration_record, out)
        inputs, outputs = [args.reference, *args.masks, args.registration_record], [path]
        parameters = {"automated_registration": False, "approved_for_tractography": False}
    elif command == "approve":
        approval = approval_record(args.mask, args.reference, args.atlas, args.reviewer, args.checked)
        path = out / "approval.json"
        write_json(path, approval)
        inputs, outputs = [args.mask, args.reference, args.atlas], [path]
        parameters = {"human_attestation": True, "checks": list(args.checked)}
    elif command == "connectome":
        paths, commands = connectome_plan(args.config, out, args.threads)
        write_json(out / "plan.json", commands)
        parameters = {"threads": args.threads, "sift2_refitted_after_exclusion": True,
                      "assignment_radial_search_mm": 4, "matrix_row_zero": "atlas_label_1",
                      "nodes_sha256": digest(paths["nodes"])}
        inputs = [args.config, *paths.values()]
        if args.execute:
            before = freeze_inputs(inputs)
            write_json(out / "input_hashes_before.json", before)
            validate_mif_grid(paths["fod"], paths["reference"], out)
            validate_mif_grid(paths["five_tissue"], paths["reference"], out)
            sampling = validate_tractogram_sampling(paths["tractogram"], paths["reference"])
            write_json(out / "tractogram_sampling.json", sampling)
            for directory in ("exclusion_commands", "connectome_commands"):
                (out / directory).mkdir()
            execute(commands[:1], out / "exclusion_commands", args.threads)
            audit = audit_mask_exclusion(out / "tracks_excluded.tck", paths["mask"])
            write_json(out / "exclusion_audit.json", audit)
            execute(commands[1:], out / "connectome_commands", args.threads)
            nodes = load_nodes(paths["nodes"])
            raw = np.loadtxt(out / "connectome_raw.csv", delimiter=",")
            compact = compact_raw(raw, [int(n["label"]) for n in nodes])
            np.save(out / "connectome.npy", compact)
            assert_inputs_unchanged(before)
            write_json(out / "source_integrity.json", {"source_bytes_unchanged": True})
        else:
            status = "planned_not_executed"
        outputs = [p for p in out.iterdir() if p.is_file()]
    elif command in ("reference", "embed"):
        nodes = load_nodes(args.nodes)
        matrix = np.load(args.matrix, allow_pickle=False)
        inputs, parameters = [args.matrix, args.nodes], {"settings": settings_dict(), "nodes_sha256": digest(args.nodes)}
        references = None
        if command == "embed":
            ref_path, ref_prov = checked_artifact(args.reference_run, "gradients.npz", "reference")
            if ref_prov["parameters"] != parameters:
                raise ValueError("Reference settings or node identities differ")
            with np.load(ref_path, allow_pickle=False) as saved:
                references = dict(saved)
            parameters["reference_sha256"] = digest(ref_path)
            parameters["reference_run"] = str(Path(args.reference_run).resolve())
            inputs.append(ref_path)
            # Matrix-only imports are deliberately NOT certified as electrode-excluded.
            parameters["matrix_mask_lineage"] = "unverified_matrix_import"
            if args.connectome_run:
                conn_path, conn_prov = checked_artifact(args.connectome_run, "connectome.npy", "connectome")
                if digest(conn_path) != digest(args.matrix):
                    raise ValueError("Matrix differs from the completed masked-connectome run")
                if conn_prov["parameters"]["nodes_sha256"] != digest(args.nodes):
                    raise ValueError("Node identities differ from the masked-connectome run")
                parameters["matrix_mask_lineage"] = "completed_masked_connectome_stage"
                inputs.append(Path(args.connectome_run) / "provenance.json")
        result = hemisphere_embeddings(matrix, nodes, references)
        path = out / "gradients.npz"
        np.savez_compressed(path, **result)
        outputs = [path]
    elif command == "change":
        first, a = checked_artifact(args.first_run, "gradients.npz", "embed")
        second, b = checked_artifact(args.second_run, "gradients.npz", "embed")
        if a["parameters"] != b["parameters"] or a["parameters"]["nodes_sha256"] != digest(args.nodes):
            raise ValueError("Longitudinal comparison requires identical reference, nodes and settings")
        ref_path, _ = checked_artifact(a["parameters"]["reference_run"], "gradients.npz", "reference")
        if digest(ref_path) != a["parameters"]["reference_sha256"]:
            raise ValueError("Alignment reference changed")
        nodes, rows = load_nodes(args.nodes), []
        with np.load(first, allow_pickle=False) as g1, np.load(second, allow_pickle=False) as g2, np.load(ref_path, allow_pickle=False) as ref:
            for hemi in ("L", "R"):
                x, y = g1[f"{hemi}_coordinates"], g2[f"{hemi}_coordinates"]
                idx = g1[f"{hemi}_indices"]
                if not np.array_equal(idx, g2[f"{hemi}_indices"]):
                    raise ValueError("Node ordering changed")
                center = ref[f"{hemi}_coordinates"].mean(0)
                e1, e2 = geometry(x, center)["eccentricity"], geometry(y, center)["eccentricity"]
                d = displacement(x, y)
                for j, i in enumerate(idx):
                    rows.append({"node_index": int(i), "network": nodes[i]["network"], "hemisphere": hemi,
                                 "displacement_4d": float(d[j]), "eccentricity_reference_center_change": float(e2[j] - e1[j]),
                                 "axis_change_G1_G4": (y[j, :4] - x[j, :4]).tolist()})
        path = out / "parcel_change.json"
        write_json(path, rows)
        inputs, outputs = [first, second, args.nodes, ref_path], [path]
        parameters = {**a["parameters"], "clinical_inference": False,
                      "caution": "Imaging provenance/scan pairing must be audited separately; no symptom axes assigned"}
    record(out, command, inputs, parameters, outputs, status)


def main(argv=None):
    args = parser().parse_args(argv)
    out = None
    try:
        if args.command == "gui":
            from .gui import serve
            serve(args.data_root, args.output_root, args.port)
            return 0
        for name in ("dicom_dir", "inventory_run", "import_run", "reference_run", "connectome_run", "first_run", "second_run", "reconstruction_run", "prepared_run"):
            source = getattr(args, name, None)
            if source:
                require_separate_output(args.output_root, source)
        if args.command == "import-session":
            require_separate_output(args.output_root, read_json(Path(args.inventory_run) / "inventory.json")["source_root"])
        if args.command == "import-nifti":
            from .nifti import validate_selection
            validate_selection(read_json(args.selection), args.output_root)
        lineage_files = {"reconstruct": ("import_run", "converted.json"),
                         "prepare-session": ("reconstruction_run", "reconstruction.json"),
                         "finish-session": ("prepared_run", "prepared.json")}
        if args.command in lineage_files:
            field, filename = lineage_files[args.command]
            original = read_json(Path(getattr(args, field)) / filename).get("source_root")
            if original:
                require_separate_output(args.output_root, original)
        out = new_run(args.output_root, args.command)
        run(args, out)
    except Exception as error:
        if out is not None:
            write_json(out / "FAILED.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
