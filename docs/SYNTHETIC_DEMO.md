# Small synthetic integration demo

This input-free demonstration uses two geometrical balls, not a virtual patient.
It creates artificial data locally; there are no downloads and no patient files.
The only external tools required for this demo are **dcm2niix and MRtrix3**, already
installed on the computer, plus the package's Python dependencies.

## In the GUI

1. Start `dbs-connectome gui --data-root /your/empty/folder --output-root /your/results`.
2. Open the printed local URL. Select **Try a small synthetic demo…**.
3. Review the notice, select CPU threads, enable execution and confirm local computation.
4. Run the job. Open **View report** when it completes.

The demo has no input-path fields. Patient files cannot be substituted into this
QC-free fixture. The ordinary patient workflow still requires explicit image QC.
Without execution enabled, only a synthetic fixture and a command plan are saved.

CLI equivalent:

```sh
dbs-connectome synthetic-demo --output-root /your/results --threads 2 --execute
```

## Actual operations

- Generate two 24-slice CT/T1 DICOM series and a 44 × 24 × 24 × 68 diffusion NIfTI.
- Convert CT/T1 with dcm2niix; verify exact voxel values and world-space geometry.
  Lossless axis permutation/flipping puts the converted CT on the known reference grid.
- Import diffusion data with MRtrix and run single-shell CSD using a prescribed response.
- Detect two elongated CT candidates and combine them with the known synthetic MRI void.
  Verify coverage of every generated metal voxel; export the unapproved QC report.
- Generate 10,000 ACT/iFOD2 streamlines using the known synthetic five-tissue image.
- Exclude mask-intersecting streamlines with MRtrix and the strict continuous-segment filter.
  Independently reread the retained tractogram and require zero intersections.
- Refit SIFT2 after exclusion; check weight count, positivity/finite values and mu.
- Build the 32-node weighted connectome. Independently sum endpoint assignments and weights
  to check every edge, accounting for self-connections and unassigned endpoints.
- Export hemisphere-wise G1–G4 coordinates, all computed components, graph metrics,
  node strength/degree, edges, tool versions, hashes and a local HTML summary.

Outputs are stored under `synthetic-demo_YYYYMMDD_HHMM`. Source integrity is checked.
Tracking is stochastic: counts/weights may vary between successful runs. Acceptance is
based on explicit structural/integrity checks, not exact equality of random tractograms.

## What this does NOT test

The atlas and five-tissue map are generated ground truth. FreeSurfer, NextBrain,
anatomical registration estimation, denoise/topup/eddy and tissue-response estimation
are **not** run. Artificial directional signal is not a realistic brain diffusion model.
DICOM conversion covers artificial CT/T1, **not vendor-specific diffusion DICOM**.
The gradients are a single unaligned session; no longitudinal clinical effect is inferred.
Named anatomical tracts and FEM/VTA are outside this demonstration.
No patient QC approval is generated. This is integration evidence, not full-paper
reproduction or clinical validation.

See [the dated validation record](VALIDATION_SYNTHETIC_20261025.md).
