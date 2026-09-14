# Post_DBS_Connectome_analysis

Scripts for Analysis for brains post deep brain stimulation electrode implantation

## DBS Native Connectome — development prototype

Research software for electrode-aware, native-space structural connectomics.
**Development prototype. Not clinically validated or an end-to-end DICOM pipeline. No validated release yet.**
Target repository: https://github.com/taichisayanagi/Post_DBS_Connectome_analysis .
New code is MIT-licensed. Dependencies and atlases retain their own licenses.
No patient data, brain images, private study matrices or third-party atlases are distributed.

**[Current development progress and remaining milestones](docs/STATUS.md)**

**New: [guided DICOM intake and reconstruction adapter](docs/DICOM_WORKFLOW.md).** The browser
now inventories series, checks patient/visit selection, converts selected images and plans/runs
acquisition-gated DWI/FOD processing and candidate rigid registration. The full DICOM-to-gradient
route remains incomplete: automatic lead/mask/atlas/tractogram generation and real-data validation
are still required.

## Installation model

This is a lightweight GUI/workflow package that calls **software already installed on the user's
computer**. FSL, MRtrix3, FreeSurfer, NextBrain resources, MATLAB, SPM12, the Python interpreter,
atlases and model weights are not bundled. No all-in-one container is required.
Only the dependencies for the selected, implemented operation are required.
See [local dependencies and the planned environment checker](docs/LOCAL_DEPENDENCIES.md).
Installing these tools does not implement the reconstruction stages that are still missing below.

## Current functionality

- Explicit local DICOM conversion using dcm2niix (dry-run unless `--execute`).
- Candidate masks from **already registered, reviewed** electrode centerlines in NIfTI RAS+ millimetres.
- Study-style 6-connected dilation with an explicit 2-mm-grid check.
- Union of already anatomically registered masks with a hash-bound registration-review record.
- Human QC approval bound to the exact mask, b0 reference and parcellation hashes.
- Self-contained local browser QC: three slice sliders, mask overlay/opacity and intensity window.
  No external services or CDN. Reports contain private anatomy and must not be published.
- Prepared-input MRtrix command plans: mask-intersecting streamline exclusion, SIFT2 refit,
  weighted connectome and endpoint assignments. Execution requires `--execute --threads N`.
- Correct atlas-label-to-matrix indexing with no silent padding or node dropping.
- Study-compatible hemisphere-wise log1p / normalized-angle diffusion-map embedding:
  75% row sparsity, alpha 0.05, diffusion time 0, 10 computed / 4 retained components.
- Reference-bound longitudinal 4D parcel displacement and per-axis change.
- Timestamped, non-overwriting private run folders, hashes and decoded voxel fingerprints.

**Not implemented yet:** automatic CT lead localization;
the paper's b0 void-following algorithm and reviewed island repair; FreeSurfer/NextBrain atlas construction;
ACT tractogram generation; longitudinal transform orchestration; cohort LOSO orchestration;
CT/atlas/multi-session browser overlays; comprehensive dependency/resource validation and external end-to-end validation.

The centerline tube is a candidate-mask utility, **not** a reproduction of the study's final void-follow
mask. Its radius is not a universal artifact boundary. CT metal geometry and diffusion signal loss
need not coincide. Anatomical alignment and full-shaft void coverage must be inspected on the images.
An approval file records a human attestation; the software cannot establish that the inspection was correct.

## Quick start

Python 3.11+; development smoke tests used Python 3.12, numpy 2.3.1, scipy 1.16.2,
nibabel 5.3.3 and BrainSpace 0.1.22. External tools are separately installed.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
dbs-connectome --help
python examples/synthetic_demo.py --output-root /private/analysis
dbs-connectome gui --data-root /private/study --output-root /private/analysis
dbs-connectome convert --dicom-dir /private/input/selected_series --output-root /private/analysis
dbs-connectome qc --reference /private/b0.nii.gz --mask /private/reviewed_mask.nii.gz --output-root /private/analysis
```

All run directories follow `stage_YYYYMMDD_HHMM`. A collision aborts rather than overwriting.
MRI-derived outputs must live **outside this source checkout**, on institutional/local protected storage.
Open the generated `mask_review.html` locally. The viewer does not grant QC approval and cannot
replace CT, atlas and longitudinal registration review. It changes neither source images nor masks.

### GUI (primary user interface)

The `gui` command prints a local token-bearing URL. Open it on the same computer.
The app binds only to `127.0.0.1`, restricts file selection to the configured data/output roots,
and requires a session token. Do not port-forward or expose it to the network. It is not a
multi-user clinical system. Enter prepared input paths or use the local file browser.

The GUI shows queued/running/planned/completed/needs_qc/failed/cancelled states, elapsed time,
current MRtrix stage and log excerpts. Plan-only jobs are never marked as executed analyses.
Long-running stages with no numeric progress report do not get a fabricated percentage.
The GUI runs one job at a time. External execution requires an explicit consent checkbox;
connectome execution additionally requires a positive CPU thread count.
Full tool logs are in the private run directory. Jobs are not resumed after server restart.
For gradient jobs, the optional completed-connectome run field enables hash verification of
the matrix's electrode-exclusion lineage; omitting it is explicitly recorded as unverified.

Use the DICOM workflow tab for the implemented import/preprocessing/rigid-registration adapter.
Atlas, electrode localization/void-following and tractogram generation remain external steps.
The GUI is not evidence of complete or validated DICOM reconstruction.

### Prepared-input connectome

First prepare/register/QC the DWI products and parcellation using a separately validated pipeline.
Copy `examples/prepared_session.json` outside the repository and replace its paths.
Use a compact atlas with unique positive labels or provide the exact labels in `nodes.tsv`.
The result matrix follows the TSV's zero-based `index` column; MRtrix raw row zero is label 1.
This prototype does not perform many-to-one atlas-label merging.

```sh
dbs-connectome approve --mask /private/mask.nii.gz --reference /private/b0.nii.gz \
  --atlas /private/atlas.nii.gz --reviewer reviewer-code \
  --checked registration void_coverage full_shaft spurious_components atlas_alignment \
  --output-root /private/analysis
dbs-connectome connectome --config /private/prepared_session.json \
  --threads 8 --output-root /private/analysis
```

The second command **only plans** the expensive steps. Explicitly add `--execute` after review
and after agreeing resource use. Same grid is necessary but does not prove anatomical co-registration.
The provided tractogram must already be in the reference's world space; this is not inferable from a
TCK file alone. Execution rejects out-of-FOV vertices and tracks with segment spacing greater than
half the smallest reference voxel dimension: sparsely sampled tracks can jump over a mask.
Explicit upstream resampling and renewed validation are required in that case. Full lineage and
an independent continuous streamline-exclusion audit remain release requirements.

### Gradient analysis

Use one frozen node table across all compared sessions. Do not hard-code the study's 493-node basis
for another cohort. Zero-strength cortical nodes cause an error instead of silent per-session deletion.
Create an appropriate independent/training-only mean connectome as a reference input; the current
CLI **does not itself build LOSO references**. Each subject's two longitudinal sessions need the
same reference. Do not compare axes across separately aligned LOSO templates without further alignment.

```sh
dbs-connectome reference --matrix /private/reference_mean.npy --nodes /private/nodes.tsv \
  --output-root /private/analysis
dbs-connectome embed --matrix /private/post1.npy --nodes /private/nodes.tsv \
  --reference-run /private/analysis/reference_YYYYMMDD_HHMM --output-root /private/analysis/post1
dbs-connectome embed --matrix /private/post2.npy --nodes /private/nodes.tsv \
  --reference-run /private/analysis/reference_YYYYMMDD_HHMM --output-root /private/analysis/post2
dbs-connectome change --first-run /private/analysis/post1/embed_YYYYMMDD_HHMM \
  --second-run /private/analysis/post2/embed_YYYYMMDD_HHMM \
  --nodes /private/nodes.tsv --output-root /private/analysis
```

For a matrix generated by this tool, add `--connectome-run /private/analysis/connectome_YYYYMMDD_HHMM`
to `embed` to verify its hash against the completed exclusion pipeline. Matrix-only imports are
explicitly marked `unverified_matrix_import`; an embedding alone never proves electrode exclusion.

The reference-centred eccentricity output is explicitly named. It must not be interchanged with
session-centred eccentricity. Computed diffusion eigenvalues are **not PCA explained variance**.
G1–G4 have no automatic motor/cognitive labels. Displacement measures movement, not strengthened
connections, axonal growth, neuroprotection or clinical improvement.

## Privacy and release status

`dcm2niix -ba y` is **not complete anonymization**. DICOM, anatomy, metadata, paths, hashes,
logs and tractograms remain sensitive. Defacing does not by itself guarantee anonymization.
There is no network upload or telemetry function. `.gitignore` is not a privacy audit.
See `docs/RELEASE_CHECKLIST.md` before any public push.

## Upstream documentation

Implementation API references checked 2026-09-15 JST:

- [dcm2niix](https://github.com/rordenlab/dcm2niix)
- [MRtrix tck2connectome and endpoint assignment](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html)
- [BrainSpace GradientMaps](https://brainspace.readthedocs.io/en/latest/generated/brainspace.gradient.gradient.GradientMaps.html)

These links document dependencies, not validation of this prototype. Dependency binaries, atlases
and their licenses must be handled separately. No manuscript citation or software DOI is assigned yet.
