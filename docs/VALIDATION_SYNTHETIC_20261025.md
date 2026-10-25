# Synthetic integration validation — 2026-10-25 (JST)

Version: 0.3.0a4. Platform: local macOS arm64. No patient data used.

## Successful GUI execution

The new **Try a small synthetic demo…** control was exercised in the browser. Two
threads were selected, execution/compute consent enabled, and the job reached
**Completed**. Its result report was opened and visually inspected in the GUI.

| Check | Observed result |
|---|---|
| GUI wall time, including process startup | 5.4 s |
| Pipeline wall time | 4.45 s |
| Artificial DICOM series converted | CT and T1, 24 slices each |
| Exact voxel / world-space checks | Both passed |
| DWI | 44 × 24 × 24 × 68; four b0, 64 b=1000 directions |
| Actual generated ACT/iFOD2 tracks | 10,000 |
| After MRtrix sampled exclusion | 8,754 |
| Additional continuous-segment exclusions | 9 |
| Final retained / corresponding SIFT2 weights | 8,745 / 8,745 |
| Final mask intersections on reread | 0 |
| Connectome | 32 × 32; 218 nonzero undirected edges |
| Independent endpoint-weight maximum matrix error | 3.655 × 10⁻⁷ |
| G1–G4 | Finite; 16 nodes per hemisphere; 10 components computed |
| Original synthetic input hashes | Unchanged |
| Human clinical QC approval generated | No |

Executed programs: dcm2niix v1.0.20241211, installed MRtrix 3.0.4-221-gbb36e36b
(`mrconvert`, `dwi2fod`, `5tt2gmwmi`, `tckgen`, `tckedit`, `tcksift2`, `tck2connectome`),
and BrainSpace 0.1.22. Exact executable hashes/version text are stored in each local run.
External calls were not mocked in this integration run. Random tracking makes exact
counts/weights variable; no biological meaning or fixed numerical target is implied.

## Regression tests

176 local tests passed after adding the demo. New tests cover known atlas/tissue
truth, lossless orientation checks versus translated-image rejection, dry-run behavior,
missing-tool refusal, rejection of patient-input fields, explicit compute consent,
and an independent matrix sum that handles self/unassigned tracks.
The default CI suite does not install MRI tools; its passing status is separate from
the actual local-tool integration result above.

## Detected and repaired during development

The first run stopped because dcm2niix stored the physically equivalent CT with a
different array orientation. An exact canonical-world/voxel check already passed.
The fixture now explicitly saves a losslessly reoriented copy before same-grid
masking. The same-grid check was not weakened. No anatomical registration is inferred.

## Scope boundaries

Known synthetic atlas/5TT maps and a prescribed response replace anatomical segmentation
and response estimation. No denoise, topup/eddy, registration estimation, FreeSurfer or
NextBrain is run by this demo. DICOM diffusion-vendor handling, biological tract
anatomy, clinical outcomes, longitudinal validation and FEM/VTA are not tested here.
No patient QC approval, clinical claim or full-paper validation certificate is created.
No patient image, study matrix or private log is included in the public repository.

The earlier real-image NextBrain run independently completed left and right hemispheres
on 2026-10-24 (1 h 31 min and 2 h 01 min respectively). It resumed previously generated
anatomical intermediates; that separate evidence does not turn this demo into a validated
raw-image-to-paper pipeline. The manuscript was not changed.
