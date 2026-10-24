# Development status

Updated 2026-10-24 (JST). Version: **0.3.0a2**, GUI research preview.
This is not an end-to-end validated DICOM pipeline or a clinical tool.

| Workstream | Current state | Next acceptance check |
|---|---|---|
| Local browser GUI | Control-panel layout: directories, grouped operations, explicit settings, job table and log; input browser and cancellation retained | More failure/interruption testing; persistent restart/recovery |
| Mask QC | b0/T1/CT/5TT backgrounds, atlas boundary overlay, hash-bound manual approval | Longitudinal overlays; independent expert review |
| DICOM import | Header inventory, explicit roles/patient/visit review, private selected-series conversion and output choice | Multi-vendor and enhanced-DICOM validation |
| NIfTI intake | Private image/sidecar copies, gradient checks, source-integrity audit | More vendor/header cases |
| CT/MRI/DWI reconstruction | Acquisition-gated DWI/FOD and rigid registrations; linked FreeSurfer/NextBrain/5TT adapters | Independent full patient reconstruction |
| Electrode mask | CT candidates + constrained MRI void, manual RAS point and local add/erase editor; immutable mask adoption | Study-algorithm equivalence; independent coverage validation; automatic longitudinal registration |
| Linked connectome | QC → ACT → strict segment exclusion and audit → SIFT2 refit and weight checks → matrix → G1–G4 | Real-patient whole-chain testing |
| Gradient G1–G4 | Common/LOSO cohort orchestration, 25 measures/network, exact clinical tests and interactive heatmap/scatter; eight-setting sensitivity | Full supplementary-family and figure replay; independent acquisition validation |
| Numerical replay | Allowlisted suite for gradients, graphs, controls, sensitivity and unmasked reference benchmarks; frozen-output comparison | Complete all remaining paper coverage gates; resolve historical graph-basis lineage |
| Field / VTA-like modeling | Separate OSS-DBS 0.5.9 adapter; explicit model/tip/rotation; voltage/current; field NIfTI and local viewer | Independent localization, mesh/domain convergence, pulse-width/axon modeling; not clinically validated |
| Distribution | MIT GUI/workflow package; calls user-installed software; no all-in-one bundle | Stage-specific environment checks and tested local dependency combinations |
| Environment setup | Read-only GUI discovery via PATH/configured suite roots; stage-specific execution checks | GUI path settings, resource/license availability checks and bounded smoke tests |

## Evidence obtained so far

### Added numerical replay evidence (2026-10-24)

- Internal primary common/LOSO arrays: 28,728 values each, identical; 2,730 network table
  values identical. All 350 clinical correlations/probabilities independently rechecked.
- Historical external masked clinical table: 1,400 changes identical. The approved later
  mask revision is tracked separately, not silently substituted for this historical fixture.
- Eight gradient configurations: 192 participant measures identical, 128 correlation/P
  values within 3.4e-16, 32 axis-matching values identical.
- External unmasked benchmark: 15 identification/density summaries and three healthy-axis
  comparisons matched within 3.6e-15, including 23/24 common-reference assignments.
- Repeat-scan and synthetic-void controls: 51 frozen summary values matched within 2e-15.
  Noise/outcome comparisons are also exported; reliability alone is not clinical validity.
- Distance, donor-mask, Dice/Mantel and exclusion-cost controls: 86 frozen numeric values
  matched within 7.2e-15, including the unmasked-baseline Mantel follow-up.
- Representation perturbation-to-signal, between-subject structure preservation and spatial
  error profiles: a further 42 values matched within 1.4e-15.
- Raw/standardized G1, G1–G4, node and edge representation comparisons in both cohorts:
  50 values matched within 9.5e-15. Ten change-map concordance summaries also matched,
  including the 20,000-draw within-hemisphere parcel-permutation null.
- Study void-follow primitive matches the original implementation in 18 sessions. Replaying
  three explicit historical correction overlays reproduces all 18 adopted masks bitwise.
  This does not independently approve new masks or longitudinal registrations.
- The 92-value preoperative volumetric graph implementation matches all six stored cases
  (552 comparisons). The older surface graph CSV came from now-quarantined matrices and
  does **not** match the corrected adopted matrices; it is recorded as a lineage issue,
  not counted as a passing replay. The associated figure/table provenance needs adjudication.
- Common-reference graph and edge comparators reproduce 3/18 and 16/18 identification.
- A macOS NextBrain DataLoader workaround runs only in a child process, without modifying
  the installed tool. Real anatomical processing is in progress, not declared validated.
- Matched cross-cohort replication, unmasked-reference transfer and normalization audits:
  212 frozen values matched within 1.5e-14. Calibre-wise ICC/rho ranges: 48 values matched.
- Five lateralized definitions, left/right contrasts and symptom-dominant-side analyses:
  54,516 numerical comparisons matched within 2e-14, including all multiplicity corrections.
- The 280-test inventory, 420-test shared/session variants and 322-test shared-mask inventory:
  9,814 numerical comparisons matched within 3.4e-16. The two mask variants are explicitly
  separated; an old directory's name/comment was not trusted over its provenance note.
- Historical matched replication used ten axes, not four. The exact ten-axis replay and a
  separately labelled four-axis sensitivity calculation are both exported. Historical
  pooled-coordinate geometry and the old non-ICC variance ratio are not presented as valid
  hemisphere-aware geometry or scan-rescan ICC.
- 164 synthetic and regression checks pass locally; see the
  [numerical-replay validation record](VALIDATION_REPLAY_20261024.md).
  Earlier evidence below belongs to v0.2.0a1.

- 113 synthetic unit/integration/static-interface tests passed locally on macOS/Python 3.12. They cover source/output separation, byte-integrity guards, DICOM/NIfTI lineage, acquisition guards, geometry, mask approval hashes, label indexing,
  gradient reference integrity, GUI access controls, job execution and cancellation records.
- New CT/MRI candidate generation, MRI-only seeds, local correction audit, stale approval rejection,
  tissue/field unit checks, fixed-reference paired G1–G4 CSV values and report matrix identities are covered.
- Actual OSS-DBS voltage and current FEM completed on a homogeneous synthetic phantom; field and binary
  threshold NIfTI and browser reports were generated. This tests adapter interoperability, not patient VTA.
  See [dated validation record](VALIDATION_20261024.md) for exact scope and reproduction commands.
- A local browser QC job using an artificial phantom was queued, completed and opened in the GUI.
  Browser QA of the DICOM control-panel GUI subsequently covered installation discovery, synthetic
  series listing, selection and real dcm2niix conversion. Full patient reconstruction remains untested.
- A wheel was built and installed into a separate virtual environment using existing scientific
  dependencies; the CLI entry point and bundled GUI asset were verified outside the source checkout.
  This is a packaging smoke test, not a dependency-isolated clean installation.
- GitHub-hosted macOS and Ubuntu runners subsequently installed the Python package and passed
  the synthetic test suite: [initial CI run](https://github.com/taichisayanagi/Post_DBS_Connectome_analysis/actions/runs/34901769633).
  This does not install or validate the external MRI reconstruction tools.
- One synthetic gradient primitive was compared with the study implementation: matching aligned
  and unaligned coordinates and eigenvalues in that fixture. This is not full-study reproduction.
- Synthetic MRtrix checks cover dilation geometry and exclusion of a densely sampled test track.
  A sparse two-vertex track initially crossed the mask without exclusion; the prototype now rejects
  overly sparse imported tracks. A separate dense corner-clipping fixture also survived MRtrix's
  sampled exclusion. The new strict continuous-segment filter removes that residual track before
  SIFT2, and rereading the final tractogram confirms no intersection. The predicate was compared
  with a brute-force voxel-box oracle on 350 randomized cases, including oblique coordinates.
- The opt-in [installed-MRtrix smoke test](../examples/verify_installed_mrtrix.py) completed actual
  tckedit, tcksift2, tck2connectome and BrainSpace calls on a wholly artificial 32-node fixture:
  752 input tracks, 750 after exclusion, 750 fitted weights, symmetric matrix and finite
  hemisphere gradients. Matrix weight totals matched the fitted weights. No external calls were
  mocked in this test. Tracks/FOD/5TT were artificial prepared inputs, not generated by ACT or
  reconstructed from DICOM; this does not validate anatomy, human QC or patient outcomes.
- The linked continuation test mocks external MRI commands but runs exclusion auditing and
  gradient embedding for real. It validates artifact handoff, not MRI-tool interoperability.
- A real NIfTI intake smoke test copied six source files and confirmed identical original
  bytes, modification times and permissions. It did not execute heavy MRI reconstruction.
- A subsequent local postoperative NIfTI run completed 21 external commands through denoising,
  b0 selection, topup/eddy, bias correction, 2-mm resampling, WM/CSF FOD estimation and rigid
  T1-to-DWI registration. The ten original files retained identical hashes, modification times
  and permissions. Output-grid/rotated-gradient checks passed and selected registration views
  received technical review, not human mask/atlas approval. FreeSurfer/NextBrain and the full
  patient downstream chain remain to be validated. This is not completed input-to-gradient
  validation. Input review also
  motivated explicit b-value-driven selection for mixed reverse-PE series, with refusal tests
  for missing/mismatched b-values and no-baseline data. The reverse b-value field was checked
  in the running browser GUI. All external commands now use the private output working directory.

No real patient DICOM-to-result run has been performed with this software. No patient data or
private validation artifacts are included here. Tests are not evidence of clinical validity.

## Reading progress in the GUI

`queued` → `running` → `completed`, `needs_qc`, `failed`, or `cancelled`.
`planned` means a dry-run plan was generated, **not** that image processing occurred.
The interface reports elapsed time, current external tool and local log excerpts. It does not
invent percentages or estimated completion times for tools that do not expose them.
`needs_qc` explicitly means reconstructed images require review and missing downstream preparation;
it does not mean a connectome has been completed.

See [interface design](GUI_DESIGN.md) for the Lead-DBS-inspired control-panel organization.
See [local dependency policy](LOCAL_DEPENDENCIES.md) for user-managed tool installations.
The next major milestone is real-tool, full-chain validation including anatomical segmentation,
candidate mask assessment and a common longitudinal node basis, followed by manuscript replay.

## 日本語の要点

GUIを主な利用画面とし，CLIを解析エンジンとして維持する．進捗は工程名，状態，経過時間，ログで確認できる．
完了率を推定できない工程には架空のパーセントを表示しない．

現在は開発版であり，NIfTI／DICOM入力，撮像条件確認，前処理／FOD，位置合わせ，
FreeSurfer／NextBrain，atlas／5TT，電極マスク候補，QC後のACT／SIFT2／connectome／G1–G4を連結した．
研究のvoid-follow primitiveと明示的な修正overlayによる18マスクの数値再現を確認した．
新しい症例の画像QC，独立症例での全工程検証，論文全体の全補足解析・全図の再生成は未完了である．
研究用の完成版として論文に記載する前に，実データでの全経路検証が必要である．

See [DICOM workflow scope and verification](DICOM_WORKFLOW.md).
