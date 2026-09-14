# Development status

Updated 2026-09-15 (JST). Version: **0.1.0.dev0**, research development prototype.
This is not an end-to-end validated DICOM pipeline or a clinical tool.

| Workstream | Current state | Next acceptance check |
|---|---|---|
| Local browser GUI | Control-panel layout: directories, grouped operations, explicit settings, job table and log; input browser and cancellation retained | More failure/interruption testing; persistent restart/recovery |
| Mask QC | b0/T1/CT/5TT backgrounds, atlas boundary overlay, hash-bound manual approval | Longitudinal overlays; independent expert review |
| DICOM import | Header inventory, explicit roles/patient/visit review, private selected-series conversion and output choice | Multi-vendor and enhanced-DICOM validation |
| NIfTI intake | Private image/sidecar copies, gradient checks, source-integrity audit | More vendor/header cases |
| CT/MRI/DWI reconstruction | Acquisition-gated DWI/FOD and rigid registrations; linked FreeSurfer/NextBrain/5TT adapters | Independent full patient reconstruction |
| Electrode mask | CT metal candidates or reviewed centerlines; ambiguous counts stop; registered union utility | Reproduce study void-follow; automatic longitudinal registration |
| Linked connectome | QC → ACT → exclusion → independent segment audit → SIFT2 refit → matrix → G1–G4 | Real-tool whole-chain testing; weight audit |
| Gradient G1–G4 | Reference-bound embedding and longitudinal change implemented | Automatic training-only/LOSO reference construction and study-level equivalence |
| Distribution | MIT GUI/workflow package; calls user-installed software; no all-in-one bundle | Stage-specific environment checks and tested local dependency combinations |
| Environment setup | Read-only GUI discovery via PATH/configured suite roots; stage-specific execution checks | GUI path settings, resource/license availability checks and bounded smoke tests |

## Evidence obtained so far

- 81 synthetic unit/integration/static-interface tests passed locally on macOS/Python 3.12. They cover source/output separation, byte-integrity guards, DICOM/NIfTI lineage, acquisition guards, geometry, mask approval hashes, label indexing,
  gradient reference integrity, GUI access controls, job execution and cancellation records.
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
  overly sparse imported tracks. The new independent segment/voxel-box audit also detects
  between-vertex crossings and prevents matrix generation if a retained crossing is found.
- The linked continuation test mocks external MRI commands but runs exclusion auditing and
  gradient embedding for real. It validates artifact handoff, not MRI-tool interoperability.
- A real NIfTI intake smoke test copied six source files and confirmed identical original
  bytes, modification times and permissions. It did not execute heavy MRI reconstruction.

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
研究の最終void-followマスクの完全再現と独立症例での全工程検証は未完了である．
研究用の完成版として論文に記載する前に，実データでの全経路検証が必要である．

See [DICOM workflow scope and verification](DICOM_WORKFLOW.md).
