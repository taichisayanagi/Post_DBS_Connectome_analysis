# Development status

Updated 2026-09-15 (JST). Version: **0.1.0.dev0**, research development prototype.
This is not an end-to-end validated DICOM pipeline or a clinical tool.

| Workstream | Current state | Next acceptance check |
|---|---|---|
| Local browser GUI | Implemented: input browser, job queue, states, elapsed time, tool logs, cancellation | More failure/interruption testing; persistent restart/recovery |
| Mask QC | Implemented: three-plane b0/mask viewer, slices, opacity/window, hash-bound manual approval | CT, atlas and longitudinal overlays; independent expert review |
| DICOM import | Conversion command and dry run implemented | Explicit series inventory, pairing, acquisition checks |
| CT/MRI/DWI reconstruction | Not implemented | Lead localization, registration, preprocessing, FOD, atlas and tractogram generation |
| Electrode mask | Candidate from reviewed registered centerlines; reviewed mask union | Reproduce and generalize the study's void-follow algorithm |
| Prepared-input connectome | Plans/executes exclusion → SIFT2 refit → matrix; explicit CPU choice | Complete fixture run, independent exclusion/weight audit |
| Gradient G1–G4 | Reference-bound embedding and longitudinal change implemented | Automatic training-only/LOSO reference construction and study-level equivalence |
| Distribution | MIT source; dependency/atlas licenses separate | macOS/Linux clean-install CI, containers, then validated release |

## Evidence obtained so far

- 38 synthetic unit/integration tests passed locally on macOS/Python 3.12. They cover geometry, mask approval hashes, label indexing,
  gradient reference integrity, GUI access controls, job execution and cancellation records.
- A local browser QC job using an artificial phantom was queued, completed and opened in the GUI.
- A wheel was built and installed into a separate virtual environment using existing scientific
  dependencies; the CLI entry point and bundled GUI asset were verified outside the source checkout.
  This is a packaging smoke test, not a dependency-isolated clean installation.
- One synthetic gradient primitive was compared with the study implementation: matching aligned
  and unaligned coordinates and eigenvalues in that fixture. This is not full-study reproduction.
- Synthetic MRtrix checks cover dilation geometry and exclusion of a densely sampled test track.
  A sparse two-vertex track initially crossed the mask without exclusion; the prototype now rejects
  overly sparse imported tracks. An independent continuous intersection audit remains outstanding.

No real patient DICOM-to-result run has been performed with this software. No patient data or
private validation artifacts are included here. Tests are not evidence of clinical validity.

## Reading progress in the GUI

`queued` → `running` → `completed`, `failed`, or `cancelled`.
`planned` means a dry-run plan was generated, **not** that image processing occurred.
The interface reports elapsed time, current external tool and local log excerpts. It does not
invent percentages or estimated completion times for tools that do not expose them.

The next major milestone is an audited reconstruction path from DICOM to reviewed prepared inputs.
After that: end-to-end independent-data validation and reproducibility of the manuscript outputs.

## 日本語の要点

GUIを主な利用画面とし，CLIを解析エンジンとして維持する．進捗は工程名，状態，経過時間，ログで確認できる．
完了率を推定できない工程には架空のパーセントを表示しない．

現在は開発版であり，DICOMからの位置合わせ・前処理・atlas・tractography生成は未実装である．
研究用の完成版として論文に記載する前に，実データでの全経路検証が必要である．
