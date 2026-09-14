# DICOM workflow — development adapter

Updated 2026-09-15 (JST). This is an **incomplete, unvalidated reconstruction route**,
not a clinical analysis service or a validated implementation of the manuscript pipeline.

## What now runs

The local browser's **DICOM workflow** tab provides:

1. Read-only discovery of tools in PATH or explicitly configured suite roots.
2. Header-only DICOM inventory grouped by SeriesInstanceUID. Patient identifiers are compared
   using per-inventory keyed tokens; names and PatientID are not displayed. Study dates,
   protocol descriptions, UIDs in private manifests and file paths remain potentially identifying.
3. Explicit T1/DWI selection, optional postoperative CT/T2/reverse-b0 selection, and patient/visit
   review. Mixed/missing patient identifiers, duplicated SOPInstanceUIDs, changed source files
   and unreviewed cross-study pairing block conversion.
4. Private, selected-series copies and dcm2niix conversion. Source byte hashes are retained.
   Multiple echoes/reconstructions remain separately selectable. No image is silently discarded.
5. Acquisition-gated command planning and execution: native-grid denoising, optional reviewed
   Gibbs correction, eddy/motion correction, optional topup, ANTs bias correction, a derived 2 mm
   analysis grid, response estimation, explicitly selected two- or three-tissue CSD and mtnormalise.
6. Candidate rigid T1→b0 and optional CT/T2→T1→b0 registration. Transforms are retained.
   Numerical success does **not** establish anatomical alignment, especially near electrodes.
7. Live process logs, elapsed time, explicit thread/compute consent and a `needs_qc` outcome.

No large MRI tools, models, atlases or MATLAB runtimes are bundled or installed by these steps.
Only commands used by the chosen route are required. The environment panel distinguishes detection
from testing; it does not launch MATLAB, check out a license or read license-file contents.

## Source-data protection

Original patient data are read-only inputs. The application must not delete, move, rename,
overwrite or change their permissions. GUI output roots cannot be inside the selected input
directory (including symlink aliases). CLI DICOM and upstream-run stages also reject overlapping
input/output directories before creating a run. All products and logs use separate timestamped
private directories; existing results are never silently overwritten. Selected DICOMs are copied
to a private staging directory. Source byte hashes are compared before/after conversion and
reconstruction; a changed input prevents acceptance of the output. Original data are never used
as temporary working files.

## Explicit current boundaries

The route does not yet automatically reconstruct leads, reproduce the study's final void-follow
mask/repair procedure, build the surface-Schaefer/NextBrain atlas and five-tissue image, generate
the ACT tractogram, or orchestrate longitudinal registration and cohort reference construction.
These are **implementation gaps**, not just absent validation. Existing advanced stages accept
their externally prepared and reviewed products for mask approval, exclusion/SIFT2/connectome
generation, reference-bound gradients and longitudinal change.

Reconstruction never creates a mask approval file or a prepared-connectome config. Users must
not pass the **brain mask** as the **electrode exclusion mask**. The preview produced here is only
a brain-mask view; CT/T1 overlay and artifact checks still require an external image viewer.

## Acquisition checks and unsupported inputs

- DWI must be 4D with matching `3×N` b-vectors and N b-values. Nonzero b-vectors must have
  unit norm within 0.05; rank-deficient directions fail. No automatic gradient repair is performed.
- The initial adapter requires at least one b0 (≤50 s/mm²) and 28 directions per detected shell.
  Shell grouping uses a 100 s/mm² tolerance; this is a screening rule, not a scanner-independent
  quality guarantee. Low-direction and irregular/non-shelled acquisitions need expert handling.
- A three-tissue model requires two or more nonzero shells. Two-tissue WM+CSF is a distinct
  model and is never represented as three-tissue MSMT-CSD or as study-equivalent processing.
- Phase encoding is a reviewed NIfTI axis (`i`, `j`, `k`, optionally `-`). An anatomical AP label
  is not assumed. Missing/changed JSON values require a documented source. TotalReadoutTime
  is in seconds and is not interchangeable with effective echo spacing.
- No reverse-PE data means **no susceptibility correction**. Topup does not establish removal
  of metal-induced distortion or signal loss either.
- The paired route currently requires opposite PE, equal readout time, the same acquisition
  grid, reviewed SE-EPI contrast and b0 weighting, and a first DWI volume that is b0.
  That exact first volume is used in the pair with `-align_seepi`; the reverse b0 is averaged if 4D.
  Other acquisition designs stop instead of guessing a conversion or resampling raw data.
- Gibbs correction requires the reviewed in-plane axes. Partial Fourier/unknown settings require
  a specific acknowledgement or skipping, recorded in provenance.
- Matching headers/grids do not prove image orientation, correct patient pairing, scanner gradient
  convention, tractography accuracy or anatomical registration. Human QC remains necessary.

## Practical use

Launch `dbs-connectome gui --data-root /private/study --output-root /private/analysis`.
Use the generated local URL, then the DICOM workflow tab. Input paths and analysis products stay
on the same computer. These copies are **not fully anonymized** and must not be uploaded to GitHub.

The GUI generates its selection/acquisition JSON, so users do not have to write command scripts.
The equivalent CLI stages are `environment`, `inventory`, `import-session`, and `reconstruct`;
use each stage's `--help`. Import and reconstruction default to a plan unless `--execute` is set.
Processing pauses at the explicit development boundary above. Jobs survive a page reload through
the current server's job list (`Open result`); server-restart recovery is not implemented.

For a tiny non-patient import demonstration:

```sh
python examples/synthetic_dicom_demo.py --output-root /private/demo
```

The second series intentionally resembles an incorrectly supplied DWI: it is 3D and has no
diffusion gradient table. It must convert successfully but be **rejected for reconstruction**.
This example does not validate diffusion processing.

## Evidence and release acceptance

Synthetic tests cover patient mixing, missing identity, cross-study review, duplicate images,
source changes, immutable import lineage, gradient-table failures, wrong model/shell choice,
PE overrides, paired-b0 construction, correction order and non-execution of plan-only jobs.
A macOS GUI check converted two artificial MR series using installed dcm2niix.
Neither substitutes for a patient DICOM-to-gradient run or independent scanner/vendor testing.

Before a validated release: implement the missing adapters, compare each intermediate image and
mask against the study route, independently audit streamline exclusion, verify longitudinal
transform direction and frozen node/reference definitions, test interrupted runs, and have users
outside the development team complete a documented case without command-line assistance.

## Primary implementation references checked 2026-09-15

- [pydicom header-only reads](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html)
- [MRtrix dwifslpreproc acquisition modes and alignment prerequisites](https://userdocs.mrtrix.org/en/latest/reference/commands/dwifslpreproc.html)
- [MRtrix mrdegibbs axes and partial-Fourier limitations](https://userdocs.mrtrix.org/en/dev/reference/commands/mrdegibbs.html)
- [MRtrix dwi2fod](https://userdocs.mrtrix.org/en/latest/reference/commands/dwi2fod.html)

## 日本語

DICOMを選ぶ→患者・時点とシリーズを確認する→撮像条件を確認する→前処理を実行する，
という入口をGUIに追加した．コマンドの知識は不要だが，撮像条件が不明なときに推測で
補完する仕組みではない．施設の撮像担当者・解析担当者による確認が必要な場合がある．

電極同定，最終マスク，atlas／5TT／tractographyの自動生成・連結は残っている．
全経路を実データで検証するまでは，「DICOMから完成するソフトウェア」としての完成宣言や，
臨床判断への使用はしない．
