# Numerical-replay validation — 2026-10-24

Version: **0.3.0a1** research preview. This record is intentionally narrower than
"the entire paper has been reproduced from DICOM". It does not confer clinical validity.

## Release checks

- 151 synthetic/unit/integration checks passed locally on macOS, Python 3.12.
- A source-only wheel was built and installed in a separate virtual environment.
  Imports, CLI entry points and bundled report HTML were checked outside the checkout.
  This environment reused installed scientific dependencies; it is not a clean-environment
  external-MRI-tool validation. GitHub CI separately tests Python installation on macOS/Linux.
- The browser was used to launch the actual numerical replay, not a mocked job.
- Synthetic demonstration runs without frozen expected outputs are reported as
  `completed_with_unverified_tasks`, never as verified replay.
- Frozen comparisons reject empty comparison sets. Mismatches remain failures and are
  not silently overwritten with historical results.
- Every result records source hashes, package-source hashes and dependency versions.
  Inputs and historical outputs were read-only; no private validation files are distributed.

## Independently frozen numerical checks

| Analysis | Compared evidence | Maximum absolute difference |
|---|---:|---:|
| Internal common-reference coordinates | 28,728 values | 0 |
| Internal LOSO coordinates | 28,728 values | 0 |
| Internal network measures | 2,730 values | 0 |
| Internal clinical statistics | 350 tests × rho/P/BH/maxT | 3.4e-16 |
| Historical external masked network changes | 1,400 values | 0 |
| Eight embedding configurations | 192 participant values, 128 coefficients/P values, 32 axis matches | 3.4e-16 |
| Unmasked external benchmark | 15 identification/density and 3 healthy-axis summaries | 3.6e-15 |
| Repeat-scan/synthetic-void controls | 51 summaries | 2.0e-15 |
| Geometry/distance/mask-swap controls | 86 summaries | 7.2e-15 |
| Raw/standardized representation comparisons | 50 summaries across two cohorts | 9.5e-15 |
| Hemisphere-aware change-map concordance | 10 summaries, 20,000 permutations | 3.5e-18 |
| Original void-follow + historical repairs | 18 adopted masks | Bitwise equality |
| Unmasked preoperative volumetric graph implementation | 552 values | Matched within 1e-10 relative tolerance |

The mask comparison includes **three explicitly recorded manual correction overlays**.
Agreement of these historical masks does not approve a newly generated mask. Numerical
replay also does not prove that the historical data were free of imaging artifacts.

## Discrepancies retained, not hidden

1. The older preoperative **surface** graph table was derived from matrices now quarantined
   after label-index corrections. It differs from the current corrected matrix basis.
   It is not counted as a passing replay. Associated paper values need source adjudication.
2. A historical pooled correlation used a rounded external coefficient. The software
   retains full precision; rounded-input replay and full-precision recomputation are distinct.
3. The historical external masked fixture and a later approved mask revision are separate
   inputs. The earlier fixture is not silently replaced, nor called validation of that revision.
4. One external session has an isolated **within-hemisphere** profile. The default rejects
   this. Explicit historical compatibility requires frozen comparisons and reports the node.
5. Noise/outcome controls are exported even when unfavourable to biological interpretation.

## Raw-image boundary

Earlier local checks completed NIfTI intake and 21 real DWI/FOD/registration commands,
and actual MRtrix exclusion/SIFT2/connectome/gradient calls on an artificial prepared fixture.
A real NextBrain anatomical run was started with 12 threads using private copies of existing
upstream anatomical products. It is **not** a fresh complete DICOM-to-anatomy reconstruction.
At the time this record was written it was still running; no completion is claimed.

The study-specific reconstruction profile exposes the intended response-fit order and raw
tracking FOD. Continuous-segment streamline exclusion is stricter than the old sampled lookup.
Random tractography is not promised to reproduce historical tracks bitwise.

## Full-paper release gates still open

- Complete and inspect the real anatomical/atlas pipeline, both native-grid mask registrations,
  expert mask QC, ACT, SIFT2 and cohort node correspondence.
- Validate a complete new NIfTI/DICOM case through final results with source-integrity checks.
- Finish the legacy exhaustive families, symptom-side analyses, all caliber/representation
  robustness and cross-cohort transfer analyses, and power simulations.
- Rebuild every adopted figure/table from a single adjudicated version of its input lineage.
- Resolve the surface-graph and rounded-pooling discrepancies with the study authors.

Until these gates pass, use **"numerically checked research preview"**, not
**"validated reproduction of every paper analysis"** in software descriptions.
