# Numerical-replay validation — 2026-10-24

Version: **0.3.0a2** research preview (includes the earlier 0.3.0a1 checks). This record is intentionally narrower than
"the entire paper has been reproduced from DICOM". It does not confer clinical validity.

## Release checks

- 164 synthetic/unit/integration checks passed locally on macOS, Python 3.12.
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
| Representation perturbation, structure preservation and spatial error | 42 summaries | 1.4e-15 |
| Raw/standardized representation comparisons | 50 summaries across two cohorts | 9.5e-15 |
| Hemisphere-aware change-map concordance | 10 summaries, 20,000 permutations | 3.5e-18 |
| Original void-follow + historical repairs | 18 adopted masks | Bitwise equality |
| Unmasked preoperative volumetric graph implementation | 552 values | Matched within 1e-10 relative tolerance |
| Matched cross-cohort replication | 180 values | 0 |
| Reference transfer/group correspondence/normalization | 32 values | 1.5e-14 |
| Four-calibre ICC and clinical-correlation ranges | 48 values | 3.4e-16 |
| Five lateralized definitions and left/right contrasts | 32,620 values | 4.5e-16 |
| Symptom-dominant-side metrics and both nulls | 21,896 values | 2.0e-14 |
| Historical 280-test inventory | 1,680 values | 3.4e-16 |
| Historical 420-test shared/session variants | 5,880 values | 3.4e-16 |
| Historical 322-test shared-mask inventory | 2,254 values | 3.4e-16 |

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
6. Historical matched cross-cohort replication used all **ten** computed coordinates.
   It is reproduced with its actual dimensionality and accompanied by a separate four-axis
   sensitivity result. It must not be labelled G1–G4 replication.
7. The old exploratory `icc` column is a clipped between-subject/total variance ratio,
   **not** an ICC estimator. It is exported under an explicit non-ICC name. The historical
   subcortical inventory duplicated post1 in its time-0 slot; that slot is not called a
   measured preoperative value. Historical pooled bilateral geometry remains labelled.
8. An existing provenance note records that the old 420-test output directory, despite
   its comments, used shared-mask matrices. The distinct session-mask output is separately
   reproduced. Input schemes are explicit and hashes preserve the distinction.
9. The 420-test family has 14.6667 expected hits under its actual row-conditional exact
   nulls. The manuscript states 13.99. The software exports the computed expectation and
   complete shared-relabelling histogram; no manuscript correction has been made here.

## Statistical design and family nulls

The n=6 exact ordering/leave-one-out counts were independently enumerated. The critical
two-sided Spearman magnitude is 0.885714 at alpha 0.05, with achieved probability 24/720.
Nine power points used 200,000 bivariate-normal replicates each, seed 20261024, and agreed
with the historical rounded estimates within five combined Monte Carlo standard errors
plus the rounding bound. This is statistical agreement, not equality to unknown old draws.
The six first estimated n values at 80% power matched 7/9/10/12/15/22. Nulls are exact
through n=8 and sampled from 400,000 orderings above n=8. Every n-specific simulated power
estimate is exported; first crossing remains uncertain, not a guaranteed design prescription.

The 322-test dependency-preserving null matched the stored rounded count, mean and upper
tail: 7, 11.2444 and 0.7667. Its exact recomputed upper tail is 0.7666667. The 280-test
expectation is 9.3333, and both 420-test schemes yield 14.6667. Observed hit counts are
9, 7 and 8, respectively. The family tests are not assumed independent.

An installed-wheel GUI run completed 14 supplied real-data replay tasks; all frozen checks
passed and source hashes were unchanged. That suite does not include raw-image reconstruction.
There are now 169 local synthetic/regression tests, including statistical-audit checks.

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
- Finish the remaining conventional-graph/omnibus family audits and figure-source coverage.
  Gradient-inventory hit-count nulls and the stated exact-test power design are now implemented.
- Rebuild every adopted figure/table from a single adjudicated version of its input lineage.
- Resolve the surface-graph and rounded-pooling discrepancies with the study authors.

Until these gates pass, use **"numerically checked research preview"**, not
**"validated reproduction of every paper analysis"** in software descriptions.
