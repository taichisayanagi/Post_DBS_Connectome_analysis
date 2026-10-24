# Paper-analysis replay

The `study` operation reconstructs the primary **matrix-to-result** analysis.
It is not a replacement for validation of image reconstruction and human QC.
Inputs and private reports must remain outside this repository.

## Definitions implemented

- Three explicitly paired visits per participant; fixed node TSV and file hashes.
- Hemisphere-wise cortical log1p/normalized-angle diffusion maps: sparsity 0.75,
  alpha 0.05, diffusion time 0, ten computed components and four retained after alignment.
- Separate common-preoperative and leave-one-participant-out references. Every visit of
  one participant uses the same LOSO reference; membership is recorded.
- Session/hemisphere-centred eccentricity, network-centred spread, per-axis position
  and dispersion, and postoperative parcel/centroid/geometric-median displacement.
  The 25 measures per network preserve the frozen study definitions, including the
  pooled residual SD denominator of parcel count minus two hemisphere centroids.
- Complete-case Spearman screening with all n! permutations for 3–8 participants,
  endpoint-specific BH and shared-permutation max-statistic families, and leave-one-out
  correlations. Missing endpoints are not imputed; constant outcomes are not tested.
- Common-reference patient identification with baseline labels fixed and postoperative
  labels independently permuted. Exact for n≤6; sampled with a plus-one p-value for
  larger cohorts. No patient-specific LOSO reference is used for primary identification.
- Browser inventory, heatmap and descriptive Theil–Sen scatterplots, with uncorrected
  and adjusted probabilities displayed separately and all tested measures retained.
- Optional frozen array/CSV checks with predeclared tolerances. A mismatch stops the job.

`graph-study` exports fixed-cost binary/rank-weighted graph measures, node-strength
and raw edge changes, and graph-change clinical screening. These use the full selected
node basis. They must not be interchanged with the paper's **different unmasked,
preoperative cortical** graph predictor. Endpoint edges are not named anatomical tracts.
An optional `preoperative_graph` block explicitly supplies its separate unmasked cortical
node TSV, hashes, ordered subjects and matrix row mapping. All 21 cost-mean measures are
computed before excluding undefined/constant variables from clinical tests. The original
volumetric and corrected surface bases must never be silently interchanged.

`sensitivity-study` runs all eight frozen configurations, one-to-one absolute axis matching,
and the four declared candidate definitions. Signs are not flipped after matching.
`reference-validation` uses unmasked matrices for the common-reference identification and
healthy-axis benchmark. Its LOSO gallery is re-embedded in full under each held-out reference.
`control-study` replays same-day ICC, displacement and synthetic-void controls from reviewed
matrices, including the noise/outcome association, not only favourable controls.

`geometry-controls` recomputes near/mid/far identification, random matched-size parcel
subsets, donor-mask pull under an exact donor relabeling null, mask-Dice Mantel tests,
unmasked-baseline Mantel follow-up and exclusion cost by distance. The manifest's
`geometry_controls` block requires `geometry_reviewed`, ordered `subjects` with explicit
atlas source labels in cortical L-then-R order, image paths/hashes, registered donor/target
mask pairs and optional masked swap matrices. Distance means parcel centroid to the nearest
mask-voxel **centre**, not distance to a surface or the physical metal. Pairwise registration
and parcel-label correspondence are supplied and require independent review.
When unmasked preoperative matrices are supplied, fixed-reference perturbation-to-signal,
between-subject structure preservation and spatial error are computed for gradient, node
and edge representations. These mask perturbations are not independent scan-rescan data.

`representation-study` compares G1, G1–G4, log-weight nodal strength, degree and log edges
under raw correlation and feature-wise standardization. Only different visits are eligible
neighbours. Its margin is **mean within minus mean between** similarity, not the nearest-
neighbour margin of primary identification. Optional `change_concordance` specifies
`permutations` and `seed` for hemisphere-centred postoperative eccentricity-change maps.
This historical null independently shuffles parcel labels within patient and hemisphere;
it does **not** preserve cortical spatial autocorrelation. It is not a spin-test substitute.

`replay-suite --config ...` executes an allowlisted list of these tasks, keeps failures,
and writes a browser summary. Every task manifest is hash-bound. A task without a frozen
expected-output comparison is **computed, not verified**. The suite never issues an
image-to-paper certificate, even when every selected numerical check passes.

`comparative-study` adds three deliberately descriptive audits. Its `comparative` block
names an `endpoint` and hash-bound linked `external`, `transfer_reference` and/or `calibres`
manifests (`manifest` plus `sha256`). Nested comparative manifests are rejected. Every
calibre must retain the same participant order, node identities and endpoint values.
The reference-transfer branch requires `parcel_mapping_reviewed: true`, maps complete
parcel names with matching hemisphere/network metadata, and uses explicitly unmasked
preoperative references. Standardization of external session features is descriptive,
not an out-of-sample prediction model. `replication_components` defaults to four; the
historical ten-component setting must be explicitly selected. When ten are selected,
both the ten-axis replay and a separate four-axis sensitivity table are exported.
Calibre stability uses its historical mean-post1 reference, not primary LOSO.

`lateralized-study` recomputes five hemisphere definitions, their exact correlations,
left/right correlation contrasts, and optional symptom-side analyses. To enable the latter,
provide `symptom_side_reviewed: true` and `symptom_dominant_side` keyed by each coded subject,
with physical `L`/`R` values. Contralateral labels are derived from these reviewed clinical
labels, never inferred from imaging. The side-label null preserves the observed allocation
imbalance. Signed-position ratios are undefined and explicitly excluded. Missing or
nonpositive denominators are not replaced with small constants.

`exploratory-study` supports explicit `exploratory_inventory` choices: `legacy_140`
(default, the 280 tests with two endpoints), `legacy_60_mean_centres` and
`legacy_46_median_centres`. The latter two give 420/322 tests with seven complete endpoints
on the study's node set. Counts depend on available structures and constant measures.
These are historical-definition audits: bilateral coordinate pooling, unaligned axis
meaning, eigenvalue shares and the non-ICC variance ratio carry warnings. The legacy
subcortical time-0 slot duplicates post1 and must not be used as preoperative data.
Both all-endpoint and endpoint-wise shared-permutation max statistics are exported.
Discrete-null hit expectations are calculated from each actual permutation distribution,
not hard-coded as 5% or 3.33% of the test count.

`study-mask` exposes the original filled-brain-exit b0 void-follow primitive on a
2-mm grid. Its output is a candidate requiring review. The separately implemented
CT/MRI editor is a different algorithm. Manual repairs and longitudinal registration
cannot be silently inferred from agreement of the primitive with its old implementation.
`mask-repair` replays explicit add/remove overlays into a new candidate and records parent,
overlay, reference and output hashes. `shared-masks` registers the postoperative b0 images,
transports masks with nearest-neighbour interpolation, forms unions in both native grids,
then dilates once on the 2-mm grid. Both native-grid reviews are shown; neither is auto-approved.

The optional reconstruction profile `paper_dil1` exposes the study-specific response-fit
order, WM/CSF model, affine T1 registration, raw FOD for tracking and normalized FOD for
masked SIFT2. The public workflow's continuous-segment exclusion remains stricter than
the historical sampled MRtrix exclusion. This profile is not a bitwise historical-tractogram
replay; original random track seeds and complete end-to-end validation are not available.

## Manifest

Use the GUI's **Cohort / paper numerical replay** operation, or:

```sh
dbs-connectome study --manifest /private/reviewed/study.json --output-root /private/results
dbs-connectome graph-study --manifest /private/reviewed/study.json --output-root /private/graphs
```

The version-1 JSON manifest contains:

- `schema_version: 1`, `pairing_reviewed: true`.
- `nodes`: absolute TSV path and `nodes_sha256`.
- `subjects`: unique coded `id` and exactly three `visits`, ordered `pre`, `post1`, `post2`.
- Each visit: `matrix`, `sha256`, and `mask_scheme` (`virtual`, `shared_postoperative`,
  or `none` for the baseline). Masked visits require `exclusion_evidence` and its SHA-256.
  Optional `matrix_indices` explicitly maps noncompact matrix rows into TSV order.
- `clinical`: endpoint names, each with a precise `definition` and `values` keyed by every
  participant ID. Use JSON `null` for unavailable values. Do not substitute zeros.
- Optional `expected`: checks of kind `array`, `metrics_csv`, or `correlations_csv`, with
  `path`, `sha256`, `actual`, array `key` or CSV `columns`, `atol` and `rtol`.
- `table_csv` also accepts explicit row `keys`, `key_map` and `column_map`. `numbers_json`
  checks explicitly listed nested paths (or `actual`/`expected` path pairs). Graph-specific
  checks live in `graph_expected`, so gradient checks are not accidentally applied to graphs.
- Optional `identification_samples` (1,000–10,000,000) and `seed` (default 42).

Exclusion evidence records which historical matrix was selected. It does **not**
independently prove streamline exclusion. The matrix replay never grants upstream QC.

The default zero-profile policy is `reject`. A recorded external study session contains
an isolated ipsilateral cortical profile. For explicit **historical replay only**, with
frozen expected outputs supplied, `legacy_zero_profile_policy: "brainspace_normalized_angle"`
retains the original BrainSpace handling. Every affected node is listed in the report.
No node is dropped or imputed. Numerical agreement under this compatibility policy does
not establish that the isolated profile is scientifically interpretable.

For `sensitivity-study`, declare `analysis_role: "embedding_sensitivity"` and supply a
`sensitivity_reference` with path/hash, array key, exact subjects/visits, `reference_mode: "loso"`
and mask scheme. For `reference-validation`, declare `analysis_role: "unmasked_reference_validation"`
and unmasked visits; an optional `healthy_reference` supplies path/hash, delimiter, component,
explicit reference rows and shared-parcel comparison indices. No atlas-name heuristic fills gaps.
Control cases have explicit definitions and ordered, hash-bound subject matrices in
`control_sessions`: `retest`, `simulation_baseline`, `simulation_void`, `simulation_rim`,
and `simulation_retrack`. They must share the fixed node basis and reference design.

## Statistical design audit

`power-study --config examples/statistical_design.json --output-root /private/results`
is also exposed in the GUI. It enumerates the n=6 permutation/leave-one-out boundaries,
then runs explicitly seeded bivariate-normal power simulations. The generating parameter
is Pearson r, while the decision statistic is Spearman rho. Power is not asserted to be
bitwise-reproducible against an old run with an unknown seed. Above n=8 the permutation
null is sampled, not exact. The output includes every evaluated n, Monte Carlo uncertainty,
and the first estimated threshold crossing. No clinical effect size is selected automatically.

`exploratory-study` also exports a shared-participant-relabelling histogram of nominal hit
counts. It preserves dependence and tied ranks rather than multiplying the number of tests
by a nominal five-percent rate. Historical counts that do not agree remain discrepancies.

## Remaining full-paper acceptance gates

- Reexecute and validate raw-image reconstruction through anatomy, mask QC, ACT and SIFT2.
- Validate new shared-mask registration and review both native grids. Historical correction
  overlays have been replayed; fresh image QC is a different requirement.
- Validate the separate unmasked preoperative graph basis and all supplementary sensitivity,
  reliability, artifact and healthy-reference analyses.
- Remaining supplementary coverage includes conventional-graph/omnibus audits
  and all adopted figure sources. Legacy 280/322/420 inventories and their hit-count nulls,
  lateralized/symptom-side analyses and comparative audits are separate operations;
  supplying one passing primary-analysis manifest does not execute them automatically.
- Rebuild every adopted figure/table from recomputed results and verify its source values.
- Complete expert image QC before any new patient-derived outputs are accepted.

Do not describe these remaining gates as complete because primary gradient arrays agree.
