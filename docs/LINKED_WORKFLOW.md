# NIfTI and DICOM linked workflow

Version 0.1.0.dev0. Research development code, not a validated clinical product.

## Inputs and local execution

Start with T1-weighted anatomy and **raw 4D magnitude DWI** with matching FSL-format
bval and 3-by-N bvec files, either as NIfTI or selected DICOM series. Optional postoperative
CT supplies metal candidates; optional T2 provides an additional registration output.
MRI-only automatic lead detection is not implemented. Already preprocessed DWI, ADC and FA
maps are not interchangeable with raw input.

NIfTI input is copied to a private timestamped run directory. A missing JSON becomes an
empty, explicitly marked metadata record: PE direction and readout time are never invented.
The operator must supply reviewed acquisition settings. NIfTI does not independently establish
patient identity. DICOM identifiers plus human series/visit review are used in DICOM mode.
No patient data are uploaded. Source content, permissions and names are not changed.
External commands also run with the private output directory as their working directory, so
tools using current-directory scratch files do not write into the launch or patient directory.

Reverse-PE input may be a single reviewed b0 image or a series with matching b-values.
For multi-volume input, b-values are mandatory: only volumes with b ≤ 50 s/mm² are
selected before averaging. Diffusion-weighted reverse volumes are not included in topup.
Both PE directions must have matching grids and reviewed contrast/readout metadata.
The selected thread count is forwarded to both topup and eddy; the installed FSL version
must support their `--nthr` option. The local topup help was checked for that option.

## Linked stages

1. `import-nifti` or `inventory` → `import-session`: private imported artifacts with hashes.
2. `reconstruct`: native-resolution denoising, optional Gibbs correction, eddy/topup as applicable,
   bias correction, final 2-mm grid, CSD/FOD and rigid T1/CT/DWI registration. Review these outputs.
3. `prepare-session`: private FreeSurfer reconstruction, subject surface Schaefer mapping,
   NextBrain segmentation, explicit hybrid labels, five-tissue images and a candidate electrode mask.
4. Human image QC: inspect registration, full-shaft signal void, spurious components and atlas alignment.
   Approval binds the exact reference, mask and atlas hashes. It is never generated automatically.
5. `finish-session`: gray–white-interface seeded iFOD2 ACT tractography, mask exclusion,
   continuous segment/voxel exclusion and reread audit, SIFT2 refit, weight checks, endpoint
   assignment and hemisphere G1–G4.

The default streamline request is 1,000,000, FOD cutoff 0.06, integration step 0.5 mm and
downsampling factor 1. These generalized settings are explicit and **not** a bit-equivalent
replay of the original study scripts. Reduced streamline counts are technical smoke tests.
An independent or training-only gradient reference must be supplied for aligned comparison.
Without one, the workflow produces **UNALIGNED_SINGLE_SESSION** coordinates only.
It does not automatically build a LOSO reference or certify longitudinal scan matching.

## Installed resources

FSL, MRtrix3, ANTs, FreeSurfer and the full NextBrain atlas are installed separately.
No MRI tool, model weight, FreeSurfer license or third-party atlas is redistributed.
Only tools used by the selected route are required; this route does not call MATLAB/SPM12.
FreeSurfer resources are read from the operator's trusted installation. Subject outputs go
to a newly created private `subjects/analysis_subject`, never an existing patient subject.

Set the FreeSurfer installation, fsaverage directory, LH/RH Schaefer400 seven-network annotations,
NextBrain full-atlas directory and an expert-reviewed common node TSV in the GUI.
The installed NextBrain `segment.py` command adapter was inspected in FreeSurfer 8.0.0;
other versions require verification. Its internal shell calls require processing/resource paths
without whitespace or shell metacharacters; the adapter refuses incompatible paths.

The node TSV has `index,label,name,hemisphere,tissue,network,source_label` columns (tab separated).
Index is zero-based; output labels are consecutive 1..N. Cortical source labels are Schaefer
1..200 LH and 201..400 RH, checked against actual annotation order. Noncortical source labels
are original NextBrain IDs and must be explicitly selected by an expert. No tissue class is
inferred from a guessed anatomical name. Cortex takes precedence. Overlapping or absent
assignments cause refusal. The study's 493-node basis is **not** a universal new-cohort default.
The software does not silently discard missing parcels per visit.

## Candidate masks and remaining scientific limits

CT input must have reviewed HU scaling. The candidate detector operates on CT registered into
the DWI grid, identifies elongated high-intensity components and stops unless the number
matches the declared one or two leads. The 2000-HU starting value is configurable and unvalidated
across scanners. The mask is a 3.5-mm-radius piecewise-line tube with one six-connected dilation
on a 2-mm grid. It does **not** follow the study's measured b0 void, guarantee full-shaft coverage,
identify contacts, establish a target nucleus or prove MRI safety.

If candidates are ambiguous, supply reviewed centerlines or use the Advanced prepared-input
workflow with a separately validated mask. Longitudinal work needs registered union masks;
the existing union operation accepts already registered masks with a hash-bound registration
attestation. Automatic longitudinal registration and mask-union orchestration remain unimplemented.
After MRtrix exclusion, a continuous segment/closed-voxel-box filter removes entire streamlines
that still intersect the mask between sampled vertices. Retained vertices and their order are
not changed. The final tractogram is reread using the same predicate before SIFT2 is fitted.
This predicate is independent of MRtrix's sampled mask lookup; the filter and reread audit are
not two independent algorithms. Any remaining intersection, sparse/invalid geometry or empty
tractogram aborts the run. These checks validate only the supplied geometric mask, not its
coverage of susceptibility artifacts. The stricter exclusion is not established as equivalent
to the study pipeline.

`strict_exclusion.json` records additional removals and `exclusion_audit.json` the final count.
`sift2_audit.json` records the weight count, sum, zero-weight count, mu and file hashes.
Weight count must match the final tractogram; weights must be finite, nonnegative and not all
zero; mu must be finite and positive. The matrix uses the fitted SIFT2 weights **without**
additional mu scaling. Mu is retained for explicit downstream analysis, not silently applied.

## Validation boundaries

The local synthetic suite covers input safety, label mapping, tissue normalization, CT candidate
ambiguity, between-vertex mask crossings and artifact handoff through gradient estimation.
The full continuation integration test **mocks external MRI tools** while running the independent
mask audit and gradient routines; it tests wiring, not biological validity or tool interoperability.
The separate opt-in `examples/verify_installed_mrtrix.py` runs real MRtrix exclusion, SIFT2 and
matrix construction plus BrainSpace on artificial prepared data. It takes no patient input,
creates no human approval and uses one thread. Run it only with a new output root outside the
source checkout. This tests downstream numerical interoperability, not ACT generation or MRI
reconstruction; its isotropic synthetic FOD is not a biological model.
A subsequent single local postoperative NIfTI run completed diffusion preprocessing, WM/CSF
FOD estimation and T1-to-DWI rigid registration using installed MRI tools. Source hashes,
modification times and permissions were unchanged; numerical geometry checks and selected
registration views were reviewed. Anatomical segmentation and the full downstream patient
chain are not yet validated. No independent full raw-image-to-gradient validation or
complete manuscript replay has been performed. Do not cite this package as having generated
the reported clinical results or as a validated biomarker implementation.

## Minimal CLI entry points

```sh
dbs-connectome import-nifti --selection /private/config/nifti.json --output-root /private/results
dbs-connectome reconstruct --import-run /private/results/IMPORT_RUN --choices /private/config/acquisition.json --threads 12 --output-root /private/results
dbs-connectome prepare-session --reconstruction-run /private/results/RECON_RUN --settings /private/config/anatomy.json --threads 12 --output-root /private/results
dbs-connectome finish-session --prepared-run /private/results/PREP_RUN --approval /private/results/APPROVAL_RUN/approval.json --reference-run /private/results/TRAINING_REFERENCE --threads 12 --output-root /private/results
```

External stages default to a plan; add `--execute` only after input review and resource approval.
The GUI writes equivalent configurations from its controls. See the example schemas in `examples/`.
