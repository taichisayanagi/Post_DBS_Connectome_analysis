# Release gates

Repository owner/name and MIT license were confirmed on 2026-09-15. The GUI is the primary
entry point; the CLI remains the execution backend. A clearly labelled development snapshot
may be shared after source/privacy review and synthetic tests. That is NOT a validated release.
The end-to-end and independent-data gates below apply before a scientific/validated release.

1. Confirm contribution ownership; preserve the approved MIT license for new code.
2. Audit all tracked files AND all Git history for patient identifiers, tokens, scans, matrices,
   logs, machine paths, institutional confidential material and non-redistributable atlas resources.
   Do not initialize this repository by copying the clinical project's Git history.
3. Freeze manuscript-compatible parameters and distinguish legacy, primary and sensitivity profiles.
   A centerline tube must never be advertised as the study's void-follow algorithm.
4. Implement and test DICOM inventory, pairing, gradient directions, reverse-PE policy, transforms,
   lead localization, artifact coverage, surface atlas generation, ACT and acquisition-dependent FOD.
   Do not infer shell model, PE direction or timing from filename heuristics.
5. Test RAS/LPS, sform/qform, voxel/world transformations, anisotropic and oblique acquisitions,
   registered longitudinal unions, isolated nodes, mapping indices, all-zero/malformed inputs,
   interruption recovery, provenance and stale QC approvals.
6. Independently verify that excluded tractograms contain no mask-intersecting streamlines,
   that SIFT2 weights correspond to the final tractogram, and that all inputs share an audited lineage.
7. Reproduce the existing study locally against frozen outputs with predeclared numerical tolerances.
   Never distribute those clinical fixtures without appropriate approval.
8. End-to-end test on independent DICOM series from both sites and at least one new subject/protocol.
   Log unsuccessful runs and manual correction burden, not only successful cases.
9. Quantify mask agreement/coverage, registration error, exclusion fraction, matrix differences,
   gradient subspace stability and longitudinal metric agreement. A unit-test pass is not MRI validation.
10. Test clean installations on macOS arm64 and supported Linux; pin dependencies and test containers.
    FreeSurfer and atlas licenses/weights are not bundled without permission.
11. Verify reproducible synthetic demo and CI. Resolve resource controls and GUI requirements.
12. Tag a release, archive it to obtain a real persistent identifier, then add the ACTUAL URL/version
    and verified validation results to the paper. Do not create a placeholder DOI or claim availability early.

The prototype is research-only. It does not establish MRI device safety, electrode migration,
DBS target engagement, stimulation settings or patient-specific clinical recommendations.
