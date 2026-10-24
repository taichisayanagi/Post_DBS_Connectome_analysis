# Optional electric-field / VTA-like simulation

## Scope and installation

This MIT application launches the separately installed **OSS-DBSv2** solver. It does not bundle
OSS-DBS, NGSolve, NEURON, electrode geometry or their licenses. Follow upstream installation and
license terms in an independent environment. Do not install it into an existing clinical/scientific
environment without reviewing dependency changes. Adapter smoke-tested with PyPI `ossdbs==0.5.9`
on macOS arm64 / Python 3.12 on 2026-10-24. Other versions are not validated by this release.

```sh
export OSSDBS_PYTHON=/absolute/path/to/ossdbs_environment/bin/python
dbs-connectome gui --data-root /private/study_inputs --output-root /private/study_results
```

The interpreter path is chosen by the local operator before starting the server, not executable
code submitted by the browser. The main application never auto-installs or upgrades the solver.

## Required independent review

- Segmentation labels: 0 unknown/outside, 1 CSF, 2 WM, 3 GM, 4 blood. A raw MRI, parcel atlas or
  electrode-exclusion mask is not an acceptable conductivity tissue map. A 5TT argmax candidate can
  be generated from a loaded preparation; pathology maps to unknown, not blood. Coarse DWI-resolution
  segmentation may be inappropriate for a contact-scale FEM model and needs separate review.
- Explicit lead model, physical tip and a shaft-direction point in the segmentation's RAS-mm space.
  Neither the end of the CT artifact nor the end of MRI void establishes the physical tip.
- Exact mapping between clinical contact numbers and OSS-DBS's **1-based contact order**. The IPG
  name does not identify the lead. Directional models additionally require independently verified
  axial rotation; this cannot be recovered from a shaft centerline alone.
- Explicit voltage (V) or current (mA) amplitudes per contact. Current converts to A internally.
  In voltage mode, zero entries are inactive floating contacts, not grounded contact returns.
  Current-mode contacts are floating current sources/sinks; the outer boundary balances current.

## What the model actually computes

One lead, grounded spherical outer boundary, isotropic tissue properties using OSS-DBS ColeCole4,
quasi-static FEM at one user-selected **field evaluation frequency**. That frequency is **not the
DBS pulse repetition rate**. A regular lattice samples electric-field magnitude; threshold converts
V/mm to V/m for the solver and back for the viewer. The binary thresholded field is labelled
**VTA-like**, not demonstrated neural recruitment. NIfTI volumes, solver VTK/geometry files,
settings, engine version, commands and hashes remain in the private timestamped output directory.

No axon activation, pulse-width dependence, anisotropic conductivity, encapsulation layer,
multi-lead interaction, waveform optimization or automatic programming is implemented by this
adapter. Mesh size, lattice size, boundary placement and tissue labeling can materially change the
result. Mesh/domain convergence must be assessed independently. The reported binary voxel volume
can differ from the solver's mesh-integrated volume; neither is a validated patient VTA measurement.

The segmentation must cover the domain bounding box. The displayed field covers only the selected
regular lattice, not the whole brain. Empty thresholded output is possible and is not automatically
an error. Software amplitude bounds are input checks, not clinical recommendations.

**Artifact-exclusion mask ≠ physical electrode ≠ electric field ≠ VTA-like threshold.**
Do not substitute one for another, infer an anatomical tip automatically, or use these outputs for
patient treatment decisions or MRI-device safety decisions.

## Upstream interface sources checked 2026-10-24

- [Input settings](https://oss-dbsv2.readthedocs.io/en/latest/input_settings.html)
- [Electrodes](https://oss-dbsv2.readthedocs.io/en/latest/electrodes.html)
- [Point analysis](https://oss-dbsv2.readthedocs.io/en/latest/point_analysis.html)
- [Volume conductor](https://oss-dbsv2.readthedocs.io/en/latest/volume_conductor_model.html)
- [Tutorial](https://oss-dbsv2.readthedocs.io/en/latest/tutorial.html)
- [Upstream source](https://github.com/SFB-ELAINE/OSS-DBSv2)

Configuration keys and units were checked against documentation and installed 0.5.9 source and
then exercised in actual voltage/current homogeneous-phantom FEM runs. These sources document the
solver interface; they do not validate this application's patient-specific results.
