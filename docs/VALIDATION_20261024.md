# Validation record — 2026-10-24 JST, v0.2.0a1

This is a **research-preview acceptance record**, not clinical validation or a full patient
DICOM-to-gradient validation. No patient images were used for the new tests recorded here.

| Check | Observed result | Scope / exclusions |
|---|---|---|
| Python unit/integration/interface suite | 113 passed on macOS arm64 / Python 3.12 | Some external reconstruction commands are mocked; not whole-pipeline MRI validation |
| CT + MRI synthetic editor | Two shaft candidates, connected void, manual-edit guards, unchanged source bytes/mtime/mode | Artificial anatomy; no independent clinical mask agreement |
| Corrected mask handoff | New preparation, old input unchanged, stale QC refused | Synthetic preparation |
| MRtrix downstream | Actual tckedit → continuous audit/filter → tcksift2 → tck2connectome → BrainSpace completed | 752 synthetic tracks, 750 retained, 750 weights; no DICOM reconstruction or actual tckgen/ACT generation in this fixture |
| OSS-DBS voltage mode | Actual 0.5.9 FEM, field NIfTI, binary threshold NIfTI, HTML viewer | One lead, coarse homogeneous phantom, 1 thread; no mesh/domain convergence study |
| OSS-DBS current mode | Same output contract completed; mA→A and return-current balance tested | Same phantom; not patient VTA validation |
| Common-reference pairing | Mean verified numerically; signed G1–G4 deltas and 4D displacement match direct arrays | Synthetic completed-connectome fixtures; no axis-identity or clinical validation |
| Browser interaction | CT/MRI candidate generated and handed to main workflow; external FEM queued and completed | Synthetic input only; not expert clinical QC |
| Wheel packaging | CLI version and all five HTML assets verified in a separate venv outside checkout | Existing scientific site-packages reused; not a clean dependency-isolated installation |

## Reproduce

```sh
python -m unittest discover -s tests -v
python examples/verify_installed_mrtrix.py --output-root /private/checks/mrtrix
python examples/verify_gui_release.py --output-root /private/checks/editor
python examples/verify_reports.py --output-root /private/checks/reports
OSSDBS_PYTHON=/path/to/solver/bin/python python examples/verify_gui_release.py \
  --output-root /private/checks/voltage --simulate
OSSDBS_PYTHON=/path/to/solver/bin/python python examples/verify_gui_release.py \
  --output-root /private/checks/current --simulate --current
```

Use different output parents or timestamps when rerunning. Existing runs are never overwritten.
The optional FEM example uses coarse numerical-test settings; they are not clinical recommendations.
Reports and full solver logs stay outside the repository. All newly generated example inputs are
artificial and deterministically generated in code, not de-identified patient data.

## Remaining evidence gaps

- Independent patient DICOM/NIfTI → segmentation/ACT → final connectome/gradient completion.
- Exact agreement with the study's original artifact algorithm and frozen analysis outputs.
- Multi-vendor acquisitions, severe metal artifacts, failed registrations and manual correction burden.
- Longitudinal mask registration/union review, same-node coverage and independent repeat-scan reliability.
- Physical contact localization, directional rotation, FEM convergence and validated neural activation.
- Cross-platform external MRI/solver compatibility beyond the specific local checks above.

Do not turn this record into a statement that all input protocols or patients are supported.
