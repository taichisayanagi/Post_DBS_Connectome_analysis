# Electrode localization and exclusion editor

## Inputs and coordinate contract

Reference b0, registered CT, registered MRI/b0 and registered brain mask must share grid and anatomy.
Identical affine matrices are not evidence of registration. The GUI shows canonical RAS+ voxel planes;
oblique planes remain oblique, not resampled anatomical orthogonal planes. Point coordinates are
NIfTI RAS millimetres. Displayed image hashes bind every generation request to the loaded images.
Study-style dilation currently requires an isotropic, non-sheared 2-mm reference grid.

## Candidate algorithm

1. Automatic CT mode detects elongated high-HU intracranial components with explicit expected lead
   count. Ambiguous candidates stop. Edited/MRI-only mode uses operator-provided shaft polylines.
2. A tube around each polyline forms a conservative geometric seed. CT-thresholded voxels are
   retained within the selected shaft corridor; these are metal-artifact candidates, not an exact
   estimate of the physical metal boundary or a reconstructed contact array.
3. Optional MRI extraction identifies dark voxels relative to a positive-intensity local annulus
   around the corridor. Only 6-connected components inside the corridor and brain mask touching
   the tube are retained. The dark/surround threshold and corridor are explicit. Native dark anatomy,
   distortion, registration errors and coil artifacts can still be mistaken for or missed as void.
4. Union of tube, CT candidate and MRI void receives explicit 6-connected dilation. The final mask,
   metal candidate and void are saved separately. Boundary-reaching void triggers a warning.
5. Manual sphere additions/erasures are applied after dilation. Erasure requires an audit reason;
   removal of detected metal/void is flagged. Point edits and mask edits create new outputs, never
   overwrite source images. They do not imply that erased signal is valid.

Virtual mode creates geometric exclusion only and cannot infer a real MRI void. It must not be
presented as the implanted electrode or as a biophysical stimulation model.

## Review and downstream handoff

The orange editor lines are projected centerlines, not the final voxel-mask boundary. Generate and
open the mask report after edits. Inspect registration, complete signal-void coverage, the whole shaft,
spurious components and atlas alignment. The report alone never grants approval.
Adopting a candidate into an existing anatomical preparation creates a new preparation and forces
fresh hash-bound QC. Downstream tractography is regenerated and its final exclusion audited before SIFT2.

This is a new candidate algorithm with synthetic validation, not a proven replay of the paper's
original void-follow implementation. Thresholds/radii are not universal artifact boundaries. It
does not estimate physical contact centers, tip location or directional rotation. Those must be
independently determined for stimulation modeling.
