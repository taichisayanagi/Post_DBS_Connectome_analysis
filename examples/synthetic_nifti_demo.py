"""Create artificial intake files only; NOT a biological or reconstruction phantom."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from dbs_connectome.provenance import new_run, write_json

p = argparse.ArgumentParser()
p.add_argument('--output-root', required=True)
args = p.parse_args()
out = new_run(args.output_root, 'synthetic_nifti')
rng = np.random.default_rng(22)
shape = (24, 24, 24)
x = np.indices(shape).astype(float)
radius = np.sqrt(((x - 11.5)**2).sum(0))
brain = (radius < 10).astype(np.float32)
t1 = brain * (100 + 20 * rng.random(shape))
dwi = np.stack([t1 * (.6 + .4*rng.random(shape)) for _ in range(33)], -1).astype(np.float32)
affine = np.diag([2., 2., 2., 1.])
nib.save(nib.Nifti1Image(t1, affine), out / 'T1.nii.gz')
nib.save(nib.Nifti1Image(dwi, affine), out / 'dwi.nii.gz')
vectors = rng.normal(size=(3, 32)); vectors /= np.linalg.norm(vectors, axis=0)
np.savetxt(out / 'dwi.bvec', np.column_stack([np.zeros(3), vectors]))
np.savetxt(out / 'dwi.bval', np.r_[0, np.full(32, 1000)])
write_json(out / 'dwi.json', {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': .08,
                            'Synthetic': True, 'Notice': 'Artificial intake test, not realistic DWI'})
print(out)
