import itertools
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
from dbs_connectome.geometry_controls import parcel_distances,mask_dice,mantel,donor_null,representation_perturbation
from dbs_connectome.provenance import digest
from dbs_connectome.provenance import write_json
from dbs_connectome.study import expected_checks

class GeometryControlTests(unittest.TestCase):
    def test_perturbation_units_and_identity(self):
        rng=np.random.default_rng(8)
        m=rng.uniform(1,2,(4,9,9));m=(m+m.transpose(0,2,1))/2
        for x in m:np.fill_diagonal(x,0)
        # Vary graph degree so its structure comparison is defined.
        for i in range(4):m[i,i,(i+1)%9]=m[i,(i+1)%9,i]=0
        g=rng.normal(size=(4,9,4));distance=np.tile([10,20,25,35,45,55,65,75,85],(4,1))
        result=representation_perturbation(m,m,g,g,distance)
        self.assertEqual(result['perturbation_to_signal']['G1-G4']['ratio'],0)
        self.assertAlmostEqual(result['structure_preservation']['G1-G4']['r'],1)
        self.assertEqual(result['spatial_error']['gradient']['far'],0)

    def test_empty_frozen_check_cannot_certify_replay(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'expected.json';write_json(p,{'value':1})
            spec={'expected':[{'kind':'numbers_json','path':str(p),'sha256':digest(p),'actual':'result','paths':[]}]}
            with self.assertRaises(ValueError):expected_checks(spec,{'result':{'value':1}},[])

    def test_world_distance_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a=np.zeros((5,5,5),np.int16);a[3,1,1]=7;a[1,4,1]=9
            m=np.zeros_like(a);m[1,1,1]=1
            def save(name,data,affine):
                p=root/name;nib.save(nib.Nifti1Image(data,affine),p)
                return {'path':str(p),'sha256':digest(p)}
            atlas=save('atlas.nii',a,np.diag([2.,3.,4.,1.]));mask=save('mask.nii',m,np.diag([2.,3.,4.,1.]))
            np.testing.assert_allclose(parcel_distances(atlas,mask,[9,7]),[9,4])
            self.assertEqual(mask_dice(mask,mask),1)
            shifted=save('shift.nii',m,np.eye(4))
            with self.assertRaises(ValueError):mask_dice(mask,shifted)
            with self.assertRaises(ValueError):parcel_distances(atlas,mask,[8])
            with self.assertRaises(ValueError):parcel_distances(atlas,mask,[7,7])
            mask['sha256']='stale'
            with self.assertRaises(ValueError):parcel_distances(atlas,mask,[7])

    def test_mantel_against_exhaustive_oracle(self):
        a=np.array([[0.,.2,.8],[.2,0,.5],[.8,.5,0]])
        b=np.array([[0.,.3,.9],[.3,0,.1],[.9,.1,0]])
        got=mantel(a,b);iu=np.triu_indices(3,1)
        null=[spearmanr(a[np.ix_(p,p)][iu],b[iu]).statistic for p in itertools.permutations(range(3))]
        obs=spearmanr(a[iu],b[iu]).statistic
        self.assertEqual(got['n_permutations'],6)
        self.assertAlmostEqual(got['p_one_sided'],sum(v>=obs-1e-12 for v in null)/6)

    def test_donor_no_self_and_exact_null(self):
        delta=np.array([[0.,.1,.3],[.8,0,.1],[.4,.2,0]])
        donors=[2,0,1];got=donor_null(delta,donors)
        def stat(ds):
            return np.mean([delta[i,d]-np.mean([delta[i,j] for j in range(3) if j not in (i,d)]) for i,d in enumerate(ds)])
        values=[stat(p) for p in itertools.product([1,2],[0,2],[0,1])]
        self.assertEqual(got['n_relabelings'],8)
        self.assertAlmostEqual(got['obs_pull_toward_donor'],stat(donors))
        self.assertAlmostEqual(got['p_one_sided'],sum(v>=stat(donors)-1e-15 for v in values)/8)
        with self.assertRaises(ValueError):donor_null(delta,[0,1,2])
