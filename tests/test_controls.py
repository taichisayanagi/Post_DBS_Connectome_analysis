import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dbs_connectome.controls import icc21,paired_signflip,control_matrix
from dbs_connectome.provenance import digest


class ControlTests(unittest.TestCase):
    def test_icc_identity_and_systematic_bias(self):
        x=np.arange(8,dtype=float)
        self.assertAlmostEqual(icc21(x,x),1)
        self.assertLess(icc21(x,x+10),.2)
        self.assertLess(icc21(x,-x),0)
        self.assertTrue(np.isnan(icc21(np.ones(5),np.ones(5))))

    def test_signflip_independent_enumeration(self):
        a=np.array([2,4,5,9.]);b=np.array([1,2,4,3.])
        got=paired_signflip(a,b)
        sums=[sum((a-b)*s) for s in itertools.product([-1,1],repeat=4)]
        self.assertEqual(got['p_greater'],sum(v>=sum(a-b) for v in sums)/16)
        self.assertEqual(got['n_signflips'],16)

    def test_control_hash_and_mapping(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'matrix.npy';m=np.ones((5,5));np.fill_diagonal(m,0);np.save(p,m)
            item={'matrix':str(p),'sha256':digest(p),'mask_scheme':'none','matrix_indices':[1,3,4]}
            got=control_matrix(item,[{}, {}, {}]);self.assertEqual(got.shape,(3,3))
            item['matrix_indices']=[1,1,4]
            with self.assertRaises(ValueError):control_matrix(item,[{}, {}, {}])
            item['sha256']='changed'
            with self.assertRaises(ValueError):control_matrix(item,[{}, {}, {}])
