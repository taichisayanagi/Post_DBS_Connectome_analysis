import unittest
import numpy as np
from dbs_connectome.representations import visit_excluded_identification,change_concordance

class RepresentationTests(unittest.TestCase):
    def test_visit_filter_and_reference_identity(self):
        rng=np.random.default_rng(71);base=rng.normal(size=(4,20))
        x=np.repeat(base,3,axis=0)+rng.normal(0,.01,(12,20))
        subjects=np.repeat(np.arange(4),3);visits=np.tile(np.arange(3),4)
        for z in (False,True):
            result=visit_excluded_identification(x,subjects,visits,z)
            self.assertEqual(result['acc'],1)
            self.assertTrue(all(visits[i]!=visits[j] for i,j in enumerate(result['nearest'])))

    def test_concordance_is_rotation_invariant_and_deterministic(self):
        rng=np.random.default_rng(4);g=rng.normal(size=(4,3,20,4));hemi=np.array(['L']*10+['R']*10)
        a,rows,null=change_concordance(g,hemi,['A','B','C','D'],100,11)
        q,_=np.linalg.qr(rng.normal(size=(4,4)))
        b,_,other=change_concordance(g@q,hemi,['A','B','C','D'],100,11)
        self.assertAlmostEqual(a['observed_mean_node_weighted_rho'],b['observed_mean_node_weighted_rho'])
        np.testing.assert_allclose(null,other,atol=1e-14)
        self.assertEqual(len(rows),6)
        self.assertTrue(0<a['one_sided_p_for_positive_concordance']<=1)
