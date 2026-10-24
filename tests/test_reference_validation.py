import unittest
import numpy as np

from dbs_connectome.reference_validation import loso_gallery
from dbs_connectome.study_metrics import identification


class ReferenceValidationTests(unittest.TestCase):
    def test_complete_gallery_held_out_reference(self):
        rng=np.random.default_rng(410)
        matrices=[]
        for s in range(3):
            a=rng.uniform(.5,10,(32,32));a=(a+a.T)/2;np.fill_diagonal(a,0)
            visits=[]
            for t in range(3):
                v=a+rng.uniform(0,.005,(32,32));v=(v+v.T)/2;np.fill_diagonal(v,0);visits.append(v)
            matrices.append(visits)
        nodes=[{'tissue':'cortex','hemisphere':'L' if i<16 else 'R'} for i in range(32)]
        result=loso_gallery(np.asarray(matrices),nodes)
        self.assertEqual(result['correct'],9)
        self.assertEqual(len(result['nearest']),9)

    def test_identification_summaries_ignore_self_diagonal(self):
        rng=np.random.default_rng(5);base=rng.normal(size=(3,20));f=np.repeat(base,3,axis=0)+rng.normal(size=(9,20))*.01
        r,s=identification(f,3)
        self.assertTrue(np.isfinite([r[k] for k in ('margin_mean','margin_min','within_mean','between_mean')]).all())
        self.assertGreater(r['margin_min'],0)
        self.assertEqual(r['null_count'],36)
