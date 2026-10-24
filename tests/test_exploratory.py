import unittest
import numpy as np
from dbs_connectome.exploratory import exploratory_metrics,historical_variance_ratio,pooled_inventory
from dbs_connectome.comparative import NETWORKS

class ExploratoryTests(unittest.TestCase):
    def test_inventory_and_non_icc(self):
        rng=np.random.default_rng(16)
        nodes=[{'name':f'{h}_{net}_{j}','hemisphere':h,'network':net,'tissue':'cortex'} for h in ('L','R') for net in NETWORKS for j in range(2)]
        nodes += [{'name':name,'tissue':'subcortex'} for name in ('thalamus','putamen','caudate','pallidum','accumbens')]
        n=len(nodes);matrices=rng.random((4,3,n,n));g=rng.normal(size=(4,3,28,4));lam=rng.random((4,3,2,10))
        metrics,excluded,structures=exploratory_metrics(g,lam,matrices,nodes,np.arange(28),['L','L','R','R'])
        self.assertEqual(len(metrics),148);self.assertEqual(len(structures),5);self.assertEqual(excluded,[])
        values=metrics['putamen_cortex_absolute']['values']
        np.testing.assert_array_equal(values[:,0],values[:,1])
        self.assertEqual(historical_variance_ratio(np.ones((4,3))),0)
        self.assertTrue(0<=historical_variance_ratio(g[:,:,0,0])<=1)
        full,_,_=pooled_inventory(g,lam,nodes,np.arange(28),True)
        reduced,_,_=pooled_inventory(g,lam,nodes,np.arange(28),False)
        self.assertEqual(len(full),59)  # no posterior-attention name in this artificial atlas
        self.assertEqual(len(reduced),45)
        self.assertFalse(np.allclose(full['Vis_SomMot_distance']['values'],reduced['Vis_SomMot_distance']['values']))
