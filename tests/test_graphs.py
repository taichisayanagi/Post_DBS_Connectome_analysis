import unittest
import numpy as np
from dbs_connectome.graphs import fixed_cost, graph_features, node_edge_change
from dbs_connectome.nextbrain_runner import serial_loader


class GraphTests(unittest.TestCase):
    def test_density_ties_are_exact_and_deterministic(self):
        m=np.ones((20,20))-np.eye(20)
        a,b=fixed_cost(m,.1)
        self.assertEqual(a.sum()/2,19)
        np.testing.assert_array_equal(a,fixed_cost(m,.1)[0])
        self.assertTrue(np.all(b[a>0]==10/19))

    def test_refuse_zero_padding(self):
        with self.assertRaises(ValueError):fixed_cost(np.zeros((20,20)),.1)

    def test_disconnected_efficiency_and_lcc(self):
        m=np.zeros((5,5));m[0,1]=m[1,0]=3;m[1,2]=m[2,1]=2
        metrics=graph_features(m,(.2,),modularity=False)
        self.assertAlmostEqual(metrics['binary_global_efficiency_cost20'],.25)
        self.assertAlmostEqual(metrics['largest_component_fraction_cost20'],.6)
        self.assertAlmostEqual(metrics['binary_characteristic_path_lcc_cost20'],4/3)

    def test_node_zero_baseline_not_infinity(self):
        a=np.zeros((3,3));b=a.copy();b[0,1]=b[1,0]=2
        nodes=[dict(name=str(i),network='A',hemisphere='L') for i in range(3)]
        n,e=node_edge_change(a,b,nodes)
        self.assertIsNone(n[0]['strength_percent']);self.assertIsNone(e[0]['percent'])
        self.assertEqual(n[0]['strength_change'],2)

    def test_nextbrain_wrapper_keeps_numerical_settings_separate(self):
        def fake(*args,**kwargs):return args,kwargs
        _,kw=serial_loader(fake)([1],num_workers=2,prefetch_factor=2,batch_size=1)
        self.assertEqual(kw['num_workers'],0);self.assertIsNone(kw['prefetch_factor'])
        self.assertEqual(kw['batch_size'],1)


if __name__=='__main__':unittest.main()
