import unittest
import numpy as np
from dbs_connectome.lateralized import session_geometry, symptom_value, correlation_null, adjusted

class LateralizedTests(unittest.TestCase):
    def test_session_geometry_translation_and_rotation(self):
        rng = np.random.default_rng(1); g = rng.normal(size=(20,4)); sel=np.arange(20)<6; whole=np.ones(20,bool)
        a=session_geometry(g,sel,whole); b=session_geometry(g+13,sel,whole)
        for k in a:self.assertAlmostEqual(a[k],b[k])
        q,_=np.linalg.qr(rng.normal(size=(4,4))); c=session_geometry(g@q,sel,whole)
        for k in a:
            if not k.startswith('G'):self.assertAlmostEqual(a[k],c[k])

    def test_signed_ratio_not_defined(self):
        a={'delta':2,'post1':2,'post2':4};b={'delta':1,'post1':2,'post2':3}
        ratio='contralateral_to_ipsilateral_change_ratio'
        self.assertTrue(np.isnan(symptom_value(a,b,'G1_position_mean',True,ratio)))
        self.assertAlmostEqual(symptom_value(a,b,'eccentricity_mean',True,ratio),.75)
        self.assertEqual(symptom_value(a,b,'parcel_displacement_mean',True,ratio),.5)
        self.assertEqual(symptom_value(a,b,'eccentricity_mean',False,'contralateral_minus_ipsilateral'),1)

    def test_exact_null_floor_and_family(self):
        rho,p,null=correlation_null(np.arange(6),np.arange(6))
        self.assertAlmostEqual(rho,1);self.assertAlmostEqual(p,2/720)
        rows=[{'family':'a','rho':rho,'p':p},{'family':'a','rho':rho,'p':p}]
        adjusted(rows,[null,null],'p','rho','test',('family',))
        self.assertEqual(rows[0]['p_maxT_test'],p)
        with self.assertRaises(ValueError):adjusted(rows,[null,null[:-1]],'p','rho','bad',('family',))
