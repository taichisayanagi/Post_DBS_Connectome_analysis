import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dbs_connectome.sensitivity import candidate_values, match_axes, setting_family
from dbs_connectome.study import expected_checks
from dbs_connectome.study_metrics import interval_metrics, session_metrics
from dbs_connectome.provenance import digest


class SensitivityTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(81)
        g=rng.normal(size=(5,3,36,6))
        nets=np.tile(np.repeat(['Vis','SomMot','Cont'],6),2)
        hemis=np.repeat(['L','R'],18)
        return g,nets,hemis

    def test_primary_candidates_match_general_metrics(self):
        g,nets,hemis=self.fixture()
        values=candidate_values(g,nets,hemis,{0:0,1:1,2:2,3:3})
        for i in range(len(g)):
            a=session_metrics(g[i,1,:,:4],nets,hemis)
            b=session_metrics(g[i,2,:,:4],nets,hemis)
            d=interval_metrics(g[i,1,:,:4],g[i,2,:,:4],nets,hemis)
            self.assertAlmostEqual(values['Vis_parcel_displacement_mean'][i],d['Vis','parcel_displacement_mean'])
            for name,net,metric in [('SomMot_eccentricity_median','SomMot','eccentricity_median'),('Cont_G2_position_median','Cont','G2_position_median'),('Vis_G1_dispersion_sd','Vis','G1_dispersion_sd')]:
                self.assertAlmostEqual(values[name][i],b[net,metric]-a[net,metric])

    def test_one_to_one_mapping_and_no_sign_flip(self):
        g,nets,hemis=self.fixture()
        primary=g[:,:,:,[4,5,0,1]]*np.array([-1,1,1,-1])
        mapping,c=match_axes(g,primary)
        self.assertEqual(mapping,{0:4,1:5,2:0,3:1})
        self.assertAlmostEqual(c[4,0],-1)
        self.assertEqual(len(setting_family()),8)

    def test_generic_frozen_table_column_mapping(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'expected.csv';p.write_text('id,score_old\nA,1.25\nB,2.5\n')
            check={'kind':'table_csv','actual':'table','path':str(p),'sha256':digest(p),
                   'keys':['subject'],'key_map':{'subject':'id'},'columns':['score'],'column_map':{'score':'score_old'}}
            rows=[{'subject':'A','score':1.25},{'subject':'B','score':2.5}]
            result=expected_checks({'expected':[check]},{'table':rows},[])
            self.assertTrue(result[0]['passed'])
            rows[0]['score']=4
            self.assertFalse(expected_checks({'expected':[check]},{'table':rows},[])[0]['passed'])

    def test_missing_network_rejected(self):
        g,nets,hemis=self.fixture();nets[nets=='Vis']='Other'
        with self.assertRaises(ValueError):candidate_values(g,nets,hemis,{0:0,1:1})
