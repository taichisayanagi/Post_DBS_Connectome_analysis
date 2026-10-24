import json
from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import spearmanr

from dbs_connectome.statistical_audit import (family_hit_null, n6_ordering_audit,
    normal_spearman_power, null_critical, rank_numerator, run_power)


class StatisticalAuditTests(unittest.TestCase):
    def test_rank_formula_and_exact_threshold(self):
        x=np.arange(6); rows=np.array(list(permutations(x)))
        actual=rank_numerator(rows,x)/210.
        expected=np.array([spearmanr(x,y).statistic for y in rows])
        np.testing.assert_allclose(actual,expected,atol=1e-15)
        c=null_critical(6,.05,100,12)
        self.assertEqual(c['critical_absolute_numerator'],186)
        self.assertEqual(c['null_draws'],720)
        self.assertAlmostEqual(c['achieved_null_probability'],24/720)

    def test_ordering_and_loo_counts(self):
        v=n6_ordering_audit()
        self.assertEqual(v['groups']['abs_rho_ge_0.942857']['count'],12)
        self.assertEqual(v['groups']['abs_rho_ge_0.885714']['count'],24)
        self.assertEqual(v['groups']['abs_rho_eq_0.828571']['count'],18)
        self.assertEqual(v['weak_abs_rho_lt_06_count'],546)
        self.assertEqual(v['weak_any_strict_sign_flip_count'],370)
        self.assertEqual(v['weak_any_sign_flip_or_zero_count'],416)

    def test_shared_hit_null_preserves_dependence(self):
        x=np.arange(6);rows=np.array(list(permutations(x)))
        null=rank_numerator(rows,x)/210.
        audit=family_hit_null([null,null],[2/720,2/720])
        self.assertAlmostEqual(audit['expected_hits'],2*24/720)
        self.assertAlmostEqual(audit['p_null_ge_observed'],24/720)
        self.assertEqual(audit['histogram'],[{'hits':0,'permutations':696},{'hits':2,'permutations':24}])

    def test_power_reproducible_and_perfect(self):
        a=normal_spearman_power(6,.8,186,1000,31)
        self.assertEqual(a,normal_spearman_power(6,.8,186,1000,31))
        self.assertEqual(normal_spearman_power(6,1.,186,100,31)['power'],1.)
        self.assertEqual(normal_spearman_power(6,-1.,186,100,31)['power'],1.)
        self.assertEqual(null_critical(9,.05,1000,31),null_critical(9,.05,1000,31))

    def test_run_does_not_claim_historical_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs=root/'inputs';inputs.mkdir();out=root/'out';out.mkdir()
            config=inputs/'spec.json';config.write_text(json.dumps({'schema_version':1,'scope':'statistical_design_audit',
                'seed':11,'power_replicates':100,'sample_size_replicates':100,'null_draws_above_n8':100,
                'generating_pearson':[1.],'sample_size_generating_pearson':[1.],'n_max':6}))
            run_power(config,out)
            result=json.loads((out/'power_result.json').read_text())
            self.assertEqual(result['status'],'computed_statistical_design_audit')
            self.assertEqual(result['sample_size_first_estimate_at_target'],{'1.0':6})
            self.assertTrue(result['source_bytes_unchanged'])
            self.assertTrue((out/'power_review.html').is_file())
