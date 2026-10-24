import unittest
import numpy as np
from dbs_connectome.comparative import (NETWORKS, agreement_icc, matched_metrics,
                                       rank_correlation, replication_summary, calibre_summary,
                                       exact_parcel_mapping, normalization_margins)


class ComparativeTests(unittest.TestCase):
    def fixture(self, axes=4):
        rng = np.random.default_rng(15)
        m = rng.random((4, 3, 28, 28)); m += m.transpose(0, 1, 3, 2)
        for x in m.reshape(-1, 28, 28): np.fill_diagonal(x, 0)
        g = rng.normal(size=(4, 3, 28, axes))
        nodes = [{'network': net} for net in NETWORKS for _ in range(4)]
        return m, g, nodes, np.arange(28)

    def test_icc_absolute_agreement_penalizes_offset(self):
        x = np.arange(6.)
        self.assertAlmostEqual(agreement_icc(np.column_stack([x, x, x])), 1)
        self.assertLess(agreement_icc(np.column_stack([x, x+10, x+20])), .1)
        self.assertTrue(np.isnan(agreement_icc(np.ones((3, 3)))))

    def test_metrics_match_direct_definition(self):
        m, g, nodes, order = self.fixture(10)
        metrics = matched_metrics(m, g, nodes, order)
        self.assertEqual([len(x) for x in metrics.values()], [28, 28, 28])
        a, b = g[:, 1], g[:, 2]
        expected = np.median(np.linalg.norm(b[:, :4]-b.mean(1)[:, None], axis=2), axis=1)-np.median(np.linalg.norm(a[:, :4]-a.mean(1)[:, None], axis=2), axis=1)
        np.testing.assert_allclose(metrics['gradient']['Vis_ecc'], expected)
        w = np.log1p(m)
        np.testing.assert_allclose(metrics['edge']['Vis-Vis'], w[:, 2, :4, :4].mean((1, 2))-w[:, 1, :4, :4].mean((1, 2)))
        with self.assertRaises(ValueError):
            matched_metrics(np.zeros_like(m), g, nodes, order)

    def test_replication_reports_constants_and_signs(self):
        metrics = matched_metrics(*self.fixture())
        result, rows = replication_summary(metrics, metrics, np.arange(4), np.arange(4))
        for value in result.values():
            self.assertEqual(value['sign_agree'], 1)
            self.assertAlmostEqual(value['rho_corr'], 1)
        self.assertTrue(any(not row['included'] for row in rows))  # dense graph degree is constant

    def test_calibre_keys_and_dimension_difference_are_explicit(self):
        m, g, nodes, order = self.fixture(10)
        ten = matched_metrics(m, g, nodes, order)
        four = matched_metrics(m, g[:, :, :, :4], nodes, order)
        self.assertFalse(np.allclose(ten['gradient']['Vis_ecc'], four['gradient']['Vis_ecc']))
        summary, rows = calibre_summary({'a': four, 'b': four, 'c': four}, np.arange(4))
        self.assertEqual(summary['gradient']['n'], 14)
        self.assertEqual(summary['node']['n'], 7)
        self.assertAlmostEqual(summary['gradient']['icc_med'], 1)
        self.assertEqual(summary['gradient']['span_med'], 0)
        self.assertEqual(len(rows), 63)

    def test_rank_correlation_does_not_impute(self):
        self.assertTrue(np.isnan(rank_correlation([1, 1, 1], [1, 2, 3])))
        with self.assertRaises(ValueError): rank_correlation([1, np.nan, 3], [1, 2, 3])

    def test_name_mapping_requires_matching_networks(self):
        nodes = [{'name': 'A', 'network': 'Vis', 'hemisphere': 'L', 'tissue': 'cortex'},
                 {'name': 'B', 'network': 'Cont', 'hemisphere': 'R', 'tissue': 'cortex'}]
        a, b, mapping = exact_parcel_mapping(nodes, nodes[::-1])
        np.testing.assert_array_equal(a, [0, 1]); np.testing.assert_array_equal(b, [1, 0])
        np.testing.assert_array_equal(mapping, [0, 1])
        wrong = [dict(nodes[0], network='Default'), nodes[1]]
        with self.assertRaises(ValueError): exact_parcel_mapping(nodes, wrong)

    def test_normalization_variants_are_exported(self):
        m, g, _, order = self.fixture()
        result = normalization_margins(m, g, order)
        self.assertEqual(set(result), {'G1', 'G1-G4', 'node_strength', 'edges_log1p'})
        for metrics in result.values(): self.assertEqual(set(metrics), {'z', 'rank', 'center'})
