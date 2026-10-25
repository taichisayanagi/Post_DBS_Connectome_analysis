"""Synthetic-only boundary/fixture tests. External tools tested separately, without mocks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from dbs_connectome.cli import parser
from dbs_connectome.gradients import load_nodes
from dbs_connectome.gui import JobManager
from dbs_connectome.provenance import read_json
from dbs_connectome.synthetic import build_fixture, run_synthetic, verify_dicom_conversion, verify_matrix_weights


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_fixture_has_all_nodes_and_valid_tissues(self):
        files = build_fixture(self.root/'inputs')
        atlas = nib.load(files['atlas']).get_fdata()
        self.assertEqual(set(np.unique(atlas)), set(range(33)))
        self.assertEqual(len(load_nodes(self.root/'inputs/nodes.tsv')), 32)
        five = nib.load(files['five_tissue']).get_fdata()
        brain = nib.load(files['brain']).get_fdata() > 0
        np.testing.assert_array_equal(five.sum(-1), brain)
        self.assertEqual(len(list((self.root/'inputs').rglob('*.dcm'))), 48)
        self.assertTrue(read_json(self.root/'inputs/SYNTHETIC_ONLY.json')['synthetic_only'])

    def test_conversion_check_accepts_lossless_reorientation_and_rejects_translation(self):
        files = build_fixture(self.root/'inputs')
        image = nib.load(files['CT'])
        moved = image.as_reoriented(np.array([[0,-1],[1,1],[2,1]]))
        target = self.root/'reoriented.nii.gz'
        nib.save(moved, target)
        verify_dicom_conversion(files['CT'], target)
        affine = image.affine.copy()
        affine[0,3] += 2
        bad = self.root/'shifted.nii.gz'
        nib.save(nib.Nifti1Image(image.get_fdata(), affine), bad)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            verify_dicom_conversion(files['CT'], bad)

    def test_plan_never_executes_tools_or_claims_completion(self):
        out = self.root/'run'
        out.mkdir()
        with patch('dbs_connectome.synthetic.execute', side_effect=AssertionError('must not run')):
            run_synthetic(out, execute_tools=False)
        self.assertEqual(read_json(out/'provenance.json')['status'], 'planned_not_executed')
        self.assertFalse((out/'synthetic_result.json').exists())
        self.assertFalse(list(out.rglob('approval.json')))

    def test_missing_tools_fail_before_generating_outputs(self):
        out = self.root/'run'
        out.mkdir()
        with patch('dbs_connectome.synthetic.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'installed tools'):
                run_synthetic(out, execute_tools=True)
        self.assertFalse(list(out.iterdir()))

    def test_gui_requires_consent_and_rejects_patient_inputs(self):
        data = self.root/'data'
        data.mkdir()
        manager = JobManager(data, self.root/'output')
        try:
            body = {'stage':'synthetic-demo', 'inputs':{}, 'threads':2, 'execute':True}
            with self.assertRaisesRegex(ValueError, 'confirmation'):
                manager.arguments(body, self.root)
            body['approve_compute'] = True
            self.assertIn('--execute', manager.arguments(body, self.root))
            body['inputs'] = {'reference':str(data/'patient.nii.gz')}
            with self.assertRaisesRegex(ValueError, 'no patient'):
                manager.arguments(body, self.root)
        finally:
            manager.pool.shutdown(wait=True)

    def test_cli_has_no_patient_input_option(self):
        args = parser().parse_args(['synthetic-demo','--output-root',str(self.root)])
        self.assertFalse(args.execute)
        self.assertFalse(hasattr(args, 'reference'))

    def test_matrix_audit_accounts_for_self_and_unassigned_tracks(self):
        pairs, weights = self.root/'pairs.txt', self.root/'weights.txt'
        np.savetxt(pairs, [[1,2],[2,1],[1,1],[0,2]], fmt='%d')
        np.savetxt(weights, [2.,3.,7.,11.])
        good = np.array([[0.,5.],[5.,0.]])
        result = verify_matrix_weights(good, pairs, weights)
        self.assertEqual(result['assigned_nonself_tracks'], 2)
        self.assertEqual(result['discarded_self_or_unassigned_tracks'], 2)
        with self.assertRaisesRegex(ValueError, 'independently summed'):
            verify_matrix_weights(good*2, pairs, weights)


if __name__ == '__main__':
    unittest.main()
