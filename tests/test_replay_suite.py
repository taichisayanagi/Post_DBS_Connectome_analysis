import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dbs_connectome.provenance import digest,write_json
from dbs_connectome.replay_suite import run_suite,suite_paths


class ReplaySuiteTests(unittest.TestCase):
    def fixture(self,root):
        inputs=root/'inputs';inputs.mkdir();out=root/'out';out.mkdir()
        nodes=inputs/'nodes';nodes.write_text('synthetic fixture')
        manifest=inputs/'manifest.json';write_json(manifest,{'nodes':str(nodes),'subjects':[]})
        config=inputs/'suite.json';spec={'schema_version':1,'scope':'numerical_replay','tasks':[
            {'id':'A','operation':'study','manifest':str(manifest),'manifest_sha256':digest(manifest)},
            {'id':'B','operation':'study','manifest':str(manifest),'manifest_sha256':digest(manifest)}]}
        write_json(config,spec)
        return config,out,spec

    def test_continues_after_failure_and_never_certifies_images(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,spec=self.fixture(Path(d))
            def fake(manifest,target):
                if target.name=='A':raise ValueError('synthetic failure')
                write_json(target/'result.json',{'checks':[{'passed':True}]})
            with patch('dbs_connectome.replay_suite.OPERATIONS',{'study':(fake,'result.json')}):
                with self.assertRaises(ValueError):run_suite(config,out)
            result=json.loads((out/'suite_result.json').read_text())
            self.assertFalse(result['full_image_to_paper_validated'])
            self.assertEqual([r['status'] for r in result['tasks']],['failed','verified_numeric_replay'])
            self.assertTrue((out/'suite_review.html').exists())

    def test_uncompared_is_not_verified(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,spec=self.fixture(Path(d))
            def fake(manifest,target):write_json(target/'result.json',{})
            with patch('dbs_connectome.replay_suite.OPERATIONS',{'study':(fake,'result.json')}):run_suite(config,out)
            result=json.loads((out/'suite_result.json').read_text())
            self.assertEqual(result['status'],'completed_with_unverified_tasks')

    def test_changed_or_unsafe_tasks_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            config,out,spec=self.fixture(Path(d))
            spec['tasks'][0]['operation']='shell'
            with self.assertRaises(ValueError):suite_paths(spec)
            spec['tasks'][0]['operation']='study';spec['tasks'][0]['manifest_sha256']='wrong'
            with self.assertRaises(ValueError):suite_paths(spec)
