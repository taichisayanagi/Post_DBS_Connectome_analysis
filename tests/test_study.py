import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import spearmanr

from dbs_connectome.cli import parser
from dbs_connectome.provenance import digest
from dbs_connectome.study import expected_checks, load_manifest, run_study
from dbs_connectome.study_metrics import exact_family, fisher_pool, identification, interval_metrics, session_metrics


class StudyTests(unittest.TestCase):
    def test_cli(self):
        self.assertEqual(parser().parse_args(['study','--manifest','m.json','--output-root','/tmp/o']).command, 'study')

    def test_exact_with_ties_matches_independent_scipy(self):
        x=np.array([[1,3],[1,2],[2,1],[4,2],[5,7]],float);y=np.array([1,4,2,3,3.])
        got=exact_family(x,y)
        null=np.array([[spearmanr(x[:,j],y[list(p)]).statistic for j in range(2)] for p in itertools.permutations(range(5))])
        rho=np.array([spearmanr(x[:,j],y).statistic for j in range(2)])
        np.testing.assert_allclose(got['rho'],rho,atol=1e-14)
        np.testing.assert_allclose(got['p_exact'],(abs(null)>=abs(rho)-1e-12).mean(0))
        np.testing.assert_allclose(got['p_maxT'],(abs(null).max(1)[:,None]>=abs(rho)-1e-12).mean(0))
        self.assertEqual(got['n_permutations'],120)

    def test_constant_and_nan_rejected(self):
        for x in (np.ones((4,2)), np.array([[1,2],[2,np.nan],[3,5],[6,4]])):
            with self.assertRaises(ValueError):exact_family(x,np.arange(4))

    def test_rotation_and_translation_invariance(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(32,4));b=a+rng.normal(size=a.shape)*.1
        hemi=np.array(['L']*16+['R']*16);net=np.tile(np.array(['A']*8+['B']*8),2)
        old=session_metrics(a,net,hemi);pair=interval_metrics(a,b,net,hemi)
        aa=a.copy();bb=b.copy()
        for h in ('L','R'):
            q=np.linalg.qr(rng.normal(size=(4,4)))[0];shift=rng.normal(size=4)*10;m=hemi==h
            aa[m]=a[m]@q+shift;bb[m]=b[m]@q+shift
        new=session_metrics(aa,net,hemi);newpair=interval_metrics(aa,bb,net,hemi)
        for k in old:
            if not k[1].startswith('G'):self.assertAlmostEqual(old[k],new[k],places=12)
        for k in pair:self.assertAlmostEqual(pair[k],newpair[k],places=9)
        self.assertEqual(len(old)+len(pair),50)

    def test_identification_null_against_brute_force(self):
        rng=np.random.default_rng(3);base=rng.normal(size=(3,20));f=np.repeat(base,3,axis=0)+rng.normal(size=(9,20))*.2
        got,_=identification(f,3);distribution=np.zeros(10,int)
        for a in itertools.permutations(range(3)):
            for b in itertools.permutations(range(3)):
                labels=np.repeat(np.arange(3),3);labels[1::3]=a;labels[2::3]=b
                distribution[(labels==labels[got['nearest']]).sum()]+=1
        self.assertEqual(got['null_distribution'],distribution.tolist())
        self.assertEqual(got['correct'],9)

    def test_fisher_refuses_perfect_or_small(self):
        with self.assertRaises(ValueError):fisher_pool([1,.5],[6,8],3)
        with self.assertRaises(ValueError):fisher_pool([.8,.5],[3,8],3)
        got=fisher_pool([.8,.5],[6,8],3)
        self.assertGreater(got['p_bonferroni'],got['p_two'])

    def fixture(self,root):
        source=root/'inputs';source.mkdir();out=root/'outputs';out.mkdir()
        nodes=source/'nodes.tsv'
        nodes.write_text('index\tlabel\tname\themisphere\ttissue\tnetwork\n'+''.join(f'{i}\t{i+1}\tparcel{i}\t'+('L' if i<16 else 'R')+'\tcortex\t'+('A' if i%16<8 else 'B')+'\n' for i in range(32)))
        evidence=source/'evidence.json';evidence.write_text('{"fixture":"synthetic only"}')
        subjects=[];rng=np.random.default_rng(13)
        for s in range(3):
            visits=[]
            for t,name in enumerate(('pre','post1','post2')):
                w=rng.uniform(.1,10,(32,32));w=(w+w.T)/2;np.fill_diagonal(w,0)
                p=source/f'{s}_{t}.npy';np.save(p,w)
                visits.append(dict(timepoint=name,matrix=str(p),sha256=digest(p),mask_scheme='virtual' if t==0 else 'shared_postoperative',exclusion_evidence=str(evidence),exclusion_evidence_sha256=digest(evidence)))
            subjects.append(dict(id=f'S{s+1}',visits=visits))
        spec=dict(schema_version=1,pairing_reviewed=True,nodes=str(nodes),nodes_sha256=digest(nodes),subjects=subjects,
                  clinical={'score':{'definition':'synthetic change','values':{'S1':2,'S2':3,'S3':1}},'constant':{'definition':'all same','values':{'S1':0,'S2':0,'S3':0}}})
        path=source/'manifest.json';path.write_text(json.dumps(spec))
        return path,out,spec

    def test_full_synthetic_replay_and_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as t:
            path,out,spec=self.fixture(Path(t));before={str(p):digest(p) for p in path.parent.iterdir()}
            run_study(path,out)
            result=json.loads((out/'study_result.json').read_text())
            self.assertEqual(result['replay_status'],'not_compared')
            self.assertEqual(result['n_cortical'],32)
            self.assertTrue((out/'study_review.html').exists())
            self.assertEqual(len(result['clinical_exclusions']),1)
            self.assertEqual(before,{str(p):digest(p) for p in path.parent.iterdir()})
            spec['expected']=[dict(kind='array',actual='loso',key='loso',path=str(out/'study_gradients.npz'),sha256=digest(out/'study_gradients.npz'))]
            checks=expected_checks(spec,{'loso':np.load(out/'study_gradients.npz')['loso']+.01},[])
            self.assertFalse(checks[0]['passed'])

    def test_manifest_subject_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as t:
            path,out,spec=self.fixture(Path(t))
            spec['clinical']['score']['values']['extra']=1
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'matching subject'):load_manifest(path)
            del spec['clinical']['score']['values']['extra'];spec['subjects'][0]['visits'][0]['sha256']='wrong'
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'Changed'):load_manifest(path)


if __name__=='__main__':unittest.main()
