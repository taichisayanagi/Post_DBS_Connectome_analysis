"""Generate an artificial cohort for the GUI numerical-replay workflow.

No MRI, patient, clinical or frozen paper data are distributed. Expected values
are deliberately NOT generated from this same implementation and called validation.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from dbs_connectome.provenance import digest,new_run,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    out=new_run(args.output_root,'synthetic_cohort_inputs')
    nets=['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
    node_rows=[]
    for h in ('L','R'):
        for net in nets:
            for j in range(4):
                i=len(node_rows)
                node_rows.append({'index':i,'label':i+1,'name':f'synthetic_{h}_{net}_{j+1}',
                                  'hemisphere':h,'tissue':'cortex','network':net})
    nodes=out/'synthetic_nodes.tsv'
    with nodes.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(node_rows[0]),delimiter='\t');writer.writeheader();writer.writerows(node_rows)
    evidence=out/'synthetic_not_an_exclusion_audit.json'
    write_json(evidence,{'synthetic_only':True,'warning':'No images or streamlines exist; mask_scheme labels only exercise the interface.'})
    rng=np.random.default_rng(2061);subjects=[];n=len(node_rows)
    for i in range(6):
        code=f'DEMO{i+1}'
        base=rng.lognormal(0,.7,(n,n));base=(base+base.T)/2
        visits=[]
        for j,visit in enumerate(('pre','post1','post2')):
            matrix=base+rng.uniform(0,.2*(j+1),(n,n));matrix=(matrix+matrix.T)/2;np.fill_diagonal(matrix,0)
            path=out/f'{code}_{visit}.npy';np.save(path,matrix)
            visits.append({'timepoint':visit,'matrix':str(path),'sha256':digest(path),
                           'mask_scheme':'virtual' if j==0 else 'shared_postoperative',
                           'exclusion_evidence':str(evidence),'exclusion_evidence_sha256':digest(evidence)})
        subjects.append({'id':code,'visits':visits})
    manifest={'schema_version':1,'pairing_reviewed':True,'synthetic_only':True,
              'nodes':str(nodes),'nodes_sha256':digest(nodes),'subjects':subjects,
              'clinical':{'synthetic_score':{'definition':'Artificial numbers; not any patient or clinical scale',
                         'values':dict(zip([s['id'] for s in subjects],[-4,2,9,1,-2,7]))}}}
    path=out/'synthetic_study_manifest.json';write_json(path,manifest)
    config=out/'synthetic_suite.json'
    write_json(config,{'schema_version':1,'scope':'numerical_replay','tasks':[
        {'id':'primary','operation':'study','manifest':str(path),'manifest_sha256':digest(path)},
        {'id':'graphs','operation':'graph-study','manifest':str(path),'manifest_sha256':digest(path)}]})
    print('Synthetic inputs:',out)
    print('In the GUI choose Run reviewed numerical replay suite and select:',config)
    print('Choose a SEPARATE output root. Expected status: computed_not_frozen_verified, never validated.')


if __name__=='__main__':main()
