"""Unmasked cross-acquisition axis and identification benchmark.

Kept separate from masked postoperative clinical associations by design.
"""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from .gradients import Settings,embed
from .provenance import (assert_inputs_unchanged,digest,freeze_inputs,read_json,
                         record,require_separate_output,write_json)
from .study import clean,cohort_embeddings,expected_checks,load_manifest,manifest_paths,isolated_profiles
from .study_metrics import identification


def loso_gallery(matrices,nodes,legacy_zero_profiles=False):
    n=len(matrices);labels=np.repeat(np.arange(n),3);correct=0;assignments=[]
    ix={h:np.array([i for i,v in enumerate(nodes) if v['tissue']=='cortex' and v['hemisphere']==h]) for h in ('L','R')}
    for held in range(n):
        mean=matrices[[i for i in range(n) if i!=held],0].mean(0)
        refs={h:embed(mean,index,legacy_zero_profiles=legacy_zero_profiles)[0] for h,index in ix.items()}
        # Every gallery member is embedded under THIS held-out reference.
        features=np.array([np.vstack([embed(m,index,reference=refs[h],legacy_zero_profiles=legacy_zero_profiles)[0][:,:4] for h,index in ix.items()]).ravel()
                           for m in matrices.reshape(-1,matrices.shape[-1],matrices.shape[-1])])
        sim=np.corrcoef(features);np.fill_diagonal(sim,-np.inf)
        nearest=np.argmax(sim,axis=1)
        for row in np.flatnonzero(labels==held):
            best=sim[row,nearest[row]]
            if np.count_nonzero(np.isclose(sim[row],best,atol=1e-12,rtol=0))!=1:
                raise ValueError('Ambiguous held-out reference gallery assignment')
            correct+=labels[row]==labels[nearest[row]]
            assignments.append(int(nearest[row]))
    return {'correct':int(correct),'n_sessions':3*n,'nearest':assignments,
            'rule':'Re-embed the entire gallery under each held-out-subject reference; assess only its three query scans.'}


def run_reference_validation(manifest,out):
    out=Path(out);initial=read_json(manifest);paths=[manifest,*manifest_paths(initial)]
    for p in paths:require_separate_output(out,Path(p).parent)
    frozen=freeze_inputs(paths)
    spec,nodes,matrices=load_manifest(manifest,allow_unmasked=True)
    if spec.get('analysis_role')!='unmasked_reference_validation' or any(v['mask_scheme']!='none' for s in spec['subjects'] for v in s['visits']):
        raise ValueError('This benchmark requires explicitly unmasked visits; it is not the clinical analysis')
    legacy=spec.get('legacy_zero_profile_policy','reject')
    if legacy not in ('reject','brainspace_normalized_angle') or (legacy!='reject' and not spec.get('expected')):
        raise ValueError('Isolated profiles require explicit frozen historical-output verification')
    legacy=legacy!='reject'
    isolated=isolated_profiles(matrices,nodes,[s['id'] for s in spec['subjects']],['pre','post1','post2'])
    g,lam,order,_=cohort_embeddings(matrices,nodes,'common',legacy)
    n=len(matrices)
    settings={'sampled_pairs':spec.get('identification_samples',100000),
              'seed':spec.get('seed',42),'batch_size':spec.get('identification_batch_size',10000)}
    gradient,sim=identification(g.reshape(n*3,-1),n,**settings)
    cortical=matrices[:,:,order][:,:,:,order]
    upper=np.triu_indices(len(order),1)
    edges= np.array([m[upper] for m in cortical.reshape(-1,len(order),len(order))])
    edge,_=identification(edges,n,**settings)
    result={'mask_scheme':'none','isolated_profiles':isolated,'legacy_zero_profile_compatibility':legacy,
            'gradient':gradient,'edge':edge,'loso_gallery':loso_gallery(matrices,nodes,legacy),
            'density_all':float(np.mean(edges>0)*100),'density_pre':float(np.mean(edges[::3]>0)*100)}
    if spec.get('healthy_reference'):
        ref=spec['healthy_reference']
        if digest(ref['path'])!=ref.get('sha256'):
            raise ValueError('Healthy benchmark changed')
        h=np.loadtxt(ref['path'],delimiter=ref.get('delimiter',','))
        rows=np.asarray(ref['rows']);compare=np.asarray(ref['comparison_indices']);column=ref.get('component',0)
        if rows.dtype.kind not in 'iu' or rows.ndim!=1 or len(rows)!=len(order) or len(set(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(h)):
            raise ValueError('Healthy benchmark parcel mapping invalid')
        if compare.dtype.kind not in 'iu' or compare.ndim!=1 or len(set(compare))!=len(compare) or np.any(compare<0) or np.any(compare>=len(order)) or len(compare)<3:
            raise ValueError('Explicit shared-parcel comparison indices required')
        healthy=h[rows,column]
        mean=matrices[:,0].mean(0)
        group=[]
        for hemi in ('L','R'):
            ix=np.array([i for i in order if nodes[i]['hemisphere']==hemi])
            group.extend(embed(mean,ix,Settings())[0][:,0])
        group=np.asarray(group)
        r=float(np.corrcoef(group[compare],healthy[compare])[0,1])
        aligned=group*(-1 if r<0 else 1)
        nets=np.array([nodes[i]['network'] for i in order])
        unique=sorted(set(nets))
        hm={k:float(healthy[nets==k].mean()) for k in unique}
        gm={k:float(aligned[nets==k].mean()) for k in unique}
        result['healthy_axis']={'r':abs(r),'rho':abs(float(spearmanr(group[compare],healthy[compare]).statistic)),
                               'rank':float(spearmanr(list(gm.values()),list(hm.values())).statistic),
                               'network_means':gm,'healthy_network_means':hm,'n_compared':len(compare),
                               'display_sign':-1 if r<0 else 1}
    np.savez_compressed(out/'reference_validation_gradients.npz',common=g,lambdas=lam,cortical_indices=order,similarity=sim)
    checks=expected_checks(spec,{'reference_result':result},[])
    assert_inputs_unchanged(frozen);passed=all(c['passed'] for c in checks)
    result.update(checks=checks,status='completed' if passed else 'failed_replay',source_bytes_unchanged=True,
                  warning='Unmasked reference benchmark is not a validation of electrode exclusion or longitudinal clinical effects.')
    write_json(out/'reference_validation_result.json',clean(result))
    record(out,'reference-validation',paths,settings,sorted(out.glob('*')),'completed' if passed else 'failed_replay')
    if not passed:raise ValueError('Reference validation differs from frozen outputs')
