"""Predeclared embedding sensitivity, not a search for a significant result.

Setting-audit matrices and the axis-matching reference must each have explicit
lineage. The adopted setting audit uses shared postoperative masks. A separately
labelled session-mask sensitivity must never be mistaken for that primary basis.
"""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gradients import Settings
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, read_json,
                         record, require_separate_output, write_json)
from .study import (clean, cohort_embeddings, expected_checks, load_manifest,
                    manifest_paths, write_csv)
from .study_metrics import exact_family


def setting_family():
    primary = Settings()
    return {
        "primary_top25_align10": primary,
        "primary_top25_align4": replace(primary,n_components=4),
        "primary_top25_align6": replace(primary,n_components=6),
        "log_no_threshold": replace(primary,sparsity=None),
        "raw_no_threshold_alpha05": replace(primary,log_transform=False,sparsity=None),
        "zarkali_like": replace(primary,log_transform=False,sparsity=None,alpha=.5),
        "log_top10": replace(primary,sparsity=.9),
        "log_top50": replace(primary,sparsity=.5),
    }


def match_axes(gradients, primary):
    if gradients.shape[:-1] != primary.shape[:-1] or primary.shape[-1] != 4:
        raise ValueError("Sensitivity and reference subject/visit/parcel axes differ")
    k=gradients.shape[-1]
    correlation=np.corrcoef(gradients.reshape(-1,k).T,primary.reshape(-1,4).T)[:k,k:]
    if not np.isfinite(correlation).all():
        raise ValueError("Undefined axis correspondence")
    source,target=linear_sum_assignment(-np.abs(correlation))
    mapping={int(t):int(s) for s,t in zip(source,target)}
    if set(mapping)!=set(range(4)):
        raise ValueError("Four one-to-one source axis matches are required")
    return mapping,correlation


def candidate_values(gradients, networks, hemispheres, mapping):
    """Frozen four candidate definitions; no sign flipping after axis matching."""
    g=np.asarray(gradients)
    nets,hemis=np.asarray(networks),np.asarray(hemispheres)
    if g.ndim!=4 or g.shape[1]!=3 or g.shape[2]!=len(nets) or len(hemis)!=len(nets) or g.shape[-1]<4:
        raise ValueError("Expected subject x three visits x cortical parcels x components")
    for net in ('Vis','SomMot','Cont'):
        if any(np.count_nonzero((nets==net)&(hemis==h))<2 for h in ('L','R')):
            raise ValueError("Each candidate network needs at least two parcels per hemisphere")
    displacement=np.linalg.norm(g[:,2,nets=='Vis',:4]-g[:,1,nets=='Vis',:4],axis=-1).mean(1)
    ecc=[];position=[];spread=[]
    for t in (1,2):
        e=[];p=[];r=[]
        for h in ('L','R'):
            hm=hemis==h
            center=g[:,t,hm,:4].mean(1,keepdims=True)
            e.append(np.linalg.norm(g[:,t,hm&(nets=='SomMot'),:4]-center,axis=-1))
            p.append(g[:,t,hm&(nets=='Cont'),mapping[1]]-g[:,t,hm,mapping[1]].mean(1,keepdims=True))
            v=g[:,t,hm&(nets=='Vis'),mapping[0]]
            r.append(v-v.mean(1,keepdims=True))
        ecc.append(np.median(np.concatenate(e,axis=1),axis=1))
        position.append(np.median(np.concatenate(p,axis=1),axis=1))
        residual=np.concatenate(r,axis=1)
        spread.append(np.sqrt(np.sum(residual**2,axis=1)/(residual.shape[1]-2)))
    return {'Vis_parcel_displacement_mean':displacement,
            'SomMot_eccentricity_median':ecc[1]-ecc[0],
            'Cont_G2_position_median':position[1]-position[0],
            'Vis_G1_dispersion_sd':spread[1]-spread[0]}


def run_sensitivity(manifest, out):
    out=Path(out)
    initial=read_json(manifest)
    paths=[manifest,*manifest_paths(initial)]
    for path in paths:
        require_separate_output(out,Path(path).parent)
    frozen=freeze_inputs(paths)
    spec,nodes,matrices=load_manifest(manifest,allow_session_masks=True)
    if spec.get('analysis_role')!='embedding_sensitivity':
        raise ValueError('Explicit embedding_sensitivity role required; not the primary analysis')
    ids=[s['id'] for s in spec['subjects']]
    ref=spec.get('sensitivity_reference',{})
    if ref.get('reference_mode')!='loso' or ref.get('subjects')!=ids or ref.get('timepoints')!=['pre','post1','post2']:
        raise ValueError('Hash-bound LOSO axis reference and exact subject/visit order required')
    if not ref.get('mask_scheme') or digest(ref['path'])!=ref.get('sha256'):
        raise ValueError('Axis reference changed or lacks mask lineage')
    with np.load(ref['path'],allow_pickle=False) as z:
        primary=np.asarray(z[ref['key']])
    rows=[];settings_rows=[];correlations=[]
    for name,settings in setting_family().items():
        g,_,order,_=cohort_embeddings(matrices,nodes,'loso',settings=settings,retain_all=True)
        mapping,corr=match_axes(g,primary)
        metadata={'setting':name,**asdict(settings)}
        for axis in range(4):
            metadata[f'matched_G{axis+1}_source_index']=mapping[axis]+1
            metadata[f'matched_G{axis+1}_signed_correlation']=float(corr[mapping[axis],axis])
            metadata[f'matched_G{axis+1}_abs_correlation']=float(abs(corr[mapping[axis],axis]))
        settings_rows.append(metadata)
        values=candidate_values(g,[nodes[i]['network'] for i in order],[nodes[i]['hemisphere'] for i in order],mapping)
        for metric,x in values.items():
            rows.extend({'setting':name,'subject':s,'metric':metric,'value':float(v)} for s,v in zip(ids,x))
        for outcome,endpoint in spec['clinical'].items():
            valid=np.array([endpoint['values'][s] is not None for s in ids])
            y=np.array([endpoint['values'][s] for s in ids if endpoint['values'][s] is not None],float)
            if len(y)<3 or np.ptp(y)==0:
                continue
            fields=[k for k,v in values.items() if np.ptp(v[valid])>0]
            if not fields:
                continue
            stats=exact_family(np.array([values[k][valid] for k in fields]).T,y)
            for j,metric in enumerate(fields):
                correlations.append({'setting':name,'metric':metric,'outcome':outcome,'n':len(y),
                                     'family_size':len(fields),
                                     **{k:float(stats[k][j]) for k in ('rho','p_exact','q_bh','p_maxT')}})
        print(f'[step sensitivity] {name} complete',flush=True)
    tables={'embedding_candidate_subject_values':rows,'embedding_candidate_correlations':correlations,'embedding_settings':settings_rows}
    for name,table in tables.items():
        if table:write_csv(out/(name+'.csv'),table)
    checks=expected_checks(spec,tables,rows,correlations)
    assert_inputs_unchanged(frozen)
    passed=all(c['passed'] for c in checks)
    write_json(out/'sensitivity_result.json',clean({'status':'completed' if passed else 'failed_replay',
        'checks':checks,'n_settings':len(settings_rows),'source_bytes_unchanged':True,
        'reference_mask_scheme':ref['mask_scheme'],
        'session_mask_schemes':sorted({v['mask_scheme'] for s in spec['subjects'] for v in s['visits']}),
        'axis_sign_policy':'no post-hoc sign flipping; absolute correspondence used only for one-to-one matching',
        'warning':'Sensitivity variants are not independent replication. No setting is chosen by its clinical p-value. Sparse unstable high-order axes must not be assigned functional identities.'}))
    record(out,'sensitivity-study',paths,{'settings':{k:asdict(v) for k,v in setting_family().items()}},sorted(out.glob('*')),'completed' if passed else 'failed_replay')
    if not passed:raise ValueError('Sensitivity output does not match frozen expected results')
