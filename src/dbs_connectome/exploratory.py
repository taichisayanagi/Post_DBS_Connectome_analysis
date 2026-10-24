"""Reproduce the historical pooled-coordinate exploratory metric inventory.

This module labels rather than endorses historical geometry and reliability
definitions. It never selects favourable correlations or invents anatomy.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

from .comparative import NETWORKS, clinical_values
from .lateralized import correlation_null
from .provenance import (assert_inputs_unchanged, freeze_inputs, read_json, record,
                         require_separate_output, write_json)
from .study import (clean, cohort_embeddings, expected_checks, load_manifest,
                    manifest_paths, write_csv)
from .statistical_audit import family_hit_null

JP = dict(zip(NETWORKS, ('視覚', '体性運動', '背側注意', '腹側注意', '辺縁系', '前頭頭頂制御', 'DMN')))


def exploratory_metrics(g, eigenvalues, matrices, nodes, order, contralateral):
    n = len(g); nets = np.array([nodes[i]['network'] for i in order]); hemis = np.array([nodes[i]['hemisphere'] for i in order])
    if g.shape != (n, 3, len(order), 4) or len(contralateral) != n or any(h not in ('L', 'R') for h in contralateral):
        raise ValueError('Four-axis gradients and reviewed contralateral hemispheres required')
    eccentricity = np.linalg.norm(g-g.mean(2, keepdims=True), axis=3)
    metrics, exclusions = {}, []
    def add(key, label, family, function):
        values = np.array([[function(s, t) for t in range(3)] for s in range(n)])
        if not np.isfinite(values).all() or np.std(values[:, 2]-values[:, 1]) == 0:
            exclusions.append({'metric': key, 'reason': 'Nonfinite or constant change; correlation undefined'})
        else: metrics[key] = {'legacy_metric': label, 'family': family, 'values': values}
    def hemi(s, contra): return contralateral[s] if contra else ('L' if contralateral[s] == 'R' else 'R')
    def sel(net, h=None): return (nets == net) & ((hemis == h) if h else True)
    def ecc(s, t, mask): return float(eccentricity[s, t, mask].mean())
    def disp(s, t, mask):
        v = g[s, t, mask]; return float(np.linalg.norm(v-v.mean(0), axis=1).mean())
    for net in NETWORKS:
        for side, pick, label in (('contra', True, '対側'), ('ipsi', False, '同側')):
            add(f'{net}_{side}_ecc', f'ecc_{JP[net]}_{label}', 'laterality', lambda s,t,net=net,pick=pick:ecc(s,t,sel(net,hemi(s,pick))))
            add(f'{net}_{side}_disp', f'disp_{JP[net]}_{label}', 'laterality', lambda s,t,net=net,pick=pick:disp(s,t,sel(net,hemi(s,pick))))
    for side, pick, label in (('contra', True, '対側'), ('ipsi', False, '同側')):
        add(f'cortex_{side}_ecc', f'ecc_皮質全体_{label}', 'laterality', lambda s,t,pick=pick:ecc(s,t,hemis==hemi(s,pick)))
    motor = np.isin(nets, ('SomMot','DorsAttn','Cont'))
    add('motor3_ecc','ecc_運動系3網','motor_restriction',lambda s,t:ecc(s,t,motor))
    add('motor3_disp','disp_運動系3網','motor_restriction',lambda s,t:disp(s,t,motor))
    for side,pick,label in (('contra',True,'対側'),('ipsi',False,'同側')):
        add(f'motor3_{side}_ecc',f'ecc_運動系3網_{label}','motor_restriction',lambda s,t,pick=pick:ecc(s,t,motor&(hemis==hemi(s,pick))))
    add('motor2_ecc','ecc_体性運動+背側注意','motor_restriction',lambda s,t:ecc(s,t,np.isin(nets,('SomMot','DorsAttn'))))
    for net in NETWORKS:
        for axis in range(4):
            add(f'{net}_G{axis+1}_median',f'medpos_{JP[net]}_G{axis+1}','signed_positions',lambda s,t,net=net,axis=axis:float(np.median(g[s,t,sel(net),axis])))
    for axis in range(4):
        add(f'cortex_G{axis+1}_median',f'medpos_皮質全体_G{axis+1}','signed_positions',lambda s,t,axis=axis:float(np.median(g[s,t,:,axis])))
    for a,b in combinations(NETWORKS,2):
        add(f'{a}_{b}_distance',f'btw_{JP[a]}⟷{JP[b]}','network_distances',lambda s,t,a=a,b=b:float(np.linalg.norm(np.median(g[s,t,sel(a)],axis=0)-np.median(g[s,t,sel(b)],axis=0))))
    for axis in range(4):
        add(f'G{axis+1}_kendall',f'順序保存_G{axis+1}','ordering',lambda s,t,axis=axis:float(kendalltau(g[s,1,:,axis],g[s,t,:,axis]).statistic))
    add('cortex_displacement','|移動量|_皮質全体','displacement',lambda s,t:float(np.linalg.norm(g[s,t]-g[s,1],axis=1).mean()))
    for net in NETWORKS:
        add(f'{net}_displacement',f'|移動量|_{JP[net]}','displacement',lambda s,t,net=net:float(np.linalg.norm(g[s,t,sel(net)]-g[s,1,sel(net)],axis=1).mean()))
    for axis in range(4):
        add(f'G{axis+1}_eigenvalue_share',f'説明分散_G{axis+1}','eigenvalues_range',lambda s,t,axis=axis:float(eigenvalues[s,t].mean(0)[axis]/eigenvalues[s,t].mean(0).sum()*100))
        add(f'G{axis+1}_range',f'range_G{axis+1}','eigenvalues_range',lambda s,t,axis=axis:float(np.ptp(g[s,t,:,axis])))
    selected_structures = {}
    for name,jp,substring in [('thalamus','視床','thalam'),('putamen','被殻','putamen'),('caudate','尾状核','caudate'),('pallidum','淡蒼球','pallid'),('accumbens','側坐核','accumbens')]:
        ix = np.array([i for i,node in enumerate(nodes) if node['tissue']!='cortex' and substring in node['name'].lower()],dtype=int)
        if not len(ix):continue
        selected_structures[name] = [nodes[i]['name'] for i in ix]
        for net in (*NETWORKS,'cortex'):
            target = order if net=='cortex' else order[nets==net]
            label = '皮質全体' if net=='cortex' else JP[net]
            # The historical time-0 value duplicated post1. Preserve but explicitly flag it.
            add(f'{name}_{net}_absolute',f'abs_{jp}→{label}','subcortical_absolute',lambda s,t,ix=ix,target=target:float(matrices[s,max(t,1)][np.ix_(ix,target)].sum()))
    return metrics, exclusions, selected_structures


def historical_variance_ratio(values):
    total = values.ravel().var(ddof=1)
    return float(np.clip(values.mean(1).var(ddof=1)/total,0,1)) if total>0 else 0.


def pooled_inventory(g, eigenvalues, nodes, order, include_positions):
    nets = np.array([nodes[i]['network'] for i in order]); n = len(g)
    ecc = np.linalg.norm(g-g.mean(2, keepdims=True), axis=3)
    metrics, exclusions = {}, []
    def add(name, legacy, family, f):
        values = np.array([[f(s,t) for t in range(3)] for s in range(n)])
        if not np.isfinite(values).all() or np.std(values[:,2]-values[:,1]) == 0:
            exclusions.append({'metric':name,'reason':'Nonfinite or constant change'})
        else: metrics[name] = {'legacy_metric':legacy,'family':family,'values':values}
    for axis in range(4):
        add(f'G{axis+1}_eigenvalue_share',f'説明分散_G{axis+1}','eigenvalues_range',lambda s,t,axis=axis:float(eigenvalues[s,t].mean(0)[axis]/eigenvalues[s,t].mean(0).sum()*100))
        add(f'G{axis+1}_range',f'range_G{axis+1}','eigenvalues_range',lambda s,t,axis=axis:float(np.ptp(g[s,t,:,axis])))
    add('spectral_energy','spectral_energy','eigenvalues_range',lambda s,t:float((ecc[s,t]**2).mean()))
    add('cortex_ecc','ecc_皮質全体','cortex',lambda s,t:float(ecc[s,t].mean()))
    for net in NETWORKS:
        mask=nets==net
        add(f'{net}_ecc',f'ecc_{JP[net]}','cortex',lambda s,t,mask=mask:float(ecc[s,t,mask].mean()))
        add(f'{net}_disp',f'disp_{JP[net]}','cortex',lambda s,t,mask=mask:float(np.linalg.norm(g[s,t,mask]-g[s,t,mask].mean(0),axis=1).mean()))
    posterior=np.array(['DorsAttn_Post' in nodes[i]['name'] for i in order])
    if posterior.any():
        add('DorsAttn_posterior_ecc','ecc_背側注意後部','cortex',lambda s,t:float(ecc[s,t,posterior].mean()))
    center=np.mean if include_positions else np.median
    for a,b in combinations(NETWORKS,2):
        add(f'{a}_{b}_distance',f'btw_{JP[a]}⟷{JP[b]}','network_distances',lambda s,t,a=a,b=b:float(np.linalg.norm(center(g[s,t,nets==a],axis=0)-center(g[s,t,nets==b],axis=0))))
    if include_positions:
        for net in NETWORKS:
            for axis in range(2):
                add(f'{net}_G{axis+1}_median',f'medpos_{JP[net]}_G{axis+1}','signed_positions',lambda s,t,net=net,axis=axis:float(np.median(g[s,t,nets==net,axis])))
    return metrics,exclusions,{}


def run_exploratory(manifest,out):
    out=Path(out);initial=read_json(manifest);paths=[manifest,*manifest_paths(initial)]
    for p in paths:require_separate_output(out,Path(p).parent)
    mode=initial.get('exploratory_inventory','legacy_140')
    if mode not in ('legacy_140','legacy_60_mean_centres','legacy_46_median_centres'):
        raise ValueError('Unknown exploratory inventory')
    frozen=freeze_inputs(paths);spec,nodes,matrices=load_manifest(manifest,allow_session_masks=mode=='legacy_60_mean_centres')
    subjects=[s['id'] for s in spec['subjects']];side=spec.get('symptom_dominant_side',{})
    if set(side)!=set(subjects) or spec.get('symptom_side_reviewed') is not True or any(s not in ('L','R') for s in side.values()):
        raise ValueError('Reviewed symptom-dominant sides required; do not infer from imaging')
    contra=['R' if side[s]=='L' else 'L' for s in subjects]
    g,lam,order,_=cohort_embeddings(matrices,nodes,'loso')
    if mode=='legacy_140':
        metrics,excluded,structures=exploratory_metrics(g,lam,matrices,nodes,order,contra)
    else:
        metrics,excluded,structures=pooled_inventory(g,lam,nodes,order,mode=='legacy_60_mean_centres')
    outcomes={name:clinical_values(spec,name) for name in spec['clinical']}
    rows,nulls,subject_values=[],[],[]
    legacy_outcomes={'motor_improvement_percent':'改善率%','motor_improvement_points':'絶対点数',
        'preoperative_off':'術前OFF重症度','preoperative_on':'術前ON重症度',
        'levodopa_response_percent':'L-dopa反応性%','preoperative_moca':'術前MoCA','preoperative_mmse':'術前MMSE'}
    for name,record_ in metrics.items():
        values=record_['values'];delta=values[:,2]-values[:,1];ratio=historical_variance_ratio(values)
        for s,v,d in zip(subjects,values,delta):
            subject_values.append({'subject':s,'metric':name,'pre_value_legacy':v[0],'post1':v[1],'post2':v[2],'delta':d})
        for outcome,y in outcomes.items():
            rho,p,null=correlation_null(delta,y)
            rows.append({'metric':name,'legacy_metric':record_['legacy_metric'],'family':record_['family'],
                'outcome':outcome,'legacy_outcome':legacy_outcomes.get(outcome,outcome),'rho':rho,'p':p,
                'historical_variance_ratio_not_icc':ratio,'same_sign':int(max((delta>0).sum(),(delta<0).sum())),
                'mean_change':float(delta.mean())})
            nulls.append(null)
    maximum=abs(np.stack(nulls)).max(0)
    for row in rows:row['p_fwe']=float(np.mean(maximum>=abs(row['rho'])-1e-12))
    for endpoint in outcomes:
        ix=[i for i,r in enumerate(rows) if r['outcome']==endpoint]
        mx=abs(np.stack([nulls[i] for i in ix])).max(0)
        for i in ix:rows[i]['p_fwe_clin']=float(np.mean(mx>=abs(rows[i]['rho'])-1e-12))
    write_csv(out/'exploratory_correlations.csv',rows);write_csv(out/'exploratory_subject_values.csv',subject_values)
    hit_audit=family_hit_null(nulls,[r['p'] for r in rows])
    result={'inventory':mode,'n_metrics':len(metrics),'n_tests':len(rows),'n_nominal_p_lt_05':sum(r['p']<.05 for r in rows),
        'exact_discrete_null_expected_hits_p_lt_05':hit_audit['expected_hits'],
        'dependency_preserving_hit_null':hit_audit,
        'n_fwe_lt_05':sum(r['p_fwe']<.05 for r in rows),'excluded_metrics':excluded,'subcortical_name_matches':structures,
        'warning':'Historical exploratory inventory pools independently embedded hemispheres. The legacy column named ICC is only a clipped between-subject/total variance ratio, not an ICC estimator. Subcortical time-0 entries duplicate post1 and are not preoperative measurements. Eigenvalue shares are not PCA explained-variance estimates.'}
    checks=expected_checks(spec,{'exploratory_correlations':rows,'exploratory_result':result},[])
    assert_inputs_unchanged(frozen);passed=all(c['passed'] for c in checks)
    result.update(checks=checks,source_bytes_unchanged=True,status='completed' if passed else 'failed_replay')
    write_json(out/'exploratory_result.json',clean(result))
    record(out,'exploratory-study',paths,{'reference':'loso','scope':'historical_definition_audit'},sorted(out.glob('*')),result['status'])
    if not passed:raise ValueError('Historical exploratory inventory differs from frozen outputs')
