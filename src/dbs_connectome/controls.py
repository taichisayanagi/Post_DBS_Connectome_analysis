"""Hash-bound matrix replay of repeat-scan and synthetic-void controls.

These calculations do not generate a new MRI or prove that a synthetic void
models all metal effects. Both favourable and unfavourable controls are exported.
"""
import itertools
from pathlib import Path

import numpy as np

from .gradients import Settings, embed, validate_matrix
from .provenance import (assert_inputs_unchanged,digest,freeze_inputs,read_json,
                         record,require_separate_output,write_json)
from .study import clean,cohort_embeddings,expected_checks,load_manifest,manifest_paths,write_csv
from .study_metrics import exact_family,interval_metrics,session_metrics


def icc21(first, second):
    """Two-way random-effects single-measure absolute-agreement ICC(2,1)."""
    x=np.column_stack((first,second)).astype(float)
    if len(x)<3 or not np.isfinite(x).all():
        raise ValueError('ICC requires at least three paired finite observations')
    n,k=x.shape;grand=x.mean()
    msr=k*np.sum((x.mean(1)-grand)**2)/(n-1)
    msc=n*np.sum((x.mean(0)-grand)**2)/(k-1)
    mse=np.sum((x-x.mean(1,keepdims=True)-x.mean(0,keepdims=True)+grand)**2)/((n-1)*(k-1))
    denominator=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denominator) if denominator else float('nan')


def paired_signflip(first, second):
    d=np.asarray(first,float)-np.asarray(second,float)
    if d.ndim!=1 or not 3<=len(d)<=16 or not np.isfinite(d).all():
        raise ValueError('Exact paired sign-flip supports 3..16 finite pairs')
    null=np.array([np.mean(d*np.asarray(sign)) for sign in itertools.product([1,-1],repeat=len(d))])
    return {'mean_difference':float(d.mean()),'p_greater':float(np.mean(null>=d.mean()-1e-15)),
            'p_two':float(np.mean(abs(null)>=abs(d.mean())-1e-15)),'n_signflips':len(null)}


def control_matrix(item,nodes):
    path=Path(item['matrix'])
    if digest(path)!=item.get('sha256'):
        raise ValueError('Control matrix changed')
    m=validate_matrix(np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.loadtxt(path,delimiter=','))
    if not item.get('mask_scheme'):
        raise ValueError('Explicit control mask scheme required')
    if item['mask_scheme']!='none':
        if not item.get('exclusion_evidence') or digest(item['exclusion_evidence'])!=item.get('exclusion_evidence_sha256'):
            raise ValueError('Control exclusion evidence missing or changed')
    ix=np.asarray(item.get('matrix_indices',list(range(len(m)))))
    if ix.dtype.kind not in 'iu' or len(ix)!=len(nodes) or len(set(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(m)):
        raise ValueError('Control node mapping differs from the fixed basis')
    return m[np.ix_(ix,ix)]


def run_controls(manifest,out):
    out=Path(out);initial=read_json(manifest);paths=[manifest,*manifest_paths(initial)]
    for path in paths:require_separate_output(out,Path(path).parent)
    frozen=freeze_inputs(paths)
    spec,nodes,matrices=load_manifest(manifest)
    cases=spec.get('control_sessions',{})
    if not cases or any(not c.get('definition') for c in cases.values()):
        raise ValueError('Named control cases with explicit definitions are required')
    allowed={'retest','simulation_baseline','simulation_void','simulation_rim','simulation_retrack'}
    if not set(cases)<=allowed:
        raise ValueError('Unknown control role')
    legacy=spec.get('legacy_zero_profile_policy','reject')
    if legacy not in ('reject','brainspace_normalized_angle') or (legacy!='reject' and not spec.get('expected')):
        raise ValueError('Isolated profiles require explicit frozen historical-output verification')
    legacy=legacy!='reject'
    ids=[s['id'] for s in spec['subjects']];n=len(ids)
    g,_,order,_=cohort_embeddings(matrices,nodes,'loso',legacy)
    nets=np.array([nodes[i]['network'] for i in order]);hemis=np.array([nodes[i]['hemisphere'] for i in order])
    indices={h:np.array([i for i,r in enumerate(nodes) if r['tissue']=='cortex' and r['hemisphere']==h]) for h in ('L','R')}
    settings=Settings();refs={}
    for i in range(n):
        mean=matrices[[j for j in range(n) if j!=i],0].mean(0)
        refs[i]={h:embed(mean,ix,settings,legacy_zero_profiles=legacy)[0] for h,ix in indices.items()}
    gs={'pre':g[:,0],'post1':g[:,1],'post2':g[:,2]}
    for name,case in cases.items():
        if [s['id'] for s in case['subjects']]!=ids:
            raise ValueError('Control subject order does not match the cohort')
        values=[]
        for i,item in enumerate(case['subjects']):
            m=control_matrix(item,nodes)
            values.append(np.vstack([embed(m,ix,settings,refs[i][h],legacy_zero_profiles=legacy)[0][:,:4] for h,ix in indices.items()]))
        gs[name]=np.asarray(values)
        print(f'[step control] {name} embedded for {n} participants',flush=True)
    np.savez_compressed(out/'control_gradients.npz',subjects=np.array(ids),cortical_indices=order,**gs)
    sm={k:[session_metrics(v,nets,hemis) for v in a] for k,a in gs.items()}
    def interval(a,b):return [interval_metrics(x,y,nets,hemis) for x,y in zip(gs[a],gs[b])]
    observed=interval('post1','post2');key=('Vis','parcel_displacement_mean')
    observed_values=np.array([r[key] for r in observed])
    result={'scope':'matrix_replay_of_artifact_and_repeat_scan_controls','available':list(cases),
            'reference':'patient-specific LOSO; repeat-scan gallery is a conditional reliability check, not unbiased clinical prediction'}
    if 'retest' in gs:
        rt=interval('post2','retest');rt_values=np.array([r[key] for r in rt])
        rows=[]
        for network,metric in sorted(sm['post2'][0]):
            a=[r[network,metric] for r in sm['post2']];b=[r[network,metric] for r in sm['retest']]
            rows.append({'network':network,'metric':metric,'icc21':icc21(a,b)})
        write_csv(out/'retest_icc.csv',rows)
        iccs={f'{r["network"]}|{r["metric"]}':r['icc21'] for r in rows}
        v=np.array(list(iccs.values()))
        result['retest_icc']={'n_metrics':len(v),'median':float(np.nanmedian(v)),
            'frac_ge_0_6':float(np.mean(v>=.6)),'frac_lt_0_4':float(np.mean(v<.4)),
            'candidates':{k:iccs[k] for k in ('SomMot|eccentricity_median','Vis|G1_dispersion_sd')}}
        gallery=g.reshape(n*3,-1);lab=np.repeat(np.arange(n),3);hits=0;margins=[];same=[]
        for i in range(n):
            correlations=np.corrcoef(np.vstack([gs['retest'][i].ravel(),gallery]))[0,1:]
            hits+=lab[np.argmax(correlations)]==i
            margins.append(correlations[lab==i].max()-correlations[lab!=i].max())
            same.append(np.corrcoef(gs['retest'][i].ravel(),gs['post2'][i].ravel())[0,1])
        result['retest_identification']={'correct':int(hits),'of':n,'margin_median':float(np.median(margins)),'r_same_session_median':float(np.median(same))}
        test=paired_signflip(observed_values,rt_values)
        result['vis_displacement_obs_vs_retest']={'obs':observed_values.tolist(),'retest':rt_values.tolist(),
            'ratio_median':float(np.median(observed_values/rt_values)),
            'p_one_sided_signflip':test['p_greater'],'corr_retest_with_obs':float(np.corrcoef(rt_values,observed_values)[0,1])}
        # Export the noise/outcome association too, never only the favourable reliability result.
        family=np.column_stack((observed_values,rt_values,observed_values-rt_values))
        clinical=[]
        for outcome,endpoint in spec['clinical'].items():
            valid=np.array([endpoint['values'][s] is not None for s in ids])
            y=np.array([endpoint['values'][s] for s in ids if endpoint['values'][s] is not None],float)
            if len(y)<3 or np.ptp(y)==0:continue
            if np.any(np.ptp(family[valid],axis=0)==0):continue
            stats=exact_family(family[valid],y)
            for j,name in enumerate(('observed_displacement','same_day_displacement','observed_minus_same_day')):
                clinical.append({'metric':name,'outcome':outcome,'n':len(y),**{k:float(stats[k][j]) for k in ('rho','p_exact','p_greater','p_less','q_bh','p_maxT')}})
        if clinical:write_csv(out/'noise_clinical_correlations.csv',clinical)
        result['noise_clinical_correlations']=clinical
    sim={'simulation_baseline','simulation_void','simulation_retrack'}
    if sim<=set(gs):
        def error(a,b):return np.linalg.norm(gs[a]-gs[b],axis=-1)
        void=error('simulation_void','simulation_baseline');floor=error('simulation_retrack','simulation_baseline')
        scale=np.linalg.norm(gs['pre']-gs['pre'].mean(0),axis=-1).std(0)+1e-12
        result['sim_position_error']={'void_vs_exclusion_only_median_sd':float(np.median(void/scale)),
            'retrack_floor_median_sd':float(np.median(floor/scale)),
            'ratio':float(np.median(void)/np.median(floor)),
            'per_subject_ratio':[float(np.median(a)/np.median(b)) for a,b in zip(void,floor)]}
        pairs=[('simulation_baseline','simulation_retrack','tractography floor'),('simulation_baseline','simulation_void','void inside mask')]
        if 'simulation_rim' in gs:
            rim=error('simulation_rim','simulation_baseline')
            result['sim_position_error'].update(void_beyond_mask_median_sd=float(np.median(rim/scale)),ratio_B=float(np.median(rim)/np.median(floor)))
            pairs.append(('simulation_void','simulation_rim','void change between visits (rim beyond mask)'))
        false={}
        for a,b,label in pairs:
            v=np.array([r[key] for r in interval(a,b)])
            false[label]={'mean':float(v.mean()),'over_obs_between_sd':float(v.mean()/observed_values.std(ddof=1)),'over_obs_mean':float(v.mean()/observed_values.mean())}
        result['sim_false_vis_displacement']=false
    checks=expected_checks(spec,{'control_result':result},[])
    assert_inputs_unchanged(frozen)
    passed=all(c['passed'] for c in checks)
    result.update(checks=checks,status='completed' if passed else 'failed_replay',source_bytes_unchanged=True,
        warning='Matrix replay does not validate generation of the simulation images or new tractography. Report the repeat-scan noise comparator alongside clinical correlations.')
    write_json(out/'control_result.json',clean(result))
    record(out,'control-study',paths,{'settings':settings.__dict__},sorted(out.glob('*')),'completed' if passed else 'failed_replay')
    if not passed:raise ValueError('Control replay differs from frozen outputs')
