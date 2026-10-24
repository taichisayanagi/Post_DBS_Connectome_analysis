"""Descriptive matched-map identification and change-map concordance audits."""
import itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

from .provenance import (assert_inputs_unchanged, freeze_inputs, read_json,
                         record, require_separate_output, write_json)
from .study import (clean, cohort_embeddings, expected_checks, isolated_profiles,
                    load_manifest, manifest_paths, write_csv)


def visit_excluded_identification(features, subjects, visits, standardize=False):
    x = np.asarray(features, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite session-by-feature matrix required')
    if standardize:
        x = (x-x.mean(0))/(x.std(0)+1e-12)
    if np.any(np.ptp(x, axis=1) == 0):
        raise ValueError('Constant session feature vector')
    similarity = np.corrcoef(x)
    labels, times = np.asarray(subjects), np.asarray(visits)
    similarity[times[:, None] == times[None, :]] = -np.inf
    nearest = np.argmax(similarity, axis=1)
    within, between = [], []
    for i in range(len(x)):
        if np.count_nonzero(np.isclose(similarity[i], similarity[i, nearest[i]], atol=1e-12, rtol=0)) != 1:
            raise ValueError('Ambiguous cross-visit nearest neighbour')
        same = similarity[i, (labels == labels[i]) & np.isfinite(similarity[i])]
        other = similarity[i, (labels != labels[i]) & np.isfinite(similarity[i])]
        if not len(same) or not len(other): raise ValueError('Missing cross-visit comparison')
        within.append(same.mean()); between.append(other.mean())
    return {'acc': float(np.mean(labels == labels[nearest])),
            'margin': float(np.mean(within)-np.mean(between)),
            'nearest': nearest.tolist(),
            'margin_definition': 'mean within-subject similarity minus mean between-subject similarity; not nearest-neighbour margin'}


def change_concordance(gradients, hemispheres, subjects, permutations=20000, seed=20260731):
    """Legacy spatial null: no cortical spatial-autocorrelation preservation."""
    g = np.asarray(gradients, float); hemispheres = np.asarray(hemispheres)
    if g.shape[:2] != (len(subjects), 3) or g.shape[2] != len(hemispheres) or g.shape[3] != 4 or not np.isfinite(g).all():
        raise ValueError('Expected subject x three visits x cortical parcels x four axes')
    if not isinstance(permutations, int) or not 1 <= permutations <= 1000000:
        raise ValueError('Spatial permutation count must be 1..1000000')
    changed, ranks = {}, {}
    for h in ('L', 'R'):
        values = g[:, :, hemispheres == h]
        ecc = np.linalg.norm(values-values.mean(2, keepdims=True), axis=3)
        changed[h] = ecc[:, 2]-ecc[:, 1]
        r = np.stack([rankdata(row) for row in changed[h]])
        r -= r.mean(1, keepdims=True); length = np.linalg.norm(r, axis=1, keepdims=True)
        if np.any(length == 0): raise ValueError('Constant change map cannot be correlated')
        ranks[h] = r/length
    pairs = list(itertools.combinations(range(len(subjects)), 2)); rows = []
    total = len(hemispheres)
    def pair_values(v):
        return [sum(float(np.dot(v[h][i], v[h][j]))*v[h].shape[1] for h in ('L','R'))/total for i,j in pairs]
    observed = pair_values(ranks)
    for (i,j), value in zip(pairs, observed):
        rows.append({'subject_1': subjects[i], 'subject_2': subjects[j],
                     'left_rho': float(np.dot(ranks['L'][i],ranks['L'][j])),
                     'right_rho': float(np.dot(ranks['R'][i],ranks['R'][j])), 'node_weighted_rho': value})
    rng = np.random.default_rng(seed); null = np.empty(permutations)
    for k in range(permutations):
        shuffled = {h: np.stack([rng.permutation(row) for row in ranks[h]]) for h in ('L','R')}
        null[k] = np.mean(pair_values(shuffled))
    mean = float(np.mean(observed))
    return {'observed_mean_node_weighted_rho': mean, 'pairwise_min': min(observed), 'pairwise_max': max(observed),
            'positive_pairs': int(np.count_nonzero(np.asarray(observed)>0)), 'n_subject_pairs': len(pairs),
            'n_permutations': permutations, 'seed': seed,
            'one_sided_p_for_positive_concordance': float((1+np.count_nonzero(null>=mean))/(permutations+1)),
            'null_mean': float(null.mean()), 'null_95th_percentile': float(np.quantile(null,.95)),
            'warning': 'Parcel permutations are independent within subject and hemisphere; this is not a spatial-autocorrelation-preserving null.'}, rows, null


def run_representations(manifest, out):
    out=Path(out); initial=read_json(manifest); paths=[manifest,*manifest_paths(initial)]
    for p in paths: require_separate_output(out,Path(p).parent)
    frozen=freeze_inputs(paths); spec,nodes,matrices=load_manifest(manifest)
    subjects=[s['id'] for s in spec['subjects']]; n=len(subjects)
    legacy=spec.get('legacy_zero_profile_policy','reject')
    if legacy not in ('reject','brainspace_normalized_angle') or (legacy!='reject' and not spec.get('expected')):
        raise ValueError('Historical zero-profile compatibility requires frozen verification')
    g,_,order,_=cohort_embeddings(matrices,nodes,'common',legacy!='reject')
    cort=matrices[:,:,order][:,:,:,order].reshape(3*n,len(order),len(order)); weights=np.log1p(cort)
    upper=np.triu_indices(len(order),1)
    features={'G1':g[:,:,:,0].reshape(3*n,-1),'G1-G4':g.reshape(3*n,-1),
              'node_strength':weights.sum(2),'node_degree':(weights>0).sum(2).astype(float),
              'edges_log1p':weights[:,upper[0],upper[1]]}
    labels=np.repeat(np.arange(n),3); visits=np.tile(np.arange(3),n)
    result={'representations':{},'isolated_profiles':isolated_profiles(matrices,nodes,subjects,['pre','post1','post2']),
            'legacy_zero_profile_compatibility':legacy!='reject',
            'gallery_rule':'Only other visits can serve as neighbours; differs from unrestricted primary identification.'}
    for name,f in features.items():
        result['representations'][name]={'dim':f.shape[1],
            'raw':visit_excluded_identification(f,labels,visits),
            'z':visit_excluded_identification(f,labels,visits,True)}
    if spec.get('change_concordance'):
        cfg=spec['change_concordance']; loso,_,_,_=cohort_embeddings(matrices,nodes,'loso',legacy!='reject')
        result['concordance'],rows,null=change_concordance(loso,[nodes[i]['hemisphere'] for i in order],subjects,
            cfg.get('permutations',20000),cfg.get('seed',20260731))
        write_csv(out/'pairwise_spatial_correlations.csv',rows);np.save(out/'change_concordance_null.npy',null)
    checks=expected_checks(spec,{'representation_result':result},[])
    assert_inputs_unchanged(frozen); passed=all(c['passed'] for c in checks)
    result.update(checks=checks,source_bytes_unchanged=True,status='completed' if passed else 'failed_replay',
        warning='Descriptive representation comparisons. No classifier training or demonstration of gradient superiority.')
    write_json(out/'representation_result.json',clean(result))
    record(out,'representation-study',paths,{'change_concordance':spec.get('change_concordance')},sorted(out.glob('*')),result['status'])
    if not passed:raise ValueError('Representation audit differs from frozen outputs')
