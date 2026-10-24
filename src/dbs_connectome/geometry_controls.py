"""Spatial/mask controls with explicit, hash-bound geometry and matrix inputs.

Registered masks and parcel labels are supplied, not inferred from filenames.
These tests measure residual mask effects; a null result is not proof of absence.
"""
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

from .controls import control_matrix
from .gradients import embed
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, read_json,
                         record, require_separate_output, write_json)
from .study import clean, cohort_embeddings, expected_checks, load_manifest, manifest_paths
from .study_metrics import identification


def image(spec):
    if digest(spec['path']) != spec.get('sha256'):
        raise ValueError('Spatial-control image changed after review')
    im = nib.load(spec['path'])
    data = np.asarray(im.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError('Finite 3D spatial-control image required')
    return im, data


def parcel_distances(atlas_spec, mask_spec, labels):
    """Centroid to nearest mask-voxel CENTRE, in world mm (not boundary distance)."""
    ai, a = image(atlas_spec); mi, m = image(mask_spec)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or len(set(labels)) != len(labels) or np.any(labels <= 0):
        raise ValueError('Explicit unique positive source atlas labels required')
    voxels = np.argwhere(m > .5)
    if not len(voxels):
        raise ValueError('Distance control requires a nonempty mask')
    tree = cKDTree(nib.affines.apply_affine(mi.affine, voxels))
    distances = []
    for label in labels:
        points = np.argwhere(a == label)
        if not len(points):
            raise ValueError(f'Source atlas label absent: {label}')
        centroid = nib.affines.apply_affine(ai.affine, points.mean(0))
        distances.append(float(tree.query(centroid)[0]))
    return np.asarray(distances)


def mask_dice(own_spec, registered_spec):
    own, a = image(own_spec); other, b = image(registered_spec)
    if a.shape != b.shape or not np.allclose(own.affine, other.affine, atol=1e-5, rtol=0):
        raise ValueError('Dice requires reviewed masks on the same target grid')
    a, b = a > 0, b > 0
    if not a.any() or not b.any():
        raise ValueError('Empty mask cannot establish pairwise overlap')
    return float(2 * np.count_nonzero(a & b) / (a.sum() + b.sum()))


def mantel(dice, similarity):
    n = len(dice); upper = np.triu_indices(n, 1)
    y = similarity[upper] if np.asarray(similarity).ndim == 2 else similarity
    if not 3 <= n <= 8 or len(y) != len(upper[0]):
        raise ValueError('Exact Mantel test supports 3..8 matched subjects')
    observed = float(spearmanr(dice[upper], y).statistic)
    if not np.isfinite(observed):
        raise ValueError('Constant or nonfinite Mantel inputs')
    null = np.array([spearmanr(dice[np.ix_(p, p)][upper], y).statistic
                     for p in itertools.permutations(range(n))])
    return {'rho': observed, 'p_one_sided': float(np.mean(null >= observed - 1e-12)),
            'p_two_sided': float(np.mean(abs(null) >= abs(observed) - 1e-12)),
            'n_permutations': len(null)}


def donor_null(delta, donors):
    n = len(delta)
    if not 3 <= n <= 7 or len(donors) != n or any(d == i or not 0 <= d < n for i, d in enumerate(donors)):
        raise ValueError('Exact donor null requires 3..7 subjects and non-self donors')
    scores = np.zeros_like(delta)
    for i in range(n):
        for d in range(n):
            if i != d:
                scores[i, d] = delta[i, d] - np.mean([delta[i, q] for q in range(n) if q not in (i, d)])
    observed = float(np.mean(scores[np.arange(n), donors]))
    choices = [[q for q in range(n) if q != i] for i in range(n)]
    null = np.array([scores[np.arange(n), p].mean() for p in itertools.product(*choices)])
    return {'obs_pull_toward_donor': observed, 'p_one_sided': float(np.mean(null >= observed - 1e-15)),
            'n_relabelings': len(null), 'null_q95': float(np.percentile(null, 95))}


def representation_perturbation(first, second, first_gradients, second_gradients, distance):
    """Fixed-reference mask sensitivity, not scan-rescan reliability."""
    n, parcels = first.shape[:2]; upper=np.triu_indices(parcels,1)
    a,b=np.log1p(first),np.log1p(second)
    def features(w,g):
        return {'G1':g[:,:,0], 'G1-G4':g.reshape(n,-1), 'node_strength':w.sum(2),
                'node_degree':(w>0).sum(2).astype(float), 'edges_log1p':w[:,upper[0],upper[1]]}
    fa,fb=features(a,first_gradients),features(b,second_gradients)
    between_subject=np.triu_indices(n,1); perturbation={};preservation={}
    for name in fa:
        x,y=fa[name],fb[name]
        within=float(abs(x-y).mean());between=float(x.std(0).mean())
        perturbation[name]={'within':within,'between':between,'ratio':within/max(between,1e-12)}
        sx=np.corrcoef((x-x.mean(0))/(x.std(0)+1e-12))[between_subject]
        sy=np.corrcoef((y-y.mean(0))/(y.std(0)+1e-12))[between_subject]
        preservation[name]={'r':float(np.corrcoef(sx,sy)[0,1]),'rho':float(spearmanr(sx,sy).statistic),
                            'disp':float(abs(sx-sy).mean())}
    errors={'gradient':np.linalg.norm(first_gradients-second_gradients,axis=2),
            'node':abs(a.sum(2)-b.sum(2)), 'edge':abs(a-b).mean(2)}
    denominators={'gradient':np.linalg.norm(first_gradients-first_gradients.mean(0),axis=2).std(0),
                  'node':a.sum(2).std(0),'edge':a.mean(2).std(0)}
    profiles={}
    for name,errors_ in errors.items():
        z=errors_/(denominators[name]+1e-12); values=[]
        for lower,upper_ in [(0,30),(30,60),(60,np.inf)]:
            per_subject=[]
            for i in range(n):
                select=(distance[i]>=lower)&(distance[i]<upper_)
                if select.any():per_subject.append(float(np.median(z[i,select])))
            values.append(float(np.mean(per_subject)) if per_subject else float('nan'))
        profiles[name]=dict(zip(('near','mid','far'),values));profiles[name]['ratio']=values[0]/max(values[2],1e-12)
    return {'perturbation_to_signal':perturbation,'structure_preservation':preservation,'spatial_error':profiles}


def run_geometry_controls(manifest, out):
    out = Path(out); initial = read_json(manifest); paths = [manifest, *manifest_paths(initial)]
    for p in paths: require_separate_output(out, Path(p).parent)
    frozen = freeze_inputs(paths)
    spec, nodes, matrices = load_manifest(manifest)
    cfg = spec['geometry_controls']; subjects = [s['id'] for s in spec['subjects']]; n = len(subjects)
    if cfg.get('geometry_reviewed') is not True or n > 7:
        raise ValueError('Explicit geometry review and <=7 subjects required')
    g, _, order, _ = cohort_embeddings(matrices, nodes, 'common')
    ix = {h: np.array([i for i in order if nodes[i]['hemisphere'] == h]) for h in ('L', 'R')}
    refs = {h: embed(matrices[:, 0].mean(0), v)[0] for h, v in ix.items()}
    def fixed(m, references=refs):
        return np.vstack([embed(m, ix[h], reference=references[h])[0][:, :4] for h in ('L', 'R')])
    def ident(features):
        result, _ = identification(features.reshape(3*n, -1), n)
        return {'acc': result['correct'], 'p': result['p'], 'null_mean': result['null_mean']}
    entries = {v['id']: v for v in cfg['subjects']}
    if len(entries) != len(cfg['subjects']) or set(entries) != set(subjects):
        raise ValueError('Spatial-control subject IDs must match exactly')
    distance = np.stack([parcel_distances(entries[s]['atlas'], entries[s]['distance_mask'],
                                         entries[s]['cortical_source_labels']) for s in subjects])
    if distance.shape != (n, len(order)):
        raise ValueError('Parcel-distance order must equal cortical L-then-R order')
    median = np.median(distance, axis=0); rng = np.random.default_rng(cfg.get('seed', 20261015))
    draws = cfg.get('random_subsets', 200)
    if not isinstance(draws, int) or not 1 <= draws <= 10000:
        raise ValueError('random_subsets must be 1..10000')
    result = {'full': {**ident(g), 'n_parcels': len(order)},
              'distance_mm': {'median_of_parcel_medians': float(np.median(median)),
                              'q': np.percentile(median, [10, 25, 50, 75, 90]).tolist()}}
    labels = np.repeat(np.arange(n), 3)
    for name, select in [('near_lt30mm', median < 30), ('mid_30_60mm', (median >= 30) & (median < 60)), ('far_ge60mm', median >= 60)]:
        count = int(select.sum())
        if count < 2: raise ValueError('Distance bin has fewer than two parcels')
        measured = ident(g[:, :, select]); random_acc = []
        for _ in range(draws):
            chosen = rng.choice(len(order), count, replace=False)
            sim = np.corrcoef(g[:, :, chosen].reshape(n*3, -1)); np.fill_diagonal(sim, -np.inf)
            random_acc.append(int(np.count_nonzero(labels == labels[np.argmax(sim, axis=1)])))
        result[name] = {**measured, 'n_parcels': count, 'random_subset_acc_median': float(np.median(random_acc)),
                        'random_subset_acc_q05_q95': np.percentile(random_acc, [5, 95]).tolist(),
                        'frac_random_ge': float(np.mean(np.asarray(random_acc) >= measured['acc']))}
    base_sim = np.corrcoef(g.reshape(n*3, -1)); result['swap'] = {}
    for name, arm in cfg.get('swaps', {}).items():
        t = {'post1': 1, 'post2': 2}[arm['timepoint']]
        mapping = {v['id']: v for v in arm['subjects']}
        if len(mapping) != n or set(mapping) != set(subjects): raise ValueError('Swap subject pairing invalid')
        changed = g.copy()
        for si, s in enumerate(subjects): changed[si, t] = fixed(control_matrix(mapping[s], nodes))
        sim = np.corrcoef(changed.reshape(n*3, -1)); delta = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j: delta[i, j] = (sim[3*i+t, 3*j:3*j+3] - base_sim[3*i+t, 3*j:3*j+3]).mean()
        donors = [subjects.index(mapping[s]['donor']) for s in subjects]
        result['swap'][name] = {**donor_null(delta, donors), 'identification': ident(changed)}
    # Own target masks and independently registered donor masks are never re-registered here.
    dice = np.full((n, n), np.nan)
    for pair in cfg.get('registered_pairs', []):
        i, j = subjects.index(pair['donor']), subjects.index(pair['target'])
        if i == j or np.isfinite(dice[i, j]): raise ValueError('Duplicate/self registered mask pair')
        dice[i, j] = mask_dice(entries[subjects[j]]['overlap_mask'], pair['mask'])
    if cfg.get('registered_pairs'):
        if not np.isfinite(dice[~np.eye(n, dtype=bool)]).all(): raise ValueError('All directed mask pairs required')
        dice = (dice + dice.T)/2; similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j: similarity[i, j] = base_sim[np.ix_([3*i+1, 3*i+2], [3*j+1, 3*j+2])].mean()
        result['mantel'] = {**mantel(dice, similarity), 'dice_range': [float(np.nanmin(dice)), float(np.nanmax(dice))]}
    if all('unmasked_preoperative' in entries[s] for s in subjects):
        pre = np.stack([control_matrix(entries[s]['unmasked_preoperative'], nodes) for s in subjects])
        if any(entries[s]['unmasked_preoperative']['mask_scheme'] != 'none' for s in subjects):
            raise ValueError('Exclusion-cost comparator must be unmasked')
        r2 = {h: embed(pre.mean(0), v)[0] for h, v in ix.items()}
        a = np.stack([fixed(m, r2) for m in pre]); b = np.stack([fixed(m, r2) for m in matrices[:, 0]])
        error = np.linalg.norm(a-b, axis=2)
        sd = np.linalg.norm(a-a.mean(0), axis=2).std(0) + 1e-12
        z = error/sd; bins = [0, 15, 30, 45, 60, 75, 200]; profile = []
        result['representation_robustness']=representation_perturbation(
            pre[:,order][:,:,order],matrices[:,0][:,order][:,:,order],a,b,distance)
        for lo, hi in zip(bins[:-1], bins[1:]):
            select = (distance >= lo) & (distance < hi)
            profile.append({'bin_mm': [lo, hi], 'n': int(select.sum()), 'median_err_sd': float(np.median(z[select])) if select.any() else None})
        result['distance_error'] = {'profile': profile, 'spearman_err_vs_dist': float(spearmanr(distance.ravel(), z.ravel()).statistic)}
        if cfg.get('registered_pairs'):
            unmasked = np.stack([fixed(m) for m in pre]).reshape(n, -1)
            sp = np.corrcoef(unmasked); sq = np.corrcoef(g[:, 2].reshape(n, -1)); upper = np.triu_indices(n, 1)
            x, y = rankdata(sp[upper]), rankdata(sq[upper]); residual = y - np.polyval(np.polyfit(x, y, 1), x)
            result['mantel_followup'] = {'preop_NOMASK': mantel(dice, sp), 'postop2_shared_mask': mantel(dice, sq),
                                        'postop2_given_preopNOMASK': mantel(dice, residual)}
    checks = expected_checks(spec, {'geometry_result': result}, [])
    assert_inputs_unchanged(frozen); passed = all(c['passed'] for c in checks)
    result.update(checks=checks, source_bytes_unchanged=True, status='completed' if passed else 'failed_replay',
                  warning='Spatial distance is centroid to mask-voxel centre. Registered geometry requires independent QC. Weak/null mask associations do not establish biological specificity.')
    np.savez_compressed(out/'geometry_controls.npz', distances_mm=distance, directed_mask_dice=dice)
    write_json(out/'geometry_result.json', clean(result))
    record(out, 'geometry-controls', paths, {'random_subsets': draws}, sorted(out.glob('*')), result['status'])
    if not passed: raise ValueError('Spatial controls differ from frozen outputs')
