"""Exact small-sample boundaries and explicit, seeded Monte Carlo power.

These calculations concern a statistical design, not clinical prediction. The
generating parameter is a Pearson correlation of bivariate-normal observations;
the decision rule is a two-sided Spearman permutation test.
"""
from itertools import permutations
import html
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .provenance import (assert_inputs_unchanged, freeze_inputs, read_json, record,
                         require_separate_output, write_json)


def family_hit_null(nulls, observed_p, alpha=.05):
    """Shared relabelling, with ties and dependence between tests preserved.

    Columns must represent the SAME participant permutation for every row.
    No independence approximation and no nominal alpha-times-test-count shortcut.
    """
    nulls = np.asarray(nulls, dtype=float)
    observed_p = np.asarray(observed_p, dtype=float)
    if (nulls.ndim != 2 or nulls.shape[0] != len(observed_p)
            or not nulls.size or not np.isfinite(nulls).all()
            or not np.isfinite(observed_p).all() or not 0 < alpha < 1):
        raise ValueError('Finite aligned permutation rows and observed probabilities required')
    counts = np.zeros(nulls.shape[1], dtype=int)
    for values in abs(nulls):
        p = (len(values)-np.searchsorted(np.sort(values), values-1e-12, side='left'))/len(values)
        counts += p < alpha
    observed = int(np.sum(observed_p < alpha))
    unique, frequency = np.unique(counts, return_counts=True)
    return {
        'threshold': f'p < {alpha}', 'observed_hits': observed,
        'expected_hits': float(counts.mean()),
        'observed_to_expected': float(observed/counts.mean()) if counts.mean() else None,
        'null_sd': float(counts.std(ddof=0)),
        'p_null_ge_observed': float(np.mean(counts >= observed)),
        'percentile_strictly_below_observed': float(np.mean(counts < observed)),
        'n_shared_permutations': len(counts),
        'histogram': [{'hits': int(k), 'permutations': int(v)} for k,v in zip(unique,frequency)],
        'method': 'Exact row-conditional p values under shared participant relabelling; inclusive tails; strict p threshold',
    }


def rank_numerator(first, second):
    """Integer numerator of untied Spearman rho; denominator n*(n*n-1)."""
    first, second = np.asarray(first), np.asarray(second)
    n = first.shape[-1]
    return n*(n*n-1)-6*np.sum((first-second)**2, axis=-1)


def null_critical(n, alpha, draws, seed):
    if not 3 <= n <= 100 or not 0 < alpha < 1 or draws < 100:
        raise ValueError('Invalid null specification')
    denominator = n*(n*n-1)
    if n <= 8:
        order = np.array(list(permutations(range(n))), dtype=np.int64)
        absolute = abs(rank_numerator(order, np.arange(n)))
        method = 'exact_enumeration'
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed,n,0]))
        # Chunking bounds memory even for larger prospective sample sizes.
        absolute = np.empty(draws, dtype=np.int64)
        for start in range(0, draws, 10000):
            size = min(10000, draws-start)
            order = rng.random((size,n)).argsort(axis=1)
            absolute[start:start+size] = abs(rank_numerator(order, np.arange(n)))
        method = 'sampled_permutation_null'
    support, count = np.unique(absolute, return_counts=True)
    tail = np.cumsum(count[::-1])[::-1]/len(absolute)
    eligible = np.flatnonzero(tail <= alpha)
    threshold = int(support[eligible[0]]) if len(eligible) else denominator+1
    return {'n': n, 'method': method, 'null_draws': len(absolute),
            'critical_absolute_numerator': threshold, 'denominator': denominator,
            'critical_absolute_spearman': threshold/denominator,
            'achieved_null_probability': float(tail[eligible[0]]) if len(eligible) else 0.}


def normal_spearman_power(n, generating_pearson, critical, draws, seed, stream=0):
    if not -1 <= generating_pearson <= 1 or draws < 100:
        raise ValueError('Invalid generating correlation or replicate count')
    rng = np.random.default_rng(np.random.SeedSequence([seed,n,stream,1]))
    hits = 0
    for start in range(0, draws, 10000):
        size = min(10000, draws-start)
        samples = rng.standard_normal((size,n,2))
        x = samples[:,:,0]
        y = generating_pearson*x + math.sqrt(max(0.,1-generating_pearson**2))*samples[:,:,1]
        rx = x.argsort(axis=1).argsort(axis=1)
        ry = y.argsort(axis=1).argsort(axis=1)
        hits += int(np.sum(abs(rank_numerator(rx,ry)) >= critical))
    value = hits/draws
    se = math.sqrt(value*(1-value)/draws)
    return {'n': n, 'generating_pearson': generating_pearson, 'replicates': draws,
            'detections': hits, 'power': value, 'monte_carlo_se': se,
            'mc_95_interval_normal': [max(0.,value-1.96*se),min(1.,value+1.96*se)]}


def n6_ordering_audit():
    x = np.arange(6)
    order = np.array(list(permutations(x)))
    numerator = rank_numerator(order,x)
    rho = numerator/210.
    loo = []
    for k in range(6):
        keep = x != k
        ranks = rankdata(order[:,keep], axis=1)-1
        loo.append(rank_numerator(ranks,np.arange(5))/120.)
    loo = np.asarray(loo).T
    groups = {}
    for label, selected in (
        ('abs_rho_ge_0.942857',abs(numerator)>=198),
        ('abs_rho_ge_0.885714',abs(numerator)>=186),
        ('abs_rho_eq_0.828571',abs(numerator)==174)):
        values = loo[selected]
        groups[label] = {'count': int(selected.sum()), 'positive_tail': int(np.sum(selected&(rho>0))),
            'loo_abs_range': [float(abs(values).min()),float(abs(values).max())],
            'any_strict_sign_flip_count': int(np.sum(np.any(values*rho[selected,None]<-1e-12,axis=1)))}
    selected = abs(rho)<.6
    product = loo[selected]*rho[selected,None]
    return {'total_orderings': 720, 'groups': groups,
        'weak_abs_rho_lt_06_count': int(selected.sum()),
        'weak_any_strict_sign_flip_count': int(np.sum(np.any(product < -1e-12,axis=1))),
        'weak_any_sign_flip_or_zero_count': int(np.sum(np.any(product <= 1e-12,axis=1))),
        'mean_abs_rho': float(abs(rho).mean()),
        'p_abs_rho_ge_07': float(np.mean(abs(rho)>=.7))}


def run_power(config,out):
    out = Path(out); require_separate_output(out,Path(config).parent)
    frozen = freeze_inputs([config]); spec = read_json(config)
    if spec.get('schema_version') != 1 or spec.get('scope') != 'statistical_design_audit':
        raise ValueError('Explicit statistical_design_audit config required')
    seed = spec.get('seed')
    if not isinstance(seed,int) or not 0 <= seed < 2**32:
        raise ValueError('Explicit uint32 seed required')
    alpha = float(spec.get('alpha',.05)); target = float(spec.get('target_power',.8))
    nmin,nmax = spec.get('n_min',6),spec.get('n_max',30)
    draws = spec.get('power_replicates',200000); sizedraws = spec.get('sample_size_replicates',100000)
    nulldraws = spec.get('null_draws_above_n8',400000)
    if not all(isinstance(v,int) for v in (nmin,nmax,draws,sizedraws,nulldraws)) or not 3 <= nmin <= nmax <= 100:
        raise ValueError('Integer sample-size range 3..100 required')
    if not .001 <= alpha <= .2 or not .5 <= target < 1 or not all(100 <= v <= 1000000 for v in (draws,sizedraws,nulldraws)):
        raise ValueError('Invalid simulation limits')
    correlations = spec.get('generating_pearson', [.3,.5,.6,.7,.8,.886,.9,.943,.95])
    sizes = spec.get('sample_size_generating_pearson',[.943,.886,.829,.771,.714,.6])
    if (not correlations or len(correlations)>32 or len(sizes)>32
            or any(not np.isfinite(v) or not 0<=v<=1 for v in correlations+sizes)):
        raise ValueError('Specify at most 32 finite nonnegative generating correlations per audit')
    nulls = {6:null_critical(6,alpha,nulldraws,seed)}
    powers = []
    for i,r in enumerate(correlations):
        print(f'[step power {i+1}/{len(correlations)}] n=6 generating Pearson r={r}',flush=True)
        powers.append(normal_spearman_power(6,r,nulls[6]['critical_absolute_numerator'],draws,seed,i))
    trajectories = {str(r):[] for r in sizes}; first = {str(r):None for r in sizes}
    for n in range(nmin,nmax+1):
        pending = [i for i,r in enumerate(sizes) if first[str(r)] is None]
        if not pending: break
        print(f'[step sample-size n={n}] {len(pending)} generating correlations remaining',flush=True)
        if n not in nulls: nulls[n] = null_critical(n,alpha,nulldraws,seed)
        for i in pending:
            r = sizes[i]
            point = normal_spearman_power(n,r,nulls[n]['critical_absolute_numerator'],sizedraws,seed,1000+i)
            trajectories[str(r)].append(point)
            if point['power'] >= target: first[str(r)] = n
    result = {'status':'computed_statistical_design_audit','seed':seed,'alpha':alpha,
        'target_power':target,'n6_exact_orderings':n6_ordering_audit(),
        'power_n6':powers,'sample_size_first_estimate_at_target':first,
        'sample_size_trajectories':trajectories,'null_thresholds':list(nulls.values()),
        'warning':'Generating r is Pearson correlation under a bivariate-normal model, not observed Spearman rho. Sampled nulls above n=8 and power curves have Monte Carlo error. First threshold crossing is an estimate, not a guaranteed recruitment target. Original seeds must be known for bitwise replay. No patient-image validation is implied.'}
    assert_inputs_unchanged(frozen); result['source_bytes_unchanged'] = True
    write_json(out/'power_result.json',result)
    rows = ''.join(f'<tr><td>{p["generating_pearson"]}</td><td>{p["power"]:.4f}</td><td>{p["monte_carlo_se"]:.5f}</td></tr>' for p in powers)
    table = ''.join(f'<tr><td>{html.escape(r)}</td><td>{n if n is not None else "Not reached"}</td></tr>' for r,n in first.items())
    (out/'power_review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Statistical design audit</title><style>body{font:15px system-ui;background:white;color:#222;margin:28px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}p{max-width:900px}</style><h1>Statistical design audit</h1><p>'+html.escape(result['warning'])+'</p><h2>Detection probability at n=6</h2><table><tr><th>Generating Pearson r</th><th>Power</th><th>Monte Carlo SE</th></tr>'+rows+'</table><h2>First estimated sample size at target power</h2><table><tr><th>Generating Pearson r</th><th>n</th></tr>'+table+'</table></html>',encoding='utf-8')
    record(out,'power-study',[config],spec,[out/'power_result.json',out/'power_review.html'],result['status'])
