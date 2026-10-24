"""Lateralized and symptom-dominant-side audits of reviewed four-axis gradients."""
from itertools import combinations, permutations
from pathlib import Path
import numpy as np

from .comparative import NETWORKS, clinical_values
from .provenance import (assert_inputs_unchanged, freeze_inputs, read_json, record,
                         require_separate_output, write_json)
from .study import (clean, cohort_embeddings, expected_checks, load_manifest,
                    manifest_paths, write_csv)
from .study_metrics import geometric_median, ranks

SESSION_METRICS = ('eccentricity_mean', 'eccentricity_median', 'centroid_eccentricity',
    'dispersion_4d_mean', 'dispersion_4d_geomedian',
    *(f'G{a}_{s}' for a in range(1, 5) for s in ('position_mean', 'position_median', 'dispersion_sd', 'dispersion_mad')))
INTERVAL_METRICS = ('parcel_displacement_mean', 'parcel_displacement_median', 'centroid_shift', 'geomedian_shift')
DEFINITIONS = ('left_only', 'right_only', 'hemisphere_parcel_weighted',
               'hemisphere_equal_weighted', 'historical_bilateral_pooled')
SYMPTOM_DEFINITIONS = ('symptom_contralateral', 'symptom_ipsilateral',
    'contralateral_minus_ipsilateral', 'contralateral_to_ipsilateral_change_ratio')


def session_geometry(g, selected, whole, historical=False):
    points = g[selected]; center = points.mean(0); origin = g[whole].mean(0)
    if len(points) < 2: raise ValueError('Each hemisphere-network needs at least two parcels')
    distances = np.linalg.norm(points-origin, axis=1)
    residual = points-center
    result = {'eccentricity_mean': float(distances.mean()), 'eccentricity_median': float(np.median(distances)),
        'centroid_eccentricity': float(np.linalg.norm(center-origin)),
        'dispersion_4d_mean': float(np.linalg.norm(residual, axis=1).mean()),
        'dispersion_4d_geomedian': float(np.median(np.linalg.norm(points-geometric_median(points), axis=1)))}
    for a in range(4):
        position = points[:, a] if historical else points[:, a]-origin[a]
        result[f'G{a+1}_position_mean'] = float(position.mean())
        result[f'G{a+1}_position_median'] = float(np.median(position))
        result[f'G{a+1}_dispersion_sd'] = float(np.std(points[:, a] if historical else residual[:, a], ddof=1))
        deviations = points[:, a]-np.median(points[:, a]) if historical else residual[:, a]
        result[f'G{a+1}_dispersion_mad'] = float(np.median(np.abs(deviations)))
    return result


def interval_geometry(first, second):
    displacement = np.linalg.norm(second-first, axis=1)
    return {'parcel_displacement_mean': float(displacement.mean()),
        'parcel_displacement_median': float(np.median(displacement)),
        'centroid_shift': float(np.linalg.norm(second.mean(0)-first.mean(0))),
        'geomedian_shift': float(np.linalg.norm(geometric_median(second)-geometric_median(first)))}


def safety_class(definition, metric):
    if definition == 'historical_bilateral_pooled':
        return 'interpretable_parcelwise_rotation_invariant' if metric.startswith('parcel_displacement_') else 'not_interpretable_independent_coordinate_pooling'
    if '_position_' in metric: return 'signed_axis_reference_dependent'
    if '_dispersion_s' in metric or '_dispersion_mad' in metric: return 'axis_identity_dependent'
    return 'interpretable_rotation_invariant_within_hemisphere'


def lateral_values(gradients, networks, hemispheres, subjects):
    g = np.asarray(gradients); networks, hemispheres = np.asarray(networks), np.asarray(hemispheres)
    if g.shape != (len(subjects), 3, len(networks), 4) or len(hemispheres) != len(networks) or not np.isfinite(g).all():
        raise ValueError('Expected finite participant x three visits x parcel x four-axis gradients')
    rows = []
    for si, subject in enumerate(subjects):
        for network in NETWORKS:
            selected = networks == network
            by_side, intervals, counts = {}, {}, {}
            for h in ('L', 'R'):
                hemi = hemispheres == h; mask = selected & hemi; counts[h] = int(mask.sum())
                by_side[h] = [session_geometry(g[si, t], mask, hemi) for t in (1, 2)]
                intervals[h] = interval_geometry(g[si, 1, mask], g[si, 2, mask])
            for definition in DEFINITIONS:
                if definition in ('left_only', 'right_only'):
                    h = 'L' if definition == 'left_only' else 'R'
                    s, v = by_side[h], intervals[h]
                elif definition == 'historical_bilateral_pooled':
                    s = [session_geometry(g[si, t], selected, np.ones(len(networks), bool), True) for t in (1, 2)]
                    v = interval_geometry(g[si, 1, selected], g[si, 2, selected])
                else:
                    weight = counts['L']/sum(counts.values()) if definition == 'hemisphere_parcel_weighted' else .5
                    s = [{k: weight*by_side['L'][t][k]+(1-weight)*by_side['R'][t][k] for k in SESSION_METRICS} for t in (0, 1)]
                    v = {k: weight*intervals['L'][k]+(1-weight)*intervals['R'][k] for k in INTERVAL_METRICS}
                for metric in SESSION_METRICS+INTERVAL_METRICS:
                    interval = metric in INTERVAL_METRICS
                    rows.append({'subject': subject, 'definition': definition, 'network': network, 'metric': metric,
                        'post1': None if interval else s[0][metric], 'post2': None if interval else s[1][metric],
                        'delta': v[metric] if interval else s[1][metric]-s[0][metric],
                        'safety_class': safety_class(definition, metric)})
    return rows


def bh(p):
    p = np.asarray(p); order = np.argsort(p); q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1])
    return q


def correlation_null(values, outcome):
    x, y = ranks(values), ranks(outcome)
    order = np.array(list(permutations(range(len(y)))))
    observed, null = float(y@x), y[order]@x
    return observed, float(np.mean(abs(null) >= abs(observed)-1e-12)), null


def adjusted(rows, nulls, p_key, observed_key, suffix, group_keys):
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(tuple(row[k] for k in group_keys), []).append(i)
    for indices in groups.values():
        p = [rows[i][p_key] for i in indices]
        q = bh(p)
        arrays = [nulls[i] for i in indices]
        if len({a.shape for a in arrays}) != 1:
            raise ValueError('Null orderings differ; never truncate them to manufacture a shared maxT null')
        null = np.stack(arrays); valid = np.isfinite(null).all(0)
        if not valid.any(): raise ValueError('No common valid null ordering')
        maximum = abs(null[:, valid]).max(0)
        for j, i in enumerate(indices):
            rows[i]['q_bh_'+suffix] = float(q[j])
            rows[i]['p_maxT_'+suffix] = float(np.mean(maximum >= abs(rows[i][observed_key])-1e-12))


def symptom_value(left, right, metric, right_contralateral, definition):
    c, i = (right, left) if right_contralateral else (left, right)
    if definition == 'symptom_contralateral': return c['delta']
    if definition == 'symptom_ipsilateral': return i['delta']
    if definition == 'contralateral_minus_ipsilateral': return c['delta']-i['delta']
    if definition != SYMPTOM_DEFINITIONS[-1]: raise ValueError('Unknown symptom definition')
    if metric in INTERVAL_METRICS: return c['delta']/i['delta'] if i['delta'] > 0 else float('nan')
    if '_position_' in metric: return float('nan')
    values = [c['post1'], c['post2'], i['post1'], i['post2']]
    if not np.isfinite(values).all() or min(values) <= 0: return float('nan')
    return (c['post2']/c['post1'])/(i['post2']/i['post1'])


def run_lateralized(manifest, out):
    out = Path(out); initial = read_json(manifest); paths = [manifest, *manifest_paths(initial)]
    for p in paths: require_separate_output(out, Path(p).parent)
    frozen = freeze_inputs(paths); spec, nodes, matrices = load_manifest(manifest)
    subjects = [s['id'] for s in spec['subjects']]
    g, _, order, _ = cohort_embeddings(matrices, nodes, 'loso')
    values = lateral_values(g, [nodes[i]['network'] for i in order], [nodes[i]['hemisphere'] for i in order], subjects)
    outcomes = {name: clinical_values(spec, name) for name in spec['clinical']}
    lookup = {(r['subject'], r['network'], r['metric'], r['definition']): r for r in values}
    rows, nulls, contrasts, contrast_nulls = [], [], [], []
    for definition in DEFINITIONS:
        for net in NETWORKS:
            for metric in SESSION_METRICS+INTERVAL_METRICS:
                x = np.array([lookup[s, net, metric, definition]['delta'] for s in subjects])
                for name, y in outcomes.items():
                    rho, p, null = correlation_null(x, y)
                    rows.append({'definition': definition, 'network': net, 'metric': metric, 'outcome': name,
                        'safety_class': safety_class(definition, metric), 'rho': rho, 'p_exact': p,
                        'mean_delta': float(x.mean()), 'median_delta': float(np.median(x)),
                        'direction_consistency': int(max((x > 0).sum(), (x < 0).sum()))})
                    nulls.append(null)
    adjusted(rows, nulls, 'p_exact', 'rho', 'definition_outcome', ('definition', 'outcome'))
    adjusted(rows, nulls, 'p_exact', 'rho', 'all_definitions_outcome', ('outcome',))
    indexed = {(r['definition'], r['network'], r['metric'], r['outcome']): i for i, r in enumerate(rows)}
    for net in NETWORKS:
        for metric in SESSION_METRICS+INTERVAL_METRICS:
            for name in outcomes:
                li, ri = (indexed[d, net, metric, name] for d in ('left_only', 'right_only'))
                a, b = rows[li], rows[ri]; delta = a['rho']-b['rho']; null = nulls[li]-nulls[ri]
                contrasts.append({'network': net, 'metric': metric, 'outcome': name,
                    'rho_left': a['rho'], 'rho_right': b['rho'], 'p_left_exact': a['p_exact'], 'p_right_exact': b['p_exact'],
                    'rho_left_minus_right': delta, 'p_lateral_difference_exact': float(np.mean(abs(null) >= abs(delta)-1e-12))})
                contrast_nulls.append(null)
    adjusted(contrasts, contrast_nulls, 'p_lateral_difference_exact', 'rho_left_minus_right', 'lateral_difference_outcome', ('outcome',))
    write_csv(out/'lateral_subject_values.csv', values); write_csv(out/'lateral_correlations.csv', rows)
    write_csv(out/'left_right_contrasts.csv', contrasts)
    arrays = {'lateral_values': values, 'lateral_correlations': rows, 'left_right_contrasts': contrasts}
    result = {'n_lateral_tests': len(rows), 'n_contrast_tests': len(contrasts),
        'warning': 'All associations are exploratory. Independently embedded hemispheres cannot be interpreted as one shared geometry; signed axes depend on the reference.'}
    side = spec.get('symptom_dominant_side')
    if side:
        if set(side) != set(subjects) or any(v not in ('L', 'R') for v in side.values()) or spec.get('symptom_side_reviewed') is not True:
            raise ValueError('Explicit reviewed symptom-dominant side required for every participant')
        right = np.array([side[s] == 'L' for s in subjects])
        assignments = list(combinations(range(len(subjects)), int(right.sum())))
        srows, snulls, side_nulls, svalues, exclusions = [], [], [], [], []
        for net in NETWORKS:
            for metric in SESSION_METRICS+INTERVAL_METRICS:
                left = [lookup[s, net, metric, 'left_only'] for s in subjects]
                right_values = [lookup[s, net, metric, 'right_only'] for s in subjects]
                for definition in SYMPTOM_DEFINITIONS:
                    def vector(assignment):
                        return np.array([symptom_value(a, b, metric, i in assignment, definition) for i, (a, b) in enumerate(zip(left, right_values))])
                    x = vector(set(np.where(right)[0]))
                    if not np.isfinite(x).all():
                        exclusions.append({'network': net, 'metric': metric, 'definition': definition, 'reason': 'Undefined ratio; signed positions or nonpositive denominator'})
                        continue
                    svalues.extend({'subject': s, 'network': net, 'metric': metric, 'definition': definition, 'value': float(v)} for s, v in zip(subjects, x))
                    for name, y in outcomes.items():
                        rho, p, null = correlation_null(x, y)
                        side_null = []
                        for assignment in assignments:
                            try: side_null.append(float(ranks(vector(set(assignment)))@ranks(y)))
                            except ValueError: side_null.append(float('nan'))
                        side_null = np.asarray(side_null); valid = np.isfinite(side_null)
                        srows.append({'network': net, 'metric': metric, 'definition': definition, 'outcome': name,
                            'rho': rho, 'p_outcome_exact': p, 'p_side_label_exact': float(np.mean(abs(side_null[valid]) >= abs(rho)-1e-12)),
                            'mean_value': float(x.mean()), 'median_value': float(np.median(x)),
                            'direction_consistency': int(max((x > 0).sum(), (x < 0).sum()))})
                        snulls.append(null); side_nulls.append(side_null)
        for grouping, suffix in [(('definition', 'outcome'), 'within_definition'), (('outcome',), 'all_definitions')]:
            adjusted(srows, snulls, 'p_outcome_exact', 'rho', 'outcome_'+suffix, grouping)
            adjusted(srows, side_nulls, 'p_side_label_exact', 'rho', 'side_label_'+suffix, grouping)
        arrays.update(symptom_values=svalues, symptom_correlations=srows)
        result.update(n_symptom_tests=len(srows), n_side_assignments=len(assignments), side_label_p_floor=1/len(assignments),
            excluded_symptom_definitions=exclusions, symptom_side_definition='Clinically documented symptom-dominant side, not symptom onset side; whole-body motor score is not a lateralized outcome')
        write_csv(out/'symptom_subject_values.csv', svalues); write_csv(out/'symptom_correlations.csv', srows)
    checks = expected_checks(spec, arrays, [])
    assert_inputs_unchanged(frozen); passed = all(c['passed'] for c in checks)
    result.update(checks=checks, source_bytes_unchanged=True, status='completed' if passed else 'failed_replay')
    write_json(out/'lateralized_result.json', clean(result))
    record(out, 'lateralized-study', paths, {'reference': 'loso', 'symptom_side_reviewed': spec.get('symptom_side_reviewed', False)}, sorted(out.glob('*')), result['status'])
    if not passed: raise ValueError('Lateralized audit differs from frozen outputs')
