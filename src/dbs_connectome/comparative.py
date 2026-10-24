"""Explicit, hash-bound cross-cohort and mask-calibre sensitivity audits.

Historical pooled-coordinate definitions are preserved, not silently promoted to
the hemisphere-aware primary analysis. Descriptive replication is not prediction.
"""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

from .gradients import Settings, embed
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, read_json,
                         record, require_separate_output, write_json)
from .representations import visit_excluded_identification
from .study import (clean, cohort_embeddings, expected_checks, isolated_profiles,
                    load_manifest, manifest_paths, write_csv)

NETWORKS = ('Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default')


def rank_correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite([a, b]).all():
        raise ValueError('Paired finite vectors required; no automatic imputation')
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def agreement_icc(matrix):
    """Two-way random-effects, single-measure, absolute-agreement ICC(2,1)."""
    x = np.asarray(matrix, float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('ICC requires a finite subject-by-calibre table with >=2 per dimension')
    n, k = x.shape
    mean, row, col = x.mean(), x.mean(1), x.mean(0)
    msr = k*((row-mean)**2).sum()/(n-1)
    msc = n*((col-mean)**2).sum()/(k-1)
    mse = ((x-row[:, None]-col[None, :]+mean)**2).sum()/((n-1)*(k-1))
    denominator = msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denominator) if denominator else float('nan')


def linked_cohort(item, frozen_expected):
    if digest(item['manifest']) != item.get('sha256'):
        raise ValueError('Linked cohort manifest changed after review')
    spec, nodes, matrices = load_manifest(item['manifest'], allow_session_masks=True)
    if spec.get('comparative'):
        raise ValueError('Nested comparative studies are forbidden')
    policy = spec.get('legacy_zero_profile_policy', 'reject')
    if policy not in ('reject', 'brainspace_normalized_angle') or (policy != 'reject' and not frozen_expected):
        raise ValueError('Historical zero-profile handling requires frozen comparisons')
    return spec, nodes, matrices, policy != 'reject'


def clinical_values(spec, endpoint):
    if endpoint not in spec['clinical']:
        raise ValueError('Requested endpoint missing from linked cohort')
    values = [spec['clinical'][endpoint]['values'][s['id']] for s in spec['subjects']]
    if any(v is None for v in values):
        raise ValueError('Comparative audit needs complete explicitly defined outcomes; no imputation')
    return np.asarray(values, float)


def matched_metrics(matrices, gradients, nodes, order):
    """Historical 28-per-representation definition, with dimensionality explicit.

    The pooled bilateral centroid differs from the primary hemisphere-aware
    centroid. Within-network edge blocks include their zero diagonal, as in the
    frozen historical code. Degree is unthresholded positive-edge count.
    """
    g, m = np.asarray(gradients), np.asarray(matrices)
    if g.shape[:2] != m.shape[:2] or g.shape[2] != len(order) or g.shape[1] != 3:
        raise ValueError('Gradient/matrix session axes do not match')
    networks = np.array([nodes[i]['network'] for i in order])
    weights = np.log1p(m[:, :, order][:, :, :, order])
    out = {'gradient': {}, 'node': {}, 'edge': {}}
    for network in NETWORKS:
        sel = networks == network
        if not sel.any(): raise ValueError(f'Missing network: {network}')
        a, b = g[:, 1], g[:, 2]
        va, vb = a[:, sel], b[:, sel]
        e1 = np.median(np.linalg.norm(va-a.mean(1)[:, None], axis=2), axis=1)
        e2 = np.median(np.linalg.norm(vb-b.mean(1)[:, None], axis=2), axis=1)
        d1 = np.median(np.linalg.norm(va-va.mean(1)[:, None], axis=2), axis=1)
        d2 = np.median(np.linalg.norm(vb-vb.mean(1)[:, None], axis=2), axis=1)
        out['gradient'][network+'_ecc'] = e2-e1
        out['gradient'][network+'_disp'] = d2-d1
        out['gradient'][network+'_G1pos'] = vb[:, :, 0].mean(1)-va[:, :, 0].mean(1)
        out['gradient'][network+'_move'] = np.linalg.norm(vb-va, axis=2).mean(1)
        strength = weights[:, :, sel].sum(3).mean(2)
        degree = (weights[:, :, sel] > 0).sum(3).mean(2)
        if np.any(strength[:, 1] == 0) or np.any(degree[:, 1] == 0):
            raise ValueError('Zero baseline for relative node change; do not divide or impute')
        out['node'][network+'_strength'] = strength[:, 2]-strength[:, 1]
        out['node'][network+'_degree'] = degree[:, 2]-degree[:, 1]
        out['node'][network+'_str_rel'] = (strength[:, 2]-strength[:, 1])/strength[:, 1]
        out['node'][network+'_deg_rel'] = (degree[:, 2]-degree[:, 1])/degree[:, 1]
        for target in ('Vis', 'SomMot', 'DorsAttn', 'Default'):
            block = weights[:, :, sel][:, :, :, networks == target].mean((2, 3))
            out['edge'][network+'-'+target] = block[:, 2]-block[:, 1]
    return out


def replication_summary(first, second, y_first, y_second):
    result, rows = {}, []
    for representation in ('gradient', 'node', 'edge'):
        if list(first[representation]) != list(second[representation]):
            raise ValueError('Metric identities/order differ across cohorts')
        keys = list(first[representation])
        a = np.array([rank_correlation(first[representation][k], y_first) for k in keys])
        b = np.array([rank_correlation(second[representation][k], y_second) for k in keys])
        valid = np.isfinite(a) & np.isfinite(b)
        for key, x, y, ok in zip(keys, a, b, valid):
            rows.append({'representation': representation, 'metric': key,
                         'first_rho': x, 'second_rho': y, 'included': bool(ok)})
        a, b = a[valid], b[valid]
        if not len(a): raise ValueError('No finite matched clinical correlations')
        top = np.argsort(-np.abs(a))[:5]
        result[representation] = {
            'n': len(a), 'sign_agree': float(np.mean(np.sign(a) == np.sign(b))),
            'rho_corr': float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and np.std(a)*np.std(b) > 0 else float('nan'),
            'top5_sign': float(np.mean(np.sign(a[top]) == np.sign(b[top]))),
            'first': a.tolist(), 'second': b.tolist(),
            'excluded_constant_metrics': [k for k, ok in zip(keys, valid) if not ok]}
    return result, rows


def calibre_summary(metrics_by_calibre, outcome):
    result, rows = {}, []
    for representation, suffixes in (('gradient', ('_ecc', '_move')), ('node', ('_strength',))):
        keys = [k for k in next(iter(metrics_by_calibre.values()))[representation] if k.endswith(suffixes)]
        iccs, spans = [], []
        for metric in keys:
            x = np.column_stack([v[representation][metric] for v in metrics_by_calibre.values()])
            rhos = np.array([rank_correlation(x[:, i], outcome) for i in range(x.shape[1])])
            icc = agreement_icc(x)
            span = float(np.nanmax(rhos)-np.nanmin(rhos)) if np.isfinite(rhos).any() else float('nan')
            iccs.append(icc); spans.append(span)
            for i, name in enumerate(metrics_by_calibre):
                rows.append({'representation': representation, 'metric': metric, 'calibre': name,
                             'rho': rhos[i], 'icc21_across_calibres': icc, 'rho_span': span})
        result[representation] = {'n': len(keys), 'icc_med': float(np.median(iccs)),
            'span_med': float(np.median(spans)), 'icc_all': iccs, 'span_all': spans}
    return result, rows


def exact_parcel_mapping(first_nodes, second_nodes):
    """Map by reviewed full parcel identity, never by inferred missing index."""
    first = [i for h in ('L', 'R') for i, n in enumerate(first_nodes) if n['tissue'] == 'cortex' and n['hemisphere'] == h]
    second = [i for h in ('L', 'R') for i, n in enumerate(second_nodes) if n['tissue'] == 'cortex' and n['hemisphere'] == h]
    names = [second_nodes[i]['name'] for i in second]
    if len(names) != len(set(names)) or len({first_nodes[i]['name'] for i in first}) != len(first):
        raise ValueError('Ambiguous cortical parcel names')
    mapping = []
    for index in first:
        node = first_nodes[index]
        if node['name'] not in names: raise ValueError('Missing cross-cohort parcel identity')
        j = names.index(node['name']); target = second_nodes[second[j]]
        if any(node[k] != target[k] for k in ('hemisphere', 'network', 'tissue')):
            raise ValueError('Cross-cohort parcel metadata mismatch')
        mapping.append(j)
    return np.array(first), np.array(second), np.array(mapping)


def transfer_audit(reference_nodes, reference_matrices, external_nodes, external_matrices, legacy=False):
    a, b, mapping = exact_parcel_mapping(reference_nodes, external_nodes)
    settings = Settings(); mean_a = reference_matrices[:, 0].mean(0); mean_b = external_matrices[:, 0].mean(0)
    ia = {h: a[[reference_nodes[i]['hemisphere'] == h for i in a]] for h in ('L', 'R')}
    ib = {h: b[[external_nodes[i]['hemisphere'] == h for i in b]] for h in ('L', 'R')}
    ra = {h: embed(mean_a, ix, settings)[0] for h, ix in ia.items()}
    rb = {h: embed(mean_b, ix, settings, legacy_zero_profiles=legacy)[0] for h, ix in ib.items()}
    ga = np.vstack(list(ra.values())); gb = np.vstack(list(rb.values()))[mapping]
    grad_signed = [float(np.corrcoef(ga[:, i], gb[:, i])[0, 1]) for i in range(4)]
    result = {'group_correspondence': {}, 'gradient_signed_correspondence': grad_signed,
              'parcel_names': [reference_nodes[i]['name'] for i in a],
              'external_cortical_indices_matching_internal': mapping.tolist(),
              'warning': 'Absolute group-map correlations are descriptive; independently estimated axes are not symptom labels. Standardization uses all external sessions and is not an out-of-sample classifier.'}
    upper = np.triu_indices(len(a), 1)
    for name in ('raw', 'log1p', 'rank'):
        x, y = mean_a[np.ix_(a, a)], mean_b[np.ix_(b, b)]
        if name == 'log1p': x, y = np.log1p(x), np.log1p(y)
        elif name == 'rank': x, y = rankdata(x.ravel()).reshape(x.shape), rankdata(y.ravel()).reshape(y.shape)
        node_r = float(np.corrcoef(x.sum(1), y.sum(1)[mapping])[0, 1])
        edge_r = float(np.corrcoef(x[upper], y[np.ix_(mapping, mapping)][upper])[0, 1])
        result['group_correspondence'][name] = {'gradient_G1_G4': np.abs(grad_signed).tolist(), 'node': abs(node_r), 'edge': abs(edge_r)}
    labels = np.repeat(np.arange(len(external_matrices)), 3); visits = np.tile(np.arange(3), len(external_matrices))
    result['reference_identification'] = {}
    for name, refs in (('own', rb), ('external', ra)):
        use = ib if name == 'own' else {h: b[mapping[[reference_nodes[i]['hemisphere'] == h for i in a]]] for h in ('L', 'R')}
        features = [np.vstack([embed(m, ix, settings, refs[h], legacy_zero_profiles=legacy)[0][:, :4] for h, ix in use.items()]).ravel() for subject in external_matrices for m in subject]
        result['reference_identification'][name] = visit_excluded_identification(features, labels, visits, True)
    return result


def normalization_margins(matrices, gradients, order):
    n = len(matrices); w = np.log1p(matrices[:, :, order][:, :, :, order]).reshape(3*n, len(order), len(order))
    upper = np.triu_indices(len(order), 1)
    features = {'G1': gradients[:, :, :, 0].reshape(3*n, -1), 'G1-G4': gradients[:, :, :, :4].reshape(3*n, -1),
                'node_strength': w.sum(2), 'edges_log1p': w[:, upper[0], upper[1]]}
    result = {}; labels = np.repeat(np.arange(n), 3); visits = np.tile(np.arange(3), n)
    for name, values in features.items():
        result[name] = {}
        for mode in ('z', 'rank', 'center'):
            x = (values-values.mean(0))/(values.std(0)+1e-12) if mode == 'z' else (np.apply_along_axis(rankdata, 0, values) if mode == 'rank' else values-values.mean(0))
            result[name][mode] = visit_excluded_identification(x, labels, visits)['margin']
    return result


def run_comparative(manifest, out):
    out = Path(out); initial = read_json(manifest); paths = [manifest, *manifest_paths(initial)]
    for path in paths: require_separate_output(out, Path(path).parent)
    frozen = freeze_inputs(paths)
    spec, nodes, matrices = load_manifest(manifest)
    cfg = spec.get('comparative', {}); endpoint = cfg.get('endpoint')
    y = clinical_values(spec, endpoint)
    ids = [s['id'] for s in spec['subjects']]
    result = {'scope': 'descriptive_comparative_audits', 'clinical_endpoint': endpoint,
              'endpoint_definition': spec['clinical'][endpoint]['definition'],
              'warning': 'No prospective prediction or proof of superiority. Historical pooled bilateral centroids differ from the primary hemisphere-aware analysis.'}
    if cfg.get('external'):
        es, en, em, legacy = linked_cohort(cfg['external'], bool(spec.get('expected')))
        dimensions = cfg.get('replication_components', 4)
        if dimensions not in (4, 10): raise ValueError('Use four retained axes or explicitly labelled legacy ten-axis replay')
        ga, _, order, _ = cohort_embeddings(matrices, nodes, 'common', retain_all=True)
        gb, _, eo, _ = cohort_embeddings(em, en, 'common', legacy, retain_all=True)
        ye = clinical_values(es, endpoint)
        ma = matched_metrics(matrices, ga[:, :, :, :dimensions], nodes, order)
        mb = matched_metrics(em, gb[:, :, :, :dimensions], en, eo)
        result['replication'], rows = replication_summary(ma, mb, y, ye)
        result['replication_dimensions'] = dimensions
        result['external_endpoint_definition'] = es['clinical'][endpoint]['definition']
        result['external_isolated_profiles'] = isolated_profiles(em, en, [s['id'] for s in es['subjects']], ['pre', 'post1', 'post2'])
        write_csv(out/'matched_replication.csv', rows)
        if dimensions == 10:
            result['dimensionality_warning'] = 'Historical matched-replication code used all ten computed axes. These results must not be described as G1-G4.'
            result['four_axis_sensitivity'], rows4 = replication_summary(
                matched_metrics(matrices, ga[:, :, :, :4], nodes, order),
                matched_metrics(em, gb[:, :, :, :4], en, eo), y, ye)
            write_csv(out/'matched_replication_four_axes.csv', rows4)
        if cfg.get('transfer_reference'):
            if cfg.get('parcel_mapping_reviewed') is not True:
                raise ValueError('Cross-cohort parcel mapping requires explicit review')
            rs, rn, rm, rl = linked_cohort(cfg['transfer_reference'], bool(spec.get('expected')))
            if rl or any(s['visits'][0]['mask_scheme'] != 'none' for s in rs['subjects']):
                raise ValueError('Transfer reference must explicitly use unmasked preoperative matrices')
            result['transfer'] = transfer_audit(rn, rm, en, em, legacy)
            result['normalization_margins'] = normalization_margins(matrices, ga, order)
    if cfg.get('calibres'):
        calibres = cfg['calibres']
        if not 3 <= len(calibres) <= 8: raise ValueError('Specify three to eight explicit mask-calibre variants')
        metrics = {}; all_values = {}
        for name, item in calibres.items():
            cs, cn, cm, legacy = linked_cohort(item, bool(spec.get('expected')))
            if [s['id'] for s in cs['subjects']] != ids or cn != nodes or not np.array_equal(clinical_values(cs, endpoint), y):
                raise ValueError('Calibre study subject order, nodes or clinical outcomes differ')
            # Historical reference is each calibre's mean post1, not the preoperative template.
            reference_matrices = cm.copy(); reference_matrices[:, 0] = cm[:, 1]
            g, _, order, _ = cohort_embeddings(reference_matrices, cn, 'common', legacy)
            metrics[name] = matched_metrics(cm, g, cn, order)
            all_values[name] = metrics[name]
        result['calibre'], rows = calibre_summary(metrics, y)
        result['calibre_reference'] = 'Mean post1 within each calibre; four retained components; distinct from primary LOSO'
        result['calibre_schemes'] = {name: [v['mask_scheme'] for v in read_json(item['manifest'])['subjects'][0]['visits']] for name, item in calibres.items()}
        write_csv(out/'calibre_metric_stability.csv', rows)
        write_json(out/'calibre_subject_values.json', clean(all_values))
    if 'replication' not in result and 'calibre' not in result:
        raise ValueError('No comparative audit configured')
    checks = expected_checks(spec, {'comparative_result': result}, [])
    assert_inputs_unchanged(frozen)
    passed = all(check['passed'] for check in checks)
    result.update(checks=checks, source_bytes_unchanged=True, status='completed' if passed else 'failed_replay')
    write_json(out/'comparative_result.json', clean(result))
    record(out, 'comparative-study', paths, cfg, sorted(out.glob('*')), result['status'])
    if not passed: raise ValueError('Comparative audit differs from frozen outputs')
