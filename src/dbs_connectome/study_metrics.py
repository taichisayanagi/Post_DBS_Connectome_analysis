"""Study-defined hemisphere geometry and small-cohort inference.

These functions preserve the definitions in the frozen hemisphere-aware study
analysis. They do not infer anatomy, symptom meaning, or prospective validity.
"""
from itertools import permutations
from math import factorial

import numpy as np
from scipy.stats import rankdata, norm


def ranks(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Ranks require one finite vector; missingness must be explicit")
    r = rankdata(x)
    r -= r.mean()
    length = np.linalg.norm(r)
    if length == 0:
        raise ValueError("Constant variable has no defined correlation")
    return r / length


def exact_family(features, outcome):
    """All n! permutations, common across a complete-case, two-sided family."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y) or not 3 <= len(y) <= 8:
        raise ValueError("Exact family requires 3..8 subjects and subjects-by-features")
    rx = np.column_stack([ranks(v) for v in x.T])
    ry = ranks(y)
    perm = np.asarray(list(permutations(range(len(y)))), dtype=int)
    observed = ry @ rx
    null = ry[perm] @ rx
    absolute = np.abs(null)
    p = (absolute >= np.abs(observed)[None, :] - 1e-12).mean(axis=0)
    max_null = absolute.max(axis=1)
    fwe = (max_null[:, None] >= np.abs(observed)[None, :] - 1e-12).mean(axis=0)
    order = np.argsort(p, kind="stable")
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    loo = np.full((len(y), x.shape[1]), np.nan)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        for j in range(x.shape[1]):
            try:
                loo[i, j] = ranks(x[keep, j]) @ ranks(y[keep])
            except ValueError:
                pass
    return {"rho": observed, "p_exact": p, "q_bh": q, "p_maxT": fwe,
            "p_greater": (null >= observed[None, :] - 1e-12).mean(0),
            "p_less": (null <= observed[None, :] + 1e-12).mean(0),
            "loo_rho": loo, "n_permutations": len(perm)}


def geometric_median(points, tolerance=1e-10):
    """Frozen study's Weiszfeld implementation, including its near-point stop."""
    current = points.mean(axis=0)
    for _ in range(1000):
        distances = np.linalg.norm(points - current, axis=1)
        if np.any(distances < tolerance):
            return points[np.argmin(distances)].copy()
        weights = 1.0 / distances
        updated = np.sum(points * weights[:, None], axis=0) / weights.sum()
        if np.linalg.norm(updated - current) < tolerance:
            return updated
        current = updated
    return current


def _geometry_inputs(g, networks, hemispheres):
    g = np.asarray(g, dtype=float)
    networks, hemispheres = np.asarray(networks), np.asarray(hemispheres)
    if g.shape != (len(networks), 4) or len(networks) != len(hemispheres) or not np.isfinite(g).all():
        raise ValueError("Expected finite parcel-by-four gradients with matching metadata")
    if set(hemispheres) != {"L", "R"}:
        raise ValueError("Both independently embedded cortical hemispheres are required")
    for network in sorted(set(networks)):
        if any(np.count_nonzero((networks == network) & (hemispheres == h)) < 2 for h in ("L", "R")):
            raise ValueError("Each network requires at least two parcels in each hemisphere")
    return g, networks, hemispheres


def session_metrics(g, networks, hemispheres):
    """20 measures/network; session hemisphere centroids, not reference centroids."""
    g, networks, hemispheres = _geometry_inputs(g, networks, hemispheres)
    output = {}
    for network in sorted(set(networks)):
        ecc, disp, pos, residual = [], [], [], []
        for h in ("L", "R"):
            hemi = hemispheres == h
            points = g[hemi & (networks == network)]
            whole = g[hemi].mean(axis=0)
            center = points.mean(axis=0)
            ecc.extend(np.linalg.norm(points - whole, axis=1))
            disp.extend(np.linalg.norm(points - center, axis=1))
            pos.extend(points - whole)
            residual.extend(points - center)
        pos, residual = np.asarray(pos), np.asarray(residual)
        output[network, "eccentricity_mean"] = float(np.mean(ecc))
        output[network, "eccentricity_median"] = float(np.median(ecc))
        output[network, "dispersion_4d_mean"] = float(np.mean(disp))
        output[network, "dispersion_4d_median"] = float(np.median(disp))
        for a in range(4):
            output[network, f"G{a+1}_position_mean"] = float(pos[:, a].mean())
            output[network, f"G{a+1}_position_median"] = float(np.median(pos[:, a]))
            output[network, f"G{a+1}_dispersion_sd"] = float(np.sqrt(np.sum(residual[:, a]**2) / (len(residual)-2)))
            output[network, f"G{a+1}_dispersion_mad"] = float(np.median(np.abs(residual[:, a])))
    return output


def interval_metrics(first, second, networks, hemispheres):
    first, networks, hemispheres = _geometry_inputs(first, networks, hemispheres)
    second, _, _ = _geometry_inputs(second, networks, hemispheres)
    output = {}
    for network in sorted(set(networks)):
        displacement, shifts, geoshifts = [], [], []
        for h in ("L", "R"):
            mask = (hemispheres == h) & (networks == network)
            a, b = first[mask], second[mask]
            displacement.extend(np.linalg.norm(b - a, axis=1))
            shifts.append(np.linalg.norm(b.mean(axis=0) - a.mean(axis=0)))
            geoshifts.append(np.linalg.norm(geometric_median(b) - geometric_median(a)))
        output[network, "parcel_displacement_mean"] = float(np.mean(displacement))
        output[network, "parcel_displacement_median"] = float(np.median(displacement))
        output[network, "centroid_shift_mean_hemispheres"] = float(np.mean(shifts))
        output[network, "centroid_shift_rms_hemispheres"] = float(np.sqrt(np.mean(np.asarray(shifts)**2)))
        output[network, "geomedian_shift_mean_hemispheres"] = float(np.mean(geoshifts))
    return output


def identification(features, subject_count, sampled_pairs=100000, seed=42, batch_size=10000):
    """Three visits/subject; fix baseline labels and permute each follow-up."""
    x = np.asarray(features, dtype=float)
    n = int(subject_count)
    if n < 3 or x.ndim != 2 or x.shape[0] != 3*n or not np.isfinite(x).all():
        raise ValueError("Identification requires a balanced subject-major three-visit array")
    if np.any(x.std(axis=1) == 0):
        raise ValueError("Constant feature vector: Pearson identification undefined")
    sim = np.corrcoef(x)
    np.fill_diagonal(sim, -np.inf)
    # Refuse ambiguous nearest neighbours rather than choosing a favourable tie.
    nearest = np.argmax(sim, axis=1)
    best = sim[np.arange(3*n), nearest]
    if np.any(np.sum(np.isclose(sim, best[:, None], rtol=0, atol=1e-12), axis=1) > 1):
        raise ValueError("Tied nearest neighbours require an explicit identification policy")
    actual = np.repeat(np.arange(n), 3)
    correct = int(np.count_nonzero(actual == actual[nearest]))
    upper=np.triu_indices(3*n,1)
    within=actual[upper[0]]==actual[upper[1]]
    margins=np.array([sim[i,actual==actual[i]].max()-sim[i,actual!=actual[i]].max() for i in range(3*n)])
    distribution = np.zeros(3*n+1, dtype=np.int64)
    total = factorial(n)**2
    if total <= 1000000:
        pp = np.asarray(list(permutations(range(n))), dtype=int)
        labels = np.tile(actual, (len(pp), 1))
        labels[:, 2::3] = pp
        for p in pp:
            labels[:, 1::3] = p
            score = (labels == labels[:, nearest]).sum(1)
            distribution += np.bincount(score, minlength=3*n+1)
        pvalue = distribution[correct:].sum() / total
        method = "exact_visit_stratified"
    else:
        if not isinstance(sampled_pairs, int) or not 1000 <= sampled_pairs <= 10000000:
            raise ValueError("Sampled identification null requires 1000..10000000 pairs")
        if not isinstance(batch_size,int) or not 1000<=batch_size<=200000:
            raise ValueError("Permutation batch size must be 1000..200000")
        rng = np.random.default_rng(seed)
        for start in range(0, sampled_pairs, batch_size):
            size = min(batch_size, sampled_pairs-start)
            labels = np.tile(actual, (size, 1))
            labels[:, 1::3] = np.argsort(rng.random((size, n)), axis=1)
            labels[:, 2::3] = np.argsort(rng.random((size, n)), axis=1)
            distribution += np.bincount((labels == labels[:, nearest]).sum(1), minlength=3*n+1)
        pvalue = (distribution[correct:].sum()+1) / (sampled_pairs+1)
        method = "sampled_visit_stratified_plus_one"
    return {"correct": correct, "n_sessions": 3*n, "nearest": nearest.tolist(),
            "p": float(pvalue), "null_method": method, "null_count": int(distribution.sum()),
            "null_distribution": distribution.tolist(), "seed": seed,"batch_size":batch_size,
            "null_mean":float(np.dot(np.arange(3*n+1),distribution)/distribution.sum()),
            "null_max":int(np.flatnonzero(distribution)[-1]),
            "margin_mean":float(margins.mean()),"margin_min":float(margins.min()),
            "within_mean":float(sim[upper][within].mean()),"within_min":float(sim[upper][within].min()),
            "between_mean":float(sim[upper][~within].mean()),"between_max":float(sim[upper][~within].max())}, sim


def fisher_pool(coefficients, sizes, family_size):
    r, n = np.asarray(coefficients, float), np.asarray(sizes, int)
    if r.ndim != 1 or len(r) < 2 or r.shape != n.shape or np.any(n <= 3) or np.any(np.abs(r) >= 1) or not np.isfinite(r).all():
        raise ValueError("Fisher pooling requires finite |r|<1 and n>3 in each independent cohort")
    if not isinstance(family_size, int) or family_size < 1:
        raise ValueError("State the multiplicity family size")
    weights = n-3
    z = np.average(np.arctanh(r), weights=weights)
    se = 1/np.sqrt(weights.sum())
    p = float(2*norm.sf(abs(z/se)))
    return {"r": float(np.tanh(z)), "p_two": p, "p_bonferroni": min(1., p*family_size),
            "ci_approx": np.tanh(z + np.array([-1, 1])*norm.ppf(.975)*se).tolist(),
            "warning": "Approximate Fisher-z synthesis of Spearman coefficients; discovery selection is not corrected."}
