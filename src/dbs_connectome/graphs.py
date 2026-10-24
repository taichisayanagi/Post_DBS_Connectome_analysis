"""Deterministic fixed-cost graph definitions and anatomical-label summaries.

No tract name is inferred from two endpoint labels. Disconnected paths contribute
zero to efficiency; path length is explicitly restricted to the largest component.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata
import bct
import networkx as nx

from .gradients import validate_matrix


def fixed_cost(matrix, cost):
    m = validate_matrix(matrix)
    iu = np.triu_indices(len(m), 1)
    values = m[iu]
    positive = np.flatnonzero(values > 0)
    count = int(round(cost * len(values)))
    if not 0 < cost < 1 or count < 1 or len(positive) < count:
        raise ValueError("Insufficient positive edges for requested density")
    chosen = positive[np.lexsort((positive, -values[positive]))[:count]]
    binary = np.zeros_like(m)
    binary[iu[0][chosen], iu[1][chosen]] = 1
    binary += binary.T
    weighted = np.zeros_like(m)
    weighted[iu[0][chosen], iu[1][chosen]] = rankdata(values[chosen], method="average") / count
    weighted += weighted.T
    return binary, weighted


def largest_component(binary):
    count, membership = connected_components(csr_matrix(binary), directed=False)
    sizes = np.bincount(membership, minlength=count)
    return np.flatnonzero(membership == np.flatnonzero(sizes == sizes.max())[0])


def deterministic_louvain(matrix, seeds=range(32)):
    candidates = []
    for seed in seeds:
        labels, q = bct.community_louvain(matrix, gamma=1, seed=seed)
        mapping, canon = {}, []
        for value in labels:
            if int(value) not in mapping:mapping[int(value)] = len(mapping)+1
            canon.append(mapping[int(value)])
        candidates.append((float(q), tuple(canon), seed))
    best = max(r[0] for r in candidates)
    chosen = min((r for r in candidates if np.isclose(r[0], best, atol=1e-12, rtol=0)), key=lambda r:(r[1],r[2]))
    return chosen[0], len(set(chosen[1])), chosen[2]


def weighted_local_efficiency(weighted):
    """Frozen study definition; cube-root lengths on each neighbour subgraph."""
    result = np.zeros(len(weighted))
    for i in range(len(weighted)):
        neighbors = np.flatnonzero(weighted[i] > 0)
        degree = len(neighbors)
        if degree < 2:
            continue
        w = weighted[np.ix_(neighbors, neighbors)]
        lengths = np.zeros_like(w)
        lengths[w > 0] = np.cbrt(1/w[w > 0])
        distance = shortest_path(csr_matrix(lengths), directed=False, unweighted=False)
        inverse = np.zeros_like(distance)
        valid = np.isfinite(distance) & (distance > 0)
        inverse[valid] = 1/distance[valid]
        sym = 2*np.cbrt(weighted[i,neighbors])
        result[i] = np.sum(np.outer(sym,sym)*(inverse+inverse.T))/2/(4*degree*(degree-1))
    return result


def graph_features(matrix, costs=(.03, .05, .07), modularity=True, extended=True):
    m = validate_matrix(matrix)
    n = len(m)
    values = m[np.triu_indices(n, 1)]
    output = {"total_edge_weight": float(values.sum()), "raw_positive_edge_density": float(np.mean(values > 0))}
    across = {}
    for cost in costs:
        binary, weighted = fixed_cost(m, cost)
        lengths = np.zeros_like(m)
        lengths[weighted > 0] = 1 / weighted[weighted > 0]
        db = shortest_path(csr_matrix(binary), directed=False, unweighted=True)
        dw = shortest_path(csr_matrix(lengths), directed=False, unweighted=False)
        keep = largest_component(binary)
        off = ~np.eye(len(keep), dtype=bool)
        inverse = np.zeros_like(dw)
        valid = np.isfinite(dw) & (dw > 0)
        inverse[valid] = 1 / dw[valid]
        metrics = {
            "binary_global_efficiency": float(bct.efficiency_bin(binary)),
            "binary_mean_clustering": float(np.mean(bct.clustering_coef_bu(binary))),
            "binary_transitivity": float(bct.transitivity_bu(binary)),
            "binary_characteristic_path_lcc": float(db[np.ix_(keep,keep)][off].mean()),
            "binary_assortativity": float(bct.assortativity_bin(binary, flag=0)),
            "rank_global_efficiency": float(inverse.sum() / (n*(n-1))),
            "rank_mean_clustering": float(np.mean(bct.clustering_coef_wu(weighted))),
            "rank_transitivity": float(bct.transitivity_wu(weighted)),
            "rank_characteristic_path_lcc": float(dw[np.ix_(keep,keep)][off].mean()),
            "rank_assortativity": float(bct.assortativity_wei(weighted, flag=0)),
            "largest_component_fraction": float(len(keep)/n)}
        if modularity:
            for kind, a in (("binary",binary),("rank",weighted)):
                q, count, seed = deterministic_louvain(a)
                metrics[kind+"_louvain_modularity"] = q
                metrics[kind+"_louvain_n_communities"] = float(count)
                metrics[kind+"_louvain_selected_seed"] = float(seed)
        if extended:
            binary_bc = bct.betweenness_bin(binary)/((n-1)*(n-2))
            weighted_bc_map = nx.betweenness_centrality(nx.from_numpy_array(lengths),normalized=True,weight="weight")
            weighted_bc = np.array([weighted_bc_map[i] for i in range(n)])
            metrics.update(binary_mean_local_efficiency=float(np.mean(bct.efficiency_bin(binary,local=True))),
                           rank_mean_local_efficiency=float(np.mean(weighted_local_efficiency(weighted))),
                           binary_mean_betweenness_normalized=float(binary_bc.mean()),
                           binary_max_betweenness_normalized=float(binary_bc.max()),
                           rank_mean_betweenness_normalized=float(weighted_bc.mean()),
                           rank_max_betweenness_normalized=float(weighted_bc.max()))
        for name, value in metrics.items():
            output[f"{name}_cost{round(cost*100):02d}"] = value
            if "selected_seed" not in name:
                across.setdefault(name, []).append(value)
    output.update({name+"_cost_mean": float(np.mean(v)) for name,v in across.items()})
    return output


def node_edge_change(first, second, nodes):
    """Raw changes; undefined baseline percentages stay missing, never infinite."""
    a, b = validate_matrix(first), validate_matrix(second)
    if a.shape != b.shape or len(a) != len(nodes):
        raise ValueError("Fixed node basis is required for edge/node differences")
    node_rows=[]
    for i,r in enumerate(nodes):
        before,after=float(a[i].sum()),float(b[i].sum())
        node_rows.append({"index":i,"name":r["name"],"network":r["network"],"hemisphere":r["hemisphere"],
                          "strength_post1":before,"strength_post2":after,"strength_change":after-before,
                          "strength_percent":100*(after-before)/before if before>0 else None,
                          "degree_post1":int(np.count_nonzero(a[i])),"degree_post2":int(np.count_nonzero(b[i]))})
    edges=[]
    for i,j in zip(*np.triu_indices(len(a),1)):
        if a[i,j] == 0 and b[i,j] == 0:continue
        edges.append({"source_index":int(i),"target_index":int(j),"post1":float(a[i,j]),"post2":float(b[i,j]),
                      "delta":float(b[i,j]-a[i,j]),"percent":float(100*(b[i,j]-a[i,j])/a[i,j]) if a[i,j]>0 else None})
    return node_rows,edges


def run_graph_study(manifest, out):
    """Same fixed-basis matrices; all summaries retain explicit scope and costs."""
    from pathlib import Path
    from .study import clean, load_manifest, manifest_paths, write_csv, expected_checks
    from .study_metrics import exact_family, identification
    from .provenance import read_json, freeze_inputs, assert_inputs_unchanged, require_separate_output, write_json, record
    out=Path(out)
    paths=[manifest,*manifest_paths(read_json(manifest))]
    for path in paths:require_separate_output(out,Path(path).parent)
    frozen=freeze_inputs(paths)
    spec,nodes,matrices=load_manifest(manifest)
    rows=[]
    graph=np.empty(matrices.shape[:2],dtype=object)
    # Full frozen basis is deliberate: this is the paper's identification comparator,
    # not its different unmasked preoperative cortical predictor.
    for i,subject in enumerate(spec['subjects']):
        for j,visit in enumerate(('pre','post1','post2')):
            graph[i,j]=graph_features(matrices[i,j],modularity=spec.get('graph_modularity',True),extended=spec.get('graph_extended',True))
            rows.append({'subject':subject['id'],'timepoint':visit,'basis':'full_frozen_matrix',**graph[i,j]})
        nn,ee=node_edge_change(matrices[i,1],matrices[i,2],nodes)
        write_csv(out/(subject['id']+'_node_change.csv'),nn)
        write_csv(out/(subject['id']+'_edge_change.csv'),ee)
        print(f"[step {i+1}/{len(spec['subjects'])}] graph / node / edge summaries",flush=True)
    write_csv(out/'graph_metrics.csv',rows)
    ids=[s['id'] for s in spec['subjects']]
    fields=[k for k in graph[0,0] if 'selected_seed' not in k]
    changes=np.array([[graph[i,2][k]-graph[i,1][k] for k in fields] for i in range(len(ids))])
    clinical=[];excluded=[]
    for name,endpoint in spec['clinical'].items():
        valid=np.array([endpoint['values'][s] is not None for s in ids])
        y=np.array([endpoint['values'][s] for s in ids if endpoint['values'][s] is not None],float)
        if len(y)<3 or np.ptp(y)==0:
            excluded.append({'outcome':name,'reason':'insufficient or constant outcome'});continue
        finite=np.isfinite(changes[valid]).all(0)
        usable=finite & np.array([np.ptp(v) > 0 if np.isfinite(v).all() else False for v in changes[valid].T])
        for k,ok in zip(fields,usable):
            if not ok:excluded.append({'outcome':name,'metric':k,'reason':'undefined or constant graph change'})
        if not usable.any():continue
        result=exact_family(changes[valid][:,usable],y)
        for j,k in enumerate(np.array(fields)[usable]):
            clinical.append({'outcome':name,'metric':k,'n':len(y),'family_size':int(usable.sum()),
                             **{key:float(result[key][j]) for key in ('rho','p_exact','q_bh','p_maxT')}})
    if clinical:write_csv(out/'graph_clinical_correlations.csv',clinical)
    selected=['binary_global_efficiency_cost_mean','binary_characteristic_path_lcc_cost_mean','binary_transitivity_cost_mean','binary_assortativity_cost_mean']
    features=np.array([[g[k] for k in selected] for g in graph.ravel()])
    graph_ident,_=identification(features,len(ids),spec.get('identification_samples',100000))
    upper=np.triu_indices(len(nodes),1)
    edges=np.array([m[upper] for m in matrices.reshape(-1,len(nodes),len(nodes))])
    edge_ident,_=identification(edges,len(ids),spec.get('identification_samples',100000))
    result={'scope':'full_frozen_basis_graphs_and_raw_node_edge_changes','graph_identification':graph_ident,
            'edge_identification':edge_ident,'clinical_exclusions':excluded,
            'warning':'Different from the unmasked preoperative cortical graph predictor. Endpoint edges are not anatomically segmented white-matter bundles.'}
    if spec.get('preoperative_graph'):
        result['preoperative_graph'] = preoperative_graphs(spec, out)
    checks=expected_checks({'expected':spec.get('graph_expected',[])},{'graph_result':result,'graph_metrics':rows,'graph_clinical_correlations':clinical},[])
    result['checks']=checks
    passed=all(c['passed'] for c in checks)
    assert_inputs_unchanged(frozen)
    write_json(out/'graph_result.json',clean(result))
    record(out,'graph-study',paths,{'costs':[.03,.05,.07],'louvain_seeds':32,'basis':'full_frozen_matrix'},sorted(out.glob('*')),'completed' if passed else 'failed_replay')
    if not passed:raise ValueError('Graph replay differs from frozen expected outputs')


def preoperative_graphs(spec, out):
    """Separate hash-bound cortical unmasked basis; never reuse masked baseline."""
    from pathlib import Path
    from .gradients import load_nodes
    from .provenance import digest
    from .study import write_csv
    from .study_metrics import exact_family
    pre=spec['preoperative_graph']
    ids=[s['id'] for s in spec['subjects']]
    if pre.get('basis') != 'unmasked_preoperative_cortex' or digest(pre['nodes']) != pre.get('nodes_sha256'):
        raise ValueError('Explicit unchanged unmasked preoperative cortical basis required')
    nodes=load_nodes(pre['nodes'])
    if any(n['tissue'] != 'cortex' for n in nodes) or [s['id'] for s in pre['subjects']] != ids:
        raise ValueError('Preoperative cortical nodes / subject pairing mismatch')
    rows=[]
    for subject in pre['subjects']:
        if subject.get('mask_scheme') != 'none' or digest(subject['matrix']) != subject.get('sha256'):
            raise ValueError('Preoperative matrix changed or is masked')
        p=Path(subject['matrix'])
        m=validate_matrix(np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.loadtxt(p,delimiter=','))
        ix=np.asarray(subject.get('matrix_indices',list(range(len(m)))))
        if ix.ndim!=1 or ix.dtype.kind not in 'iu' or len(ix)!=len(nodes) or len(set(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(m)):
            raise ValueError('Invalid preoperative cortical matrix mapping')
        row={'subject':subject['id'],**graph_features(m[np.ix_(ix,ix)])}
        rows.append(row)
        print(f"[step preoperative] {len(rows)}/{len(ids)} unmasked cortical graph",flush=True)
    write_csv(Path(out)/'preoperative_cortical_graphs.csv',rows)
    fields=[k for k in rows[0] if k.endswith('_cost_mean')]
    x=np.array([[r[k] for k in fields] for r in rows])
    correlations=[];exclusions=[]
    for name,endpoint in spec['clinical'].items():
        valid=np.array([endpoint['values'][s] is not None for s in ids])
        y=np.array([endpoint['values'][s] for s in ids if endpoint['values'][s] is not None])
        if len(y)<3 or np.ptp(y)==0:
            exclusions.append({'outcome':name,'reason':'insufficient or constant outcome'});continue
        usable=np.array([np.isfinite(v).all() and np.ptp(v)>0 for v in x[valid].T])
        if not usable.any():continue
        result=exact_family(x[valid][:,usable],y)
        for j,k in enumerate(np.asarray(fields)[usable]):
            correlations.append({'outcome':name,'metric':k,'n':len(y),'family_size':int(usable.sum()),
                                 **{key:float(result[key][j]) for key in ('rho','p_exact','q_bh','p_maxT')}})
    if correlations:write_csv(Path(out)/'preoperative_cortical_correlations.csv',correlations)
    return {'basis':pre['basis'],'n_nodes':len(nodes),'n_subjects':len(ids),'clinical_exclusions':exclusions,
            'warning':'Exploratory postoperative outcomes associated with baseline structure; not a prospective prediction model.'}
