"""One explicit study-compatible embedding primitive; no clinical inference."""

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np
from brainspace.gradient import GradientMaps


@dataclass(frozen=True)
class Settings:
    log_transform: bool = True
    sparsity: float = 0.75
    alpha: float = 0.05
    diffusion_time: int = 0
    n_components: int = 10
    n_retained: int = 4
    random_state: int = 42


def validate_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Connectome must be square")
    if not np.isfinite(m).all() or np.any(m < 0):
        raise ValueError("Weights must be finite and nonnegative")
    if not np.allclose(m, m.T, atol=1e-10, rtol=1e-7):
        raise ValueError("Asymmetric connectome: do not silently symmetrize")
    if np.any(np.diag(m) != 0):
        raise ValueError("Self-connections must already be excluded")
    return m


def load_nodes(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        nodes = list(csv.DictReader(stream, delimiter="\t"))
    required = {"index", "label", "name", "hemisphere", "tissue", "network"}
    if not nodes or not required.issubset(nodes[0]):
        raise ValueError("Node TSV must define index,label,name,hemisphere,tissue,network")
    if [int(row["index"]) for row in nodes] != list(range(len(nodes))):
        raise ValueError("Node indices must be zero-based, consecutive and in matrix order")
    labels = [int(row["label"]) for row in nodes]
    if min(labels) < 1 or len(labels) != len(set(labels)):
        raise ValueError("Atlas labels must be unique positive integers")
    if any(row["hemisphere"] not in ("L", "R", "M") or row["tissue"] not in ("cortex", "subcortex") for row in nodes):
        raise ValueError("Unknown hemisphere or tissue code")
    # Derive/check Schaefer network identity from its actual name, not a second stale table.
    for row in nodes:
        if row["name"].startswith("7Networks_"):
            parts = row["name"].split("_")
            if parts[1] != row["hemisphere"] + "H" or parts[2] != row["network"]:
                raise ValueError("Schaefer name disagrees with hemisphere/network metadata")
    return nodes


def compact_raw(raw, labels):
    """MRtrix CSV without -keep_unassigned: row 0 is atlas label 1."""
    raw = validate_matrix(raw)
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer) or labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional integer sequence")
    if len(labels) == 0 or len(np.unique(labels)) != len(labels) or np.any(labels < 1) or np.any(labels > len(raw)):
        raise ValueError("Labels missing, duplicated or out of bounds; never pad silently")
    return raw[np.ix_(labels - 1, labels - 1)]


def embed(matrix, indices, settings=Settings(), reference=None):
    matrix = validate_matrix(matrix)
    indices = np.asarray(indices, dtype=int)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(matrix)):
        raise ValueError("Invalid or duplicated embedding indices")
    if len(indices) <= settings.n_components + 1:
        raise ValueError("Too few nodes for the requested eigensystem")
    if not 1 <= settings.n_retained <= settings.n_components:
        raise ValueError("Invalid retained component count")
    x = matrix[np.ix_(indices, indices)]
    if np.any(x.sum(1) == 0):
        raise ValueError("Isolated cortical node: freeze and document a common valid basis first")
    if settings.log_transform:
        x = np.log1p(x)
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (len(indices), settings.n_components) or not np.isfinite(reference).all():
            raise ValueError("Reference shape/content mismatch; align all computed components before retention")
    model = GradientMaps(n_components=settings.n_components, approach="dm", kernel="normalized_angle",
                         alignment="procrustes" if reference is not None else None,
                         random_state=settings.random_state)
    model.fit(x, reference=reference, sparsity=settings.sparsity,
              alpha=settings.alpha, diffusion_time=settings.diffusion_time)
    result = model.gradients_ if reference is None else model.aligned_
    if isinstance(result, list):
        result = result[0]
    if not np.isfinite(result).all():
        raise ValueError("Embedding returned non-finite coordinates")
    return np.asarray(result), np.asarray(model.lambdas_)


def hemisphere_embeddings(matrix, nodes, references=None, settings=Settings()):
    if len(nodes) != len(matrix):
        raise ValueError("Matrix and node table sizes differ")
    result = {}
    for hemi in ("L", "R"):
        indices = [i for i, n in enumerate(nodes) if n["hemisphere"] == hemi and n["tissue"] == "cortex"]
        ref = None if references is None else references[f"{hemi}_coordinates"]
        g, eigenvalues = embed(matrix, indices, settings, ref)
        result[f"{hemi}_coordinates"] = g
        result[f"{hemi}_indices"] = np.asarray(indices)
        result[f"{hemi}_eigenvalues"] = eigenvalues
    return result


def geometry(coordinates, reference_centroid, k=4):
    """Centroid is explicit: cohort-reference and session-centred metrics differ."""
    g = np.asarray(coordinates)[:, :k]
    center = np.asarray(reference_centroid)[:k]
    if center.shape != (k,) or g.shape[1] != k or not np.isfinite(center).all():
        raise ValueError("Invalid centroid or retained dimensionality")
    return {"eccentricity": np.linalg.norm(g - center, axis=1), "range": np.ptp(g, axis=0)}


def displacement(first, second, k=4):
    """Only call for the same nodes aligned to the SAME reference/settings."""
    if first.shape != second.shape or first.shape[1] < k:
        raise ValueError("Gradient shapes differ")
    return np.linalg.norm(second[:, :k] - first[:, :k], axis=1)


def settings_dict():
    return asdict(Settings())
