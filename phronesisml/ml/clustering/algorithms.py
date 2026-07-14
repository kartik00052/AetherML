"""Clustering algorithms — KMeans, DBSCAN, Agglomerative Clustering.

Evaluates multiple clustering algorithms and selects the best one based
on internal metrics (silhouette score).  All algorithms use sklearn
implementations.

Resource bounds:
- ``max_k`` (default 10): maximum number of clusters to try for KMeans.
- DBSCAN and Agglomerative use automatic parameter selection based on
  dataset characteristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterResult:
    """Result of clustering analysis."""

    algorithm: str
    labels: list[int]
    n_clusters: int
    silhouette_score: float | None
    davies_bouldin_score: float | None
    calinski_harabasz_score: float | None
    params: dict[str, Any] = field(default_factory=dict)
    all_results: list[dict[str, Any]] = field(default_factory=list)


def run_clustering(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    max_k: int = 10,
    algorithms: list[str] | None = None,
    random_state: int = 42,
) -> ClusterResult:
    """Run clustering on the given features and return the best result.

    Args:
        df: DataFrame containing numeric features.
        feature_names: Columns to cluster on.  If ``None``, uses all
            numeric columns.
        max_k: Maximum clusters to try for KMeans.
        algorithms: Which algorithms to try.  Default:
            ``["kmeans", "dbscan", "agglomerative"]``.
        random_state: Random seed for reproducibility.

    Returns:
        A ``ClusterResult`` with labels, scores, and metadata.
    """
    if feature_names is None:
        _num_types = ("float64", "int64", "float32", "int32")
        feature_names = [c for c in df.columns if df[c].dtype in _num_types]

    if len(feature_names) < 2:
        msg = "At least 2 numeric features required for clustering."
        raise ValueError(msg)

    X = df[feature_names].values

    if algorithms is None:
        algorithms = ["kmeans", "dbscan", "agglomerative"]

    all_results: list[dict[str, Any]] = []

    for algo in algorithms:
        try:
            if algo == "kmeans":
                result = _run_kmeans(X, max_k, random_state)
            elif algo == "dbscan":
                result = _run_dbscan(X)
            elif algo == "agglomerative":
                result = _run_agglomerative(X, max_k, random_state)
            else:
                logger.warning("Unknown clustering algorithm: %s", algo)
                continue
            all_results.append(result)
        except Exception as exc:
            logger.warning("Clustering algorithm %s failed: %s", algo, exc)

    if not all_results:
        msg = "All clustering algorithms failed."
        raise RuntimeError(msg)

    # Select best by silhouette score
    valid = [r for r in all_results if r["silhouette_score"] is not None]
    best = max(valid, key=lambda r: r["silhouette_score"]) if valid else all_results[0]

    logger.info(
        "Best clustering: %s (silhouette=%.4f, n_clusters=%d)",
        best["algorithm"],
        best.get("silhouette_score", 0),
        best["n_clusters"],
    )

    return ClusterResult(
        algorithm=best["algorithm"],
        labels=best["labels"],
        n_clusters=best["n_clusters"],
        silhouette_score=best.get("silhouette_score"),
        davies_bouldin_score=best.get("davies_bouldin_score"),
        calinski_harabasz_score=best.get("calinski_harabasz_score"),
        params=best.get("params", {}),
        all_results=all_results,
    )


def _run_kmeans(
    X: np.ndarray,
    max_k: int,
    random_state: int,
) -> dict[str, Any]:
    """Run KMeans with automatic k selection via silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    best_score = -1.0
    best_k = 2
    best_labels = None

    max_k = min(max_k, X.shape[0] - 1, X.shape[1] + 1)
    max_k = max(max_k, 2)

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        msg = "KMeans failed for all k values."
        raise RuntimeError(msg)

    sil = float(best_score)
    db = _safe_score(davies_bouldin_score, X, best_labels)
    ch = _safe_score(calinski_harabasz_score, X, best_labels)

    return {
        "algorithm": "kmeans",
        "labels": best_labels.tolist(),
        "n_clusters": best_k,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch,
        "params": {"n_clusters": best_k, "n_init": 10},
    }


def _run_dbscan(X: np.ndarray) -> dict[str, Any]:
    """Run DBSCAN with automatic eps estimation."""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    # Estimate eps from k-distance graph (k=5)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(5, X_scaled.shape[0]))
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    # Use knee point estimation: median of sorted distances
    eps = float(np.median(k_distances))

    db = DBSCAN(eps=eps, min_samples=max(2, X_scaled.shape[1]))
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return {
            "algorithm": "dbscan",
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "params": {"eps": eps, "min_samples": max(2, X_scaled.shape[1])},
        }

    # Filter noise points for scoring
    mask = labels != -1
    if mask.sum() < 2:
        return {
            "algorithm": "dbscan",
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "params": {"eps": eps, "min_samples": max(2, X_scaled.shape[1])},
        }

    sil = _safe_score(silhouette_score, X_scaled[mask], labels[mask])
    db_score = _safe_score(davies_bouldin_score, X_scaled[mask], labels[mask])
    ch = _safe_score(calinski_harabasz_score, X_scaled[mask], labels[mask])

    return {
        "algorithm": "dbscan",
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "silhouette_score": sil,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch,
        "params": {"eps": eps, "min_samples": max(2, X_scaled.shape[1])},
    }


def _run_agglomerative(
    X: np.ndarray,
    max_k: int,
    random_state: int,
) -> dict[str, Any]:
    """Run Agglomerative Clustering with automatic k selection."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    best_score = -1.0
    best_k = 2
    best_labels = None

    max_k = min(max_k, X.shape[0] - 1, X.shape[1] + 1)
    max_k = max(max_k, 2)

    for k in range(2, max_k + 1):
        try:
            ac = AgglomerativeClustering(n_clusters=k)
            labels = ac.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        msg = "Agglomerative Clustering failed for all k values."
        raise RuntimeError(msg)

    sil = float(best_score)
    db = _safe_score(davies_bouldin_score, X, best_labels)
    ch = _safe_score(calinski_harabasz_score, X, best_labels)

    return {
        "algorithm": "agglomerative",
        "labels": best_labels.tolist(),
        "n_clusters": best_k,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch,
        "params": {"n_clusters": best_k, "linkage": "ward"},
    }


def _safe_score(func: Any, *args: Any) -> float | None:
    """Call a scoring function, returning None on failure."""
    try:
        return float(func(*args))
    except Exception:
        return None
