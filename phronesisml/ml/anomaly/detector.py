"""Anomaly detection — Isolation Forest, Local Outlier Factor.

Evaluates multiple anomaly detection algorithms and returns anomaly
scores and labels.  All algorithms use sklearn implementations.

Resource bounds:
- ``contamination`` (default 0.1): expected fraction of anomalies.
- Maximum dataset size for LOF: 10000 rows (uses sampling for larger).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Result of anomaly detection analysis."""

    algorithm: str
    labels: list[int]
    anomaly_scores: list[float]
    n_anomalies: int
    contamination: float
    params: dict[str, Any] = field(default_factory=dict)
    all_results: list[dict[str, Any]] = field(default_factory=list)


def detect_anomalies(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    contamination: float = 0.1,
    algorithms: list[str] | None = None,
    random_state: int = 42,
) -> AnomalyResult:
    """Detect anomalies in the given features.

    Args:
        df: DataFrame containing numeric features.
        feature_names: Columns to analyze.  If ``None``, uses all
            numeric columns.
        contamination: Expected fraction of anomalies (0.0–0.5).
        algorithms: Which algorithms to try.  Default:
            ``["isolation_forest", "lof"]``.
        random_state: Random seed for reproducibility.

    Returns:
        An ``AnomalyResult`` with labels, scores, and metadata.
    """
    if feature_names is None:
        _num_types = ("float64", "int64", "float32", "int32")
        feature_names = [c for c in df.columns if df[c].dtype in _num_types]

    if len(feature_names) < 1:
        msg = "At least 1 numeric feature required for anomaly detection."
        raise ValueError(msg)

    X = df[feature_names].values

    if algorithms is None:
        algorithms = ["isolation_forest", "lof"]

    all_results: list[dict[str, Any]] = []

    for algo in algorithms:
        try:
            if algo == "isolation_forest":
                result = _run_isolation_forest(X, contamination, random_state)
            elif algo == "lof":
                result = _run_lof(X, contamination)
            else:
                logger.warning("Unknown anomaly algorithm: %s", algo)
                continue
            all_results.append(result)
        except Exception as exc:
            logger.warning("Anomaly algorithm %s failed: %s", algo, exc)

    if not all_results:
        msg = "All anomaly detection algorithms failed."
        raise RuntimeError(msg)

    # Use Isolation Forest as primary (more robust)
    primary = None
    for r in all_results:
        if r["algorithm"] == "isolation_forest":
            primary = r
            break
    if primary is None:
        primary = all_results[0]

    logger.info(
        "Anomaly detection complete: %s, n_anomalies=%d (of %d)",
        primary["algorithm"],
        primary["n_anomalies"],
        len(X),
    )

    return AnomalyResult(
        algorithm=primary["algorithm"],
        labels=primary["labels"],
        anomaly_scores=primary["anomaly_scores"],
        n_anomalies=primary["n_anomalies"],
        contamination=contamination,
        params=primary.get("params", {}),
        all_results=all_results,
    )


def _run_isolation_forest(
    X: np.ndarray,
    contamination: float,
    random_state: int,
) -> dict[str, Any]:
    """Run Isolation Forest for anomaly detection."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    labels_raw = iso.fit_predict(X_scaled)
    scores = -iso.score_samples(X_scaled)  # higher = more anomalous

    # Convert: -1 (anomaly) → 1, 1 (normal) → 0
    labels = (labels_raw == -1).astype(int).tolist()
    anomaly_scores = scores.tolist()
    n_anomalies = int(sum(labels))

    return {
        "algorithm": "isolation_forest",
        "labels": labels,
        "anomaly_scores": anomaly_scores,
        "n_anomalies": n_anomalies,
        "params": {"contamination": contamination, "n_estimators": 100},
    }


def _run_lof(
    X: np.ndarray,
    contamination: float,
) -> dict[str, Any]:
    """Run Local Outlier Factor for anomaly detection."""
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    # Cap dataset size for LOF (O(n^2) memory)
    max_rows = 10000
    if X_scaled.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        indices = rng.choice(X_scaled.shape[0], size=max_rows, replace=False)
        X_lof = X_scaled[indices]
    else:
        X_lof = X_scaled
        indices = None

    n_neighbors = min(20, X_lof.shape[0] - 1)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    labels_raw = lof.fit_predict(X_lof)
    scores = -lof.negative_outlier_factor_

    labels_partial = (labels_raw == -1).astype(int).tolist()
    scores_partial = scores.tolist()

    # Map back to full dataset if sampled
    if indices is not None:
        labels = [0] * X_scaled.shape[0]
        anomaly_scores = [0.0] * X_scaled.shape[0]
        for i, idx in enumerate(indices):
            labels[idx] = labels_partial[i]
            anomaly_scores[idx] = scores_partial[i]
    else:
        labels = labels_partial
        anomaly_scores = scores_partial

    n_anomalies = int(sum(labels))

    return {
        "algorithm": "lof",
        "labels": labels,
        "anomaly_scores": anomaly_scores,
        "n_anomalies": n_anomalies,
        "params": {"contamination": contamination, "n_neighbors": n_neighbors},
    }
