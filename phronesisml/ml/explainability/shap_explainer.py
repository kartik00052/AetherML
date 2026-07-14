"""SHAP-based model explainability with resource-bounded computation.

.. deprecated::
    This module is maintained for backward compatibility.  New code
    should import from ``phronesisml.ml.explainability.service`` directly.

    All logic has been moved to ``ExplainabilityService`` in
    ``service.py``.  This module delegates to the service while
    preserving the original function signatures.

Selects the appropriate SHAP explainer based on model class (tree
explainer for tree-based models, linear explainer for linear models,
PermutationExplainer for model-agnostic cases, KernelExplainer as
universal fallback) and computes global feature importance via mean
absolute SHAP values.

Resource bounds:
- ``max_samples`` (default 100): caps the number of rows used for SHAP
  value computation.  Full-dataset SHAP on non-tree explainers can be
  extremely slow — this prevents resource exhaustion if exposed via API.
- If the dataset exceeds the cap, a random sample is drawn and the
  output flags that explanations are based on a sample, not the full
  dataset.  This is visible in the result, not a silent truncation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default resource bound for SHAP computation.
DEFAULT_MAX_SAMPLES = 100

# Tree-based model class name prefixes (case-insensitive check).
_TREE_MODEL_KEYWORDS = frozenset(
    {
        "forest",
        "boosting",
        "tree",
        "xgb",
        "lgbm",
        "catboost",
        "extra tree",
    }
)

# Linear model class name prefixes (case-insensitive check).
_LINEAR_MODEL_KEYWORDS = frozenset(
    {
        "linear",
        "logistic",
        "ridge",
        "lasso",
        "elastic",
        "sgd",
    }
)


def compute_shap_explanations(
    model: Any,
    X: np.ndarray[Any, Any],
    feature_names: list[str],
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[str, Any]:
    """Compute SHAP-based feature importance explanations.

    This function delegates to ``ExplainabilityService.compute_explanations()``
    while preserving the original function signature for backward compatibility.

    Args:
        model: A trained sklearn-compatible estimator.
        X: Feature matrix (n_samples, n_features).
        feature_names: Names of the feature columns.
        max_samples: Maximum number of rows to use for SHAP computation.
            Enforced as a hard ceiling — if ``len(X) > max_samples``,
            a random sample is drawn and ``sampled`` is set to ``True``.

    Returns:
        A dict with keys: ``feature_importance`` (dict mapping feature
        name → mean absolute SHAP value), ``explainer_type`` (str),
        ``sampled`` (bool), ``n_samples_used`` (int),
        ``max_samples`` (int).

    """
    from phronesisml.ml.explainability.service import (
        ExplainConfig,
        compute_explanations,
    )

    config = ExplainConfig(max_samples=max_samples)
    return compute_explanations(model, X, feature_names, config=config)
