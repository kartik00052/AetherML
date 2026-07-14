"""Task detection — unified ML problem identification.

Detects whether a dataset is suited for supervised learning
(classification/regression), unsupervised learning (clustering),
anomaly detection, or analytics-only exploration.
"""

from phronesisml.ml.task_detection.detector import detect_task

__all__ = ["detect_task"]
