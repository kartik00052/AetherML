"""Data profiling — descriptive statistics and distribution analysis.

``data.profilers.stats`` contains the real profiling logic,
called by the EDA agent.
"""

from aetherml.data.profilers.stats import profile_dataset

__all__ = ["profile_dataset"]
