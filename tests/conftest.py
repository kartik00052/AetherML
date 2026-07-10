"""Shared test fixtures for AetherML tests."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherml.engines.pandas_engine import PandasEngine


@pytest.fixture
def pandas_engine() -> PandasEngine:
    """Return a PandasEngine instance for testing."""
    return PandasEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a sample DataFrame with mixed types and some nulls."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [30, 25, 35, 28, 32],
            "salary": [75000.0, 65000.0, 85000.0, 70000.0, 90000.0],
            "department": ["Engineering", "Marketing", "Engineering", "Marketing", "Engineering"],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Return an empty DataFrame (zero rows)."""
    return pd.DataFrame(columns=["name", "age", "salary"])


@pytest.fixture
def zero_columns_df() -> pd.DataFrame:
    """Return a DataFrame with zero columns but some rows."""
    return pd.DataFrame(index=range(5))


@pytest.fixture
def all_null_column_df() -> pd.DataFrame:
    """Return a DataFrame where one column is entirely null."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
            "empty_col": [None, None, None],
        }
    )


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """Return a DataFrame with a single row."""
    return pd.DataFrame(
        {
            "name": ["Alice"],
            "age": [30],
            "salary": [75000.0],
        }
    )


@pytest.fixture
def duplicate_rows_df() -> pd.DataFrame:
    """Return a DataFrame with duplicate rows."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", "Charlie"],
            "age": [30, 25, 30, 35],
        }
    )
