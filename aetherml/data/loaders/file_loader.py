"""Data loaders — read data from various sources via the active engine.

Loaders are thin wrappers around ``BaseEngine.read()`` that add:
- Format detection from file extensions
- Uniform return type (Pandas DataFrame via ``engine.collect()``)
- Error wrapping with ``DataLoadError``

Loaders do NOT own data — they are pure functions that return DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aetherml.engines.base_engine import BaseEngine
from aetherml.exceptions import DataLoadError

logger = logging.getLogger(__name__)

# Supported formats and their file extensions
_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "csv": [".csv", ".tsv"],
    "parquet": [".parquet", ".pq"],
    "json": [".json", ".jsonl", ".ndjson"],
    "feather": [".feather", ".arrow"],
    "excel": [".xlsx", ".xls"],
}


def detect_format(path: str | Path) -> str:
    """Detect file format from the file extension.

    Returns:
        Format name string (e.g. ``"csv"``, ``"parquet"``).

    Raises:
        DataLoadError: If the format cannot be determined.

    """
    suffix = Path(path).suffix.lower()
    for fmt, extensions in _FORMAT_EXTENSIONS.items():
        if suffix in extensions:
            return fmt
    msg = f"Cannot detect format from extension '{suffix}' for path: {path}"
    raise DataLoadError(msg)


def load_file(
    path: str | Path,
    engine: BaseEngine,
    format: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a file into a Pandas DataFrame via the given engine.

    Args:
        path: Path to the data file.
        engine: The active computation engine.
        format: Explicit format override (``"csv"``, ``"parquet"``, etc.).
            If ``None``, the format is auto-detected from the file extension.
        **kwargs: Additional arguments forwarded to the engine's reader
            (e.g. ``sep=";"``, ``header=False``).

    Returns:
        A Pandas DataFrame with the loaded data.

    Raises:
        DataLoadError: If loading fails for any reason.

    """
    path = Path(path)
    if not path.exists():
        msg = f"Data file does not exist: {path}"
        raise DataLoadError(msg)

    if format is None:
        format = detect_format(path)

    logger.info("Loading %s file from %s", format, path)

    try:
        df = engine.read(path, **kwargs)
        # Normalise to Pandas via collect()
        result = engine.collect(df)
        logger.info("Loaded %d rows, %d columns", result.shape[0], result.shape[1])
        return result
    except DataLoadError:
        raise
    except Exception as exc:
        msg = f"Failed to load data from {path}: {exc}"
        raise DataLoadError(msg) from exc


def infer_format(path: str | Path) -> str:
    """Public alias for ``detect_format``."""
    return detect_format(path)
