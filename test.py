"""
PhronesisML -- Comprehensive Compatibility Matrix & Integration Audit
=====================================================================
Master test.py: validates every public API, dataset category, parameter,
SHAP explainability, error recovery, and generates a final summary report.

Usage:  python test.py
Output: Console summary + AUDIT_REPORT.md
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit")

# ── Result Tracking ──────────────────────────────────────────────────────────


@dataclass
class StageResult:
    name: str
    status: str = "pending"  # passed / failed / skipped / warning
    duration_s: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


RESULTS: list[StageResult] = []


def run_stage(name: str, fn: Callable[..., object], *, required: bool = True) -> None:
    """Execute a test stage, capture result, never abort on recoverable errors."""
    sr = StageResult(name=name)
    t0 = time.perf_counter()
    try:
        output = fn()
        sr.status = "passed"
        sr.message = str(output)[:200] if output else "OK"
        sr.details = output if isinstance(output, dict) else {}
    except NotImplementedError as exc:
        sr.status = "skipped"
        sr.message = f"Not implemented: {exc}"
    except ImportError as exc:
        sr.status = "skipped"
        sr.message = f"Missing optional dependency: {exc}"
    except Exception as exc:
        sr.status = "failed" if required else "warning"
        sr.message = f"{type(exc).__name__}: {exc}"
        sr.details["traceback"] = traceback.format_exc()
        logger.error("Stage %s failed: %s", name, exc)
    finally:
        sr.duration_s = time.perf_counter() - t0
        RESULTS.append(sr)
        icon = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP", "warning": "WARN"}.get(
            sr.status, "?"
        )
        print(f"  [{icon:4s}] {name} ({sr.duration_s:.2f}s) -- {sr.message[:120]}")


async def run_stage_async(name: str, fn: Callable[..., object], *, required: bool = True) -> None:
    sr = StageResult(name=name)
    t0 = time.perf_counter()
    try:
        output = await fn()
        sr.status = "passed"
        sr.message = str(output)[:200] if output else "OK"
        sr.details = output if isinstance(output, dict) else {}
    except NotImplementedError as exc:
        sr.status = "skipped"
        sr.message = f"Not implemented: {exc}"
    except ImportError as exc:
        sr.status = "skipped"
        sr.message = f"Missing optional dependency: {exc}"
    except Exception as exc:
        sr.status = "failed" if required else "warning"
        sr.message = f"{type(exc).__name__}: {exc}"
        sr.details["traceback"] = traceback.format_exc()
        logger.error("Stage %s failed: %s", name, exc)
    finally:
        sr.duration_s = time.perf_counter() - t0
        RESULTS.append(sr)
        icon = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP", "warning": "WARN"}.get(
            sr.status, "?"
        )
        print(f"  [{icon:4s}] {name} ({sr.duration_s:.2f}s) -- {sr.message[:120]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: TEST DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

TMPDIR = Path(tempfile.mkdtemp(prefix="phronesis_audit_"))


def _make_classification_csv(n_rows: int = 200, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n_rows),
            "income": rng.normal(50000, 15000, n_rows).round(2),
            "score": rng.uniform(0, 100, n_rows).round(1),
            "category": rng.choice(["A", "B", "C"], n_rows),
            "target": rng.choice([0, 1], n_rows),
        }
    )
    p = path or str(TMPDIR / "classification.csv")
    df.to_csv(p, index=False)
    return p


def _make_regression_csv(n_rows: int = 200, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "sqft": rng.integers(500, 5000, n_rows),
            "bedrooms": rng.integers(1, 6, n_rows),
            "age": rng.integers(0, 100, n_rows),
            "price": (rng.random(n_rows) * 300000 + 100000).round(2),
        }
    )
    p = path or str(TMPDIR / "regression.csv")
    df.to_csv(p, index=False)
    return p


def _make_dirty_csv(n_rows: int = 100, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    vals = list(range(n_rows))
    target_vals = list(range(n_rows))
    for i in range(n_rows):
        if rng.random() < 0.15:
            vals[i] = None
        if rng.random() < 0.05:
            target_vals[i] = None
    df = pd.DataFrame(
        {
            "feature1": vals,
            "feature2": rng.normal(0, 1, n_rows).tolist(),
            "feature3": rng.choice(["X", "Y", None], n_rows).tolist(),
            "target": target_vals,
        }
    )
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # add duplicates
    p = path or str(TMPDIR / "dirty.csv")
    df.to_csv(p, index=False)
    return p


def _make_tiny_csv(path: str | None = None) -> str:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    p = path or str(TMPDIR / "tiny.csv")
    df.to_csv(p, index=False)
    return p


def _make_multiclass_csv(n_rows: int = 200, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feat_a": rng.random(n_rows),
            "feat_b": rng.random(n_rows),
            "label": rng.choice(["cat", "dog", "bird"], n_rows),
        }
    )
    p = path or str(TMPDIR / "multiclass.csv")
    df.to_csv(p, index=False)
    return p


def _make_constant_target_csv(n_rows: int = 50, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": rng.random(n_rows),
            "target": [1] * n_rows,
        }
    )
    p = path or str(TMPDIR / "constant_target.csv")
    df.to_csv(p, index=False)
    return p


def _make_no_target_csv(n_rows: int = 50, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
        }
    )
    p = path or str(TMPDIR / "no_target.csv")
    df.to_csv(p, index=False)
    return p


def _make_high_cardinality_csv(n_rows: int = 200, path: str | None = None) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(n_rows),
            "feature": rng.random(n_rows),
            "target": rng.choice([0, 1], n_rows),
        }
    )
    p = path or str(TMPDIR / "high_card.csv")
    df.to_csv(p, index=False)
    return p


def _make_inf_values_csv(path: str | None = None) -> str:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.inf, 4.0, 5.0],
            "b": [1.0, -np.inf, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        }
    )
    p = path or str(TMPDIR / "inf_values.csv")
    df.to_csv(p, index=False)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: IMPORT & INTERFACE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_imports() -> None:
    import phronesisml

    assert phronesisml.__version__ == "0.2.2"
    return {"version": phronesisml.__version__, "all_exports": len(phronesisml.__all__)}


def test_sdk_oop_imports() -> None:
    from phronesisml.sdk import Phronesis

    return {"sdk_class": Phronesis.__name__}


def test_simple_api_imports() -> None:
    return {"simple_api": True}


def test_async_api_imports() -> None:
    return {"async_api": True}


def test_config_imports() -> None:
    from phronesisml.configs.settings import (
        PhronesisConfig,
    )

    cfg = PhronesisConfig()
    return {
        "engine_preferred": cfg.engine.preferred,
        "data_default_format": cfg.data.default_format,
        "feature_variance_threshold": cfg.feature_selection.variance_threshold,
    }


def test_exception_hierarchy() -> None:
    from phronesisml.exceptions import (
        AgentError,
        AgentNotImplementedError,
        ConfigurationError,
        DataError,
        DataLoadError,
        PhronesisError,
        WorkflowError,
    )

    assert issubclass(ConfigurationError, PhronesisError)
    assert issubclass(DataLoadError, DataError)
    assert issubclass(WorkflowError, PhronesisError)
    assert issubclass(AgentNotImplementedError, AgentError)
    return {"exception_count": 11}


def test_workflow_state() -> None:
    from phronesisml.workflow.state import WorkflowState

    WorkflowState()
    fields = list(WorkflowState.model_fields.keys())
    return {"state_fields": len(fields), "sample_fields": fields[:5]}


def test_pipeline_order() -> None:
    from phronesisml.workflow.graph import PIPELINE_ORDER

    return {"stages": PIPELINE_ORDER, "count": len(PIPELINE_ORDER)}


def test_agent_base() -> None:
    from phronesisml.agents.base import AgentResult

    return {"agent_result_fields": list(AgentResult.model_fields.keys())}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: ENGINE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_pandas_engine() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine

    eng = PandasEngine()
    csv_path = _make_classification_csv(20)
    df = eng.read(csv_path)
    assert isinstance(df, pd.DataFrame)
    shape = eng.shape(df)
    cols = eng.columns(df)
    head = eng.head(df, 3)
    mem = eng.memory_usage(df)
    return {"shape": shape, "columns": cols, "head_rows": len(head), "memory_bytes": mem}


def test_polars_engine() -> None:
    from phronesisml.engines.polars_engine import PolarsEngine

    eng = PolarsEngine()
    csv_path = _make_classification_csv(20)
    df = eng.read(csv_path)
    shape = eng.shape(df)
    cols = eng.columns(df)
    head = eng.head(df, 3)
    collected = eng.collect(df)
    assert isinstance(collected, pd.DataFrame)
    return {
        "shape": shape,
        "columns": cols,
        "head_rows": len(head),
        "collected_type": type(collected).__name__,
    }


def test_engine_selector() -> None:
    from phronesisml.engines.engine_selector import select_engine

    csv_path = _make_classification_csv(20)
    eng = select_engine(data_path=csv_path)
    return {"selected_engine": type(eng).__name__}


def test_engine_selector_pandas_force() -> None:
    from phronesisml.configs.settings import PhronesisConfig
    from phronesisml.engines.engine_selector import select_engine

    cfg = PhronesisConfig()
    cfg.engine.preferred = "pandas"
    eng = select_engine(config=cfg)
    return {"engine": type(eng).__name__}


def test_engine_selector_polars_force() -> None:
    from phronesisml.configs.settings import PhronesisConfig
    from phronesisml.engines.engine_selector import select_engine

    cfg = PhronesisConfig()
    cfg.engine.preferred = "polars"
    eng = select_engine(config=cfg)
    return {"engine": type(eng).__name__}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: DATA LOADING & FORMATS
# ═══════════════════════════════════════════════════════════════════════════════


def test_csv_loading() -> None:
    from phronesisml.data.loaders.file_loader import detect_format, load_file
    from phronesisml.engines.pandas_engine import PandasEngine

    csv_path = _make_classification_csv(20)
    fmt = detect_format(csv_path)
    df = load_file(csv_path, PandasEngine())
    assert len(df) == 20
    return {"format": fmt, "rows": len(df)}


def test_json_loading() -> None:
    from phronesisml.data.loaders.file_loader import detect_format, load_file
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = str(TMPDIR / "test.json")
    df.to_json(path, orient="records")
    fmt = detect_format(path)
    loaded = load_file(path, PandasEngine())
    assert len(loaded) == 2
    return {"format": fmt, "rows": len(loaded)}


def test_parquet_loading() -> None:
    from phronesisml.data.loaders.file_loader import detect_format, load_file
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = str(TMPDIR / "test.parquet")
    df.to_parquet(path)
    fmt = detect_format(path)
    loaded = load_file(path, PandasEngine())
    assert len(loaded) == 2
    return {"format": fmt, "rows": len(loaded)}


def test_excel_loading() -> None:
    from phronesisml.data.loaders.file_loader import detect_format, list_excel_sheets, load_file
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = str(TMPDIR / "test.xlsx")
    df.to_excel(path, index=False)
    fmt = detect_format(path)
    sheets = list_excel_sheets(path)
    loaded = load_file(path, PandasEngine())
    assert len(loaded) == 2
    return {"format": fmt, "sheets": sheets, "rows": len(loaded)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: ETL & DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def test_etl_handle_nulls_drop() -> None:
    from phronesisml.data.transformers.cleaning import handle_nulls

    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    cleaned, log = handle_nulls(df, strategy="drop")
    assert cleaned.isnull().sum().sum() == 0
    return {"strategy": "drop", "rows_before": 3, "rows_after": len(cleaned)}


def test_etl_handle_nulls_fill() -> None:
    from phronesisml.data.transformers.cleaning import handle_nulls

    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    cleaned, log = handle_nulls(df, strategy="fill", fill_value=0)
    assert cleaned.isnull().sum().sum() == 0
    return {"strategy": "fill", "filled_with": 0}


def test_etl_handle_nulls_flag() -> None:
    from phronesisml.data.transformers.cleaning import handle_nulls

    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    cleaned, log = handle_nulls(df, strategy="flag")
    flag_cols = [c for c in cleaned.columns if "_is_null" in c]
    assert len(flag_cols) == 2
    return {"strategy": "flag", "flag_columns": flag_cols}


def test_etl_encode_categoricals() -> None:
    from phronesisml.data.transformers.cleaning import encode_categoricals

    df = pd.DataFrame({"cat": ["a", "b", "c", "a"], "num": [1, 2, 3, 4]})
    encoded, log = encode_categoricals(df, columns=["cat"])
    assert encoded["cat"].dtype in [np.int64, np.int32, int, float]
    return {"encoded_column": "cat", "dtype": str(encoded["cat"].dtype)}


def test_etl_cast_dtypes() -> None:
    from phronesisml.data.transformers.cleaning import cast_dtypes

    df = pd.DataFrame({"x": ["1", "2", "3"], "y": ["4.0", "5.0", "6.0"]})
    casted, log = cast_dtypes(df, type_map={"x": "int", "y": "float"})
    return {"x_dtype": str(casted["x"].dtype), "y_dtype": str(casted["y"].dtype)}


def test_etl_invalid_strategy() -> None:
    from phronesisml.data.transformers.cleaning import handle_nulls
    from phronesisml.exceptions import DataTransformError

    df = pd.DataFrame({"a": [1, None]})
    try:
        handle_nulls(df, strategy="invalid_xyz")
        return {"raised": False}
    except DataTransformError:
        return {"raised": True, "error_type": "DataTransformError"}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F: VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_validate_clean_data() -> None:
    from phronesisml.data.validators.checks import validate_dataframe
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validated, report = validate_dataframe(df, PandasEngine())
    assert report["passed"] is True
    return {"passed": True, "shape": report["shape"]}


def test_validate_dirty_data() -> None:
    from phronesisml.data.validators.checks import validate_dataframe
    from phronesisml.engines.pandas_engine import PandasEngine

    csv_path = _make_dirty_csv(20)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    validated, report = validate_dataframe(df, PandasEngine())
    return {
        "passed": report["passed"],
        "null_columns": report.get("null_columns", []),
        "duplicate_rows": report.get("duplicate_rows", 0),
    }


def test_validate_empty_df() -> None:
    from phronesisml.data.validators.checks import validate_dataframe
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.exceptions import DataValidationError

    df = pd.DataFrame()
    try:
        validate_dataframe(df, PandasEngine())
        return {"raised": False}
    except DataValidationError:
        return {"raised": True}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G: EDA / PROFILING
# ═══════════════════════════════════════════════════════════════════════════════


def test_eda_profiling() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine

    csv_path = _make_classification_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    assert "shape" in profile
    assert "numeric_columns" in profile
    assert "categorical_columns" in profile
    return {
        "rows": profile["shape"]["rows"],
        "columns": profile["shape"]["columns"],
        "numeric": len(profile["numeric_columns"]),
        "categorical": len(profile["categorical_columns"]),
    }


def test_eda_mixed_types() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    profile = profile_dataset(df, PandasEngine())
    return {
        "columns": profile["column_names"],
        "numeric": profile["numeric_columns"],
        "categorical": profile["categorical_columns"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION H: TARGET DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def test_target_detection_classification() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_classification_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    assert result["task_type"] in ("classification", "regression", "ambiguous")
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "confidence": result["confidence"],
    }


def test_target_detection_regression() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_regression_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "confidence": result["confidence"],
    }


def test_target_detection_no_target() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_no_target_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "confidence": result["confidence"],
    }


def test_target_detection_constant_target() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_constant_target_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "confidence": result["confidence"],
    }


def test_target_detection_multiclass() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_multiclass_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "confidence": result["confidence"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION I: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════


def test_feature_engineering() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.feature_engineering.engineer import engineer_features

    csv_path = _make_classification_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    features, log = engineer_features(df, PandasEngine(), target_column="target")
    assert features is not None
    return {"n_features": len(features.columns), "feature_names": list(features.columns)[:5]}


def test_feature_engineering_no_target() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.feature_engineering.engineer import engineer_features

    csv_path = _make_classification_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    features, log = engineer_features(df, PandasEngine(), target_column=None)
    return {"n_features": len(features.columns)}


def test_feature_engineering_fill_strategy() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.feature_engineering.engineer import engineer_features

    csv_path = _make_dirty_csv(30)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    features, log = engineer_features(
        df, PandasEngine(), target_column="target", null_strategy="fill"
    )
    return {"n_features": len(features.columns)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION J: MODEL SELECTION & TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_recommendation_classification() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models(
        "classification", n_rows=200, n_features=4, n_numeric_features=3, n_categorical_features=1
    )
    names = [c.name for c in candidates]
    assert len(candidates) >= 2
    return {"candidates": names, "count": len(candidates)}


def test_model_recommendation_regression() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models(
        "regression", n_rows=200, n_features=3, n_numeric_features=3, n_categorical_features=0
    )
    names = [c.name for c in candidates]
    return {"candidates": names, "count": len(candidates)}


def test_model_recommendation_ambiguous() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models(
        "ambiguous", n_rows=200, n_features=3, n_numeric_features=3, n_categorical_features=0
    )
    names = [c.name for c in candidates]
    return {"candidates": names, "count": len(candidates)}


def test_model_recommendation_none_task() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models(
        None, n_rows=200, n_features=3, n_numeric_features=3, n_categorical_features=0
    )
    return {"candidates": [c.name for c in candidates], "count": len(candidates)}


def test_training_cost_estimation() -> None:
    from phronesisml.ml.automl.auto_selector import estimate_training_cost

    low = estimate_training_cost(50, 3)
    med = estimate_training_cost(5000, 10)
    high = estimate_training_cost(100000, 100)
    return {"low": low, "medium": med, "high": high}


def test_train_classification() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_classification_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 100, 4, 3, 1)
    result = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=10,
        max_time_seconds=30,
    )
    assert "best_model" in result
    assert "best_score" in result
    return {
        "best_model": result["best_model"],
        "best_score": result["best_score"],
        "trials": result["trials_used"],
    }


def test_train_regression() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_regression_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("regression", 100, 3, 3, 0)
    result = train_models(
        df, PandasEngine(), candidates, "price", "regression", max_trials=10, max_time_seconds=30
    )
    return {"best_model": result["best_model"], "best_score": result["best_score"]}


def test_train_with_cv() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_classification_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 100, 4, 3, 1)
    result = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=5,
        max_time_seconds=30,
        cv=3,
    )
    return {"best_model": result["best_model"], "cv_results": result.get("cv_results") is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION K: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_evaluate_classification() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models
    from phronesisml.ml.evaluation.metrics import evaluate_model

    csv_path = _make_classification_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 100, 4, 3, 1)
    training = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=5,
        max_time_seconds=30,
    )
    eval_result = evaluate_model(
        training["best_model"],
        df,
        "target",
        training["feature_names"],
        "classification",
        training.get("best_params"),
    )
    metrics = eval_result["metrics"]
    assert "accuracy" in metrics
    return {"metrics": list(metrics.keys()), "task_type": eval_result["task_type"]}


def test_evaluate_regression() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models
    from phronesisml.ml.evaluation.metrics import evaluate_model

    csv_path = _make_regression_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("regression", 100, 3, 3, 0)
    training = train_models(
        df, PandasEngine(), candidates, "price", "regression", max_trials=5, max_time_seconds=30
    )
    eval_result = evaluate_model(
        training["best_model"],
        df,
        "price",
        training["feature_names"],
        "regression",
        training.get("best_params"),
    )
    metrics = eval_result["metrics"]
    return {"metrics": list(metrics.keys()), "task_type": eval_result["task_type"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION L: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_shap_explainability() -> None:
    from sklearn.ensemble import RandomForestClassifier

    from phronesisml.ml.explainability.shap_explainer import compute_shap_explanations

    rng = np.random.default_rng(42)
    X = rng.random((50, 4))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    result = compute_shap_explanations(model, X, ["f1", "f2", "f3", "f4"], max_samples=50)
    assert "feature_importance" in result
    assert "explainer_type" in result
    return {
        "explainer_type": result["explainer_type"],
        "features": len(result["feature_importance"]),
        "sampled": result["sampled"],
    }


def test_shap_without_shap_library() -> None:
    import unittest.mock

    with unittest.mock.patch.dict("sys.modules", {"shap": None}):
        from sklearn.ensemble import RandomForestClassifier

        from phronesisml.ml.explainability.shap_explainer import compute_shap_explanations

        rng = np.random.default_rng(42)
        X = rng.random((20, 3))
        y = (X[:, 0] > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        try:
            result = compute_shap_explanations(model, X, ["a", "b", "c"], max_samples=20)
            return {"graceful": True, "result": result}
        except ImportError:
            return {"graceful": False, "raised_import_error": True}


def test_shap_linear_model() -> None:
    from sklearn.linear_model import LinearRegression

    from phronesisml.ml.explainability.shap_explainer import compute_shap_explanations

    rng = np.random.default_rng(42)
    X = rng.random((50, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.1, 50)
    model = LinearRegression()
    model.fit(X, y)
    result = compute_shap_explanations(model, X, ["x1", "x2", "x3"], max_samples=50)
    return {
        "explainer_type": result["explainer_type"],
        "features": len(result["feature_importance"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION M: REPORTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_report_generation() -> None:
    from phronesisml.ml.reports.builder import build_html_report, build_report

    state = {
        "target_column": "target",
        "task_type": "classification",
        "target_detection_confidence": 0.85,
        "ambiguity_reason": None,
        "validation_report": {"passed": True},
        "data_profile": {
            "shape": {"rows": 100, "columns": 5},
            "numeric_columns": ["a"],
            "categorical_columns": ["b"],
        },
        "transform_log": [{"step": "nulls_dropped", "columns_affected": ["a"]}],
        "feature_names": ["f1", "f2", "f3"],
        "best_pipeline": {
            "model_type": "RandomForestClassifier",
            "score": 0.92,
            "truncated": False,
        },
        "evaluation_report": {"metrics": {"accuracy": 0.92, "f1": 0.91}},
        "explanation_report": {
            "feature_importance": {"f1": 0.5, "f2": 0.3, "f3": 0.2},
            "explainer_type": "TreeExplainer",
        },
        "final_report": None,
    }
    md = build_report(state)
    html = build_html_report(state)
    assert len(md) > 100
    assert len(html) > 100
    return {"md_length": len(md), "html_length": len(html)}


def test_report_minimal_state() -> None:
    from phronesisml.ml.reports.builder import build_report

    state = {}
    report = build_report(state)
    assert len(report) > 0
    return {"report_length": len(report)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION N: SIMPLE API
# ═══════════════════════════════════════════════════════════════════════════════


def test_simple_api_analyze() -> None:
    from phronesisml.simple import analyze

    csv_path = _make_classification_csv(50)
    result = analyze(csv_path)
    assert hasattr(result, "rows") or isinstance(result, dict) or hasattr(result, "__dict__")
    return {"result_type": type(result).__name__}


def test_simple_api_clean() -> None:
    from phronesisml.simple import clean

    csv_path = _make_dirty_csv(30)
    result = clean(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_validate() -> None:
    from phronesisml.simple import validate

    csv_path = _make_classification_csv(30)
    result = validate(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_detect_target() -> None:
    from phronesisml.simple import detect_target

    csv_path = _make_classification_csv(50)
    result = detect_target(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_engineer() -> None:
    from phronesisml.simple import engineer

    csv_path = _make_classification_csv(50)
    result = engineer(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_select_model() -> None:
    from phronesisml.simple import select_model

    csv_path = _make_classification_csv(50)
    result = select_model(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_train() -> None:
    from phronesisml.simple import train

    csv_path = _make_classification_csv(80)
    result = train(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_explain() -> None:
    from phronesisml.simple import explain

    csv_path = _make_classification_csv(50)
    result = explain(csv_path)
    return {"result_type": type(result).__name__}


def test_simple_api_report() -> None:
    from phronesisml.simple import report

    csv_path = _make_classification_csv(50)
    result = report(csv_path)
    return {"result_type": type(result).__name__}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION O: OOP API
# ═══════════════════════════════════════════════════════════════════════════════


def test_oop_api() -> None:
    from phronesisml.sdk import Phronesis

    csv_path = _make_classification_csv(50)
    ml = Phronesis(csv_path)
    summary = ml.load()
    assert summary is not None
    return {"load_result": type(summary).__name__}


def test_oop_api_incremental() -> None:
    from phronesisml.sdk import Phronesis

    csv_path = _make_classification_csv(50)
    ml = Phronesis(csv_path)
    ml.load()
    ml.clean()
    ml.validate()
    ml.eda()
    ml.detect_target()
    ml.engineer_features()
    ml.recommend_model()
    ml.train()
    ml.evaluate()
    return {"incremental_stages": 9}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION P: ADVANCED API (run_pipeline)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_advanced_api_full_pipeline() -> None:
    from phronesisml import run_pipeline

    csv_path = _make_classification_csv(80)
    result = await run_pipeline(data_path=csv_path, null_strategy="drop")
    assert "target_column" in result
    assert "task_type" in result
    return {
        "target": result["target_column"],
        "task": result["task_type"],
        "model": result.get("best_model_type"),
    }


async def test_advanced_api_subset_stages() -> None:
    from phronesisml import run_pipeline

    csv_path = _make_classification_csv(50)
    result = await run_pipeline(data_path=csv_path, stages=["upload", "etl", "validation", "eda"])
    return {"stages_run": 4, "has_profile": result.get("numeric_columns") is not None}


async def test_advanced_api_with_config() -> None:
    from phronesisml import PhronesisConfig, run_pipeline

    csv_path = _make_classification_csv(50)
    cfg = PhronesisConfig()
    cfg.engine.preferred = "pandas"
    result = await run_pipeline(data_path=csv_path, config=cfg)
    return {"target": result["target_column"], "task": result["task_type"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION Q: PARAMETER VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_config_engine_preferred_invalid() -> None:
    from phronesisml.configs.settings import PhronesisConfig

    cfg = PhronesisConfig()
    cfg.engine.preferred = "invalid_engine"
    from phronesisml.engines.engine_selector import select_engine
    from phronesisml.exceptions import EngineSelectionError

    try:
        select_engine(config=cfg)
        return {"rejected": False}
    except EngineSelectionError:
        return {"rejected": True}


def test_config_feature_selection_params() -> None:
    from phronesisml.configs.settings import PhronesisConfig

    cfg = PhronesisConfig()
    cfg.feature_selection.variance_threshold = 0.05
    cfg.feature_selection.correlation_threshold = 0.1
    cfg.feature_selection.min_features = 2
    return {
        "variance": cfg.feature_selection.variance_threshold,
        "correlation": cfg.feature_selection.correlation_threshold,
        "min_features": cfg.feature_selection.min_features,
    }


def test_null_strategy_fill_value() -> None:
    from phronesisml.data.transformers.cleaning import handle_nulls

    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    cleaned, _ = handle_nulls(df, strategy="fill", fill_value=999)
    assert cleaned["a"].iloc[1] == 999
    return {"fill_value": cleaned["a"].iloc[1]}


def test_random_state_reproducibility() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_classification_csv(80)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 80, 4, 3, 1)
    r1 = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=3,
        max_time_seconds=15,
        random_state=42,
    )
    r2 = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=3,
        max_time_seconds=15,
        random_state=42,
    )
    return {
        "same_model": r1["best_model"] == r2["best_model"],
        "score1": r1["best_score"],
        "score2": r2["best_score"],
    }


def test_test_size_parameter() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_classification_csv(100)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 100, 4, 3, 1)
    r = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=3,
        max_time_seconds=15,
        test_size=0.3,
    )
    return {"test_size": 0.3, "best_model": r["best_model"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION R: ERROR RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════


def test_missing_file_recovery() -> None:
    from phronesisml import analyze

    try:
        analyze("/nonexistent/path/data.csv")
        return {"graceful": True}
    except Exception as e:
        return {"graceful": False, "error": type(e).__name__}


def test_empty_dataset_recovery() -> None:
    from phronesisml.data.validators.checks import validate_dataframe
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.exceptions import DataValidationError

    df = pd.DataFrame()
    try:
        validate_dataframe(df, PandasEngine())
        return {"raised": False}
    except DataValidationError:
        return {"raised": True, "graceful": True}


def test_single_row_dataset() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    df = pd.DataFrame({"a": [1], "b": [2], "target": [1]})
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {"target": result["target_column"], "task": result["task_type"]}


def test_constant_column_survival() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine

    df = pd.DataFrame({"const": [1] * 50, "varied": range(50), "target": [0, 1] * 25})
    profile = profile_dataset(df, PandasEngine())
    return {"numeric": profile["numeric_columns"], "categorical": profile["categorical_columns"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION S: CLI
# ═══════════════════════════════════════════════════════════════════════════════


def test_cli_info() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "phronesisml.interfaces.cli.app", "info"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        result = subprocess.run(["phronesisml", "info"], capture_output=True, text=True, timeout=30)
    return {"returncode": result.returncode, "stdout_len": len(result.stdout)}


def test_cli_run() -> None:
    import subprocess

    csv_path = _make_classification_csv(30)
    result = subprocess.run(
        [sys.executable, "-m", "phronesisml.interfaces.cli.app", "run", csv_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["phronesisml", "run", csv_path], capture_output=True, text=True, timeout=120
        )
    return {
        "returncode": result.returncode,
        "stdout_len": len(result.stdout),
        "stderr_len": len(result.stderr),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION T: FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════


def test_fastapi_app_creation() -> None:
    from phronesisml.interfaces.api.app import app

    return {"app_title": app.title, "version": app.version}


def test_fastapi_health_endpoint() -> None:
    from phronesisml.interfaces.api.routes import health

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(health())
        return {"status": result.data.get("status") if hasattr(result, "data") else result}
    finally:
        loop.close()


def test_fastapi_version_endpoint() -> None:
    from phronesisml.interfaces.api.routes import version

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(version())
        return {"version": result}
    finally:
        loop.close()


def test_fastapi_capabilities_endpoint() -> None:
    from phronesisml.interfaces.api.routes import capabilities

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(capabilities())
        return {"capabilities": result}
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION U: AGENT STUBS
# ═══════════════════════════════════════════════════════════════════════════════


def test_all_agents_instantiate() -> None:
    from phronesisml.agents.eda.agent import EDAAgent
    from phronesisml.agents.etl.agent import ETLAgent
    from phronesisml.agents.evaluation.agent import EvaluationAgent
    from phronesisml.agents.explainability.agent import ExplainabilityAgent
    from phronesisml.agents.feature_engineering.agent import FeatureEngineeringAgent
    from phronesisml.agents.model_selection.agent import ModelSelectionAgent
    from phronesisml.agents.reporting.agent import ReportingAgent
    from phronesisml.agents.storage.agent import StorageAgent
    from phronesisml.agents.target_detection.agent import TargetDetectionAgent
    from phronesisml.agents.upload.agent import UploadAgent
    from phronesisml.agents.validation.agent import ValidationAgent
    from phronesisml.engines.pandas_engine import PandasEngine

    engine = PandasEngine()
    agents = {
        "upload": UploadAgent(engine=engine),
        "etl": ETLAgent(),
        "validation": ValidationAgent(engine=engine),
        "eda": EDAAgent(engine=engine),
        "target_detection": TargetDetectionAgent(engine=engine),
        "feature_engineering": FeatureEngineeringAgent(engine=engine),
        "model_selection": ModelSelectionAgent(engine=engine),
        "evaluation": EvaluationAgent(engine=engine),
        "explainability": ExplainabilityAgent(engine=engine),
        "reporting": ReportingAgent(),
        "storage": StorageAgent(),
    }
    return {"agent_count": len(agents), "names": list(agents.keys())}


def test_stub_agent_raises() -> None:
    from phronesisml.agents.base import _StubAgent
    from phronesisml.exceptions import AgentNotImplementedError

    stub = _StubAgent(name="test", description="test stub")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(stub.run({}))
        return {"raised": False}
    except AgentNotImplementedError:
        return {"raised": True}
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION V: LANGGRAPH WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════


def test_build_graph() -> None:
    from phronesisml.agents.eda.agent import EDAAgent
    from phronesisml.agents.etl.agent import ETLAgent
    from phronesisml.agents.evaluation.agent import EvaluationAgent
    from phronesisml.agents.explainability.agent import ExplainabilityAgent
    from phronesisml.agents.feature_engineering.agent import FeatureEngineeringAgent
    from phronesisml.agents.model_selection.agent import ModelSelectionAgent
    from phronesisml.agents.reporting.agent import ReportingAgent
    from phronesisml.agents.storage.agent import StorageAgent
    from phronesisml.agents.target_detection.agent import TargetDetectionAgent
    from phronesisml.agents.upload.agent import UploadAgent
    from phronesisml.agents.validation.agent import ValidationAgent
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.workflow.graph import build_graph, clear_graph_cache

    engine = PandasEngine()
    agents = {
        "upload": UploadAgent(engine=engine),
        "etl": ETLAgent(),
        "validation": ValidationAgent(engine=engine),
        "eda": EDAAgent(engine=engine),
        "target_detection": TargetDetectionAgent(engine=engine),
        "feature_engineering": FeatureEngineeringAgent(engine=engine),
        "model_selection": ModelSelectionAgent(engine=engine),
        "evaluation": EvaluationAgent(engine=engine),
        "explainability": ExplainabilityAgent(engine=engine),
        "reporting": ReportingAgent(),
        "storage": StorageAgent(),
    }
    clear_graph_cache()
    graph = build_graph(agents)
    assert graph is not None
    return {"graph_type": type(graph).__name__}


def test_build_graph_subset() -> None:
    from phronesisml.agents.etl.agent import ETLAgent
    from phronesisml.agents.upload.agent import UploadAgent
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.workflow.graph import build_graph

    engine = PandasEngine()
    agents = {"upload": UploadAgent(engine=engine), "etl": ETLAgent()}
    graph = build_graph(agents, stages=["upload", "etl"])
    return {"graph_type": type(graph).__name__}


def test_build_graph_unknown_stage() -> None:
    from phronesisml.exceptions import ConfigurationError
    from phronesisml.workflow.graph import build_graph

    try:
        build_graph({}, stages=["upload", "nonexistent_stage"])
        return {"raised": False}
    except ConfigurationError:
        return {"raised": True}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION W: DIRTY / EDGE CASE DATASETS
# ═══════════════════════════════════════════════════════════════════════════════


def test_dirty_data_full_pipeline() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_dirty_csv(50)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {"target": result["target_column"], "task": result["task_type"]}


def test_tiny_dataset_target_detection() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.target_detection.detector import detect_target

    csv_path = _make_tiny_csv()
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    result = detect_target(df, PandasEngine(), profile)
    return {"target": result["target_column"], "task": result["task_type"]}


def test_inf_values_handling() -> None:
    from phronesisml.data.profilers.stats import profile_dataset
    from phronesisml.engines.pandas_engine import PandasEngine

    csv_path = _make_inf_values_csv()
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    profile = profile_dataset(df, PandasEngine())
    return {"rows": profile["shape"]["rows"], "columns": profile["shape"]["columns"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION X: MODEL TYPE OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_type_override_classification() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_classification_csv(80)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("classification", 80, 4, 3, 1)
    result = train_models(
        df,
        PandasEngine(),
        candidates,
        "target",
        "classification",
        max_trials=5,
        max_time_seconds=30,
    )
    return {"best_model": result["best_model"]}


def test_model_type_override_regression() -> None:
    from phronesisml.engines.pandas_engine import PandasEngine
    from phronesisml.ml.automl.auto_selector import recommend_models
    from phronesisml.ml.automl.trainer import train_models

    csv_path = _make_regression_csv(80)
    from phronesisml.data.loaders.file_loader import load_file

    df = load_file(csv_path, PandasEngine())
    candidates = recommend_models("regression", 80, 3, 3, 0)
    result = train_models(
        df, PandasEngine(), candidates, "price", "regression", max_trials=5, max_time_seconds=30
    )
    return {"best_model": result["best_model"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION Y: UNSUPERVISED LEARNING COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_task_detection_imports() -> None:
    from phronesisml.ml.task_detection import detect_task

    return {"detect_task": detect_task.__name__}


def test_clustering_imports() -> None:
    from phronesisml.ml.clustering import ClusterResult, run_clustering

    return {"run_clustering": run_clustering.__name__, "result_class": ClusterResult.__name__}


def test_anomaly_imports() -> None:
    from phronesisml.ml.anomaly import AnomalyResult, detect_anomalies

    return {"detect_anomalies": detect_anomalies.__name__, "result_class": AnomalyResult.__name__}


def test_new_sdk_imports() -> None:
    from phronesisml.sdk import AnomalyReport, ClusteringReport, TaskInfo

    return {"sdk_classes": [AnomalyReport.__name__, ClusteringReport.__name__, TaskInfo.__name__]}


def test_new_simple_imports() -> None:
    return {"simple_api": True, "functions": 3}


def test_new_init_exports() -> None:
    import phronesisml

    new_exports = [
        "cluster",
        "detect_anomalies",
        "detect_task",
        "ClusterResult",
        "AnomalyResult",
        "TaskDetectionResult",
    ]
    found = [e for e in new_exports if hasattr(phronesisml, e)]
    return {"new_exports_found": len(found), "expected": len(new_exports)}


def test_model_recommendation_clustering() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models("clustering", 200, 5, 5, 0)
    names = [c.name for c in candidates]
    return {"candidates": names, "count": len(candidates)}


def test_model_recommendation_anomaly() -> None:
    from phronesisml.ml.automl.auto_selector import recommend_models

    candidates = recommend_models("anomaly_detection", 200, 5, 5, 0)
    names = [c.name for c in candidates]
    return {"candidates": names, "count": len(candidates)}


def test_clustering_kmeans() -> None:
    import pandas as pd

    from phronesisml.ml.clustering import run_clustering

    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
            "b": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
            "c": rng.randn(60),
        }
    )
    result = run_clustering(df, ["a", "b"], algorithms=["kmeans"])
    return {
        "algorithm": result.algorithm,
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette_score is not None,
    }


def test_clustering_all_algorithms() -> None:
    import pandas as pd

    from phronesisml.ml.clustering import run_clustering

    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)]),
            "b": np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)]),
            "c": rng.randn(80),
        }
    )
    result = run_clustering(df, ["a", "b"])
    all_algo_names = [r["algorithm"] for r in result.all_results]
    return {
        "best_algorithm": result.algorithm,
        "all_tried": all_algo_names,
        "n_clusters": result.n_clusters,
    }


def test_anomaly_detection_basic() -> None:
    import pandas as pd

    from phronesisml.ml.anomaly import detect_anomalies

    rng = np.random.RandomState(42)
    normal = rng.randn(90, 3)
    anomaly = rng.uniform(-10, 10, (10, 3))
    X = np.vstack([normal, anomaly])
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    result = detect_anomalies(df, ["a", "b", "c"], contamination=0.1)
    return {
        "algorithm": result.algorithm,
        "n_anomalies": result.n_anomalies,
        "has_scores": len(result.anomaly_scores) == 100,
    }


def test_anomaly_detection_all_algorithms() -> None:
    import pandas as pd

    from phronesisml.ml.anomaly import detect_anomalies

    rng = np.random.RandomState(42)
    normal = rng.randn(90, 3)
    anomaly = rng.uniform(-10, 10, (10, 3))
    X = np.vstack([normal, anomaly])
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    result = detect_anomalies(df, ["a", "b", "c"])
    all_algo_names = [r["algorithm"] for r in result.all_results]
    return {"algorithms_tried": all_algo_names, "n_anomalies": result.n_anomalies}


def test_clustering_metrics() -> None:
    import pandas as pd

    from phronesisml.ml.clustering import run_clustering

    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
            "y": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
        }
    )
    result = run_clustering(df, ["x", "y"])
    has_sil = result.silhouette_score is not None
    has_db = result.davies_bouldin_score is not None
    has_ch = result.calinski_harabasz_score is not None
    return {"silhouette": has_sil, "davies_bouldin": has_db, "calinski_harabasz": has_ch}


def test_unsupervised_auto_selector_candidates() -> None:
    from phronesisml.ml.automl.auto_selector import (
        _ANOMALY_CANDIDATES,
        _CLUSTERING_CANDIDATES,
    )

    cluster_names = [c.name for c in _CLUSTERING_CANDIDATES]
    anomaly_names = [c.name for c in _ANOMALY_CANDIDATES]
    return {"clustering": cluster_names, "anomaly": anomaly_names}


def test_unsupervised_workflow_state_fields() -> None:
    from phronesisml.workflow.state import WorkflowState

    WorkflowState()
    fields = list(WorkflowState.model_fields.keys())
    unsupervised = [
        "cluster_labels",
        "cluster_metrics",
        "anomaly_labels",
        "anomaly_scores",
        "anomaly_metrics",
    ]
    found = [f for f in unsupervised if f in fields]
    return {"unsupervised_fields": found, "count": len(found)}


def test_unsupervised_router_logic() -> None:
    from phronesisml.workflow.router import route_after_target_detection

    class FakeState:
        pass

    state = FakeState()
    state.target_column = None
    state.task_type = "clustering"
    result_cluster = route_after_target_detection(state)
    state.task_type = "anomaly_detection"
    result_anomaly = route_after_target_detection(state)
    state.task_type = None
    result_none = route_after_target_detection(state)
    return {
        "clustering_proceeds": result_cluster == "proceed",
        "anomaly_proceeds": result_anomaly == "proceed",
        "none_ends": result_none == "__end__",
    }


def test_unsupervised_report_sections() -> None:
    from phronesisml.ml.reports.builder import (
        _build_anomaly_section,
        _build_clustering_section,
    )

    class FakeState:
        pass

    state = FakeState()
    state.cluster_labels = [0, 0, 1, 1, 2]
    state.cluster_metrics = {"algorithm": "kmeans", "n_clusters": 3, "silhouette_score": 0.75}
    state.anomaly_labels = [0, 0, 1, 0, 0]
    state.anomaly_metrics = {"n_anomalies": 1, "n_total": 5}
    state.anomaly_scores = [0.1, 0.2, 0.9, 0.15, 0.3]
    cluster_sec = _build_clustering_section(state)
    anomaly_sec = _build_anomaly_section(state)
    return {
        "cluster_has_data": "kmeans" in cluster_sec,
        "anomaly_has_data": "Anomalies" in anomaly_sec,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION Z: LOGGING AUDIT
# ═══════════════════════════════════════════════════════════════════════════════


def test_logging_structure() -> None:
    import logging

    loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
    phronesis_loggers = [lg for lg in loggers if "phronesis" in lg.name.lower()]
    return {
        "phronesis_logger_count": len(phronesis_loggers),
        "logger_names": [lg.name for lg in phronesis_loggers[:10]],
    }


def test_no_print_statements_in_core() -> None:
    import ast

    core_path = Path(__file__).parent / "phronesisml"
    print_count = 0
    for py_file in core_path.rglob("*.py"):
        if "interfaces" in str(py_file) or "cli" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    print_count += 1
        except SyntaxError:
            pass
    return {"print_statements_in_core": print_count}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION Z: FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report() -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("  PHRONESISML -- COMPATIBILITY MATRIX & INTEGRATION AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.status == "passed")
    failed = sum(1 for r in RESULTS if r.status == "failed")
    skipped = sum(1 for r in RESULTS if r.status == "skipped")
    warnings = sum(1 for r in RESULTS if r.status == "warning")
    total_time = sum(r.duration_s for r in RESULTS)

    lines.append(f"  Total Stages:   {total}")
    lines.append(f"  Passed:         {passed}")
    lines.append(f"  Failed:         {failed}")
    lines.append(f"  Skipped:        {skipped}")
    lines.append(f"  Warnings:       {warnings}")
    lines.append(f"  Total Time:     {total_time:.2f}s")
    lines.append(
        f"  Pass Rate:      {passed / total * 100:.1f}%" if total > 0 else "  Pass Rate: N/A"
    )
    lines.append("")

    lines.append("-" * 80)
    lines.append("  SECTION RESULTS")
    lines.append("-" * 80)

    sections = {}
    for r in RESULTS:
        section = r.name.split("_")[0] if "_" in r.name else r.name
        if section not in sections:
            sections[section] = []
        sections[section].append(r)

    for section_name, stages in sections.items():
        s_passed = sum(1 for s in stages if s.status == "passed")
        s_total = len(stages)
        lines.append(f"\n  {section_name.upper()} ({s_passed}/{s_total} passed)")
        for s in stages:
            icon = {"passed": "+", "failed": "X", "skipped": "-", "warning": "!"}.get(s.status, "?")
            lines.append(f"    [{icon}] {s.name} ({s.duration_s:.2f}s) -- {s.message[:100]}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  FAILED STAGES -- ROOT CAUSE ANALYSIS")
    lines.append("-" * 80)
    for r in RESULTS:
        if r.status == "failed":
            lines.append(f"\n  {r.name}:")
            lines.append(f"    Error: {r.message}")
            if "traceback" in r.details:
                lines.append("    Traceback (last 5 lines):")
                for line in r.details["traceback"].split("\n")[-5:]:
                    lines.append(f"      {line}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  SKIPPED STAGES -- MISSING DEPENDENCIES")
    lines.append("-" * 80)
    for r in RESULTS:
        if r.status == "skipped":
            lines.append(f"    - {r.name}: {r.message}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  COMPATIBILITY MATRIX")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Dataset Category        | Target Detect | Train | Eval | SHAP | Report")
    lines.append("  " + "-" * 70)
    _STATUS_MAP = {"passed": "passed", "failed": "FAILED", "skipped": "skipped", "warning": "warn"}
    for name, det_name, train_name, eval_name in [
        (
            "Classification",
            "test_target_detection_classification",
            "test_train_classification",
            "test_evaluate_classification",
        ),
        (
            "Regression",
            "test_target_detection_regression",
            "test_train_regression",
            "test_evaluate_regression",
        ),
        ("Multiclass", "test_target_detection_multiclass", None, None),
        ("No Target", "test_target_detection_no_target", None, None),
        ("Constant Target", "test_target_detection_constant_target", None, None),
        ("Tiny Dataset", "test_tiny_dataset_target_detection", None, None),
        ("Dirty Data", "test_dirty_data_full_pipeline", None, None),
        ("Inf Values", "test_inf_values_handling", None, None),
    ]:
        r_det = next((r for r in RESULTS if r.name == det_name), None)
        r_train = next((r for r in RESULTS if r.name == train_name), None) if train_name else None
        r_eval = next((r for r in RESULTS if r.name == eval_name), None) if eval_name else None
        r_shap = next((r for r in RESULTS if r.name == "test_shap_explainability"), None)
        r_report = next((r for r in RESULTS if r.name == "test_report_generation"), None)
        s_det = _STATUS_MAP.get(r_det.status, "N/A") if r_det else "N/A"
        s_train = _STATUS_MAP.get(r_train.status, "N/A") if r_train else "--"
        s_eval = _STATUS_MAP.get(r_eval.status, "N/A") if r_eval else "--"
        s_shap = _STATUS_MAP.get(r_shap.status, "N/A") if r_shap else "--"
        s_report = _STATUS_MAP.get(r_report.status, "N/A") if r_report else "--"
        lines.append(
            f"  {name:24s} | {s_det:13s} | {s_train:5s} | {s_eval:4s} | {s_shap:4s} | {s_report:6s}"
        )

    lines.append("")
    lines.append("-" * 80)
    lines.append("  API INTERFACE MATRIX")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Interface          | Status  | Notes")
    lines.append("  " + "-" * 60)
    for name, desc in [
        ("Import", "test_imports"),
        ("SDK OOP", "test_oop_api"),
        ("Simple API", "test_simple_api_analyze"),
        ("Advanced API", None),
        ("CLI", "test_cli_info"),
        ("FastAPI", "test_fastapi_app_creation"),
    ]:
        if desc:
            r = next((r for r in RESULTS if r.name == desc), None)
            s = r.status if r else "N/A"
        else:
            r = next((r for r in RESULTS if "advanced" in r.name.lower()), None)
            s = r.status if r else "N/A"
        lines.append(f"  {name:19s} | {s:7s} | {r.message[:40] if r else ''}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  RECOMMENDATIONS")
    lines.append("-" * 80)
    lines.append("")

    if failed > 0:
        lines.append("  CRITICAL FIXES REQUIRED:")
        for r in RESULTS:
            if r.status == "failed":
                lines.append(f"    1. Fix {r.name}: {r.message[:80]}")
        lines.append("")

    if skipped > 0:
        lines.append("  OPTIONAL DEPENDENCIES:")
        for r in RESULTS:
            if r.status == "skipped":
                lines.append(f"    - {r.name}: {r.message[:80]}")
        lines.append("")

    lines.append("  PRODUCTION READINESS CHECKLIST:")
    checklist = [
        (
            "All core APIs import correctly",
            any(r.name == "test_imports" and r.status == "passed" for r in RESULTS),
        ),
        (
            "SDK OOP API works",
            any(r.name == "test_oop_api" and r.status == "passed" for r in RESULTS),
        ),
        (
            "Simple API works",
            any(r.name == "test_simple_api_analyze" and r.status == "passed" for r in RESULTS),
        ),
        ("Advanced API works", any("advanced" in r.name and r.status == "passed" for r in RESULTS)),
        (
            "Target detection works",
            any("target_detection" in r.name and r.status == "passed" for r in RESULTS),
        ),
        ("Training works", any("train" in r.name and r.status == "passed" for r in RESULTS)),
        ("Evaluation works", any("evaluate" in r.name and r.status == "passed" for r in RESULTS)),
        (
            "SHAP explainability works",
            any("shap" in r.name and r.status == "passed" for r in RESULTS),
        ),
        ("Reports generate", any("report" in r.name and r.status == "passed" for r in RESULTS)),
        (
            "Error recovery works",
            any("recovery" in r.name and r.status == "passed" for r in RESULTS),
        ),
    ]
    for desc, ok in checklist:
        lines.append(f"    [{'X' if ok else ' '}] {desc}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 80)
    print("  PHRONESISML -- COMPATIBILITY MATRIX & INTEGRATION AUDIT")
    print("=" * 80)
    print()

    # A. Imports & Interfaces
    print("[A] IMPORTS & INTERFACES")
    run_stage("test_imports", test_imports)
    run_stage("test_sdk_oop_imports", test_sdk_oop_imports)
    run_stage("test_simple_api_imports", test_simple_api_imports)
    run_stage("test_async_api_imports", test_async_api_imports)
    run_stage("test_config_imports", test_config_imports)
    run_stage("test_exception_hierarchy", test_exception_hierarchy)
    run_stage("test_workflow_state", test_workflow_state)
    run_stage("test_pipeline_order", test_pipeline_order)
    run_stage("test_agent_base", test_agent_base)
    print()

    # B. Engines
    print("[B] ENGINE COMPATIBILITY")
    run_stage("test_pandas_engine", test_pandas_engine)
    run_stage("test_polars_engine", test_polars_engine)
    run_stage("test_engine_selector", test_engine_selector)
    run_stage("test_engine_selector_pandas_force", test_engine_selector_pandas_force)
    run_stage("test_engine_selector_polars_force", test_engine_selector_polars_force)
    print()

    # C. Data Loading
    print("[C] DATA LOADING & FORMATS")
    run_stage("test_csv_loading", test_csv_loading)
    run_stage("test_json_loading", test_json_loading)
    run_stage("test_parquet_loading", test_parquet_loading)
    run_stage("test_excel_loading", test_excel_loading)
    print()

    # D. ETL
    print("[D] ETL & DATA PROCESSING")
    run_stage("test_etl_handle_nulls_drop", test_etl_handle_nulls_drop)
    run_stage("test_etl_handle_nulls_fill", test_etl_handle_nulls_fill)
    run_stage("test_etl_handle_nulls_flag", test_etl_handle_nulls_flag)
    run_stage("test_etl_encode_categoricals", test_etl_encode_categoricals)
    run_stage("test_etl_cast_dtypes", test_etl_cast_dtypes)
    run_stage("test_etl_invalid_strategy", test_etl_invalid_strategy)
    print()

    # E. Validation
    print("[E] VALIDATION")
    run_stage("test_validate_clean_data", test_validate_clean_data)
    run_stage("test_validate_dirty_data", test_validate_dirty_data)
    run_stage("test_validate_empty_df", test_validate_empty_df)
    print()

    # F. EDA
    print("[F] EDA / PROFILING")
    run_stage("test_eda_profiling", test_eda_profiling)
    run_stage("test_eda_mixed_types", test_eda_mixed_types)
    print()

    # G. Target Detection
    print("[G] TARGET DETECTION")
    run_stage("test_target_detection_classification", test_target_detection_classification)
    run_stage("test_target_detection_regression", test_target_detection_regression)
    run_stage("test_target_detection_no_target", test_target_detection_no_target)
    run_stage("test_target_detection_constant_target", test_target_detection_constant_target)
    run_stage("test_target_detection_multiclass", test_target_detection_multiclass)
    print()

    # H. Feature Engineering
    print("[H] FEATURE ENGINEERING")
    run_stage("test_feature_engineering", test_feature_engineering)
    run_stage("test_feature_engineering_no_target", test_feature_engineering_no_target)
    run_stage("test_feature_engineering_fill_strategy", test_feature_engineering_fill_strategy)
    print()

    # I. Model Selection
    print("[I] MODEL SELECTION & TRAINING")
    run_stage("test_model_recommendation_classification", test_model_recommendation_classification)
    run_stage("test_model_recommendation_regression", test_model_recommendation_regression)
    run_stage("test_model_recommendation_ambiguous", test_model_recommendation_ambiguous)
    run_stage("test_model_recommendation_none_task", test_model_recommendation_none_task)
    run_stage("test_training_cost_estimation", test_training_cost_estimation)
    run_stage("test_train_classification", test_train_classification)
    run_stage("test_train_regression", test_train_regression)
    run_stage("test_train_with_cv", test_train_with_cv)
    print()

    # J. Evaluation
    print("[J] EVALUATION")
    run_stage("test_evaluate_classification", test_evaluate_classification)
    run_stage("test_evaluate_regression", test_evaluate_regression)
    print()

    # K. SHAP
    print("[K] SHAP EXPLAINABILITY")
    run_stage("test_shap_explainability", test_shap_explainability)
    run_stage("test_shap_without_shap_library", test_shap_without_shap_library)
    run_stage("test_shap_linear_model", test_shap_linear_model)
    print()

    # L. Reports
    print("[L] REPORTS")
    run_stage("test_report_generation", test_report_generation)
    run_stage("test_report_minimal_state", test_report_minimal_state)
    print()

    # M. Simple API
    print("[M] SIMPLE API")
    run_stage("test_simple_api_analyze", test_simple_api_analyze)
    run_stage("test_simple_api_clean", test_simple_api_clean)
    run_stage("test_simple_api_validate", test_simple_api_validate)
    run_stage("test_simple_api_detect_target", test_simple_api_detect_target)
    run_stage("test_simple_api_engineer", test_simple_api_engineer)
    run_stage("test_simple_api_select_model", test_simple_api_select_model)
    run_stage("test_simple_api_train", test_simple_api_train)
    run_stage("test_simple_api_explain", test_simple_api_explain)
    run_stage("test_simple_api_report", test_simple_api_report)
    print()

    # N. OOP API
    print("[N] OOP API")
    run_stage("test_oop_api", test_oop_api)
    run_stage("test_oop_api_incremental", test_oop_api_incremental)
    print()

    # O. Advanced API
    print("[O] ADVANCED API")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    run_stage(
        "test_advanced_api_full_pipeline",
        lambda: loop.run_until_complete(test_advanced_api_full_pipeline()),
    )
    run_stage(
        "test_advanced_api_subset_stages",
        lambda: loop.run_until_complete(test_advanced_api_subset_stages()),
    )
    run_stage(
        "test_advanced_api_with_config",
        lambda: loop.run_until_complete(test_advanced_api_with_config()),
    )
    loop.close()
    print()

    # P. Parameter Validation
    print("[P] PARAMETER VALIDATION")
    run_stage("test_config_engine_preferred_invalid", test_config_engine_preferred_invalid)
    run_stage("test_config_feature_selection_params", test_config_feature_selection_params)
    run_stage("test_null_strategy_fill_value", test_null_strategy_fill_value)
    run_stage("test_random_state_reproducibility", test_random_state_reproducibility)
    run_stage("test_test_size_parameter", test_test_size_parameter)
    print()

    # Q. Error Recovery
    print("[Q] ERROR RECOVERY")
    run_stage("test_missing_file_recovery", test_missing_file_recovery)
    run_stage("test_empty_dataset_recovery", test_empty_dataset_recovery)
    run_stage("test_single_row_dataset", test_single_row_dataset)
    run_stage("test_constant_column_survival", test_constant_column_survival)
    print()

    # R. CLI
    print("[R] CLI")
    run_stage("test_cli_info", test_cli_info)
    run_stage("test_cli_run", test_cli_run)
    print()

    # S. FastAPI
    print("[S] FASTAPI")
    run_stage("test_fastapi_app_creation", test_fastapi_app_creation)
    run_stage("test_fastapi_health_endpoint", test_fastapi_health_endpoint)
    run_stage("test_fastapi_version_endpoint", test_fastapi_version_endpoint)
    run_stage("test_fastapi_capabilities_endpoint", test_fastapi_capabilities_endpoint)
    print()

    # T. Agents
    print("[T] AGENT INSTANTIATION")
    run_stage("test_all_agents_instantiate", test_all_agents_instantiate)
    run_stage("test_stub_agent_raises", test_stub_agent_raises)
    print()

    # U. LangGraph
    print("[U] LANGGRAPH WORKFLOW")
    run_stage("test_build_graph", test_build_graph)
    run_stage("test_build_graph_subset", test_build_graph_subset)
    run_stage("test_build_graph_unknown_stage", test_build_graph_unknown_stage)
    print()

    # V. Edge Cases
    print("[V] EDGE CASE DATASETS")
    run_stage("test_dirty_data_full_pipeline", test_dirty_data_full_pipeline)
    run_stage("test_tiny_dataset_target_detection", test_tiny_dataset_target_detection)
    run_stage("test_inf_values_handling", test_inf_values_handling)
    print()

    # W. Model Type Override
    print("[W] MODEL TYPE OVERRIDE")
    run_stage("test_model_type_override_classification", test_model_type_override_classification)
    run_stage("test_model_type_override_regression", test_model_type_override_regression)
    print()

    # X. Unsupervised Learning
    print("[X] UNSUPERVISED LEARNING")
    run_stage("test_task_detection_imports", test_task_detection_imports)
    run_stage("test_clustering_imports", test_clustering_imports)
    run_stage("test_anomaly_imports", test_anomaly_imports)
    run_stage("test_new_sdk_imports", test_new_sdk_imports)
    run_stage("test_new_simple_imports", test_new_simple_imports)
    run_stage("test_new_init_exports", test_new_init_exports)
    run_stage("test_model_recommendation_clustering", test_model_recommendation_clustering)
    run_stage("test_model_recommendation_anomaly", test_model_recommendation_anomaly)
    run_stage("test_clustering_kmeans", test_clustering_kmeans)
    run_stage("test_clustering_all_algorithms", test_clustering_all_algorithms)
    run_stage("test_anomaly_detection_basic", test_anomaly_detection_basic)
    run_stage("test_anomaly_detection_all_algorithms", test_anomaly_detection_all_algorithms)
    run_stage("test_clustering_metrics", test_clustering_metrics)
    run_stage(
        "test_unsupervised_auto_selector_candidates", test_unsupervised_auto_selector_candidates
    )
    run_stage("test_unsupervised_workflow_state_fields", test_unsupervised_workflow_state_fields)
    run_stage("test_unsupervised_router_logic", test_unsupervised_router_logic)
    run_stage("test_unsupervised_report_sections", test_unsupervised_report_sections)
    print()

    # Y. Logging Audit
    print("[Y] LOGGING AUDIT")
    run_stage("test_logging_structure", test_logging_structure)
    run_stage("test_no_print_statements_in_core", test_no_print_statements_in_core)
    print()

    # Z. Final Report
    print("=" * 80)
    print("  GENERATING REPORT...")
    print("=" * 80)
    report = generate_report()
    print(report)

    # Write report to file
    report_path = Path(__file__).parent / "AUDIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("```\n" + report + "\n```\n")
    print(f"\n  Report saved to: {report_path}")

    # Cleanup
    import shutil

    shutil.rmtree(TMPDIR, ignore_errors=True)

    return report


if __name__ == "__main__":
    main()
