"""Minimal configuration for AetherML.

Uses Pydantic ``BaseSettings`` so that values can be overridden via
environment variables or constructor arguments.  This is the minimal
configuration needed to make the Upload → ETL vertical slice runnable.
Full configuration infrastructure is deferred to a later pass.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Engine selection preferences."""

    preferred: str | None = Field(
        default=None,
        description="Force a specific engine ('pandas', 'polars', 'spark'). None = auto-select.",
    )
    spark_master: str = Field(
        default="local[*]",
        description="Spark master URL (only used when Spark is selected).",
    )


class DataConfig(BaseModel):
    """Data loading defaults."""

    default_format: str = Field(
        default="auto",
        description=(
            "Default file format when not inferred from extension "
            "('auto', 'csv', 'parquet', 'json')."
        ),
    )
    max_memory_bytes: int = Field(
        default=500 * 1024 * 1024,  # 500 MB
        description=(
            "Memory threshold (bytes) above which Spark is preferred "
            "over in-process engines."
        ),
    )


class AetherMLConfig(BaseModel):
    """Top-level SDK configuration."""

    engine: EngineConfig = Field(default_factory=EngineConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    model_config = {"extra": "ignore"}
