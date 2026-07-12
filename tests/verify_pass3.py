"""End-to-end pipeline verification script for Pass 3."""

from __future__ import annotations

import asyncio

import aetherml


async def main() -> None:
    result = await aetherml.run_pipeline(
        data_path="tests/fixtures/sample.csv",
        stages=["upload", "etl", "validation", "eda"],
    )
    print("Pipeline result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Verify no field collisions
    assert result["row_count"] == 5, f"row_count={result['row_count']}"
    assert result["column_count"] == 4, f"column_count={result['column_count']}"
    assert result["transformations"] == 2, f"transformations={result['transformations']}"
    assert result["validation_passed"] is True, f"validation_passed={result['validation_passed']}"
    assert result["numeric_columns"] is not None, "numeric_columns should be set"
    assert result["categorical_columns"] is not None, "categorical_columns should be set"

    # After ETL encoding, all columns are numeric (categoricals label-encoded)
    assert (
        len(result["numeric_columns"]) == 4
    ), f"Expected 4 numeric columns, got {result['numeric_columns']}"

    print("\nAll assertions passed!")


if __name__ == "__main__":
    asyncio.run(main())
