"""Unit tests for profiling engine and duplicate detector."""

from __future__ import annotations

import pandas as pd

from strataforge.core.types import DataType
from strataforge.profiling.duplicates import detect_duplicates
from strataforge.profiling.profiler import profile_column, profile_dataset


def test_profile_column_numeric():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, None], name="test_metric")
    cp = profile_column(series)

    assert cp.name == "test_metric"
    assert cp.inferred_type == DataType.FLOAT
    assert cp.total_count == 6
    assert cp.null_count == 1
    assert round(cp.null_percentage, 1) == 16.7
    assert cp.min_val == 10.0
    assert cp.max_val == 50.0
    assert cp.mean_val == 30.0
    assert cp.quantiles is not None
    assert cp.quantiles.p50 == 30.0


def test_profile_column_categorical():
    series = pd.Series(["apple", "banana", "apple", "apple", "orange", None], name="fruit")
    cp = profile_column(series)

    assert cp.name == "fruit"
    assert cp.inferred_type in (DataType.CATEGORICAL, DataType.STRING)
    assert cp.total_count == 6
    assert cp.null_count == 1
    assert "apple" in cp.top_categories
    assert cp.top_categories["apple"] == 3


def test_detect_duplicates_exact_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 4],
            "name": ["a", "b", "b", "c", "d"],
        }
    )
    rep = detect_duplicates(df)

    assert rep.total_rows == 5
    assert rep.duplicate_rows_count == 2  # rows with id=2 and name='b' appear twice
    assert rep.has_critical_duplicates is True
    assert rep.duplicate_percentage == 40.0
    assert len(rep.sample_duplicates) > 0


def test_detect_duplicates_key_subset():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    rep = detect_duplicates(df, subset=["id"])
    assert rep.duplicate_rows_count == 2
    assert rep.subset_columns == ["id"]


def test_profile_dataset(baseline_dataframe: pd.DataFrame):
    prof = profile_dataset(baseline_dataframe)

    assert prof.row_count == 500
    assert prof.column_count == len(baseline_dataframe.columns)
    assert "transaction_id" in prof.columns
    assert "amount" in prof.columns
    assert prof.columns["amount"].inferred_type == DataType.FLOAT
    assert prof.duplicate_count == 0
