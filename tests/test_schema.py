"""Unit tests for schema inference and breaking change analysis."""

from __future__ import annotations

import pandas as pd

from strataforge.core.schema import compare_schemas, infer_series_type, is_compatible_type_widening
from strataforge.core.types import ChangeType, DataType


def test_infer_integer_type():
    s = pd.Series([1, 2, 3, 100, -5])
    assert infer_series_type(s) == DataType.INTEGER


def test_infer_float_type():
    s = pd.Series([1.5, 2.3, 3.14, 100.0])
    assert infer_series_type(s) == DataType.FLOAT


def test_infer_boolean_type():
    s1 = pd.Series([True, False, True, False])
    assert infer_series_type(s1) == DataType.BOOLEAN

    s2 = pd.Series(["true", "false", "true", "false"] * 10)
    assert infer_series_type(s2) == DataType.BOOLEAN


def test_infer_datetime_type():
    s = pd.Series(["2026-01-01", "2026-01-02 14:00:00", "2026-01-03"] * 10)
    assert infer_series_type(s) == DataType.DATETIME


def test_infer_categorical_type():
    # Low cardinality relative to total count
    s = pd.Series(["apple", "banana", "orange"] * 50)
    assert infer_series_type(s) == DataType.CATEGORICAL


def test_infer_string_type():
    # High cardinality unique strings
    s = pd.Series([f"random_text_token_{i}" for i in range(100)])
    assert infer_series_type(s) == DataType.STRING


def test_infer_empty_or_null_series():
    s = pd.Series([None, None, None])
    assert infer_series_type(s) == DataType.UNKNOWN


def test_is_compatible_type_widening():
    assert is_compatible_type_widening(DataType.INTEGER, DataType.FLOAT) is True
    assert is_compatible_type_widening(DataType.FLOAT, DataType.INTEGER) is False
    assert is_compatible_type_widening(DataType.STRING, DataType.INTEGER) is False


def test_compare_schemas_breaking_and_non_breaking():
    baseline_schema = {
        "id": {"inferred_type": "integer", "null_percentage": 0.0},
        "amount": {"inferred_type": "float", "null_percentage": 0.0},
        "status": {"inferred_type": "categorical", "null_percentage": 0.0},
    }

    current_schema = {
        "id": {
            "inferred_type": "integer",
            "null_percentage": 10.0,
        },  # nullability regression (breaking)
        "amount": {"inferred_type": "string", "null_percentage": 0.0},  # type mutation (breaking)
        "new_feature": {
            "inferred_type": "float",
            "null_percentage": 0.0,
        },  # added column (non-breaking)
        # status removed (breaking)
    }

    changes = compare_schemas(baseline_schema, current_schema)

    # Verify changes
    removed = [c for c in changes if c.change_kind == "removed"]
    assert len(removed) == 1
    assert removed[0].column_name == "status"
    assert removed[0].change_type == ChangeType.BREAKING

    added = [c for c in changes if c.change_kind == "added"]
    assert len(added) == 1
    assert added[0].column_name == "new_feature"
    assert added[0].change_type == ChangeType.NON_BREAKING

    type_mutated = [c for c in changes if c.change_kind == "type_changed"]
    assert len(type_mutated) == 1
    assert type_mutated[0].column_name == "amount"
    assert type_mutated[0].change_type == ChangeType.BREAKING

    null_mutated = [c for c in changes if c.change_kind == "nullability_changed"]
    assert len(null_mutated) == 1
    assert null_mutated[0].column_name == "id"
    assert null_mutated[0].change_type == ChangeType.BREAKING
