"""Schema inference engine, type detection, and breaking change analysis."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from strataforge.core.types import ChangeType, DataType, SchemaChange

# Regex heuristics for fast date and boolean detection in string columns
DATE_REGEX = re.compile(
    r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # YYYY-MM-DD
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4}"  # MM-DD-YYYY or DD-MM-YYYY
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)$"  # ISO8601
)
BOOLEAN_STRINGS = {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}


def infer_series_type(series: pd.Series) -> DataType:
    """Infer the semantic DataType of a pandas Series using statistical and type heuristics."""
    # Strip nulls for inference
    non_null = series.dropna()
    total_valid = len(non_null)

    if total_valid == 0:
        return DataType.UNKNOWN

    dtype_str = str(series.dtype).lower()

    # 1. Native boolean
    if "bool" in dtype_str:
        return DataType.BOOLEAN

    # 2. Native datetime
    if "datetime" in dtype_str or "timestamp" in dtype_str:
        return DataType.DATETIME

    # 3. Native Integer
    if "int" in dtype_str:
        # Check if it behaves as a binary flag (0/1)
        unique_vals = set(non_null.unique())
        if unique_vals.issubset({0, 1}) and total_valid >= 5:
            # Check column name or distinctness; keep as integer unless boolean semantics are dominant
            pass
        return DataType.INTEGER

    # 4. Native Float
    if "float" in dtype_str:
        return DataType.FLOAT

    # 5. String / Object / Categorical
    if "category" in dtype_str:
        return DataType.CATEGORICAL

    # If it's an object/string series, inspect contents
    sample = non_null.astype(str).str.strip().iloc[:1000]

    # Check for boolean string representations
    lowered = sample.str.lower()
    if set(lowered.unique()).issubset(BOOLEAN_STRINGS) and len(lowered.unique()) <= 2:
        return DataType.BOOLEAN

    # Check for datetime patterns before integer/float to prevent parsing timestamps as numbers
    matches_date = sample.apply(lambda s: bool(DATE_REGEX.match(s)))
    if matches_date.mean() >= 0.85:
        # Verify with pd.to_datetime on a sample
        try:
            pd.to_datetime(sample.iloc[:100], format="mixed", errors="raise")
            return DataType.DATETIME
        except Exception:
            pass

    # Check for integer representation in string
    try:
        sample.astype(int)
        return DataType.INTEGER
    except (ValueError, TypeError, OverflowError):
        pass

    # Check for float representation in string
    try:
        sample.astype(float)
        return DataType.FLOAT
    except (ValueError, TypeError, OverflowError):
        pass

    # Check for categorical: low cardinality relative to dataset size
    distinct_ratio = series.nunique() / max(len(series), 1)
    distinct_count = series.nunique()
    if (distinct_count <= 50 and total_valid >= 20) or (
        distinct_ratio < 0.05 and distinct_count <= 250
    ):
        return DataType.CATEGORICAL

    return DataType.STRING


def is_compatible_type_widening(old_type: DataType, new_type: DataType) -> bool:
    """Check if a type change is non-breaking (e.g., Integer widened to Float)."""
    if old_type == new_type:
        return True
    if old_type == DataType.INTEGER and new_type == DataType.FLOAT:
        return True
    return False


def compare_schemas(
    baseline_schema: dict[str, dict[str, Any]],
    current_schema: dict[str, dict[str, Any]],
    check_order: bool = False,
) -> list[SchemaChange]:
    """Compare baseline schema with current schema to identify breaking and non-breaking changes."""
    changes: list[SchemaChange] = []
    baseline_cols = list(baseline_schema.keys())
    current_cols = list(current_schema.keys())

    # 1. Removed columns (Breaking change)
    for col in baseline_cols:
        if col not in current_schema:
            changes.append(
                SchemaChange(
                    column_name=col,
                    change_kind="removed",
                    old_value=str(baseline_schema[col].get("inferred_type")),
                    new_value=None,
                    change_type=ChangeType.BREAKING,
                    description=f"Column '{col}' was removed from the dataset. Downstream queries expecting this column will fail.",
                )
            )

    # 2. Added columns (Non-breaking change)
    for col in current_cols:
        if col not in baseline_schema:
            changes.append(
                SchemaChange(
                    column_name=col,
                    change_kind="added",
                    old_value=None,
                    new_value=str(current_schema[col].get("inferred_type")),
                    change_type=ChangeType.NON_BREAKING,
                    description=f"New column '{col}' of type {current_schema[col].get('inferred_type')} was added.",
                )
            )

    # 3. Type changes & Nullability changes on common columns
    common_cols = [c for c in baseline_cols if c in current_schema]
    for col in common_cols:
        b_info = baseline_schema[col]
        c_info = current_schema[col]

        b_type = DataType(b_info.get("inferred_type", "unknown"))
        c_type = DataType(c_info.get("inferred_type", "unknown"))

        if b_type != c_type:
            widening = is_compatible_type_widening(b_type, c_type)
            change_type = ChangeType.NON_BREAKING if widening else ChangeType.BREAKING
            severity_desc = (
                "Compatible widening" if widening else "Incompatible breaking type mutation"
            )
            changes.append(
                SchemaChange(
                    column_name=col,
                    change_kind="type_changed",
                    old_value=b_type.value,
                    new_value=c_type.value,
                    change_type=change_type,
                    description=f"{severity_desc}: Column '{col}' changed type from '{b_type.value}' to '{c_type.value}'.",
                )
            )

        # Nullability check: if baseline was non-nullable (0% nulls) and current has nulls
        b_null_pct = b_info.get("null_percentage", 0.0)
        c_null_pct = c_info.get("null_percentage", 0.0)
        if b_null_pct == 0.0 and c_null_pct > 0.0:
            changes.append(
                SchemaChange(
                    column_name=col,
                    change_kind="nullability_changed",
                    old_value="non-nullable (0.0% null)",
                    new_value=f"nullable ({c_null_pct:.2f}% null)",
                    change_type=ChangeType.BREAKING,
                    description=f"Nullability regression: Column '{col}' previously had 0 nulls, but now contains {c_null_pct:.2f}% null values.",
                )
            )

    # 4. Column order differences
    if check_order and baseline_cols != current_cols:
        if set(baseline_cols) == set(current_cols):
            changes.append(
                SchemaChange(
                    column_name="*",
                    change_kind="order_changed",
                    old_value=" -> ".join(baseline_cols),
                    new_value=" -> ".join(current_cols),
                    change_type=ChangeType.NON_BREAKING,
                    description="Column order shifted relative to baseline.",
                )
            )

    return changes
