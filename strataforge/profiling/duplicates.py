"""Duplicate detection engine supporting exact row matching and composite key uniqueness."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class DuplicateReport(BaseModel):
    """Detailed summary of duplicate records found in a dataset."""

    total_rows: int
    duplicate_rows_count: int
    duplicate_percentage: float
    subset_columns: list[str] | None = None
    sample_duplicates: list[dict[str, Any]] = Field(default_factory=list)
    has_critical_duplicates: bool = False


def detect_duplicates(
    df: pd.DataFrame,
    subset: list[str] | str | None = None,
    max_samples: int = 5,
) -> DuplicateReport:
    """Analyze DataFrame for exact duplicate rows or key uniqueness violations."""
    total_rows = len(df)
    if total_rows == 0:
        return DuplicateReport(
            total_rows=0,
            duplicate_rows_count=0,
            duplicate_percentage=0.0,
            subset_columns=None,
            sample_duplicates=[],
            has_critical_duplicates=False,
        )

    cols = [subset] if isinstance(subset, str) else subset

    # Filter to existing columns if specified
    if cols is not None:
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            valid_cols = None
    else:
        valid_cols = None

    duplicate_mask = df.duplicated(subset=valid_cols, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicate_pct = (duplicate_count / total_rows) * 100.0

    sample_records: list[dict[str, Any]] = []
    if duplicate_count > 0:
        dup_df = df[duplicate_mask]
        # Sort by subset to show duplicates together
        if valid_cols:
            dup_df = dup_df.sort_values(by=valid_cols)
        sample = dup_df.head(max_samples)
        sample_records = sample.to_dict(orient="records")

    return DuplicateReport(
        total_rows=total_rows,
        duplicate_rows_count=duplicate_count,
        duplicate_percentage=round(duplicate_pct, 4),
        subset_columns=valid_cols,
        sample_duplicates=sample_records,
        has_critical_duplicates=duplicate_count > 0,
    )
