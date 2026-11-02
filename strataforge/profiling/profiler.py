"""Comprehensive column and dataset profiler with chunked streaming capabilities."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strataforge.core.loader import detect_format, load_dataset, load_dataset_chunks
from strataforge.core.schema import infer_series_type
from strataforge.core.types import ColumnProfile, DatasetProfile, DataType, Quantiles
from strataforge.profiling.duplicates import detect_duplicates


def profile_column(series: pd.Series, inferred_type: DataType | None = None) -> ColumnProfile:
    """Compute comprehensive statistical profile for a single pandas Series."""
    total_count = len(series)
    null_count = int(series.isna().sum())
    null_percentage = (null_count / total_count * 100.0) if total_count > 0 else 0.0

    non_null = series.dropna()
    valid_count = len(non_null)

    if inferred_type is None:
        inferred_type = infer_series_type(series)

    distinct_count = int(non_null.nunique()) if valid_count > 0 else 0
    distinct_percentage = (distinct_count / valid_count * 100.0) if valid_count > 0 else 0.0
    is_unique = distinct_count == total_count and null_count == 0

    sample_values: list[Any] = []
    if valid_count > 0:
        sample_raw = non_null.drop_duplicates().head(3).tolist()
        sample_values = [
            val.isoformat()
            if isinstance(val, pd.Timestamp)
            else (
                int(val)
                if isinstance(val, (np.integer,))
                else (float(val) if isinstance(val, (np.floating,)) else str(val))
            )
            for val in sample_raw
        ]

    min_val: float | str | None = None
    max_val: float | str | None = None
    mean_val: float | None = None
    std_val: float | None = None
    quantiles: Quantiles | None = None
    top_categories: dict[str, int] = {}
    category_frequencies: dict[str, float] = {}

    if inferred_type in (DataType.INTEGER, DataType.FLOAT) and valid_count > 0:
        numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric_series) > 0:
            min_val = float(numeric_series.min())
            max_val = float(numeric_series.max())
            mean_val = float(numeric_series.mean())
            std_val = float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0

            q_values = np.nanpercentile(numeric_series, [5, 25, 50, 75, 95])
            quantiles = Quantiles(
                p5=float(q_values[0]),
                p25=float(q_values[1]),
                p50=float(q_values[2]),
                p75=float(q_values[3]),
                p95=float(q_values[4]),
            )

    elif (
        inferred_type in (DataType.CATEGORICAL, DataType.BOOLEAN, DataType.STRING)
        and valid_count > 0
    ):
        val_counts = non_null.astype(str).value_counts()
        top_slice = val_counts.head(10)
        top_categories = {str(k): int(v) for k, v in top_slice.items()}
        category_frequencies = {str(k): float(v / valid_count) for k, v in top_slice.items()}

    elif inferred_type == DataType.DATETIME and valid_count > 0:
        try:
            dt_series = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dt_series) > 0:
                min_val = dt_series.min().isoformat()
                max_val = dt_series.max().isoformat()
        except Exception:
            pass

    return ColumnProfile(
        name=str(series.name),
        inferred_type=inferred_type,
        native_dtype=str(series.dtype),
        total_count=total_count,
        null_count=null_count,
        null_percentage=round(null_percentage, 4),
        distinct_count=distinct_count,
        distinct_percentage=round(distinct_percentage, 4),
        is_unique=is_unique,
        is_nullable=null_count > 0,
        sample_values=sample_values,
        min_val=min_val,
        max_val=max_val,
        mean_val=round(mean_val, 4) if mean_val is not None else None,
        std_val=round(std_val, 4) if std_val is not None else None,
        quantiles=quantiles,
        top_categories=top_categories,
        category_frequencies=category_frequencies,
    )


def profile_dataset(
    dataset_input: str | Path | pd.DataFrame,
    chunk_size: int | None = None,
) -> DatasetProfile:
    """Generate full dataset profile.

    If a path is provided and chunk_size is set or the file is very large CSV,
    profile via streaming chunks to conserve memory.
    """
    now_str = datetime.now(UTC).isoformat()

    if isinstance(dataset_input, pd.DataFrame):
        df = dataset_input
        dup_rep = detect_duplicates(df)
        cols_profile: dict[str, ColumnProfile] = {}
        for col in df.columns:
            cols_profile[col] = profile_column(df[col])

        return DatasetProfile(
            dataset_path="in-memory",
            file_format="dataframe",
            row_count=len(df),
            column_count=len(df.columns),
            memory_bytes=int(df.memory_usage(deep=True).sum()),
            duplicate_count=dup_rep.duplicate_rows_count,
            duplicate_percentage=dup_rep.duplicate_percentage,
            columns=cols_profile,
            created_at=now_str,
        )

    path = Path(dataset_input)
    fmt = detect_format(path)

    # If CSV/JSONL is larger than 150MB and chunk_size is requested, use streaming
    if chunk_size is not None and fmt in ("csv", "jsonl"):
        return _profile_dataset_chunked(path, chunk_size, now_str)

    # Standard fast in-memory load
    df = load_dataset(path)
    dup_rep = detect_duplicates(df)
    cols_profile = {}
    for col in df.columns:
        cols_profile[col] = profile_column(df[col])

    return DatasetProfile(
        dataset_path=str(path),
        file_format=fmt,
        row_count=len(df),
        column_count=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        duplicate_count=dup_rep.duplicate_rows_count,
        duplicate_percentage=dup_rep.duplicate_percentage,
        columns=cols_profile,
        created_at=now_str,
    )


def _profile_dataset_chunked(
    path: Path,
    chunk_size: int,
    created_at: str,
) -> DatasetProfile:
    """Memory-efficient streaming profiler for large datasets."""
    total_rows = 0
    col_nulls: dict[str, int] = {}
    col_types: dict[str, DataType] = {}
    col_native_dtypes: dict[str, str] = {}
    col_numeric_sums: dict[str, float] = {}
    col_numeric_sq_sums: dict[str, float] = {}
    col_numeric_mins: dict[str, float] = {}
    col_numeric_maxs: dict[str, float] = {}
    col_numeric_counts: dict[str, int] = {}
    col_category_counts: dict[str, dict[str, int]] = {}

    first_chunk = True
    columns: list[str] = []

    for chunk in load_dataset_chunks(path, chunk_size=chunk_size):
        chunk_len = len(chunk)
        total_rows += chunk_len

        if first_chunk:
            columns = list(chunk.columns)
            for c in columns:
                col_nulls[c] = 0
                col_native_dtypes[c] = str(chunk[c].dtype)
                inferred = infer_series_type(chunk[c])
                col_types[c] = inferred
                if inferred in (DataType.INTEGER, DataType.FLOAT):
                    col_numeric_sums[c] = 0.0
                    col_numeric_sq_sums[c] = 0.0
                    col_numeric_mins[c] = float("inf")
                    col_numeric_maxs[c] = float("-inf")
                    col_numeric_counts[c] = 0
                elif inferred in (DataType.CATEGORICAL, DataType.BOOLEAN, DataType.STRING):
                    col_category_counts[c] = {}
            first_chunk = False

        for c in columns:
            series = chunk[c]
            nulls = int(series.isna().sum())
            col_nulls[c] += nulls

            if c in col_numeric_sums:
                valid_num = pd.to_numeric(series.dropna(), errors="coerce").dropna()
                if len(valid_num) > 0:
                    cnt = len(valid_num)
                    col_numeric_counts[c] += cnt
                    s = float(valid_num.sum())
                    sq = float((valid_num**2).sum())
                    col_numeric_sums[c] += s
                    col_numeric_sq_sums[c] += sq
                    c_min = float(valid_num.min())
                    c_max = float(valid_num.max())
                    if c_min < col_numeric_mins[c]:
                        col_numeric_mins[c] = c_min
                    if c_max > col_numeric_maxs[c]:
                        col_numeric_maxs[c] = c_max

            elif c in col_category_counts:
                top_v = series.dropna().astype(str).value_counts().head(20)
                cat_dict = col_category_counts[c]
                for k, v in top_v.items():
                    cat_dict[k] = cat_dict.get(k, 0) + int(v)

    # Build ColumnProfile dictionary
    col_profiles: dict[str, ColumnProfile] = {}
    for c in columns:
        null_count = col_nulls[c]
        null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0
        inferred = col_types[c]

        min_val = None
        max_val = None
        mean_val = None
        std_val = None
        top_cats = {}
        freqs = {}

        if c in col_numeric_sums and col_numeric_counts[c] > 0:
            n = col_numeric_counts[c]
            mean_val = col_numeric_sums[c] / n
            variance = max(0.0, (col_numeric_sq_sums[c] / n) - (mean_val**2))
            std_val = float(np.sqrt(variance))
            min_val = col_numeric_mins[c]
            max_val = col_numeric_maxs[c]

        elif c in col_category_counts:
            sorted_cats = sorted(col_category_counts[c].items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            top_cats = dict(sorted_cats)
            valid_cnt = max(1, total_rows - null_count)
            freqs = {k: round(v / valid_cnt, 4) for k, v in top_cats.items()}

        col_profiles[c] = ColumnProfile(
            name=c,
            inferred_type=inferred,
            native_dtype=col_native_dtypes[c],
            total_count=total_rows,
            null_count=null_count,
            null_percentage=round(null_pct, 4),
            distinct_count=len(top_cats) if top_cats else 0,
            distinct_percentage=0.0,
            is_unique=False,
            is_nullable=null_count > 0,
            sample_values=[],
            min_val=min_val,
            max_val=max_val,
            mean_val=round(mean_val, 4) if mean_val is not None else None,
            std_val=round(std_val, 4) if std_val is not None else None,
            quantiles=None,
            top_categories=top_cats,
            category_frequencies=freqs,
        )

    return DatasetProfile(
        dataset_path=str(path),
        file_format=detect_format(path),
        row_count=total_rows,
        column_count=len(columns),
        memory_bytes=path.stat().st_size,
        duplicate_count=0,
        duplicate_percentage=0.0,
        columns=col_profiles,
        created_at=created_at,
    )
