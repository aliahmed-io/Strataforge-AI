"""Dataset loading utilities supporting CSV, JSONL, and Parquet with chunked memory streaming."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def detect_format(file_path: str | Path) -> str:
    """Infer dataset format from file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    raise ValueError(
        f"Unsupported file format '{ext}' for file {path.name}. "
        f"Supported formats: CSV (.csv), JSONL (.jsonl, .ndjson), Parquet (.parquet, .pq)."
    )


def load_dataset(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a complete dataset into memory as a pandas DataFrame."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = detect_format(path)

    try:
        if fmt == "csv":
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        elif fmt == "parquet":
            table = pq.read_table(path, memory_map=True)
            df = table.to_pandas()
            if nrows is not None:
                df = df.iloc[:nrows]
            return df
        elif fmt == "jsonl":
            return pd.read_json(path, lines=True, nrows=nrows)
        else:
            raise ValueError(f"Unhandled file format: {fmt}")
    except Exception as exc:
        raise RuntimeError(f"Error reading dataset {path}: {exc}") from exc


def load_dataset_chunks(
    file_path: str | Path,
    chunk_size: int = 50_000,
) -> Iterator[pd.DataFrame]:
    """Stream dataset in memory-efficient chunks.

    For CSV and JSONL, chunks are read iteratively.
    For Parquet, PyArrow row batches are converted to pandas DataFrames.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = detect_format(path)

    if fmt == "csv":
        reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
        for chunk in reader:
            yield chunk

    elif fmt == "parquet":
        parquet_file = pq.ParquetFile(path, memory_map=True)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()

    elif fmt == "jsonl":
        reader = pd.read_json(path, lines=True, chunksize=chunk_size)
        for chunk in reader:
            yield chunk
    else:
        raise ValueError(f"Unhandled file format: {fmt}")


def get_dataset_metadata(file_path: str | Path) -> dict[str, int | str]:
    """Retrieve fast metadata without loading entire file into memory."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = detect_format(path)
    file_size_bytes = path.stat().st_size

    if fmt == "parquet":
        meta = pq.read_metadata(path)
        return {
            "format": "parquet",
            "file_size_bytes": file_size_bytes,
            "row_count": meta.num_rows,
            "column_count": meta.num_columns,
            "column_names": meta.schema.names,
        }

    # For CSV/JSONL, read sample header
    sample = load_dataset(path, nrows=5)
    return {
        "format": fmt,
        "file_size_bytes": file_size_bytes,
        "column_count": len(sample.columns),
        "column_names": list(sample.columns),
    }
