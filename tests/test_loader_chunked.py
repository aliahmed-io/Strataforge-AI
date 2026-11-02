"""Unit tests for multi-format loading and chunk streaming."""

from __future__ import annotations

from pathlib import Path

import pytest

from strataforge.core.loader import (
    detect_format,
    get_dataset_metadata,
    load_dataset,
    load_dataset_chunks,
)


def test_detect_format():
    assert detect_format("file.csv") == "csv"
    assert detect_format("file.parquet") == "parquet"
    assert detect_format("file.pq") == "parquet"
    assert detect_format("file.jsonl") == "jsonl"
    assert detect_format("file.ndjson") == "jsonl"

    with pytest.raises(ValueError):
        detect_format("file.xlsx")


def test_load_all_supported_formats(temp_dataset_dir: Path):
    csv_df = load_dataset(temp_dataset_dir / "baseline.csv")
    parquet_df = load_dataset(temp_dataset_dir / "baseline.parquet")
    jsonl_df = load_dataset(temp_dataset_dir / "baseline.jsonl")

    assert len(csv_df) == 500
    assert len(parquet_df) == 500
    assert len(jsonl_df) == 500

    assert list(csv_df.columns) == list(parquet_df.columns)


def test_load_dataset_chunks_csv(temp_dataset_dir: Path):
    csv_file = temp_dataset_dir / "baseline.csv"
    chunks = list(load_dataset_chunks(csv_file, chunk_size=150))

    assert len(chunks) == 4  # 150 + 150 + 150 + 50 = 500
    total_chunk_rows = sum(len(c) for c in chunks)
    assert total_chunk_rows == 500


def test_load_dataset_chunks_parquet(temp_dataset_dir: Path):
    pq_file = temp_dataset_dir / "baseline.parquet"
    chunks = list(load_dataset_chunks(pq_file, chunk_size=200))
    total_chunk_rows = sum(len(c) for c in chunks)
    assert total_chunk_rows == 500


def test_dataset_metadata_fast(temp_dataset_dir: Path):
    meta_pq = get_dataset_metadata(temp_dataset_dir / "baseline.parquet")
    assert meta_pq["format"] == "parquet"
    assert meta_pq["row_count"] == 500
    assert meta_pq["column_count"] == 8

    meta_csv = get_dataset_metadata(temp_dataset_dir / "baseline.csv")
    assert meta_csv["format"] == "csv"
    assert meta_csv["column_count"] == 8
