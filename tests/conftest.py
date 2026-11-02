"""Shared pytest fixtures for Strataforge test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from strataforge.demo.generator import generate_baseline_dataset, generate_drifted_dataset


@pytest.fixture
def baseline_dataframe() -> pd.DataFrame:
    """Provide clean baseline DataFrame."""
    return generate_baseline_dataset(n_rows=500, seed=42)


@pytest.fixture
def drifted_dataframe(baseline_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Provide modified current DataFrame with intentional flaws."""
    return generate_drifted_dataset(baseline_dataframe, seed=1337)


@pytest.fixture
def temp_dataset_dir(baseline_dataframe: pd.DataFrame, drifted_dataframe: pd.DataFrame) -> Path:
    """Create temporary directory containing CSV, JSONL, and Parquet demo files."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        # CSV
        baseline_dataframe.to_csv(tmp_path / "baseline.csv", index=False)
        drifted_dataframe.to_csv(tmp_path / "current.csv", index=False)

        # Parquet
        baseline_dataframe.to_parquet(tmp_path / "baseline.parquet", index=False)
        drifted_pq = drifted_dataframe.copy()
        drifted_pq["risk_score"] = drifted_pq["risk_score"].astype(str)
        drifted_pq.to_parquet(tmp_path / "current.parquet", index=False)

        # JSONL
        baseline_dataframe.to_json(tmp_path / "baseline.jsonl", orient="records", lines=True)
        drifted_dataframe.to_json(tmp_path / "current.jsonl", orient="records", lines=True)

        yield tmp_path
