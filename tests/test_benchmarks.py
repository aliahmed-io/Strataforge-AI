"""Performance and latency benchmarks for Strataforge AI.

Ensures profiling, statistical drift, and ML anomaly detection execute within SLAs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strataforge.drift.detector import detect_drift
from strataforge.ml.anomaly import StrataforgeAnomalyModel
from strataforge.profiling.profiler import profile_dataset


@pytest.fixture
def synthetic_benchmark_df() -> pd.DataFrame:
    """Generate reproducible 10,000-row dataset for benchmark suites."""
    np.random.seed(42)
    n = 10_000
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "amount": np.random.exponential(scale=50.0, size=n),
            "customer_age": np.random.randint(18, 85, size=n),
            "score": np.random.normal(loc=0.5, scale=0.15, size=n),
            "category": np.random.choice(["retail", "travel", "grocery", "tech"], size=n),
            "is_flagged": np.random.choice([True, False], size=n),
        }
    )


def test_benchmark_profiling_10k_rows(benchmark, synthetic_benchmark_df: pd.DataFrame) -> None:
    """Profile 10,000 rows across 6 mixed-type columns."""
    result = benchmark(profile_dataset, synthetic_benchmark_df, "benchmark_10k.csv")
    assert result.row_count == 10_000
    assert result.column_count == 6


def test_benchmark_drift_detection_10k_rows(
    benchmark, synthetic_benchmark_df: pd.DataFrame
) -> None:
    """Evaluate full statistical drift pipeline on 10,000 baseline vs current records."""
    df_curr = synthetic_benchmark_df.copy()
    df_curr["amount"] = df_curr["amount"] * 1.05  # slight drift

    result = benchmark(detect_drift, synthetic_benchmark_df, df_curr)
    assert len(result.metrics) > 0


def test_benchmark_isolation_forest_scoring(
    benchmark, synthetic_benchmark_df: pd.DataFrame
) -> None:
    """Benchmark multivariate anomaly scoring on 10,000 records."""
    model = StrataforgeAnomalyModel(contamination=0.01, random_state=42)
    model.fit(synthetic_benchmark_df)

    is_anom, scores = benchmark(model.score, synthetic_benchmark_df)
    assert len(scores) == 10_000
