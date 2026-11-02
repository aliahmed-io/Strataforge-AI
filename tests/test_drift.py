"""Unit tests for statistical drift detection (PSI, KS test, categorical shifts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strataforge.core.types import Severity
from strataforge.drift.detector import detect_drift
from strataforge.drift.ks_test import evaluate_ks_drift
from strataforge.drift.psi import calculate_categorical_psi, calculate_numerical_psi


def test_numerical_psi_identical_distributions():
    rng = np.random.default_rng(42)
    b = rng.normal(loc=100.0, scale=15.0, size=1000)
    c = b.copy()

    psi_val, details = calculate_numerical_psi(b, c)
    assert psi_val < 0.01  # Should be near zero
    assert len(details) > 0


def test_numerical_psi_massive_drift():
    rng = np.random.default_rng(42)
    b = rng.normal(loc=100.0, scale=15.0, size=1000)
    c = rng.normal(loc=250.0, scale=35.0, size=1000)

    psi_val, details = calculate_numerical_psi(b, c)
    assert psi_val >= 0.25  # Significant drift


def test_categorical_psi_with_new_category():
    b_series = pd.Series(["A", "B", "C"] * 100)
    c_series = pd.Series(["A", "B", "NEW_CATEGORY"] * 100)

    psi_val, details, new_cats = calculate_categorical_psi(b_series, c_series)
    assert "NEW_CATEGORY" in new_cats
    assert psi_val > 0.10


def test_ks_test_drift_detection():
    rng = np.random.default_rng(42)
    b = rng.normal(10, 2, 500)
    c = rng.normal(15, 2, 500)

    res = evaluate_ks_drift(b, c, p_value_alpha=0.01)
    assert res["significant"] is True
    assert res["p_value"] < 0.01
    assert res["effect_size"] in ("moderate", "large")


def test_ks_test_small_sample_safeguard():
    b = np.array([1, 2, 3, 4, 5, 6, 7])
    c = np.array([2, 3, 4, 5, 6, 7, 8])

    res = evaluate_ks_drift(b, c)
    assert res["notes"] is not None
    assert "Small sample size" in res["notes"]


def test_detect_drift_pipeline(baseline_dataframe: pd.DataFrame, drifted_dataframe: pd.DataFrame):
    res = detect_drift(baseline_dataframe, drifted_dataframe)

    assert res.has_critical_drift is True
    assert res.max_psi >= 0.25
    assert res.drifted_column_count > 0

    # Ensure amount and customer_age have drift metrics
    amount_metrics = [
        m for m in res.metrics if m.column_name == "amount" and m.metric_name == "psi"
    ]
    assert len(amount_metrics) == 1
    assert amount_metrics[0].severity == Severity.CRITICAL
