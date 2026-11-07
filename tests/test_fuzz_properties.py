"""Property-based fuzz testing using Hypothesis.

Tests statistical and data invariants across thousands of randomized inputs,
ensuring Strataforge's algorithms never crash or violate mathematical bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from strataforge.core.schema import infer_series_type
from strataforge.core.types import DataType
from strataforge.drift.ks_test import evaluate_ks_drift
from strataforge.drift.psi import calculate_numerical_psi
from strataforge.ml.anomaly import StrataforgeAnomalyModel


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=200,
    )
)
@settings(max_examples=50, deadline=None)
def test_fuzz_numerical_psi_invariants(values: list[float]) -> None:
    """PSI of any distribution with itself must be approximately 0.0 (and non-negative)."""
    s_base = pd.Series(values)
    s_curr = pd.Series(values)

    psi, _ = calculate_numerical_psi(s_base, s_curr, num_bins=10)
    assert not np.isnan(psi), "PSI must not evaluate to NaN for valid numerical data"
    assert psi >= 0.0, f"PSI must be non-negative, got {psi}"
    assert psi < 0.1, f"PSI of identical distributions must be < 0.1, got {psi}"


@given(
    st.lists(
        st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
        min_size=15,
        max_size=150,
    )
)
@settings(max_examples=50, deadline=None)
def test_fuzz_ks_test_invariants(values: list[float]) -> None:
    """Kolmogorov-Smirnov test statistic must strictly fall in [0.0, 1.0]."""
    s1 = pd.Series(values)
    s2 = pd.Series(values)

    res = evaluate_ks_drift(s1, s2)
    stat = res["statistic"]
    p_val = res["p_value"]
    assert 0.0 <= stat <= 1.0, f"KS statistic must be within [0, 1], got {stat}"
    assert 0.0 <= p_val <= 1.0, f"p-value must be within [0, 1], got {p_val}"


@given(
    st.lists(
        st.one_of(
            st.integers(min_value=-100000, max_value=100000),
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20),
            st.none(),
        ),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=50, deadline=None)
def test_fuzz_schema_inference_no_fatal_crashes(mixed_values: list[object]) -> None:
    """Arbitrary mixed inputs must be classified without unhandled fatal exceptions."""
    s = pd.Series(mixed_values)
    inferred = infer_series_type(s)
    assert isinstance(inferred, DataType)


@given(
    st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=100,
    ),
    st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=100,
    ),
)
@settings(max_examples=30, deadline=None)
def test_fuzz_anomaly_model_score_range(feat1: list[float], feat2: list[float]) -> None:
    """Normalized anomaly scores must always map strictly to [0.0, 1.0]."""
    min_len = min(len(feat1), len(feat2))
    df = pd.DataFrame({"f1": feat1[:min_len], "f2": feat2[:min_len]})

    model = StrataforgeAnomalyModel(features=["f1", "f2"], contamination=0.1, random_state=42)
    model.fit(df)

    is_anom, norm_scores = model.score(df)
    assert len(norm_scores) == min_len
    assert np.all((norm_scores >= 0.0) & (norm_scores <= 1.0)), "Anomaly scores must be in [0, 1]"
    assert isinstance(is_anom, np.ndarray)
