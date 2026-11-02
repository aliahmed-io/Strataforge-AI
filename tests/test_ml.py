"""Unit tests for multivariate Isolation Forest anomaly detection and diagnostics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from strataforge.ml.anomaly import StrataforgeAnomalyModel
from strataforge.ml.explainer import explain_anomaly_row
from strataforge.ml.storage import create_baseline_snapshot, load_baseline_snapshot


def test_anomaly_model_fit_score(baseline_dataframe: pd.DataFrame):
    model = StrataforgeAnomalyModel(contamination=0.05, random_state=42)
    model.fit(baseline_dataframe)

    assert model.is_fitted is True
    assert len(model.features) > 0
    assert "amount" in model.features

    is_anom, scores = model.score(baseline_dataframe)
    assert len(scores) == len(baseline_dataframe)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_anomaly_model_flagging_outliers(baseline_dataframe: pd.DataFrame):
    model = StrataforgeAnomalyModel(contamination=0.05, random_state=42).fit(baseline_dataframe)

    # Create synthetic extreme outlier
    outlier_df = pd.DataFrame(
        [
            {
                "transaction_id": "TX-999999",
                "user_id": 9999,
                "amount": 999999.00,  # extreme amount
                "customer_age": 120,
                "risk_score": 0.9999,
            }
        ]
    )

    is_anom, scores = model.score(outlier_df)
    assert scores[0] >= 0.85  # Should score as high anomaly

    # Explain record
    rec = explain_anomaly_row(model, outlier_df, row_identifier=0)
    assert rec.anomaly_score >= 0.85
    assert rec.top_features[0].feature == "amount"
    assert rec.top_features[0].z_score > 10.0


def test_anomaly_model_serialization(baseline_dataframe: pd.DataFrame):
    model = StrataforgeAnomalyModel(contamination=0.05, random_state=42).fit(baseline_dataframe)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        loaded = StrataforgeAnomalyModel.load(tmp_path)

        assert loaded.is_fitted is True
        assert loaded.features == model.features

        _, scores_orig = model.score(baseline_dataframe.head(10))
        _, scores_load = loaded.score(baseline_dataframe.head(10))

        assert (scores_orig == scores_load).all()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_create_and_load_baseline_snapshot(baseline_dataframe: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_file = Path(tmp_dir) / "baseline.json"
        snapshot = create_baseline_snapshot(
            dataset=baseline_dataframe,
            output_path=base_file,
            fit_model=True,
        )

        assert base_file.exists()
        assert snapshot.model_artifact_path is not None
        assert (Path(tmp_dir) / snapshot.model_artifact_path).exists()

        loaded_snap, loaded_model = load_baseline_snapshot(base_file)
        assert loaded_snap.row_count == len(baseline_dataframe)
        assert loaded_model is not None
        assert loaded_model.is_fitted is True
