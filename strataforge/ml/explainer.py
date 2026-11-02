"""Diagnostic explanation engine for multivariate anomaly detection."""

from __future__ import annotations

import pandas as pd

from strataforge.core.types import AnomalyRecord
from strataforge.ml.anomaly import StrataforgeAnomalyModel

DISCLAIMER_TEXT = (
    "STATISTICAL DISCLAIMER: An anomaly indicates that this record deviates significantly "
    "from the empirical distribution and multivariate correlation patterns of the reference data. "
    "An anomaly is not automatically an error or corrupted record; legitimate rare events, "
    "high-value customers, or novel edge cases can legitimately score as anomalies. "
    "Do not drop records solely based on statistical anomaly scores without domain context."
)


def explain_anomaly_row(
    model: StrataforgeAnomalyModel,
    df: pd.DataFrame,
    row_identifier: int | str,
    id_column: str | None = None,
) -> AnomalyRecord:
    """Explain a specific row in the dataset against the baseline reference model."""
    if not model.is_fitted:
        raise RuntimeError("Anomaly model must be fitted to explain records.")

    # Locate row
    if id_column and id_column in df.columns:
        match = df[df[id_column].astype(str) == str(row_identifier)]
        if len(match) == 0:
            raise KeyError(f"Record with {id_column}='{row_identifier}' not found in dataset.")
        row_idx = int(match.index[0])
        row_series = match.iloc[0]
        rec_id = row_identifier
    else:
        try:
            row_idx = int(row_identifier)
        except ValueError as exc:
            raise ValueError(
                f"Invalid row index '{row_identifier}'. Must be an integer or valid ID."
            ) from exc

        if row_idx < 0 or row_idx >= len(df):
            raise IndexError(f"Row index {row_idx} is out of bounds (dataset has {len(df)} rows).")
        row_series = df.iloc[row_idx]
        rec_id = row_series.get(id_column, row_idx) if id_column else row_idx

    # Evaluate row
    row_df = pd.DataFrame([row_series])
    is_anomaly, norm_scores = model.score(row_df)

    # Compute raw decision function score
    scaled = model.transform(row_df)
    raw_score = float(model.model.decision_function(scaled)[0])

    record = model.explain_record(
        row_series=row_series,
        row_idx=row_idx,
        record_id=rec_id,
        anomaly_score=float(norm_scores[0]),
        raw_score=raw_score,
        is_anomaly=bool(is_anomaly[0]),
    )

    return record


def get_anomaly_score_interpretation(anomaly_score: float) -> str:
    """Provide calibrated description of what the normalized anomaly score means."""
    if anomaly_score >= 0.85:
        return "Severe statistical outlier. Extreme departure from typical reference distribution."
    elif anomaly_score >= 0.70:
        return (
            "High anomaly degree. Substantial multivariate deviation across one or more dimensions."
        )
    elif anomaly_score >= 0.50:
        return "Moderate anomaly degree. Boundary observation exhibiting atypical feature combinations."
    else:
        return "Conforms to expected reference distributions and multivariate density."
