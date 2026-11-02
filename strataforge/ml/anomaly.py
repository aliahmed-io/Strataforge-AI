"""Multivariate anomaly detection using scikit-learn Isolation Forest."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from strataforge.core.types import AnomalyRecord, FeatureImpact


class StrataforgeAnomalyModel:
    """Production-grade multivariate anomaly detector with fitted preprocessing and persistence."""

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        features: list[str] | None = None,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.features: list[str] = features or []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_stats_: dict[str, dict[str, float]] = {}

    def select_numerical_features(self, df: pd.DataFrame) -> list[str]:
        """Automatically select continuous numerical features suitable for multivariate modeling."""
        suitable = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Exclude boolean flags or high-cardinality ID columns
                n_unique = df[col].nunique()
                if n_unique > 5:
                    suitable.append(col)
        return suitable

    def fit(self, df: pd.DataFrame) -> StrataforgeAnomalyModel:
        """Fit preprocessing pipeline and Isolation Forest model on reference baseline dataset."""
        if not self.features:
            self.features = self.select_numerical_features(df)

        if not self.features:
            raise ValueError("No suitable numerical columns available for anomaly detection.")

        sub_df = df[self.features].copy()
        for col in self.features:
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")

        # Compute baseline reference statistics for explanation
        self.feature_stats_ = {}
        for col in self.features:
            valid_col = sub_df[col].dropna()
            if len(valid_col) > 0:
                self.feature_stats_[col] = {
                    "mean": float(valid_col.mean()),
                    "std": float(valid_col.std(ddof=1)) if len(valid_col) > 1 else 1.0,
                    "median": float(valid_col.median()),
                    "q25": float(valid_col.quantile(0.25)),
                    "q75": float(valid_col.quantile(0.75)),
                }
            else:
                self.feature_stats_[col] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "median": 0.0,
                    "q25": 0.0,
                    "q75": 0.0,
                }

        # Fit preprocessors only on reference baseline data
        raw_values = sub_df.values
        imputed_values = self.imputer.fit_transform(raw_values)
        scaled_values = self.scaler.fit_transform(imputed_values)

        # Fit Isolation Forest
        self.model.fit(scaled_values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess features using fitted imputer and scaler."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly model is not fitted.")
        sub_df = df[self.features].copy()
        for col in self.features:
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
        imputed = self.imputer.transform(sub_df.values)
        return self.scaler.transform(imputed)

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform observations, compute raw decision scores and normalized anomaly scores.

        Returns:
            tuple of (is_anomaly_mask, normalized_scores_0_to_1)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Anomaly model is not fitted. Call fit() or load a baseline model first."
            )

        scaled = self.transform(df)

        # Raw decision function: lower scores indicate higher anomaly severity
        raw_scores = self.model.decision_function(scaled)
        predictions = self.model.predict(scaled)  # -1 for anomalies, 1 for inliers

        is_anomaly = predictions == -1

        # Normalize score: 0.0 (very normal) to 1.0 (extreme anomaly)
        # Using sigmoid-like transformation of negative decision function
        min_s = float(np.min(raw_scores))
        max_s = float(np.max(raw_scores))
        if max_s > min_s:
            norm_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
        else:
            norm_scores = np.where(is_anomaly, 1.0, 0.0)

        return is_anomaly, np.round(norm_scores, 4)

    def explain_record(
        self,
        row_series: pd.Series,
        row_idx: int,
        record_id: Any,
        anomaly_score: float,
        raw_score: float,
        is_anomaly: bool,
    ) -> AnomalyRecord:
        """Deconstruct an observation's feature deviations against baseline reference."""
        impacts: list[FeatureImpact] = []

        for col in self.features:
            val = row_series.get(col)
            stats = self.feature_stats_.get(col, {"mean": 0.0, "std": 1.0, "median": 0.0})
            mean = stats["mean"]
            std = stats["std"] if stats["std"] > 1e-6 else 1.0
            median = stats["median"]

            try:
                num_val = float(val) if pd.notna(val) else None
            except (ValueError, TypeError):
                num_val = None

            if num_val is not None:
                z_score = round((num_val - mean) / std, 2)
                deviation = round(abs(num_val - median), 2)
                impact_weight = abs(z_score)
                expl = (
                    f"Value {num_val:g} is {abs(z_score):.1f} std deviations "
                    f"{'above' if z_score > 0 else 'below'} baseline mean ({mean:.2f})."
                )
            else:
                z_score = None
                deviation = None
                impact_weight = 2.0  # missing value impact penalty
                expl = "Missing or null value on modeled feature."

            impacts.append(
                FeatureImpact(
                    feature=col,
                    value=val,
                    baseline_mean=round(mean, 2),
                    baseline_std=round(std, 2),
                    baseline_median=round(median, 2),
                    z_score=z_score,
                    relative_deviation=deviation,
                    contribution_score=round(impact_weight, 2),
                    explanation=expl,
                )
            )

        # Sort features by contribution score descending
        impacts.sort(key=lambda x: x.contribution_score, reverse=True)
        top_deviations = [
            imp.feature for imp in impacts[:2] if (imp.z_score and abs(imp.z_score) >= 2.0)
        ]

        if top_deviations:
            diag = (
                f"Flagged primarily due to anomalous shifts in {', '.join(top_deviations)} "
                f"(normalized anomaly index: {anomaly_score:.2f})."
            )
        else:
            diag = (
                f"Flagged due to multivariate combination of subtle feature correlations "
                f"(normalized anomaly index: {anomaly_score:.2f})."
            )

        row_dict = {k: (None if pd.isna(v) else v) for k, v in row_series.to_dict().items()}

        return AnomalyRecord(
            row_index=row_idx,
            record_id=str(record_id),
            anomaly_score=anomaly_score,
            raw_decision_score=round(float(raw_score), 4),
            is_anomaly=is_anomaly,
            top_features=impacts[:5],
            row_values=row_dict,
            diagnostic_summary=diag,
        )

    def save(self, filepath: str) -> None:
        """Serialize model bundle to disk."""
        bundle = {
            "version": "1.0",
            "contamination": self.contamination,
            "random_state": self.random_state,
            "features": self.features,
            "feature_stats": self.feature_stats_,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> StrataforgeAnomalyModel:
        """Load serialized model bundle."""
        bundle = joblib.load(filepath)
        instance = cls(
            contamination=bundle["contamination"],
            random_state=bundle["random_state"],
            features=bundle["features"],
        )
        instance.feature_stats_ = bundle["feature_stats"]
        instance.imputer = bundle["imputer"]
        instance.scaler = bundle["scaler"]
        instance.model = bundle["model"]
        instance.is_fitted = bundle["is_fitted"]
        return instance
