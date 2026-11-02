"""Comprehensive dataset drift detection engine."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from strataforge.core.config import DriftConfig
from strataforge.core.types import DataType, DriftMetric, Severity
from strataforge.drift.ks_test import evaluate_ks_drift
from strataforge.drift.psi import calculate_categorical_psi, calculate_numerical_psi


class DriftAnalysisResult(BaseModel):
    """Aggregate result of drift detection across all evaluated columns."""

    metrics: list[DriftMetric] = Field(default_factory=list)
    has_critical_drift: bool = False
    has_warning_drift: bool = False
    max_psi: float = 0.0
    drifted_column_count: int = 0
    summary_description: str = ""


def detect_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: DriftConfig | None = None,
    column_types: dict[str, DataType] | None = None,
) -> DriftAnalysisResult:
    """Analyze all shared columns between baseline and current DataFrames for distribution drift."""
    if config is None:
        config = DriftConfig()

    shared_columns = [c for c in baseline_df.columns if c in current_df.columns]
    metrics: list[DriftMetric] = []
    max_psi = 0.0
    has_critical = False
    has_warning = False
    drifted_cols = set()

    for col in shared_columns:
        b_series = baseline_df[col]
        c_series = current_df[col]

        # Determine feature type
        if column_types and col in column_types:
            ftype = column_types[col]
        else:
            if pd.api.types.is_numeric_dtype(b_series) and pd.api.types.is_numeric_dtype(c_series):
                ftype = DataType.FLOAT if "float" in str(b_series.dtype) else DataType.INTEGER
            elif pd.api.types.is_bool_dtype(b_series):
                ftype = DataType.BOOLEAN
            else:
                ftype = DataType.CATEGORICAL

        # 1. Missing value rate delta check
        b_null_pct = float(b_series.isna().mean() * 100.0)
        c_null_pct = float(c_series.isna().mean() * 100.0)
        delta_null = round(c_null_pct - b_null_pct, 4)

        if delta_null > config.max_missing_delta_pct:
            severity = Severity.CRITICAL
            has_critical = True
            drifted_cols.add(col)
        elif delta_null > (config.max_missing_delta_pct / 2.0):
            severity = Severity.WARNING
            has_warning = True
            drifted_cols.add(col)
        else:
            severity = Severity.PASS

        metrics.append(
            DriftMetric(
                column_name=col,
                feature_type=ftype,
                metric_name="missing_rate_delta",
                metric_display_name="Missing Value Rate Delta",
                stat_value=round(delta_null, 2),
                threshold_warning=round(config.max_missing_delta_pct / 2.0, 2),
                threshold_critical=config.max_missing_delta_pct,
                severity=severity,
                description=(
                    f"Null rate shifted from {b_null_pct:.1f}% to {c_null_pct:.1f}% "
                    f"(Δ {delta_null:+.1f}%)."
                ),
                sample_size_current=len(c_series),
                sample_size_baseline=len(b_series),
            )
        )

        # 2. Numerical drift (PSI and KS test)
        if ftype in (DataType.INTEGER, DataType.FLOAT):
            b_clean = pd.to_numeric(b_series.dropna(), errors="coerce").dropna()
            c_clean = pd.to_numeric(c_series.dropna(), errors="coerce").dropna()

            if len(b_clean) >= 10 and len(c_clean) >= 10:
                # PSI
                psi_val, _ = calculate_numerical_psi(b_clean, c_clean)
                if psi_val > max_psi:
                    max_psi = psi_val

                if psi_val >= config.psi_critical:
                    psi_severity = Severity.CRITICAL
                    has_critical = True
                    drifted_cols.add(col)
                elif psi_val >= config.psi_warning:
                    psi_severity = Severity.WARNING
                    has_warning = True
                    drifted_cols.add(col)
                else:
                    psi_severity = Severity.PASS

                metrics.append(
                    DriftMetric(
                        column_name=col,
                        feature_type=ftype,
                        metric_name="psi",
                        metric_display_name="Population Stability Index",
                        stat_value=psi_val,
                        threshold_warning=config.psi_warning,
                        threshold_critical=config.psi_critical,
                        severity=psi_severity,
                        description=(
                            f"PSI is {psi_val:.4f} ("
                            + (
                                "Significant distribution shift"
                                if psi_val >= config.psi_critical
                                else "Moderate shift"
                                if psi_val >= config.psi_warning
                                else "Distribution stable"
                            )
                            + ")."
                        ),
                        sample_size_current=len(c_clean),
                        sample_size_baseline=len(b_clean),
                    )
                )

                # KS test
                ks_res = evaluate_ks_drift(
                    b_clean, c_clean, p_value_alpha=config.ks_p_value_threshold
                )
                d_stat = float(ks_res["statistic"])
                p_val = float(ks_res["p_value"])
                sig = bool(ks_res["significant"])

                ks_sev = Severity.WARNING if sig else Severity.PASS
                if sig:
                    has_warning = True
                    drifted_cols.add(col)

                metrics.append(
                    DriftMetric(
                        column_name=col,
                        feature_type=ftype,
                        metric_name="ks",
                        metric_display_name="Kolmogorov-Smirnov Test",
                        stat_value=d_stat,
                        p_value=p_val,
                        threshold_warning=0.10,
                        threshold_critical=0.25,
                        severity=ks_sev,
                        description=f"KS statistic D={d_stat:.4f}, p-value={p_val:.2e} (Effect: {ks_res['effect_size']}).",
                        sample_size_current=len(c_clean),
                        sample_size_baseline=len(b_clean),
                        notes=str(ks_res["notes"]) if ks_res["notes"] else None,
                    )
                )

        # 3. Categorical drift (Categorical PSI & Unseen Categories)
        elif ftype in (DataType.CATEGORICAL, DataType.BOOLEAN, DataType.STRING):
            b_str = b_series.dropna().astype(str)
            c_str = c_series.dropna().astype(str)

            if len(b_str) >= 10 and len(c_str) >= 10:
                cat_psi, _, new_cats = calculate_categorical_psi(b_str, c_str)
                if cat_psi > max_psi:
                    max_psi = cat_psi

                if cat_psi >= config.psi_critical:
                    cat_sev = Severity.CRITICAL
                    has_critical = True
                    drifted_cols.add(col)
                elif cat_psi >= config.psi_warning or new_cats:
                    cat_sev = Severity.WARNING
                    has_warning = True
                    drifted_cols.add(col)
                else:
                    cat_sev = Severity.PASS

                desc_parts = [f"Categorical PSI={cat_psi:.4f}"]
                if new_cats:
                    desc_parts.append(
                        f"Discovered {len(new_cats)} new category value(s): {new_cats[:3]}"
                    )

                metrics.append(
                    DriftMetric(
                        column_name=col,
                        feature_type=ftype,
                        metric_name="category_drift",
                        metric_display_name="Categorical Distribution Shift",
                        stat_value=cat_psi,
                        threshold_warning=config.psi_warning,
                        threshold_critical=config.psi_critical,
                        severity=cat_sev,
                        description=". ".join(desc_parts),
                        sample_size_current=len(c_str),
                        sample_size_baseline=len(b_str),
                    )
                )

    summary = (
        f"Evaluated {len(shared_columns)} shared feature(s). "
        f"{len(drifted_cols)} column(s) exhibited noticeable drift. "
        f"Maximum recorded PSI across features: {max_psi:.4f}."
    )

    return DriftAnalysisResult(
        metrics=metrics,
        has_critical_drift=has_critical,
        has_warning_drift=has_warning,
        max_psi=round(max_psi, 4),
        drifted_column_count=len(drifted_cols),
        summary_description=summary,
    )
