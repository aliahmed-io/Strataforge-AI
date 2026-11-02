"""Intelligent, evidence-based data quality recommendation engine."""

from __future__ import annotations

from strataforge.core.types import (
    AnomalyRecord,
    ChangeType,
    DriftMetric,
    Recommendation,
    SchemaChange,
    Severity,
    ValidationFinding,
)
from strataforge.profiling.duplicates import DuplicateReport


def generate_recommendations(
    schema_changes: list[SchemaChange] | None = None,
    drift_metrics: list[DriftMetric] | None = None,
    validation_findings: list[ValidationFinding] | None = None,
    duplicate_report: DuplicateReport | None = None,
    anomalies: list[AnomalyRecord] | None = None,
) -> list[Recommendation]:
    """Analyze all diagnostic findings and construct evidence-based remediation recommendations."""
    recommendations: list[Recommendation] = []

    # 1. Schema Change Recommendations
    if schema_changes:
        for chg in schema_changes:
            if chg.change_kind == "removed":
                recommendations.append(
                    Recommendation(
                        category="schema",
                        severity=Severity.CRITICAL,
                        title=f"Restore or Deprecate Missing Column: '{chg.column_name}'",
                        evidence=f"Column was present in baseline ({chg.old_value}) but is completely absent in current dataset.",
                        proposed_action=(
                            "If this was an intentional deprecation, update rules.yaml to remove the column requirement. "
                            "If accidental, check upstream ETL extraction scripts before continuing."
                        ),
                        code_snippet=(
                            f"# Ensure backward compatibility if reading upstream\n"
                            f"if '{chg.column_name}' not in df.columns:\n"
                            f"    df['{chg.column_name}'] = None  # Or assign explicit fallback default"
                        ),
                        impact_description="Prevents KeyError and broken schema contracts in downstream pipelines.",
                    )
                )
            elif chg.change_kind == "type_changed" and chg.change_type == ChangeType.BREAKING:
                recommendations.append(
                    Recommendation(
                        category="schema",
                        severity=Severity.CRITICAL,
                        title=f"Enforce Safe Type Casting for '{chg.column_name}'",
                        evidence=f"Column type mutated from '{chg.old_value}' to '{chg.new_value}'.",
                        proposed_action="Cast values to expected type or sanitize corrupt string values.",
                        code_snippet=(
                            f"# Cast '{chg.column_name}' safely back to expected type\n"
                            f"df['{chg.column_name}'] = pd.to_numeric(df['{chg.column_name}'], errors='coerce')"
                            if chg.old_value in ("integer", "float")
                            else f"df['{chg.column_name}'] = df['{chg.column_name}'].astype(str)"
                        ),
                        impact_description="Prevents type mismatch errors in ML features and database insertions.",
                    )
                )
            elif chg.change_kind == "nullability_changed":
                recommendations.append(
                    Recommendation(
                        category="missing_values",
                        severity=Severity.CRITICAL,
                        title=f"Address Null Infiltration in '{chg.column_name}'",
                        evidence=f"Column previously had 0 nulls; now has {chg.new_value}.",
                        proposed_action=(
                            "Inspect upstream source for missing payload keys, or apply an imputation strategy "
                            "such as forward-fill, median, or constant default."
                        ),
                        code_snippet=(
                            f"# Impute nulls with sensible default or drop incomplete rows\n"
                            f"df['{chg.column_name}'] = df['{chg.column_name}'].fillna(df['{chg.column_name}'].median())"
                        ),
                        impact_description="Restores non-null contract required by strict downstream schemas.",
                    )
                )

    # 2. Duplicate Records Recommendations
    if duplicate_report and duplicate_report.has_critical_duplicates:
        subset_cols = duplicate_report.subset_columns or "all columns"
        recommendations.append(
            Recommendation(
                category="duplicates",
                severity=Severity.CRITICAL,
                title=f"Deduplicate Records on Primary Key / Columns ({subset_cols})",
                evidence=(
                    f"Found {duplicate_report.duplicate_rows_count} duplicate rows "
                    f"({duplicate_report.duplicate_percentage:.2f}% of dataset)."
                ),
                proposed_action=(
                    "Deduplicate dataset preserving the latest record (keep='last') or earliest record (keep='first'), "
                    "and review upstream message broker idempotency."
                ),
                code_snippet=(
                    f"# Deduplicate keeping latest record\n"
                    f"df = df.drop_duplicates(subset={duplicate_report.subset_columns}, keep='last')"
                ),
                impact_description=f"Eliminates {duplicate_report.duplicate_rows_count} redundant records preventing distorted aggregates.",
            )
        )

    # 3. Statistical Drift Recommendations
    if drift_metrics:
        for m in drift_metrics:
            if m.severity in (Severity.CRITICAL, Severity.WARNING):
                if m.metric_name == "psi":
                    recommendations.append(
                        Recommendation(
                            category="drift",
                            severity=m.severity,
                            title=f"Calibrate Distribution Drift in Feature '{m.column_name}'",
                            evidence=f"PSI reached {m.stat_value:.4f} (threshold: {m.threshold_warning:.2f}).",
                            proposed_action=(
                                "Inspect underlying customer demographics or macro events. "
                                "If this represents a permanent organic shift, retrain downstream ML models "
                                "and generate an updated Strataforge baseline (`strataforge baseline PATH --out baseline.json`)."
                            ),
                            code_snippet=(
                                f"# Retrain baseline model or normalize feature values\n"
                                f"# strataforge baseline {m.column_name}_new.csv --out baseline.json"
                            ),
                            impact_description="Prevents silent model degradation and statistical skew in decision systems.",
                        )
                    )
                elif m.metric_name == "category_drift":
                    recommendations.append(
                        Recommendation(
                            category="drift",
                            severity=m.severity,
                            title=f"Review New or Drifting Categories in '{m.column_name}'",
                            evidence=m.description,
                            proposed_action="Validate whether new category values are legitimate business expansions or typos.",
                            code_snippet=(
                                f"# Map unexpected category outliers to '__other__' or canonical forms\n"
                                f"valid_categories = ['retail', 'grocery', 'travel']\n"
                                f"df['{m.column_name}'] = df['{m.column_name}'].apply(lambda x: x if x in valid_categories else 'other')"
                            ),
                            impact_description="Prevents unseen category errors in one-hot encoders and strict categorical pipelines.",
                        )
                    )
                elif m.metric_name == "missing_rate_delta":
                    recommendations.append(
                        Recommendation(
                            category="missing_values",
                            severity=m.severity,
                            title=f"Investigate Missing-Value Spike in '{m.column_name}'",
                            evidence=m.description,
                            proposed_action="Audit upstream ingestion API to verify whether sensor or client payload stopped transmitting this field.",
                            code_snippet=(
                                f"# Check timestamps of missing entries to locate ingestion cutoff\n"
                                f"null_records = df[df['{m.column_name}'].isna()]"
                            ),
                            impact_description="Isolates broken data producer before corrupted batch reaches downstream warehouse.",
                        )
                    )

    # 4. Explicit Rule Validation Recommendations
    if validation_findings:
        failed_findings = [f for f in validation_findings if not f.passed]
        for f in failed_findings:
            if f.rule_name.startswith("range"):
                recommendations.append(
                    Recommendation(
                        category="outliers",
                        severity=Severity.CRITICAL,
                        title=f"Clamp or Filter Out-of-Range Outliers in '{f.column_name}'",
                        evidence=f.message,
                        proposed_action="Apply bound clamping (`clip`) or filter sensor glitches outside physical bounds.",
                        code_snippet=(
                            f"# Clamp extreme outliers to allowable operational bounds\n"
                            f"df['{f.column_name}'] = df['{f.column_name}'].clip(lower={f.expected_value.split()[-1]})"
                        ),
                        impact_description="Prevents mathematical overflows, exploding gradients, and corrupted KPI calculations.",
                    )
                )

    # 5. Multivariate Anomaly Recommendations
    if anomalies:
        high_anomalies = [a for a in anomalies if a.is_anomaly or a.anomaly_score >= 0.70]
        if high_anomalies:
            top_rec = high_anomalies[0]
            top_feat_names = [imp.feature for imp in top_rec.top_features[:2]]
            recommendations.append(
                Recommendation(
                    category="anomalies",
                    severity=Severity.WARNING,
                    title="Inspect Multivariate Outlier Cluster",
                    evidence=(
                        f"Detected {len(high_anomalies)} anomalous record(s). "
                        f"Lead outlier row #{top_rec.row_index} (ID: {top_rec.record_id}) "
                        f"deviates sharply on: {', '.join(top_feat_names)}."
                    ),
                    proposed_action=(
                        "Run `strataforge explain PATH --row ROW_ID` to inspect specific anomalies. "
                        "Determine if these reflect fraud attempts, VIP outliers, or sensor failures."
                    ),
                    code_snippet=(
                        f"# Inspect specific flagged record\n"
                        f"# strataforge explain dataset.csv --row {top_rec.row_index}"
                    ),
                    impact_description="Surfaces high-risk anomalies without blindly dropping genuine data.",
                )
            )

    return recommendations
