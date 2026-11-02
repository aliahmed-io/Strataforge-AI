"""Aggregated dataset quality scoring and health index calculation."""

from __future__ import annotations

from strataforge.core.types import (
    ChangeType,
    DriftMetric,
    QualityScoreSummary,
    SchemaChange,
    Severity,
    ValidationFinding,
)
from strataforge.profiling.duplicates import DuplicateReport


def compute_quality_score(
    schema_changes: list[SchemaChange] | None = None,
    drift_metrics: list[DriftMetric] | None = None,
    validation_findings: list[ValidationFinding] | None = None,
    duplicate_report: DuplicateReport | None = None,
    anomaly_count: int = 0,
    total_rows: int = 1,
) -> QualityScoreSummary:
    """Compute normalized 0-100 data quality health score and assign letter grade."""
    score = 100.0
    critical_checks = 0
    warning_checks = 0
    passed_checks = 0
    total_checks = 0

    # 1. Schema impact
    if schema_changes:
        for chg in schema_changes:
            total_checks += 1
            if chg.change_type == ChangeType.BREAKING:
                critical_checks += 1
                score -= 25.0
            elif chg.change_type == ChangeType.NON_BREAKING:
                warning_checks += 1
                score -= 5.0
            else:
                passed_checks += 1

    # 2. Drift impact
    if drift_metrics:
        for m in drift_metrics:
            total_checks += 1
            if m.severity == Severity.CRITICAL:
                critical_checks += 1
                score -= 15.0
            elif m.severity == Severity.WARNING:
                warning_checks += 1
                score -= 5.0
            else:
                passed_checks += 1

    # 3. Validation rule impact
    if validation_findings:
        for f in validation_findings:
            total_checks += 1
            if not f.passed:
                if f.severity == Severity.CRITICAL:
                    critical_checks += 1
                    score -= 15.0
                else:
                    warning_checks += 1
                    score -= 5.0
            else:
                passed_checks += 1

    # 4. Duplicate impact
    if duplicate_report:
        total_checks += 1
        if duplicate_report.has_critical_duplicates:
            critical_checks += 1
            # Deduct proportional to duplicate percentage, max 20 pts
            deduction = min(20.0, max(5.0, duplicate_report.duplicate_percentage * 2.0))
            score -= deduction
        else:
            passed_checks += 1

    # 5. Anomaly impact
    if total_rows > 0 and anomaly_count > 0:
        total_checks += 1
        anom_pct = (anomaly_count / total_rows) * 100.0
        if anom_pct > 10.0:
            critical_checks += 1
            score -= 15.0
        elif anom_pct > 5.0:
            warning_checks += 1
            score -= 5.0
        else:
            passed_checks += 1

    score = max(0.0, min(100.0, round(score, 1)))

    if score >= 90.0:
        grade = "A"
        status = Severity.PASS
    elif score >= 80.0:
        grade = "B"
        status = Severity.INFO
    elif score >= 70.0:
        grade = "C"
        status = Severity.WARNING
    elif score >= 60.0:
        grade = "D"
        status = Severity.WARNING
    else:
        grade = "F"
        status = Severity.CRITICAL

    if critical_checks > 0 and status != Severity.CRITICAL:
        status = Severity.CRITICAL

    return QualityScoreSummary(
        health_score=score,
        grade=grade,
        status=status,
        total_checks=total_checks,
        passed_checks=passed_checks,
        warning_checks=warning_checks,
        critical_checks=critical_checks,
    )
