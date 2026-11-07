"""Rule-based data quality validation engine."""

from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel, Field

from strataforge.core.config import QualityRules
from strataforge.core.schema import infer_series_type
from strataforge.core.types import DataType, Severity, ValidationFinding


class ValidationResult(BaseModel):
    """Overall outcome of running validation rules on a dataset."""

    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    findings: list[ValidationFinding] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


def validate_dataset(df: pd.DataFrame, rules: QualityRules) -> ValidationResult:
    """Validate DataFrame against explicit configured QualityRules."""
    findings: list[ValidationFinding] = []
    failure_reasons: list[str] = []
    total_checks = 0
    passed_checks = 0
    failed_checks = 0
    warning_checks = 0

    total_rows = len(df)

    # 1. Dataset-level checks
    # min_rows
    if rules.dataset.min_rows is not None:
        total_checks += 1
        ok = total_rows >= rules.dataset.min_rows
        findings.append(
            ValidationFinding(
                rule_name="min_rows",
                column_name=None,
                severity=Severity.PASS if ok else Severity.CRITICAL,
                passed=ok,
                actual_value=total_rows,
                expected_value=f">= {rules.dataset.min_rows}",
                message=(
                    f"Dataset row count ({total_rows}) meets threshold."
                    if ok
                    else f"Dataset row count ({total_rows}) is below minimum required ({rules.dataset.min_rows})."
                ),
            )
        )
        if ok:
            passed_checks += 1
        else:
            failed_checks += 1
            failure_reasons.append(
                f"Row count ({total_rows}) below min_rows ({rules.dataset.min_rows})"
            )

    # max_rows
    if rules.dataset.max_rows is not None:
        total_checks += 1
        ok = total_rows <= rules.dataset.max_rows
        findings.append(
            ValidationFinding(
                rule_name="max_rows",
                column_name=None,
                severity=Severity.PASS if ok else Severity.CRITICAL,
                passed=ok,
                actual_value=total_rows,
                expected_value=f"<= {rules.dataset.max_rows}",
                message=(
                    f"Dataset row count ({total_rows}) is within maximum limit."
                    if ok
                    else f"Dataset row count ({total_rows}) exceeds maximum permitted ({rules.dataset.max_rows})."
                ),
            )
        )
        if ok:
            passed_checks += 1
        else:
            failed_checks += 1
            failure_reasons.append(
                f"Row count ({total_rows}) exceeds max_rows ({rules.dataset.max_rows})"
            )

    # max_duplicate_pct
    total_checks += 1
    dup_mask = df.duplicated(keep=False)
    dup_pct = (dup_mask.sum() / max(total_rows, 1)) * 100.0
    ok = dup_pct <= rules.dataset.max_duplicate_pct
    findings.append(
        ValidationFinding(
            rule_name="max_duplicate_pct",
            column_name=None,
            severity=Severity.PASS if ok else Severity.CRITICAL,
            passed=ok,
            actual_value=f"{dup_pct:.2f}%",
            expected_value=f"<= {rules.dataset.max_duplicate_pct:.2f}%",
            message=(
                f"Duplicate rows ({dup_pct:.2f}%) within acceptable tolerance."
                if ok
                else f"Duplicate rows ({dup_pct:.2f}%) exceed permitted threshold ({rules.dataset.max_duplicate_pct:.2f}%)."
            ),
        )
    )
    if ok:
        passed_checks += 1
    else:
        failed_checks += 1
        failure_reasons.append(f"Duplicate rows ({dup_pct:.2f}%) exceed limit")

    # 2. Column-level assertions
    for col_name, col_rule in rules.columns.items():
        # Check presence
        col_exists = col_name in df.columns
        if col_rule.required:
            total_checks += 1
            if not col_exists:
                failed_checks += 1
                failure_reasons.append(f"Required column '{col_name}' is missing")
                findings.append(
                    ValidationFinding(
                        rule_name="required_column",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value="missing",
                        expected_value="present",
                        message=f"Required column '{col_name}' is absent from the dataset.",
                    )
                )
                continue
            else:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="required_column",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value="present",
                        expected_value="present",
                        message=f"Required column '{col_name}' is present.",
                    )
                )

        if not col_exists:
            continue

        series = df[col_name]
        inferred = infer_series_type(series)
        null_count = int(series.isna().sum())
        null_pct = (null_count / max(total_rows, 1)) * 100.0

        # Data type assertion
        if col_rule.type is not None:
            total_checks += 1
            type_ok = inferred == col_rule.type
            # Allow compatible widening (e.g. integer allowed as float, string/categorical)
            if not type_ok and col_rule.type == DataType.FLOAT and inferred == DataType.INTEGER:
                type_ok = True
            elif not type_ok and {col_rule.type, inferred} == {
                DataType.STRING,
                DataType.CATEGORICAL,
            }:
                type_ok = True

            if type_ok:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="data_type",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value=inferred.value,
                        expected_value=col_rule.type.value,
                        message=f"Column '{col_name}' matches expected type '{col_rule.type.value}'.",
                    )
                )
            else:
                failed_checks += 1
                failure_reasons.append(
                    f"Column '{col_name}' type mismatch: expected {col_rule.type.value}, got {inferred.value}"
                )
                findings.append(
                    ValidationFinding(
                        rule_name="data_type",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=inferred.value,
                        expected_value=col_rule.type.value,
                        message=f"Column '{col_name}' expected type '{col_rule.type.value}', but found '{inferred.value}'.",
                    )
                )

        # Nullability assertion
        if not col_rule.nullable:
            total_checks += 1
            if null_count > 0:
                failed_checks += 1
                failure_reasons.append(
                    f"Column '{col_name}' contains {null_count} unexpected null values"
                )
                findings.append(
                    ValidationFinding(
                        rule_name="not_null",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=f"{null_count} nulls ({null_pct:.2f}%)",
                        expected_value="0 nulls",
                        message=f"Column '{col_name}' is marked non-nullable, but contains {null_count} null(s).",
                    )
                )
            else:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="not_null",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value="0 nulls",
                        expected_value="0 nulls",
                        message=f"Column '{col_name}' contains zero nulls as required.",
                    )
                )

        # max_missing_pct assertion
        if col_rule.max_missing_pct is not None:
            total_checks += 1
            ok = null_pct <= col_rule.max_missing_pct
            if ok:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="max_missing_pct",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value=f"{null_pct:.2f}%",
                        expected_value=f"<= {col_rule.max_missing_pct:.2f}%",
                        message=f"Column '{col_name}' missing rate ({null_pct:.2f}%) within limit.",
                    )
                )
            else:
                failed_checks += 1
                failure_reasons.append(
                    f"Column '{col_name}' missing rate ({null_pct:.2f}%) exceeds max ({col_rule.max_missing_pct:.2f}%)"
                )
                findings.append(
                    ValidationFinding(
                        rule_name="max_missing_pct",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=f"{null_pct:.2f}%",
                        expected_value=f"<= {col_rule.max_missing_pct:.2f}%",
                        message=f"Column '{col_name}' missing rate ({null_pct:.2f}%) exceeds allowed threshold ({col_rule.max_missing_pct:.2f}%).",
                    )
                )

        # Unique assertion
        if col_rule.unique:
            total_checks += 1
            distinct_cnt = int(series.nunique())
            is_unique = (distinct_cnt == total_rows) and (null_count == 0)
            if is_unique:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="unique",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value="100% unique",
                        expected_value="100% unique",
                        message=f"Column '{col_name}' has 100% unique values.",
                    )
                )
            else:
                failed_checks += 1
                dups_cnt = total_rows - distinct_cnt
                failure_reasons.append(f"Column '{col_name}' has {dups_cnt} duplicate keys")
                findings.append(
                    ValidationFinding(
                        rule_name="unique",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=f"{distinct_cnt}/{total_rows} unique ({dups_cnt} dups)",
                        expected_value="all unique",
                        message=f"Unique constraint violated on column '{col_name}' with {dups_cnt} duplicate record(s).",
                    )
                )

        # Range assertion
        if col_rule.range is not None:
            num_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
            if len(num_series) > 0:
                min_b = col_rule.range.min
                max_b = col_rule.range.max

                if min_b is not None:
                    total_checks += 1
                    below = (num_series < min_b).sum()
                    if below == 0:
                        passed_checks += 1
                        findings.append(
                            ValidationFinding(
                                rule_name="range_min",
                                column_name=col_name,
                                severity=Severity.PASS,
                                passed=True,
                                actual_value=f"min={num_series.min()}",
                                expected_value=f">= {min_b}",
                                message=f"Column '{col_name}' values all >= {min_b}.",
                            )
                        )
                    else:
                        failed_checks += 1
                        failure_reasons.append(
                            f"Column '{col_name}' has {below} values below min {min_b}"
                        )
                        findings.append(
                            ValidationFinding(
                                rule_name="range_min",
                                column_name=col_name,
                                severity=Severity.CRITICAL,
                                passed=False,
                                actual_value=f"{below} values < {min_b} (min: {num_series.min()})",
                                expected_value=f">= {min_b}",
                                message=f"Column '{col_name}' contains {below} value(s) below allowed minimum {min_b}.",
                            )
                        )

                if max_b is not None:
                    total_checks += 1
                    above = (num_series > max_b).sum()
                    if above == 0:
                        passed_checks += 1
                        findings.append(
                            ValidationFinding(
                                rule_name="range_max",
                                column_name=col_name,
                                severity=Severity.PASS,
                                passed=True,
                                actual_value=f"max={num_series.max()}",
                                expected_value=f"<= {max_b}",
                                message=f"Column '{col_name}' values all <= {max_b}.",
                            )
                        )
                    else:
                        failed_checks += 1
                        failure_reasons.append(
                            f"Column '{col_name}' has {above} values above max {max_b}"
                        )
                        findings.append(
                            ValidationFinding(
                                rule_name="range_max",
                                column_name=col_name,
                                severity=Severity.CRITICAL,
                                passed=False,
                                actual_value=f"{above} values > {max_b} (max: {num_series.max()})",
                                expected_value=f"<= {max_b}",
                                message=f"Column '{col_name}' contains {above} value(s) exceeding allowed maximum {max_b}.",
                            )
                        )

        # Allowed values assertion
        if col_rule.allowed_values is not None:
            total_checks += 1
            allowed_set = {str(v) for v in col_rule.allowed_values}
            actual_vals = series.dropna().astype(str).unique()
            unexpected = [v for v in actual_vals if v not in allowed_set]

            if not unexpected:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="allowed_values",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value="all allowed",
                        expected_value=str(col_rule.allowed_values),
                        message=f"All values in column '{col_name}' belong to allowed set.",
                    )
                )
            else:
                failed_checks += 1
                failure_reasons.append(
                    f"Column '{col_name}' contains unapproved categories: {unexpected[:3]}"
                )
                findings.append(
                    ValidationFinding(
                        rule_name="allowed_values",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=f"Unexpected: {unexpected[:5]}",
                        expected_value=str(col_rule.allowed_values),
                        message=f"Column '{col_name}' contains {len(unexpected)} unapproved category value(s): {unexpected[:3]}.",
                    )
                )

        # Regex pattern assertion
        if col_rule.regex is not None:
            total_checks += 1
            pattern = re.compile(col_rule.regex)
            str_series = series.dropna().astype(str)
            mismatches = str_series.apply(lambda s, pat=pattern: bool(pat.match(s)) is False).sum()

            if mismatches == 0:
                passed_checks += 1
                findings.append(
                    ValidationFinding(
                        rule_name="regex_pattern",
                        column_name=col_name,
                        severity=Severity.PASS,
                        passed=True,
                        actual_value="100% matched",
                        expected_value=col_rule.regex,
                        message=f"All non-null values in '{col_name}' match regex pattern.",
                    )
                )
            else:
                failed_checks += 1
                failure_reasons.append(
                    f"Column '{col_name}' has {mismatches} values violating regex {col_rule.regex}"
                )
                findings.append(
                    ValidationFinding(
                        rule_name="regex_pattern",
                        column_name=col_name,
                        severity=Severity.CRITICAL,
                        passed=False,
                        actual_value=f"{mismatches} non-matching values",
                        expected_value=col_rule.regex,
                        message=f"Column '{col_name}' contains {mismatches} value(s) failing pattern '{col_rule.regex}'.",
                    )
                )

    overall_passed = failed_checks == 0

    return ValidationResult(
        passed=overall_passed,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        warning_checks=warning_checks,
        findings=findings,
        failure_reasons=failure_reasons,
    )
