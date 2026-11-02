"""Core types, enumerations, and Pydantic schemas for Strataforge AI."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DataType(StrEnum):
    """Normalized data types inferred by Strataforge."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Severity levels for data quality findings."""

    PASS = "PASS"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class ExitCode(int, Enum):
    """CLI exit codes for deterministic CI/CD automation."""

    SUCCESS = 0
    DATA_FAILURE = 1
    SYSTEM_ERROR = 2


class ChangeType(StrEnum):
    """Schema change classifications."""

    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    NEUTRAL = "neutral"


class Quantiles(BaseModel):
    """Distribution quantiles for numerical features."""

    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None


class ColumnProfile(BaseModel):
    """Profile metrics and distribution data for a single column."""

    name: str
    inferred_type: DataType
    native_dtype: str
    total_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    distinct_percentage: float
    is_unique: bool = False
    is_nullable: bool = False
    sample_values: list[Any] = Field(default_factory=list)
    min_val: float | str | None = None
    max_val: float | str | None = None
    mean_val: float | None = None
    std_val: float | None = None
    quantiles: Quantiles | None = None
    top_categories: dict[str, int] = Field(default_factory=dict)
    category_frequencies: dict[str, float] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    """High-level profile summary of an analyzed dataset."""

    dataset_path: str
    file_format: str
    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_count: int
    duplicate_percentage: float
    columns: dict[str, ColumnProfile]
    created_at: str


class SchemaChange(BaseModel):
    """A detected schema change between baseline and current dataset."""

    column_name: str
    change_kind: str  # added, removed, type_changed, nullability_changed, order_changed
    old_value: str | None = None
    new_value: str | None = None
    change_type: ChangeType
    description: str


class DriftMetric(BaseModel):
    """Statistical drift comparison for a single feature."""

    column_name: str
    feature_type: DataType
    metric_name: str  # "psi", "ks", "missing_rate_delta", "category_drift"
    metric_display_name: str
    stat_value: float
    p_value: float | None = None
    threshold_warning: float
    threshold_critical: float
    severity: Severity
    description: str
    sample_size_current: int
    sample_size_baseline: int
    notes: str | None = None


class FeatureImpact(BaseModel):
    """Anomaly contribution breakdown for a specific feature."""

    feature: str
    value: Any
    baseline_mean: float | None = None
    baseline_std: float | None = None
    baseline_median: float | None = None
    z_score: float | None = None
    relative_deviation: float | None = None
    contribution_score: float
    explanation: str


class AnomalyRecord(BaseModel):
    """Anomalous record flagged by multivariate model."""

    row_index: int
    record_id: str | int
    anomaly_score: float  # Normalized anomaly score: higher = more anomalous (0.0 to 1.0)
    raw_decision_score: float  # Raw IsolationForest decision_function value
    is_anomaly: bool
    top_features: list[FeatureImpact] = Field(default_factory=list)
    row_values: dict[str, Any] = Field(default_factory=dict)
    diagnostic_summary: str


class ValidationFinding(BaseModel):
    """Individual assertion finding from a rule check."""

    rule_name: str
    column_name: str | None = None
    severity: Severity
    passed: bool
    actual_value: Any = None
    expected_value: Any = None
    message: str


class Recommendation(BaseModel):
    """Evidence-based automated recommendation for remediating data quality."""

    category: str  # "schema", "missing_values", "drift", "duplicates", "outliers", "validation"
    severity: Severity
    title: str
    evidence: str
    proposed_action: str
    code_snippet: str | None = None
    impact_description: str


class QualityScoreSummary(BaseModel):
    """Overall dataset health score and status."""

    health_score: float  # 0 to 100
    grade: str  # A, B, C, D, F
    status: Severity
    total_checks: int
    passed_checks: int
    warning_checks: int
    critical_checks: int
