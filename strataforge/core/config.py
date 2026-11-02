"""Configuration loader and schema validator for Strataforge rules.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from strataforge.core.types import DataType


class NumericRange(BaseModel):
    """Allowed numeric range bounds."""

    min: float | None = None
    max: float | None = None


class ColumnRule(BaseModel):
    """Validation rules configured for a single column."""

    type: DataType | None = None
    required: bool = True
    nullable: bool = True
    unique: bool = False
    range: NumericRange | None = None
    allowed_values: list[Any] | None = None
    regex: str | None = None
    max_missing_pct: float | None = Field(default=None, ge=0.0, le=100.0)


class DatasetRules(BaseModel):
    """Dataset-level validation rules."""

    min_rows: int | None = Field(default=None, ge=0)
    max_rows: int | None = Field(default=None, ge=0)
    max_duplicate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    primary_key: list[str] | str | None = None


class DriftConfig(BaseModel):
    """Statistical drift thresholds."""

    psi_warning: float = Field(default=0.10, ge=0.0)
    psi_critical: float = Field(default=0.25, ge=0.0)
    ks_p_value_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    max_missing_delta_pct: float = Field(default=5.0, ge=0.0, le=100.0)


class AnomalyConfig(BaseModel):
    """Isolation Forest anomaly detection configuration."""

    contamination: float = Field(default=0.05, gt=0.0, le=0.5)
    max_anomaly_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    features: list[str] | None = None
    random_state: int = 42


class QualityRules(BaseModel):
    """Root configuration model representing rules.yaml."""

    version: int | str = 1
    dataset_name: str | None = None
    dataset: DatasetRules = Field(default_factory=DatasetRules)
    columns: dict[str, ColumnRule] = Field(default_factory=dict)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig)
    fail_on: list[str] = Field(
        default_factory=lambda: [
            "breaking_schema",
            "validation_errors",
            "critical_drift",
            "high_anomalies",
        ]
    )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> QualityRules:
        """Load and validate rules from a YAML file with clear human-readable error reporting."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML Syntax Error in {file_path}: {exc}") from exc

        if raw_data is None:
            raw_data = {}
        elif not isinstance(raw_data, dict):
            raise ValueError(f"Configuration root in {file_path} must be a mapping/dictionary.")

        try:
            return cls.model_validate(raw_data)
        except ValidationError as err:
            formatted_errors = []
            for e in err.errors():
                loc = " -> ".join(str(item) for item in e.get("loc", []))
                msg = e.get("msg", "Invalid value")
                formatted_errors.append(f"  • [{loc}]: {msg}")
            error_msg = f"Invalid rules configuration in {file_path}:\n" + "\n".join(
                formatted_errors
            )
            raise ValueError(error_msg) from err


def generate_default_rules_yaml() -> str:
    """Generate an extensively documented default rules.yaml template."""
    return """# Strataforge AI — Data Quality Rules Configuration
# Documentation: https://github.com/strataforge/strataforge
version: 1
dataset_name: "transaction_pipeline"

# Dataset-level health criteria
dataset:
  min_rows: 10
  max_rows: 10000000
  max_duplicate_pct: 0.0  # Zero duplicates allowed
  primary_key: ["transaction_id"]

# Column-level validation rules
columns:
  transaction_id:
    type: string
    required: true
    nullable: false
    unique: true
    regex: "^TX-[0-9]{5,10}$"

  user_id:
    type: integer
    required: true
    nullable: false
    range:
      min: 1
      max: 2147483647

  amount:
    type: float
    required: true
    nullable: false
    range:
      min: 0.01
      max: 100000.00
    max_missing_pct: 0.0

  category:
    type: categorical
    required: true
    nullable: false
    allowed_values: ["retail", "grocery", "travel", "entertainment", "utility"]

  status:
    type: string
    required: true
    nullable: false
    allowed_values: ["settled", "pending", "declined", "refunded"]

  created_at:
    type: datetime
    required: true
    nullable: false

  notes:
    type: string
    required: false
    nullable: true
    max_missing_pct: 95.0

# Statistical drift thresholds
drift:
  psi_warning: 0.10      # PSI >= 0.10 triggers Warning
  psi_critical: 0.25     # PSI >= 0.25 triggers Critical Failure
  ks_p_value_threshold: 0.01  # p-value < 0.01 indicates significant drift
  max_missing_delta_pct: 5.0  # Missing rate increase > 5.0% triggers alert

# Multivariate anomaly detection (Isolation Forest)
anomalies:
  contamination: 0.05    # Expected fraction of outliers (5%)
  max_anomaly_pct: 8.0   # If anomalies exceed 8%, trigger quality alert
  random_state: 42

# Conditions that trigger process exit code 1 (failure in CI/CD)
fail_on:
  - breaking_schema
  - validation_errors
  - critical_drift
  - high_anomalies
"""
