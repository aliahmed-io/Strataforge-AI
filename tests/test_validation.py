"""Unit tests for configuration loading and rule validation engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from strataforge.core.config import QualityRules
from strataforge.core.types import DataType
from strataforge.validation.engine import validate_dataset


def test_quality_rules_from_yaml():
    yaml_content = """
    version: 1
    dataset:
      min_rows: 50
      max_rows: 1000
      max_duplicate_pct: 0.0
    columns:
      amount:
        type: float
        required: true
        nullable: false
        range:
          min: 0.01
          max: 1000.00
      category:
        type: categorical
        allowed_values: ["retail", "grocery"]
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        rules = QualityRules.from_yaml_file(tmp_path)
        assert rules.dataset.min_rows == 50
        assert "amount" in rules.columns
        assert rules.columns["amount"].type == DataType.FLOAT
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_quality_rules_invalid_yaml_error():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write("columns:\n  amount:\n    max_missing_pct: 150.0  # Invalid > 100")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError) as exc_info:
            QualityRules.from_yaml_file(tmp_path)
        assert "Invalid rules configuration" in str(exc_info.value)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_validate_dataset_pass(baseline_dataframe: pd.DataFrame):
    yaml_content = """
    version: 1
    dataset:
      min_rows: 100
      max_duplicate_pct: 0.0
    columns:
      transaction_id:
        required: true
        nullable: false
        unique: true
      amount:
        type: float
        required: true
        range:
          min: 0.0
          max: 5000.0
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        rules = QualityRules.from_yaml_file(tmp_path)
        res = validate_dataset(baseline_dataframe, rules)
        assert res.passed is True
        assert res.failed_checks == 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_validate_dataset_fail(drifted_dataframe: pd.DataFrame):
    yaml_content = """
    version: 1
    columns:
      status:
        required: true
      category:
        allowed_values: ["retail", "grocery", "travel", "entertainment", "utility"]
      transaction_id:
        unique: true
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        rules = QualityRules.from_yaml_file(tmp_path)
        res = validate_dataset(drifted_dataframe, rules)
        assert res.passed is False
        assert res.failed_checks >= 2
    finally:
        Path(tmp_path).unlink(missing_ok=True)
