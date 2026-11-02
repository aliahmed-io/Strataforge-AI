"""End-to-end CLI command and exit-code validation tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from strataforge.cli import app
from strataforge.core.types import ExitCode

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Strataforge AI" in result.output


def test_cli_profile(temp_dataset_dir: Path):
    csv_file = temp_dataset_dir / "baseline.csv"
    result = runner.invoke(app, ["profile", str(csv_file), "--no-color"])
    assert result.exit_code == ExitCode.SUCCESS.value
    assert "Analytical Summary" in result.output
    assert "transacti" in result.output
    assert "Dimensions" in result.output


def test_cli_baseline_and_compare(temp_dataset_dir: Path):
    base_csv = temp_dataset_dir / "baseline.csv"
    curr_csv = temp_dataset_dir / "current.csv"
    base_json = temp_dataset_dir / "baseline.json"

    # 1. Create baseline
    res_base = runner.invoke(
        app, ["baseline", str(base_csv), "--out", str(base_json), "--no-color"]
    )
    assert res_base.exit_code == ExitCode.SUCCESS.value
    assert base_json.exists()

    # 2. Compare baseline with itself -> Exit 0
    res_self = runner.invoke(
        app, ["compare", str(base_csv), "--baseline", str(base_json), "--no-color"]
    )
    assert res_self.exit_code == ExitCode.SUCCESS.value

    # 3. Compare baseline with flawed current -> Exit 1
    res_drift = runner.invoke(
        app, ["compare", str(curr_csv), "--baseline", str(base_json), "--no-color"]
    )
    assert res_drift.exit_code == ExitCode.DATA_FAILURE.value


def test_cli_anomalies(temp_dataset_dir: Path):
    curr_csv = temp_dataset_dir / "current.csv"
    result = runner.invoke(app, ["anomalies", str(curr_csv), "--top", "5", "--no-color"])
    assert result.exit_code == ExitCode.SUCCESS.value
    assert "Multivariate Anomaly Findings" in result.output


def test_cli_explain(temp_dataset_dir: Path):
    curr_csv = temp_dataset_dir / "current.csv"
    result = runner.invoke(app, ["explain", str(curr_csv), "--row", "0", "--no-color"])
    assert result.exit_code == ExitCode.SUCCESS.value
    assert "DIAGNOSTIC EXPLANATION" in result.output
    assert "STATISTICAL DISCLAIMER" in result.output


def test_cli_report(temp_dataset_dir: Path):
    curr_csv = temp_dataset_dir / "current.csv"
    report_file = temp_dataset_dir / "test_report.html"
    result = runner.invoke(app, ["report", str(curr_csv), "--out", str(report_file), "--no-color"])
    assert result.exit_code == ExitCode.SUCCESS.value
    assert report_file.exists()


def test_cli_invalid_file_exit_code():
    result = runner.invoke(app, ["profile", "non_existent_file.csv", "--no-color"])
    assert result.exit_code == ExitCode.SYSTEM_ERROR.value
