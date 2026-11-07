"""End-to-End User Experience & Black-Box Acceptance Tests.

Tests the CLI from the end-user's perspective:
- Error empathy (clean error messaging, no unhandled tracebacks)
- Exit codes (0 for pass, 1 for quality failure, 2 for user/file errors)
- Non-interactive flags (--compact, --no-color)
- Discovery commands (--help, init)
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from strataforge.cli import app

runner = CliRunner()


def test_user_journey_missing_file_error_empathy() -> None:
    """A user passing a non-existent file path gets a clear explanation and exit code 2."""
    result = runner.invoke(app, ["scan", "definitely_does_not_exist.csv"])
    assert result.exit_code == 2
    # Ensure there is no raw Python traceback dumped to the user
    assert "Traceback (most recent call last)" not in result.output
    assert "Error:" in result.output or "Dataset file not found" in result.output


def test_user_journey_invalid_yaml_error_empathy(tmp_path: Path) -> None:
    """A user passing a malformed YAML rule configuration gets an actionable message."""
    bad_yaml = tmp_path / "broken_rules.yaml"
    bad_yaml.write_text("rules: [this is unclosed yaml", encoding="utf-8")

    result = runner.invoke(app, ["validate", "demo_baseline.csv", "--config", str(bad_yaml)])
    assert result.exit_code == 2
    assert "Traceback (most recent call last)" not in result.output
    assert "Error" in result.output


def test_user_journey_clean_dataset_exit_code_zero() -> None:
    """A clean reference dataset with no violations returns exit code 0."""
    result = runner.invoke(
        app,
        ["validate", "demo_baseline.csv", "--config", "rules.yaml"],
    )
    assert result.exit_code == 0
    assert "passed successfully" in result.output.lower() or "pass" in result.output.lower()


def test_user_journey_corrupted_dataset_exit_code_one() -> None:
    """A dataset violating data-quality gates returns exit code 1."""
    result = runner.invoke(
        app,
        ["validate", "demo_current.csv", "--config", "rules.yaml"],
    )
    assert result.exit_code == 1
    assert (
        "critical data quality failure" in result.output.lower()
        or "failed" in result.output.lower()
    )


def test_user_journey_no_color_flag_for_ci() -> None:
    """In CI or headless environments, --no-color strips ANSI color escape codes."""
    result = runner.invoke(app, ["profile", "demo_baseline.csv", "--no-color"])
    assert result.exit_code == 0
    # ANSI escape sequences start with \x1b[
    assert "\x1b[" not in result.output
    assert "PROFILE" in result.output
    assert "STRATAFORGE" in result.output


def test_user_journey_compact_mode_for_scripts() -> None:
    """--compact mode renders simple table format suitable for logs."""
    result = runner.invoke(
        app,
        [
            "scan",
            "demo_current.csv",
            "--baseline",
            "baseline.json",
            "--config",
            "rules.yaml",
            "--compact",
        ],
    )
    assert result.exit_code == 1
    assert "QUALITY AUDIT SUMMARY" in result.output or "Scan completed" in result.output


def test_user_journey_init_scaffolding(tmp_path: Path) -> None:
    """'strataforge init' creates scaffolded configuration in target directory."""
    result = runner.invoke(app, ["init", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "rules.yaml").exists()
    assert (tmp_path / "demo_baseline.csv").exists()
    assert (tmp_path / "demo_current.csv").exists()
