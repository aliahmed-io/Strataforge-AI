"""Unit tests for standalone HTML report generation and structure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from strataforge.profiling.profiler import profile_dataset
from strataforge.reporting.html_generator import generate_html_report
from strataforge.reporting.summary import compute_quality_score


def test_html_report_standalone_content(baseline_dataframe: pd.DataFrame):
    prof = profile_dataset(baseline_dataframe)
    score = compute_quality_score(total_rows=prof.row_count)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        report_path = generate_html_report(
            profile=prof,
            score_summary=score,
            output_path=tmp_path,
        )

        with open(report_path, encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "STRATAFORGE AI" in content
        assert "HEALTH GRADE: A" in content
        assert "transaction_id" in content
        # Ensure zero external CDN links or unbundled web fonts
        assert "http://" not in content
        assert "https://" not in content
    finally:
        Path(tmp_path).unlink(missing_ok=True)
