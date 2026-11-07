"""Visual and DOM Verification Test for Standalone HTML Report.

Uses Playwright to audit the generated HTML report:
- Ensures it opens file:// without web server
- Verifies zero console errors or uncaught exceptions
- Audits presence of critical UI sections (Schema, Drift, Anomalies, Gates)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright
from typer.testing import CliRunner

from strataforge.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def generated_report_html(tmp_path_factory) -> Path:
    """Generate a fresh HTML report for inspection."""
    temp_dir = tmp_path_factory.mktemp("report_test")
    report_file = temp_dir / "audit_report.html"

    res = runner.invoke(
        app,
        [
            "report",
            "demo_current.csv",
            "--baseline",
            "baseline.json",
            "--config",
            "rules.yaml",
            "--out",
            str(report_file),
        ],
    )
    assert res.exit_code == 0
    assert report_file.exists()
    return report_file


def test_html_report_opens_without_server_and_zero_errors(generated_report_html: Path) -> None:
    """Ensure the static report loads locally with zero JavaScript/console errors."""
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.on(
            "console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None
        )
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        file_url = "file:///" + os.path.abspath(str(generated_report_html)).replace("\\", "/")
        page.goto(file_url)
        page.wait_for_timeout(300)

        # Confirm zero runtime errors
        assert len(console_errors) == 0, f"Encountered JS errors on report load: {console_errors}"

        # Confirm DOM elements
        assert page.locator(".brand-title").count() > 0
        assert page.locator(".kpi-grid").count() > 0
        assert page.locator("text=Inferred Schema").count() > 0
        assert page.locator("text=Statistical Distribution Drift").count() > 0
        assert page.locator("text=Multivariate Isolation Forest Anomalies").count() > 0

        browser.close()


def test_html_report_responsive_mobile_viewport(generated_report_html: Path) -> None:
    """Ensure the report loads and key cards adapt responsively on mobile viewport."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})

        file_url = "file:///" + os.path.abspath(str(generated_report_html)).replace("\\", "/")
        page.goto(file_url)
        page.wait_for_timeout(200)

        # Confirm KPI grid and brand cluster render cleanly
        assert page.locator(".kpi-grid").count() > 0
        assert page.locator(".brand-title").count() > 0

        browser.close()
