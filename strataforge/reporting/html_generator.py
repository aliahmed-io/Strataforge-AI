"""Standalone HTML quality report generator styled with Strataforge terminal palette."""

from __future__ import annotations

import html
from pathlib import Path

from strataforge.core.types import (
    AnomalyRecord,
    DatasetProfile,
    DriftMetric,
    QualityScoreSummary,
    Recommendation,
    SchemaChange,
    ValidationFinding,
)
from strataforge.profiling.duplicates import DuplicateReport


def generate_html_report(
    profile: DatasetProfile,
    score_summary: QualityScoreSummary,
    output_path: str | Path,
    schema_changes: list[SchemaChange] | None = None,
    drift_metrics: list[DriftMetric] | None = None,
    validation_findings: list[ValidationFinding] | None = None,
    duplicate_report: DuplicateReport | None = None,
    anomalies: list[AnomalyRecord] | None = None,
    recommendations: list[Recommendation] | None = None,
) -> str:
    """Generate a self-contained, offline-compatible HTML report document."""
    schema_changes = schema_changes or []
    drift_metrics = drift_metrics or []
    validation_findings = validation_findings or []
    anomalies = anomalies or []
    recommendations = recommendations or []

    # Format numbers
    row_count_str = f"{profile.row_count:,}"
    col_count_str = str(profile.column_count)
    memory_mb = f"{profile.memory_bytes / (1024 * 1024):.2f} MB"
    dup_str = f"{profile.duplicate_count:,} ({profile.duplicate_percentage:.2f}%)"
    health_score = score_summary.health_score
    grade = score_summary.grade

    # Color tokens
    score_color = (
        "#91C8A8" if health_score >= 80 else ("#ECC06C" if health_score >= 60 else "#E77770")
    )

    # Render Column Profiles Rows
    column_rows_html = []
    for col_name, cp in profile.columns.items():
        type_badge = f'<span class="badge badge-type">{cp.inferred_type.value}</span>'
        null_badge_class = (
            "badge-pass"
            if cp.null_count == 0
            else ("badge-crit" if cp.null_percentage > 10 else "badge-warn")
        )
        null_cell = f'<span class="badge {null_badge_class}">{cp.null_count} ({cp.null_percentage:.1f}%)</span>'
        distinct_cell = f"{cp.distinct_count:,} ({cp.distinct_percentage:.1f}%)"

        if cp.min_val is not None and cp.max_val is not None:
            stats_cell = f'<div class="cell-mono">min: {cp.min_val}<br>avg: {cp.mean_val}<br>max: {cp.max_val}</div>'
        elif cp.top_categories:
            top_cats_str = ", ".join(f"{k} ({v})" for k, v in list(cp.top_categories.items())[:3])
            stats_cell = f'<div class="cell-mono" title="{html.escape(top_cats_str)}">{html.escape(top_cats_str[:40])}...</div>'
        else:
            stats_cell = '<span class="text-muted">N/A</span>'

        samples = ", ".join(html.escape(str(s)) for s in cp.sample_values) or "None"

        column_rows_html.append(f"""
        <tr>
            <td class="col-name font-mono"><strong>{html.escape(col_name)}</strong></td>
            <td>{type_badge}</td>
            <td>{null_cell}</td>
            <td class="font-mono">{distinct_cell}</td>
            <td>{stats_cell}</td>
            <td class="font-mono text-muted text-small">{samples}</td>
        </tr>
        """)

    # Render Drift Metrics Rows
    drift_rows_html = []
    for dm in drift_metrics:
        badge_cls = (
            "badge-pass"
            if dm.severity == "PASS"
            else ("badge-warn" if dm.severity == "WARNING" else "badge-crit")
        )
        p_val_str = f"{dm.p_value:.2e}" if dm.p_value is not None else "-"
        notes_str = (
            f'<div class="text-small text-muted">{html.escape(dm.notes)}</div>' if dm.notes else ""
        )

        drift_rows_html.append(f"""
        <tr>
            <td class="font-mono"><strong>{html.escape(dm.column_name)}</strong></td>
            <td><span class="badge badge-type">{html.escape(dm.metric_display_name)}</span></td>
            <td class="font-mono">{dm.stat_value:.4f}</td>
            <td class="font-mono">{p_val_str}</td>
            <td><span class="badge {badge_cls}">{dm.severity.value}</span></td>
            <td>{html.escape(dm.description)}{notes_str}</td>
        </tr>
        """)

    # Render Schema Changes Rows
    schema_rows_html = []
    for sc in schema_changes:
        badge_cls = "badge-crit" if sc.change_type == "breaking" else "badge-warn"
        schema_rows_html.append(f"""
        <tr>
            <td class="font-mono"><strong>{html.escape(sc.column_name)}</strong></td>
            <td><span class="badge {badge_cls}">{sc.change_kind.upper()}</span></td>
            <td><span class="badge {badge_cls}">{sc.change_type.value.upper()}</span></td>
            <td class="font-mono">{html.escape(str(sc.old_value or "-"))}</td>
            <td class="font-mono">{html.escape(str(sc.new_value or "-"))}</td>
            <td>{html.escape(sc.description)}</td>
        </tr>
        """)

    # Render Anomalies Rows
    anomalies_rows_html = []
    for anom in anomalies[:25]:
        top_feats = []
        for feat in anom.top_features[:3]:
            z_str = f"{feat.z_score:+.1f}σ" if feat.z_score is not None else "divergent/null"
            top_feats.append(f"{html.escape(feat.feature)} (val: {feat.value}, z: {z_str})")
        top_feats_str = "<br>".join(top_feats)

        score_cls = "badge-crit" if anom.anomaly_score >= 0.70 else "badge-warn"

        anomalies_rows_html.append(f"""
        <tr>
            <td class="font-mono">#{anom.row_index}</td>
            <td class="font-mono">{html.escape(str(anom.record_id))}</td>
            <td><span class="badge {score_cls}">{anom.anomaly_score:.2f}</span></td>
            <td class="font-mono text-small">{top_feats_str}</td>
            <td class="text-small">{html.escape(anom.diagnostic_summary)}</td>
        </tr>
        """)

    # Render Validation Findings Rows
    validation_rows_html = []
    for vf in validation_findings:
        badge_cls = (
            "badge-pass"
            if vf.passed
            else ("badge-crit" if vf.severity == "CRITICAL" else "badge-warn")
        )
        status_txt = "PASS" if vf.passed else vf.severity.value
        validation_rows_html.append(f"""
        <tr>
            <td class="font-mono"><strong>{html.escape(vf.rule_name)}</strong></td>
            <td class="font-mono">{html.escape(str(vf.column_name or "*"))}</td>
            <td><span class="badge {badge_cls}">{status_txt}</span></td>
            <td class="font-mono text-small">{html.escape(str(vf.actual_value))}</td>
            <td class="font-mono text-small">{html.escape(str(vf.expected_value))}</td>
            <td>{html.escape(vf.message)}</td>
        </tr>
        """)

    # Render Recommendations Cards
    recs_cards_html = []
    for rec in recommendations:
        badge_cls = (
            "badge-crit"
            if rec.severity == "CRITICAL"
            else ("badge-warn" if rec.severity == "WARNING" else "badge-pass")
        )
        code_block = ""
        if rec.code_snippet:
            code_block = f"""
            <div class="code-box">
                <div class="code-header font-mono">Suggested Remediation Snippet</div>
                <pre><code>{html.escape(rec.code_snippet)}</code></pre>
            </div>
            """

        recs_cards_html.append(f"""
        <div class="rec-card">
            <div class="rec-top">
                <div class="rec-title">
                    <span class="badge {badge_cls}">{rec.severity.value}</span>
                    <span class="badge badge-type">{rec.category.upper()}</span>
                    <strong>{html.escape(rec.title)}</strong>
                </div>
            </div>
            <div class="rec-body">
                <p><strong>Evidence:</strong> {html.escape(rec.evidence)}</p>
                <p><strong>Proposed Action:</strong> {html.escape(rec.proposed_action)}</p>
                <p class="text-small text-muted"><strong>Impact:</strong> {html.escape(rec.impact_description)}</p>
                {code_block}
            </div>
        </div>
        """)

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strataforge AI — Data Quality Report</title>
    <style>
        :root {{
            --bg-base: #17151F;
            --bg-surface: #272039;
            --bg-elevated: #312847;
            --border-subtle: #3D3357;
            --accent-copper: #E9AC74;
            --color-sage: #91C8A8;
            --color-coral: #E77770;
            --color-gold: #ECC06C;
            --text-primary: #F2EFF7;
            --text-secondary: #C4BCD1;
            --text-muted: #8E83A3;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 14px;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 24px;
            margin-bottom: 24px;
        }}
        .brand-title {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: var(--accent-copper);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand-badge {{
            background: rgba(233, 172, 116, 0.15);
            color: var(--accent-copper);
            border: 1px solid rgba(233, 172, 116, 0.3);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .meta-text {{ color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 20px;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }}
        .kpi-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
        .section-header {{
            font-size: 18px;
            font-weight: 600;
            color: var(--accent-copper);
            margin: 32px 0 16px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .table-wrap {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 32px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid rgba(61, 51, 87, 0.4);
            vertical-align: middle;
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            text-transform: uppercase;
        }}
        .badge-pass {{ background: rgba(145, 200, 168, 0.15); color: var(--color-sage); border: 1px solid rgba(145, 200, 168, 0.3); }}
        .badge-warn {{ background: rgba(236, 192, 108, 0.15); color: var(--color-gold); border: 1px solid rgba(236, 192, 108, 0.3); }}
        .badge-crit {{ background: rgba(231, 119, 112, 0.15); color: var(--color-coral); border: 1px solid rgba(231, 119, 112, 0.3); }}
        .badge-type {{ background: rgba(196, 188, 209, 0.12); color: var(--text-secondary); border: 1px solid var(--border-subtle); }}
        .font-mono {{ font-family: var(--font-mono); }}
        .text-muted {{ color: var(--text-muted); }}
        .text-small {{ font-size: 12px; }}
        .rec-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
            margin-bottom: 32px;
        }}
        .rec-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 20px;
        }}
        .rec-top {{ margin-bottom: 12px; }}
        .rec-title {{ display: flex; align-items: center; gap: 10px; font-size: 16px; }}
        .rec-body p {{ margin-bottom: 8px; }}
        .code-box {{
            background: #110F17;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 12px 16px;
            margin-top: 12px;
        }}
        .code-header {{ font-size: 11px; color: var(--accent-copper); margin-bottom: 6px; }}
        pre {{ margin: 0; overflow-x: auto; font-family: var(--font-mono); font-size: 12px; color: #E0DAE8; }}
        .disclaimer-banner {{
            background: rgba(233, 172, 116, 0.08);
            border: 1px solid rgba(233, 172, 116, 0.25);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 16px;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .footer {{
            border-top: 1px solid var(--border-subtle);
            padding-top: 24px;
            margin-top: 48px;
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}
        .search-box {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-family: var(--font-mono);
            font-size: 12px;
            width: 260px;
        }}
        .search-box:focus {{ outline: 1px solid var(--accent-copper); }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div>
                <div class="brand-title">
                    <span>STRATAFORGE AI</span>
                    <span class="brand-badge">Engine v0.1.0</span>
                </div>
                <div class="meta-text">Target: {html.escape(profile.dataset_path)} &bull; Format: {
        profile.file_format.upper()
    } &bull; Generated: {profile.created_at}</div>
            </div>
            <div>
                <span class="badge" style="background: {score_color}22; color: {
        score_color
    }; border: 1px solid {score_color}66; font-size: 14px; padding: 6px 14px;">
                    HEALTH GRADE: {grade} ({health_score:.1f}/100)
                </span>
            </div>
        </header>

        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Health Score</div>
                <div class="kpi-value" style="color: {score_color};">{health_score:.1f}</div>
                <div class="kpi-sub">Composite Quality Index</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Observations</div>
                <div class="kpi-value">{row_count_str}</div>
                <div class="kpi-sub">{col_count_str} columns &bull; {memory_mb}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duplicate Records</div>
                <div class="kpi-value">{dup_str}</div>
                <div class="kpi-sub">Exact full-row duplicates</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Checks Executed</div>
                <div class="kpi-value">{score_summary.total_checks}</div>
                <div class="kpi-sub">{score_summary.passed_checks} passed &bull; {
        score_summary.critical_checks
    } critical</div>
            </div>
        </section>

        <!-- Column Profiles Section -->
        <div class="section-header">
            <span>Inferred Schema & Column Distributions</span>
            <input type="text" id="columnSearch" class="search-box" placeholder="Filter columns by name..." onkeyup="filterColumns()">
        </div>
        <div class="table-wrap">
            <table id="columnsTable">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Inferred Type</th>
                        <th>Missing / Nulls</th>
                        <th>Distinct Count</th>
                        <th>Distribution (Min / Mean / Max)</th>
                        <th>Sample Values</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(column_rows_html)}
                </tbody>
            </table>
        </div>

        <!-- Schema Changes Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Schema Evolution Changes ({len(schema_changes)})</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Change Kind</th>
                        <th>Severity</th>
                        <th>Baseline Value</th>
                        <th>Current Value</th>
                        <th>Impact Description</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(schema_rows_html)}
                </tbody>
            </table>
        </div>
        '''
        if schema_changes
        else ""
    }

        <!-- Statistical Drift Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Statistical Distribution Drift ({len(drift_metrics)})</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Metric</th>
                        <th>Statistic</th>
                        <th>p-value</th>
                        <th>Status</th>
                        <th>Diagnostic Findings</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(drift_rows_html)}
                </tbody>
            </table>
        </div>
        '''
        if drift_metrics
        else ""
    }

        <!-- Multivariate Anomalies Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Multivariate Isolation Forest Anomalies (Top Flagged)</span>
        </div>
        <div class="disclaimer-banner">
            <strong>Diagnostic Note:</strong> Multivariate anomaly scores reflect distance from reference density clusters.
            Anomalous records represent statistically unusual combinations, not necessarily corrupted records.
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Row Index</th>
                        <th>Record ID</th>
                        <th>Anomaly Score</th>
                        <th>Key Divergent Features</th>
                        <th>Diagnostic Summary</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(anomalies_rows_html)}
                </tbody>
            </table>
        </div>
        '''
        if anomalies
        else ""
    }

        <!-- Validation Rules Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Quality Rule Assertions ({len(validation_findings)})</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Rule</th>
                        <th>Column</th>
                        <th>Status</th>
                        <th>Observed</th>
                        <th>Expected</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(validation_rows_html)}
                </tbody>
            </table>
        </div>
        '''
        if validation_findings
        else ""
    }

        <!-- Intelligent Recommendations Section -->
        {
        f'''
        <div class="section-header">
            <span>Intelligent Remediation Recommendations ({len(recommendations)})</span>
        </div>
        <div class="rec-grid">
            {"".join(recs_cards_html)}
        </div>
        '''
        if recommendations
        else ""
    }

        <footer class="footer">
            Strataforge AI Engine &bull; Automated Data Quality Pipeline &bull; Offline Self-Contained Document
        </footer>
    </div>

    <script>
        function filterColumns() {{
            const input = document.getElementById("columnSearch").value.toLowerCase();
            const rows = document.querySelectorAll("#columnsTable tbody tr");
            rows.forEach(row => {{
                const colName = row.querySelector(".col-name").textContent.toLowerCase();
                if (colName.includes(input)) {{
                    row.style.display = "";
                }} else {{
                    row.style.display = "none";
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(template)

    return str(out_file.resolve())
