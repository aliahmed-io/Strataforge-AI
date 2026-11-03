"""Standalone HTML quality report generator styled with Strataforge dark cockpit palette."""

from __future__ import annotations

import html
import json
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


def _generate_svg_gauge(score: float, color: str, grade: str) -> str:
    """Generate an inline circular SVG gauge for the health score."""
    radius = 42
    circumference = 2 * 3.1415926535 * radius
    stroke_dash = (max(0.0, min(100.0, score)) / 100.0) * circumference
    dash_array = f"{stroke_dash:.1f} {circumference:.1f}"

    return f"""
    <div class="radial-gauge-wrap">
        <svg viewBox="0 0 100 100" class="radial-gauge-svg">
            <circle cx="50" cy="50" r="{radius}" class="gauge-bg" />
            <circle cx="50" cy="50" r="{radius}" class="gauge-fill"
                    style="stroke: {color}; stroke-dasharray: {dash_array};" />
        </svg>
        <div class="gauge-center-text">
            <span class="gauge-grade" style="color: {color};">{grade}</span>
            <span class="gauge-score">{score:.1f}</span>
        </div>
    </div>
    """


def _generate_mini_progress(pct: float, color_cls: str = "pass") -> str:
    """Generate a clean CSS-based mini progress bar."""
    capped = max(0.0, min(100.0, pct))
    return f"""
    <div class="mini-bar-track">
        <div class="mini-bar-fill bar-{color_cls}" style="width: {capped:.1f}%;"></div>
    </div>
    """


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

    row_count_str = f"{profile.row_count:,}"
    col_count_str = str(profile.column_count)
    memory_mb = f"{profile.memory_bytes / (1024 * 1024):.2f} MB"
    dup_str = f"{profile.duplicate_count:,} ({profile.duplicate_percentage:.2f}%)"
    health_score = score_summary.health_score
    grade = score_summary.grade

    # Dynamic status colors
    if health_score >= 80.0:
        score_color = "#91C8A8"  # Sage
        pulse_color = "#91C8A8"
        status_label = "HEALTHY"
    elif health_score >= 60.0:
        score_color = "#ECC06C"  # Gold
        pulse_color = "#ECC06C"
        status_label = "DEGRADED"
    else:
        score_color = "#E77770"  # Coral
        pulse_color = "#E77770"
        status_label = "CRITICAL FAILURES"

    gauge_svg = _generate_svg_gauge(health_score, score_color, grade)

    # 1. Render Column Profiles
    column_rows_html = []
    for col_name, cp in profile.columns.items():
        type_badge = f'<span class="badge badge-type">{cp.inferred_type.value}</span>'

        if cp.null_count == 0:
            null_badge_class = "badge-pass"
            bar_cls = "pass"
        elif cp.null_percentage > 10.0:
            null_badge_class = "badge-crit"
            bar_cls = "crit"
        else:
            null_badge_class = "badge-warn"
            bar_cls = "warn"

        null_bar = _generate_mini_progress(cp.null_percentage, bar_cls)
        null_cell = f"""
        <div class="metric-cell-group">
            <span class="badge {null_badge_class}">{cp.null_count:,} ({cp.null_percentage:.1f}%)</span>
            {null_bar if cp.null_percentage > 0 else ""}
        </div>
        """

        distinct_cell = f"""
        <div class="metric-cell-group">
            <span>{cp.distinct_count:,}</span>
            <span class="text-muted text-small">({cp.distinct_percentage:.1f}%)</span>
        </div>
        """

        if cp.min_val is not None and cp.max_val is not None:
            stats_cell = f"""
            <div class="cell-stats">
                <span class="stat-pill"><span class="stat-lbl">min</span> {cp.min_val}</span>
                <span class="stat-pill"><span class="stat-lbl">avg</span> {cp.mean_val}</span>
                <span class="stat-pill"><span class="stat-lbl">max</span> {cp.max_val}</span>
            </div>
            """
        elif cp.top_categories:
            top_3 = list(cp.top_categories.items())[:3]
            pills = "".join(
                f'<span class="category-pill">{html.escape(k)}: {v}</span>' for k, v in top_3
            )
            stats_cell = f'<div class="cell-categories">{pills}</div>'
        else:
            stats_cell = '<span class="text-muted">—</span>'

        sample_pills = (
            "".join(
                f'<span class="sample-tag font-mono">{html.escape(str(s))}</span>'
                for s in cp.sample_values[:3]
            )
            or '<span class="text-muted">—</span>'
        )

        column_rows_html.append(f"""
        <tr class="col-row" data-name="{html.escape(col_name.lower())}">
            <td class="col-name font-mono"><strong>{html.escape(col_name)}</strong></td>
            <td>{type_badge}</td>
            <td>{null_cell}</td>
            <td class="font-mono">{distinct_cell}</td>
            <td>{stats_cell}</td>
            <td><div class="sample-wrap">{sample_pills}</div></td>
        </tr>
        """)

    # 2. Render Schema Changes
    schema_rows_html = []
    for sc in schema_changes:
        badge_cls = "badge-crit" if sc.change_type.value == "breaking" else "badge-warn"
        kind_style = (
            "badge-crit" if sc.change_kind in ("removed", "nullability_changed") else "badge-warn"
        )
        schema_rows_html.append(f"""
        <tr class="issue-row severity-{sc.change_type.value}">
            <td class="font-mono"><strong>{html.escape(sc.column_name)}</strong></td>
            <td><span class="badge {kind_style}">{sc.change_kind.upper()}</span></td>
            <td><span class="badge {badge_cls}">{sc.change_type.value.upper()}</span></td>
            <td class="font-mono text-muted">{html.escape(str(sc.old_value or "—"))}</td>
            <td class="font-mono">{html.escape(str(sc.new_value or "—"))}</td>
            <td class="text-secondary">{html.escape(sc.description)}</td>
        </tr>
        """)

    # 3. Render Drift Metrics
    drift_rows_html = []
    for dm in drift_metrics:
        badge_cls = (
            "badge-pass"
            if dm.severity == "PASS"
            else ("badge-warn" if dm.severity == "WARNING" else "badge-crit")
        )
        p_val_str = f"{dm.p_value:.2e}" if dm.p_value is not None else "—"
        notes_str = (
            f'<div class="text-small text-muted note-box">{html.escape(dm.notes)}</div>'
            if dm.notes
            else ""
        )

        # Visual indicator bar for PSI
        if dm.metric_name == "psi":
            psi_fill = min(100.0, (dm.stat_value / 0.50) * 100.0)
            psi_bar_cls = (
                "crit" if dm.stat_value >= 0.25 else ("warn" if dm.stat_value >= 0.10 else "pass")
            )
            stat_cell = f"""
            <div class="metric-cell-group">
                <span class="font-mono font-bold">{dm.stat_value:.4f}</span>
                <div class="mini-bar-track" title="PSI vs 0.25 critical threshold">
                    <div class="mini-bar-fill bar-{psi_bar_cls}" style="width: {psi_fill:.1f}%;"></div>
                </div>
            </div>
            """
        else:
            stat_cell = f'<span class="font-mono font-bold">{dm.stat_value:.4f}</span>'

        drift_rows_html.append(f"""
        <tr class="issue-row severity-{dm.severity.value.lower()}">
            <td class="font-mono"><strong>{html.escape(dm.column_name)}</strong></td>
            <td><span class="badge badge-type">{html.escape(dm.metric_display_name)}</span></td>
            <td>{stat_cell}</td>
            <td class="font-mono text-muted">{p_val_str}</td>
            <td><span class="badge {badge_cls}">{dm.severity.value}</span></td>
            <td>{html.escape(dm.description)}{notes_str}</td>
        </tr>
        """)

    # 4. Render Anomalies
    anomalies_rows_html = []
    for anom in anomalies[:30]:
        top_feats = []
        for feat in anom.top_features[:3]:
            z_str = f"{feat.z_score:+.1f}σ" if feat.z_score is not None else "null/divergent"
            z_cls = "text-coral" if (feat.z_score and abs(feat.z_score) >= 3.0) else "text-gold"
            top_feats.append(
                f'<span class="anom-feat-pill font-mono">{html.escape(feat.feature)} '
                f'<span class="{z_cls}">({z_str})</span></span>'
            )
        top_feats_html = "<div class='anom-tags-wrap'>" + "".join(top_feats) + "</div>"

        score_cls = "badge-crit" if anom.anomaly_score >= 0.70 else "badge-warn"
        score_bar_cls = "crit" if anom.anomaly_score >= 0.70 else "warn"
        score_pct = anom.anomaly_score * 100.0

        # Construct JSON for row inspection
        row_json_str = html.escape(json.dumps(anom.row_values, indent=2))

        anomalies_rows_html.append(f"""
        <tr class="issue-row anom-row severity-critical" onclick="toggleRowDetails('{anom.row_index}')">
            <td class="font-mono text-muted">#{anom.row_index}</td>
            <td class="font-mono font-bold">{html.escape(str(anom.record_id))}</td>
            <td>
                <div class="metric-cell-group">
                    <span class="badge {score_cls}">{anom.anomaly_score:.2f}</span>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill bar-{score_bar_cls}" style="width: {score_pct:.1f}%;"></div>
                    </div>
                </div>
            </td>
            <td>{top_feats_html}</td>
            <td class="text-secondary text-small">{html.escape(anom.diagnostic_summary)}</td>
        </tr>
        <tr id="anom-detail-{anom.row_index}" class="anom-detail-row" style="display: none;">
            <td colspan="5">
                <div class="anom-expanded-card">
                    <div class="expanded-header font-mono">Row #{anom.row_index} Payload & Feature Attributions</div>
                    <pre class="json-box"><code>{row_json_str}</code></pre>
                </div>
            </td>
        </tr>
        """)

    # 5. Render Validation Findings
    validation_rows_html = []
    for vf in validation_findings:
        badge_cls = (
            "badge-pass"
            if vf.passed
            else ("badge-crit" if vf.severity == "CRITICAL" else "badge-warn")
        )
        status_txt = "PASS" if vf.passed else vf.severity.value
        sev_class = "pass" if vf.passed else vf.severity.value.lower()

        validation_rows_html.append(f"""
        <tr class="issue-row severity-{sev_class}">
            <td class="font-mono"><strong>{html.escape(vf.rule_name)}</strong></td>
            <td class="font-mono text-muted">{html.escape(str(vf.column_name or "*"))}</td>
            <td><span class="badge {badge_cls}">{status_txt}</span></td>
            <td class="font-mono text-small">{html.escape(str(vf.actual_value))}</td>
            <td class="font-mono text-small text-muted">{html.escape(str(vf.expected_value))}</td>
            <td class="text-secondary">{html.escape(vf.message)}</td>
        </tr>
        """)

    # 6. Render Recommendations
    recs_cards_html = []
    for idx, rec in enumerate(recommendations):
        badge_cls = (
            "badge-crit"
            if rec.severity == "CRITICAL"
            else ("badge-warn" if rec.severity == "WARNING" else "badge-pass")
        )
        code_block = ""
        if rec.code_snippet:
            snippet_id = f"snippet-{idx}"
            code_block = f"""
            <div class="code-box">
                <div class="code-header">
                    <span class="font-mono text-copper">Remediation Snippet (Python / Pandas)</span>
                    <button class="copy-btn" onclick="copySnippet('{snippet_id}', this)">Copy</button>
                </div>
                <pre><code id="{snippet_id}">{html.escape(rec.code_snippet)}</code></pre>
            </div>
            """

        recs_cards_html.append(f"""
        <div class="rec-card severity-{rec.severity.value.lower()}">
            <div class="rec-header">
                <div class="rec-title-group">
                    <span class="badge {badge_cls}">{rec.severity.value}</span>
                    <span class="badge badge-type">{rec.category.upper()}</span>
                    <h3 class="rec-title">{html.escape(rec.title)}</h3>
                </div>
            </div>
            <div class="rec-body">
                <div class="rec-field">
                    <span class="field-label">Evidence:</span>
                    <span class="field-value">{html.escape(rec.evidence)}</span>
                </div>
                <div class="rec-field">
                    <span class="field-label">Proposed Action:</span>
                    <span class="field-value">{html.escape(rec.proposed_action)}</span>
                </div>
                <div class="rec-field text-muted text-small">
                    <span class="field-label">Impact:</span>
                    <span class="field-value">{html.escape(rec.impact_description)}</span>
                </div>
                {code_block}
            </div>
        </div>
        """)

    # Assemble HTML document
    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strataforge AI — Data Quality Cockpit</title>
    <style>
        :root {{
            --bg-base: #110F17;
            --bg-surface: #191624;
            --bg-elevated: #231C33;
            --bg-card: #1D182B;
            --border-subtle: #302645;
            --border-highlight: #463763;
            --accent-copper: #E9AC74;
            --accent-copper-glow: rgba(233, 172, 116, 0.25);
            --color-sage: #91C8A8;
            --color-coral: #E77770;
            --color-gold: #ECC06C;
            --color-lavender: #B892FF;
            --text-primary: #F4F1F8;
            --text-secondary: #C7BFD4;
            --text-muted: #847999;
            --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 13.5px;
            line-height: 1.5;
            padding: 24px 32px 64px 32px;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{ max-width: 1480px; margin: 0 auto; }}

        /* Top Navigation Bar */
        .top-nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        }}
        .brand-cluster {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .logo-mark {{
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #E9AC74 0%, #B892FF 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            color: #110F17;
            font-size: 18px;
            box-shadow: 0 0 16px var(--accent-copper-glow);
        }}
        .brand-title {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand-tag {{
            font-size: 11px;
            background: rgba(233, 172, 116, 0.12);
            color: var(--accent-copper);
            border: 1px solid rgba(233, 172, 116, 0.3);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-weight: 600;
        }}
        .meta-line {{
            color: var(--text-muted);
            font-size: 12px;
            font-family: var(--font-mono);
            margin-top: 2px;
        }}
        .status-pill {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-subtle);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            font-family: var(--font-mono);
        }}
        .status-pulse {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {pulse_color};
            box-shadow: 0 0 8px {pulse_color};
        }}
        .btn-group {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: var(--font-sans);
        }}
        .btn:hover {{
            background: var(--border-highlight);
            color: var(--text-primary);
            border-color: var(--accent-copper);
        }}

        /* Executive Telemetry Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: 240px repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px 24px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            transition: border-color 0.2s;
        }}
        .kpi-card:hover {{ border-color: var(--border-highlight); }}
        .kpi-score-card {{
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 16px 20px;
        }}
        .radial-gauge-wrap {{
            position: relative;
            width: 86px;
            height: 86px;
            flex-shrink: 0;
        }}
        .radial-gauge-svg {{
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }}
        .gauge-bg {{
            fill: none;
            stroke: #2A213D;
            stroke-width: 8;
        }}
        .gauge-fill {{
            fill: none;
            stroke-width: 8;
            stroke-linecap: round;
            transition: stroke-dasharray 1s ease;
        }}
        .gauge-center-text {{
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }}
        .gauge-grade {{
            font-size: 24px;
            font-weight: 800;
            font-family: var(--font-mono);
        }}
        .gauge-score {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
            font-family: var(--font-mono);
        }}
        .kpi-label {{
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            font-family: var(--font-mono);
            color: var(--text-primary);
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 6px;
        }}

        /* Interactive Controls Bar */
        .controls-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 24px;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .filter-pills {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .filter-chip {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .filter-chip.active, .filter-chip:hover {{
            background: var(--accent-copper);
            color: #110F17;
            border-color: var(--accent-copper);
        }}
        .search-box-wrap {{
            position: relative;
            flex: 1;
            max-width: 340px;
        }}
        .search-input {{
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 8px 14px 8px 34px;
            border-radius: 6px;
            font-family: var(--font-mono);
            font-size: 12px;
            transition: border-color 0.2s;
        }}
        .search-input:focus {{
            outline: none;
            border-color: var(--accent-copper);
            box-shadow: 0 0 8px var(--accent-copper-glow);
        }}
        .search-icon {{
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 13px;
        }}

        /* Section Headings */
        .section-header {{
            font-size: 17px;
            font-weight: 700;
            color: var(--text-primary);
            margin: 32px 0 14px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-count {{
            font-size: 12px;
            background: var(--bg-surface);
            color: var(--accent-copper);
            border: 1px solid var(--border-subtle);
            padding: 2px 8px;
            border-radius: 12px;
            font-family: var(--font-mono);
        }}

        /* Tables & Wrappers */
        .table-wrap {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            overflow-x: auto;
            margin-bottom: 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            font-size: 11.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-subtle);
            white-space: nowrap;
        }}
        td {{
            padding: 12px 18px;
            border-bottom: 1px solid rgba(48, 38, 69, 0.4);
            vertical-align: middle;
        }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

        /* Badges & Metrics */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            font-family: var(--font-mono);
            text-transform: uppercase;
            white-space: nowrap;
        }}
        .badge-pass {{ background: rgba(145, 200, 168, 0.12); color: var(--color-sage); border: 1px solid rgba(145, 200, 168, 0.3); }}
        .badge-warn {{ background: rgba(236, 192, 108, 0.12); color: var(--color-gold); border: 1px solid rgba(236, 192, 108, 0.3); }}
        .badge-crit {{ background: rgba(231, 119, 112, 0.12); color: var(--color-coral); border: 1px solid rgba(231, 119, 112, 0.3); }}
        .badge-type {{ background: rgba(184, 146, 255, 0.10); color: var(--color-lavender); border: 1px solid rgba(184, 146, 255, 0.25); }}

        /* Mini progress bar */
        .metric-cell-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .mini-bar-track {{
            width: 50px;
            height: 5px;
            background: #231C33;
            border-radius: 3px;
            overflow: hidden;
            flex-shrink: 0;
        }}
        .mini-bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        .bar-pass {{ background: var(--color-sage); }}
        .bar-warn {{ background: var(--color-gold); }}
        .bar-crit {{ background: var(--color-coral); }}

        /* Stats & Tags */
        .cell-stats {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .stat-pill {{
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--border-subtle);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-family: var(--font-mono);
        }}
        .stat-lbl {{ color: var(--text-muted); font-size: 10px; margin-right: 2px; }}
        .category-pill {{
            background: rgba(184, 146, 255, 0.08);
            border: 1px solid rgba(184, 146, 255, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-family: var(--font-mono);
            color: var(--text-secondary);
        }}
        .sample-wrap {{ display: flex; gap: 6px; flex-wrap: wrap; }}
        .sample-tag {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-subtle);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            color: var(--text-secondary);
        }}
        .anom-tags-wrap {{ display: flex; gap: 6px; flex-wrap: wrap; }}
        .anom-feat-pill {{
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border-subtle);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }}

        /* Interactive Anomaly Detail Rows */
        .anom-row {{ cursor: pointer; }}
        .anom-row:hover {{ background: rgba(233, 172, 116, 0.04) !important; }}
        .anom-expanded-card {{
            background: #13101C;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 14px 18px;
            margin: 8px 0;
        }}
        .expanded-header {{
            font-size: 11.5px;
            color: var(--accent-copper);
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .json-box {{
            background: #0D0A14;
            border: 1px solid var(--border-subtle);
            padding: 10px;
            border-radius: 4px;
            font-size: 11.5px;
            color: #CFC8DD;
            max-height: 200px;
            overflow-y: auto;
        }}

        /* Recommendations Cards */
        .rec-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
            margin-bottom: 32px;
        }}
        .rec-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-left: 4px solid var(--border-subtle);
            border-radius: 10px;
            padding: 20px 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            transition: all 0.2s ease;
        }}
        .rec-card.severity-critical {{ border-left-color: var(--color-coral); }}
        .rec-card.severity-warning {{ border-left-color: var(--color-gold); }}
        .rec-card.severity-pass {{ border-left-color: var(--color-sage); }}
        .rec-title-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
        }}
        .rec-title {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .rec-field {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        .field-label {{
            font-weight: 700;
            color: var(--text-muted);
            margin-right: 6px;
        }}
        .field-value {{ color: var(--text-secondary); }}
        .code-box {{
            background: #110E1A;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 14px;
            position: relative;
        }}
        .code-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            margin-bottom: 8px;
        }}
        .copy-btn {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            font-family: var(--font-mono);
            transition: all 0.15s;
        }}
        .copy-btn:hover {{
            background: var(--accent-copper);
            color: #110F17;
        }}
        pre {{ margin: 0; overflow-x: auto; font-family: var(--font-mono); font-size: 12px; color: #E7E2F0; }}

        /* Disclaimers & Notes */
        .disclaimer-banner {{
            background: rgba(233, 172, 116, 0.06);
            border: 1px solid rgba(233, 172, 116, 0.25);
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 16px;
            font-size: 12px;
            color: var(--text-secondary);
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .note-box {{ margin-top: 4px; font-style: italic; }}

        /* Utility */
        .font-mono {{ font-family: var(--font-mono); }}
        .font-bold {{ font-weight: 700; }}
        .text-copper {{ color: var(--accent-copper); }}
        .text-sage {{ color: var(--color-sage); }}
        .text-coral {{ color: var(--color-coral); }}
        .text-gold {{ color: var(--color-gold); }}
        .text-muted {{ color: var(--text-muted); }}
        .text-secondary {{ color: var(--text-secondary); }}
        .text-small {{ font-size: 11.5px; }}

        /* Footer */
        .footer {{
            border-top: 1px solid var(--border-subtle);
            padding-top: 24px;
            margin-top: 48px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}

        @media (max-width: 900px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
            .top-nav {{ flex-direction: column; align-items: flex-start; gap: 14px; }}
            .controls-bar {{ flex-direction: column; align-items: stretch; }}
            .search-box-wrap {{ max-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Navigation -->
        <nav class="top-nav">
            <div class="brand-cluster">
                <div class="logo-mark">S</div>
                <div>
                    <div class="brand-title">
                        <span>STRATAFORGE AI</span>
                        <span class="brand-tag">v0.1.0</span>
                    </div>
                    <div class="meta-line">
                        {html.escape(profile.dataset_path)} &bull; {
        profile.file_format.upper()
    } &bull; Generated {profile.created_at[:19]}
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 14px;">
                <div class="status-pill">
                    <div class="status-pulse"></div>
                    <span>{status_label}</span>
                </div>
                <div class="btn-group">
                    <button class="btn" onclick="window.print()">Print / PDF</button>
                    <button class="btn" onclick="exportDiagnosticJson()">Export JSON</button>
                </div>
            </div>
        </nav>

        <!-- Executive Telemetry Grid -->
        <section class="kpi-grid">
            <div class="kpi-card kpi-score-card">
                {gauge_svg}
                <div>
                    <div class="kpi-label">Health Index</div>
                    <div style="font-size: 13px; color: {
        score_color
    }; font-weight: 700;">HEALTH GRADE: {grade}</div>
                    <div class="kpi-sub" style="font-size: 11px;">{score_summary.status.value}</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Observations & Footprint</div>
                <div class="kpi-value">{row_count_str}</div>
                <div class="kpi-sub">{col_count_str} dimensions &bull; {memory_mb}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duplicate Records</div>
                <div class="kpi-value">{dup_str}</div>
                <div class="kpi-sub">Exact full-row matching</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Diagnostic Checks</div>
                <div class="kpi-value">{score_summary.total_checks}</div>
                <div class="kpi-sub">{
        score_summary.passed_checks
    } passed &bull; <strong style="color: {score_color};">{
        score_summary.critical_checks
    } critical</strong></div>
            </div>
        </section>

        <!-- Interactive Controls Bar -->
        <div class="controls-bar">
            <div class="filter-pills">
                <span style="font-size: 12px; color: var(--text-muted); font-weight: 600; margin-right: 4px;">FILTER:</span>
                <button class="filter-chip active" onclick="applyFilter('all', this)">All Checks</button>
                <button class="filter-chip" onclick="applyFilter('critical', this)">Critical Only</button>
                <button class="filter-chip" onclick="applyFilter('warning', this)">Warnings Only</button>
                <button class="filter-chip" onclick="applyFilter('pass', this)">Passed Only</button>
            </div>
            <div class="search-box-wrap">
                <span class="search-icon">&#x26B2;</span>
                <input type="text" id="globalSearch" class="search-input" placeholder="Search columns, rules, or findings..." onkeyup="filterTables()">
            </div>
        </div>

        <!-- 1. Column Profiles Section -->
        <div class="section-header">
            <span>Inferred Schema & Column Distributions</span>
            <span class="section-count">{len(profile.columns)} columns</span>
        </div>
        <div class="table-wrap">
            <table id="columnsTable">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Inferred Type</th>
                        <th>Missing / Nulls</th>
                        <th>Distinct Count</th>
                        <th>Distribution (Min / Avg / Max)</th>
                        <th>Sample Values</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(column_rows_html)}
                </tbody>
            </table>
        </div>

        <!-- 2. Schema Changes Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Schema Evolution Mutations</span>
            <span class="section-count">{len(schema_changes)}</span>
        </div>
        <div class="table-wrap">
            <table id="schemaTable">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Change Kind</th>
                        <th>Classification</th>
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

        <!-- 3. Statistical Drift Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Statistical Distribution Drift</span>
            <span class="section-count">{len(drift_metrics)} tests</span>
        </div>
        <div class="table-wrap">
            <table id="driftTable">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Test Metric</th>
                        <th>Statistic / PSI</th>
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

        <!-- 4. Multivariate Anomalies Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Multivariate Isolation Forest Anomalies (Top Flagged)</span>
            <span class="section-count">{len(anomalies)} observations</span>
        </div>
        <div class="disclaimer-banner">
            <span style="font-size: 16px;">&#x2139;</span>
            <span><strong>Diagnostic Note:</strong> Multivariate anomaly scores reflect geometric distance from empirical reference clusters. An anomalous record is a statistically unusual multivariate combination, not inherently a corrupted record. Click any row to inspect raw payload values.</span>
        </div>
        <div class="table-wrap">
            <table id="anomaliesTable">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Record ID</th>
                        <th>Anomaly Index</th>
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

        <!-- 5. Quality Rules Validation Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Quality Rule Assertions</span>
            <span class="section-count">{len(validation_findings)} rules</span>
        </div>
        <div class="table-wrap">
            <table id="validationTable">
                <thead>
                    <tr>
                        <th>Rule Assertion</th>
                        <th>Target Column</th>
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

        <!-- 6. Recommendations Section if present -->
        {
        f'''
        <div class="section-header">
            <span>Intelligent Remediation Recommendations</span>
            <span class="section-count">{len(recommendations)} actions</span>
        </div>
        <div class="rec-grid">
            {"".join(recs_cards_html)}
        </div>
        '''
        if recommendations
        else ""
    }

        <footer class="footer">
            <div>STRATAFORGE AI &bull; Terminal-Native Data-Quality Cockpit</div>
            <div>Self-Contained Offline Artifact &bull; Zero CDN Dependencies</div>
        </footer>
    </div>

    <script>
        let currentFilter = 'all';

        function applyFilter(filterType, btn) {{
            currentFilter = filterType;
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            filterTables();
        }}

        function filterTables() {{
            const search = document.getElementById("globalSearch").value.toLowerCase();

            // Filter columns table
            document.querySelectorAll("#columnsTable tbody tr").forEach(row => {{
                const text = row.textContent.toLowerCase();
                const matchesSearch = text.includes(search);
                row.style.display = matchesSearch ? "" : "none";
            }});

            // Filter all issue rows across tables
            document.querySelectorAll(".issue-row").forEach(row => {{
                const text = row.textContent.toLowerCase();
                const matchesSearch = text.includes(search);
                const matchesFilter = (currentFilter === 'all') || row.classList.contains('severity-' + currentFilter);
                row.style.display = (matchesSearch && matchesFilter) ? "" : "none";
            }});
        }}

        function toggleRowDetails(rowIdx) {{
            const detailRow = document.getElementById('anom-detail-' + rowIdx);
            if (detailRow) {{
                detailRow.style.display = detailRow.style.display === 'none' ? '' : 'none';
            }}
        }}

        function copySnippet(snippetId, btn) {{
            const code = document.getElementById(snippetId).textContent;
            navigator.clipboard.writeText(code).then(() => {{
                const originalText = btn.textContent;
                btn.textContent = "Copied!";
                btn.style.background = "#91C8A8";
                btn.style.color = "#110F17";
                setTimeout(() => {{
                    btn.textContent = originalText;
                    btn.style.background = "";
                    btn.style.color = "";
                }}, 1800);
            }});
        }}

        function exportDiagnosticJson() {{
            const data = {{
                dataset: "{html.escape(profile.dataset_path)}",
                health_score: {health_score},
                grade: "{grade}",
                status: "{score_summary.status.value}",
                rows: {profile.row_count},
                columns: {profile.column_count},
                duplicates: {profile.duplicate_count},
                generated_at: "{profile.created_at}"
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'strataforge_diagnostic_summary.json';
            a.click();
            URL.revokeObjectURL(url);
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
