"""Rich terminal components styled with the Strataforge brand aesthetic."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from strataforge.core.types import (
    AnomalyRecord,
    DatasetProfile,
    DriftMetric,
    QualityScoreSummary,
    Recommendation,
    SchemaChange,
    Severity,
    ValidationFinding,
)
from strataforge.ml.explainer import DISCLAIMER_TEXT, get_anomaly_score_interpretation


def get_severity_badge(severity: Severity | str) -> Text:
    """Render a text badge with appropriate styling."""
    sev = str(severity).upper()
    if sev in ("PASS", "SEVERITY.PASS"):
        return Text(" PASS ", style="badge.pass")
    elif sev in ("INFO", "SEVERITY.INFO"):
        return Text(" INFO ", style="badge.info")
    elif sev in ("WARNING", "WARN", "SEVERITY.WARNING"):
        return Text(" WARN ", style="badge.warn")
    else:
        return Text(" CRIT ", style="badge.crit")


def render_header(console: Console, action_title: str, target_path: str = "") -> None:
    """Render branded top banner reminiscent of a sleek terminal engine."""
    banner_text = Text()
    banner_text.append("STRATAFORGE AI", style="copper")
    banner_text.append("  |  ", style="muted")
    banner_text.append(action_title.upper(), style="text")
    if target_path:
        banner_text.append(f"  [{target_path}]", style="lavender")

    console.print()
    console.print(
        Panel(
            banner_text,
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 2),
        )
    )


def render_kpi_summary(
    console: Console, profile: DatasetProfile, score: QualityScoreSummary
) -> None:
    """Render compact KPI analytical metrics table."""
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="muted", expand=True)
    table.add_column("Health Score", justify="center")
    table.add_column("Observations", justify="right")
    table.add_column("Dimensions", justify="right")
    table.add_column("Duplicates", justify="right")
    table.add_column("Quality Status", justify="center")
    table.add_column("Checks Executed", justify="right")

    score_style = (
        "sage" if score.health_score >= 80 else ("gold" if score.health_score >= 60 else "coral")
    )
    score_display = Text(f"{score.grade} ({score.health_score:.1f}/100)", style=score_style)

    table.add_row(
        score_display,
        f"{profile.row_count:,}",
        f"{profile.column_count} cols",
        f"{profile.duplicate_count:,} ({profile.duplicate_percentage:.2f}%)",
        get_severity_badge(score.status),
        f"{score.passed_checks}/{score.total_checks} passed",
    )

    console.print(
        Panel(
            table,
            title="[copper]Analytical Summary[/copper]",
            title_align="left",
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 1),
        )
    )


def render_column_profile_table(console: Console, profile: DatasetProfile) -> None:
    """Render detailed column distributions table."""
    table = Table(box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper")
    table.add_column("Column", style="text")
    table.add_column("Type", justify="center")
    table.add_column("Missing / Nulls", justify="right")
    table.add_column("Distinct", justify="right")
    table.add_column("Distribution Summary", style="text")
    table.add_column("Sample Values", style="muted")

    for col_name, cp in profile.columns.items():
        type_txt = Text(cp.inferred_type.value, style="lavender")
        null_style = (
            "sage" if cp.null_count == 0 else ("coral" if cp.null_percentage > 10 else "gold")
        )
        null_txt = Text(f"{cp.null_count:,} ({cp.null_percentage:.1f}%)", style=null_style)
        distinct_txt = f"{cp.distinct_count:,} ({cp.distinct_percentage:.1f}%)"

        if cp.min_val is not None and cp.max_val is not None:
            if cp.mean_val is not None:
                mean_str = (
                    f"{cp.mean_val:.2f}" if isinstance(cp.mean_val, float) else str(cp.mean_val)
                )
                min_str = f"{cp.min_val:.2f}" if isinstance(cp.min_val, float) else str(cp.min_val)
                max_str = f"{cp.max_val:.2f}" if isinstance(cp.max_val, float) else str(cp.max_val)
                dist_desc = f"min: {min_str} | avg: {mean_str} | max: {max_str}"
            else:
                dist_desc = f"min: {cp.min_val} | max: {cp.max_val}"
        elif cp.top_categories:
            top_3 = [f"{k}:{v}" for k, v in list(cp.top_categories.items())[:3]]
            dist_desc = ", ".join(top_3)
        else:
            dist_desc = "—"

        samples = ", ".join(str(s) for s in cp.sample_values) if cp.sample_values else "—"

        table.add_row(col_name, type_txt, null_txt, distinct_txt, dist_desc, samples)

    console.print(table)


def render_schema_changes_table(console: Console, changes: list[SchemaChange]) -> None:
    """Render schema evolution and drift diff table."""
    if not changes:
        console.print(
            Panel(
                "[sage]No schema mutations detected. Schema is 100% consistent with baseline.[/sage]",
                box=box.ROUNDED,
                border_style="plum",
            )
        )
        return

    table = Table(box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper")
    table.add_column("Column", style="text")
    table.add_column("Change", justify="center")
    table.add_column("Severity", justify="center")
    table.add_column("Baseline Value", style="muted")
    table.add_column("Current Value", style="text")
    table.add_column("Description", style="text")

    for chg in changes:
        kind_style = "coral" if chg.change_kind in ("removed", "nullability_changed") else "gold"
        kind_badge = Text(f" {chg.change_kind.upper()} ", style=f"bold {kind_style} on #272039")
        sev_badge = (
            Text(" BREAKING ", style="badge.crit")
            if chg.change_type.value == "breaking"
            else Text(" NON-BREAK ", style="badge.warn")
        )

        table.add_row(
            chg.column_name,
            kind_badge,
            sev_badge,
            str(chg.old_value or "—"),
            str(chg.new_value or "—"),
            chg.description,
        )

    console.print(
        Panel(
            table,
            title="[copper]Schema Evolution Changes[/copper]",
            title_align="left",
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 0),
        )
    )


def render_drift_table(console: Console, metrics: list[DriftMetric]) -> None:
    """Render statistical drift comparison table."""
    if not metrics:
        return

    table = Table(box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper")
    table.add_column("Feature", style="text")
    table.add_column("Test Metric", style="lavender")
    table.add_column("Statistic", justify="right", style="text")
    table.add_column("p-value", justify="right", style="muted")
    table.add_column("Status", justify="center")
    table.add_column("Findings & Limitations", style="text")

    for dm in metrics:
        p_val_str = f"{dm.p_value:.2e}" if dm.p_value is not None else "—"
        finding_text = Text(dm.description)
        if dm.notes:
            finding_text.append(f"\n[Note: {dm.notes}]", style="muted")

        stat_display = Text(f"{dm.stat_value:.4f}")
        if dm.metric_name == "psi":
            if dm.stat_value >= 0.25:
                stat_display.append(" [===!]", style="coral")
            elif dm.stat_value >= 0.10:
                stat_display.append(" [==--]", style="gold")
            else:
                stat_display.append(" [=---]", style="sage")

        table.add_row(
            dm.column_name,
            dm.metric_display_name,
            stat_display,
            p_val_str,
            get_severity_badge(dm.severity),
            finding_text,
        )

    console.print(
        Panel(
            table,
            title="[copper]Statistical Distribution Drift[/copper]",
            title_align="left",
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 0),
        )
    )


def render_anomalies_table(
    console: Console, anomalies: list[AnomalyRecord], max_rows: int = 10
) -> None:
    """Render top flagged multivariate anomaly records."""
    if not anomalies:
        console.print(
            Panel(
                "[sage]No unusual multivariate anomalies detected.[/sage]",
                box=box.ROUNDED,
                border_style="plum",
            )
        )
        return

    table = Table(box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper")
    table.add_column("Row Index", justify="right", style="muted")
    table.add_column("Record ID", style="text")
    table.add_column("Anomaly Index", justify="center")
    table.add_column("Top Contributing Features", style="text")
    table.add_column("Diagnostic Summary", style="text")

    for rec in anomalies[:max_rows]:
        score_style = "coral" if rec.anomaly_score >= 0.70 else "gold"
        score_txt = Text(f"{rec.anomaly_score:.2f} ", style=f"bold {score_style}")
        if rec.anomaly_score >= 0.85:
            score_txt.append("[====]", style="coral")
        elif rec.anomaly_score >= 0.70:
            score_txt.append("[===.]", style="coral")
        elif rec.anomaly_score >= 0.50:
            score_txt.append("[==..]", style="gold")
        else:
            score_txt.append("[=...]", style="sage")

        feats = [
            f"{f.feature} (z: {f.z_score:+.1f}σ)"
            for f in rec.top_features[:2]
            if f.z_score is not None
        ]
        feats_str = ", ".join(feats) if feats else "Multivariate correlation shift"

        table.add_row(
            f"#{rec.row_index}",
            str(rec.record_id),
            score_txt,
            feats_str,
            rec.diagnostic_summary,
        )

    console.print(
        Panel(
            table,
            title="[copper]Multivariate Anomaly Findings (Isolation Forest)[/copper]",
            title_align="left",
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 0),
        )
    )


def render_anomaly_detail(console: Console, record: AnomalyRecord) -> None:
    """Render deep-dive diagnostic card for a single row anomaly inspection."""
    header = Text()
    header.append("DIAGNOSTIC EXPLANATION — RECORD ", style="copper")
    header.append(f"#{record.row_index} (ID: {record.record_id})\n", style="text")
    header.append("Anomaly Score: ", style="muted")
    header.append(f"{record.anomaly_score:.2f} / 1.00  ", style="coral")
    header.append(f"({get_anomaly_score_interpretation(record.anomaly_score)})\n", style="text")
    header.append("Raw Decision Function: ", style="muted")
    header.append(f"{record.raw_decision_score:+.4f}\n", style="text")

    console.print(header)

    feat_table = Table(
        box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper"
    )
    feat_table.add_column("Feature", style="text")
    feat_table.add_column("Observed Value", justify="right", style="text")
    feat_table.add_column("Baseline Mean ± Std", justify="right", style="muted")
    feat_table.add_column("Z-Score Deviation", justify="right")
    feat_table.add_column("Diagnostic Explanation", style="text")

    for f in record.top_features:
        z_style = (
            "coral"
            if (f.z_score and abs(f.z_score) >= 3.0)
            else ("gold" if (f.z_score and abs(f.z_score) >= 2.0) else "text")
        )
        z_txt = Text(f"{f.z_score:+.2f}σ" if f.z_score is not None else "—", style=z_style)
        mean_std = f"{f.baseline_mean} ± {f.baseline_std}" if f.baseline_mean is not None else "—"

        feat_table.add_row(
            f.feature,
            str(f.value),
            mean_std,
            z_txt,
            f.explanation,
        )

    console.print(feat_table)
    console.print(
        Panel(f"[muted]{DISCLAIMER_TEXT}[/muted]", box=box.ROUNDED, border_style="copper")
    )


def render_recommendations(console: Console, recommendations: list[Recommendation]) -> None:
    """Render intelligent remediation recommendations with copyable snippets."""
    if not recommendations:
        return

    console.print(Text("\nRECOMMENDED ACTIONS", style="copper"))

    for i, rec in enumerate(recommendations, 1):
        content = Text()
        content.append("Category: ", style="muted")
        content.append(f"{rec.category.upper()}   ", style="lavender")
        content.append("Impact: ", style="muted")
        content.append(f"{rec.impact_description}\n\n", style="text")

        content.append("Evidence: ", style="muted")
        content.append(f"{rec.evidence}\n", style="text")

        content.append("Proposed Remediation: ", style="muted")
        content.append(f"{rec.proposed_action}\n", style="text")

        panel_title = f"{i}. {rec.title}"

        console.print(
            Panel(
                content,
                title=f"[{'coral' if rec.severity == Severity.CRITICAL else 'gold'}]{panel_title}[/]",
                box=box.ROUNDED,
                border_style="plum",
                padding=(1, 2),
            )
        )
        if rec.code_snippet:
            syntax = Syntax(rec.code_snippet, "python", theme="monokai", line_numbers=False)
            console.print(
                Panel(
                    syntax, title="Remediation Snippet", box=box.HORIZONTALS, border_style="muted"
                )
            )


def render_validation_table(console: Console, findings: list[ValidationFinding]) -> None:
    """Render rule assertion outcomes."""
    table = Table(box=box.ROUNDED, border_style="plum", show_header=True, header_style="copper")
    table.add_column("Rule", style="text")
    table.add_column("Column", style="lavender")
    table.add_column("Status", justify="center")
    table.add_column("Observed Value", style="text")
    table.add_column("Expected Assertion", style="muted")
    table.add_column("Message", style="text")

    for f in findings:
        table.add_row(
            f.rule_name,
            f.column_name or "*",
            get_severity_badge(f.severity if not f.passed else Severity.PASS),
            str(f.actual_value),
            str(f.expected_value),
            f.message,
        )

    console.print(
        Panel(
            table,
            title="[copper]Quality Rule Assertions[/copper]",
            title_align="left",
            box=box.ROUNDED,
            border_style="plum",
            padding=(0, 0),
        )
    )
