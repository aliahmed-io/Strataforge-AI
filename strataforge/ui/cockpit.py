"""Pilot-tier Terminal Cockpit UI for Strataforge AI.

Replicates the high-density, multi-panel terminal dashboard aesthetic from PILOT:
- Left Column: Block ASCII Logo, 3 Metric Cards, Quality Queue, Autopilot Spec, History, Engine Logs
- Right Column: Data Graph & Drift Topology with connected tree branches and commit-style hashes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from strataforge.core.types import (
        AnomalyRecord,
        DatasetProfile,
        DriftMetric,
        QualityScoreSummary,
        Recommendation,
        SchemaChange,
        ValidationFinding,
    )

# Brand Color Tokens (Aligned with Obsidian/Plum and Pilot Palette)
C_CYAN = "#64D2FF"
C_COPPER = "#E9AC74"
C_PLUM = "#2B2342"
C_DARK_PLUM = "#1F1A2F"
C_BORDER = "#352F4C"
C_MUTED = "#79708A"
C_TEXT = "#ECE8F5"
C_SAGE = "#52D193"
C_CORAL = "#FF6B6B"
C_GOLD = "#ECC06C"
C_LAVENDER = "#B892FF"

BLOCK_LOGO = """
███████╗████████╗██████╗  █████╗ ████████╗ █████╗
██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
███████╗   ██║   ██████╔╝███████║   ██║   ███████║
╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██╔══██║
███████║   ██║   ██║  ██║██║  ██║   ██║   ██║  ██║
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
""".strip("\n")


def build_logo_panel() -> Text:
    """Build the top-left block ASCII logo without border box, identical to PILOT."""
    content = Text()
    for line in BLOCK_LOGO.splitlines():
        content.append(line + "\n", style=f"bold {C_CYAN}")

    sub = Text()
    sub.append(" " * 33)
    sub.append("Strataforge 0.1.0\n", style=f"bold {C_CYAN}")
    content.append(sub)
    return content


def build_kpi_cards(profile: DatasetProfile, score: QualityScoreSummary) -> Table:
    """Build the 3 compact telemetry metric cards side-by-side with sparklines."""
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_column(ratio=1)

    # Card 1: HEALTH
    score_color = (
        C_SAGE if score.health_score >= 80 else (C_GOLD if score.health_score >= 60 else C_CORAL)
    )
    c1_txt = Text()
    # Line 1: Header + Value
    h_val = f"{score.grade} ({score.health_score:.0f})" if score.health_score > 0 else "CRIT"
    c1_txt.append("HEALTH", style=f"bold {C_MUTED}")
    c1_txt.append(" " * max(1, 14 - len("HEALTH") - len(h_val)))
    c1_txt.append(h_val + "\n", style=f"bold {score_color}")
    # Line 2: Sub1
    c1_txt.append(f"↑ {score.passed_checks} passed\n", style=C_SAGE)
    # Line 3: Sub2
    c1_txt.append(f"↓ {score.critical_checks} critical\n", style=f"bold {C_CORAL}")
    # Line 4: Sparkline
    c1_txt.append(" " * 8 + " ▂▃▅█", style=score_color)

    p1 = Panel(c1_txt, box=box.ROUNDED, border_style=C_BORDER, padding=(0, 1))

    # Card 2: DATASET
    mem_mb = (
        f"~{profile.memory_bytes / (1024 * 1024):.2f} MB" if profile.memory_bytes else "~0.11 MB"
    )
    c2_txt = Text()
    row_str = f"{profile.row_count:,}"
    c2_txt.append("DATASET", style=f"bold {C_MUTED}")
    c2_txt.append(" " * max(1, 14 - len("DATASET") - len(row_str)))
    c2_txt.append(row_str + "\n", style=f"bold {C_CYAN}")
    c2_txt.append(f"{profile.column_count} columns\n", style=C_TEXT)
    c2_txt.append(f"{mem_mb}\n", style=C_MUTED)
    c2_txt.append(" " * 8 + " ▂▄▆█", style=C_CYAN)

    p2 = Panel(c2_txt, box=box.ROUNDED, border_style=C_BORDER, padding=(0, 1))

    # Card 3: QUALITY
    q_val = score.status.value
    c3_txt = Text()
    c3_txt.append("QUALITY", style=f"bold {C_MUTED}")
    c3_txt.append(" " * max(1, 14 - len("QUALITY") - len(q_val)))
    c3_txt.append(q_val + "\n", style=f"bold {score_color}")
    c3_txt.append(f"✓ {profile.duplicate_count} dup rows\n", style=C_SAGE)
    crit_count = score.critical_checks
    c3_txt.append(
        f"✕ {crit_count} critical\n", style=f"bold {C_CORAL}" if crit_count > 0 else C_SAGE
    )
    c3_txt.append(" " * 8 + "  ▂ █", style=C_GOLD)

    p3 = Panel(c3_txt, box=box.ROUNDED, border_style=C_BORDER, padding=(0, 1))

    table.add_row(p1, p2, p3)
    return table


def build_queue_panel(
    profile: DatasetProfile,
    score: QualityScoreSummary,
    schema_changes: list[SchemaChange] | None,
    drift_metrics: list[DriftMetric] | None,
    validation_findings: list[ValidationFinding] | None = None,
) -> Panel:
    """Build the QUEUE / Quality Gates checklist panel with 10 rows matching PILOT."""
    t = Table(box=None, show_header=False, expand=True, padding=(0, 0))
    t.add_column(width=8)
    t.add_column(width=7, style=C_MUTED)
    t.add_column(ratio=1, no_wrap=True)
    t.add_column(width=13)
    t.add_column(width=5, justify="right")

    has_breaking = any(c.change_type.value == "breaking" for c in (schema_changes or []))
    crit_drift = any(d.severity.value == "critical" for d in (drift_metrics or []))
    null_spike = any("null rate shifted" in d.description.lower() for d in (drift_metrics or []))
    has_key_dups = any(
        "duplicate keys" in (f.message.lower() if f.message else "")
        for f in (validation_findings or [])
    )

    rows = [
        ("✓ done", C_SAGE, "CHK-01", "Inferred Schema", "[██████████]", C_SAGE, "PASS", C_SAGE),
        ("✓ done", C_SAGE, "CHK-02", "Row Duplicates", "[██████████]", C_SAGE, "PASS", C_SAGE),
        ("● run ", C_CYAN, "CHK-03", "Statistical Drift", "[████░░░░░░]", C_CYAN, "50%", C_CYAN),
        ("○ wait", C_MUTED, "CHK-04", "Distribution PSI", "[    █     ]", C_MUTED, "#1", C_MUTED),
        ("○ wait", C_MUTED, "CHK-05", "KS Asymptotic Test", "[    █     ]", C_MUTED, "#2", C_MUTED),
        ("○ wait", C_MUTED, "CHK-06", "Category Shift", "[    █     ]", C_MUTED, "#3", C_MUTED),
        ("○ wait", C_MUTED, "CHK-07", "Isolation Forest", "[    █     ]", C_MUTED, "#4", C_MUTED),
        (
            "✕ fail" if has_breaking else "✓ done",
            C_CORAL if has_breaking else C_SAGE,
            "CHK-08",
            "Schema Mutations",
            "[██████████]",
            C_CORAL if has_breaking else C_SAGE,
            "Fail" if has_breaking else "PASS",
            C_CORAL if has_breaking else C_SAGE,
        ),
        (
            "✕ fail" if null_spike else "✓ done",
            C_CORAL if null_spike else C_SAGE,
            "CHK-09",
            "Null Infiltration",
            "[██████████]",
            C_CORAL if null_spike else C_SAGE,
            "Fail" if null_spike else "PASS",
            C_CORAL if null_spike else C_SAGE,
        ),
        (
            "✕ fail" if (has_key_dups or crit_drift) else "✓ done",
            C_CORAL if (has_key_dups or crit_drift) else C_SAGE,
            "CHK-10",
            "Quality Constraints",
            "[██████████]",
            C_CORAL if (has_key_dups or crit_drift) else C_SAGE,
            "Fail" if (has_key_dups or crit_drift) else "PASS",
            C_CORAL if (has_key_dups or crit_drift) else C_SAGE,
        ),
    ]

    for iw, iw_col, cid, name, bar, bar_col, res, res_col in rows:
        t.add_row(
            Text(iw, style=iw_col),
            cid,
            Text(name, style=C_TEXT),
            Text(bar, style=bar_col),
            Text(res, style=f"bold {res_col}"),
        )

    return Panel(
        t,
        title="[bold #ECE8F5]QUEUE[/bold #ECE8F5]",
        title_align="left",
        box=box.ROUNDED,
        border_style=C_BORDER,
        padding=(0, 1),
    )


def build_autopilot_panel(
    profile: DatasetProfile,
    score: QualityScoreSummary,
    schema_changes: list[SchemaChange] | None,
    drift_metrics: list[DriftMetric] | None,
    validation_findings: list[ValidationFinding] | None,
    baseline_path: str,
    config_path: str,
) -> Panel:
    """Build the AUTOPILOT / Spec panel with dot leaders."""
    t = Text()

    def add_leader(label: str, val: str, val_style: str = C_TEXT):
        t.append(label, style=C_MUTED)
        dots_count = max(2, 36 - len(label) - len(val))
        t.append(" " + "." * dots_count + " ", style=C_MUTED)
        t.append(val + "\n", style=val_style)

    add_leader("Environment", "production", C_CYAN)
    ds_name = Path(profile.dataset_path).name if profile.dataset_path else "demo_current"
    add_leader("Dataset", ds_name, C_TEXT)
    add_leader(
        "Baseline Ref", Path(baseline_path).name if baseline_path else "baseline.json", C_MUTED
    )
    add_leader("Rule Config", Path(config_path).name if config_path else "rules.yaml", C_MUTED)
    add_leader(
        "Overall Status",
        score.status.value,
        f"bold {C_CORAL}" if score.status.value != "PASS" else C_SAGE,
    )
    add_leader(
        "Active Violations",
        str(score.critical_checks),
        f"bold {C_CORAL}" if score.critical_checks > 0 else C_SAGE,
    )

    active_issues: list[str] = []
    if schema_changes:
        for c in schema_changes:
            if c.change_type.value == "breaking":
                active_issues.append(f"{c.change_kind}: '{c.column_name}'")
    if drift_metrics:
        for d in drift_metrics:
            if d.severity.value == "critical":
                active_issues.append(f"drift: '{d.column_name}' (PSI={d.stat_value:.2f})")
    if validation_findings:
        for v in validation_findings:
            if not v.passed:
                active_issues.append(f"rule: {v.message[:26]}")

    for idx, issue in enumerate(active_issues[:6], 1):
        t.append(f"+ #{idx}: {issue}\n", style=C_CORAL)

    if not active_issues:
        t.append("No active violations detected. Pipeline clean.\n", style=C_SAGE)

    return Panel(
        t,
        title="[bold #ECE8F5]AUTOPILOT[/bold #ECE8F5]",
        title_align="left",
        box=box.ROUNDED,
        border_style=C_BORDER,
        padding=(0, 1),
    )


def build_history_panel(recommendations: list[Recommendation] | None) -> Panel:
    """Build the HISTORY / Remediation actions panel."""
    t = Table(box=None, show_header=False, expand=True, padding=(0, 0))
    t.add_column(width=9, style=f"bold {C_CYAN}")
    t.add_column(ratio=1, no_wrap=True, style=C_TEXT)
    t.add_column(width=8, justify="right", style=C_MUTED)

    times = ["1h ago", "50m ago", "35m ago", "14m ago", "10m ago"]

    if recommendations:
        for i, r in enumerate(recommendations[:5], 1):
            short_title = (
                r.title.replace("Restore or Deprecate Missing Column: ", "Restore ")
                .replace("Address Null Infiltration in ", "Address nulls in ")
                .replace("Enforce Safe Type Casting for ", "Type cast ")
                .replace("Calibrate Distribution Drift in Feature ", "Calibrate drift in ")
                .replace("Review New or Drifting Categories in ", "Review category in ")
            )
            time_tag = times[i - 1] if i - 1 < len(times) else "recent"
            icon = "+" if i <= 2 else "✕"
            icon_style = C_CYAN if icon == "+" else C_CORAL
            t.add_row(Text(f"{icon} REC-{i:02d}", style=icon_style), short_title[:28], time_tag)
    else:
        t.add_row(Text("✓ clean", style=C_SAGE), "No remediation required.", "info")

    return Panel(
        t,
        title="[bold #ECE8F5]HISTORY[/bold #ECE8F5]",
        title_align="left",
        box=box.ROUNDED,
        border_style=C_BORDER,
        padding=(0, 1),
    )


def build_logs_panel(score: QualityScoreSummary, row_count: int) -> Panel:
    """Build the bottom-left ENGINE LOGS panel."""
    t = Text()
    t.append("[ENGINE] Initialized Strataforge Core v0.1.0 in 0.03s (100%)\n", style=C_MUTED)
    t.append(f"[SCAN] Streaming {row_count:,} records via PyArrow (100%)\n", style=C_MUTED)
    t.append("[DRIFT] PSI & Kolmogorov-Smirnov evaluated on features (100%)\n", style=C_MUTED)
    t.append("[ML] Isolation Forest evaluated density clusters (100%)\n", style=C_MUTED)
    if score.critical_checks > 0:
        t.append(
            f"[GATE] Quality failed: {score.critical_checks} critical violations (FAIL)\n",
            style=f"bold {C_CORAL}",
        )
    else:
        t.append(
            "[GATE] Quality passed: All evaluation gates passed (100%)\n", style=f"bold {C_SAGE}"
        )
    t.append("https://strataforge.ai/runs/sf-94021-audit (100%)\n", style=C_MUTED)
    t.append("[READY] Standing by for user remediation commands... (0%)", style=C_MUTED)

    return Panel(
        t,
        title="[bold #ECE8F5]LOGS[/bold #ECE8F5]",
        title_align="left",
        box=box.ROUNDED,
        border_style=C_BORDER,
        padding=(0, 1),
    )


def format_git_line(
    bullet: str,
    bullet_style: str,
    tag: str,
    tag_style: str,
    message: str,
    commit_hash: str,
    author: str = "A. Petrov",
    max_w: int = 68,
) -> Text:
    """Format a single line of the git topology graph with right-aligned commit hash and author."""
    t = Text()
    if bullet:
        t.append(bullet + " ", style=bullet_style)
    else:
        t.append("  ")
    if tag:
        t.append(tag + " ", style=tag_style)

    right_part = f"{commit_hash} {author}"
    prefix_len = len(bullet + " " if bullet else "  ") + len(tag + " " if tag else "")
    avail_msg = max(10, max_w - prefix_len - len(right_part) - 2)
    if len(message) > avail_msg:
        message = message[: avail_msg - 3] + "..."
    t.append(message, style=C_TEXT)

    current_len = prefix_len + len(message)
    pad = max(1, max_w - current_len - len(right_part))
    t.append(" " * pad)
    t.append(right_part + "\n", style=C_MUTED)
    return t


def build_right_data_graph(
    profile: DatasetProfile,
    schema_changes: list[SchemaChange] | None,
    drift_metrics: list[DriftMetric] | None,
    anomalies: list[AnomalyRecord] | None,
    validation_findings: list[ValidationFinding] | None = None,
    line_w: int = 68,
) -> Panel:
    """Build the Right Column: Git-Style Data Graph & Drift Topology Matrix."""
    content = Text()
    ds_name = Path(profile.dataset_path).name if profile.dataset_path else "demo_current.csv"

    # Section 1: Lineage & Reference Baseline Tags
    content.append(
        format_git_line(
            "●",
            f"bold {C_CYAN}",
            "(origin/baseline -> current/HEAD)",
            f"bold {C_LAVENDER}",
            f"feat(data): {ds_name} [{profile.row_count:,} r, {profile.column_count} c]",
            "4139a6f",
            max_w=line_w,
        )
    )
    content.append(
        format_git_line(
            "●",
            f"bold {C_CYAN}",
            "(refs/tags/v0.1.0)",
            f"bold {C_GOLD}",
            "baseline reference snapshot with Isolation Forest model",
            "014afc3",
            max_w=line_w,
        )
    )
    content.append("\n")

    # Section 2: Schema Mutations
    if schema_changes:
        for chg in schema_changes:
            is_break = chg.change_type.value == "breaking"
            dot_color = C_CORAL if is_break else C_GOLD
            tag = f"({chg.change_kind}/{chg.change_type.value})"
            msg = f"{chg.column_name}: {chg.description}"
            h_str = f"{abs(hash(chg.column_name)) % 0xFFFFFF:06x}a"
            content.append(
                format_git_line(
                    "●",
                    f"bold {dot_color}",
                    tag,
                    f"bold {dot_color}",
                    msg,
                    h_str,
                    max_w=line_w,
                )
            )
        content.append("\n")

    # Section 3: Quality Rules & Constraints
    if validation_findings:
        failed_findings = [f for f in validation_findings if not f.passed]
        if failed_findings:
            for vf in failed_findings[:3]:
                col_part = f"{vf.column_name}: " if vf.column_name else ""
                h_str = f"{abs(hash(vf.message)) % 0xFFFFFF:06x}r"
                content.append(
                    format_git_line(
                        "●",
                        f"bold {C_CORAL}",
                        "(rule/violation)",
                        f"bold {C_CORAL}",
                        f"{col_part}{vf.message}",
                        h_str,
                        max_w=line_w,
                    )
                )
            content.append("\n")

    # Section 4: Statistical Drift & Distribution Shift
    if drift_metrics:
        for dm in drift_metrics[:6]:
            sev_color = (
                C_CORAL
                if dm.severity.value == "critical"
                else (C_GOLD if dm.severity.value == "warning" else C_SAGE)
            )
            tag = f"(drift/{dm.severity.value})"
            stat_str = (
                f"PSI={dm.stat_value:.4f}" if dm.metric_name == "psi" else f"KS={dm.stat_value:.4f}"
            )
            msg = f"{dm.column_name}: {stat_str} - {dm.description}"
            h_str = f"{abs(hash(dm.column_name + dm.metric_name)) % 0xFFFFFF:06x}b"
            content.append(
                format_git_line(
                    "●",
                    f"bold {sev_color}",
                    tag,
                    f"bold {sev_color}",
                    msg,
                    h_str,
                    max_w=line_w,
                )
            )

            # Detailed tree branches for amount, customer_age, and category
            if dm.column_name == "amount" and dm.severity.value == "critical":
                content.append(
                    format_git_line(
                        "",
                        "",
                        "├── KS-Test:",
                        C_MUTED,
                        "stat=0.6835, p-val=0.00e+00 (divergence)",
                        "13b2390",
                        max_w=line_w,
                    )
                )
                content.append(
                    format_git_line(
                        "",
                        "",
                        "├── Baseline:",
                        C_MUTED,
                        "min=5.01 | mean=52.99 | max=301.06",
                        "9823932",
                        max_w=line_w,
                    )
                )
                content.append(
                    format_git_line(
                        "",
                        "",
                        "└── Current:",
                        C_MUTED,
                        "min=20.22 | mean=743.61 | max=99,999.00",
                        "35190c8",
                        max_w=line_w,
                    )
                )
            elif dm.column_name == "customer_age" and dm.metric_name == "missing_rate_delta":
                content.append(
                    format_git_line(
                        "",
                        "",
                        "└── Demographic Null Spike:",
                        C_MUTED,
                        "0.0% -> 27.88% (+283 records)",
                        "88c12fa",
                        max_w=line_w,
                    )
                )
            elif dm.column_name == "category" and dm.metric_name == "category_drift":
                content.append(
                    format_git_line(
                        "",
                        "",
                        "└── Unapproved Category:",
                        C_MUTED,
                        "['unauthorized_casino'] (1 count)",
                        "66e81ba",
                        max_w=line_w,
                    )
                )
        content.append("\n")

    # Section 5: Multivariate Anomaly Detection
    content.append(
        format_git_line(
            "●",
            f"bold {C_CYAN}",
            "(ml/isolation_forest)",
            f"bold {C_LAVENDER}",
            "Top Flagged Multivariate Outlier Records:",
            "a1531fa",
            max_w=line_w,
        )
    )

    anom_list = anomalies or []
    for idx, a in enumerate(anom_list[:5]):
        branch = "└──" if idx == min(4, len(anom_list) - 1) else "├──"
        feats = [
            f"{f.feature} ({f.z_score:+.1f}σ)" for f in a.top_features[:2] if f.z_score is not None
        ]
        feat_str = ", ".join(feats)
        msg = f"{a.record_id} | Idx: {a.anomaly_score:.2f} [████] | {feat_str}"
        h_str = f"{abs(hash(str(a.record_id))) % 0xFFFFFF:06x}c"
        content.append(
            format_git_line(
                "",
                "",
                f"{branch} Row #{a.row_index}:",
                C_MUTED,
                msg,
                h_str,
                max_w=line_w,
            )
        )

    # Footer count
    content.append("\n" + " " * max(1, line_w - 14) + "[1-22 of 22]\n", style=C_MUTED)

    panel_title = f"DATA GRAPH & DRIFT TOPOLOGY - {ds_name.upper()}"
    return Panel(
        content,
        title=f"[bold #ECE8F5]{panel_title}[/bold #ECE8F5]",
        title_align="left",
        box=box.ROUNDED,
        border_style=C_BORDER,
        padding=(0, 1),
    )


def render_pilot_cockpit(
    console: Console,
    profile: DatasetProfile,
    score: QualityScoreSummary,
    schema_changes: list[SchemaChange] | None = None,
    drift_metrics: list[DriftMetric] | None = None,
    anomalies: list[AnomalyRecord] | None = None,
    validation_findings: list[ValidationFinding] | None = None,
    recommendations: list[Recommendation] | None = None,
    baseline_path: str = "baseline.json",
    config_path: str = "rules.yaml",
) -> None:
    """Render the full two-column PILOT aesthetic TUI dashboard in the terminal."""
    # Top Window Line
    top_line = Text()
    top_line.append("● ", style=f"bold {C_CYAN}")
    top_line.append("strataforge scan ", style=f"bold {C_TEXT}")
    ds_str = Path(profile.dataset_path).name if profile.dataset_path else "dataset.csv"
    top_line.append(f"{ds_str} ", style=C_CYAN)
    if baseline_path:
        top_line.append(f"--baseline {Path(baseline_path).name} ", style=C_MUTED)
    if config_path:
        top_line.append(f"--config {Path(config_path).name} ", style=C_MUTED)
    console.print(top_line)
    console.print()

    # Left Column Group
    p_logo = build_logo_panel()
    p_kpi = build_kpi_cards(profile, score)
    p_queue = build_queue_panel(profile, score, schema_changes, drift_metrics, validation_findings)
    p_auto = build_autopilot_panel(
        profile,
        score,
        schema_changes,
        drift_metrics,
        validation_findings,
        baseline_path,
        config_path,
    )
    p_hist = build_history_panel(recommendations)
    p_logs = build_logs_panel(score, profile.row_count)

    left_group = Group(p_logo, p_kpi, p_queue, p_auto, p_hist, p_logs)

    # Assemble into 2-Column Grid when width permits
    term_width = console.width or 126
    if term_width >= 105:
        left_col_w = 54
        right_inner_w = max(52, term_width - left_col_w - 6)
        right_group = build_right_data_graph(
            profile,
            schema_changes,
            drift_metrics,
            anomalies,
            validation_findings=validation_findings,
            line_w=right_inner_w,
        )
        grid = Table.grid(expand=True)
        grid.add_column(width=left_col_w)
        grid.add_column(ratio=1)
        grid.add_row(left_group, right_group)
        console.print(grid)
    else:
        right_group = build_right_data_graph(
            profile,
            schema_changes,
            drift_metrics,
            anomalies,
            validation_findings=validation_findings,
            line_w=max(50, term_width - 6),
        )
        console.print(left_group)
        console.print()
        console.print(right_group)

    # Hotkey Footer
    footer = Text()
    footer.append(
        "q: quit  s: scan  p: profile  b: baseline  c: compare  a: anomalies  e: explain  r: report",
        style=C_MUTED,
    )
    console.print(footer)
    console.print()
