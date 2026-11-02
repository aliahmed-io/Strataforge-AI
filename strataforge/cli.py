"""Terminal-native CLI application for Strataforge AI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.panel import Panel

from strataforge.core.config import QualityRules, generate_default_rules_yaml
from strataforge.core.loader import load_dataset
from strataforge.core.schema import compare_schemas
from strataforge.core.types import ExitCode, Severity
from strataforge.demo.generator import write_demo_files
from strataforge.drift.detector import detect_drift
from strataforge.ml.anomaly import StrataforgeAnomalyModel
from strataforge.ml.explainer import explain_anomaly_row
from strataforge.ml.storage import create_baseline_snapshot, load_baseline_snapshot
from strataforge.profiling.duplicates import detect_duplicates
from strataforge.profiling.profiler import profile_dataset
from strataforge.recommendations.engine import generate_recommendations
from strataforge.recommendations.fix_preview import preview_recommended_fixes
from strataforge.reporting.html_generator import generate_html_report
from strataforge.reporting.summary import compute_quality_score
from strataforge.ui.components import (
    render_anomalies_table,
    render_anomaly_detail,
    render_column_profile_table,
    render_drift_table,
    render_header,
    render_kpi_summary,
    render_recommendations,
    render_schema_changes_table,
    render_validation_table,
)
from strataforge.ui.theme import create_console
from strataforge.validation.engine import validate_dataset

app = typer.Typer(
    name="strataforge",
    help="Strataforge AI — Terminal-native intelligent data-quality analysis engine.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("profile")
def profile_cmd(
    path: Annotated[Path, typer.Argument(help="Path to CSV, JSONL, or Parquet dataset.")],
    chunk_size: Annotated[
        int | None, typer.Option("--chunk-size", "-c", help="Chunk size for streaming large files.")
    ] = None,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable colored output for plain logs.")
    ] = False,
) -> None:
    """Display inferred schema, missing-value counts, duplicates, and column distributions."""
    console = create_console(no_color)
    try:
        render_header(console, "Profile", str(path))
        with console.status(
            "[plum]Profiling dataset schema and distributions...[/plum]", spinner="dots"
        ):
            prof = profile_dataset(path, chunk_size=chunk_size)

        score = compute_quality_score(
            duplicate_report=detect_duplicates(load_dataset(path, nrows=5000)),
            total_rows=prof.row_count,
        )
        render_kpi_summary(console, prof, score)
        render_column_profile_table(console, prof)
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Execution Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("baseline")
def baseline_cmd(
    path: Annotated[Path, typer.Argument(help="Path to clean reference dataset.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Target path for baseline JSON snapshot.")
    ] = Path("baseline.json"),
    contamination: Annotated[
        float, typer.Option("--contamination", help="Isolation Forest expected contamination rate.")
    ] = 0.05,
    no_model: Annotated[
        bool, typer.Option("--no-model", help="Skip fitting multivariate Isolation Forest model.")
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Create a versioned schema and statistical baseline snapshot."""
    console = create_console(no_color)
    try:
        render_header(console, "Create Baseline", str(path))
        with console.status(
            "[plum]Generating reference snapshot and fitting anomaly model...[/plum]",
            spinner="dots",
        ):
            snapshot = create_baseline_snapshot(
                dataset=path,
                output_path=out,
                fit_model=not no_model,
                contamination=contamination,
            )

        console.print(
            Panel(
                f"[sage]Baseline successfully saved to: [bold]{out.resolve()}[/bold][/sage]\n"
                f"Observations indexed: {snapshot.row_count:,} | Columns: {snapshot.column_count}\n"
                f"Model Artifact: {snapshot.model_artifact_path or 'None (skipped)'}",
                box=box.ROUNDED,
                border_style="plum",
            )
        )
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Baseline Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("compare")
def compare_cmd(
    path: Annotated[Path, typer.Argument(help="Current dataset to evaluate.")],
    baseline: Annotated[
        Path, typer.Option("--baseline", "-b", help="Path to baseline.json snapshot.")
    ] = Path("baseline.json"),
    fix_preview: Annotated[
        bool, typer.Option("--fix-preview", help="Preview proposed safe remediation steps.")
    ] = False,
    ci: Annotated[
        bool,
        typer.Option(
            "--ci", help="CI mode: exit 1 if critical schema changes or drift are detected."
        ),
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Detect schema mutations and statistical distribution drift against a baseline."""
    console = create_console(no_color)
    try:
        render_header(console, "Compare Against Baseline", f"{path.name} vs {baseline.name}")
        with console.status(
            "[plum]Comparing schemas and calculating statistical drift...[/plum]", spinner="dots"
        ):
            snapshot, _ = load_baseline_snapshot(baseline)
            curr_df = load_dataset(path)
            curr_prof = profile_dataset(curr_df)

            # Schema diff
            baseline_schema_dict = {
                k: {"inferred_type": v.inferred_type.value, "null_percentage": v.null_percentage}
                for k, v in snapshot.profile.columns.items()
            }
            current_schema_dict = {
                k: {"inferred_type": v.inferred_type.value, "null_percentage": v.null_percentage}
                for k, v in curr_prof.columns.items()
            }
            schema_changes = compare_schemas(baseline_schema_dict, current_schema_dict)

            # Drift detection
            base_df = (
                load_dataset(snapshot.dataset_source)
                if Path(snapshot.dataset_source).is_file()
                else None
            )

            if base_df is not None:
                drift_res = detect_drift(base_df, curr_df)
                drift_metrics = drift_res.metrics
            else:
                drift_metrics = []

            dup_rep = detect_duplicates(curr_df)
            recs = generate_recommendations(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                duplicate_report=dup_rep,
            )
            score = compute_quality_score(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                duplicate_report=dup_rep,
                total_rows=curr_prof.row_count,
            )

        render_kpi_summary(console, curr_prof, score)
        render_schema_changes_table(console, schema_changes)
        render_drift_table(console, drift_metrics)
        render_recommendations(console, recs)

        if fix_preview:
            preview_df, applied = preview_recommended_fixes(curr_df, recs)
            console.print("\n[copper]FIX PREVIEW (SIMULATION ONLY - NO FILES MODIFIED)[/copper]")
            for a in applied:
                console.print(f"  • {a}")

        has_breaking = any(c.change_type.value == "breaking" for c in schema_changes)
        has_crit_drift = any(m.severity == Severity.CRITICAL for m in drift_metrics)

        if (ci or True) and (has_breaking or has_crit_drift):
            console.print(
                "\n[coral]Data quality comparison failed: critical breaking changes or drift detected.[/coral]"
            )
            raise typer.Exit(code=ExitCode.DATA_FAILURE.value)

        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Comparison Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("anomalies")
def anomalies_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset to scan for multivariate anomalies.")],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="Optional baseline JSON to use fitted model."),
    ] = None,
    contamination: Annotated[
        float, typer.Option("--contamination", "-c", help="Expected contamination rate.")
    ] = 0.05,
    top: Annotated[
        int, typer.Option("--top", "-t", help="Number of top anomalies to display.")
    ] = 10,
    out_scores: Annotated[
        Path | None,
        typer.Option("--out-scores", help="Optional CSV output for row anomaly scores."),
    ] = None,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Run Isolation Forest anomaly detection on suitable numerical features."""
    console = create_console(no_color)
    try:
        render_header(console, "Multivariate Anomaly Detection", str(path))
        with console.status(
            "[plum]Fitting/evaluating Isolation Forest on numerical features...[/plum]",
            spinner="dots",
        ):
            df = load_dataset(path)

            if baseline and baseline.is_file():
                _, model = load_baseline_snapshot(baseline)
                if model is None:
                    model = StrataforgeAnomalyModel(contamination=contamination).fit(df)
            else:
                model = StrataforgeAnomalyModel(contamination=contamination).fit(df)

            is_anom, scores = model.score(df)

            # Build list of AnomalyRecord
            anom_indices = [i for i, flagged in enumerate(is_anom) if flagged]
            # Also sort by score descending
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            anomaly_records = []
            for idx in ranked_indices[: max(top, len(anom_indices))]:
                raw_score = float(model.model.decision_function(model.transform(df.iloc[[idx]]))[0])
                rec = model.explain_record(
                    row_series=df.iloc[idx],
                    row_idx=idx,
                    record_id=df.iloc[idx].get("transaction_id", df.iloc[idx].get("id", idx)),
                    anomaly_score=float(scores[idx]),
                    raw_score=raw_score,
                    is_anomaly=bool(is_anom[idx]),
                )
                anomaly_records.append(rec)

            if out_scores:
                scores_df = df.copy()
                scores_df["strataforge_anomaly_score"] = scores
                scores_df["strataforge_is_anomaly"] = is_anom
                scores_df.to_csv(out_scores, index=False)
                console.print(f"[sage]Anomaly scores exported to: {out_scores}[/sage]")

        render_anomalies_table(console, anomaly_records, max_rows=top)
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Anomaly Detection Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("explain")
def explain_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset containing the row to explain.")],
    row: Annotated[
        str, typer.Option("--row", "-r", help="Row index or record ID to inspect.")
    ] = "0",
    baseline: Annotated[
        Path | None, typer.Option("--baseline", "-b", help="Optional baseline JSON snapshot.")
    ] = None,
    id_col: Annotated[
        str | None, typer.Option("--id-col", help="Column to match row ID against.")
    ] = None,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Inspect an identified anomaly using previously saved analysis results."""
    console = create_console(no_color)
    try:
        render_header(console, "Explain Anomaly Record", f"Row: {row}")
        with console.status(
            "[plum]Extracting feature attributions and reference deviations...[/plum]",
            spinner="dots",
        ):
            df = load_dataset(path)

            if baseline and baseline.is_file():
                _, model = load_baseline_snapshot(baseline)
                if model is None:
                    model = StrataforgeAnomalyModel().fit(df)
            else:
                model = StrataforgeAnomalyModel().fit(df)

            record = explain_anomaly_row(model, df, row_identifier=row, id_column=id_col)

        render_anomaly_detail(console, record)
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Explain Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("validate")
def validate_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset to validate.")],
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to rules.yaml.")] = Path(
        "rules.yaml"
    ),
    fix_preview: Annotated[
        bool, typer.Option("--fix-preview", help="Preview proposed remediation transformations.")
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Apply explicit data-quality rules configured in rules.yaml."""
    console = create_console(no_color)
    try:
        render_header(console, "Validate Rules", str(path))
        with console.status(
            "[plum]Evaluating assertions from rules.yaml...[/plum]", spinner="dots"
        ):
            rules = QualityRules.from_yaml_file(config)
            df = load_dataset(path)
            res = validate_dataset(df, rules)
            recs = generate_recommendations(validation_findings=res.findings)

        render_validation_table(console, res.findings)
        render_recommendations(console, recs)

        if fix_preview:
            preview_df, applied = preview_recommended_fixes(df, recs)
            console.print("\n[copper]FIX PREVIEW (SIMULATION ONLY - NO FILES MODIFIED)[/copper]")
            for a in applied:
                console.print(f"  • {a}")

        if not res.passed:
            console.print(
                f"\n[coral]Data quality validation failed with {res.failed_checks} assertion error(s).[/coral]"
            )
            for reason in res.failure_reasons[:5]:
                console.print(f"  [coral][X][/coral] {reason}")
            raise typer.Exit(code=ExitCode.DATA_FAILURE.value)

        console.print("\n[sage]All data quality rules passed successfully.[/sage]")
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Validation Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("scan")
def scan_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset to analyze.")],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="Optional baseline for drift comparison."),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Optional rules.yaml to validate.")
    ] = None,
    fix_preview: Annotated[
        bool, typer.Option("--fix-preview", help="Preview proposed fixes.")
    ] = False,
    ci: Annotated[bool, typer.Option("--ci", help="CI mode: exit 1 on critical findings.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Analyze a dataset and print a comprehensive quality report."""
    console = create_console(no_color)
    try:
        render_header(console, "Comprehensive Quality Scan", str(path))
        with console.status(
            "[plum]Scanning dataset, evaluating distributions & anomalies...[/plum]", spinner="dots"
        ):
            df = load_dataset(path)
            prof = profile_dataset(df)
            dup_rep = detect_duplicates(df)

            schema_changes = []
            drift_metrics = []
            if baseline and baseline.is_file():
                snapshot, snap_model = load_baseline_snapshot(baseline)
                baseline_schema_dict = {
                    k: {
                        "inferred_type": v.inferred_type.value,
                        "null_percentage": v.null_percentage,
                    }
                    for k, v in snapshot.profile.columns.items()
                }
                curr_schema_dict = {
                    k: {
                        "inferred_type": v.inferred_type.value,
                        "null_percentage": v.null_percentage,
                    }
                    for k, v in prof.columns.items()
                }
                schema_changes = compare_schemas(baseline_schema_dict, curr_schema_dict)
                if Path(snapshot.dataset_source).is_file():
                    base_df = load_dataset(snapshot.dataset_source)
                    drift_metrics = detect_drift(base_df, df).metrics

            # Run anomaly model
            anomaly_records = []
            try:
                model = StrataforgeAnomalyModel().fit(df)
                is_anom, scores = model.score(df)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
                for idx in top_indices:
                    raw_s = float(model.model.decision_function(model.transform(df.iloc[[idx]]))[0])
                    rec = model.explain_record(
                        row_series=df.iloc[idx],
                        row_idx=idx,
                        record_id=df.iloc[idx].get("transaction_id", df.iloc[idx].get("id", idx)),
                        anomaly_score=float(scores[idx]),
                        raw_score=raw_s,
                        is_anomaly=bool(is_anom[idx]),
                    )
                    anomaly_records.append(rec)
            except Exception:
                pass

            # Validate rules if provided
            validation_findings = []
            if config and config.is_file():
                rules = QualityRules.from_yaml_file(config)
                v_res = validate_dataset(df, rules)
                validation_findings = v_res.findings

            score = compute_quality_score(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                validation_findings=validation_findings,
                duplicate_report=dup_rep,
                anomaly_count=len([a for a in anomaly_records if a.is_anomaly]),
                total_rows=prof.row_count,
            )

            recs = generate_recommendations(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                validation_findings=validation_findings,
                duplicate_report=dup_rep,
                anomalies=anomaly_records,
            )

        render_kpi_summary(console, prof, score)
        render_column_profile_table(console, prof)

        if schema_changes:
            render_schema_changes_table(console, schema_changes)
        if drift_metrics:
            render_drift_table(console, drift_metrics)
        if anomaly_records:
            render_anomalies_table(console, anomaly_records, max_rows=5)
        if validation_findings:
            render_validation_table(console, validation_findings)

        render_recommendations(console, recs)

        if fix_preview:
            preview_df, applied = preview_recommended_fixes(df, recs)
            console.print("\n[copper]FIX PREVIEW (SIMULATION ONLY - NO FILES MODIFIED)[/copper]")
            for a in applied:
                console.print(f"  • {a}")

        if (ci or score.status == Severity.CRITICAL) and score.critical_checks > 0:
            console.print("\n[coral]Scan completed with critical data quality failures.[/coral]")
            raise typer.Exit(code=ExitCode.DATA_FAILURE.value)

        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Scan Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("report")
def report_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset to analyze.")],
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (currently 'html').")
    ] = "html",
    out: Annotated[Path, typer.Option("--out", "-o", help="Target output file path.")] = Path(
        "report.html"
    ),
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="Optional baseline for drift comparison."),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Optional rules.yaml to validate.")
    ] = None,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Export a standalone HTML data-quality report."""
    console = create_console(no_color)
    try:
        render_header(console, "Generate Report", str(path))
        with console.status(
            f"[plum]Building standalone {format.upper()} report...[/plum]", spinner="dots"
        ):
            df = load_dataset(path)
            prof = profile_dataset(df)
            dup_rep = detect_duplicates(df)

            schema_changes = []
            drift_metrics = []
            if baseline and baseline.is_file():
                snapshot, _ = load_baseline_snapshot(baseline)
                baseline_schema_dict = {
                    k: {
                        "inferred_type": v.inferred_type.value,
                        "null_percentage": v.null_percentage,
                    }
                    for k, v in snapshot.profile.columns.items()
                }
                curr_schema_dict = {
                    k: {
                        "inferred_type": v.inferred_type.value,
                        "null_percentage": v.null_percentage,
                    }
                    for k, v in prof.columns.items()
                }
                schema_changes = compare_schemas(baseline_schema_dict, curr_schema_dict)
                if Path(snapshot.dataset_source).is_file():
                    base_df = load_dataset(snapshot.dataset_source)
                    drift_metrics = detect_drift(base_df, df).metrics

            # Model anomalies
            anomaly_records = []
            try:
                model = StrataforgeAnomalyModel().fit(df)
                is_anom, scores = model.score(df)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                    :15
                ]
                for idx in ranked_indices:
                    raw_s = float(model.model.decision_function(model.transform(df.iloc[[idx]]))[0])
                    rec = model.explain_record(
                        row_series=df.iloc[idx],
                        row_idx=idx,
                        record_id=df.iloc[idx].get("transaction_id", df.iloc[idx].get("id", idx)),
                        anomaly_score=float(scores[idx]),
                        raw_score=raw_s,
                        is_anomaly=bool(is_anom[idx]),
                    )
                    anomaly_records.append(rec)
            except Exception:
                pass

            validation_findings = []
            if config and config.is_file():
                rules = QualityRules.from_yaml_file(config)
                v_res = validate_dataset(df, rules)
                validation_findings = v_res.findings

            score = compute_quality_score(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                validation_findings=validation_findings,
                duplicate_report=dup_rep,
                anomaly_count=len([a for a in anomaly_records if a.is_anomaly]),
                total_rows=prof.row_count,
            )

            recs = generate_recommendations(
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                validation_findings=validation_findings,
                duplicate_report=dup_rep,
                anomalies=anomaly_records,
            )

            html_path = generate_html_report(
                profile=prof,
                score_summary=score,
                output_path=out,
                schema_changes=schema_changes,
                drift_metrics=drift_metrics,
                validation_findings=validation_findings,
                duplicate_report=dup_rep,
                anomalies=anomaly_records,
                recommendations=recs,
            )

        console.print(
            Panel(
                f"[sage]Standalone HTML report successfully generated:[/sage]\n"
                f"[bold]{html_path}[/bold]\n"
                f"Health Score: {score.grade} ({score.health_score:.1f}/100) | Checks: {score.total_checks}",
                box=box.ROUNDED,
                border_style="plum",
            )
        )
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Report Generation Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


@app.command("init")
def init_cmd(
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir", "-d", help="Directory where project files should be initialized."
        ),
    ] = Path("."),
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
) -> None:
    """Create an example configuration, rules.yaml, and synthetic demo datasets."""
    console = create_console(no_color)
    try:
        render_header(console, "Initialize Project", str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        rules_path = out_dir / "rules.yaml"
        if not rules_path.exists():
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(generate_default_rules_yaml())
            console.print(
                f"  [sage][OK][/sage] Created configuration template: [bold]{rules_path}[/bold]"
            )
        else:
            console.print(
                f"  [gold][!][/gold] Configuration already exists: [bold]{rules_path}[/bold]"
            )

        base_csv, current_csv = write_demo_files(out_dir)
        console.print(
            f"  [sage][OK][/sage] Created clean baseline demo dataset: [bold]{base_csv}[/bold]"
        )
        console.print(
            f"  [sage][OK][/sage] Created flawed current demo dataset: [bold]{current_csv}[/bold]"
        )

        console.print(
            Panel(
                "[copper]Quickstart Next Steps:[/copper]\n"
                "1. [text]strataforge profile demo_baseline.csv[/text]\n"
                "2. [text]strataforge baseline demo_baseline.csv --out baseline.json[/text]\n"
                "3. [text]strataforge compare demo_current.csv --baseline baseline.json[/text]\n"
                "4. [text]strataforge validate demo_current.csv --config rules.yaml[/text]\n"
                "5. [text]strataforge scan demo_current.csv --baseline baseline.json --config rules.yaml[/text]\n"
                "6. [text]strataforge report demo_current.csv --baseline baseline.json --config rules.yaml[/text]",
                title="[copper]Initialization Complete[/copper]",
                box=box.ROUNDED,
                border_style="plum",
            )
        )
        raise typer.Exit(code=ExitCode.SUCCESS.value)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[coral]Init Error: {exc}[/coral]")
        raise typer.Exit(code=ExitCode.SYSTEM_ERROR.value) from exc


if __name__ == "__main__":
    app()
