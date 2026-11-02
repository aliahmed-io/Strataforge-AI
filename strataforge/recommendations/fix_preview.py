"""Safe, non-destructive fix preview for proposing data remediation transformations."""

from __future__ import annotations

import pandas as pd

from strataforge.core.types import Recommendation


def preview_recommended_fixes(
    df: pd.DataFrame,
    recommendations: list[Recommendation],
) -> tuple[pd.DataFrame, list[str]]:
    """Simulate applying recommended fixes on an in-memory copy without altering input files.

    Returns:
        tuple of (remediated_df_sample, list_of_applied_action_descriptions)
    """
    preview_df = df.copy()
    applied_actions: list[str] = []

    for rec in recommendations:
        if rec.category == "duplicates":
            before_len = len(preview_df)
            # Check for primary key / subset columns in title or code snippet
            subset = None
            for col in preview_df.columns:
                if f"'{col}'" in rec.title or (rec.code_snippet and f"'{col}'" in rec.code_snippet):
                    subset = [col]
                    break
            preview_df = preview_df.drop_duplicates(subset=subset, keep="last")
            dropped = before_len - len(preview_df)
            applied_actions.append(f"Deduplicated dataset: removed {dropped} duplicate row(s).")

        elif rec.category == "missing_values":
            # Extract column name from title
            col_target = None
            for col in preview_df.columns:
                if f"'{col}'" in rec.title or f"'{col}'" in rec.evidence:
                    col_target = col
                    break
            if col_target and col_target in preview_df.columns:
                nulls_before = preview_df[col_target].isna().sum()
                if nulls_before > 0:
                    if pd.api.types.is_numeric_dtype(preview_df[col_target]):
                        fill_val = preview_df[col_target].median()
                    else:
                        mode_vals = preview_df[col_target].mode()
                        fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
                    preview_df[col_target] = preview_df[col_target].fillna(fill_val)
                    applied_actions.append(
                        f"Imputed {nulls_before} missing value(s) in '{col_target}' with {fill_val}."
                    )

        elif rec.category == "outliers":
            # Extract column target
            col_target = None
            for col in preview_df.columns:
                if f"'{col}'" in rec.title or f"'{col}'" in rec.evidence:
                    col_target = col
                    break
            if (
                col_target
                and col_target in preview_df.columns
                and pd.api.types.is_numeric_dtype(preview_df[col_target])
            ):
                # Clamp to 1st and 99th percentiles for preview
                p01 = preview_df[col_target].quantile(0.01)
                p99 = preview_df[col_target].quantile(0.99)
                preview_df[col_target] = preview_df[col_target].clip(lower=p01, upper=p99)
                applied_actions.append(
                    f"Clamped '{col_target}' bounds to empirical range [{p01:.2f}, {p99:.2f}]."
                )

    return preview_df, applied_actions
