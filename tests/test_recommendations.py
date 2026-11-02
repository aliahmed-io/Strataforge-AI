"""Unit tests for intelligent recommendations and non-destructive fix previews."""

from __future__ import annotations

import pandas as pd

from strataforge.core.types import ChangeType, SchemaChange
from strataforge.profiling.duplicates import DuplicateReport
from strataforge.recommendations.engine import generate_recommendations
from strataforge.recommendations.fix_preview import preview_recommended_fixes


def test_generate_recommendations_schema_and_duplicates():
    changes = [
        SchemaChange(
            column_name="status",
            change_kind="removed",
            old_value="categorical",
            new_value=None,
            change_type=ChangeType.BREAKING,
            description="Column was removed",
        )
    ]
    dup_report = DuplicateReport(
        total_rows=100,
        duplicate_rows_count=5,
        duplicate_percentage=5.0,
        has_critical_duplicates=True,
    )

    recs = generate_recommendations(schema_changes=changes, duplicate_report=dup_report)
    assert len(recs) == 2
    assert any(r.category == "schema" for r in recs)
    assert any(r.category == "duplicates" for r in recs)


def test_fix_preview_does_not_modify_original():
    original_df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "val": [10.0, None, 20.0, 30.0],
        }
    )
    copy_orig = original_df.copy()

    dup_report = DuplicateReport(
        total_rows=4,
        duplicate_rows_count=2,
        duplicate_percentage=50.0,
        subset_columns=["id"],
        has_critical_duplicates=True,
    )
    recs = generate_recommendations(duplicate_report=dup_report)

    preview_df, applied = preview_recommended_fixes(original_df, recs)

    # Assert original was untouched
    pd.testing.assert_frame_equal(original_df, copy_orig)
    # Assert preview was remediated
    assert len(preview_df) == 3
    assert len(applied) > 0
