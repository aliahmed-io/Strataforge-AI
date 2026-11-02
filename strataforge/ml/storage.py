"""Baseline snapshot management and artifact serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from strataforge.core.types import DatasetProfile
from strataforge.ml.anomaly import StrataforgeAnomalyModel
from strataforge.profiling.profiler import profile_dataset


class BaselineSnapshot(BaseModel):
    """Complete baseline snapshot capturing schema, distributions, and model metadata."""

    version: str = "1.0"
    engine_version: str = "0.1.0"
    created_at: str
    dataset_source: str
    row_count: int
    column_count: int
    profile: DatasetProfile
    model_artifact_path: str | None = None
    numerical_features: list[str] = Field(default_factory=list)


def create_baseline_snapshot(
    dataset: pd.DataFrame | str | Path,
    output_path: str | Path,
    fit_model: bool = True,
    contamination: float = 0.05,
) -> BaselineSnapshot:
    """Create and persist a complete baseline JSON snapshot and optional model artifact."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, (str, Path)):
        df = pd.read_csv(dataset) if str(dataset).endswith(".csv") else pd.read_parquet(dataset)
        source_name = str(dataset)
    else:
        df = dataset
        source_name = "in-memory"

    profile = profile_dataset(df)

    model_path_str = None
    num_features = []

    if fit_model:
        model = StrataforgeAnomalyModel(contamination=contamination)
        num_features = model.select_numerical_features(df)
        if num_features:
            model.fit(df)
            model_artifact = out_file.with_suffix(".model.joblib")
            model.save(str(model_artifact))
            model_path_str = model_artifact.name

    snapshot = BaselineSnapshot(
        version="1.0",
        engine_version="0.1.0",
        created_at=profile.created_at,
        dataset_source=source_name,
        row_count=profile.row_count,
        column_count=profile.column_count,
        profile=profile,
        model_artifact_path=model_path_str,
        numerical_features=num_features,
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(snapshot.model_dump_json(indent=2))

    return snapshot


def load_baseline_snapshot(
    path: str | Path,
) -> tuple[BaselineSnapshot, StrataforgeAnomalyModel | None]:
    """Load baseline snapshot and any accompanying anomaly model."""
    snap_path = Path(path)
    if not snap_path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {snap_path}")

    with open(snap_path, encoding="utf-8") as f:
        data = json.load(f)

    snapshot = BaselineSnapshot.model_validate(data)
    model = None

    if snapshot.model_artifact_path:
        model_file = snap_path.parent / snapshot.model_artifact_path
        if model_file.is_file():
            model = StrataforgeAnomalyModel.load(str(model_file))

    return snapshot, model
