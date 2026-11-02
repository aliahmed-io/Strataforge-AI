"""Synthetic demonstration dataset generators with seeded reproducibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_baseline_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic, clean reference baseline dataset."""
    rng = np.random.default_rng(seed)

    # 1. Primary keys
    transaction_ids = [f"TX-{100000 + i}" for i in range(n_rows)]

    # 2. User IDs (100 to 500)
    user_ids = rng.integers(100, 500, size=n_rows)

    # 3. Transaction Amount (Log-normal distribution around $45, std $25)
    amounts = np.round(rng.lognormal(mean=3.8, sigma=0.6, size=n_rows), 2)
    # Ensure min amount is $1.00
    amounts = np.clip(amounts, 1.00, 2500.00)

    # 4. Categories with stable distribution
    categories = rng.choice(
        ["retail", "grocery", "travel", "entertainment", "utility"],
        size=n_rows,
        p=[0.35, 0.30, 0.15, 0.12, 0.08],
    )

    # 5. Transaction Status
    statuses = rng.choice(
        ["settled", "pending", "declined", "refunded"],
        size=n_rows,
        p=[0.88, 0.08, 0.03, 0.01],
    )

    # 6. Customer Age
    ages = rng.integers(18, 75, size=n_rows)

    # 7. Risk Score (0.0 to 1.0, beta distribution skewed low)
    risk_scores = np.round(rng.beta(a=1.5, b=8.0, size=n_rows), 4)

    # 8. Timestamps
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    time_deltas = pd.to_timedelta(rng.integers(0, 86400 * 30, size=n_rows), unit="s")
    created_at = (base_time + time_deltas).strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "user_id": user_ids,
            "amount": amounts,
            "category": categories,
            "status": statuses,
            "customer_age": ages,
            "risk_score": risk_scores,
            "created_at": created_at,
        }
    )

    return df


def generate_drifted_dataset(
    baseline_df: pd.DataFrame,
    seed: int = 1337,
) -> pd.DataFrame:
    """Generate modified current dataset containing intentional data-quality flaws:

    1. Distribution drift on 'amount' (PSI > 0.30 shift toward higher transactions)
    2. Missing value spike on 'customer_age' (jumps from 0% to ~28% nulls)
    3. Duplicate primary keys (intentional duplicate rows)
    4. Extreme numerical outliers in 'amount' (e.g. $999,999 sensor/fraud spike)
    5. Corrupted types in 'risk_score' (strings injected into float column)
    6. Unapproved categorical values in 'category' (e.g. 'crypto_gambling', 'unknown_vendor')
    7. Removed baseline column: 'status' dropped
    8. Added new column: 'device_fingerprint'
    """
    rng = np.random.default_rng(seed)
    df = baseline_df.copy()

    n = len(df)

    # 1. Distribution drift: transaction amount shifts significantly higher (macro inflation/campaign)
    drifted_amounts = np.round(rng.lognormal(mean=5.2, sigma=0.8, size=n), 2)
    df["amount"] = np.clip(drifted_amounts, 5.00, 15000.00)

    # 2. Missing value spike: introduce 28% nulls into customer_age
    null_mask = rng.random(size=n) < 0.28
    df.loc[null_mask, "customer_age"] = np.nan

    # 3. Duplicate rows: duplicate 15 random rows
    dup_indices = rng.choice(n, size=15, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. Extreme outliers: inject anomalies into 5 rows
    outlier_indices = [5, 42, 108, 250, 412]
    for idx in outlier_indices:
        if idx < len(df):
            df.loc[idx, "amount"] = 99999.00
            df.loc[idx, "risk_score"] = 0.9999

    # 5. Unapproved categorical values
    unapproved_indices = [12, 88, 190, 310]
    for idx in unapproved_indices:
        if idx < len(df):
            df.loc[idx, "category"] = "unauthorized_casino"

    # 6. Corrupted type: inject text into numerical risk_score
    type_corrupt_indices = [20, 55]
    df["risk_score"] = df["risk_score"].astype(object)
    for idx in type_corrupt_indices:
        if idx < len(df):
            df.loc[idx, "risk_score"] = "CORRUPTED_VALUE"

    # 7. Breaking schema change: Drop 'status' column
    if "status" in df.columns:
        df = df.drop(columns=["status"])

    # 8. Added column: 'device_fingerprint'
    df["device_fingerprint"] = [f"dev_{rng.integers(1000, 9999)}" for _ in range(len(df))]

    return df


def write_demo_files(output_dir: str | Path = ".") -> tuple[Path, Path]:
    """Generate and write both baseline.csv and current_flawed.csv to specified directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = generate_baseline_dataset(n_rows=1000, seed=42)
    flawed_df = generate_drifted_dataset(base_df, seed=1337)

    base_path = out_dir / "demo_baseline.csv"
    flawed_path = out_dir / "demo_current.csv"

    base_df.to_csv(base_path, index=False)
    flawed_df.to_csv(flawed_path, index=False)

    return base_path, flawed_path
