"""Kolmogorov-Smirnov two-sample test with statistical sample-size safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def evaluate_ks_drift(
    baseline_values: pd.Series | np.ndarray,
    current_values: pd.Series | np.ndarray,
    p_value_alpha: float = 0.01,
) -> dict[str, float | str | bool | None]:
    """Run two-sample Kolmogorov-Smirnov test with sample size diagnostics.

    Returns structured dictionary with statistic, p-value, effect size, and limitation notes.
    """
    b = np.asarray(baseline_values)
    c = np.asarray(current_values)

    b = b[~np.isnan(b)]
    c = c[~np.isnan(c)]

    n_baseline = len(b)
    n_current = len(c)

    if n_baseline < 5 or n_current < 5:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": "insufficient_data",
            "sample_size_baseline": n_baseline,
            "sample_size_current": n_current,
            "notes": "Insufficient observations (< 5 samples) to conduct a valid KS test.",
        }

    # Execute 2-sample KS test
    res = stats.ks_2samp(b, c)
    d_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Effect size interpretation for Kolmogorov-Smirnov D statistic:
    # D < 0.05: Negligible difference
    # 0.05 <= D < 0.15: Small difference
    # 0.15 <= D < 0.30: Moderate difference
    # D >= 0.30: Substantial / Large shift
    if d_stat < 0.05:
        effect_desc = "negligible"
    elif d_stat < 0.15:
        effect_desc = "small"
    elif d_stat < 0.30:
        effect_desc = "moderate"
    else:
        effect_desc = "large"

    notes = []
    # Sample-size safeguard 1: Small sample power limitation
    if n_baseline < 30 or n_current < 30:
        notes.append(
            "Small sample size (< 30) limits statistical power; test may fail to reject null hypothesis."
        )

    # Sample-size safeguard 2: Massive sample over-sensitivity
    if n_baseline > 10_000 and n_current > 10_000 and p_val < p_value_alpha and d_stat < 0.05:
        notes.append(
            f"High statistical power (N > 10,000) flagged p < {p_value_alpha}, but D-statistic ({d_stat:.3f}) "
            f"shows a negligible real-world effect size. This change is unlikely to disrupt downstream models."
        )

    # Significant if p-value is below alpha and D-statistic is non-negligible
    is_significant = bool(p_val < p_value_alpha and d_stat >= 0.05)

    return {
        "statistic": round(d_stat, 4),
        "p_value": round(p_val, 6),
        "significant": is_significant,
        "effect_size": effect_desc,
        "sample_size_baseline": n_baseline,
        "sample_size_current": n_current,
        "notes": " ".join(notes) if notes else None,
    }
