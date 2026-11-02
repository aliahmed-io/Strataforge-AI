"""Population Stability Index (PSI) calculation for numerical and categorical distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_numerical_psi(
    baseline_values: pd.Series | np.ndarray,
    current_values: pd.Series | np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, list[dict[str, float | str]]]:
    """Calculate Population Stability Index (PSI) for continuous numerical features.

    Returns:
        tuple of (psi_value, list_of_bin_diagnostics)
    """
    b = np.asarray(baseline_values)
    c = np.asarray(current_values)

    b = b[~np.isnan(b)]
    c = c[~np.isnan(c)]

    if len(b) == 0 or len(c) == 0:
        return 0.0, []

    # If identical constant distribution, no drift
    if np.all(b == b[0]) and np.all(c == c[0]):
        return (0.0 if b[0] == c[0] else 1.0), []

    # Determine bin edges from baseline quantiles
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(b, quantiles * 100)
    bin_edges = np.unique(bin_edges)  # handle tied quantiles

    if len(bin_edges) < 2:
        bin_edges = np.array([np.min(b) - 1e-5, np.max(b) + 1e-5])

    # Extend outer edges to infinity to catch current data outside baseline range
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequencies in bins
    b_counts, _ = np.histogram(b, bins=bin_edges)
    c_counts, _ = np.histogram(c, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    b_prop = (b_counts + epsilon) / (len(b) + epsilon * len(b_counts))
    c_prop = (c_counts + epsilon) / (len(c) + epsilon * len(c_counts))

    # PSI calculation
    psi_bins = (c_prop - b_prop) * np.log(c_prop / b_prop)
    total_psi = float(np.sum(psi_bins))

    bin_details = []
    for i in range(len(b_counts)):
        lower = f"{bin_edges[i]:.2f}" if np.isfinite(bin_edges[i]) else "-∞"
        upper = f"{bin_edges[i + 1]:.2f}" if np.isfinite(bin_edges[i + 1]) else "+∞"
        bin_details.append(
            {
                "bin": f"[{lower}, {upper})",
                "baseline_prop": float(round(b_prop[i], 4)),
                "current_prop": float(round(c_prop[i], 4)),
                "psi_contribution": float(round(psi_bins[i], 5)),
            }
        )

    return round(total_psi, 4), bin_details


def calculate_categorical_psi(
    baseline_series: pd.Series,
    current_series: pd.Series,
    epsilon: float = 1e-4,
) -> tuple[float, list[dict[str, float | str]], list[str]]:
    """Calculate PSI across discrete category distributions.

    Returns:
        tuple of (psi_value, list_of_category_proportions, list_of_new_categories)
    """
    b_non_null = baseline_series.dropna().astype(str)
    c_non_null = current_series.dropna().astype(str)

    if len(b_non_null) == 0 or len(c_non_null) == 0:
        return 0.0, [], []

    b_cats = set(b_non_null.unique())
    c_cats = set(c_non_null.unique())
    all_cats = sorted(b_cats.union(c_cats))
    new_cats = sorted(c_cats - b_cats)

    b_val_counts = b_non_null.value_counts()
    c_val_counts = c_non_null.value_counts()

    total_b = len(b_non_null)
    total_c = len(c_non_null)

    cat_details = []
    total_psi = 0.0

    for cat in all_cats:
        b_cnt = b_val_counts.get(cat, 0)
        c_cnt = c_val_counts.get(cat, 0)

        p = (b_cnt + epsilon) / (total_b + epsilon * len(all_cats))
        q = (c_cnt + epsilon) / (total_c + epsilon * len(all_cats))

        cat_psi = (q - p) * np.log(q / p)
        total_psi += cat_psi

        cat_details.append(
            {
                "category": cat,
                "baseline_prop": float(round(p, 4)),
                "current_prop": float(round(q, 4)),
                "psi_contribution": float(round(cat_psi, 5)),
                "is_new": cat in new_cats,
            }
        )

    return round(float(total_psi), 4), cat_details, new_cats
