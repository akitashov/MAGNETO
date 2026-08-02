#!/usr/bin/env python3
"""S09c. Cluster-bootstrap confidence intervals for Table 1 primary rows.

Computes cell-cluster bootstrap 95% confidence intervals for the five primary
manuscript rows in Table 1. Uses the same residual datasets, exposure windows,
eligibility filters, and bootstrap implementation as the existing
temperature/scenario confidence intervals (S04b / S08 / S09b).

Outputs
-------
results/supplementary_checks/table1_bootstrap_summary.csv
    One row per Table 1 row with observed rho, CI bounds, and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports of supplementary_checks helpers and shared utilities.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import atomic_write, PROJECT_ROOT
from magneto_lib import compute_stats
from supplementary_checks._supplementary_checks_common import (
    attach_exposure,
    build_sii_exposure,
    cell_cluster_bootstrap_rho,
    load_all_residuals,
    make_supplementary_checks_dirs,
    SUPPLEMENTARY_CHECKS_RESULTS,
)

OUTCOME = "sif_771nm"
N_BOOT = 1000
CI_LEVEL = 0.95
CLUSTER_DEFINITION = "0.5° × 0.5° grid cell (lat_id, lon_id)"

# Five primary Table 1 rows: (display_label, method, window_days, sample_type)
TABLE1_ROWS = [
    ("Full sample", "harmonic", 28, "pooled_full"),
    ("Full sample (pairwise matched)", "cyclic_spline", 28, "pairwise_matched"),
    ("Full sample", "harmonic", 21, "pooled_full"),
    ("Persistently vegetated", "harmonic", 28, "control_vegetated"),
    ("Sahara control", "harmonic", 28, "control_Sahara"),
]


def _derived_seed(sample_type: str, method: str, window: int) -> int:
    """Deterministic seed from sample, method and window."""
    return (
        int(hashlib.sha256(f"table1_bootstrap_{sample_type}_{method}_{window}".encode()).hexdigest(), 16)
        % 2**31
    )


def _filter_sample(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """Return observations for one analytical sample."""
    if sample_type in ("pooled_full", "pairwise_matched"):
        return df.copy()
    if sample_type == "control_vegetated":
        return df[df["is_vegetated"] == True].copy()  # noqa: E712
    if sample_type == "control_Sahara":
        return df[df["is_Sahara"] == True].copy()  # noqa: E712
    raise ValueError(f"Unknown sample_type: {sample_type}")


def _residual_col(method: str, sample_type: str) -> str:
    """Residual column name for the matched dataset."""
    if sample_type == "pairwise_matched":
        return f"residual_{method}"
    return "residual"


def _load_dataset(method: str, sample_type: str, all_residuals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load the correct residual dataset for a Table 1 row."""
    if sample_type == "pairwise_matched":
        return all_residuals["matched"]
    return all_residuals[method]


def build_summary() -> pd.DataFrame:
    """Build bootstrap summary for the six Table 1 rows."""
    all_residuals = load_all_residuals(OUTCOME)
    rows = []

    for sample_label, method, window, sample_type in TABLE1_ROWS:
        df = _load_dataset(method, sample_type, all_residuals)
        df = attach_exposure(df, build_sii_exposure([window]))
        sub = _filter_sample(df, sample_type)
        residual_col = _residual_col(method, sample_type)
        sii_col = f"sii_{window}d"

        needed = [residual_col, sii_col, "lat_id", "lon_id"]
        valid = sub[needed].dropna()
        n_obs = len(valid)
        n_cells = valid[["lat_id", "lon_id"]].drop_duplicates().shape[0]

        observed_rho = float(compute_stats(
            valid[residual_col].values, valid[sii_col].values
        )["spearman_rho"])

        seed = _derived_seed(sample_type, method, window)
        boot = cell_cluster_bootstrap_rho(
            sub, residual_col, sii_col, n_boot=N_BOOT, ci=CI_LEVEL, seed=seed
        )

        rows.append({
            "sample_label": sample_label,
            "sample_type": sample_type,
            "method": method,
            "sii_window_days": window,
            "sii_window": f"sii_{window}d",
            "observed_rho": observed_rho,
            "bootstrap_ci_low": boot["boot_ci_lo"],
            "bootstrap_ci_high": boot["boot_ci_hi"],
            "boot_rho_median": boot["boot_rho_median"],
            "n_bootstrap": N_BOOT,
            "n_cells": n_cells,
            "n_observations": n_obs,
            "seed": seed,
            "cluster_definition": CLUSTER_DEFINITION,
        })

    return pd.DataFrame(rows)


def _validate(summary: pd.DataFrame) -> None:
    """Confirm observed rho matches the existing fixed-window table."""
    fixed = pd.read_csv(PROJECT_ROOT / "results" / "fixed_window_results.csv")
    for _, row in summary.iterrows():
        ref = fixed[
            (fixed["sample_type"] == row["sample_type"])
            & (fixed["method"] == row["method"])
            & (fixed["sii_window"] == row["sii_window"])
        ]
        if ref.empty:
            continue
        expected = float(ref.iloc[0]["spearman_rho"])
        if not (pd.isna(row["observed_rho"]) and pd.isna(expected)):
            assert np.isclose(row["observed_rho"], expected, atol=1e-6), (
                f"Observed rho mismatch for {row['sample_type']} {row['method']} {row['sii_window']}: "
                f"{row['observed_rho']} vs {expected}"
            )

    # All Table 1 samples have well more than 50 cells; CI must be finite.
    assert summary["bootstrap_ci_low"].notna().all(), "All Table 1 CI lower bounds must be finite"
    assert summary["bootstrap_ci_high"].notna().all(), "All Table 1 CI upper bounds must be finite"

    print("[VALIDATION] All Table 1 bootstrap checks passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation assertions (not recommended).",
    )
    args = parser.parse_args(argv)

    make_supplementary_checks_dirs()

    print("[S09c] Computing Table 1 cluster-bootstrap confidence intervals ...")
    summary = build_summary()

    if not args.skip_validation:
        _validate(summary)

    summary_path = SUPPLEMENTARY_CHECKS_RESULTS / "table1_bootstrap_summary.csv"
    atomic_write(summary, summary_path, fmt="csv")

    print(f"[OK] Summary: {summary_path} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
