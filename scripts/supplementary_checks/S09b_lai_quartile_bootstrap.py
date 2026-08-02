#!/usr/bin/env python3
"""S09b. Cluster-bootstrap confidence intervals for LAI-stratified associations.

Computes cell-cluster bootstrap 95% confidence intervals for the SII–SIF
Spearman association within persistent-vegetation and LAI-quartile strata.
Uses the same residual datasets, exposure windows, eligibility filters, and
bootstrap implementation as the existing temperature/scenario confidence
intervals (S04b / S08).

Outputs
-------
results/supplementary_checks/lai_quartile_bootstrap_draws.parquet
    One row per bootstrap replicate.
results/supplementary_checks/lai_quartile_bootstrap_summary.csv
    One row per method × window × stratum with observed rho, CI bounds, and
    provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    load_residual_dataset,
    make_supplementary_checks_dirs,
    SUPPLEMENTARY_CHECKS_RESULTS,
)

OUTCOME = "sif_771nm"
WINDOWS = [21, 28]
METHODS = ["harmonic", "cyclic_spline"]
N_BOOT = 1000
CI_LEVEL = 0.95
ELIGIBILITY_MIN_CELLS = 50
CLUSTER_DEFINITION = "0.5° × 0.5° grid cell (lat_id, lon_id)"


def _derived_seed(method: str, window: int, group_key: str) -> int:
    """Deterministic seed from method, window and stratum label."""
    return (
        int(hashlib.sha256(f"lai_bootstrap_{method}_{window}_{group_key}".encode()).hexdigest(), 16)
        % 2**31
    )


def _filter_group(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """Return observations for one analytical stratum."""
    if group_key == "control_vegetated":
        return df[df["is_vegetated"] == True].copy()
    if group_key.startswith("lai_quartile_"):
        q = int(group_key.split("_")[-1])
        return df[df["lai_quartile"] == q].copy()
    raise ValueError(f"Unknown group: {group_key}")


def _lai_bounds(boundaries: dict[str, dict], group_key: str) -> tuple[float | None, float | None]:
    """Return (lower, upper) LAI bounds for display; control_vegetated gets thresholds."""
    if group_key == "control_vegetated":
        # Thresholds correspond to the functional vegetated definition.
        return (1.0, None)
    q = group_key.split("_")[-1]
    b = boundaries.get(q, {})
    return (b.get("min"), b.get("max"))


def _compute_observed_rho(df: pd.DataFrame, residual_col: str, sii_col: str) -> float:
    """Observed Spearman rho on finite pairs."""
    valid = df[[residual_col, sii_col]].dropna()
    if len(valid) < 10:
        return float("nan")
    return float(compute_stats(valid[residual_col].values, valid[sii_col].values)["spearman_rho"])


def _bootstrap_draws(
    df: pd.DataFrame,
    residual_col: str,
    sii_col: str,
    n_boot: int,
    seed: int,
) -> list[float]:
    """Return the full vector of bootstrap rho estimates for archiving."""
    rng = np.random.default_rng(seed)
    needed = [residual_col, sii_col, "lat_id", "lon_id"]
    valid = df[needed].dropna()
    if len(valid) < 10:
        return []

    cells = valid[["lat_id", "lon_id"]].drop_duplicates().values
    cell_idx = valid[["lat_id", "lon_id"]].apply(tuple, axis=1).values
    cell_map = {tuple(c): i for i, c in enumerate(cells)}
    labels = np.array([cell_map[t] for t in cell_idx])
    n_cells = len(cells)

    x = valid[residual_col].values.astype(float)
    y = valid[sii_col].values.astype(float)

    rhos = []
    for _ in range(n_boot):
        sampled_cells = rng.integers(0, n_cells, size=n_cells)
        keep = np.isin(labels, sampled_cells)
        if keep.sum() < 10:
            continue
        ix = np.where(keep)[0]
        boot_rho = float(compute_stats(x[ix], y[ix])["spearman_rho"])
        rhos.append(boot_rho)
    return rhos


def build_summary_and_draws() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build bootstrap summary and draws tables."""
    boundaries = json.loads(
        (PROJECT_ROOT / "data" / "interim" / "lai_quartile_boundaries.json").read_text(encoding="utf-8")
    )

    groups = ["control_vegetated"] + [f"lai_quartile_{q}" for q in (1, 2, 3, 4)]

    summary_rows: list[dict] = []
    draw_rows: list[dict] = []

    for method in METHODS:
        residuals = load_residual_dataset(method, OUTCOME)
        for window in WINDOWS:
            df = attach_exposure(residuals, build_sii_exposure([window]))
            sii_col = f"sii_{window}d"

            for group_key in groups:
                sub = _filter_group(df, group_key)
                n_obs = len(sub)
                n_cells = sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]
                eligible = n_cells >= ELIGIBILITY_MIN_CELLS
                seed = _derived_seed(method, window, group_key)

                observed_rho = _compute_observed_rho(sub, "residual", sii_col)

                if eligible:
                    rhos = _bootstrap_draws(sub, "residual", sii_col, N_BOOT, seed)
                    n_finite = len(rhos)
                    if n_finite >= N_BOOT // 2:
                        ci_low = float(np.quantile(rhos, (1 - CI_LEVEL) / 2))
                        ci_high = float(np.quantile(rhos, 1 - (1 - CI_LEVEL) / 2))
                    else:
                        ci_low = ci_high = float("nan")
                        n_finite = 0
                else:
                    rhos = []
                    ci_low = ci_high = float("nan")
                    n_finite = 0

                lo_bound, hi_bound = _lai_bounds(boundaries, group_key)
                lai_quartile = None if group_key == "control_vegetated" else int(group_key.split("_")[-1])

                summary_rows.append({
                    "lai_quartile": lai_quartile,
                    "lai_lower_bound": lo_bound,
                    "lai_upper_bound": hi_bound,
                    "method": method,
                    "sii_window_days": window,
                    "observed_rho": observed_rho,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_bootstrap": n_finite,
                    "n_cells": n_cells,
                    "n_observations": n_obs,
                    "seed": seed,
                    "cluster_definition": CLUSTER_DEFINITION,
                    "eligibility_status": "eligible" if eligible else f"ineligible: fewer_than_{ELIGIBILITY_MIN_CELLS}_cells",
                })

                for rep_id, rho in enumerate(rhos, start=1):
                    draw_rows.append({
                        "method": method,
                        "sii_window_days": window,
                        "group": group_key,
                        "lai_quartile": lai_quartile,
                        "replicate_id": rep_id,
                        "bootstrap_rho": rho,
                    })

    summary = pd.DataFrame(summary_rows)
    draws = pd.DataFrame(draw_rows)
    return summary, draws


def _validate(summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    """Run focused validation assertions."""
    # Observed rho must match the existing fixed-window table for harmonic 28-day rows.
    fixed = pd.read_csv(PROJECT_ROOT / "results" / "fixed_window_results.csv")

    for _, row in summary.iterrows():
        if row["method"] != "harmonic" or row["sii_window_days"] != 28:
            continue
        if pd.notna(row["lai_quartile"]):
            ref = fixed[
                (fixed["sample_type"] == "lai_quartile")
                & (fixed["method"] == "harmonic")
                & (fixed["sii_window"] == "sii_28d")
                & (fixed["lai_quartile"] == f"Q{int(row['lai_quartile'])}")
            ]
        else:
            ref = fixed[
                (fixed["sample_type"] == "control_vegetated")
                & (fixed["method"] == "harmonic")
                & (fixed["sii_window"] == "sii_28d")
            ]
        if not ref.empty:
            expected = float(ref.iloc[0]["spearman_rho"])
            if not (pd.isna(row["observed_rho"]) and pd.isna(expected)):
                assert np.isclose(row["observed_rho"], expected, atol=1e-6), (
                    f"Observed rho mismatch for {row['method']} {row['sii_window_days']} Q{row['lai_quartile']}: "
                    f"{row['observed_rho']} vs {expected}"
                )

    # Eligible rows must have 1000 finite bootstrap estimates.
    eligible = summary[summary["eligibility_status"] == "eligible"]
    assert (eligible["n_bootstrap"] == N_BOOT).all(), "Eligible rows must have 1000 bootstrap estimates"

    # No row may fall below the eligibility cell threshold without being marked ineligible.
    assert (
        summary[summary["n_cells"] < ELIGIBILITY_MIN_CELLS]["eligibility_status"].str.startswith("ineligible").all()
    ), "Rows below cell threshold must be marked ineligible"

    # Quartiles are mutually exclusive and collectively exhaustive within each method/window.
    for (method, window), g in summary.groupby(["method", "sii_window_days"]):
        quart = g[g["lai_quartile"].notna()]
        total_obs = quart["n_observations"].sum()
        # Compare to full sample for that method/window.
        residuals = load_residual_dataset(method, OUTCOME)
        full_obs = len(residuals)
        assert total_obs == full_obs, (
            f"Quartile observation sum {total_obs} != full sample {full_obs} for {method} {window}"
        )

    print("[VALIDATION] All LAI-quartile bootstrap checks passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation assertions (not recommended).",
    )
    args = parser.parse_args(argv)

    make_supplementary_checks_dirs()

    print("[S09b] Computing LAI-quartile cluster-bootstrap confidence intervals ...")
    summary, draws = build_summary_and_draws()

    if not args.skip_validation:
        _validate(summary, draws)

    draws_path = SUPPLEMENTARY_CHECKS_RESULTS / "lai_quartile_bootstrap_draws.parquet"
    summary_path = SUPPLEMENTARY_CHECKS_RESULTS / "lai_quartile_bootstrap_summary.csv"

    atomic_write(draws, draws_path, fmt="parquet")
    atomic_write(summary, summary_path, fmt="csv")

    print(f"[OK] Draws: {draws_path} ({len(draws)} rows)")
    print(f"[OK] Summary: {summary_path} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
