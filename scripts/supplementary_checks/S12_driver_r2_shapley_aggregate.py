#!/usr/bin/env python3
"""
S12_driver_r2_shapley_aggregate.py — Plotting-source aggregate for the
Shapley/LMG R² decomposition.

Reads the S11 long table and produces a wide table with one row per
scenario × temperature × window. Does not recompute models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _Common import PROJECT_ROOT, atomic_write

TOL = 1e-10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate Shapley/LMG R² decomposition")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Read from and write to results/supplementary_checks/smoke/.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = (
        PROJECT_ROOT / "results" / "supplementary_checks" / "smoke"
        if args.smoke_test
        else PROJECT_ROOT / "results" / "supplementary_checks"
    )

    input_path = out_dir / "driver_r2_shapley_long.csv"
    output_path = out_dir / "driver_r2_shapley_by_window.csv"

    print("=" * 64)
    print("MAGNETO supplementary checks — Driver R² Shapley/LMG aggregate")
    print("=" * 64)
    print(f"Smoke mode: {args.smoke_test}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{input_path}\n"
            "Run S11_driver_r2_shapley.py first."
        )

    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df):,} rows")

    required = ["scenario", "temp_bin_label", "window_days", "driver",
                "shapley_r2", "r2_full", "beta_standardized", "n_obs",
                "n_cells", "eligible", "eligibility_reason"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in S11 output: {missing}")

    # Pivot driver-specific fields.
    pivot_r2 = df.pivot_table(
        index=["scenario", "temp_bin_label", "window_days"],
        columns="driver",
        values="shapley_r2",
    ).reset_index()
    pivot_r2.columns.name = None
    pivot_r2 = pivot_r2.rename(
        columns={d: f"{d.lower()}_shapley_r2" for d in ["SII", "PAR", "VPD"]}
    )

    pivot_beta = df.pivot_table(
        index=["scenario", "temp_bin_label", "window_days"],
        columns="driver",
        values="beta_standardized",
    ).reset_index()
    pivot_beta.columns.name = None
    pivot_beta = pivot_beta.rename(
        columns={d: f"{d.lower()}_beta_standardized" for d in ["SII", "PAR", "VPD"]}
    )

    # One set of model-level metadata per group.
    meta = (
        df.groupby(["scenario", "temp_bin_label", "window_days"])
        .agg(
            r2_full=("r2_full", "first"),
            n_obs=("n_obs", "first"),
            n_cells=("n_cells", "first"),
            eligible=("eligible", "all"),
            eligibility_reason=("eligibility_reason", lambda x: "; ".join(sorted(set(x.dropna())))),
        )
        .reset_index()
    )

    out = meta.merge(pivot_r2, on=["scenario", "temp_bin_label", "window_days"], how="left")
    out = out.merge(pivot_beta, on=["scenario", "temp_bin_label", "window_days"], how="left")

    out["sii_shapley_percent"] = 100.0 * out["sii_shapley_r2"]
    out["par_shapley_percent"] = 100.0 * out["par_shapley_r2"]
    out["vpd_shapley_percent"] = 100.0 * out["vpd_shapley_r2"]
    out["r2_full_percent"] = 100.0 * out["r2_full"]

    # Validate additive decomposition for eligible rows.
    eligible = out[out["eligible"]].copy()
    if not eligible.empty:
        shapley_sum = (
            eligible["sii_shapley_r2"]
            + eligible["par_shapley_r2"]
            + eligible["vpd_shapley_r2"]
        )
        max_err = float((shapley_sum - eligible["r2_full"]).abs().max())
        if max_err > TOL:
            bad = eligible[(shapley_sum - eligible["r2_full"]).abs() > TOL]
            raise ValueError(
                f"Shapley sum != r2_full (max error {max_err:.2e}):\n"
                + bad[["scenario", "temp_bin_label", "window_days", "r2_full"]].to_string(index=False)
            )
        print(f"Additive decomposition validated (max error {max_err:.2e})")

    out = out.sort_values(["scenario", "temp_bin_label", "window_days"]).reset_index(drop=True)

    column_order = [
        "scenario", "temp_bin_label", "window_days",
        "sii_shapley_r2", "par_shapley_r2", "vpd_shapley_r2",
        "sii_shapley_percent", "par_shapley_percent", "vpd_shapley_percent",
        "r2_full", "r2_full_percent",
        "sii_beta_standardized", "par_beta_standardized", "vpd_beta_standardized",
        "n_obs", "n_cells", "eligible", "eligibility_reason",
    ]
    out = out[[c for c in column_order if c in out.columns]]

    atomic_write(out, output_path, fmt="csv")
    print(f"Written: {output_path} ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
