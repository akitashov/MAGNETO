#!/usr/bin/env python3
"""Table S7. Shapley/LMG decomposition at the common 28-day window.

Reports the exact additive allocation of full-model R² among SII, PAR and VPD
for each scenario and temperature bin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_int, fmt_number, load_csv, project_root, write_table


def build_table_s7() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    df = load_csv(root / "results" / "supplementary_checks" / "driver_r2_shapley_by_window.csv")

    df = df[df["window_days"] == 28].copy()

    scenario_map = {
        "full_sample": "Full sample",
        "persistently_vegetated": "Persistently vegetated",
        "control_strict_low_lai": "Strict low-LAI control",
        "Sahara": "Sahara",
        "SAA": "SAA",
    }
    df["Scenario"] = df["scenario"].map(scenario_map)

    temp_order = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]
    df["temp_bin_label"] = pd.Categorical(df["temp_bin_label"], categories=temp_order, ordered=True)
    df = df.sort_values(["Scenario", "temp_bin_label"]).reset_index(drop=True)

    temp_label_map = {
        "Frozen": "Frozen",
        "Cold": "Cold",
        "Cool": "Cool",
        "Optimum": "Optimum",
        "Warm_Stress": "Warm stress",
        "Extreme_Heat": "Extreme heat",
    }
    df["Temperature regime"] = df["temp_bin_label"].map(temp_label_map)

    display = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Temperature regime": df["Temperature regime"],
        "Full-model R² (percentage points)": df["r2_full_percent"].apply(lambda x: fmt_number(x, 3)),
        "SII contribution (R² percentage points)": df["sii_shapley_percent"].apply(lambda x: fmt_number(x, 3)),
        "PAR contribution (R² percentage points)": df["par_shapley_percent"].apply(lambda x: fmt_number(x, 3)),
        "VPD contribution (R² percentage points)": df["vpd_shapley_percent"].apply(lambda x: fmt_number(x, 3)),
        "n observations": df["n_obs"].apply(fmt_int),
        "n cells": df["n_cells"].apply(fmt_int),
        "Eligible": df["eligible"].apply(lambda x: "Yes" if x else "No"),
    })

    notes = [
        "Contributions are exact Shapley/LMG allocations of ordinary full-model R², reported in R² percentage points (100 × R²).",
        "SII + PAR + VPD contributions sum to the full-model R² within numerical tolerance for every eligible model.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s7()
    write_table(df, "tableS7_shapley", "Table S7. Shapley/LMG decomposition of residual SIF variance at the 28-day common window", notes=notes, landscape=True)
    print(f"[OK] Table S7: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
