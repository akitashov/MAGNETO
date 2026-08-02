#!/usr/bin/env python3
"""Table S2. Analytical sample and scenario definitions.

Combines control definitions, diagnostics, and fixed-window sample sizes. No
computations are rerun.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import load_csv, load_json, load_parquet, project_root, write_table


def build_table_s2() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    defs = load_json(root / "results" / "control_definitions.json")
    fixed = load_csv(root / "results" / "fixed_window_results.csv")
    landcover = load_csv(root / "results" / "supplementary_checks" / "landcover_analysis.csv")
    env_input = load_parquet(root / "data" / "processed" / "environmental_driver_input.parquet")

    # Sample definitions and sizes for main scenarios.
    main = [
        ("Full sample", "All land-only 0.5° cells passing QC", "pooled_full"),
        ("Persistently vegetated", "Median LAI ≥ 1.0, 10th percentile LAI ≥ 0.30, at least 10 valid LAI observations", "control_vegetated"),
        ("Strict low-LAI control", "Median LAI ≤ 0.10, 90th percentile LAI ≤ 0.30, at least 10 valid LAI observations", "control_strict_low_lai"),
        ("Sahara", "Geographic box 15–32° N, 18° W–40° E", "control_Sahara"),
    ]

    rows = []
    for label, definition, sample_type in main:
        sub = fixed[
            (fixed["sample_type"] == sample_type)
            & (fixed["method"] == "harmonic")
            & (fixed["sii_window"] == "sii_28d")
        ]
        if sub.empty:
            n_cells = None
            n_obs = None
        else:
            n_cells = sub.iloc[0]["n_cells"]
            n_obs = sub.iloc[0]["n_obs"]
        rows.append({
            "Scenario / class": label,
            "Definition / threshold": definition,
            "n cells": n_cells,
            "n observations": n_obs,
        })

    # SAA is not in fixed_window_results; derive counts from the processed input.
    saa_cells = env_input.loc[env_input["is_SAA"] == True, ["lat_id", "lon_id"]].drop_duplicates()
    saa_obs = env_input[env_input["is_SAA"] == True]
    rows.append({
        "Scenario / class": "SAA",
        "Definition / threshold": "Rectangular geographic approximation 50° S–0°, 81°–35° W of the South Atlantic Anomaly",
        "n cells": len(saa_cells),
        "n observations": len(saa_obs),
    })

    # Land-cover classes from pooled landcover analysis.
    lc_sub = landcover[
        (landcover["method"] == "harmonic")
        & (landcover["sii_window"] == "sii_28d")
        & (landcover["sample_type"].str.startswith("landcover_"))
        & (~landcover["sample_type"].str.startswith("landcover_temp_"))
    ].copy()
    lc_sub["class"] = lc_sub["sample_type"].str.replace("landcover_", "")
    lc_cells = 0
    lc_obs = 0
    for _, r in lc_sub.sort_values("class").iterrows():
        class_name = r["class"]
        display_class = "Barren land-cover class" if class_name == "Barren" else class_name
        rows.append({
            "Scenario / class": display_class,
            "Definition / threshold": f"Dominant 2022 MCD12C1 class = {class_name}",
            "n cells": r["n_cells"],
            "n observations": r["n_obs"],
        })
        lc_cells += r["n_cells"]
        lc_obs += r["n_obs"]

    # Coverage gap between full sample and summed land-cover classes.
    full_cells = int(fixed.loc[
        (fixed["sample_type"] == "pooled_full")
        & (fixed["method"] == "harmonic")
        & (fixed["sii_window"] == "sii_28d"),
        "n_cells",
    ].iloc[0])
    full_obs = int(fixed.loc[
        (fixed["sample_type"] == "pooled_full")
        & (fixed["method"] == "harmonic")
        & (fixed["sii_window"] == "sii_28d"),
        "n_obs",
    ].iloc[0])
    rows.append({
        "Scenario / class": "Other / unclassified / not assigned",
        "Definition / threshold": "Cells not assigned to one of the six aggregated land-cover classes",
        "n cells": full_cells - lc_cells,
        "n observations": full_obs - lc_obs,
    })

    df = pd.DataFrame(rows)

    notes = [
        "Cell and observation counts correspond to the harmonic-detrended, 28-day SII window analysis.",
        "Strict low-LAI control criteria: median LAI ≤ 0.10, 90th percentile LAI ≤ 0.30, at least 10 valid LAI observations; persistently vegetated criteria: median LAI ≥ 1.0, 10th percentile LAI ≥ 0.30, at least 10 valid LAI observations.",
        f"The six aggregated land-cover classes sum to {lc_cells:,} cells and {lc_obs:,} observations; the residual {full_cells - lc_cells:,} cells and {full_obs - lc_obs:,} observations are listed separately as Other / unclassified / not assigned.",
    ]
    return df, notes


def main() -> int:
    df, notes = build_table_s2()
    write_table(df, "tableS2_sample_definitions", "Table S2. Analytical sample and scenario definitions", notes=notes)
    print(f"[OK] Table S2: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
