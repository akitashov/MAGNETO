#!/usr/bin/env python3
"""Table S3. Fixed-window SII–SIF results.

Combines harmonic and cyclic-spline temperature-scenario outputs for the 21- and
28-day SII windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_ci, fmt_int, fmt_number, fmt_p, load_csv, project_root, write_table


def build_table_s3() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    root = project_root()
    har = load_csv(root / "results" / "supplementary_checks" / "temperature_scenarios_harmonic.csv")
    spl = load_csv(root / "results" / "supplementary_checks" / "temperature_scenarios_spline.csv")

    # Map internal scenario names to manuscript labels.
    scenario_map = {
        "pooled": "Full sample",
        "control_vegetated": "Persistently vegetated",
        "control_strict_low_lai": "Strict low-LAI control",
        "landcover_barren": "Barren land-cover class",
        "Sahara": "Sahara control",
        "SAA": "SAA control",
    }

    har["Method"] = "Harmonic"
    spl["Method"] = "Cyclic spline"
    df = pd.concat([har, spl], ignore_index=True)
    df["Scenario"] = df["scenario"].map(scenario_map)

    # Temperature order and human-readable labels.
    temp_order = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]
    temp_label_map = {
        "Frozen": "Frozen",
        "Cold": "Cold",
        "Cool": "Cool",
        "Optimum": "Optimum",
        "Warm_Stress": "Warm stress",
        "Extreme_Heat": "Extreme heat",
    }
    df["temp_class"] = pd.Categorical(df["temp_class"], categories=temp_order, ordered=True)
    df["Temp. regime"] = df["temp_class"].map(temp_label_map)

    # Eligibility: suppress inferential values for under-powered groups.
    df["eligible"] = df["n_cells"] >= 50
    df["eligibility_reason"] = df["eligible"].apply(lambda x: "" if x else "fewer_than_50_cells")

    # Add explicit not-estimable rows for missing scenario × temperature combinations.
    # Sahara × Frozen is present in the input (1 cell, 3 observations) but not in the
    # temperature-scenario output because it is below the estimation threshold.
    missing_rows = []
    for method in ["Harmonic", "Cyclic spline"]:
        for window in [21, 28]:
            missing_rows.append({
                "Scenario": "Sahara control",
                "Temp. regime": "Frozen",
                "Method": method,
                "window_days": window,
                "rho": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_year": float("nan"),
                "p_circular": float("nan"),
                "p_block": float("nan"),
                "n_obs": 3,
                "n_cells": 1,
                "eligible": False,
                "eligibility_reason": "fewer_than_50_cells",
            })
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    df = df.sort_values(["Scenario", "Method", "window_days", "temp_class"]).reset_index(drop=True)

    def fmt_if_eligible(row, value):
        if row["eligible"] and pd.notna(value):
            return value
        return float("nan")

    # Full source table retains eligibility metadata.
    full = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Temp. regime": df["Temp. regime"],
        "Method": df["Method"],
        "Window (d)": df["window_days"].apply(fmt_int),
        "Spearman rho": df.apply(lambda r: fmt_number(fmt_if_eligible(r, r["rho"]), 3), axis=1),
        "95% bootstrap CI": df.apply(lambda r: fmt_ci(fmt_if_eligible(r, r["ci_low"]), fmt_if_eligible(r, r["ci_high"])), axis=1),
        "n obs.": df["n_obs"].apply(fmt_int),
        "n cells": df["n_cells"].apply(fmt_int),
        "Year perm. p": df.apply(lambda r: fmt_p(fmt_if_eligible(r, r["p_year"]), empirical=True), axis=1),
        "Circular-shift p": df.apply(lambda r: fmt_p(fmt_if_eligible(r, r["p_circular"]), empirical=True), axis=1),
        "Block p": df.apply(lambda r: fmt_p(fmt_if_eligible(r, r["p_block"]), empirical=True), axis=1),
        "Eligible": df["eligible"].apply(lambda x: "Yes" if x else "No"),
    })

    # Compact display table for DOCX/Markdown drops the eligibility flag.
    display = full.drop(columns=["Eligible"]).copy()

    notes = [
        "Empirical temporal-surrogate p-values are floored at 0.001 (B = 1000).",
        "Confidence intervals are cluster-bootstrap 95 % intervals resampling spatial grid cells.",
        "Inferential values are suppressed for combinations with fewer than 50 cells; such rows are retained in the source CSV and marked Ineligible.",
        "Sahara × Frozen is retained as a not-estimable row (1 cell, 3 observations).",
    ]
    return display, full, notes


def main() -> int:
    display, full, notes = build_table_s3()
    write_table(
        display,
        "tableS3_fixed_window_results",
        "Table S3. Fixed-window SII–SIF results by scenario, temperature regime and detrending method",
        notes=notes,
        landscape=True,
        full_df=full,
    )
    print(f"[OK] Table S3: {len(display)} display rows, {len(full)} source rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
