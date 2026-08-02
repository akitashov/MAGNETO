#!/usr/bin/env python3
"""Table S4. Temporal-surrogate inference.

Summarizes the three temporal null models (year permutation, circular shift,
30-day block permutation) for the primary analysis rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_ci, fmt_int, fmt_method, fmt_number, fmt_p, load_csv, project_root, write_table


def build_table_s4() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    sur = load_csv(root / "results" / "surrogate_summary.csv")

    sample_map = {
        "pooled_full": "Full sample",
        "pairwise_matched": "Pairwise matched",
        "control_vegetated": "Persistently vegetated",
        "control_strict_low_lai": "Strict low-LAI control",
        "control_Sahara": "Sahara control",
    }
    basis_map = {
        "pooled_full": "Native method-specific sample",
        "pairwise_matched": "Pairwise matched",
        "control_vegetated": "Native method-specific sample",
        "control_strict_low_lai": "Native method-specific sample",
        "control_Sahara": "Native method-specific sample",
    }
    mode_map = {
        "year_perm": "Year permutation",
        "circ_shift": "Circular shift",
        "block_perm": "30-day block permutation",
    }

    df = sur[sur["sample_type"].isin(sample_map.keys())].copy()
    df["Sample"] = df["sample_type"].map(sample_map)
    df["Sample basis"] = df["sample_type"].map(basis_map)
    df["Surrogate mode"] = df["surrogate_mode"].map(mode_map)
    df["Method"] = df["method"].apply(fmt_method)
    df["SII window (days)"] = df["sii_window"].str.extract(r'(\d+)').astype(int)

    df = df.sort_values(["Sample basis", "Sample", "Method", "SII window (days)", "Surrogate mode"]).reset_index(drop=True)

    display = pd.DataFrame({
        "Sample basis": df["Sample basis"],
        "Sample": df["Sample"],
        "Method": df["Method"],
        "SII window (days)": df["SII window (days)"].apply(fmt_int),
        "Surrogate mode": df["Surrogate mode"],
        "Observed rho": df["rho_obs"].apply(lambda x: fmt_number(x, 3)),
        "Null median": df["null_q50"].apply(lambda x: fmt_number(x, 3)),
        "Null 95% interval": df.apply(lambda r: fmt_ci(r["null_q025"], r["null_q975"]), axis=1),
        "Empirical p": df["p_value"].apply(lambda x: fmt_p(x, empirical=True)),
        "B": df["n_completed"].apply(fmt_int),
    })

    notes = [
        "Empirical p-values are plus-one p-values from B = 1000 surrogate realizations, floored at 0.001.",
        "Null 95 % interval is the 2.5th–97.5th percentile of the surrogate null distribution.",
        "SAA surrogate rows are not present in results/surrogate_summary.csv; SAA inference is reported in Table S3 and Table S5.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s4()
    write_table(df, "tableS4_temporal_surrogate", "Table S4. Temporal-surrogate inference for the primary analysis rows", notes=notes, landscape=True)
    print(f"[OK] Table S4: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
