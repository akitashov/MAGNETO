#!/usr/bin/env python3
"""Table 1. Primary SII–SIF associations and key robustness analyses.

Builds a compact summary of the main manuscript rows from existing fixed-window
and surrogate-summary tables. No analytical computations are rerun.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import (
    fmt_ci,
    fmt_int,
    fmt_method,
    fmt_number,
    fmt_p,
    load_csv,
    project_root,
    write_table,
)


def build_table1() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    fixed = load_csv(root / "results" / "fixed_window_results.csv")
    surrogates = load_csv(root / "results" / "surrogate_summary.csv")
    boot = load_csv(root / "results" / "supplementary_checks" / "table1_bootstrap_summary.csv")

    # Map requested labels to internal sample_type values.
    rows = [
        ("Full sample", "harmonic", "sii_28d", "pooled_full"),
        ("Full sample (pairwise matched)", "cyclic_spline", "sii_28d", "pairwise_matched"),
        ("Full sample", "harmonic", "sii_21d", "pooled_full"),
        ("Persistently vegetated", "harmonic", "sii_28d", "control_vegetated"),
        ("Sahara control", "harmonic", "sii_28d", "control_Sahara"),
    ]

    out_rows = []
    for sample_label, method, window, sample_type in rows:
        row = fixed[
            (fixed["sample_type"] == sample_type)
            & (fixed["method"] == method)
            & (fixed["sii_window"] == window)
        ]
        if row.empty:
            out_rows.append({
                "Sample": sample_label,
                "Detrending method": fmt_method(method),
                "SII window (days)": int(window.split("_")[1].replace("d", "")),
                "n cells": None,
                "n observations": None,
                "Spearman ρ [95% CI]": None,
                "Effect per 100 nT": None,
                "Maximum empirical p": None,
            })
            continue

        r = row.iloc[0]

        # Pivot surrogate p-values.
        s = surrogates[
            (surrogates["sample_type"] == sample_type)
            & (surrogates["method"] == method)
            & (surrogates["sii_window"] == window)
        ]
        p_map = {}
        for _, sr in s.iterrows():
            p_map[sr["surrogate_mode"]] = sr["p_value"]
        p_values = [p_map.get(m) for m in ("year_perm", "circ_shift", "block_perm")]
        p_values = [p for p in p_values if pd.notna(p)]
        p_max = max(p_values) if p_values else None

        # Cluster-bootstrap CI for this row.
        boot_match = boot[
            (boot["sample_type"] == sample_type)
            & (boot["method"] == method)
            & (boot["sii_window"] == window)
        ]
        boot_r = boot_match.iloc[0] if not boot_match.empty else None

        out_rows.append({
            "Sample": sample_label,
            "Detrending method": fmt_method(method),
            "SII window (days)": int(window.split("_")[1].replace("d", "")),
            "n cells": r["n_cells"],
            "n observations": r["n_obs"],
            "Spearman ρ [95% CI]": (
                r["spearman_rho"],
                boot_r["bootstrap_ci_low"] if boot_r is not None else None,
                boot_r["bootstrap_ci_high"] if boot_r is not None else None,
            ),
            "Effect per 100 nT": r.get("ols_slope_per_100nT"),
            "Maximum empirical p": p_max,
        })

    df = pd.DataFrame(out_rows)

    # Format for display (Markdown/DOCX).
    def _fmt_rho_ci(row):
        rho, lo, hi = row["Spearman ρ [95% CI]"]
        if rho is None or pd.isna(rho):
            return ""
        rho_s = fmt_number(rho, 3)
        ci_s = fmt_ci(lo, hi, 3)
        return f"{rho_s} {ci_s}".strip()

    display = pd.DataFrame({
        "Sample": df["Sample"],
        "Detrending method": df["Detrending method"],
        "SII window (days)": df["SII window (days)"].apply(fmt_int),
        "n cells": df["n cells"].apply(fmt_int),
        "n observations": df["n observations"].apply(fmt_int),
        "Spearman ρ [95% CI]": df.apply(_fmt_rho_ci, axis=1),
        "Effect per 100 nT": df["Effect per 100 nT"].apply(lambda x: fmt_number(x, 4)),
        "Maximum empirical p": df["Maximum empirical p"].apply(lambda x: fmt_p(x, empirical=True)),
    })

    notes = [
        "Maximum empirical p is the maximum of the three temporal-surrogate p-values (year permutation, circular shift, 30-day block permutation); it is a conservative summary, not a single formal test. Values are floored at 0.001 (B = 1000 realizations).",
        "Spearman ρ values are followed by 95% cluster-bootstrap confidence intervals obtained from 1,000 replicates by resampling spatial grid cells and retaining all observations within each selected cell.",
        "Effect per 100 nT = OLS slope of SIF residual on SII expressed per 100 nT of cumulative geomagnetic activity.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table1()
    write_table(
        df,
        "table1_primary_results",
        "Table 1. Primary SII–SIF associations and key robustness analyses",
        notes=notes,
        landscape=True,
    )
    print(f"[OK] Table 1: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
