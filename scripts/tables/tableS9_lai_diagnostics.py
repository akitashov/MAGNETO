#!/usr/bin/env python3
"""Table S9. LAI diagnostics.

Reports persistent-vegetation thresholds and LAI-strata associations using
existing fixed-window and surrogate-summary tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_ci, fmt_int, fmt_number, fmt_p, load_csv, load_json, project_root, write_table


def build_table_s9() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    defs = load_json(root / "results" / "control_definitions.json")
    boundaries = load_json(root / "data" / "interim" / "lai_quartile_boundaries.json")
    fixed = load_csv(root / "results" / "fixed_window_results.csv")
    surrogates = load_csv(root / "results" / "surrogate_summary.csv")
    boot = load_csv(root / "results" / "supplementary_checks" / "lai_quartile_bootstrap_summary.csv")

    rows = []

    # Persistent vegetation threshold row.
    veg_def = defs["functional"]["vegetated"]["criteria"]
    veg = fixed[
        (fixed["sample_type"] == "control_vegetated")
        & (fixed["method"] == "harmonic")
        & (fixed["sii_window"] == "sii_28d")
    ]
    if not veg.empty:
        r = veg.iloc[0]
        s = surrogates[
            (surrogates["sample_type"] == "control_vegetated")
            & (surrogates["method"] == "harmonic")
            & (surrogates["sii_window"] == "sii_28d")
        ]
        p_map = {sr["surrogate_mode"]: sr["p_value"] for _, sr in s.iterrows()}
        boot_match = boot[
            (boot["method"] == "harmonic")
            & (boot["sii_window_days"] == 28)
            & (boot["lai_quartile"].isna())
        ]
        boot_r = boot_match.iloc[0] if not boot_match.empty else None
        rows.append({
            "LAI stratum / threshold": "Persistently vegetated",
            "Definition": "Median LAI ≥ 1.0, 10th percentile LAI ≥ 0.30, at least 10 valid LAI observations",
            "n observations": r["n_obs"],
            "n cells": r["n_cells"],
            "Spearman rho": r["spearman_rho"],
            "95% bootstrap CI": fmt_ci(
                boot_r["bootstrap_ci_low"] if boot_r is not None else None,
                boot_r["bootstrap_ci_high"] if boot_r is not None else None,
            ),
            "Year-permutation p": p_map.get("year_perm"),
            "Circular-shift p": p_map.get("circ_shift"),
            "30-day-block p": p_map.get("block_perm"),
        })

    # LAI quartile rows.
    quartiles = fixed[
        (fixed["sample_type"] == "lai_quartile")
        & (fixed["method"] == "harmonic")
        & (fixed["sii_window"] == "sii_28d")
    ].copy()
    quartiles["lai_quartile"] = quartiles["lai_quartile"].astype(str)
    # Surrogate summary stores lai_quartile as float; convert to Q1…Q4 for matching.
    surrogates["lai_quartile_q"] = (
        pd.to_numeric(surrogates["lai_quartile"], errors="coerce")
        .dropna()
        .astype(int)
        .apply(lambda x: f"Q{x}")
    )
    for _, r in quartiles.sort_values("lai_quartile").iterrows():
        q = r["lai_quartile"]
        q_num = int(q[1])
        b = boundaries.get(q, {})
        lo = b.get("min")
        hi = b.get("max")
        if lo is not None and hi is not None:
            if q == "Q4":
                definition = f"Global LAI quartile {q}: {fmt_number(lo, 3)} ≤ median LAI ≤ {fmt_number(hi, 3)}"
            else:
                definition = f"Global LAI quartile {q}: {fmt_number(lo, 3)} ≤ median LAI < {fmt_number(hi, 3)}"
        else:
            definition = f"Global LAI quartile {q}"
        s = surrogates[
            (surrogates["sample_type"] == "lai_quartile")
            & (surrogates["method"] == "harmonic")
            & (surrogates["sii_window"] == "sii_28d")
            & (surrogates["lai_quartile_q"] == q)
        ]
        p_map = {sr["surrogate_mode"]: sr["p_value"] for _, sr in s.iterrows()}
        boot_match = boot[
            (boot["method"] == "harmonic")
            & (boot["sii_window_days"] == 28)
            & (boot["lai_quartile"] == q_num)
        ]
        boot_r = boot_match.iloc[0] if not boot_match.empty else None
        rows.append({
            "LAI stratum / threshold": f"LAI quartile {q}",
            "Definition": definition,
            "n observations": r["n_obs"],
            "n cells": r["n_cells"],
            "Spearman rho": r["spearman_rho"],
            "95% bootstrap CI": fmt_ci(
                boot_r["bootstrap_ci_low"] if boot_r is not None else None,
                boot_r["bootstrap_ci_high"] if boot_r is not None else None,
            ),
            "Year-permutation p": p_map.get("year_perm"),
            "Circular-shift p": p_map.get("circ_shift"),
            "30-day-block p": p_map.get("block_perm"),
        })

    df = pd.DataFrame(rows)

    display = pd.DataFrame({
        "LAI stratum / threshold": df["LAI stratum / threshold"],
        "Definition": df["Definition"],
        "n observations": df["n observations"].apply(fmt_int),
        "n cells": df["n cells"].apply(fmt_int),
        "Spearman rho": df["Spearman rho"].apply(lambda x: fmt_number(x, 3)),
        "95% bootstrap CI": df["95% bootstrap CI"],
        "Year-permutation p": df["Year-permutation p"].apply(lambda x: fmt_p(x, empirical=True)),
        "Circular-shift p": df["Circular-shift p"].apply(lambda x: fmt_p(x, empirical=True)),
        "30-day-block p": df["30-day-block p"].apply(lambda x: fmt_p(x, empirical=True)),
    })

    notes = [
        "Confidence intervals were obtained from 1,000 cluster-bootstrap replicates by resampling spatial grid cells while retaining all observations within each selected cell.",
        "Bootstrap confidence intervals quantify spatial sampling uncertainty, whereas temporal-surrogate p-values evaluate the association against alternative temporal null structures; a confidence interval excluding zero alongside a nonsignificant surrogate p-value is therefore not a contradiction.",
        "Empirical temporal-surrogate p-values are floored at 0.001 (B = 1000).",
        "LAI quartile boundaries are based on the distribution of cell-level median LAI across the full analytical grid. Displayed cut points are rounded; assignment used the unrounded values in data/interim/lai_quartile_boundaries.json.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s9()
    write_table(df, "tableS9_lai_diagnostics", "Table S9. LAI-stratification of SII–SIF associations", notes=notes, landscape=True)
    print(f"[OK] Table S9: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
