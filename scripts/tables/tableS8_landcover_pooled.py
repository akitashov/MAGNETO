#!/usr/bin/env python3
"""Table S8. Pooled land-cover associations.

Summarizes the SII–SIF association within each dominant land-cover class for the
pooled (non-temperature-stratified) sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_int, fmt_method, fmt_number, fmt_p, load_csv, project_root, write_table


def build_table_s8() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    effects = load_csv(root / "results" / "supplementary_checks" / "landcover_analysis.csv")
    surrogates = load_csv(root / "results" / "supplementary_checks" / "landcover_analysis_surrogates.csv")

    df = effects[
        (effects["method"].isin(["harmonic", "cyclic_spline"]))
        & (effects["sii_window"] == "sii_28d")
        & (effects["sample_type"].str.startswith("landcover_"))
        & (~effects["sample_type"].str.startswith("landcover_temp_"))
    ].copy()
    df["Class"] = df["sample_type"].str.replace("landcover_", "")
    df["Method"] = df["method"].apply(fmt_method)

    # Pivot surrogate p-values.
    s_sub = surrogates[
        (surrogates["method"].isin(["harmonic", "cyclic_spline"]))
        & (surrogates["sii_window"] == "sii_28d")
    ].copy()
    s_sub["Class"] = s_sub["sample_type"].str.replace("landcover_", "")
    s_sub["Method"] = s_sub["method"].apply(fmt_method)

    p_pivot = s_sub.pivot_table(
        index=["Class", "Method"],
        columns="surrogate_mode",
        values="p_value",
        aggfunc="first",
    ).reset_index()

    merged = pd.merge(df, p_pivot, on=["Class", "Method"], how="left")
    merged["p_max"] = merged[["year_perm", "circ_shift", "block_perm"]].max(axis=1)

    class_order = ["Barren", "Shrubland/Savanna", "Savanna", "Grassland", "Cropland", "Forest"]
    merged["Class"] = pd.Categorical(merged["Class"], categories=class_order, ordered=True)
    merged = merged.sort_values(["Class", "Method"]).reset_index(drop=True)

    display = pd.DataFrame({
        "Land-cover class": merged["Class"].astype(str),
        "Method": merged["Method"],
        "Spearman rho": merged["spearman_rho"].apply(lambda x: fmt_number(x, 3)),
        "n observations": merged["n_obs"].apply(fmt_int),
        "n cells": merged["n_cells"].apply(fmt_int),
        "Year-permutation p": merged["year_perm"].apply(lambda x: fmt_p(x, empirical=True)),
        "Circular-shift p": merged["circ_shift"].apply(lambda x: fmt_p(x, empirical=True)),
        "30-day-block p": merged["block_perm"].apply(lambda x: fmt_p(x, empirical=True)),
        "Maximum empirical p across surrogate modes": merged["p_max"].apply(lambda x: fmt_p(x, empirical=True)),
    })

    notes = [
        "Empirical temporal-surrogate p-values are floored at 0.001 (B = 1000).",
        "Maximum empirical p across surrogate modes = max(year-permutation p, circular-shift p, 30-day-block p); it is a conservative summary, not an additional statistical test.",
        "Pooled associations use the 28-day SII window across all temperature regimes within each land-cover class.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s8()
    write_table(df, "tableS8_landcover_pooled", "Table S8. Pooled land-cover associations with SII", notes=notes, landscape=True)
    print(f"[OK] Table S8: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
