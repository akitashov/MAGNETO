#!/usr/bin/env python3
"""Table S10. Additional SIF wavelength sensitivity.

Reports SIF 757 nm and provider-derived SIF 740 nm associations with SII for the
outcome-specific full sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_int, fmt_method, fmt_number, fmt_p, load_csv, project_root, write_table


def build_table_s10() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    effects = load_csv(root / "results" / "supplementary_outcomes_results.csv")
    surrogates = load_csv(root / "results" / "supplementary_outcomes_surrogate_summary.csv")

    df = effects[
        (effects["sample_type"] == "outcome_specific_full")
        & (effects["outcome"].isin(["sif_757nm", "sif_740nm"]))
    ].copy()

    # Pivot surrogate p-values.
    s_sub = surrogates[surrogates["sample_type"] == "outcome_specific_full"].copy()
    p_pivot = s_sub.pivot_table(
        index=["outcome", "method", "sii_window"],
        columns="surrogate_mode",
        values="p_value",
        aggfunc="first",
    ).reset_index()

    merged = pd.merge(df, p_pivot, on=["outcome", "method", "sii_window"], how="left")

    outcome_map = {
        "sif_757nm": "SIF 757 nm (direct)",
        "sif_740nm": "SIF 740 nm (provider-derived)",
    }
    merged["Outcome"] = merged["outcome"].map(outcome_map)
    merged["Method"] = merged["method"].apply(fmt_method)
    merged["SII window (days)"] = merged["sii_window_days"].astype(int)

    merged = merged.sort_values(["Outcome", "Method", "SII window (days)"]).reset_index(drop=True)

    display = pd.DataFrame({
        "Outcome": merged["Outcome"],
        "Method": merged["Method"],
        "SII window (days)": merged["SII window (days)"].apply(fmt_int),
        "Spearman rho": merged["spearman_rho"].apply(lambda x: fmt_number(x, 3)),
        "Effect per 1 SD SII": merged["std_effect_per_1sd_sii"].apply(lambda x: fmt_number(x, 5)),
        "n observations": merged["n_obs"].apply(fmt_int),
        "n cells": merged["n_cells"].apply(fmt_int),
        "Year-permutation p": merged["year_perm"].apply(lambda x: fmt_p(x, empirical=True)),
        "Circular-shift p": merged["circ_shift"].apply(lambda x: fmt_p(x, empirical=True)),
        "30-day-block p": merged["block_perm"].apply(lambda x: fmt_p(x, empirical=True)),
    })

    notes = [
        "SIF 740 nm is provider-derived as 0.75 × (SIF 757 nm + 1.5 × SIF 771 nm). It is algebraically derived from SIF 757 nm and SIF 771 nm and therefore is not an independent wavelength replication.",
        "Empirical temporal-surrogate p-values are floored at 0.001 (B = 1000).",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s10()
    write_table(df, "tableS10_sif_wavelength", "Table S10. Sensitivity of the SII association to SIF 757 nm and provider-derived SIF 740 nm", notes=notes, landscape=True)
    print(f"[OK] Table S10: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
