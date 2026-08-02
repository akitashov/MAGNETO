#!/usr/bin/env python3
"""Table S6. Environmental-driver adjustment summary.

Aggregates the full SII × PAR × VPD GPU cube over all PAR/VPD window
specifications. The source CSV contains all SII windows; the DOCX/Markdown view
is restricted to a compact subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_int, fmt_number, fmt_percent, load_csv, project_root, write_table


def build_table_s6() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    root = project_root()
    full = load_csv(root / "results" / "environmental_driver_matrix_full_summary.csv")

    scenario_map = {
        "full_sample": "Full sample",
        "persistently_vegetated": "Persistently vegetated",
    }
    full = full[full["scenario"].isin(scenario_map.keys())].copy()
    full["Scenario"] = full["scenario"].map(scenario_map)

    temp_order = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]
    temp_label_map = {
        "Frozen": "Frozen",
        "Cold": "Cold",
        "Cool": "Cool",
        "Optimum": "Optimum",
        "Warm_Stress": "Warm stress",
        "Extreme_Heat": "Extreme heat",
    }
    full["temp_bin_label"] = pd.Categorical(full["temp_bin_label"], categories=temp_order, ordered=True)
    full["Temperature regime"] = full["temp_bin_label"].map(temp_label_map)
    full = full.sort_values(["Scenario", "temp_bin_label", "sii_window"]).reset_index(drop=True)

    compact_windows = [1, 7, 14, 21, 28]
    compact = full[full["sii_window"].isin(compact_windows)].copy()

    def make_display(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Scenario": df["Scenario"],
            "Temperature regime": df["Temperature regime"],
            "SII window (days)": df["sii_window"].apply(fmt_int),
            "Median adjusted βₛᵢᵢ per 100 nT": (df["median_beta_sii"] * 100).apply(lambda x: fmt_number(x, 6)),
            "IQR βₛᵢᵢ per 100 nT": df.apply(lambda r: f"[{fmt_number(r['q25_beta_sii'] * 100, 6)}, {fmt_number(r['q75_beta_sii'] * 100, 6)}]", axis=1),
            "Fraction βₛᵢᵢ < 0": df["share_beta_sii_negative"].apply(lambda x: fmt_number(x, 3)),
            "Valid PAR×VPD models": df["n_models"].apply(fmt_int),
            "Median n": df["median_n"].apply(fmt_int),
        })

    display_full = make_display(full)
    display_compact = make_display(compact)

    notes = [
        "Median adjusted βₛᵢᵢ and inter-quartile range (IQR) summarize the distribution of SII coefficients across all 28 × 28 PAR × VPD accumulation-window specifications.",
        "βₛᵢᵢ is scaled per 100 nT of trailing-mean SII exposure; the raw coefficient is per 1 nT.",
        "Fraction βₛᵢᵢ < 0 is the share of PAR × VPD specifications yielding a negative SII coefficient.",
        "The source CSV contains the complete 1–28 day SII window results; this DOCX table shows a compact subset.",
    ]
    return display_compact, display_full, notes


def main() -> int:
    compact, full, notes = build_table_s6()
    write_table(compact, "tableS6_environmental_driver", "Table S6. Environmental-driver adjustment summary across PAR × VPD specifications", notes=notes, landscape=True, full_df=full)
    print(f"[OK] Table S6: {len(full)} source rows, {len(compact)} display rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
