#!/usr/bin/env python3
"""Table S5. Alternative geomagnetic and solar-activity comparisons.

Combines Kp, F10.7 and SAA-control results. Uses matched observations where the
output supports them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import fmt_int, fmt_number, fmt_p, load_csv, project_root, write_table


def build_table_s5() -> tuple[pd.DataFrame, list[str]]:
    root = project_root()
    kp = load_csv(root / "results" / "supplementary_checks" / "kp_comparison.csv")
    f107 = load_csv(root / "results" / "supplementary_checks" / "f107_comparison.csv")
    saa = load_csv(root / "results" / "supplementary_checks" / "saa_control.csv")

    rows = []

    def add_row(comparison, sample, temp, exposure, unit_col, unit_label, r, matched_note=None):
        if unit_col == "ols_slope_per_100nT":
            effect = r.get("ols_slope_per_100nT")
        else:
            effect = r.get("ols_slope_per_1nT")
        rows.append({
            "Comparison": comparison,
            "Sample": sample,
            "Temperature regime": temp,
            "Exposure": exposure,
            "Exposure unit": unit_label,
            "Spearman rho": r["spearman_rho"],
            "Effect per unit": effect,
            "n observations": r["n_obs"],
            "n cells": r["n_cells"],
            "Nominal p": r["spearman_p"],
            "Matched sample note": matched_note or "",
        })

    # Kp comparison: SII reference and Kp from the same file (matched sample).
    sii_kp = kp[
        (kp["sample_type"] == "pooled_sii")
        & (kp["method"] == "harmonic")
        & (kp["sii_window_days"] == 28)
    ]
    if not sii_kp.empty:
        add_row("SII (Kp-matched reference)", "Full sample", "Pooled", "SII", "ols_slope_per_100nT", "per 100 nT",
                sii_kp.iloc[0], matched_note="Matched to Kp comparison")
    kp_sub = kp[
        (kp["sample_type"] == "pooled_kp")
        & (kp["method"] == "harmonic")
        & (kp["sii_window_days"] == 28)
    ]
    if not kp_sub.empty:
        add_row("Kp comparator", "Full sample", "Pooled", "Kp", "ols_slope_per_1nT", "per 1 Kp unit",
                kp_sub.iloc[0], matched_note="Same observations as SII (Kp-matched reference)")

    # F10.7 comparison: use the F10.7-matched SII reference, then F10.7 itself.
    sii_f107 = f107[
        (f107["sample_type"] == "pooled_sii")
        & (f107["method"] == "harmonic")
        & (f107["sii_window_days"] == 28)
    ]
    if not sii_f107.empty:
        add_row("SII (F10.7-matched reference)", "Full sample", "Pooled", "SII", "ols_slope_per_100nT", "per 100 nT",
                sii_f107.iloc[0], matched_note=f"Matched to F10.7 comparison, n = {int(sii_f107.iloc[0]['n_obs']):,}")
    f107_sub = f107[
        (f107["sample_type"] == "pooled_f10_7")
        & (f107["method"] == "harmonic")
        & (f107["sii_window_days"] == 28)
    ]
    if not f107_sub.empty:
        add_row("F10.7 comparator", "Full sample", "Pooled", "F10.7", "ols_slope_per_1nT", "per 1 sfu",
                f107_sub.iloc[0], matched_note="Same observations as SII (F10.7-matched reference)")

    # SAA control.
    saa_sub = saa[
        (saa["sample_type"] == "control_SAA")
        & (saa["method"] == "harmonic")
        & (saa["sii_window_days"] == 28)
    ]
    if not saa_sub.empty:
        add_row("SAA control", "SAA control", "Pooled", "SII", "ols_slope_per_100nT", "per 100 nT",
                saa_sub.iloc[0])

    df = pd.DataFrame(rows)

    display = pd.DataFrame({
        "Comparison": df["Comparison"],
        "Sample": df["Sample"],
        "Temperature regime": df["Temperature regime"],
        "Exposure": df["Exposure"],
        "Exposure unit": df["Exposure unit"],
        "Spearman rho": df["Spearman rho"].apply(lambda x: fmt_number(x, 3)),
        "Effect per unit": df["Effect per unit"].apply(lambda x: fmt_number(x, 5)),
        "n observations": df["n observations"].apply(fmt_int),
        "n cells": df["n cells"].apply(fmt_int),
        "Nominal p": df["Nominal p"].apply(lambda x: fmt_p(x, empirical=False)),
    })

    f10_7_matched_n = int(sii_f107.iloc[0]["n_obs"]) if not sii_f107.empty else None
    kp_full_n = int(sii_kp.iloc[0]["n_obs"]) if not sii_kp.empty else None

    notes = [
        "SII, Kp and F10.7 comparisons use their respective 28-day trailing-mean exposures.",
        (
            f"The F10.7 comparison uses the matched SII reference that shares the same observations "
            f"(n = {f10_7_matched_n:,}); the main SII reference from the Kp file uses the full pooled sample "
            f"(n = {kp_full_n:,})."
            if f10_7_matched_n is not None and kp_full_n is not None
            else "Sample sizes are read from the source comparison CSVs."
        ),
        "Effect per unit is exposure-specific: SII is per 100 nT, Kp is per 1 Kp unit (raw OMNI2 values are in 0.1-Kp units; divide by 10 to obtain ordinary Kp), F10.7 is per 1 solar flux unit (sfu).",
        "Nominal Spearman p-values are reported for descriptive comparison only; empirical temporal-surrogate p-values are shown in Table S4.",
    ]
    return display, notes


def main() -> int:
    df, notes = build_table_s5()
    write_table(df, "tableS5_kp_f107", "Table S5. Alternative geomagnetic and solar-activity comparisons", notes=notes, landscape=True)
    print(f"[OK] Table S5: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
