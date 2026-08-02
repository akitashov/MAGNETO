#!/usr/bin/env python3
"""
08_fixed_window.py — Fixed-window statistics for pooled, matched, control, and
exploratory subgroup samples.

For each sample, method, and SII window, computes Spearman rho and OLS slope,
plus diagnostics (n_obs, n_cells, n_years, SII SD, residual SD).
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
from _Common import *
from magneto_lib import build_daily_sii, compute_sii_windows, build_strata, run_fixed_window


def main() -> int:
    print("=" * 64)
    print("MAGNETO — Fixed-Window Statistics")
    print("=" * 64)
    setup_dirs()

    # ── Build SII exposure ────────────────────────────────────────────
    print("\n[1/4] Building SII exposure …")
    omni = build_daily_sii()
    sii_win = compute_sii_windows(omni, SII_RAW_COL, SII_WINDOWS)
    print(f"  SII daily range: {omni['date'].min().date()} → {omni['date'].max().date()}")
    print(f"  Windows: {SII_WINDOWS}")

    # ── Load residual datasets ────────────────────────────────────────
    print("\n[2/4] Loading residual datasets …")
    harm = pd.read_parquet(FILE_RES_HARMONIC, engine=PARQUET_ENGINE)
    harm = harm[harm["pass_flag"] == True].copy() if "pass_flag" in harm.columns else harm
    harm["date"] = pd.to_datetime(harm["date"])
    harm["method"] = "harmonic"
    print(f"  harmonic valid: {len(harm):,}")

    cyc = pd.read_parquet(FILE_RES_CYCLIC, engine=PARQUET_ENGINE)
    cyc = cyc[cyc["fit_status"] == "ok"].copy() if "fit_status" in cyc.columns else cyc
    cyc["date"] = pd.to_datetime(cyc["date"])
    cyc["method"] = "cyclic_spline"
    print(f"  cyclic_spline valid: {len(cyc):,}")

    mk = pd.read_parquet(FILE_MATCHED_HC, engine=PARQUET_ENGINE)
    mk["date"] = pd.to_datetime(mk["date"])
    print(f"  matched: {len(mk):,}")

    for df in [harm, cyc, mk]:
        df = df.merge(sii_win, on="date", how="left")

    # attach SII windows to each dataset
    harm = harm.merge(sii_win, on="date", how="left")
    cyc = cyc.merge(sii_win, on="date", how="left")
    mk = mk.merge(sii_win, on="date", how="left")

    # ── Build strata ──────────────────────────────────────────────────
    print("\n[3/4] Building analysis strata …")
    all_strata = []

    # Core pooled full samples
    all_strata.extend(build_strata(harm, "pooled_full", "harmonic", "residual", group_cols=None))
    all_strata.extend(build_strata(cyc, "pooled_full", "cyclic_spline", "residual", group_cols=None))

    # Pairwise matched
    all_strata.extend(build_strata(mk, "pairwise_matched", "harmonic", "residual_harmonic", group_cols=None))
    all_strata.extend(build_strata(mk, "pairwise_matched", "cyclic_spline", "residual_cyclic_spline", group_cols=None))

    # Functional and geographic controls (only well-defined, pre-specified controls)
    control_flags = ["is_strict_low_lai", "is_vegetated", "is_Sahara"]
    for flag in control_flags:
        if flag not in harm.columns:
            continue
        for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
            sub = df[df[flag] == True].copy()
            if len(sub) == 0:
                continue
            name = flag.replace("is_", "")
            all_strata.extend(build_strata(sub, f"control_{name}", method, col, group_cols=None))

    # Exploratory subgroup samples
    for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
        all_strata.extend(build_strata(df, "lai_quartile", method, col, group_cols=["lai_quartile"]))
        if "temp_bin_label" in df.columns:
            all_strata.extend(build_strata(df, "temperature", method, col, group_cols=["temp_bin_label"]))
            all_strata.extend(build_strata(df, "temperature_lai", method, col, group_cols=["temp_bin_label", "lai_quartile"]))

    # ── Compute statistics ────────────────────────────────────────────
    print("\n[4/4] Computing statistics …")
    rows = run_fixed_window(all_strata)
    if not rows:
        print("[FAIL] No statistical results computed")
        return 1

    df_out = pd.DataFrame(rows)
    # enrich grouping metadata
    if "lai_quartile" in df_out.columns:
        df_out["lai_quartile"] = df_out["lai_quartile"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "")
    if "temp_bin_label" in df_out.columns:
        df_out["temperature_class"] = df_out["temp_bin_label"].fillna("")
    else:
        df_out["temperature_class"] = ""

    # Add run provenance directly so that stage-manifest hashes match the final files.
    prov = run_provenance_dict(
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC],
        producing_stage="08_fixed_window.py",
    )
    for k, v in prov.items():
        df_out[k] = v

    # group_name / grouping_type for reporting
    def grouping_type(r):
        if r["sample_type"] == "pooled_full":
            return "pooled"
        if r["sample_type"] == "pairwise_matched":
            return "pairwise"
        if r["sample_type"].startswith("control_"):
            return "control"
        if r["sample_type"] == "lai_quartile":
            return "lai_quartile"
        if r["sample_type"] == "temperature":
            return "temperature"
        if r["sample_type"] == "temperature_lai":
            return "temperature_lai"
        return "other"

    df_out["grouping_type"] = df_out.apply(grouping_type, axis=1)

    def group_name(r):
        if r["sample_type"].startswith("control_"):
            return r["sample_type"].replace("control_", "")
        if r["sample_type"] == "lai_quartile":
            return r["lai_quartile"]
        if r["sample_type"] == "temperature":
            return r["temperature_class"]
        if r["sample_type"] == "temperature_lai":
            return f"{r['temperature_class']}_{r['lai_quartile']}".strip("_")
        return "all"

    df_out["group_name"] = df_out.apply(group_name, axis=1)

    # order columns
    front = ["sample_type", "grouping_type", "group_name", "method", "sii_window", "sii_window_days",
             "temperature_class", "lai_quartile"]
    cols = front + [c for c in df_out.columns if c not in front]
    df_out = df_out[[c for c in cols if c in df_out.columns]]

    atomic_write(df_out, FILE_FIXED)
    print(f"\n  Written: {FILE_FIXED}")
    print(f"  Rows: {len(df_out)}")
    print(f"  Sample types: {df_out['sample_type'].nunique()}")
    print(f"  Methods: {df_out['method'].nunique()}")
    print(f"\n[OK] 08_fixed_window complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
