#!/usr/bin/env python3
"""
09_surrogates.py — Temporal surrogate tests.

Surrogate analyses are run in priority order:
  1. pooled full harmonic (primary + secondary window)
  2. pooled full cyclic spline
  3. harmonic–spline pairwise matched
  4. strict low-LAI control
  5. persistently vegetated reference
  6. Sahara geographic control
  7. LAI quartiles
  8. temperature strata
  9. temperature × LAI (only after core analyses complete)

All surrogates transform the raw daily SII series and recompute the 21- and
28-day rolling windows.
"""
from __future__ import annotations
import sys, argparse, os
import numpy as np
import pandas as pd
from _Common import *
from magneto_lib import (
    build_daily_sii, compute_sii_windows, build_strata,
    run_surrogates, year_permutation, circular_shift, block_permutation,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-surrogates", type=int, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    n_surr = args.n_surrogates
    if n_surr is None:
        n_surr = SURROGATE_SMOKE_N if args.smoke_test else SURROGATE_N
    seed = SURROGATE_SEED
    print(f"[INFO] Surrogates: N={n_surr}, seed={seed}")

    setup_dirs()

    # ── Load datasets ─────────────────────────────────────────────────
    print("[INFO] Loading residual datasets …")
    harm = pd.read_parquet(FILE_RES_HARMONIC, engine=PARQUET_ENGINE)
    harm = harm[harm["pass_flag"] == True].copy() if "pass_flag" in harm.columns else harm
    harm["date"] = pd.to_datetime(harm["date"])

    cyc = pd.read_parquet(FILE_RES_CYCLIC, engine=PARQUET_ENGINE)
    cyc = cyc[cyc["fit_status"] == "ok"].copy() if "fit_status" in cyc.columns else cyc
    cyc["date"] = pd.to_datetime(cyc["date"])

    mk = pd.read_parquet(FILE_MATCHED_HC, engine=PARQUET_ENGINE)
    mk["date"] = pd.to_datetime(mk["date"])

    sii_win = compute_sii_windows(build_daily_sii(), SII_RAW_COL, SII_WINDOWS)
    harm = harm.merge(sii_win, on="date", how="left")
    cyc = cyc.merge(sii_win, on="date", how="left")
    mk = mk.merge(sii_win, on="date", how="left")

    # ── Build strata in priority order ────────────────────────────────
    print("[INFO] Building strata …")
    strata = []

    # 1–2. Pooled full samples
    strata.extend(build_strata(harm, "pooled_full", "harmonic", "residual", group_cols=None))
    strata.extend(build_strata(cyc, "pooled_full", "cyclic_spline", "residual", group_cols=None))

    # 3. Pairwise matched
    strata.extend(build_strata(mk, "pairwise_matched", "harmonic", "residual_harmonic", group_cols=None))
    strata.extend(build_strata(mk, "pairwise_matched", "cyclic_spline", "residual_cyclic_spline", group_cols=None))

    # 4–6. Core controls
    control_flags = [
        ("is_strict_low_lai", "control_strict_low_lai"),
        ("is_vegetated", "control_vegetated"),
        ("is_Sahara", "control_Sahara"),
    ]
    for flag, sample_type in control_flags:
        if flag not in harm.columns:
            continue
        for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
            sub = df[df[flag] == True].copy()
            if len(sub) == 0:
                continue
            strata.extend(build_strata(sub, sample_type, method, col, group_cols=None))

    # 7. LAI quartiles
    for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
        strata.extend(build_strata(df, "lai_quartile", method, col, group_cols=["lai_quartile"]))

    # 8. Temperature strata (exploratory)
    if "temp_bin_label" in harm.columns:
        for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
            strata.extend(build_strata(df, "temperature", method, col, group_cols=["temp_bin_label"]))

    # 9. Temperature × LAI (exploratory; run only after core analyses complete)
    if "temp_bin_label" in harm.columns:
        for df, method, col in [(harm, "harmonic", "residual"), (cyc, "cyclic_spline", "residual")]:
            strata.extend(build_strata(df, "temperature_lai", method, col, group_cols=["temp_bin_label", "lai_quartile"]))

    print(f"[INFO] {len(strata)} strata selected for surrogate testing")

    # ── Run surrogates ────────────────────────────────────────────────
    print("[INFO] Running temporal surrogates …")
    df_summary, df_null = run_surrogates(strata, n_surr=n_surr, seed=seed)

    # Add run provenance directly; report stage must not modify numeric tables.
    prov = run_provenance_dict(
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC],
        producing_stage="09_surrogates.py",
    )
    for k, v in prov.items():
        df_summary[k] = v
        df_null[k] = v

    atomic_write(df_summary, FILE_SURROGATES)
    atomic_write(df_null, FILE_SURROGATE_NULL)

    n_sig = (df_summary["p_value"] < 0.05).sum()
    print(f"[OK] Summary: {len(df_summary)} rows, {n_sig} with p<0.05 → {FILE_SURROGATES}")
    print(f"[OK] Null distributions: {len(df_null)} rows → {FILE_SURROGATE_NULL}")

    completed = df_summary[df_summary["n_completed"] > 0]
    zero_p = (completed["p_value"] == 0.0).sum()
    if zero_p > 0:
        print(f"[WARN] {zero_p} rows with p=0 (should be impossible)")
    nan_p = completed["p_value"].isna().sum()
    if nan_p > 0:
        print(f"[WARN] {nan_p} rows with NaN p-value")

    multi = df_summary[df_summary["n_completed"] > 1]
    zero_sd = (multi["null_sd"] == 0.0).sum()
    if zero_sd > 0:
        print(f"[WARN] {zero_sd} rows with null_sd = 0")

    print("[DONE]")


if __name__ == "__main__":
    main()
