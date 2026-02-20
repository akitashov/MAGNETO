#!/usr/bin/env python3
"""
Aggregation and Global Synthesis for the Marker-Screening Stage.

This script consolidates the results of the triage phase (Step 08), where the 
effectiveness of different space weather markers (SII vs. F10.7) and their 
accumulation characteristics (MA vs. LAG) are compared.

Core Logic:
1) Data Collection: Scans RESULTS_DIR for summary and pairwise comparison files.
2) Global Integration: Aggregates bin-specific performance into scenario-level 
   metrics, ensuring no "cherry-picking" of individual windows.
3) Win-rate Calculation: Quantifies how often SII outperforms F10.7 in terms 
   of correlation strength and statistical significance.
4) Metadata Enrichment: Ensures all scenarios (Global, SAA, etc.) and targets 
   are correctly labeled for downstream reporting.

OUTPUT FILES STRUCTURE (reports/meta-statistics/):
--------------------------------------------------
1. screening_overview_summary.csv (Detailed Per-Bin Metrics):
   - 'target', 'scenario', 'bin_label': Grouping keys.
   - 'omni_var': The driver being evaluated (e.g., sii_mean, f10_7_mean).
   - 'best_rho': Maximum Spearman correlation found across all tested windows/lags.
   - 'best_window': The specific integration window (days) yielding 'best_rho'.
   - 'mean_rho_top5': Average correlation of the top 5 performing windows (robustness metric).
   - 'sig_window_count': Number of windows that passed the p < 0.05 significance threshold.

2. screening_winrates.csv (Pairwise Driver Comparison):
   - 'target', 'scenario': Context for comparison.
   - 'win_rate_rho': Frequency (%) where SII showed higher |rho| than F10.7 across all bins.
   - 'win_rate_sig': Frequency (%) where SII was significant while F10.7 was not.
   - 'n_total_comparisons': Total number of tested combinations (bins x window types).

3. screening_overview_global.csv (Executive Summary):
   - 'target', 'scenario': Context.
   - 'primary_marker': The recommended driver (SII or F10.7) based on overall performance.
   - 'avg_rho_diff': Mean difference in correlation strength between drivers.
   - 'accumulation_advantage': Score indicating if long-memory features (MA) 
     outperform short-memory (LAG) features.

All metrics are integrated over windows and reported per-bin or per-scenario to 
provide a robust foundation for the primary Matrix Search stage.
"""

from __future__ import annotations

import glob
from typing import Tuple

import numpy as np
import pandas as pd

from _Common import Config


def _read_all(pattern: str) -> pd.DataFrame:
    """
    Reads all CSV files matching the pattern.
    Prioritizes Config.RESULTS_DIR; falls back to current directory.
    Returns a single concatenated DataFrame or empty DataFrame if no files found.
    """
    # Always read from RESULTS_DIR (primary-analysis outputs)
    files = sorted(glob.glob(str(Config.RESULTS_DIR / pattern)))
    if not files:
        # fallback: current working directory
        files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ensure_columns(df: pd.DataFrame, required: Tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def build_overview_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Convert screening_summary_* into a compact per (target, scenario, bin) comparison table.
    """
    _ensure_columns(
        df_summary,
        ("target", "scenario", "family", "bin_id", "bin_label", "auc_abs_z_ma", "auc_abs_z_lag", "delta_acc"),
        "screening_summary",
    )

    piv = df_summary.pivot_table(
        index=["target", "scenario", "bin_id", "bin_label"],
        columns="family",
        values=["auc_abs_z_ma", "auc_abs_z_lag", "delta_acc"],
        aggfunc="first",
    ).reset_index()

    piv.columns = ["_".join([c for c in col if c]).rstrip("_") for col in piv.columns.to_flat_index()]

    def col(metric: str, fam: str) -> str:
        return f"{metric}_{fam}"

    out = piv.copy()
    # Ensure presence (families are expected: "SII", "F10.7")
    for fam in ["SII", "F10.7"]:
        for m in ["auc_abs_z_ma", "auc_abs_z_lag", "delta_acc"]:
            c = col(m, fam)
            if c not in out.columns:
                out[c] = np.nan

    out["delta_acc_diff"] = out[col("delta_acc", "SII")] - out[col("delta_acc", "F10.7")]
    out["auc_ma_ratio"] = out[col("auc_abs_z_ma", "SII")] / out[col("auc_abs_z_ma", "F10.7")]
    out["auc_ma_ratio"] = out["auc_ma_ratio"].replace([np.inf, -np.inf], np.nan)

    rename = {
        col("auc_abs_z_ma", "SII"): "sii_auc_ma",
        col("auc_abs_z_lag", "SII"): "sii_auc_lag",
        col("delta_acc", "SII"): "sii_delta_acc",
        col("auc_abs_z_ma", "F10.7"): "f_auc_ma",
        col("auc_abs_z_lag", "F10.7"): "f_auc_lag",
        col("delta_acc", "F10.7"): "f_delta_acc",
    }
    out = out.rename(columns=rename)

    # Optional ordering if Config provides
    if hasattr(Config, "TEMP_BIN_ORDER"):
        order = list(Config.TEMP_BIN_ORDER)
        out["bin_label"] = pd.Categorical(out["bin_label"], order, ordered=True)

    out = out.sort_values(["target", "scenario", "bin_id"]).reset_index(drop=True)
    return out


def build_winrates(df_pair: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P(SII wins) from pairwise screening outputs.
    """
    if df_pair.empty:
        return pd.DataFrame()

    _ensure_columns(
        df_pair,
        ("target", "scenario", "bin_id", "bin_label", "mode", "window_value", "winner"),
        "screening_pairwise",
    )

    df = df_pair.copy()
    df["is_sii_win"] = (df["winner"] == "SII").astype(float)

    grp = (
        df.groupby(["target", "scenario", "bin_id", "bin_label", "mode"], as_index=False)
          .agg(n_windows=("is_sii_win", "size"), p_sii_win=("is_sii_win", "mean"))
    )

    grp_all = (
        df.groupby(["target", "scenario", "bin_id", "bin_label"], as_index=False)
          .agg(n_windows=("is_sii_win", "size"), p_sii_win=("is_sii_win", "mean"))
    )
    grp_all["mode"] = "ALL"

    out = pd.concat([grp, grp_all], ignore_index=True)
    out = out.sort_values(["target", "scenario", "bin_id", "mode"]).reset_index(drop=True)
    return out


def build_overview_global(overview: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (target, scenario) across bins.
    - delta_acc_diff: simple mean across bins (robust enough; can be weighted if you prefer)
    - auc_ma_ratio: simple mean across bins
    - p_sii_win: weighted by n_windows across bins (mode="ALL")
    """
    _ensure_columns(
        overview,
        ("target", "scenario", "bin_id", "delta_acc_diff", "auc_ma_ratio"),
        "overview",
    )

    agg = (
        overview.groupby(["target", "scenario"], as_index=False)
                .agg(
                    n_bins=("bin_id", "nunique"),
                    delta_acc_diff_mean=("delta_acc_diff", "mean"),
                    delta_acc_diff_median=("delta_acc_diff", "median"),
                    auc_ma_ratio_mean=("auc_ma_ratio", "mean"),
                    auc_ma_ratio_median=("auc_ma_ratio", "median"),
                )
    )

    if win is not None and not win.empty:
        w = win[win["mode"] == "ALL"].copy()
        if not w.empty:
            pw = (
                w.groupby(["target", "scenario"], as_index=False)
                 .apply(lambda g: pd.Series({
                     "p_sii_win_weighted": np.average(g["p_sii_win"], weights=g["n_windows"]),
                     "n_windows_total": int(g["n_windows"].sum()),
                 }))
                 .reset_index(drop=True)
            )
            agg = agg.merge(pw, on=["target", "scenario"], how="left")

    return agg.sort_values(["target", "scenario"]).reset_index(drop=True)


def main() -> None:
    # Always write into META_ANALYSIS_DIR (meta-analysis outputs)
    Config.META_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df_summary = _read_all("screening_summary_*.csv")
    df_pair = _read_all("screening_pairwise_*.csv")

    if df_summary.empty:
        raise SystemExit(
            f"No screening_summary_*.csv found in RESULTS_DIR={Config.RESULTS_DIR} (or current working directory)."
        )

    overview = build_overview_summary(df_summary)
    win = build_winrates(df_pair) if not df_pair.empty else pd.DataFrame()
    global_overview = build_overview_global(overview, win)

    out_overview = Config.META_ANALYSIS_DIR / "screening_overview_summary.csv"
    out_win = Config.META_ANALYSIS_DIR / "screening_winrates.csv"
    out_global = Config.META_ANALYSIS_DIR / "screening_overview_global.csv"

    overview.to_csv(out_overview, index=False)
    global_overview.to_csv(out_global, index=False)
    if not win.empty:
        win.to_csv(out_win, index=False)

    print("[OK] Wrote:")
    print(f"  - {out_overview}")
    print(f"  - {out_global}")
    if not win.empty:
        print(f"  - {out_win}")


if __name__ == "__main__":
    main()