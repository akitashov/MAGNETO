#!/usr/bin/env python3
"""
S01_window_scan_771.py — Sweep SII exposure window length for SIF 771 nm.

Pools all valid observations and, separately, the harmonic/cyclic-spline matched
sample. The scan is descriptive only: it does not select a post-hoc "best"
window and does not rerun the main fixed-window inference.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

# Ensure the supplementary checks helpers and the core scripts are importable.
_sys_path_inserted = False
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
    _sys_path_inserted = True
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _supplementary_checks_common import (
    load_residual_dataset,
    load_matched_residuals,
    build_sii_exposure,
    attach_exposure,
    build_supplementary_checks_strata,
    run_supplementary_checks_fixed_window,
    attach_provenance,
    SUPPLEMENTARY_CHECKS_RESULTS,
    SUPPLEMENTARY_CHECKS_FIGURES,
    make_supplementary_checks_dirs,
    atomic_write,
    savefig,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Window-length scan for SIF 771 nm")
    p.add_argument("--outcome", default="sif_771nm")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"])
    p.add_argument("--figure", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    if args.windows is None:
        lo, hi = cfg["windows"]["full_scan"]
        windows = list(range(lo, hi + 1))
    else:
        windows = args.windows

    print("=" * 64)
    print("MAGNETO supplementary checks — Window scan (SIF 771 nm)")
    print("=" * 64)
    print(f"Windows: {windows[0]}..{windows[-1]} ({len(windows)} lengths)")

    # Build SII exposure once for all requested windows.
    print("\n[1/4] Building SII exposure ...")
    exposure = build_sii_exposure(windows)

    # Load residuals.
    print("[2/4] Loading residuals ...")
    residuals = {}
    for method in args.methods:
        residuals[method] = load_residual_dataset(method, args.outcome)
        residuals[method] = attach_exposure(residuals[method], exposure)
        print(f"  {method}: {len(residuals[method]):,} rows")

    matched_raw = load_matched_residuals(args.outcome)
    matched_raw = attach_exposure(matched_raw, exposure)
    print(f"  matched raw: {len(matched_raw):,} rows")

    # Build strata.
    print("[3/4] Building strata ...")
    all_strata = []
    for method in args.methods:
        df = residuals[method]
        all_strata.extend(build_supplementary_checks_strata(
            df, "pooled_full", method, "residual", windows=windows
        ))

    # Matched sample: same observations, different residual columns.
    if "harmonic" in args.methods:
        all_strata.extend(build_supplementary_checks_strata(
            matched_raw.copy(), "pairwise_matched", "harmonic",
            "residual_harmonic", windows=windows
        ))
    if "cyclic_spline" in args.methods:
        all_strata.extend(build_supplementary_checks_strata(
            matched_raw.copy(), "pairwise_matched", "cyclic_spline",
            "residual_cyclic_spline", windows=windows
        ))

    print(f"  Strata: {len(all_strata)}")

    # Compute fixed-window statistics.
    print("[4/4] Computing statistics ...")
    df_out = run_supplementary_checks_fixed_window(all_strata)
    if df_out.empty:
        print("[WARN] No statistics computed")
        return 0

    # Clean column names / labels.
    df_out["outcome"] = args.outcome
    df_out["sample_label"] = df_out["sample_type"].apply(
        lambda x: "matched" if "matched" in x else "pooled"
    )

    from _Common import FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC
    df_out = attach_provenance(
        df_out,
        stage="S01_window_scan_771.py",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC],
    )

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["window_scan_csv"]
    atomic_write(df_out, out_csv)
    print(f"\n  Written: {out_csv}")
    print(f"  Rows: {len(df_out)}")

    if args.figure:
        make_window_scan_figure(df_out, cfg)

    return 0


def supplementary_checks_provenance_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("analysis_", "execution_", "run_",
                                                   "git_commit", "config_hash",
                                                   "input_manifest_hash", "analysis_dataset_hash",
                                                   "generation_timestamp", "producing_stage",
                                                   "supplementary_checks_"))]


def make_window_scan_figure(df: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    methods = sorted(df["method"].unique())
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        for sample, linestyle in [("pooled", "-"), ("matched", "--")]:
            ss = sub[sub["sample_label"] == sample]
            if ss.empty:
                continue
            ss = ss.sort_values("sii_window_days")
            ax.plot(ss["sii_window_days"], ss["spearman_rho"],
                    label=sample, linestyle=linestyle, marker="o", markersize=3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("SII window length (days)")
        ax.set_ylabel("Spearman rho")
        ax.set_title(method)
        ax.legend(title="Sample")
        ax.grid(True, alpha=0.3)
    fig.suptitle("SIF 771 nm correlation vs. SII exposure window length")
    out_png = SUPPLEMENTARY_CHECKS_FIGURES / cfg["outputs"]["window_scan_fig"]
    savefig(out_png, fig)
    print(f"  Figure: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
