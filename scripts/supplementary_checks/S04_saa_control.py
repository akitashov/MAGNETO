#!/usr/bin/env python3
"""
S04_saa_control.py — South Atlantic Anomaly (SAA) control analysis.

The SAA is a region of weakened geomagnetic field and elevated particle flux.
This script repeats the primary harmonic/cyclic-spline analysis on SAA cells
only, with full temporal-surrogate null models. It is reported as an
instrumental / high-particle-flux diagnostic, not as primary evidence.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from _supplementary_checks_common import (
    load_residual_dataset,
    build_sii_exposure,
    attach_exposure,
    build_supplementary_checks_strata,
    run_supplementary_checks_fixed_window,
    run_supplementary_checks_surrogates,
    attach_provenance,
    decode_region_flag,
    SUPPLEMENTARY_CHECKS_RESULTS,
    SUPPLEMENTARY_CHECKS_FIGURES,
    make_supplementary_checks_dirs,
    atomic_write,
    savefig,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAA control analysis")
    p.add_argument("--outcome", default="sif_771nm")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"])
    p.add_argument("--n-surrogates", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--figure", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    windows = args.windows if args.windows else cfg["windows"]["fixed"]
    n_surr = args.n_surrogates if args.n_surrogates is not None else cfg["surrogates"]["n"]
    seed = args.seed if args.seed is not None else cfg["surrogates"]["seed"]

    print("=" * 64)
    print("MAGNETO supplementary checks — SAA control analysis")
    print("=" * 64)
    print(f"Windows: {windows}")
    print(f"Surrogates: {n_surr}")

    print("\n[1/4] Building SII exposure ...")
    sii_exp = build_sii_exposure(windows)

    print("[2/4] Loading residuals and selecting SAA cells ...")
    all_strata = []
    for method in args.methods:
        df = load_residual_dataset(method, args.outcome)
        df["is_SAA"] = decode_region_flag(df["region_flags"], "SAA")
        # Exclude desert/polar cells that happen to carry the SAA bit.
        df["is_Desert"] = decode_region_flag(df["region_flags"], "DESERT")
        df["is_Polar"] = decode_region_flag(df["region_flags"], "POLAR")
        saa = df[df["is_SAA"] & ~df["is_Desert"] & ~df["is_Polar"]].copy()
        print(f"  {method}: {len(saa):,} SAA observations")
        if saa.empty:
            continue
        saa = attach_exposure(saa, sii_exp)
        all_strata.extend(build_supplementary_checks_strata(
            saa, "control_SAA", method, "residual",
            windows=windows, group_cols=None
        ))
        all_strata.extend(build_supplementary_checks_strata(
            saa, "temperature_SAA", method, "residual",
            windows=windows, group_cols=["temp_bin_label"]
        ))

    if not all_strata:
        print("[WARN] No SAA strata built")
        return 0

    print(f"  Strata: {len(all_strata)}")

    print("[3/4] Computing observed statistics ...")
    df_obs = run_supplementary_checks_fixed_window(all_strata)
    if df_obs.empty:
        print("[WARN] No observed statistics computed")
        return 0
    df_obs["outcome"] = args.outcome

    from _Common import FILE_RES_HARMONIC, FILE_RES_CYCLIC
    df_obs = attach_provenance(
        df_obs,
        stage="S04_saa_control.py",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC],
    )

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["saa_control_csv"]
    atomic_write(df_obs, out_csv)
    print(f"\n  Written: {out_csv}")

    print("[4/4] Running temporal surrogates ...")
    # Use only pooled SAA strata for surrogates (temperature profiles are too thin).
    pooled_strata = [st for st in all_strata if st["sample_type"] == "control_SAA"]
    if not pooled_strata:
        print("[WARN] No pooled SAA strata for surrogates")
        return 0

    surr_summary, surr_null = run_supplementary_checks_surrogates(pooled_strata, n_surr=n_surr, seed=seed)
    if surr_summary.empty:
        print("[WARN] No surrogate results")
    else:
        surr_summary["outcome"] = args.outcome
        surr_summary = attach_provenance(
            surr_summary,
            stage="S04_saa_control.py:surrogates",
            dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC],
        )
        out_surr = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["saa_control_surrogate_csv"]
        atomic_write(surr_summary, out_surr)
        print(f"  Written: {out_surr}")

        null_path = SUPPLEMENTARY_CHECKS_RESULTS / "saa_control_null_distributions.parquet"
        surr_null["outcome"] = args.outcome
        atomic_write(surr_null, null_path)
        print(f"  Written: {null_path}")

    if args.figure:
        make_saa_figure(df_obs, cfg)

    return 0


def make_saa_figure(df: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    windows = sorted(df["sii_window_days"].unique())
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(
        1, len(windows), figsize=(5 * len(windows), 4.5), squeeze=False
    )
    for j, w in enumerate(windows):
        ax = axes[0][j]
        sub = df[df["sii_window_days"] == w]
        if sub.empty:
            ax.set_visible(False)
            continue
        bars = ax.bar(sub["method"], sub["spearman_rho"], color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        for bar, rho, nobs, ncells in zip(
            bars, sub["spearman_rho"], sub["n_obs"], sub["n_cells"]
        ):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"n={nobs:,}\ncells={ncells}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=7,
            )
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"SAA control, {w} days")
        ax.set_ylim(
            min(0, sub["spearman_rho"].min() * 1.2),
            max(0, sub["spearman_rho"].max() * 1.2),
        )
    fig.suptitle("SIF 771 nm association inside the South Atlantic Anomaly")
    out_png = SUPPLEMENTARY_CHECKS_FIGURES / cfg["outputs"]["saa_control_fig"]
    savefig(out_png, fig)
    print(f"  Figure: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
