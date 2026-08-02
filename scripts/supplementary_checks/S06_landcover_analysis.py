#!/usr/bin/env python3
"""
S06_landcover_analysis.py — Pooled and temperature-stratified analysis by land-cover class.

This analysis is exploratory. Land-cover class is correlated with latitude,
climate, and seasonality, so results should not be interpreted as independent
biological effects.
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
    load_landcover_cell_grid,
    SUPPLEMENTARY_CHECKS_DATA,
    SUPPLEMENTARY_CHECKS_RESULTS,
    SUPPLEMENTARY_CHECKS_FIGURES,
    make_supplementary_checks_dirs,
    atomic_write,
    savefig,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Land-cover class analysis")
    p.add_argument("--outcome", default="sif_771nm")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"])
    p.add_argument("--min-cells-temperature", type=int, default=50)
    p.add_argument("--n-surrogates", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--figure", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    windows = args.windows if args.windows else cfg["windows"]["fixed"]
    n_surr = args.n_surrogates if args.n_surrogates is not None else cfg["surrogates"]["n"]
    seed = args.seed if args.seed is not None else cfg["surrogates"]["seed"]

    print("=" * 64)
    print("MAGNETO supplementary checks — Land-cover class analysis")
    print("=" * 64)

    print("\n[1/4] Loading land-cover grid ...")
    lc_grid = load_landcover_cell_grid()
    if lc_grid is None:
        print("\n[SKIP] Land-cover grid not prepared. Run S05 first.")
        return 0

    # Exclude non-vegetated / mixed classes from the biological sensitivity analysis.
    excluded = {"No_Data", "Water", "Urban", "Snow/Ice", "Wetlands", "Mixed/Other Vegetation"}
    lc_grid = lc_grid[~lc_grid["land_cover_class"].isin(excluded)].copy()
    classes = sorted(lc_grid["land_cover_class"].unique())
    print(f"  Classes retained: {classes}")

    print("[2/4] Building SII exposure ...")
    sii_exp = build_sii_exposure(windows)

    print("[3/4] Building strata by land-cover class ...")
    all_strata = []
    class_cell_counts = {}
    for method in args.methods:
        df = load_residual_dataset(method, args.outcome)
        df = attach_exposure(df, sii_exp)
        df = df.merge(lc_grid[["lat_id", "lon_id", "land_cover_class"]],
                      on=["lat_id", "lon_id"], how="inner")

        for lc_class, sub in df.groupby("land_cover_class"):
            n_cells = sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]
            class_cell_counts[(method, lc_class)] = n_cells
            all_strata.extend(build_supplementary_checks_strata(
                sub.copy(), f"landcover_{lc_class}", method, "residual",
                windows=windows, group_cols=None
            ))
            if n_cells >= args.min_cells_temperature:
                all_strata.extend(build_supplementary_checks_strata(
                    sub.copy(), f"landcover_temp_{lc_class}", method, "residual",
                    windows=windows, group_cols=["temp_bin_label"]
                ))

    if not all_strata:
        print("[WARN] No land-cover strata built")
        return 0

    print(f"  Strata: {len(all_strata)}")

    print("[4/4] Computing statistics and surrogates ...")
    df_obs = run_supplementary_checks_fixed_window(all_strata)
    if df_obs.empty:
        print("[WARN] No observed statistics computed")
        return 0
    df_obs["outcome"] = args.outcome
    df_obs["land_cover_class"] = df_obs["sample_type"].apply(
        # Order matters: remove the longer prefix first.
        lambda x: x.replace("landcover_temp_", "").replace("landcover_", "")
    )

    from _Common import FILE_RES_HARMONIC, FILE_RES_CYCLIC
    lc_grid_path = SUPPLEMENTARY_CHECKS_DATA / cfg["outputs"]["landcover_grid_parquet"]
    df_obs = attach_provenance(
        df_obs,
        stage="S06_landcover_analysis.py",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, lc_grid_path],
    )

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["landcover_analysis_csv"]
    atomic_write(df_obs, out_csv)
    print(f"\n  Written: {out_csv}")

    # Surrogates for pooled land-cover classes only.
    pooled_strata = [st for st in all_strata if st["sample_type"].startswith("landcover_") and
                     "_temp_" not in st["sample_type"]]
    if pooled_strata:
        print("  Running surrogates for pooled land-cover classes ...")
        surr_summary, surr_null = run_supplementary_checks_surrogates(pooled_strata, n_surr=n_surr, seed=seed)
        surr_summary["outcome"] = args.outcome
        surr_summary = attach_provenance(
            surr_summary,
            stage="S06_landcover_analysis.py:surrogates",
            dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, lc_grid_path],
        )
        out_surr = SUPPLEMENTARY_CHECKS_RESULTS / "landcover_analysis_surrogates.csv"
        atomic_write(surr_summary, out_surr)
        print(f"  Written: {out_surr}")
        null_path = SUPPLEMENTARY_CHECKS_RESULTS / "landcover_analysis_null_distributions.parquet"
        surr_null["outcome"] = args.outcome
        atomic_write(surr_null, null_path)
        print(f"  Written: {null_path}")

    if args.figure:
        make_landcover_figure(df_obs, cfg)

    return 0


def make_landcover_figure(df: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    # Use only pooled land-cover class estimates; temperature-stratified rows have
    # a non-missing temp_bin_label.
    df = df[df["temp_bin_label"].isna()].copy()

    windows = sorted(df["sii_window_days"].unique())
    methods = sorted(df["method"].unique())
    classes = sorted(df["land_cover_class"].unique())
    fig, axes = plt.subplots(
        len(windows), len(methods),
        figsize=(6.5 * len(methods), 4.5 * len(windows)),
        squeeze=False, sharey=True,
    )
    for i, w in enumerate(windows):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            sub = df[(df["sii_window_days"] == w) & (df["method"] == method)]
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.bar(sub["land_cover_class"], sub["spearman_rho"])
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Spearman rho")
            ax.set_title(f"{method}, {w} days")
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(
                sub["land_cover_class"], rotation=45, ha="right", fontsize=8
            )
    fig.suptitle("SIF 771 nm association by land-cover class (exploratory)")
    fig.tight_layout()
    out_png = SUPPLEMENTARY_CHECKS_FIGURES / cfg["outputs"]["landcover_analysis_fig"]
    savefig(out_png, fig)
    print(f"  Figure: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
