#!/usr/bin/env python3
"""
S02_kp_comparison.py — Compare SIF 771 nm associations for SII vs Kp index.

For each method and fixed window (21, 28 days) the script computes Spearman rho
and effect sizes on the exact same observations for the two exposures. This is a
robustness check: Kp is a geomagnetic proxy, not the primary SII exposure.
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
    build_omni_exposure,
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
    p = argparse.ArgumentParser(description="SII vs Kp comparison")
    p.add_argument("--outcome", default="sif_771nm")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"])
    p.add_argument("--figure", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    windows = args.windows if args.windows else cfg["windows"]["fixed"]
    print("=" * 64)
    print("MAGNETO supplementary checks — SII vs Kp comparison")
    print("=" * 64)
    print(f"Windows: {windows}")

    print("\n[1/3] Building exposures ...")
    sii_exp = build_sii_exposure(windows)
    kp_exp = build_omni_exposure("kp_mean", windows, "kp")

    print("[2/3] Loading residuals and building strata ...")
    all_strata = []
    for method in args.methods:
        df = load_residual_dataset(method, args.outcome)
        df = attach_exposure(df, sii_exp)
        df = attach_exposure(df, kp_exp)

        # Enforce identical observation sets for the two exposures by dropping
        # rows missing either SII or Kp for any requested window.
        all_exp_cols = [f"{p}_{w}d" for p in ("sii", "kp") for w in windows]
        df_clean = df.dropna(subset=all_exp_cols + [args.outcome])

        # Pooled + temperature-class strata for both exposures.
        for exposure_prefix in ("sii", "kp"):
            for group_cols, sample_type in [
                (None, f"pooled_{exposure_prefix}"),
                (["temp_bin_label"], f"temperature_{exposure_prefix}"),
            ]:
                strata = build_supplementary_checks_strata(
                    df_clean, sample_type, method, "residual",
                    windows=windows, group_cols=group_cols
                )
                # Rename sii_col to the actual exposure column used.
                for st in strata:
                    w = st["sii_window_days"]
                    st["sii_col"] = f"{exposure_prefix}_{w}d"
                    st["exposure"] = exposure_prefix
                all_strata.extend(strata)

    print(f"  Strata: {len(all_strata)}")

    print("[3/3] Computing statistics ...")
    df_out = run_supplementary_checks_fixed_window(all_strata)
    if df_out.empty:
        print("[WARN] No statistics computed")
        return 0

    df_out["outcome"] = args.outcome
    df_out["exposure"] = df_out["sample_type"].apply(lambda x: x.split("_")[-1])

    from _Common import FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_OMNI
    df_out = attach_provenance(
        df_out,
        stage="S02_kp_comparison.py",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_OMNI],
    )

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["kp_comparison_csv"]
    atomic_write(df_out, out_csv)
    print(f"\n  Written: {out_csv}")
    print(f"  Rows: {len(df_out)}")

    if args.figure:
        make_kp_figure(df_out, cfg)

    return 0


def make_kp_figure(df: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    windows = sorted(df["sii_window_days"].unique())
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(
        len(windows), len(methods), figsize=(4 * len(methods), 3.5 * len(windows)),
        squeeze=False, sharey=True,
    )
    temp_order = cfg["temperature"]["classes"]
    for i, w in enumerate(windows):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            sub = df[(df["sii_window_days"] == w) & (df["method"] == method)]
            temp = sub[sub["sample_type"].str.startswith("temperature")].copy()
            if temp.empty:
                ax.set_visible(False)
                continue
            temp = temp.sort_values("temp_bin_label", key=lambda s: s.map({v: k for k, v in enumerate(temp_order)}))
            pivot = temp.pivot_table(
                index="temp_bin_label", columns="exposure", values="spearman_rho", aggfunc="first"
            )
            pivot = pivot.reindex([c for c in temp_order if c in pivot.index])
            pivot.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Temperature class")
            ax.set_ylabel("Spearman rho")
            ax.set_title(f"{method}, {w} days")
            ax.legend(title="Exposure")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("SIF 771 nm: SII vs Kp across temperature classes")
    out_png = SUPPLEMENTARY_CHECKS_FIGURES / cfg["outputs"]["kp_comparison_fig"]
    savefig(out_png, fig)
    print(f"  Figure: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
