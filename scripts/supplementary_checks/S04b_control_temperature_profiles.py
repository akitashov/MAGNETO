#!/usr/bin/env python3
"""
S04b_control_temperature_profiles.py — Temperature-stratified analysis for
functional/geographic control scenarios.

The core fixed-window table only reports pooled control estimates. This script
computes real per-temperature-class estimates for Vegetated, strict low-LAI
control, Sahara and SAA cells, together with temporal-surrogate null
distributions. The outputs are used by S07 (graphical abstract matrix) and S08
(temperature scenario figure).
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
    make_supplementary_checks_dirs,
    atomic_write,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Control temperature profiles")
    p.add_argument("--outcome", default="sif_771nm")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"])
    p.add_argument("--n-surrogates", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


SCENARIO_FILTERS = {
    "control_vegetated": (
        "Vegetated",
        lambda df: df["is_vegetated"] == True,  # noqa: E712
    ),
    "control_strict_low_lai": (
        "Strict low-LAI control",
        lambda df: df["is_strict_low_lai"] == True,  # noqa: E712
    ),
    "control_Sahara": (
        "Sahara",
        lambda df: df["is_Sahara"] == True,  # noqa: E712
    ),
    "control_SAA": (
        "SAA",
        lambda df: (
            decode_region_flag(df["region_flags"], "SAA") &
            ~decode_region_flag(df["region_flags"], "DESERT") &
            ~decode_region_flag(df["region_flags"], "POLAR")
        ),
    ),
}


def _scenario_filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {sid: fn(df) for sid, (_, fn) in SCENARIO_FILTERS.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    windows = args.windows if args.windows else cfg["windows"]["fixed"]
    n_surr = args.n_surrogates if args.n_surrogates is not None else cfg["surrogates"]["n"]
    seed = args.seed if args.seed is not None else cfg["surrogates"]["seed"]

    print("=" * 64)
    print("MAGNETO supplementary checks — Control temperature profiles")
    print("=" * 64)
    print(f"Windows: {windows}")
    print(f"Surrogates: {n_surr}")

    print("\n[1/3] Building SII exposure ...")
    sii_exp = build_sii_exposure(windows)

    print("[2/3] Loading residuals and building temperature strata ...")
    all_strata = []
    for method in args.methods:
        df = load_residual_dataset(method, args.outcome)
        df = attach_exposure(df, sii_exp)
        for scenario_id, mask in _scenario_filters(df).items():
            sub = df[mask].copy()
            if sub.empty:
                continue
            label = SCENARIO_FILTERS[scenario_id][0]
            print(f"  {method} / {label}: {len(sub):,} obs")
            all_strata.extend(build_supplementary_checks_strata(
                sub, f"temperature_{scenario_id}", method, "residual",
                windows=windows, group_cols=["temp_bin_label"]
            ))

    if not all_strata:
        print("[WARN] No temperature strata built")
        return 0

    print(f"  Total strata: {len(all_strata)}")

    print("[3/3] Computing statistics, bootstrap CIs and surrogates ...")
    df_obs = run_supplementary_checks_fixed_window(all_strata, bootstrap=True, n_boot=1000, boot_seed=seed)
    if df_obs.empty:
        print("[WARN] No observed statistics")
        return 0
    df_obs["outcome"] = args.outcome

    from _Common import FILE_RES_HARMONIC, FILE_RES_CYCLIC
    df_obs = attach_provenance(
        df_obs,
        stage="S04b_control_temperature_profiles.py",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC],
    )

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["control_temperature_csv"]
    atomic_write(df_obs, out_csv)
    print(f"\n  Written: {out_csv}")

    surr_summary, surr_null = run_supplementary_checks_surrogates(all_strata, n_surr=n_surr, seed=seed)
    surr_summary["outcome"] = args.outcome
    surr_summary = attach_provenance(
        surr_summary,
        stage="S04b_control_temperature_profiles.py:surrogates",
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC],
    )
    out_surr = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["control_temperature_surrogate_csv"]
    atomic_write(surr_summary, out_surr)
    print(f"  Written: {out_surr}")

    null_path = SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles_null_distributions.parquet"
    surr_null["outcome"] = args.outcome
    atomic_write(surr_null, null_path)
    print(f"  Written: {null_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
