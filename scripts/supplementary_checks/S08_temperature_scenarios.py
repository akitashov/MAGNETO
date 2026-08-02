#!/usr/bin/env python3
"""
S08_temperature_scenarios.py — Assemble temperature-scenario tables for Figure 1.

This stage is pure collection: it reads already-computed core fixed-window
results, control-temperature profiles, surrogate summaries, and bootstrap
confidence intervals, then writes two canonical CSV tables used by the
visualization script.

Outputs:
    results/supplementary_checks/temperature_scenarios_harmonic.csv
    results/supplementary_checks/temperature_scenarios_spline.csv

Each table contains:
    method, window_days, scenario, temp_class, rho,
    ci_low, ci_high, p_year, p_circular, p_block, n_obs, n_cells
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _supplementary_checks_common import (
    load_main_fixed_window_results,
    load_main_surrogate_summary,
    load_residual_dataset,
    build_sii_exposure,
    attach_exposure,
    cell_cluster_bootstrap_rho,
    SUPPLEMENTARY_CHECKS_RESULTS,
    make_supplementary_checks_dirs,
    atomic_write,
    load_supplementary_checks_config,
)


# Map upstream sample_type labels to the scenario vocabulary expected by the
# Figure 1 plotting script.
SCENARIO_MAP = {
    "temperature": "pooled",
    "temperature_control_vegetated": "control_vegetated",
    "temperature_control_strict_low_lai": "control_strict_low_lai",
    "temperature_control_Sahara": "Sahara",
    "temperature_control_SAA": "SAA",
}

# The visualization script uses "spline" for the cyclic-spline method.
METHOD_MAP = {"cyclic_spline": "spline"}

SURROGATE_MODE_TO_P = {
    "year_perm": "p_year",
    "circ_shift": "p_circular",
    "block_perm": "p_block",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Assemble temperature-scenario tables for Figure 1"
    )
    p.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[21, 28],
        help="Exposure window(s) in days to include (default: 21 28).",
    )
    return p.parse_args(argv)


def _bootstrap_main_ci(method: str, window: int) -> dict[str, tuple[float, float]]:
    """Cell-cluster bootstrap CI for the main temperature pooled profile."""
    sii_col = f"sii_{window}d"
    residuals = load_residual_dataset(method, "sif_771nm")
    residuals = attach_exposure(residuals, build_sii_exposure([window]))

    boot_map: dict[str, tuple[float, float]] = {}
    valid = residuals.dropna(subset=["temp_bin_label", "residual", sii_col])
    for temp, sub in valid.groupby("temp_bin_label"):
        seed = int(
            hashlib.sha256(f"{method}_{window}_{temp}".encode()).hexdigest(), 16
        ) % 2**31
        ci = cell_cluster_bootstrap_rho(
            sub, "residual", sii_col, n_boot=1000, seed=seed
        )
        boot_map[temp] = (ci["boot_ci_lo"], ci["boot_ci_hi"])
    return boot_map


def _load_main_temperature(method: str, windows: list[int]) -> pd.DataFrame:
    """Load the main 'temperature' pooled profile for one detrending method."""
    fixed = load_main_fixed_window_results()
    surr = load_main_surrogate_summary()

    out_method = METHOD_MAP.get(method, method)
    frames: list[pd.DataFrame] = []

    for window in windows:
        sub_fixed = fixed[
            (fixed["method"] == method)
            & (fixed["sii_window_days"] == window)
            & (fixed["sample_type"] == "temperature")
        ]
        if sub_fixed.empty:
            continue

        sub_surr = surr[
            (surr["method"] == method)
            & (surr["sii_window"] == f"sii_{window}d")
            & (surr["sample_type"] == "temperature")
        ]

        df = sub_fixed[
            ["temp_bin_label", "spearman_rho", "n_obs", "n_cells"]
        ].copy()
        df = df.rename(columns={"spearman_rho": "rho"})
        df["scenario"] = "pooled"
        df["method"] = out_method
        df["window_days"] = window

        boot_map = _bootstrap_main_ci(method, window)
        df["ci_low"] = df["temp_bin_label"].map(
            lambda t: boot_map.get(t, (np.nan, np.nan))[0]
        )
        df["ci_high"] = df["temp_bin_label"].map(
            lambda t: boot_map.get(t, (np.nan, np.nan))[1]
        )

        for mode, col in SURROGATE_MODE_TO_P.items():
            ss = sub_surr[sub_surr["surrogate_mode"] == mode]
            p_map = dict(zip(ss["temp_bin_label"], ss["p_value"]))
            df[col] = df["temp_bin_label"].map(p_map)

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_control_temperature(method: str, windows: list[int]) -> pd.DataFrame:
    """Load control temperature profiles (Vegetated/Barren/Sahara/SAA)."""
    obs_path = SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles.csv"
    surr_path = SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles_surrogates.csv"
    if not obs_path.exists():
        return pd.DataFrame()

    obs = pd.read_csv(obs_path)
    obs = obs[obs["method"] == method]
    if obs.empty:
        return pd.DataFrame()

    surr = pd.read_csv(surr_path) if surr_path.exists() else pd.DataFrame()
    surr = surr[surr["method"] == method]

    out_method = METHOD_MAP.get(method, method)
    frames: list[pd.DataFrame] = []

    for window in windows:
        sub_obs = obs[obs["sii_window_days"] == window]
        if sub_obs.empty:
            continue

        sub_surr = surr[surr["sii_window"] == f"sii_{window}d"]

        df = sub_obs[
            [
                "sample_type",
                "temp_bin_label",
                "spearman_rho",
                "boot_ci_lo",
                "boot_ci_hi",
                "n_obs",
                "n_cells",
            ]
        ].copy()
        df = df.rename(
            columns={
                "spearman_rho": "rho",
                "boot_ci_lo": "ci_low",
                "boot_ci_hi": "ci_high",
            }
        )
        df["scenario"] = df["sample_type"].map(SCENARIO_MAP)
        df["method"] = out_method
        df["window_days"] = window

        for mode, col in SURROGATE_MODE_TO_P.items():
            ss = sub_surr[sub_surr["surrogate_mode"] == mode]
            p_map = dict(
                zip(zip(ss["sample_type"], ss["temp_bin_label"]), ss["p_value"])
            )
            df[col] = df.apply(
                lambda r: p_map.get((r["sample_type"], r["temp_bin_label"]), np.nan),
                axis=1,
            )

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_landcover_temperature(method: str, windows: list[int]) -> pd.DataFrame:
    """Load land-cover-class temperature profiles for Barren cells."""
    path = SUPPLEMENTARY_CHECKS_RESULTS / "landcover_temperature_inference.csv"
    if not path.exists():
        return pd.DataFrame()

    obs = pd.read_csv(path)
    obs = obs[(obs["method"] == method) & (obs["land_cover_class"] == "Barren")]
    if obs.empty:
        return pd.DataFrame()

    out_method = METHOD_MAP.get(method, method)
    frames: list[pd.DataFrame] = []

    for window in windows:
        sub = obs[obs["sii_window_days"] == window]
        if sub.empty:
            continue
        df = sub[
            [
                "temp_bin_label",
                "spearman_rho",
                "boot_ci_lo",
                "boot_ci_hi",
                "n_obs",
                "n_cells",
                "p_year_perm",
                "p_circ_shift",
                "p_block_perm",
            ]
        ].copy()
        df = df.rename(
            columns={
                "spearman_rho": "rho",
                "boot_ci_lo": "ci_low",
                "boot_ci_hi": "ci_high",
                "p_year_perm": "p_year",
                "p_circ_shift": "p_circular",
                "p_block_perm": "p_block",
            }
        )
        df["scenario"] = "landcover_barren"
        df["method"] = out_method
        df["window_days"] = window
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_figure_df(method: str, windows: list[int]) -> pd.DataFrame:
    """Combine main, control, and land-cover temperature scenarios for one method."""
    parts = [
        _load_main_temperature(method, windows),
        _load_control_temperature(method, windows),
        _load_landcover_temperature(method, windows),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        return df

    df = df.rename(columns={"temp_bin_label": "temp_class"})
    df = df[
        [
            "method",
            "window_days",
            "scenario",
            "temp_class",
            "rho",
            "ci_low",
            "ci_high",
            "p_year",
            "p_circular",
            "p_block",
            "n_obs",
            "n_cells",
        ]
    ]
    return df


def main(argv=None) -> int:
    args = parse_args(argv)
    load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    print("=" * 64)
    print("MAGNETO supplementary checks — Temperature scenario tables")
    print("=" * 64)

    for method in ("harmonic", "cyclic_spline"):
        suffix = "harmonic" if method == "harmonic" else "spline"
        df = build_figure_df(method, args.window)
        if df.empty:
            print(f"[WARN] No {suffix} data available")
            continue

        out_csv = SUPPLEMENTARY_CHECKS_RESULTS / f"temperature_scenarios_{suffix}.csv"
        atomic_write(df, out_csv)
        print(f"  Written: {out_csv} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
