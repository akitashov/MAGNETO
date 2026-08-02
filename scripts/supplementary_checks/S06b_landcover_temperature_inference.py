#!/usr/bin/env python3
"""
S06b_landcover_temperature_inference.py — Land-cover × temperature inference.

Exploratory heterogeneity analysis of the temporal SII–SIF association across
land-cover classes and temperature regimes.  The stage uses the validated
harmonic and cyclic-spline residual datasets, the existing trailing-mean SII
exposure windows, and the project's temporal-surrogate engine.

No figures are produced; rendering is delegated to separate visualization
scripts.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from _supplementary_checks_common import (
    attach_provenance,
    atomic_write,
    build_sii_exposure,
    attach_exposure,
    dataset_hash_of,
    load_landcover_cell_grid,
    load_residual_dataset,
    load_supplementary_checks_config,
    make_supplementary_checks_dirs,
    run_supplementary_checks_surrogates,
    supplementary_checks_provenance,
    SUPPLEMENTARY_CHECKS_DATA,
    SUPPLEMENTARY_CHECKS_RESULTS,
)
from _Common import (
    FILE_RES_HARMONIC,
    FILE_RES_CYCLIC,
    PROJECT_ROOT,
)
from magneto_lib import compute_stats

SURROGATE_MODES = ["year_perm", "circ_shift", "block_perm"]
SURROGATE_MODE_TO_P = {
    "year_perm": "p_year_perm",
    "circ_shift": "p_circ_shift",
    "block_perm": "p_block_perm",
}
SURROGATE_MODE_TO_COUNT_PREFIX = {
    "year_perm": ("n_year_completed", "n_year_failed"),
    "circ_shift": ("n_circular_completed", "n_circular_failed"),
    "block_perm": ("n_block_completed", "n_block_failed"),
}

EXCLUDED_LANDCOVER = {
    "No_Data",
    "Water",
    "Urban",
    "Snow/Ice",
    "Wetlands",
    "Mixed/Other Vegetation",
}

CONTRASTS = [
    ("Forest", "Barren", "Forest_minus_Barren"),
    ("Cropland", "Barren", "Cropland_minus_Barren"),
]

VEGETATED_CLASSES = {"Forest", "Cropland", "Grassland", "Savanna", "Shrubland/Savanna"}

OUTPUT_NAMES = {
    "inference_csv": "landcover_temperature_inference.csv",
    "null_parquet": "landcover_temperature_null_distributions.parquet",
    "bootstrap_parquet": "landcover_temperature_bootstrap.parquet",
    "contrasts_csv": "landcover_temperature_contrasts.csv",
    "composition_csv": "landcover_temperature_composition.csv",
    "lai_composition_csv": "landcover_lai_composition.csv",
    "audit_json": "landcover_temperature_audit.json",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Land-cover × temperature inference for the SII–SIF 771 nm association. "
            "This is an exploratory heterogeneity analysis, not a causal effect-modification test."
        )
    )
    p.add_argument("--outcome", default="sif_771nm",
                   help="Outcome variable name (default: sif_771nm).")
    p.add_argument("--methods", nargs="+", default=["harmonic", "cyclic_spline"],
                   help="Detrending methods to evaluate (default: harmonic cyclic_spline).")
    p.add_argument("--windows", type=int, nargs="+", default=None,
                   help="SII exposure windows in days (default from config: 21 28).")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Number of spatial cluster bootstrap replicates (default: 1000).")
    p.add_argument("--n-surrogates", type=int, default=1000,
                   help="Number of surrogates per null model (default: 1000).")
    p.add_argument("--min-cells", type=int, default=50,
                   help="Minimum unique spatial cells for eligibility (default: 50).")
    p.add_argument("--seed", type=int, default=None,
                   help="Base random seed (default from config).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 20 bootstrap and 20 surrogate replicates.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if outputs already exist.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for S06b outputs (default: results/supplementary_checks; "
                        "in smoke mode: results/supplementary_checks/smoke).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _method_seed(base_seed: int, method: str) -> int:
    return int(
        hashlib.sha256(f"s06b_method_seed:{base_seed}:{method}".encode()).hexdigest(),
        16,
    ) % (2 ** 32)


def _stratum_seed(method_seed: int, window: int, label_a: str, label_b: str,
                  kind: str) -> int:
    return int(
        hashlib.sha256(
            f"s06b_stratum_seed:{method_seed}:{window}:{label_a}:{label_b}:{kind}".encode()
        ).hexdigest(),
        16,
    ) % (2 ** 32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _required_paths(cfg: dict) -> dict[str, Path]:
    return {
        "lc_grid": SUPPLEMENTARY_CHECKS_DATA / cfg["outputs"]["landcover_grid_parquet"],
        "qc": PROJECT_ROOT / "data" / "interim" / "global_qc.parquet",
        "lai_summary": PROJECT_ROOT / "data" / "interim" / "lai_cell_summary.parquet",
    }


def _check_upstream_inputs(cfg: dict, methods: list[str]) -> tuple[dict[str, Path], list[str]]:
    paths = _required_paths(cfg)
    missing = []
    for name, p in paths.items():
        if not p.exists():
            missing.append(str(p))

    residuals = {
        "harmonic": FILE_RES_HARMONIC,
        "cyclic_spline": FILE_RES_CYCLIC,
    }
    for method in methods:
        p = residuals.get(method)
        if p is None:
            missing.append(f"unknown method '{method}'")
        elif not p.exists():
            missing.append(str(p))

    return paths, missing


def _ensure_landcover_grid(lc_grid_path: Path) -> int:
    """Run S05 only if the land-cover grid is missing."""
    if lc_grid_path.exists():
        return 0
    print("[INFO] Land-cover grid missing; running S05_prepare_landcover_grid.py ...")
    script = Path(__file__).with_name("S05_prepare_landcover_grid.py")
    rc = subprocess.run(
        [sys.executable, str(script), "--restrict-to-qc-cells"],
        cwd=PROJECT_ROOT,
    ).returncode
    if rc != 0:
        print(f"[ERROR] S05_prepare_landcover_grid.py failed (rc={rc})")
    return rc


def _load_landcover_residuals(method: str, outcome: str, windows: list[int],
                              lc_grid: pd.DataFrame) -> pd.DataFrame:
    residuals = load_residual_dataset(method, outcome)
    sii_exp = build_sii_exposure(windows)
    residuals = attach_exposure(residuals, sii_exp)
    residuals = residuals.merge(
        lc_grid[["lat_id", "lon_id", "land_cover_class"]],
        on=["lat_id", "lon_id"],
        how="inner",
    )
    return residuals


# ---------------------------------------------------------------------------
# Spatial cluster bootstrap
# ---------------------------------------------------------------------------

def _cluster_bootstrap_rho(df: pd.DataFrame, residual_col: str, sii_col: str,
                           n_boot: int, seed: int, min_obs: int = 10,
                           pbar_desc: str = "Bootstrap") -> dict[str, Any]:
    """Cell-cluster bootstrap with replacement; preserves within-cell multiplicity."""
    valid = df[[residual_col, sii_col, "lat_id", "lon_id"]].dropna()
    cells = valid[["lat_id", "lon_id"]].drop_duplicates().values
    n_cells = len(cells)

    if n_cells == 0 or len(valid) < min_obs:
        return {
            "boot_rhos": np.full(n_boot, np.nan),
            "boot_ci_lo": np.nan,
            "boot_ci_hi": np.nan,
            "n_completed": 0,
            "n_failed": n_boot,
        }

    cell_idx = valid[["lat_id", "lon_id"]].apply(tuple, axis=1).values
    cell_map = {tuple(c): i for i, c in enumerate(cells)}
    labels = np.array([cell_map[t] for t in cell_idx])
    indices_by_cell = [np.where(labels == i)[0] for i in range(n_cells)]

    x = valid[residual_col].values.astype(float)
    y = valid[sii_col].values.astype(float)
    rng = np.random.default_rng(seed)

    rhos = np.empty(n_boot, dtype=float)
    rhos[:] = np.nan
    n_failed = 0

    for i in tqdm(range(n_boot), desc=pbar_desc, leave=False):
        sampled = rng.integers(0, n_cells, size=n_cells)
        bootstrap_indices = np.concatenate([indices_by_cell[j] for j in sampled])
        if len(bootstrap_indices) < min_obs:
            n_failed += 1
            continue
        stats = compute_stats(x[bootstrap_indices], y[bootstrap_indices])
        rho = stats["spearman_rho"]
        if not np.isfinite(rho):
            n_failed += 1
            continue
        rhos[i] = rho

    valid_rhos = rhos[np.isfinite(rhos)]
    if len(valid_rhos) >= 0.5 * n_boot:
        ci_lo = float(np.quantile(valid_rhos, 0.025))
        ci_hi = float(np.quantile(valid_rhos, 0.975))
    else:
        ci_lo = ci_hi = np.nan

    return {
        "boot_rhos": rhos,
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "n_completed": int(len(valid_rhos)),
        "n_failed": int(n_failed),
    }


def _cluster_bootstrap_delta(g1: pd.DataFrame, g2: pd.DataFrame, sii_col: str,
                             n_boot: int, seed: int, min_obs: int = 10) -> np.ndarray:
    """Independent cluster bootstrap of two disjoint groups, returning rho1 - rho2."""
    valid1 = g1[["residual", sii_col, "lat_id", "lon_id"]].dropna()
    valid2 = g2[["residual", sii_col, "lat_id", "lon_id"]].dropna()
    cells1 = valid1[["lat_id", "lon_id"]].drop_duplicates().values
    cells2 = valid2[["lat_id", "lon_id"]].drop_duplicates().values

    if len(cells1) == 0 or len(cells2) == 0:
        return np.full(n_boot, np.nan)

    def _setup(group_valid):
        cells = group_valid[["lat_id", "lon_id"]].drop_duplicates().values
        cell_idx = group_valid[["lat_id", "lon_id"]].apply(tuple, axis=1).values
        cell_map = {tuple(c): i for i, c in enumerate(cells)}
        labels = np.array([cell_map[t] for t in cell_idx])
        indices_by_cell = [np.where(labels == i)[0] for i in range(len(cells))]
        x = group_valid["residual"].values.astype(float)
        y = group_valid[sii_col].values.astype(float)
        return cells, indices_by_cell, x, y

    cells1, idx1, x1, y1 = _setup(valid1)
    cells2, idx2, x2, y2 = _setup(valid2)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    deltas[:] = np.nan

    for i in range(n_boot):
        sampled1 = rng.integers(0, len(cells1), size=len(cells1))
        sampled2 = rng.integers(0, len(cells2), size=len(cells2))
        ix1 = np.concatenate([idx1[j] for j in sampled1])
        ix2 = np.concatenate([idx2[j] for j in sampled2])
        if len(ix1) < min_obs or len(ix2) < min_obs:
            continue
        rho1 = compute_stats(x1[ix1], y1[ix1])["spearman_rho"]
        rho2 = compute_stats(x2[ix2], y2[ix2])["spearman_rho"]
        if np.isfinite(rho1) and np.isfinite(rho2):
            deltas[i] = rho1 - rho2

    return deltas


# ---------------------------------------------------------------------------
# Strata and inference
# ---------------------------------------------------------------------------

def _build_strata(df: pd.DataFrame, method: str, windows: list[int],
                  min_cells: int) -> tuple[list[dict], list[dict]]:
    strata: list[dict] = []
    ineligible: list[dict] = []

    grouped = df.groupby(["land_cover_class", "temp_bin_label"], sort=True)
    for (lc_class, temp), sub in grouped:
        n_cells = sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]
        n_obs = len(sub)
        n_years = int(sub["year"].nunique()) if "year" in sub.columns else np.nan

        if n_cells < min_cells:
            for w in windows:
                ineligible.append({
                    "method": method,
                    "land_cover_class": lc_class,
                    "temp_bin_label": temp,
                    "sii_window_days": w,
                    "eligible": False,
                    "exclusion_reason": f"fewer_than_{min_cells}_cells",
                    "n_obs": n_obs,
                    "n_cells": n_cells,
                    "n_years": n_years,
                })
            continue

        for w in windows:
            sii_col = f"sii_{w}d"
            if sii_col not in sub.columns:
                continue
            valid = sub[["residual", sii_col]].dropna()
            if len(valid) < 10:
                ineligible.append({
                    "method": method,
                    "land_cover_class": lc_class,
                    "temp_bin_label": temp,
                    "sii_window_days": w,
                    "eligible": False,
                    "exclusion_reason": "fewer_than_10_valid_observations",
                    "n_obs": n_obs,
                    "n_cells": n_cells,
                    "n_years": n_years,
                })
                continue

            strata.append({
                "sample_type": "landcover_temperature",
                "method": method,
                "residual_col": "residual",
                "sii_col": sii_col,
                "sii_window_days": w,
                "land_cover_class": lc_class,
                "temp_bin_label": temp,
                "data": sub.copy(),
            })

    return strata, ineligible


def _run_inference_for_method(method: str, df: pd.DataFrame, windows: list[int],
                              min_cells: int, n_boot: int, n_surrogates: int,
                              seed: int) -> tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, list[dict]]:
    strata, ineligible = _build_strata(df, method, windows, min_cells)
    method_seed = _method_seed(seed, method)

    rows: list[dict] = []
    boot_records: list[dict] = []

    for st in tqdm(strata, desc=f"Inference strata ({method})", leave=False):
        sub = st["data"]
        residual_col = st["residual_col"]
        sii_col = st["sii_col"]
        window = st["sii_window_days"]
        lc_class = st["land_cover_class"]
        temp = st["temp_bin_label"]

        valid = sub[[residual_col, sii_col]].dropna()
        obs_stats = compute_stats(
            valid[residual_col].values.astype(float),
            valid[sii_col].values.astype(float),
        )
        n_cells = sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]
        n_years = int(sub["year"].nunique()) if "year" in sub.columns else np.nan

        str_seed = _stratum_seed(method_seed, window, lc_class, temp, "bootstrap")
        boot = _cluster_bootstrap_rho(
            sub, residual_col, sii_col, n_boot, str_seed,
            pbar_desc=f"Bootstrap {method} {lc_class} {temp}",
        )

        rows.append({
            "method": method,
            "sii_window_days": window,
            "land_cover_class": lc_class,
            "temp_bin_label": temp,
            "eligible": True,
            "exclusion_reason": "",
            "n_obs": int(obs_stats["n_obs"]),
            "n_cells": n_cells,
            "n_years": n_years,
            "spearman_rho": obs_stats["spearman_rho"],
            "boot_ci_lo": boot["boot_ci_lo"],
            "boot_ci_hi": boot["boot_ci_hi"],
            "p_year_perm": np.nan,
            "p_circ_shift": np.nan,
            "p_block_perm": np.nan,
            "p_display": np.nan,
            "n_year_completed": np.nan,
            "n_year_failed": np.nan,
            "n_circular_completed": np.nan,
            "n_circular_failed": np.nan,
            "n_block_completed": np.nan,
            "n_block_failed": np.nan,
            "seed": str_seed,
            "n_bootstrap": n_boot,
            "n_boot_completed": boot["n_completed"],
            "n_surrogates": n_surrogates,
        })

        for rep, rho in enumerate(boot["boot_rhos"]):
            boot_records.append({
                "method": method,
                "window": window,
                "land_cover_class": lc_class,
                "temp_bin_label": temp,
                "replicate": rep,
                "rho": rho,
            })

    if strata:
        surr_seed = method_seed
        surr_summary, surr_null = run_supplementary_checks_surrogates(
            strata, n_surr=n_surrogates, seed=surr_seed
        )
        surr_summary = surr_summary.rename(columns={"p_value": "p_val"})
        surr_summary["sii_window_days"] = (
            surr_summary["sii_window"]
            .str.replace("sii_", "", regex=False)
            .str.replace("d", "", regex=False)
            .astype(int)
        )

        for row in rows:
            for mode in SURROGATE_MODES:
                p_col = SURROGATE_MODE_TO_P[mode]
                comp_col, fail_col = SURROGATE_MODE_TO_COUNT_PREFIX[mode]
                match = surr_summary[
                    (surr_summary["method"] == row["method"]) &
                    (surr_summary["sii_window_days"] == row["sii_window_days"]) &
                    (surr_summary["land_cover_class"] == row["land_cover_class"]) &
                    (surr_summary["temp_bin_label"] == row["temp_bin_label"]) &
                    (surr_summary["surrogate_mode"] == mode)
                ]
                if not match.empty:
                    row[p_col] = float(match["p_val"].iloc[0])
                    row[comp_col] = int(match["n_completed"].iloc[0])
                    row[fail_col] = int(match["n_failed"].iloc[0])

            pvals = [row[c] for c in SURROGATE_MODE_TO_P.values() if np.isfinite(row[c])]
            row["p_display"] = float(np.max(pvals)) if pvals else np.nan
    else:
        surr_null = pd.DataFrame()

    # Ensure the full method × window × land-cover × temperature grid is
    # represented, even for combinations with no observations.
    observed_keys = set()
    for row in rows:
        observed_keys.add(
            (row["method"], row["sii_window_days"], row["land_cover_class"], row["temp_bin_label"])
        )
    for inel in ineligible:
        observed_keys.add(
            (inel["method"], inel["sii_window_days"], inel["land_cover_class"], inel["temp_bin_label"])
        )

    for w in windows:
        for lc_class in sorted(df["land_cover_class"].unique()):
            for temp in sorted(df["temp_bin_label"].dropna().unique()):
                key = (method, w, lc_class, temp)
                if key not in observed_keys:
                    ineligible.append({
                        "method": method,
                        "sii_window_days": w,
                        "land_cover_class": lc_class,
                        "temp_bin_label": temp,
                        "eligible": False,
                        "exclusion_reason": "no_observations",
                        "n_obs": 0,
                        "n_cells": 0,
                        "n_years": 0,
                    })

    inference_df = pd.DataFrame(rows)
    boot_df = pd.DataFrame(boot_records)
    return inference_df, boot_df, surr_null, ineligible


# ---------------------------------------------------------------------------
# Pre-specified contrasts
# ---------------------------------------------------------------------------

def _run_contrasts(residuals_by_method: dict[str, pd.DataFrame],
                   inference_df: pd.DataFrame, windows: list[int],
                   n_boot: int, seed: int, min_cells: int) -> pd.DataFrame:
    records: list[dict] = []

    for method, df in residuals_by_method.items():
        method_seed = _method_seed(seed, method)
        for window in windows:
            sii_col = f"sii_{window}d"
            if sii_col not in df.columns:
                continue
            for temp in sorted(df["temp_bin_label"].dropna().unique()):
                sub = df[(df["temp_bin_label"] == temp) & (df[sii_col].notna())].copy()
                if sub.empty:
                    continue

                groups: dict[str, pd.DataFrame] = {}
                for lc in sub["land_cover_class"].unique():
                    groups[lc] = sub[sub["land_cover_class"] == lc].copy()

                veg_sub = sub[sub["land_cover_class"].isin(VEGETATED_CLASSES)].copy()
                if not veg_sub.empty:
                    groups["pooled_vegetated"] = veg_sub

                all_contrasts = CONTRASTS + [
                    ("pooled_vegetated", "Barren", "Vegetated_minus_Barren")
                ]
                for g1_name, g2_name, contrast_name in all_contrasts:
                    if g1_name not in groups or g2_name not in groups:
                        continue
                    g1 = groups[g1_name]
                    g2 = groups[g2_name]
                    n1 = g1[["lat_id", "lon_id"]].drop_duplicates().shape[0]
                    n2 = g2[["lat_id", "lon_id"]].drop_duplicates().shape[0]
                    eligible = (n1 >= min_cells) and (n2 >= min_cells)
                    exclusion_reason = (
                        "" if eligible else "one_or_both_groups_below_min_cells"
                    )

                    valid1 = g1[["residual", sii_col]].dropna()
                    valid2 = g2[["residual", sii_col]].dropna()
                    rho1 = (
                        compute_stats(valid1["residual"].values.astype(float),
                                      valid1[sii_col].values.astype(float))["spearman_rho"]
                        if len(valid1) >= 10 else np.nan
                    )
                    rho2 = (
                        compute_stats(valid2["residual"].values.astype(float),
                                      valid2[sii_col].values.astype(float))["spearman_rho"]
                        if len(valid2) >= 10 else np.nan
                    )
                    delta = rho1 - rho2 if np.isfinite(rho1) and np.isfinite(rho2) else np.nan

                    ci_lo = ci_hi = np.nan
                    if eligible and np.isfinite(delta):
                        str_seed = _stratum_seed(
                            method_seed, window, contrast_name, temp, "contrast"
                        )
                        deltas = _cluster_bootstrap_delta(g1, g2, sii_col, n_boot, str_seed)
                        valid_d = deltas[np.isfinite(deltas)]
                        if len(valid_d) >= 0.5 * n_boot:
                            ci_lo = float(np.quantile(valid_d, 0.025))
                            ci_hi = float(np.quantile(valid_d, 0.975))

                    records.append({
                        "method": method,
                        "sii_window_days": window,
                        "temp_bin_label": temp,
                        "contrast": contrast_name,
                        "group_1": g1_name,
                        "group_2": g2_name,
                        "rho_group_1": rho1,
                        "rho_group_2": rho2,
                        "delta_rho": delta,
                        "boot_ci_lo": ci_lo,
                        "boot_ci_hi": ci_hi,
                        "n_cells_group_1": n1,
                        "n_cells_group_2": n2,
                        "eligible": eligible,
                        "exclusion_reason": exclusion_reason,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Composition diagnostics
# ---------------------------------------------------------------------------

def _build_composition_tables(qc_df: pd.DataFrame, lc_grid: pd.DataFrame,
                              lai_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Land-cover × temperature composition.
    df = qc_df.merge(
        lc_grid[["lat_id", "lon_id", "land_cover_class"]],
        on=["lat_id", "lon_id"],
        how="inner",
    )
    df = df[~df["land_cover_class"].isin(EXCLUDED_LANDCOVER)]
    df = df.dropna(subset=["temp_bin_label"])

    temp_rows: list[dict] = []
    for (lc, temp), sub in df.groupby(["land_cover_class", "temp_bin_label"], sort=True):
        year_props = sub["year"].value_counts(normalize=True)
        temp_rows.append({
            "land_cover_class": lc,
            "temp_bin_label": temp,
            "n_obs": len(sub),
            "n_cells": sub[["lat_id", "lon_id"]].drop_duplicates().shape[0],
            "n_years": int(sub["year"].nunique()),
            "median_latitude": float(sub["lat"].median()),
            "latitude_iqr": float(sub["lat"].quantile(0.75) - sub["lat"].quantile(0.25)),
            "median_temperature_c": float(sub["temp_c_ma10"].median()),
            "temperature_iqr_c": float(
                sub["temp_c_ma10"].quantile(0.75) - sub["temp_c_ma10"].quantile(0.25)
            ),
            "year_balance": float(year_props.max()) if len(year_props) else np.nan,
        })

    temp_comp = pd.DataFrame(temp_rows)

    # Add per-cell LAI summaries.
    lai_med = lai_summary.groupby(["lat_id", "lon_id"], as_index=False).agg(
        median_lai=("median_lai", "median"),
        lai_iqr=("median_lai", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    )
    lai_by_cell = lai_med.merge(
        lc_grid[["lat_id", "lon_id", "land_cover_class"]],
        on=["lat_id", "lon_id"],
        how="inner",
    )
    lai_by_cell = lai_by_cell[~lai_by_cell["land_cover_class"].isin(EXCLUDED_LANDCOVER)]
    lai_by_cell = lai_by_cell.dropna(subset=["median_lai"])

    lai_agg = lai_by_cell.groupby("land_cover_class").agg(
        median_lai=("median_lai", "median"),
        lai_iqr=("median_lai", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).reset_index()

    temp_comp = temp_comp.merge(lai_agg, on="land_cover_class", how="left")
    temp_comp = temp_comp.rename(columns={
        "median_lai": "median_lai_cell_median",
        "lai_iqr": "lai_iqr_cell_median",
    })

    # Land-cover × LAI-quartile cross-tab.
    lc_lai = lai_summary.merge(
        lc_grid[["lat_id", "lon_id", "land_cover_class"]],
        on=["lat_id", "lon_id"],
        how="inner",
    )
    lc_lai = lc_lai[~lc_lai["land_cover_class"].isin(EXCLUDED_LANDCOVER)]
    lc_lai = lc_lai.dropna(subset=["lai_quartile"])

    lai_rows: list[dict] = []
    for (lc, q), sub in lc_lai.groupby(["land_cover_class", "lai_quartile"], sort=True):
        total_lc = lc_lai.loc[lc_lai["land_cover_class"] == lc, "n_lai"].sum()
        total_q = lc_lai.loc[lc_lai["lai_quartile"] == q, "n_lai"].sum()
        lai_rows.append({
            "land_cover_class": lc,
            "lai_quartile": q,
            "n_cells": len(sub),
            "n_obs": int(sub["n_lai"].sum()),
            "proportion_within_landcover": (
                float(sub["n_lai"].sum() / total_lc) if total_lc > 0 else np.nan
            ),
            "proportion_within_lai_quartile": (
                float(sub["n_lai"].sum() / total_q) if total_q > 0 else np.nan
            ),
            "median_lai": float(sub["median_lai"].median()),
        })

    lai_comp = pd.DataFrame(lai_rows)
    return temp_comp, lai_comp


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _build_audit(args, paths: dict[str, Path], output_paths: dict[str, Path],
                 inference_df: pd.DataFrame, contrasts_df: pd.DataFrame,
                 boot_df: pd.DataFrame, null_df: pd.DataFrame,
                 comp_df: pd.DataFrame, lai_comp_df: pd.DataFrame,
                 ineligible_all: list[dict], warnings: list[str],
                 n_boot: int, n_surr: int) -> dict[str, Any]:
    methods = sorted(inference_df["method"].dropna().unique()) if not inference_df.empty else args.methods
    windows = sorted(inference_df["sii_window_days"].dropna().unique()) if not inference_df.empty else args.windows
    lc_classes = sorted(inference_df["land_cover_class"].dropna().unique()) if not inference_df.empty else []
    temp_classes = sorted(inference_df["temp_bin_label"].dropna().unique()) if not inference_df.empty else []

    eligible_df = inference_df[inference_df["eligible"] == True] if not inference_df.empty else inference_df
    boot_requested = int(eligible_df["n_bootstrap"].sum()) if not eligible_df.empty else 0
    boot_completed = int(eligible_df["n_boot_completed"].sum()) if not eligible_df.empty else 0
    boot_failed = boot_requested - boot_completed

    surr_totals: dict[str, dict[str, int]] = {}
    for mode in SURROGATE_MODES:
        comp_col, fail_col = SURROGATE_MODE_TO_COUNT_PREFIX[mode]
        surr_totals[mode] = {
            "requested": int(eligible_df["n_surrogates"].sum()) if not eligible_df.empty else 0,
            "completed": int(eligible_df[comp_col].sum()) if comp_col in inference_df.columns else 0,
            "failed": int(eligible_df[fail_col].sum()) if fail_col in inference_df.columns else 0,
        }

    input_files = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in paths.items()}
    input_files["residual_harmonic"] = str(FILE_RES_HARMONIC.relative_to(PROJECT_ROOT))
    input_files["residual_cyclic_spline"] = str(FILE_RES_CYCLIC.relative_to(PROJECT_ROOT))

    input_mtimes = {}
    input_row_counts = {}
    for k, v in paths.items():
        if v.exists():
            input_mtimes[k] = datetime.fromtimestamp(v.stat().st_mtime, tz=timezone.utc).isoformat()
            if v.suffix == ".parquet":
                input_row_counts[k] = int(len(pd.read_parquet(v)))
            elif v.suffix == ".csv":
                input_row_counts[k] = int(len(pd.read_csv(v)))
            else:
                input_row_counts[k] = None

    output_files = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in output_paths.items()}
    output_row_counts = {
        "inference": len(inference_df),
        "bootstrap": len(boot_df),
        "null_distributions": len(null_df),
        "contrasts": len(contrasts_df),
        "composition": len(comp_df),
        "lai_composition": len(lai_comp_df),
    }

    return {
        "stage": "S06b_landcover_temperature_inference.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": supplementary_checks_provenance("S06b")["supplementary_checks_git_commit"],
        "command_line_arguments": vars(args),
        "input_files": input_files,
        "input_modification_times": input_mtimes,
        "input_row_counts": input_row_counts,
        "methods": methods,
        "windows": [int(w) for w in windows],
        "land_cover_classes": lc_classes,
        "temperature_classes": temp_classes,
        "seeds": {
            "base": args.seed,
        },
        "bootstrap": {
            "requested_per_stratum": n_boot,
            "completed_total": boot_completed,
            "failed_total": boot_failed,
        },
        "surrogates": surr_totals,
        "n_boot_actual": n_boot,
        "n_surr_actual": n_surr,
        "ineligible_subgroups": ineligible_all,
        "excluded_land_cover_classes": sorted(EXCLUDED_LANDCOVER),
        "vegetated_class_definition": sorted(VEGETATED_CLASSES),
        "contrast_definitions": [
            {"group_1": a, "group_2": b, "name": c} for a, b, c in CONTRASTS
        ] + [{"group_1": "pooled_vegetated", "group_2": "Barren",
              "name": "Vegetated_minus_Barren",
              "pooled_vegetated_classes": sorted(VEGETATED_CLASSES)}],
        "output_files": output_files,
        "output_row_counts": output_row_counts,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    windows = args.windows if args.windows else cfg["windows"]["fixed"]
    n_boot = 20 if args.smoke else args.n_bootstrap
    n_surr = 20 if args.smoke else args.n_surrogates
    seed = args.seed if args.seed is not None else cfg["surrogates"]["seed"]
    min_cells = args.min_cells

    print("=" * 64)
    print("MAGNETO supplementary checks — S06b land-cover × temperature inference")
    print("=" * 64)
    print(f"Methods: {args.methods}")
    print(f"Windows: {windows}")
    print(f"Bootstrap replicates: {n_boot}")
    print(f"Surrogates per null model: {n_surr}")
    print(f"Min cells: {min_cells}")
    print(f"Smoke: {args.smoke}")
    print(f"Base seed: {seed}")

    # Check / prepare inputs.
    paths, missing = _check_upstream_inputs(cfg, args.methods)
    if missing:
        print("[ERROR] Required upstream inputs missing:")
        for m in missing:
            print(f"  - {m}")
        return 1

    rc = _ensure_landcover_grid(paths["lc_grid"])
    if rc != 0:
        return rc

    paths, missing = _check_upstream_inputs(cfg, args.methods)
    if missing:
        print("[ERROR] Inputs still missing after S05:")
        for m in missing:
            print(f"  - {m}")
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SUPPLEMENTARY_CHECKS_RESULTS / "smoke" if args.smoke else SUPPLEMENTARY_CHECKS_RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {k: output_dir / v for k, v in OUTPUT_NAMES.items()}
    if not args.force and all(p.exists() for p in output_paths.values()):
        print("[INFO] S06b outputs already exist; use --force to rerun.")
        return 0
    print(f"Output directory: {output_dir}")

    # Load shared data.
    print("\n[1/5] Loading land-cover grid and composition inputs ...")
    lc_grid = load_landcover_cell_grid()
    if lc_grid is None:
        print("[ERROR] Land-cover grid could not be loaded.")
        return 1
    lc_grid = lc_grid[~lc_grid["land_cover_class"].isin(EXCLUDED_LANDCOVER)].copy()

    qc_cols = ["date", "year", "lat_id", "lon_id", "lat", "lon",
               "temp_c_ma10", "temp_bin_label", "lai"]
    qc_df = pd.read_parquet(paths["qc"], columns=qc_cols)
    lai_summary = pd.read_parquet(
        paths["lai_summary"],
        columns=["lat_id", "lon_id", "n_lai", "median_lai", "q25_lai",
                 "q75_lai", "lai_quartile"],
    )

    dataset_files = [
        FILE_RES_HARMONIC,
        FILE_RES_CYCLIC,
        paths["lc_grid"],
        paths["qc"],
        paths["lai_summary"],
    ]

    warnings: list[str] = []

    # Run inference per method.
    print("[2/5] Running land-cover × temperature inference ...")
    all_inference: list[pd.DataFrame] = []
    all_boot: list[pd.DataFrame] = []
    all_null: list[pd.DataFrame] = []
    ineligible_all: list[dict] = []
    residuals_by_method: dict[str, pd.DataFrame] = {}

    for method in args.methods:
        print(f"  Method: {method}")
        df = _load_landcover_residuals(method, args.outcome, windows, lc_grid)
        residuals_by_method[method] = df
        inf, boot, null, inel = _run_inference_for_method(
            method, df, windows, min_cells, n_boot, n_surr, seed
        )
        all_inference.append(inf)
        all_boot.append(boot)
        all_null.append(null)
        ineligible_all.extend(inel)

    inference_df = pd.concat(all_inference, ignore_index=True) if all_inference else pd.DataFrame()
    boot_df = pd.concat(all_boot, ignore_index=True) if all_boot else pd.DataFrame()
    null_df = pd.concat(all_null, ignore_index=True) if all_null else pd.DataFrame()

    # Contrasts.
    print("[3/5] Computing pre-specified contrasts ...")
    contrasts_df = _run_contrasts(
        residuals_by_method, inference_df, windows, n_boot, seed, min_cells
    )

    # Composition diagnostics.
    print("[4/5] Building composition diagnostics ...")
    comp_df, lai_comp_df = _build_composition_tables(qc_df, lc_grid, lai_summary)

    # Attach provenance.
    print("[5/5] Writing outputs ...")
    if not inference_df.empty:
        attach_provenance(inference_df, "S06b_landcover_temperature_inference.py", dataset_files)
    if not boot_df.empty:
        attach_provenance(boot_df, "S06b_landcover_temperature_inference.py:bootstrap", dataset_files)
    if not null_df.empty:
        attach_provenance(null_df, "S06b_landcover_temperature_inference.py:surrogates", dataset_files)
    if not contrasts_df.empty:
        attach_provenance(contrasts_df, "S06b_landcover_temperature_inference.py:contrasts", dataset_files)
    if not comp_df.empty:
        attach_provenance(comp_df, "S06b_landcover_temperature_inference.py:composition", dataset_files)
    if not lai_comp_df.empty:
        attach_provenance(lai_comp_df, "S06b_landcover_temperature_inference.py:lai_composition", dataset_files)

    # Append ineligible rows to inference output so they are not silently dropped.
    if ineligible_all:
        ineligible_df = pd.DataFrame(ineligible_all)
        ineligible_df["eligible"] = False
        for col in inference_df.columns:
            if col not in ineligible_df.columns:
                ineligible_df[col] = np.nan
        inference_df = pd.concat(
            [inference_df, ineligible_df[inference_df.columns]], ignore_index=True
        )

    # Write outputs.
    atomic_write(inference_df, output_paths["inference_csv"])
    print(f"  Written: {output_paths['inference_csv']} ({len(inference_df)} rows)")
    atomic_write(boot_df, output_paths["bootstrap_parquet"])
    print(f"  Written: {output_paths['bootstrap_parquet']} ({len(boot_df)} rows)")
    atomic_write(null_df, output_paths["null_parquet"])
    print(f"  Written: {output_paths['null_parquet']} ({len(null_df)} rows)")
    atomic_write(contrasts_df, output_paths["contrasts_csv"])
    print(f"  Written: {output_paths['contrasts_csv']} ({len(contrasts_df)} rows)")
    atomic_write(comp_df, output_paths["composition_csv"])
    print(f"  Written: {output_paths['composition_csv']} ({len(comp_df)} rows)")
    atomic_write(lai_comp_df, output_paths["lai_composition_csv"])
    print(f"  Written: {output_paths['lai_composition_csv']} ({len(lai_comp_df)} rows)")

    audit = _build_audit(
        args, paths, output_paths, inference_df, contrasts_df, boot_df,
        null_df, comp_df, lai_comp_df, ineligible_all, warnings,
        n_boot=n_boot, n_surr=n_surr,
    )
    output_paths["audit_json"].write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"  Written: {output_paths['audit_json']}")

    n_eligible = int((inference_df["eligible"] == True).sum()) if "eligible" in inference_df.columns else 0
    n_ineligible = int((inference_df["eligible"] == False).sum()) if "eligible" in inference_df.columns else 0
    print(f"\nEligible strata: {n_eligible}; ineligible strata: {n_ineligible}")
    print("S06b complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
