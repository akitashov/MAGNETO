#!/usr/bin/env python3
"""
S11_driver_r2_shapley.py — Exact Shapley/LMG decomposition of full-model R²
across SII, PAR, and VPD.

For every scenario × temperature regime × window, the script builds one common
complete-case sample and fits all seven nonempty subsets of predictors.
Predictor contributions are obtained by averaging each predictor's marginal
R² increase over the six possible orderings. This guarantees:

    shapley_r2_SII + shapley_r2_PAR + shapley_r2_VPD == r2_full

The result is a descriptive variance allocation, not a causal attribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _Common import PROJECT_ROOT, PARQUET_ENGINE, TEMP_LABELS, atomic_write, full_sha256
from _supplementary_checks_common import git_commit

warnings.filterwarnings("ignore")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "environmental_driver_input.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "supplementary_checks"
SMOKE_DIR = OUTPUT_DIR / "smoke"

TARGET = "sif_771nm"
METHOD = "harmonic"
WINDOWS_FULL = list(range(1, 29))
WINDOWS_SMOKE = [1, 7, 14, 21, 28]
MIN_CELLS = 50
TOL = 1e-10
DRIVER_ORDER = ["SII", "PAR", "VPD"]
DRIVER_COL_PATTERNS = {
    "SII": "sii_mean_ma{w}",
    "PAR": "par_ma{w}",
    "VPD": "vpd_ma{w}",
}
SCENARIOS = {
    "full_sample": lambda df: pd.Series(True, index=df.index),
    "persistently_vegetated": lambda df: df["is_vegetated"],
    "control_strict_low_lai": lambda df: df["is_strict_low_lai"],
    "Sahara": lambda df: df["is_Sahara"],
    "SAA": lambda df: df["is_SAA"],
}
SUBSETS = [
    frozenset(["SII"]),
    frozenset(["PAR"]),
    frozenset(["VPD"]),
    frozenset(["SII", "PAR"]),
    frozenset(["SII", "VPD"]),
    frozenset(["PAR", "VPD"]),
    frozenset(["SII", "PAR", "VPD"]),
]
SUBSET_R2_KEYS = {
    frozenset(["SII"]): "r2_sii",
    frozenset(["PAR"]): "r2_par",
    frozenset(["VPD"]): "r2_vpd",
    frozenset(["SII", "PAR"]): "r2_sii_par",
    frozenset(["SII", "VPD"]): "r2_sii_vpd",
    frozenset(["PAR", "VPD"]): "r2_par_vpd",
    frozenset(["SII", "PAR", "VPD"]): "r2_full",
}
PERMUTATIONS = list(permutations(DRIVER_ORDER))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Shapley/LMG R² decomposition of SII, PAR, VPD")
    p.add_argument("--smoke-test", action="store_true", help="Smoke mode.")
    p.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: results/supplementary_checks; smoke: .../smoke).")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS.keys()), default=None)
    return p.parse_args(argv)


def _resolve_backend(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        try:
            import cupy as cp  # noqa: F401
            return "gpu"
        except Exception as exc:
            raise RuntimeError("GPU backend requested but CuPy is not available") from exc
    # auto
    try:
        import cupy as cp  # noqa: F401
        return "gpu"
    except Exception:
        return "cpu"


def _array_module(backend: str):
    if backend == "gpu":
        import cupy as cp
        return cp
    return np


def _to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def _ols_r2(x: np.ndarray, y: np.ndarray, xp) -> tuple[float, np.ndarray]:
    """
    Fit y = b0 + x @ b + e with an intercept and return (R², coefficients).
    x and y are 2-D/1-D NumPy or CuPy arrays in float64.
    """
    n = x.shape[0]
    ones = xp.ones((n, 1), dtype=xp.float64)
    X = xp.concatenate([ones, x], axis=1)
    coeffs, *_ = xp.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    rss = float(xp.sum((y - y_pred) ** 2))
    tss = float(xp.sum((y - y.mean()) ** 2))
    if tss <= 0:
        r2 = 0.0
    else:
        r2 = 1.0 - rss / tss
    # Numerical safety
    if r2 < 0 and r2 > -TOL:
        r2 = 0.0
    if r2 > 1 and r2 < 1 + TOL:
        r2 = 1.0
    coeffs_cpu = _to_cpu(coeffs)
    return float(r2), coeffs_cpu


def _standardized_coeffs(x: np.ndarray, y: np.ndarray, xp) -> np.ndarray:
    """Return standardized OLS coefficients for the three predictors."""
    y_std = (y - y.mean()) / y.std()
    x_std = (x - x.mean(axis=0)) / x.std(axis=0)
    # Replace NaN std with 1 to avoid division by zero; coefficients will be zero
    x_std = xp.where(xp.isfinite(x_std), x_std, xp.zeros_like(x_std))
    _, coeffs = _ols_r2(x_std, y_std, xp)
    return coeffs[1:]  # drop intercept


def _fit_all_subsets(x_full: np.ndarray, y: np.ndarray, backend: str) -> dict:
    """Fit all seven nonempty predictor subsets and return R² values + full coeffs."""
    xp = _array_module(backend)
    x_gpu = xp.asarray(x_full, dtype=xp.float64)
    y_gpu = xp.asarray(y, dtype=xp.float64)

    r2 = {}
    for subset in SUBSETS:
        idx = [DRIVER_ORDER.index(d) for d in subset]
        r2_val, _ = _ols_r2(x_gpu[:, idx], y_gpu, xp)
        r2[subset] = r2_val

    _, full_coeffs = _ols_r2(x_gpu, y_gpu, xp)
    std_coeffs = _standardized_coeffs(x_gpu, y_gpu, xp)
    return {
        "r2": r2,
        "beta_raw": full_coeffs[1:],
        "beta_standardized": std_coeffs,
    }


def _shapley_values(r2_map: dict) -> dict[str, float]:
    """Average marginal R² increments over all six predictor orderings."""
    sums = {d: 0.0 for d in DRIVER_ORDER}
    for perm in PERMUTATIONS:
        current_set = frozenset()
        current_r2 = 0.0
        for d in perm:
            new_set = current_set | frozenset([d])
            new_r2 = r2_map.get(new_set, 0.0)
            sums[d] += new_r2 - current_r2
            current_set = new_set
            current_r2 = new_r2
    return {d: sums[d] / len(PERMUTATIONS) for d in DRIVER_ORDER}


def _coefficient_sign(beta: float) -> int:
    if not np.isfinite(beta) or abs(beta) < TOL:
        return 0
    return int(np.sign(beta))


def main(argv=None) -> int:
    args = parse_args(argv)
    smoke = args.smoke_test
    backend = _resolve_backend(args.backend)

    windows = args.windows if args.windows is not None else (WINDOWS_SMOKE if smoke else WINDOWS_FULL)
    scenarios = args.scenarios if args.scenarios is not None else (
        ["full_sample"] if smoke else list(SCENARIOS.keys())
    )
    temp_bins = ["Cool"] if smoke else list(TEMP_LABELS)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SMOKE_DIR if smoke else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_long_csv = output_dir / "driver_r2_shapley_long.csv"
    out_models_csv = output_dir / "driver_r2_shapley_models.csv"
    out_json = output_dir / "driver_r2_shapley_metadata.json"

    resolved_columns = {d: p.format(w=windows[0]) for d, p in DRIVER_COL_PATTERNS.items()}

    print("=" * 64)
    print("MAGNETO supplementary checks — Driver R² Shapley/LMG decomposition")
    print("=" * 64)
    print(f"Smoke mode: {smoke}")
    print(f"Backend: {backend}")
    print(f"Windows: {windows}")
    print(f"Scenarios: {scenarios}")
    print(f"Temperature bins: {temp_bins}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{INPUT_PATH}\n"
            "Run 10_prepare_environmental_driver_input.py first."
        )

    df = pd.read_parquet(INPUT_PATH, engine=PARQUET_ENGINE)
    available_scenarios = [s for s in scenarios if s in SCENARIOS]
    missing_scenarios = set(scenarios) - set(available_scenarios)
    if missing_scenarios:
        print(f"[WARN] Scenarios not available: {sorted(missing_scenarios)}")

    long_rows = []
    model_rows = []
    n_fail = 0
    max_decomp_error = 0.0

    for scenario in tqdm(available_scenarios, desc="Scenarios"):
        scen_df = df[SCENARIOS[scenario](df)].copy()

        for temp in tqdm(temp_bins, desc=f"Bins ({scenario})", leave=False):
            bin_df = scen_df[scen_df["temp_bin_label"] == temp]
            n_cells = bin_df[["lat_id", "lon_id"]].drop_duplicates().shape[0]
            eligible_cells = n_cells >= MIN_CELLS

            for w in windows:
                feature_cols = {d: p.format(w=w) for d, p in DRIVER_COL_PATTERNS.items()}
                missing_cols = [c for c in feature_cols.values() if c not in bin_df.columns]

                model_base = {
                    "target": TARGET,
                    "method": METHOD,
                    "scenario": scenario,
                    "temp_bin_label": temp,
                    "window_days": w,
                    "n_cells": n_cells,
                    "backend": backend,
                }

                if missing_cols:
                    reason = "column_not_found"
                    model_rows.append({
                        **model_base,
                        **{k: np.nan for k in SUBSET_R2_KEYS.values()},
                        "shapley_sum": np.nan,
                        "decomposition_error": np.nan,
                        "n_obs": 0,
                        "eligible": False,
                        "eligibility_reason": reason,
                        "calculation_status": "missing_feature_column",
                    })
                    for d in DRIVER_ORDER:
                        long_rows.append({
                            **model_base,
                            "driver": d,
                            "feature_column": feature_cols[d],
                            "shapley_r2": np.nan,
                            "shapley_r2_percent": np.nan,
                            "share_of_explained_r2": np.nan,
                            "r2_full": np.nan,
                            "r2_full_percent": np.nan,
                            "beta_raw": np.nan,
                            "beta_standardized": np.nan,
                            "coefficient_sign": np.nan,
                            "n_obs": 0,
                            "eligible": False,
                            "eligibility_reason": reason,
                            "calculation_status": "missing_feature_column",
                        })
                    n_fail += 1
                    continue

                cols = ["residual"] + list(feature_cols.values()) + ["lat_id", "lon_id"]
                cc = bin_df[cols].dropna()
                n_obs = len(cc)

                if not eligible_cells:
                    reason = "fewer_than_50_cells"
                    model_rows.append({
                        **model_base,
                        **{k: np.nan for k in SUBSET_R2_KEYS.values()},
                        "shapley_sum": np.nan,
                        "decomposition_error": np.nan,
                        "n_obs": n_obs,
                        "eligible": False,
                        "eligibility_reason": reason,
                        "calculation_status": "skipped_underpowered_group",
                    })
                    for d in DRIVER_ORDER:
                        long_rows.append({
                            **model_base,
                            "driver": d,
                            "feature_column": feature_cols[d],
                            "shapley_r2": np.nan,
                            "shapley_r2_percent": np.nan,
                            "share_of_explained_r2": np.nan,
                            "r2_full": np.nan,
                            "r2_full_percent": np.nan,
                            "beta_raw": np.nan,
                            "beta_standardized": np.nan,
                            "coefficient_sign": np.nan,
                            "n_obs": n_obs,
                            "eligible": False,
                            "eligibility_reason": reason,
                            "calculation_status": "skipped_underpowered_group",
                        })
                    continue

                if n_obs < 4:
                    reason = "too_few_complete_cases"
                    n_fail += 1
                    model_rows.append({
                        **model_base,
                        **{k: np.nan for k in SUBSET_R2_KEYS.values()},
                        "shapley_sum": np.nan,
                        "decomposition_error": np.nan,
                        "n_obs": n_obs,
                        "eligible": False,
                        "eligibility_reason": reason,
                        "calculation_status": "insufficient_data",
                    })
                    for d in DRIVER_ORDER:
                        long_rows.append({
                            **model_base,
                            "driver": d,
                            "feature_column": feature_cols[d],
                            "shapley_r2": np.nan,
                            "shapley_r2_percent": np.nan,
                            "share_of_explained_r2": np.nan,
                            "r2_full": np.nan,
                            "r2_full_percent": np.nan,
                            "beta_raw": np.nan,
                            "beta_standardized": np.nan,
                            "coefficient_sign": np.nan,
                            "n_obs": n_obs,
                            "eligible": False,
                            "eligibility_reason": reason,
                            "calculation_status": "insufficient_data",
                        })
                    continue

                try:
                    y = cc["residual"].values.astype(np.float64)
                    x_full = cc[list(feature_cols.values())].values.astype(np.float64)
                    fit = _fit_all_subsets(x_full, y, backend)
                    r2_map = fit["r2"]
                    shapley = _shapley_values(r2_map)
                    r2_full = r2_map[frozenset(["SII", "PAR", "VPD"])]
                    shapley_sum = sum(shapley.values())
                    decomp_error = abs(shapley_sum - r2_full)
                    max_decomp_error = max(max_decomp_error, decomp_error)

                    if decomp_error > TOL:
                        raise ValueError(
                            f"Shapley decomposition error {decomp_error:.2e} exceeds tolerance {TOL}"
                        )

                    if not all(-TOL <= v <= r2_full + TOL for v in shapley.values()):
                        raise ValueError("Negative Shapley contribution larger than tolerance")

                    beta_raw = fit["beta_raw"]
                    beta_std = fit["beta_standardized"]

                    model_row = {
                        **model_base,
                        SUBSET_R2_KEYS[frozenset(["SII"])]: r2_map[frozenset(["SII"])],
                        SUBSET_R2_KEYS[frozenset(["PAR"])]: r2_map[frozenset(["PAR"])],
                        SUBSET_R2_KEYS[frozenset(["VPD"])]: r2_map[frozenset(["VPD"])],
                        SUBSET_R2_KEYS[frozenset(["SII", "PAR"])]: r2_map[frozenset(["SII", "PAR"])],
                        SUBSET_R2_KEYS[frozenset(["SII", "VPD"])]: r2_map[frozenset(["SII", "VPD"])],
                        SUBSET_R2_KEYS[frozenset(["PAR", "VPD"])]: r2_map[frozenset(["PAR", "VPD"])],
                        SUBSET_R2_KEYS[frozenset(["SII", "PAR", "VPD"])]: r2_full,
                        "shapley_sum": shapley_sum,
                        "decomposition_error": decomp_error,
                        "n_obs": n_obs,
                        "eligible": True,
                        "eligibility_reason": "",
                        "calculation_status": "ok",
                    }
                    model_rows.append(model_row)

                    for i, d in enumerate(DRIVER_ORDER):
                        share = shapley[d] / r2_full if r2_full > TOL else np.nan
                        long_rows.append({
                            **model_base,
                            "driver": d,
                            "feature_column": feature_cols[d],
                            "shapley_r2": shapley[d],
                            "shapley_r2_percent": 100.0 * shapley[d],
                            "share_of_explained_r2": share,
                            "r2_full": r2_full,
                            "r2_full_percent": 100.0 * r2_full,
                            "beta_raw": beta_raw[i],
                            "beta_standardized": beta_std[i],
                            "coefficient_sign": _coefficient_sign(beta_std[i]),
                            "n_obs": n_obs,
                            "eligible": True,
                            "eligibility_reason": "",
                            "calculation_status": "ok",
                        })

                except Exception as exc:
                    n_fail += 1
                    reason = f"calculation_error: {type(exc).__name__}"
                    model_rows.append({
                        **model_base,
                        **{k: np.nan for k in SUBSET_R2_KEYS.values()},
                        "shapley_sum": np.nan,
                        "decomposition_error": np.nan,
                        "n_obs": n_obs,
                        "eligible": False,
                        "eligibility_reason": reason,
                        "calculation_status": "calculation_failed",
                    })
                    for d in DRIVER_ORDER:
                        long_rows.append({
                            **model_base,
                            "driver": d,
                            "feature_column": feature_cols[d],
                            "shapley_r2": np.nan,
                            "shapley_r2_percent": np.nan,
                            "share_of_explained_r2": np.nan,
                            "r2_full": np.nan,
                            "r2_full_percent": np.nan,
                            "beta_raw": np.nan,
                            "beta_standardized": np.nan,
                            "coefficient_sign": np.nan,
                            "n_obs": n_obs,
                            "eligible": False,
                            "eligibility_reason": reason,
                            "calculation_status": "calculation_failed",
                        })

    long_df = pd.DataFrame(long_rows)
    models_df = pd.DataFrame(model_rows)

    if long_df.empty:
        print("[WARN] No rows produced.")
        return 0

    long_df = long_df.sort_values(
        ["scenario", "temp_bin_label", "window_days", "driver"]
    ).reset_index(drop=True)
    models_df = models_df.sort_values(
        ["scenario", "temp_bin_label", "window_days"]
    ).reset_index(drop=True)

    # ── Runtime validations ─────────────────────────────────────────────────
    long_key = ["scenario", "temp_bin_label", "window_days", "driver"]
    if long_df.duplicated(long_key, keep=False).any():
        raise ValueError(
            "Duplicate scenario × temperature × window × driver rows found."
        )

    if not set(long_df["driver"].unique()).issubset(set(DRIVER_ORDER)):
        raise ValueError(f"Unexpected drivers: {long_df['driver'].unique().tolist()}")

    eligible_long = long_df[long_df["eligible"]]
    if not eligible_long.empty:
        if (eligible_long["shapley_r2"] < -TOL).any():
            raise ValueError("Negative Shapley R² found")
        grouped = eligible_long.groupby(["scenario", "temp_bin_label", "window_days"])["shapley_r2"].sum()
        full_r2 = eligible_long.groupby(["scenario", "temp_bin_label", "window_days"])["r2_full"].first()
        if not np.allclose(grouped.values, full_r2.values, atol=TOL):
            raise ValueError("Shapley contributions do not sum to full-model R²")

    eligible_models = models_df[models_df["eligible"]]
    if not eligible_models.empty:
        if not eligible_models["r2_full"].between(-TOL, 1 + TOL).all():
            raise ValueError("R² outside [0, 1] within tolerance")
        for k in ["r2_sii", "r2_par", "r2_vpd", "r2_sii_par", "r2_sii_vpd", "r2_par_vpd"]:
            if not eligible_models[k].between(-TOL, 1 + TOL).all():
                raise ValueError(f"Subset R² {k} outside [0, 1] within tolerance")

    atomic_write(long_df, out_long_csv, fmt="csv")
    atomic_write(models_df, out_models_csv, fmt="csv")
    print(f"\n[OK] Written: {out_long_csv} ({len(long_df):,} rows)")
    print(f"[OK] Written: {out_models_csv} ({len(models_df):,} rows)")

    # ── Metadata JSON ───────────────────────────────────────────────────────
    metadata = {
        "stage": "S11_driver_r2_shapley.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit() or "",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "input_size_bytes": INPUT_PATH.stat().st_size,
        "input_modified_utc": datetime.fromtimestamp(
            INPUT_PATH.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "resolved_columns": resolved_columns,
        "target": TARGET,
        "method": METHOD,
        "smoke": smoke,
        "backend": backend,
        "scenarios": scenarios,
        "temperature_bins": temp_bins,
        "windows": windows,
        "drivers": DRIVER_ORDER,
        "n_long_rows": len(long_df),
        "n_model_rows": len(models_df),
        "n_eligible_models": int(models_df["eligible"].sum()),
        "n_ineligible_models": int((~models_df["eligible"]).sum()),
        "n_calculation_failures": n_fail,
        "numerical_tolerance": TOL,
        "max_decomposition_error": float(max_decomp_error),
        "decomposition_tests_passed": True,
        "output_long_csv": str(out_long_csv.relative_to(PROJECT_ROOT)),
        "output_models_csv": str(out_models_csv.relative_to(PROJECT_ROOT)),
        "output_long_csv_sha256": full_sha256(out_long_csv),
        "output_models_csv_sha256": full_sha256(out_models_csv),
    }
    out_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"[OK] Written: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
