#!/usr/bin/env python3
"""
11_environmental_driver_matrix_gpu.py — Multivariate environmental-driver matrix.

This is a minimal adapter around the validated v1 GPU engine
(`v1/scripts/11_Matrix_Search_GPU.py`). It keeps the numerical core unchanged
and only replaces the data loader, scenario definitions, and output paths with
their current-pipeline equivalents.

Model fitted per temperature bin:

    SIF_resid = β₀ + β_SII·SII + β_PAR·PAR + β_VPD·VPD + ε

For every combination of:
    target × method × scenario × temp_bin × sii_window × par_window × vpd_window

the script records the OLS coefficients, t-statistics, p-values, sample size,
effective sample size, and residual sum of squares.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cupy as cp
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _Common import (
    PROJECT_ROOT,
    PARQUET_ENGINE,
    TEMP_LABELS,
    atomic_write,
    full_sha256,
)

warnings.filterwarnings("ignore")


# ── Paths and defaults ──────────────────────────────────────────────────────
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "environmental_driver_input.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "environmental_driver_matrix"
COMBINED_PATH = PROJECT_ROOT / "results" / "environmental_driver_matrix.parquet"
AUDIT_PATH = PROJECT_ROOT / "results" / "environmental_driver_matrix_audit.json"

DEFAULT_TARGET = "sif_771nm"
DEFAULT_METHOD = "harmonic"
DEFAULT_WINDOWS = list(range(1, 29))

SCENARIOS = {
    "full_sample": lambda df: pd.Series(True, index=df.index),
    "persistently_vegetated": lambda df: df["is_vegetated"],
    "control_strict_low_lai": lambda df: df["is_strict_low_lai"],
    "Sahara": lambda df: df["is_Sahara"],
    "SAA": lambda df: df["is_SAA"],
}


def _git_commit() -> str:
    try:
        import subprocess
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            )
            .strip()
        )
    except Exception:
        return ""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Environmental-driver matrix search (GPU)")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
    )
    p.add_argument(
        "--sii-windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    p.add_argument(
        "--par-windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    p.add_argument(
        "--vpd-windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    p.add_argument(
        "--temperature-bins",
        nargs="+",
        default=TEMP_LABELS,
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=50,
        help="Minimum observations in a temperature bin to fit models",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a scenario if its CSV already exists and is non-empty",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: overrides defaults to a tiny grid",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-scenario CSVs (default: results/environmental_driver_matrix).",
    )
    p.add_argument(
        "--combined-parquet",
        type=Path,
        default=None,
        help="Path for combined Parquet output (default: results/environmental_driver_matrix.parquet).",
    )
    p.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Path for audit JSON (default: results/environmental_driver_matrix_audit.json).",
    )
    return p.parse_args(argv)


def _apply_smoke_defaults(args):
    """Shrink the grid to a demonstrative smoke test."""
    args.scenarios = ["full_sample"]
    args.temperature_bins = ["Cool"]
    args.sii_windows = [28]
    args.par_windows = [1, 7, 14, 21, 28]
    args.vpd_windows = [1, 7, 14, 21, 28]
    return args


# ── GPU OLS solver (verbatim from v1) ───────────────────────────────────────

def solve_ols_gpu(y_vec: np.ndarray, X_mat: np.ndarray):
    """Solve (X.T X)^-1 X.T y on GPU. Returns betas, t-stats, rss, n."""
    try:
        # Use float64 for numerical stability in the full-cube decomposition.
        y_gpu = cp.array(y_vec, dtype=cp.float64)
        X_gpu = cp.array(X_mat, dtype=cp.float64)

        N, K = X_mat.shape
        xtx = X_gpu.T @ X_gpu
        xty = X_gpu.T @ y_gpu

        if cp.linalg.det(xtx) == 0:
            return None

        B = cp.linalg.solve(xtx, xty)
        residuals = y_gpu - (X_gpu @ B)
        rss = cp.sum(residuals ** 2)

        df = N - K
        if df <= 0:
            return None

        var_res = rss / df
        inv_xtx = cp.linalg.inv(xtx)
        se = cp.sqrt(cp.diag(inv_xtx) * var_res)
        t_stats = B / se

        return {
            "beta": cp.asnumpy(B),
            "t": cp.asnumpy(t_stats),
            "rss": float(rss),
            "n": N,
        }
    except Exception:
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _window_cols(prefix: str, windows: Iterable[int]) -> dict[int, str]:
    return {w: f"{prefix}_ma{w}" for w in windows}


def _scenario_out_path(target: str, scenario: str, output_dir: Path | None = None) -> Path:
    out_dir = output_dir if output_dir is not None else OUTPUT_DIR
    return out_dir / f"matrix_search_{target}_{scenario}.csv"


def _log_memory(label: str = "") -> None:
    """Print GPU and host memory status with a timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_total = props["totalGlobalMem"]
        gpu_free = cp.cuda.runtime.memGetInfo()[0]
        gpu_used = gpu_total - gpu_free
        gpu_pct = 100.0 * gpu_used / gpu_total
        print(
            f"[{ts}] MEMORY {label} | GPU: "
            f"{gpu_used / 1024**3:.2f}/{gpu_total / 1024**3:.2f} GB "
            f"({gpu_pct:.1f}%) | Free: {gpu_free / 1024**3:.2f} GB"
        )
    except Exception as exc:
        print(f"[{ts}] MEMORY {label} | GPU memory query failed: {exc}")

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(
            f"[{ts}] MEMORY {label} | Host RAM: "
            f"{vm.used / 1024**3:.2f}/{vm.total / 1024**3:.2f} GB "
            f"({vm.percent}%) | Free: {vm.available / 1024**3:.2f} GB"
        )
    except Exception:
        pass


def _run_scenario(
    df_input: pd.DataFrame,
    *,
    target: str,
    method: str,
    scenario: str,
    sii_windows: list[int],
    par_windows: list[int],
    vpd_windows: list[int],
    temp_bins: list[str],
    min_obs: int,
) -> tuple[pd.DataFrame, dict]:
    """Run the matrix for one scenario. Returns results + diagnostics."""
    mask = SCENARIOS[scenario](df_input)
    df_scen = df_input[mask].copy()

    sii_cols = _window_cols("sii_mean", sii_windows)
    par_cols = _window_cols("par", par_windows)
    vpd_cols = _window_cols("vpd", vpd_windows)

    avail_sii = {w: c for w, c in sii_cols.items() if c in df_scen.columns}
    avail_par = {w: c for w, c in par_cols.items() if c in df_scen.columns}
    avail_vpd = {w: c for w, c in vpd_cols.items() if c in df_scen.columns}

    results = []
    n_failed = 0
    n_singular = 0

    # Progress over bins; inner loops over windows are too fine-grained for tqdm.
    for b_label in temp_bins:
        bin_df = df_scen[df_scen["temp_bin_label"] == b_label]
        if bin_df.empty:
            continue

        valid_mask = np.isfinite(bin_df["residual"].values)
        bin_df = bin_df[valid_mask]
        if len(bin_df) < min_obs:
            continue

        y = bin_df["residual"].values.astype(np.float64)
        N = len(y)

        # Effective sample size via lag-1 autocorrelation (Chelton/Pyper).
        if N > 10:
            r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isnan(r1):
                r1 = 0.5
            n_eff = N * (1 - r1) / (1 + r1)
        else:
            n_eff = N

        sii_data = {w: bin_df[c].values.astype(np.float64) for w, c in avail_sii.items()}
        par_data = {w: bin_df[c].values.astype(np.float64) for w, c in avail_par.items()}
        vpd_data = {w: bin_df[c].values.astype(np.float64) for w, c in avail_vpd.items()}
        ones = np.ones(N, dtype=np.float64)

        n_bin_models = len(sii_data) * len(par_data) * len(vpd_data)
        bin_start = datetime.now(timezone.utc)
        print(
            f"[{bin_start.strftime('%Y-%m-%dT%H:%M:%SZ')}] START {scenario} / {b_label} | "
            f"N={N:,} | models={n_bin_models:,}"
        )
        _log_memory(f"{scenario}/{b_label} start")

        for w_sii, v_sii in tqdm(
            sii_data.items(),
            desc=f"{scenario} {b_label} SII",
            leave=False,
            disable=len(sii_data) < 2,
        ):
            for w_par, v_par in par_data.items():
                for w_vpd, v_vpd in vpd_data.items():
                    X = np.column_stack([v_sii, v_par, v_vpd, ones])
                    try:
                        res = solve_ols_gpu(y, X)
                    except Exception as exc:
                        print(
                            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
                            f"GPU ERROR {scenario}/{b_label} SII={w_sii} PAR={w_par} VPD={w_vpd}: {exc}"
                        )
                        n_failed += 1
                        continue
                    if res is None:
                        n_failed += 1
                        continue
                    if np.any(np.isnan(res["t"])) or np.any(np.isinf(res["t"])):
                        n_singular += 1
                        continue

                    dof = max(1, int(n_eff) - 4)
                    p_vals = 2 * (1 - stats.t.cdf(np.abs(res["t"]), df=dof))

                    results.append({
                        "target": target,
                        "method": method,
                        "scenario": scenario,
                        "temp_bin_label": b_label,
                        "sii_window": w_sii,
                        "par_window": w_par,
                        "vpd_window": w_vpd,
                        "n": res["n"],
                        "n_eff": float(n_eff),
                        "rss": res["rss"],
                        "beta_sii": float(res["beta"][0]),
                        "beta_par": float(res["beta"][1]),
                        "beta_vpd": float(res["beta"][2]),
                        "t_sii": float(res["t"][0]),
                        "t_par": float(res["t"][1]),
                        "t_vpd": float(res["t"][2]),
                        "p_sii": float(p_vals[0]),
                        "p_par": float(p_vals[1]),
                        "p_vpd": float(p_vals[2]),
                    })

        bin_elapsed = datetime.now(timezone.utc) - bin_start
        print(
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] DONE {scenario} / {b_label} | "
            f"elapsed={bin_elapsed.total_seconds():.1f}s | results={len(results):,} | "
            f"failed={n_failed} | singular={n_singular}"
        )
        _log_memory(f"{scenario}/{b_label} done")

    df = pd.DataFrame(results)
    diag = {
        "scenario": scenario,
        "input_rows": int(len(df_scen)),
        "output_rows": int(len(df)),
        "n_failed": int(n_failed),
        "n_singular": int(n_singular),
    }
    return df, diag


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args = _apply_smoke_defaults(args)

    print("=" * 64)
    print("MAGNETO — Environmental-driver matrix search (GPU)")
    print("=" * 64)
    print(f"Target:   {args.target}")
    print(f"Method:   {args.method}")
    print(f"Scenarios: {args.scenarios}")
    print(f"SII windows:   {args.sii_windows}")
    print(f"PAR windows:   {args.par_windows}")
    print(f"VPD windows:   {args.vpd_windows}")
    print(f"Temperature bins: {args.temperature_bins}")
    print(f"Min obs per bin:  {args.min_obs}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{INPUT_PATH}\n"
            "Run 10_prepare_environmental_driver_input.py first."
        )

    print(f"\n[1/3] Loading input table …")
    df_input = pd.read_parquet(INPUT_PATH, engine=PARQUET_ENGINE)
    df_input = df_input[df_input["temp_bin_label"].isin(args.temperature_bins)].copy()
    print(f"  Rows after temperature filter: {len(df_input):,}")

    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    combined_path = args.combined_parquet if args.combined_parquet is not None else COMBINED_PATH
    audit_path = args.audit_json if args.audit_json is not None else AUDIT_PATH

    expected_rows = (
        len(args.scenarios)
        * len(args.temperature_bins)
        * len(args.sii_windows)
        * len(args.par_windows)
        * len(args.vpd_windows)
    )
    print(f"  Expected matrix rows: {expected_rows:,}")
    print(f"  Output Parquet: {combined_path}")
    print(f"  Audit JSON:     {audit_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _log_memory("after load")

    all_results: list[pd.DataFrame] = []
    scenario_diagnostics: list[dict] = []

    n_total_models = 0
    for scenario in args.scenarios:
        out_path = _scenario_out_path(args.target, scenario, output_dir=output_dir)
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 100:
            print(f"\n  [SKIP] {scenario}: {out_path} exists")
            existing = pd.read_csv(out_path)
            all_results.append(existing)
            scenario_diagnostics.append({
                "scenario": scenario,
                "input_rows": np.nan,
                "output_rows": len(existing),
                "n_failed": 0,
                "n_singular": 0,
                "skipped": True,
            })
            continue

        scen_start = datetime.now(timezone.utc)
        print(
            f"\n[{scen_start.strftime('%Y-%m-%dT%H:%M:%SZ')}] [2/3] Running scenario: {scenario}"
        )
        df_scen, diag = _run_scenario(
            df_input,
            target=args.target,
            method=args.method,
            scenario=scenario,
            sii_windows=sorted(args.sii_windows),
            par_windows=sorted(args.par_windows),
            vpd_windows=sorted(args.vpd_windows),
            temp_bins=args.temperature_bins,
            min_obs=args.min_obs,
        )
        scen_elapsed = datetime.now(timezone.utc) - scen_start
        print(
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"Scenario {scenario} done | elapsed={scen_elapsed.total_seconds():.1f}s | "
            f"rows: {len(df_scen):,} | failed: {diag['n_failed']} | singular: {diag['n_singular']}"
        )
        if not df_scen.empty:
            atomic_write(df_scen, out_path, fmt="csv")
            print(f"  Written: {out_path}")
            all_results.append(df_scen)
        scenario_diagnostics.append(diag)
        n_total_models += len(df_scen)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        _log_memory(f"after {scenario}")

    # ── Combined Parquet output ────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        atomic_write(combined, combined_path, fmt="parquet")
        print(f"\n[3/3] Combined matrix written: {combined_path} ({len(combined):,} rows)")
    else:
        combined = pd.DataFrame()
        print("\n[3/3] No results to combine")

    # ── Audit manifest ─────────────────────────────────────────────────────
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_path": str(INPUT_PATH),
        "input_hash": full_sha256(INPUT_PATH),
        "output_dir": str(output_dir),
        "combined_path": str(combined_path),
        "target": args.target,
        "method": args.method,
        "scenarios": args.scenarios,
        "temperature_bins": args.temperature_bins,
        "sii_windows": sorted(args.sii_windows),
        "par_windows": sorted(args.par_windows),
        "vpd_windows": sorted(args.vpd_windows),
        "min_obs": args.min_obs,
        "number_of_input_rows": int(len(df_input)),
        "number_of_output_rows": int(len(combined)),
        "number_of_failed_models": int(sum(d.get("n_failed", 0) for d in scenario_diagnostics)),
        "number_of_singular_models": int(sum(d.get("n_singular", 0) for d in scenario_diagnostics)),
        "scenario_diagnostics": scenario_diagnostics,
        "gpu_name": str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")),
    }
    atomic_write(audit, audit_path, fmt="json")
    print(f"  Audit: {audit_path}")
    _log_memory("final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
