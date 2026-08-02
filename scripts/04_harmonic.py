#!/usr/bin/env python3
"""
04_harmonic.py — Harmonic leave-one-year-out detrending.

Model per cell × target year:
    SIF = intercept + linear calendar-time trend
          + sin(annual phase) + cos(annual phase) + residual

Target year is fully excluded from the fit; predictions use only out-of-year data.
"""
from __future__ import annotations
import sys, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from _Common import *


def fit_cell_harmonic(df_cell: pd.DataFrame, target: str = "sif_771nm") -> pd.DataFrame:
    """Fit harmonic model with calendar-time linear trend, leave-one-year-out."""
    years = sorted(df_cell["year"].unique())
    results = []
    period = 365.25

    for test_year in years:
        train = df_cell[df_cell["year"] != test_year]
        test = df_cell[df_cell["year"] == test_year]

        if len(train) < HARMONIC_MIN_OBS or train["year"].nunique() < HARMONIC_MIN_YEARS:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, len(train), train["year"].nunique(), "insufficient_data"))
            continue

        t_doy = train["doy"].values.astype(float)
        t_cal = train["year"].values.astype(float) + t_doy / period
        cal_mean = t_cal.mean()
        X = np.column_stack([
            np.ones_like(t_doy),
            t_cal - cal_mean,
            np.sin(2.0 * np.pi * t_doy / period),
            np.cos(2.0 * np.pi * t_doy / period),
        ])
        y = train[target].values.astype(float)

        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_v, y_v = X[valid], y[valid]

        if len(y_v) < 4:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, len(y_v), train["year"].nunique(), "insufficient_valid_obs"))
            continue

        try:
            coefs, residues, rank, sv = np.linalg.lstsq(X_v, y_v, rcond=None)
        except np.linalg.LinAlgError:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, len(y_v), train["year"].nunique(), "lstsq_failed"))
            continue

        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        y_pred_train = X_v @ coefs
        ss_res = np.sum((y_v - y_pred_train) ** 2)
        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        resid_sd = float(np.sqrt(ss_res / (len(y_v) - rank))) if len(y_v) > rank else np.nan

        t_test_doy = test["doy"].values.astype(float)
        t_test_cal = test["year"].values.astype(float) + t_test_doy / period
        X_test = np.column_stack([
            np.ones_like(t_test_doy),
            t_test_cal - cal_mean,
            np.sin(2.0 * np.pi * t_test_doy / period),
            np.cos(2.0 * np.pi * t_test_doy / period),
        ])
        y_pred = X_test @ coefs
        y_obs = test[target].values

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "date": row["date"], "year": row["year"], "doy": row["doy"],
                "lat_id": row["lat_id"], "lon_id": row["lon_id"],
                "observed_sif": float(y_obs[i]),
                "expected_sif": float(y_pred[i]),
                "residual": float(y_obs[i] - y_pred[i]),
                "method": "harmonic",
                "n_train_obs": len(train),
                "n_train_years": train["year"].nunique(),
                "r_squared": r_sq,
                "residual_sd": resid_sd,
                "condition_number": float(cond),
                "intercept": float(coefs[0]),
                "trend_coef": float(coefs[1]),
                "sin_coef": float(coefs[2]),
                "cos_coef": float(coefs[3]),
                "design_rank": int(rank),
                "pass_flag": True,
                "fail_reason": "",
            })

    return pd.DataFrame(results)


def _fail_row(row, target, n_obs, n_years, reason):
    return {
        "date": row["date"], "year": row["year"], "doy": row["doy"],
        "lat_id": row["lat_id"], "lon_id": row["lon_id"],
        "observed_sif": row[target], "expected_sif": np.nan, "residual": np.nan,
        "method": "harmonic", "n_train_obs": n_obs, "n_train_years": n_years,
        "r_squared": np.nan, "residual_sd": np.nan, "condition_number": np.nan,
        "intercept": np.nan, "trend_coef": np.nan, "sin_coef": np.nan, "cos_coef": np.nan,
        "design_rank": np.nan, "pass_flag": False, "fail_reason": reason,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGNETO harmonic leave-one-year-out detrending")
    parser.add_argument("--target", type=str, default="sif_771nm",
                        help="SIF column to detrend (default: sif_771nm)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: config file_harmonic_analysis)")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    target = args.target
    output_path = Path(args.output) if args.output else FILE_RES_HARMONIC

    print("=" * 64)
    print(f"MAGNETO — Harmonic Detrending ({target})")
    print("=" * 64)
    setup_dirs()

    print("\n[1/3] Loading QC dataset …")
    df = pd.read_parquet(FILE_QC, engine=PARQUET_ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  QC rows: {len(df):,}")

    if target not in df.columns:
        print(f"[FAIL] Target column not found in QC: {target}")
        return 1

    print("\n[2/3] Joining LAI/control membership …")
    members = pd.read_parquet(FILE_LAI_MEMBERS, engine=PARQUET_ENGINE)
    df = df.merge(members, on=["lat_id", "lon_id"], how="left")
    print(f"  After membership join: {len(df):,}")

    if ERA5_TEMP_COL not in df.columns:
        print(f"  [WARN] {ERA5_TEMP_COL} not in QC — temperature will be missing")

    print("\n[3/3] Fitting harmonic model per cell …")
    cells = df.groupby(["lat_id", "lon_id"])
    all_results = []
    for (lat_id, lon_id), df_cell in tqdm(cells, total=cells.ngroups, desc="harmonic cells"):
        r = fit_cell_harmonic(df_cell, target=target)
        if len(r) > 0:
            all_results.append(r)

    if not all_results:
        print("[FAIL] No results produced — check input data")
        return 1

    df_out = pd.concat(all_results, ignore_index=True)

    # join temperature metadata
    temp_cols = [c for c in [ERA5_TEMP_COL, "temp_bin_id", "temp_bin_label"] if c in df.columns]
    if temp_cols:
        df_temp = df[["date", "lat_id", "lon_id"] + temp_cols].copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out = df_out.merge(df_temp, on=["date", "lat_id", "lon_id"], how="left")

    # join membership columns again on the residual keys (safety)
    member_cols = [c for c in members.columns if c not in ["lat_id", "lon_id"]]
    df_out = df_out.merge(members, on=["lat_id", "lon_id"], how="left")

    atomic_write(df_out, output_path)
    n_pass = df_out["pass_flag"].sum()
    print(f"\n  Harmonic residuals: {n_pass:,}/{len(df_out):,} passed")
    print(f"  Written: {output_path}")
    if n_pass == 0:
        print("[WARN] Zero passing fits — check minimum thresholds")
    print(f"\n[OK] 04_harmonic complete — {len(df_out):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
