#!/usr/bin/env python3
"""
05_cyclic_spline.py — Corrected cyclic cubic regression spline detrending.

Model per cell × target year:
    SIF = intercept + linear calendar-time trend
          + cyclic_cubic_regression_spline(seasonal_phase, df=8)
          + residual

A single climatological phase variable is used for train and test, excluding
29 February and wrapping correctly at the year boundary. The test-year design
matrix is built from the training design_info to guarantee identical basis.
"""
from __future__ import annotations
import sys, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from patsy import dmatrix, build_design_matrices
from _Common import *

MAX_CONDITION_NUMBER = 1e8
MAX_PREDICTION_RANGE = 10.0
MIN_SEASONAL_COVERAGE = 0.5


def _seasonal_coverage(seasonal_pos):
    if len(seasonal_pos) == 0:
        return 0.0
    counts, _ = np.histogram(seasonal_pos, bins=np.linspace(0, 1, 11))
    return float((counts > 0).sum()) / 10.0


def _fail_row(row, target, n_obs, n_years, reason):
    return {
        "date": row["date"], "year": row["year"], "doy": row["doy"],
        "lat_id": row["lat_id"], "lon_id": row["lon_id"],
        "observed_sif": row[target], "expected_sif": np.nan, "residual": np.nan,
        "method": "cyclic_spline", "spline_df": CYCLIC_SPLINE_DF,
        "design_rank": np.nan, "condition_number": np.nan,
        "n_train_obs": n_obs, "n_train_years": n_years,
        "r_squared": np.nan, "residual_sd": np.nan,
        "prediction_range": np.nan, "seasonal_coverage": np.nan,
        "fit_status": "failed", "fail_reason": reason,
    }


def _phase_from_doy(doy, year):
    """Climatological phase in [0, 1), excluding Feb 29."""
    clim = mmdd_to_clim_index(pd.Series(doy), pd.Series(year))
    # Jan 1 -> 0, Dec 31 -> 364/365
    return (clim - 1) / 365.0


def fit_cell_cyclic_spline(df_cell, target="sif_771nm"):
    years = sorted(df_cell["year"].unique())
    results = []
    spline_df = CYCLIC_SPLINE_DF
    period = 365.25

    for test_year in years:
        train = df_cell[df_cell["year"] != test_year].copy()
        test = df_cell[df_cell["year"] == test_year].copy()

        n_raw_obs = len(train)
        n_raw_years = train["year"].nunique()

        if n_raw_obs < CYCLIC_SPLINE_MIN_OBS or n_raw_years < CYCLIC_SPLINE_MIN_YEARS:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, n_raw_obs, n_raw_years, "insufficient_data"))
            continue

        # phase variable (consistent for train and test, leap-year safe)
        train["phase"] = _phase_from_doy(train["doy"].values, train["year"].values).values
        test["phase"] = _phase_from_doy(test["doy"].values, test["year"].values).values

        cal_time_train = train["year"].values.astype(float) + train["doy"].values.astype(float) / period
        cal_time_mean = float(cal_time_train.mean())
        cal_time_ctr_train = cal_time_train - cal_time_mean

        y_train = train[target].values.astype(float)
        phase_train = train["phase"].values.astype(float)

        valid = (np.isfinite(y_train) & np.isfinite(phase_train) & np.isfinite(cal_time_ctr_train))
        if valid.sum() < spline_df + 2:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, valid.sum(), n_raw_years, "too_few_valid_obs"))
            continue

        phase_train = phase_train[valid]
        cal_time_ctr_train = cal_time_ctr_train[valid]
        y_train = y_train[valid]
        n_valid = len(y_train)
        n_valid_years = len(np.unique(train["year"].values[valid]))

        try:
            data_train = pd.DataFrame({
                "phase": phase_train,
                "cal_time_ctr": cal_time_ctr_train,
            })
            formula = (
                f"cc(phase, df={spline_df}, constraints='center', "
                f"lower_bound=0.0, upper_bound=1.0) + cal_time_ctr"
            )
            X_df = dmatrix(formula, data_train, return_type="dataframe")
            design_info = X_df.design_info
        except Exception as exc:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, n_valid, n_valid_years, f"patsy_dmatrix:{exc}"))
            continue

        X = X_df.values.astype(float)
        col_names = list(X_df.columns)
        intercept_cols = [c for c in col_names if c.strip().lower() == "intercept"]
        if len(intercept_cols) != 1:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, n_valid, n_valid_years, f"intercept_count={len(intercept_cols)}"))
            continue

        try:
            coefs, _, rank, sv = np.linalg.lstsq(X, y_train, rcond=None)
        except np.linalg.LinAlgError as exc:
            for _, row in test.iterrows():
                results.append(_fail_row(row, target, n_valid, n_valid_years, f"lstsq:{exc}"))
            continue

        design_rank = int(rank)
        full_rank = X.shape[1]
        condition_number = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else np.inf

        y_pred_train = X @ coefs
        ss_res = float(np.sum((y_train - y_pred_train) ** 2))
        ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_df = max(1, n_valid - design_rank)
        residual_sd = float(np.sqrt(ss_res / residual_df))
        seasonal_cov = _seasonal_coverage(phase_train)

        fit_ok = True
        failure = ""
        if design_rank < full_rank:
            fit_ok, failure = False, f"rank_deficient:{design_rank}/{full_rank}"
        elif condition_number > MAX_CONDITION_NUMBER:
            fit_ok, failure = False, f"cond_number:{condition_number:.1f}"
        elif not np.all(np.isfinite(coefs)):
            fit_ok, failure = False, "coefs_nan_inf"
        elif not np.isfinite(r_squared) or r_squared < 0.0:
            fit_ok, failure = False, f"r_squared:{r_squared:.4f}"
        elif not np.isfinite(residual_sd):
            fit_ok, failure = False, "residual_sd_nan"
        elif seasonal_cov < MIN_SEASONAL_COVERAGE:
            fit_ok, failure = False, f"seasonal_coverage:{seasonal_cov:.2f}"

        phase_test = test["phase"].values.astype(float)
        cal_time_test = test["year"].values.astype(float) + test["doy"].values.astype(float) / period
        cal_time_ctr_test = cal_time_test - cal_time_mean
        y_obs_test = test[target].values.astype(float)

        y_pred_full = np.full(len(test), np.nan, dtype=float)
        if fit_ok:
            try:
                data_test = pd.DataFrame({"phase": phase_test, "cal_time_ctr": cal_time_ctr_test})
                X_test_df = build_design_matrices([design_info], data_test, return_type="dataframe")[0]
                X_test = X_test_df.values.astype(float)
                y_pred_arr = X_test @ coefs
                if not np.all(np.isfinite(y_pred_arr)):
                    fit_ok, failure = False, "prediction_nan_inf"
                else:
                    pred_range = float(np.ptp(y_pred_arr))
                    if pred_range > MAX_PREDICTION_RANGE:
                        fit_ok, failure = False, f"pred_range:{pred_range:.1f}"
                    else:
                        y_pred_full[X_test_df.index] = y_pred_arr
            except Exception as exc:
                fit_ok, failure = False, f"prediction:{exc}"

        for i, (_, row) in enumerate(test.iterrows()):
            if fit_ok and np.isfinite(y_pred_full[i]):
                yp = float(y_pred_full[i])
                resid = float(y_obs_test[i] - yp)
                pr = float(np.ptp(y_pred_full))
            else:
                yp = np.nan
                resid = np.nan
                pr = np.nan

            results.append({
                "date": row["date"], "year": row["year"], "doy": row["doy"],
                "lat_id": row["lat_id"], "lon_id": row["lon_id"],
                "observed_sif": float(y_obs_test[i]),
                "expected_sif": yp,
                "residual": resid,
                "method": "cyclic_spline",
                "spline_df": spline_df,
                "design_rank": design_rank,
                "condition_number": condition_number,
                "n_train_obs": n_valid,
                "n_train_years": n_valid_years,
                "r_squared": r_squared,
                "residual_sd": residual_sd,
                "prediction_range": pr,
                "seasonal_coverage": seasonal_cov,
                "fit_status": "ok" if fit_ok and np.isfinite(yp) else "failed",
                "fail_reason": failure if (fit_ok and np.isfinite(yp)) else "nan_phase_or_prediction",
            })

    return pd.DataFrame(results)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGNETO corrected cyclic-spline detrending")
    parser.add_argument("--target", type=str, default="sif_771nm",
                        help="SIF column to detrend (default: sif_771nm)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: config file_cyclic_spline_analysis)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    target = args.target
    output_path = Path(args.output) if args.output else FILE_RES_CYCLIC

    print("=" * 64)
    print(f"MAGNETO — Cyclic spline detrending ({target})")
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

    print("\n[3/3] Fitting cyclic spline model per cell …")
    cells = df.groupby(["lat_id", "lon_id"])
    all_r = []
    for _, df_cell in tqdm(cells, total=cells.ngroups, desc="cyclic_spline cells"):
        r = fit_cell_cyclic_spline(df_cell, target=target)
        if len(r) > 0:
            all_r.append(r)

    df_out = pd.concat(all_r, ignore_index=True)
    n_ok = (df_out["fit_status"] == "ok").sum()
    print(f"\n  Cyclic spline residuals: {n_ok:,}/{len(df_out):,} fits ok")

    # join temperature metadata
    temp_cols = [c for c in [ERA5_TEMP_COL, "temp_bin_id", "temp_bin_label"] if c in df.columns]
    if temp_cols:
        df_temp = df[["date", "lat_id", "lon_id"] + temp_cols].copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out = df_out.merge(df_temp, on=["date", "lat_id", "lon_id"], how="left")

    member_cols = [c for c in members.columns if c not in ["lat_id", "lon_id"]]
    df_out = df_out.merge(members, on=["lat_id", "lon_id"], how="left")

    atomic_write(df_out, output_path)
    print(f"  Written: {output_path}")
    print(f"\n[OK] 05_cyclic_spline complete — {len(df_out):,} rows")


if __name__ == "__main__":
    main()
