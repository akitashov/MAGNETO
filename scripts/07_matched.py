#!/usr/bin/env python3
"""
07_matched.py — Build strict pairwise matched sample (harmonic ∩ cyclic_spline).

All intersections are INNER: only observations present in both residual datasets
with valid fits are retained. Grouping metadata (LAI quartile, controls,
temperature) is carried through and verified to be identical across methods.
"""
from __future__ import annotations
import sys, hashlib, numpy as np, pandas as pd
from _Common import *


# Method-specific diagnostics that must NOT be carried into the matched sample.
_DIAGNOSTIC_COLS = {
    "method", "n_train_obs", "n_train_years", "r_squared", "residual_sd",
    "condition_number", "intercept", "trend_coef", "sin_coef", "cos_coef",
    "design_rank", "pass_flag", "fail_reason", "fit_status", "spline_df",
    "prediction_range", "seasonal_coverage",
}


def load_valid(method: str) -> pd.DataFrame:
    """Load valid residuals for a given method, keeping only keys + metadata + residuals."""
    path_map = {
        "harmonic": FILE_RES_HARMONIC,
        "cyclic_spline": FILE_RES_CYCLIC,
    }
    path = path_map[method]
    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    if "fit_status" in df.columns:
        df = df[df["fit_status"] == "ok"].copy()
    elif "pass_flag" in df.columns:
        df = df[df["pass_flag"] == True].copy()

    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year.astype("int32")

    df = df.rename(columns={
        "residual": f"residual_{method}",
        "observed_sif": f"observed_sif_{method}",
        "expected_sif": f"expected_sif_{method}",
    })

    # Drop diagnostic columns; retain observation keys, year/doy, temperature,
    # LAI quartile, control memberships, and method-specific residual columns.
    drop_cols = [c for c in df.columns if c in _DIAGNOSTIC_COLS]
    df = df.drop(columns=drop_cols)
    return df


def _validate_metadata_equality(df: pd.DataFrame) -> pd.DataFrame:
    """Check that metadata columns merged from both methods are identical."""
    cols = list(df.columns)
    suffix_pairs = {}
    for c in cols:
        if c.endswith("_x"):
            base = c[:-2]
            partner = base + "_y"
            if partner in cols:
                suffix_pairs[base] = (c, partner)

    mismatches = []
    for base, (cx, cy) in suffix_pairs.items():
        if df[cx].dtype.kind in "iufc" and df[cy].dtype.kind in "iufc":
            diff = ~np.isclose(df[cx].astype(float), df[cy].astype(float), equal_nan=True)
        else:
            diff = df[cx].astype(str) != df[cy].astype(str)
        n_diff = int(diff.sum())
        if n_diff > 0:
            mismatches.append(f"{base}: {n_diff} mismatches")
        # keep the _x copy and drop _y
        df = df.drop(columns=[cy]).rename(columns={cx: base})

    if mismatches:
        diag_path = RESULTS / "matched_metadata_mismatches.txt"
        diag_path.write_text("\n".join(mismatches))
        raise ValueError(
            f"Matched metadata mismatch for {len(mismatches)} columns: {mismatches[:5]}. "
            f"Details: {diag_path}"
        )
    return df


def main() -> int:
    print("=" * 64)
    print("MAGNETO — Matched Harmonic × Cyclic-Spline Sample")
    print("=" * 64)
    setup_dirs()

    print("\n[1/3] Loading method residuals …")
    methods = {}
    for m in RESIDUAL_METHODS:
        df = load_valid(m)
        print(f"  {m}: {len(df):,} valid observations")
        methods[m] = df

    print("\n[2/3] Building strict pairwise intersection …")
    keys_h = set(zip(methods["harmonic"]["date"], methods["harmonic"]["lat_id"],
                     methods["harmonic"]["lon_id"], methods["harmonic"]["year"]))
    keys_c = set(zip(methods["cyclic_spline"]["date"], methods["cyclic_spline"]["lat_id"],
                     methods["cyclic_spline"]["lon_id"], methods["cyclic_spline"]["year"]))
    common_keys = keys_h & keys_c
    print(f"  Common observation keys: {len(common_keys):,}")

    mk = pd.DataFrame(list(common_keys), columns=["date", "lat_id", "lon_id", "year"])
    mk["date"] = pd.to_datetime(mk["date"])
    mk["lat_id"] = mk["lat_id"].astype(int)
    mk["lon_id"] = mk["lon_id"].astype(int)
    mk["year"] = mk["year"].astype(int)

    for m in RESIDUAL_METHODS:
        methods[m]["date"] = pd.to_datetime(methods[m]["date"])
        methods[m]["lat_id"] = methods[m]["lat_id"].astype(int)
        methods[m]["lon_id"] = methods[m]["lon_id"].astype(int)
        methods[m]["year"] = methods[m]["year"].astype(int)
        mk = mk.merge(methods[m], on=["date", "lat_id", "lon_id", "year"], how="inner")

    print("\n[3/3] Validating matched metadata …")
    mk = _validate_metadata_equality(mk)

    # Single canonical metadata columns remain; no _x/_y suffixes.
    mk["obs_hash"] = mk.apply(
        lambda r: hashlib.sha256(f"{r['date'].date()}_{r['lat_id']}_{r['lon_id']}".encode()).hexdigest(),
        axis=1,
    )

    print(f"\n  Matched sample: {len(mk):,} observations, "
          f"{mk[['lat_id','lon_id']].drop_duplicates().shape[0]:,} cells, "
          f"{mk['year'].nunique()} years")

    atomic_write(mk, FILE_MATCHED_HC)
    print(f"  Written: {FILE_MATCHED_HC}")

    print("\n[OK] 07_matched complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
