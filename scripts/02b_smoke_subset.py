#!/usr/bin/env python3
"""
02b_smoke_subset.py — Smoke-test subsetting after controls are defined.

When --smoke-test is passed, this stage rewrites the QC dataset to a small,
structurally representative subset with explicit minimum quotas for:
  - strict low-LAI functional controls
  - persistently vegetated reference
  - Sahara geographic control
  - each LAI quartile
  - each temperature class
  - additional high-observation land cells

The full QC is retained only when smoke mode is off.
"""
from __future__ import annotations
import sys, argparse
import numpy as np
import pandas as pd
from _Common import *


# Minimum quotas for a structurally representative smoke subset.
_QUOTAS = {
    "is_strict_low_lai": 10,
    "is_vegetated": 50,
    "is_Sahara": 50,
    "lai_quartile": 20,  # per quartile
    "temp_class": 20,    # per class
}


def _cell_temp_classes(qc: pd.DataFrame) -> pd.DataFrame:
    """Return a long table of (lat_id, lon_id, temp_bin_label) for cells."""
    if "temp_bin_label" not in qc.columns:
        return pd.DataFrame(columns=["lat_id", "lon_id", "temp_bin_label"])
    return qc[["lat_id", "lon_id", "temp_bin_label"]].dropna().drop_duplicates()


def smoke_subset_qc(qc: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a deterministic, representative smoke subset of QC rows plus diagnostics."""
    qc = qc.merge(members, on=["lat_id", "lon_id"], how="left")
    qc["year"] = qc["date"].dt.year
    qc = qc[qc["year"].isin(SMOKE_YEARS)].copy()

    # Per-cell summary for quota selection
    cell_summary = qc.groupby(["lat_id", "lon_id"], as_index=False).agg(
        n_obs=("date", "size"),
    )
    cell_summary = cell_summary.merge(members, on=["lat_id", "lon_id"], how="left")
    cell_summary["lai_quartile"] = cell_summary["lai_quartile"].fillna(-1).astype(int)

    selected = set()
    diag = {"total_cells_requested": SMOKE_CELL_SAMPLE_SIZE}

    # 1. Required functional/geographic controls
    for flag in ["is_strict_low_lai", "is_vegetated", "is_Sahara"]:
        if flag not in cell_summary.columns:
            diag[flag] = 0
            continue
        sub = cell_summary[cell_summary[flag] == True].sort_values(["lat_id", "lon_id"])
        quota = _QUOTAS[flag]
        pick = sub.head(quota)
        selected.update((int(r.lat_id), int(r.lon_id)) for r in pick.itertuples())
        diag[flag] = len(pick)
        print(f"  [SMOKE] {flag}: selected {len(pick)} cells ({sub.shape[0]} available)")

    # 2. LAI quartile coverage
    diag["lai_quartile"] = {}
    for q in sorted(cell_summary["lai_quartile"].unique()):
        if q < 1:
            continue
        sub = cell_summary[cell_summary["lai_quartile"] == q].sort_values(["lat_id", "lon_id"])
        already = set((int(r.lat_id), int(r.lon_id)) for r in sub.itertuples()) & selected
        pick = sub.head(_QUOTAS["lai_quartile"] + len(already))
        new = [(int(r.lat_id), int(r.lon_id)) for r in pick.itertuples()
               if (int(r.lat_id), int(r.lon_id)) not in selected]
        # take only the new ones up to quota
        new = new[:_QUOTAS["lai_quartile"]]
        selected.update(new)
        diag["lai_quartile"][f"Q{q}"] = len(new)
        print(f"  [SMOKE] LAI Q{q}: added {len(new)} new cells")

    # 3. Temperature class coverage
    diag["temp_class"] = {}
    cell_temps = _cell_temp_classes(qc)
    for label in TEMP_LABELS:
        cells_in_class = cell_temps[cell_temps["temp_bin_label"] == label][["lat_id", "lon_id"]].drop_duplicates()
        cells_in_class = cells_in_class.merge(cell_summary[["lat_id", "lon_id", "n_obs"]], on=["lat_id", "lon_id"], how="left")
        cells_in_class = cells_in_class.sort_values(["lat_id", "lon_id"])
        already = set((int(r.lat_id), int(r.lon_id)) for r in cells_in_class.itertuples()) & selected
        pick = cells_in_class.head(_QUOTAS["temp_class"] + len(already))
        new = [(int(r.lat_id), int(r.lon_id)) for r in pick.itertuples()
               if (int(r.lat_id), int(r.lon_id)) not in selected]
        new = new[:_QUOTAS["temp_class"]]
        selected.update(new)
        diag["temp_class"][label] = len(new)
        print(f"  [SMOKE] Temp {label}: added {len(new)} new cells")

    # 4. Fill remaining quota with highest-observation cells
    remaining = cell_summary[~cell_summary.apply(
        lambda r: (int(r.lat_id), int(r.lon_id)) in selected, axis=1
    )].sort_values("n_obs", ascending=False)
    n_extra = max(0, SMOKE_CELL_SAMPLE_SIZE - len(selected))
    if n_extra > 0 and len(remaining) > 0:
        extra = remaining.head(n_extra)
        selected.update((int(r.lat_id), int(r.lon_id)) for r in extra.itertuples())
        diag["extra_high_obs"] = len(extra)
        print(f"  [SMOKE] extra high-obs cells: {len(extra)}")
    else:
        diag["extra_high_obs"] = 0

    selected_df = pd.DataFrame(list(selected), columns=["lat_id", "lon_id"])
    out = qc.merge(selected_df, on=["lat_id", "lon_id"], how="inner").copy()
    out = out.drop(columns=[c for c in ["year"] if c in out.columns])

    n_cells = out[["lat_id", "lon_id"]].drop_duplicates().shape[0]
    diag["final_cells"] = n_cells
    diag["final_observations"] = len(out)
    return out, diag


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-surrogates", type=int, default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.smoke_test:
        print("[SKIP] 02b_smoke_subset: not in smoke mode")
        return 0

    print("=" * 64)
    print("MAGNETO — Smoke Subset (controls-aware)")
    print("=" * 64)

    print("\n[1/3] Loading QC …")
    qc = pd.read_parquet(FILE_QC, engine=PARQUET_ENGINE)
    qc["date"] = pd.to_datetime(qc["date"])
    print(f"  QC rows: {len(qc):,}, cells: {qc[['lat_id', 'lon_id']].drop_duplicates().shape[0]:,}")

    print("\n[2/3] Loading membership …")
    members = pd.read_parquet(FILE_LAI_MEMBERS, engine=PARQUET_ENGINE)
    print(f"  Membership rows: {len(members):,}")

    print("\n[3/3] Building smoke subset …")
    sub, diag = smoke_subset_qc(qc, members)
    n_cells = sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]
    print(f"  Smoke subset: {len(sub):,} rows, {n_cells:,} cells")
    print(f"  Date range: {sub['date'].min().date()} → {sub['date'].max().date()}")

    # Remove membership columns that will be re-joined downstream; keep
    # observation keys and QC columns (year/doy are needed by detrending).
    keep_cols = ["date", "year", "doy", "lat_id", "lon_id", "lat", "lon",
                 "sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index",
                 "lai", "cloud_fraction", "aerosol_fraction",
                 "region_flags", "quality_flag"]
    # Preserve temperature columns if already joined (smoke_subset runs after era5_join).
    for tc in [ERA5_TEMP_COL, "temp_bin_id", "temp_bin_label"]:
        if tc in sub.columns and tc not in keep_cols:
            keep_cols.append(tc)
    drop_cols = [c for c in members.columns if c not in ["lat_id", "lon_id"]]
    sub = sub.drop(columns=[c for c in drop_cols if c in sub.columns])
    # Ensure year/doy exist
    if "year" not in sub.columns:
        sub["year"] = sub["date"].dt.year.astype("int32")
    if "doy" not in sub.columns:
        sub["doy"] = sub["date"].dt.dayofyear.astype("int32")

    atomic_write(sub, FILE_QC)
    print(f"  Overwritten: {FILE_QC}")

    # Write smoke-subset diagnostics (flatten nested dicts to a single row).
    flat_diag = {
        "total_cells_requested": SMOKE_CELL_SAMPLE_SIZE,
        "is_strict_low_lai": diag.get("is_strict_low_lai", 0),
        "is_vegetated": diag.get("is_vegetated", 0),
        "is_Sahara": diag.get("is_Sahara", 0),
        "extra_high_obs": diag.get("extra_high_obs", 0),
        "final_cells": diag["final_cells"],
        "final_observations": diag["final_observations"],
        "run_id": current_run_id(),
        "run_type": current_run_type(),
    }
    for k, v in diag.get("lai_quartile", {}).items():
        flat_diag[f"lai_quartile_{k}"] = v
    for k, v in diag.get("temp_class", {}).items():
        flat_diag[f"temp_class_{k}"] = v
    atomic_write([flat_diag], FILE_SMOKE_SUBSET_DIAGNOSTICS)
    print(f"  Diagnostics: {FILE_SMOKE_SUBSET_DIAGNOSTICS}")

    print("\n[OK] 02b_smoke_subset complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
