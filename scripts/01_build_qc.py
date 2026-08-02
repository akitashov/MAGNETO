#!/usr/bin/env python3
"""
01_build_qc.py — MAGNETO QC Dataset Builder.

Builds the observation-level QC table used by downstream detrending and
inference. Keeps all valid land observations; no dependent-variable truncation
is applied in the primary analysis.
"""
from __future__ import annotations
import sys, argparse
from datetime import datetime
from _Common import *
import numpy as np, pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-surrogates", type=int, default=None)
    parser.add_argument(
        "--allow-modis-rematch",
        action="store_true",
        help="Allow the old independent nearest-date MODIS join when SIF lacks embedded MODIS variables",
    )
    return parser.parse_args(argv)


def apply_analysis_area_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply analysis-area filter to the already land-only SIF dataset.

    The upstream OCO-2 ETL removed water retrievals before aggregation; this
    function only excludes POLAR retrievals (bit 4) from the analysis area.
    """
    rf = df["region_flags"].values.astype(np.int64)
    is_land = ((rf & 16) != 0) | ((rf & 32) != 0)
    is_polar = (rf & 4) != 0
    return df.loc[is_land & ~is_polar].copy()


def build_modis_date_map(modis_dates, all_ords):
    """Map every ordinal to the nearest MODIS 8-day period date."""
    modis_ord = np.array([d.toordinal() for d in modis_dates], dtype=np.int64)
    idx = np.searchsorted(modis_ord, all_ords, side="left")
    idx = np.clip(idx, 1, len(modis_ord) - 1)
    left = modis_ord[idx - 1]
    right = modis_ord[idx]
    nearest = np.where(np.abs(all_ords - left) <= np.abs(all_ords - right), left, right)
    return {o: pd.Timestamp(datetime.fromordinal(int(n))) for o, n in zip(all_ords, nearest)}


def main() -> int:
    args = parse_args()
    smoke = args.smoke_test

    print("=" * 64)
    print("MAGNETO — Build QC Dataset" + (" [SMOKE]" if smoke else ""))
    print("=" * 64)
    setup_dirs()

    # ── Load SIF ──────────────────────────────────────────────────────
    print("\n[1/6] Loading SIF …")
    sif = pd.read_feather(FILE_SIF)
    n_raw = len(sif)
    n_cells_raw = sif[["lat_id", "lon_id"]].drop_duplicates().shape[0]
    print(f"  Raw SIF: {n_raw:,} rows, {n_cells_raw:,} cells")

    # Embedded MODIS variables from the upstream SIF ETL prevent an independent,
    # potentially inconsistent nearest-date MODIS rematch in QC.
    embedded_modis = {"lai", "cloud_fraction", "aerosol_fraction"}.issubset(sif.columns)
    has_source_keys = {"source_modis_date", "source_modis_lat_id", "source_modis_lon_id"}.issubset(sif.columns)
    if embedded_modis:
        print("  Using embedded MODIS variables from SIF (lai, cloud_fraction, aerosol_fraction).")
    else:
        print("  [WARN] SIF does not contain embedded MODIS variables; QC will fall back to a MODIS join.")

    if has_source_keys:
        print("  Validating SIF source keys against MODIS candidate …")
        modis_keys = pd.read_parquet(
            FILE_MODIS, engine=PARQUET_ENGINE, columns=["date", "lat_id", "lon_id"]
        )
        modis_keys["date"] = pd.to_datetime(modis_keys["date"]).dt.normalize()
        modis_key_set = set(
            zip(
                modis_keys["date"].apply(lambda d: d.toordinal()),
                modis_keys["lat_id"].astype(int),
                modis_keys["lon_id"].astype(int),
            )
        )
        del modis_keys

        sif_source_ords = sif["source_modis_date"].apply(lambda d: d.toordinal())
        sif_key_set = set(
            zip(
                sif_source_ords.astype(int),
                sif["source_modis_lat_id"].astype(int),
                sif["source_modis_lon_id"].astype(int),
            )
        )
        missing_keys = sif_key_set - modis_key_set
        if missing_keys:
            raise RuntimeError(
                f"{len(missing_keys):,} SIF source keys are missing from MODIS candidate. "
                "Example: " + str(next(iter(missing_keys)))
            )
        print(f"  All {len(sif_key_set):,} SIF source keys exist in MODIS candidate.")

    # ── Analysis-area filter ──────────────────────────────────────────
    print("\n[2/6] Applying analysis-area filter …")
    print(f"  Input SIF is already land-only from upstream ETL.")
    print(f"  Keep:  (region_flags & 16) | (region_flags & 32)  → land retrievals")
    print(f"  Exclude: POLAR (bit 4)")
    sif_land = apply_analysis_area_filter(sif)
    n_land = len(sif_land)
    print(f"  After polar exclusion: {n_land:,} rows (removed {n_raw - n_land:,})")

    # ── Smoke subsetting is performed after control membership is assigned
    # in 02b_smoke_subset.py so that barren/vegetated/geographic controls are
    # guaranteed to be represented.
    if smoke:
        print("\n[SMOKE] Smoke subsetting deferred to stage 02b (controls-aware).")

    if embedded_modis:
        # ── Use embedded MODIS variables ────────────────────────────────
        print("\n[3/6] Using embedded MODIS variables; skipping separate MODIS load/join …")
        merged = sif_land.copy()
        n_merged = len(merged)
        print(f"  Embedded MODIS coverage: {n_merged:,} rows ({100*n_merged/n_raw:.1f}% of raw SIF)")
    else:
        if not args.allow_modis_rematch:
            raise RuntimeError(
                "SIF input does not contain embedded MODIS variables. "
                "Re-run the upstream SIF ETL or pass --allow-modis-rematch to use the old independent join."
            )

        # ── Load MODIS ────────────────────────────────────────────────────
        print("\n[3/6] Loading MODIS …")
        modis = pd.read_parquet(FILE_MODIS, engine=PARQUET_ENGINE)
        n_modis = len(modis)
        print(f"  MODIS: {n_modis:,} rows")

        # ── Build date snap map ───────────────────────────────────────────
        print("\n[4/6] Building MODIS 8-day snap map …")
        modis_dates = sorted(modis["date"].unique())
        min_ord = modis_dates[0].toordinal()
        max_ord = modis_dates[-1].toordinal()
        all_ords = np.arange(min_ord, max_ord + 1, dtype=np.int64)
        print(f"  MODIS date range: {modis_dates[0].date()} → {modis_dates[-1].date()}")
        date_map = build_modis_date_map(modis_dates, all_ords)

        sif_land["date_ord"] = sif_land["date"].map(lambda d: d.toordinal())
        sif_land["date_snapped"] = sif_land["date_ord"].map(date_map)
        n_snapped = sif_land["date_snapped"].notna().sum()
        print(f"  SIF snapped: {n_snapped:,}/{n_land:,}")

        # ── Join SIF ↔ MODIS ─────────────────────────────────────────────
        print("\n[5/6] Preparing MODIS for join …")
        modis_cols = ["date", "lat_id", "lon_id", "lai", "cloud_fraction", "aerosol_fraction", "quality_flag"]
        modis_j = modis[modis_cols].rename(columns={"date": "date_snapped"})
        sif_cells = sif_land[["lat_id", "lon_id"]].drop_duplicates()
        modis_j = modis_j.merge(sif_cells, on=["lat_id", "lon_id"], how="inner")
        print(f"  MODIS rows after SIF-cell filter: {len(modis_j):,}")

        print("\n[6/6] Joining SIF ↔ MODIS …")
        sif_j = sif_land.drop(columns=["date_ord"])
        merged = sif_j.merge(modis_j, on=["date_snapped", "lat_id", "lon_id"], how="inner")
        n_merged = len(merged)
        print(f"  Merged: {n_merged:,} rows ({100*n_merged/n_raw:.1f}% of raw SIF)")

    merged = merged.rename(columns={"latitude": "lat", "longitude": "lon"})
    merged["year"] = merged["date"].dt.year.astype("int32")
    merged["doy"] = merged["date"].dt.dayofyear.astype("int32")

    # ── QC Flow ───────────────────────────────────────────────────────
    print("\n[7/7] Applying QC filters …")
    qc = merged.copy()
    steps = []

    def _rec(desc, df):
        steps.append({"step": desc, "n_rows": len(df),
                      "n_cells": df[["lat_id", "lon_id"]].drop_duplicates().shape[0]})
        print(f"  {desc:45s} {len(df):>10,} rows")

    _rec("1. Merged SIF×MODIS", qc)

    qc = qc[qc["cloud_fraction"].notna() & (qc["cloud_fraction"] <= QC_CLOUD_MAX)]
    _rec(f"2. cloud_fraction ≤ {QC_CLOUD_MAX}", qc)

    qc = qc[qc["aerosol_fraction"].notna() & (qc["aerosol_fraction"] <= QC_AEROSOL_MAX)]
    _rec(f"3. aerosol_fraction ≤ {QC_AEROSOL_MAX}", qc)

    qc = qc[(qc["lat"] >= QC_LAT_MIN) & (qc["lat"] <= QC_LAT_MAX)]
    _rec(f"4. lat ∈ [{QC_LAT_MIN}, {QC_LAT_MAX}]", qc)

    # SIF dependent variable is NOT truncated in the primary analysis.
    # The old sif_771nm >= 0.001 threshold is available as a sensitivity flag only.

    # Flow table
    qc_flow = pd.DataFrame(steps)
    qc_flow["removed"] = qc_flow["n_rows"].diff().abs().fillna(0).astype(int)
    qc_flow["pct_remaining"] = (100.0 * qc_flow["n_rows"] / qc_flow["n_rows"].iloc[0]).round(2)
    atomic_write(qc_flow, FILE_QC_FLOW)
    print(f"\n  QC flow table → {FILE_QC_FLOW}")

    out_cols = ["date", "year", "doy", "lat_id", "lon_id", "lat", "lon",
                "sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index",
                "lai", "cloud_fraction", "aerosol_fraction",
                "region_flags", "quality_flag"]
    if has_source_keys:
        out_cols.extend([
            "source_modis_date", "source_modis_lat_id", "source_modis_lon_id",
            "modis_match_mode", "temporal_offset_days",
            "spatial_offset_lat_id", "spatial_offset_lon_id", "spatial_distance",
        ])
    qc = qc[[c for c in out_cols if c in qc.columns]].copy()

    atomic_write(qc, FILE_QC)
    n_cells_final = qc[["lat_id", "lon_id"]].drop_duplicates().shape[0]

    print(f"\n{'='*64}")
    print(f"QC SUMMARY")
    print(f"  Raw SIF:              {n_raw:>10,} rows, {n_cells_raw:>6,} cells")
    print(f"  After land mask:      {n_land:>10,} rows")
    print(f"  After SIF×MODIS join: {n_merged:>10,} rows")
    print(f"  After QC filters:     {len(qc):>10,} rows, {n_cells_final:>6,} cells")
    print(f"  Date range:           {qc['date'].min().date()} → {qc['date'].max().date()}")
    print(f"  Output:               {FILE_QC}")
    print(f"[OK] 01_build_qc complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
