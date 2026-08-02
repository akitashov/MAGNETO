#!/usr/bin/env python3
"""E01_modis_etl.py — MODIS LAI/FPAR upstream ETL.

Reads the University of Hamburg ICDC 0.5° global MOD15A2H Collection 6.1
NetCDF files and conservatively regrids LAI and diagnostic variables onto
the canonical 0.5° target grid used by OCO-2 SIF and ERA5 (cell centres at
integer and half-integer degrees).

The ICDC source grid has centres offset by 0.25° from the target grid
(e.g. source centres at .25/.75, target centres at .0/.5).  A source cell
therefore overlaps exactly four target cells, each receiving 25% of the
source area.  Continuous variables are area-weighted across the four
overlapping targets; ``quality_flag`` is assigned deterministically from
the north-east overlapping source cell and is retained only as a diagnostic.

Output is ``data/interim/modis_extract_candidate.parquet``; promotion to the
canonical ``modis_extract.parquet`` occurs only if the candidate passes the
ETL equivalence audit.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (  # noqa: E402
    PARQUET_ENGINE,
    PROJECT_ROOT,
    atomic_write,
    setup_dirs,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml

_PIPELINE_CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
_PIPELINE_CFG = yaml.safe_load(_PIPELINE_CFG_PATH.read_text(encoding="utf-8"))
_MCFG = _PIPELINE_CFG.get("modis", {})

INPUT_DIR = PROJECT_ROOT / _MCFG.get("input_dir", "data/raw/MODIS")
FILE_PATTERN = _MCFG.get("file_pattern", "*.nc")
TARGET_RESOLUTION = float(_MCFG.get("target_resolution", 0.5))
PERIOD_DAYS = int(_MCFG.get("period_days", 8))
LAST_START_DOY = int(_MCFG.get("last_start_doy", 361))

VAR_MAPPING = {
    "lai": "lai",
    "primary_qualityflag": "quality_flag",
    "cloudfraction": "cloud_fraction",
    "aerosolfraction": "aerosol_fraction",
}

OUTPUT_PATH = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_modis"]
CANDIDATE_PATH = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_candidate.parquet")

import _etl_promote as _promote


VALUE_COLS = ["lai", "quality_flag", "cloud_fraction", "aerosol_fraction"]
CONTINUOUS_COLS = ["lai", "cloud_fraction", "aerosol_fraction"]


def _snap_modis_date_to_period_start(
    dates: pd.Series,
    period_days: int,
    last_start_doy: int,
) -> pd.Series:
    """Snap timestamps to the start of the MODIS compositing period."""
    dt = pd.to_datetime(dates, errors="coerce")
    years = dt.dt.year
    doys = dt.dt.dayofyear
    snapped = ((doys - 1) // period_days) * period_days + 1
    snapped = np.minimum(snapped, last_start_doy)
    return pd.to_datetime(
        years.astype(str) + snapped.astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce",
    )


def _normalize_percent(series: pd.Series, name: str) -> pd.Series:
    """Normalize percentage-like variables to a 0–100 scale if needed."""
    mx = series.max()
    if pd.isna(mx):
        return series
    if mx <= 1.5:
        print(f"[INFO] {name} normalized: 0–1 → 0–100")
        return series * 100.0
    if mx <= 100.0:
        return series
    print(f"[WARN] {name} unexpected range: min={series.min()}, max={mx}")
    return series


def _conservative_regrid(df: pd.DataFrame, res: float) -> pd.DataFrame:
    """Explode source 0.5° cells onto the four overlapping target 0.5° cells.

    Source grid (ICDC): centres at n*res + res/2, bounds [c-res/2, c+res/2].
    Target grid (OCO-2/ERA5): centres at n*res, bounds [c-res/2, c+res/2].
    Because the two grids are offset by exactly res/2, each source cell
    overlaps four target cells with equal area (res*res/4, i.e. weight 0.25).
    """
    if df.empty:
        return df

    lat_s = df["latitude"].astype("float64").to_numpy()
    lon_s = df["longitude"].astype("float64").to_numpy()

    # Lower-left target index for the source cell.
    n_lat = np.floor(lat_s / res).astype("int64")
    n_lon = np.floor(lon_s / res).astype("int64")

    # Quality flag comes from the deterministic north-east overlapping source
    # cell, whose centre is floor(target/res)*res + res/2.
    # For a target at (n_lat+d_lat)*res, that source centre is (n_lat+d_lat)*res + res/2.
    lat_orig = df["latitude"].to_numpy()
    lon_orig = df["longitude"].to_numpy()

    fragments = []
    for d_lat, d_lon in ((0, 0), (0, 1), (1, 0), (1, 1)):
        frag = df.copy()
        target_lat = (n_lat + d_lat) * res
        target_lon = (n_lon + d_lon) * res
        frag["latitude"] = target_lat.astype("float32")
        frag["longitude"] = target_lon.astype("float32")

        qf_source_lat = target_lat + res / 2
        qf_source_lon = target_lon + res / 2
        is_qf_source = (
            np.isclose(lat_orig, qf_source_lat, atol=res * 1e-3)
            & np.isclose(lon_orig, qf_source_lon, atol=res * 1e-3)
        )
        if "quality_flag" in frag.columns:
            frag["quality_flag"] = np.where(
                is_qf_source, frag["quality_flag"], np.nan
            )
        fragments.append(frag)

    return pd.concat(fragments, ignore_index=True)


def process_modis_file(file_path: Path) -> pd.DataFrame:
    """Process a single MODIS NetCDF file into the standardized target grid."""
    with xr.open_dataset(file_path) as ds:
        missing = [v for v in VAR_MAPPING if v not in ds.variables and v not in ds.data_vars]
        if missing:
            raise RuntimeError(f"Missing variables in {file_path.name}: {missing}")

        subset = ds[list(VAR_MAPPING.keys())]
        df = subset.to_dataframe().reset_index()

    rename_dict = dict(VAR_MAPPING)
    if "lat" in df.columns:
        rename_dict["lat"] = "latitude"
    if "lon" in df.columns:
        rename_dict["lon"] = "longitude"
    if "time" in df.columns:
        rename_dict["time"] = "date"
    df = df.rename(columns=rename_dict)

    if not {"latitude", "longitude", "date"}.issubset(df.columns):
        raise RuntimeError(f"Required coordinates missing in {file_path.name}")

    res = float(TARGET_RESOLUTION)

    # Normalize percentage variables before regridding so weights are applied
    # to the correct scale.
    for src_name, dst_name in [("cloudfraction", "cloud_fraction"),
                               ("aerosolfraction", "aerosol_fraction")]:
        if dst_name in df.columns:
            df[dst_name] = _normalize_percent(df[dst_name], dst_name)

    # Conservative regrid from ICDC offset grid to canonical target grid.
    df = _conservative_regrid(df, res)
    if df.empty:
        return df

    df["date"] = _snap_modis_date_to_period_start(
        df["date"], period_days=PERIOD_DAYS, last_start_doy=LAST_START_DOY
    )
    df = df.dropna(subset=["date"])

    # Aggregate to (date, target_lat, target_lon).  Mean over the four
    # overlapping source cells implements the equal-area weighted average;
    # NaN source values are ignored automatically.  quality_flag is already
    # NaN except for the deterministic source cell, so mean picks it up.
    df = df.groupby(["date", "latitude", "longitude"], as_index=False).mean(numeric_only=True)

    df["lat_id"] = (df["latitude"] * 100).round().astype("int16")
    df["lon_id"] = (df["longitude"] * 100).round().astype("int16")

    for col in VALUE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Drop rows where all continuous variables are missing (open ocean / inland
    # water).  quality_flag alone is not sufficient to keep a row.
    df = df.dropna(subset=CONTINUOUS_COLS, how="all").reset_index(drop=True)

    return df


def _expected_period_starts(years: set[int], period_days: int, last_start_doy: int) -> set[pd.Timestamp]:
    """Return the canonical 8-day period-start dates for the given years."""
    expected: set[pd.Timestamp] = set()
    starts = list(range(1, last_start_doy + 1, period_days))
    for year in years:
        for doy in starts:
            expected.add(pd.Timestamp(pd.to_datetime(f"{year}{str(doy).zfill(3)}", format="%Y%j")))
    return expected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MAGNETO MODIS ETL")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing candidate output before writing (does not touch canonical)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Process all files and report errors in the manifest instead of failing fast",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("MAGNETO — MODIS ETL (conservative regrid to 0.5° target)")
    print("=" * 64)
    setup_dirs()

    if args.clean:
        if CANDIDATE_PATH.exists():
            CANDIDATE_PATH.unlink()
            print(f"[INFO] Removed existing candidate: {CANDIDATE_PATH}")

    files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No MODIS files matching {FILE_PATTERN} in {INPUT_DIR}")

    print(f"[INFO] Found {len(files):,} MODIS files in {INPUT_DIR}")
    print(f"[INFO] Target grid:     {TARGET_RESOLUTION}°")
    print(f"[INFO] Canonical:       {OUTPUT_PATH}")
    print(f"[INFO] Candidate:       {CANDIDATE_PATH}")

    total_rows = 0
    out_cols = ["date", "latitude", "longitude", "lat_id", "lon_id"] + VALUE_COLS

    _append_engine = "fastparquet"

    tmp_path = CANDIDATE_PATH.with_suffix(CANDIDATE_PATH.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    manifest = {
        "expected_composites": [],
        "successful_files": [],
        "empty_files": [],
        "erroneous_files": [],
    }
    observed_dates: set[pd.Timestamp] = set()
    observed_years: set[int] = set()
    fail_fast = not args.continue_on_error

    for i, file_path in enumerate(tqdm(files, unit="file", desc="MODIS ETL")):
        try:
            df = process_modis_file(file_path)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] {file_path.name}: {msg}")
            manifest["erroneous_files"].append({"file": str(file_path), "error": msg})
            if fail_fast:
                break
            continue

        if df.empty:
            print(f"[WARN] {file_path.name}: no rows after processing")
            manifest["empty_files"].append(str(file_path))
            continue

        # Record observed composites for manifest and gap detection.
        file_dates = pd.to_datetime(df["date"].dropna().unique())
        observed_dates.update(file_dates)
        observed_years.update(pd.to_datetime(file_dates).year.tolist())

        df = df[[c for c in out_cols if c in df.columns]]
        total_rows += len(df)

        if not tmp_path.exists():
            df.to_parquet(tmp_path, engine=_append_engine, index=False)
        else:
            df.to_parquet(tmp_path, engine=_append_engine, index=False, append=True)

        manifest["successful_files"].append({
            "file": str(file_path),
            "composite_dates": [str(d) for d in file_dates],
            "rows": int(len(df)),
        })

        del df
        if i % 10 == 0:
            gc.collect()

    # Fail fast if any file failed and we are not continuing.
    if manifest["erroneous_files"] and fail_fast:
        _write_manifest(manifest, observed_years, observed_dates)
        raise RuntimeError(
            f"{len(manifest['erroneous_files'])} MODIS file(s) failed processing. "
            "Use --continue-on-error to process remaining files and produce a full error manifest."
        )

    if not tmp_path.exists():
        raise RuntimeError("No MODIS rows were processed.")

    # Move the validated incremental file to the candidate path without loading
    # the full dataset back into memory.
    os.replace(str(tmp_path), str(CANDIDATE_PATH))

    _write_manifest(manifest, observed_years, observed_dates)

    spec = {
        "name": "modis_extract",
        "fmt": "parquet",
        "required": ["date", "lat_id", "lon_id", "lai", "cloud_fraction", "aerosol_fraction"],
        "key_cols": ["date", "lat_id", "lon_id"],
        "metric_cols": ["lai", "cloud_fraction", "aerosol_fraction", "quality_flag"],
        "grid_cols": ["lat_id", "lon_id"],
    }
    report = _promote.compare_and_promote(OUTPUT_PATH, CANDIDATE_PATH, spec)
    if not report["passed"]:
        print("[STOP] MODIS candidate differs from canonical. Promotion blocked; operator review required.")
        return 2

    print(f"\n{'='*64}")
    print(f"MODIS ETL SUMMARY")
    print(f"  Files found:     {len(files):,}")
    print(f"  Successful:      {len(manifest['successful_files']):,}")
    print(f"  Empty:           {len(manifest['empty_files']):,}")
    print(f"  Erroneous:       {len(manifest['erroneous_files']):,}")
    print(f"  Missing periods: {len(manifest.get('missing_composites', [])):,}")
    print(f"  Total rows:      {total_rows:,}")
    print(f"  Output:          {OUTPUT_PATH}")
    print(f"[OK] E01_modis_etl complete")
    return 0


def _write_manifest(
    manifest: dict,
    observed_years: set[int],
    observed_dates: set[pd.Timestamp],
) -> None:
    """Write the MODIS ETL manifest and update missing-composite list."""
    if observed_years:
        expected = _expected_period_starts(observed_years, PERIOD_DAYS, LAST_START_DOY)
        missing = sorted(expected - observed_dates)
        manifest["expected_composites"] = sorted(str(d) for d in expected)
        manifest["missing_composites"] = [str(d) for d in missing]
    else:
        manifest["expected_composites"] = []
        manifest["missing_composites"] = []

    manifest_dir = PROJECT_ROOT / "results" / "etl_promotion_audits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "modis_etl_manifest.json"
    atomic_write(manifest, manifest_path, fmt="json")
    print(f"[INFO] MODIS ETL manifest written to {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
