#!/usr/bin/env python3
"""
MODIS LAI/FPAR Data Processing Pipeline.

Logic:
1.  Iterates through MODIS NetCDF files (CMG/Global usually 0.05 deg).
2.  Extracts variables: LAI, Quality, Cloud Fraction, Aerosol Fraction.
3.  AGGREGATION: Downsamples data to a 0.5-degree grid (matching ERA5/SIF).
    - Computes the mean for physical values (LAI, Clouds).
    - Snaps coordinates to the nearest 0.5 grid center.
4.  INDEXING: Generates integer IDs (lat_id, lon_id) for reliable joining.
5.  Saves the result to a compressed Parquet file.

CONFIGURATION:
--------------
Paths and Constants are imported from `_Common.py`.

OUTPUT FILE DESCRIPTION:
------------------------
Format: .parquet
Columns structure:
1.  date              (timestamp): Time of observation (snapped to 8-day period start).
2.  latitude          (float32):   Grid latitude (0.5 deg center).
3.  longitude         (float32):   Grid longitude (0.5 deg center).
4.  lat_id            (int16):     Integer latitude ID (lat * 100).
5.  lon_id            (int16):     Integer longitude ID (lon * 100).
6.  lai               (float32):   Leaf Area Index (mean).
7.  quality_flag      (float32):   Quality flag (mean).
8.  cloud_fraction    (float32):   Cloud fraction (mean).
9.  aerosol_fraction  (float32):   Aerosol fraction (mean).
"""

from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _Common import Config

warnings.filterwarnings("ignore")


# ----------------------------
# Defaults (overridable via Config.*)
# ----------------------------
DEFAULT_VAR_MAPPING = {
    "lai": "lai",
    "primary_qualityflag": "quality_flag",
    "cloudfraction": "cloud_fraction",
    "aerosolfraction": "aerosol_fraction",
}
DEFAULT_FILE_PATTERN = "*.nc"

def normalize_percent(series, name):
    """
    Normalize percentage-like variables to 0–100 scale if needed.

    Parameters:
        series (pd.Series): Input values.
        name (str): Variable name (for logging).

    Returns:
        pd.Series:
            Normalized series in 0–100 range when possible.
            Original series otherwise.
    """
    mx = series.max()
    mn = series.min()

    if pd.isna(mx):
        return series

    if mx <= 1.5:
        print(f"[INFO] {name} normalized: 0–1 → 0–100")
        return series * 100.0

    if mx <= 100.0:
        return series

    print(f"[WARN] {name} unexpected range: min={mn}, max={mx}")
    return series

def snap_modis_date_to_period_start(dates: pd.Series, period_days: int, last_start_doy: int) -> pd.Series:
    """
    Snap timestamps to the start of the MODIS compositing period.
    Default logic: 8-day buckets starting at DOY 1,9,17,... capped to 361.
    """
    dt = pd.to_datetime(dates, errors="coerce")
    years = dt.dt.year
    doys = dt.dt.dayofyear
    snapped = ((doys - 1) // period_days) * period_days + 1
    snapped = np.minimum(snapped, last_start_doy)
    return pd.to_datetime(years.astype(str) + snapped.astype(str).str.zfill(3), format="%Y%j", errors="coerce")


def process_modis_file(file_path, var_mapping: dict, target_res: float, period_days: int, last_start_doy: int,
                       verbose_errors: bool) -> pd.DataFrame:
    """
    Process a single MODIS NetCDF file into standardized grid format.

    Parameters:
        file_path (Path): Path to NetCDF file.
        var_mapping (dict): Mapping from source variables to standard names.
        target_res (float): Target spatial resolution (degrees).
        period_days (int): MODIS compositing period length.
        last_start_doy (int): Last valid period start DOY.
        verbose_errors (bool): Print detailed errors if True.

    Returns:
        pd.DataFrame:
            Columns:
            - date
            - latitude, longitude
            - lat_id, lon_id
            - lai
            - quality_flag
            - cloud_fraction
            - aerosol_fraction

            Empty DataFrame if file is invalid.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # ensure all required variables exist
            required_vars = list(var_mapping.keys())
            missing = [v for v in required_vars if v not in ds.variables and v not in ds.data_vars]
            if missing:
                return pd.DataFrame()

            subset = ds[required_vars]
            df = subset.to_dataframe().reset_index()

        # Rename to standard column names
        rename_dict = dict(var_mapping)
        if "lat" in df.columns:
            rename_dict["lat"] = "latitude"
        if "lon" in df.columns:
            rename_dict["lon"] = "longitude"
        if "time" in df.columns:
            rename_dict["time"] = "date"

        df = df.rename(columns=rename_dict)

        # Guard: must have key coordinates/time
        if not {"latitude", "longitude", "date"}.issubset(df.columns):
            return pd.DataFrame()

        # Snap to target grid
        res = float(target_res)
        df["latitude"] = (np.round(df["latitude"].astype("float64") / res) * res).astype("float32")
        df["longitude"] = (np.round(df["longitude"].astype("float64") / res) * res).astype("float32")

        # Snap time to MODIS period start (e.g., 8-day)
        df["date"] = snap_modis_date_to_period_start(df["date"], period_days=period_days, last_start_doy=last_start_doy)

        # Drop rows with broken timestamps after snapping
        df = df.dropna(subset=["date"])

        # Aggregate to (date, lat, lon)
        df = df.groupby(["date", "latitude", "longitude"], as_index=False).mean(numeric_only=True)

        # IDs for joining
        df["lat_id"] = (df["latitude"] * 100).round().astype("int16")
        df["lon_id"] = (df["longitude"] * 100).round().astype("int16")

        # Enforce float32 on key vars if present
        for col in ["lai", "quality_flag", "cloud_fraction", "aerosol_fraction"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        return df

    except Exception as e:
        if verbose_errors:
            tqdm.write(f"[MODIS ERROR] {file_path.name}: {e}")
        return pd.DataFrame()


def main() -> None:
    # Pull config from _Common (with safe defaults)
    input_dir = getattr(Config, "MODIS_INPUT_DIR", Config.DATA_RAW / "MODIS")
    file_pattern = getattr(Config, "MODIS_FILE_PATTERN", DEFAULT_FILE_PATTERN)
    var_mapping = getattr(Config, "MODIS_VAR_MAPPING", DEFAULT_VAR_MAPPING)

    period_days = int(getattr(Config, "MODIS_PERIOD_DAYS", 8))
    last_start_doy = int(getattr(Config, "MODIS_PERIOD_LAST_DOY", 361))

    parquet_engine = getattr(Config, "MODIS_PARQUET_ENGINE", "fastparquet")
    gc_every = int(getattr(Config, "MODIS_GC_EVERY_N_FILES", 10))
    verbose_errors = bool(getattr(Config, "MODIS_VERBOSE_ERRORS", False))

    out_path = Config.FILE_MODIS_PARQUET
    Config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Clean previous artifact
    if out_path.exists():
        os.remove(out_path)

    files = sorted(input_dir.glob(file_pattern))
    total_rows = 0

    with tqdm(files, unit="file", desc="MODIS ETL") as pbar:
        for i, file_path in enumerate(pbar):
            df = process_modis_file(
                file_path=file_path,
                var_mapping=var_mapping,
                target_res=Config.TARGET_RESOLUTION,
                period_days=period_days,
                last_start_doy=last_start_doy,
                verbose_errors=verbose_errors,
            )

            if df.empty:
                continue

            total_rows += len(df)

            # Append per-file (keeps RAM bounded)
            if not out_path.exists():
                df.to_parquet(out_path, engine=parquet_engine, index=False)
            else:
                df.to_parquet(out_path, engine=parquet_engine, index=False, append=True)

            del df
            if gc_every > 0 and (i % gc_every == 0):
                gc.collect()

    print(f"[SUCCESS] MODIS saved: {total_rows:,} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
