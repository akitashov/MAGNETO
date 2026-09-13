#!/usr/bin/env python3
"""
OCO-2 SIF Data Processing Pipeline with MODIS Environmental Filtering and Region Flagging.

Logic:
1.  CLEANUP: Removes artifacts from previous runs (old .feather/parquet files) to ensure consistency.
2.  MODIS LOAD: Loads environmental data (Cloud, Aerosol, LAI) with spatial dilation to fill gaps.
3.  ITERATION: Iterates through OCO-2 NetCDF files (Lite format) year by year.
4.  GRID SNAPPING: Maps raw SIF points to the nearest 0.5-degree grid center.
5.  FLAGGING: Calculates `region_flags` (int16) for each cell:
    - Applies Geographic masks (SAA, Sahara, Control, Polar) defined in `_Common`.
    - Applies Environmental masks (High/Low LAI) based on MODIS data.
6.  MERGE & FILTER: Joins SIF with MODIS data based on location and time.
    - Filters out Ocean/Absolute No-Data (LAI < Absolute Min).
    - Note: Low LAI pixels (Desert/Barren) are KEPT but flagged as `LOW_LAI` (bit 16)
      to allow for "Negative Control" analysis.
    - Filters out Cloudy/Aerosol contaminated pixels.
7.  INDEX CALCULATION:
    - SIF_Stress_Index = SIF_757 / SIF_771 (Threshold: raw values > 0.001 to avoid div/0).
    - SIF_PSI_PSII_Ratio = SIF_740 / SIF_757 (Threshold: raw values > 0.001).
8.  AGGREGATION: Computes daily means for each grid cell and saves to yearly chunks.
9.  FINAL MERGE: Combines all yearly chunks into a single output file.

CONFIGURATION:
--------------
Paths and Constants are imported from `_Common.py`.

OUTPUT FILE DESCRIPTION:
------------------------
File: data/interim/sif_aggregated.feather
Format: Binary Feather DataFrame
Granularity: Daily per Grid Cell (0.5 deg)

Columns structure:
1.  date               (timestamp): Date of observation (YYYY-MM-DD).
2.  latitude           (float32):   Grid center latitude (-90.0 to 90.0).
3.  longitude          (float32):   Grid center longitude (-180.0 to 180.0).
4.  lat_id             (int16):     Integer latitude ID (lat * 100).
5.  lon_id             (int16):     Integer longitude ID (lon * 100).
6.  sif_740nm          (float32):   Mean Solar Induced Fluorescence at 740nm.
7.  sif_757nm          (float32):   Mean Solar Induced Fluorescence at 757nm.
8.  sif_771nm          (float32):   Mean Solar Induced Fluorescence at 771nm.
9. sif_stress_index   (float32):   Calculated Stress Index (757nm / 771nm).
10. count              (int64):     Number of raw OCO-2 points aggregated in this cell.
11. region_flags       (int16):     Bitwise mask encoding region and vegetation status.
       (e.g., cell can be SAA | HIGH_LAI).
"""

import os
import re
import shutil
import pandas as pd
import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from tqdm import tqdm
import gc
from typing import List, Optional
import warnings
from _Common import Config, RegionFlag

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR_OCO2 = Config.OCO2_INPUT_DIR
START_YEAR = Config.OCO2_START_YEAR
END_YEAR = Config.OCO2_END_YEAR_INCLUSIVE

SIF_MIN_THRESHOLD = Config.SIF_MIN_THRESHOLD
FILTERS = Config.SIF_FILTERS

FILES_SUBBATCH = Config.OCO2_FILES_SUBBATCH


# ==============================================================================
# HELPERS
# ==============================================================================

def get_modis_period_date(target_dates):
    """
    Convert daily timestamps to the MODIS 8-day compositing period start date.

    Parameters
    ----------
    target_dates : pd.Series[datetime64]
        Daily dates.

    Returns
    -------
    pd.Series[datetime64]
        Period start date for each input day (year-aware).
    """
    years = target_dates.dt.year
    start_of_years = pd.to_datetime(years.astype(str) + "-01-01")
    doy = target_dates.dt.dayofyear - 1
    period_start_doy = (doy // 8) * 8
    return start_of_years + pd.to_timedelta(period_start_doy, unit='D')

def clean_previous_artifacts():
    """
    Remove outputs from previous runs to ensure reproducible results.

    Deletes:
    - Final merged SIF feather (if present)
    - Intermediate yearly sif_aggregated_*.feather shards
    - SIF model directory (if present)
    """
    if Config.FILE_SIF_FINAL.exists(): Config.FILE_SIF_FINAL.unlink(missing_ok=True)
    for f in Config.DATA_INTERIM.glob('sif_aggregated_*.feather'): f.unlink()
    if Config.DIR_SIF_MODEL.exists(): shutil.rmtree(Config.DIR_SIF_MODEL)

# ==============================================================================
# LOGIC
# ==============================================================================

class ModisHandler:
    """
    Year-scoped MODIS accessor and batch filter for SIF aggregation.

    Loads MODIS parquet in a year slice and provides:
    - MODIS presence detection (has_modis)
    - Quality filters (cloud/aerosol/LAI thresholds)
    """
    def __init__(self, parquet_path):
        self.path = parquet_path
        self.data = None

    def load_year(self, year):
        try:
            filters = [
                ("date", ">=", pd.Timestamp(f"{year}-01-01")),
                ("date", "<=", pd.Timestamp(f"{year}-12-31")),
            ]
            cols = Config.MODIS_COLS_FOR_SIF
            df_real = pq.read_table(self.path, columns=cols, filters=filters).to_pandas()

            expanded = []
            for d_lat, d_lon in Config.modis_dilation_shifts():
                df_s = df_real.copy()
                if d_lat != 0:
                    df_s["lat_id"] = df_s["lat_id"] + d_lat
                if d_lon != 0:
                    df_s["lon_id"] = df_s["lon_id"] + d_lon
                expanded.append(df_s)

            self.data = (
                pd.concat(expanded, ignore_index=True)
                .drop_duplicates(subset=["date", "lat_id", "lon_id"], keep="first")
            )

            gc.collect()
        except Exception as e:
            print(f"[ERROR] MODIS Load: {e}")
            self.data = pd.DataFrame()
            raise

    def filter_batch(self, sif_df):
        """
        Merge a SIF batch with MODIS data and apply MODIS-based validity filters.

        Parameters
        ----------
        sif_df : pd.DataFrame
            Must include: date, lat_id, lon_id (and SIF columns).

        Returns
        -------
        pd.DataFrame
            Subset of sif_df rows that have matching MODIS data and pass filters.
        """
        if self.data is None or self.data.empty: return pd.DataFrame(columns=sif_df.columns)

        sif_merged = sif_df.copy()
        sif_merged['modis_date_key'] = get_modis_period_date(sif_merged['date'])

        merged = pd.merge(sif_merged, self.data, left_on=['modis_date_key', 'lat_id', 'lon_id'],
                          right_on=['date', 'lat_id', 'lon_id'], how='left', suffixes=('', '_modis'))

        # Presence of ANY MODIS sky/land variable should count as "has_modis"
        has_modis = (
            merged["cloud_fraction"].notna()
            | merged["aerosol_fraction"].notna()
            | merged["lai"].notna()
        )

        temp_df = merged.copy()

        # Fill for filtering (keep has_modis separate!)
        temp_df["cloud_fraction"] = temp_df["cloud_fraction"].fillna(999)
        temp_df["aerosol_fraction"] = temp_df["aerosol_fraction"].fillna(999)
        temp_df["lai"] = temp_df["lai"].fillna(-1)

        mask_cloud = temp_df["cloud_fraction"] < FILTERS["cloud_max"]
        mask_aerosol = temp_df["aerosol_fraction"] < FILTERS["aerosol_max"]
        mask_lai = temp_df["lai"] >= FILTERS["lai_min"]

        final_mask = has_modis & mask_cloud & mask_aerosol & mask_lai
        filtered_df = merged[final_mask].copy()

        if 'quality_flag_x' in filtered_df.columns:
            filtered_df = filtered_df.rename(columns={'quality_flag_x': 'quality_flag'})

        return filtered_df

def aggregate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute region_flags and aggregate raw SIF observations to daily 0.5° grid cells.

    Input
    -----
    df columns (minimum):
    - date, latitude, longitude, lat_id, lon_id
    - sif_740nm, sif_757nm, sif_771nm
    Optionally:
    - lai (used to apply LOW_LAI/HIGH_LAI flags)

    Output
    ------
    Aggregated DataFrame with:
    - date, latitude, longitude, lat_id, lon_id, region_flags
    - sif_740nm, sif_757nm, sif_771nm (mean)
    - count (number of raw points)
    - sif_stress_index = sif_757nm / sif_771nm (masked by threshold)
    """

    # 1. Calculate Region Flags (Bitwise)
    # We use vectorization for speed

    # A. Geographic Flags (already computed in _Common)
    flags = Config.get_geo_flags_vectorized(
        df["latitude"].values,
        df["longitude"].values
    )

    # B. Environmental Flags (LAI-based)
    if "lai" in df.columns:
        flags = Config.apply_lai_flags(flags, df["lai"].values)

    df["region_flags"] = flags.astype(np.int16)

    # 2. Aggregation
    # Include 'region_flags' in grouping to preserve distinct flags per cell (usually static, but LAI might vary)

    group_keys = ["date", "lat_id", "lon_id", "latitude", "longitude", "region_flags"]

    agg_rules = { 'sif_740nm': 'mean', 'sif_757nm': 'mean', 'sif_771nm': 'mean' }
    aggregated = df.groupby(group_keys).agg(agg_rules).reset_index()

    count_df = df.groupby(group_keys).size().reset_index(name='count')
    aggregated = pd.merge(aggregated, count_df, on=group_keys)

    # Indices
    aggregated['sif_stress_index'] = np.where(
        aggregated['sif_771nm'] > SIF_MIN_THRESHOLD,
        aggregated['sif_757nm'] / aggregated['sif_771nm'],
        np.nan
    ).astype(np.float32)

    aggregated.replace([np.inf, -np.inf], np.nan, inplace=True)
    return aggregated

# Anchored regex for OCO-2 Lite SIF filenames: oco2_LtSIF_YYMMDD_*.nc4
OCO2_FILENAME_RE = re.compile(r"^oco2_LtSIF_(\d{6})_.*\.nc4$")


def parse_oco2_filename(filename: str, expected_year: int) -> tuple[pd.Timestamp, str]:
    """
    Parse the acquisition date from an OCO-2 Lite SIF filename.

    Returns
    -------
    (date, status) where status is 'parsed', 'rejected_malformed', or
    'rejected_year_mismatch'.
    """
    m = OCO2_FILENAME_RE.match(filename)
    if not m:
        return pd.NaT, "rejected_malformed"

    date_str = m.group(1)
    try:
        date = pd.to_datetime(date_str, format="%y%m%d")
    except Exception:
        return pd.NaT, "rejected_malformed"

    # Allow a short tolerance for files whose date straddles the calendar year.
    if abs(date.year - expected_year) > 0:
        return date, "rejected_year_mismatch"
    return date, "parsed"


def process_single_file_raw(sif_file, expected_year: int):
    """
    Parse one OCO-2 LtSIF NetCDF file into a row-wise DataFrame of valid observations.

    Raises
    ------
    ValueError
        If the filename is malformed, the year mismatches, or the file lacks
        required variables/coordinates.
    """
    date, status = parse_oco2_filename(sif_file.name, expected_year)
    if status == "rejected_malformed":
        raise ValueError(f"OCO-2 filename does not match expected pattern: {sif_file.name}")
    if status == "rejected_year_mismatch":
        raise ValueError(
            f"OCO-2 filename year mismatch: {sif_file.name} parsed date {date.date()} "
            f"does not match expected year {expected_year}"
        )

    with xr.open_dataset(sif_file) as ds:
        if "Latitude" not in ds:
            raise ValueError(f"Latitude coordinate missing in {sif_file.name}")
        lat = ds["Latitude"].values
        lon = ds["Longitude"].values

        sif_740 = ds["Daily_SIF_740nm"].values.flatten()
        sif_757 = ds["Daily_SIF_757nm"].values.flatten()
        sif_771 = ds["Daily_SIF_771nm"].values.flatten()

        if "SimplyGoodOrBadQualityFlag" in ds:
            q_mask = (ds["SimplyGoodOrBadQualityFlag"].values.flatten() == 0)
        elif "Quality_Flag" in ds:
            q_mask = (ds["Quality_Flag"].values.flatten() == 0)
        else:
            raise ValueError(f"No recognized quality flag in {sif_file.name}")

    valid_idx = np.where(q_mask)[0]
    if len(valid_idx) == 0:
        # No valid observations is not an error; return empty frame with correct schema.
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "latitude": pd.Series(dtype="float32"),
            "longitude": pd.Series(dtype="float32"),
            "lat_id": pd.Series(dtype="int16"),
            "lon_id": pd.Series(dtype="int16"),
            "sif_740nm": pd.Series(dtype="float32"),
            "sif_757nm": pd.Series(dtype="float32"),
            "sif_771nm": pd.Series(dtype="float32"),
        })

    lat_vals = lat.flatten()[valid_idx]
    lon_vals = lon.flatten()[valid_idx]

    # Snap to canonical grid and derive IDs from cell centers.
    lat_s, lon_s = Config.snap_to_canonical(lat_vals, lon_vals)
    lat_id, lon_id = Config.latlon_to_ids(lat_vals, lon_vals)

    return pd.DataFrame({
        "date": date,
        "latitude": lat_s,
        "longitude": lon_s,
        "lat_id": lat_id,
        "lon_id": lon_id,
        "sif_740nm": sif_740[valid_idx],
        "sif_757nm": sif_757[valid_idx],
        "sif_771nm": sif_771[valid_idx],
    })

def process_year(year_files, year, modis, pbar, manifest):
    """
    Process all SIF files for a single year:
    - load MODIS slice for that year
    - parse SIF files, sub-batch in memory
    - filter by MODIS availability/quality
    - aggregate to daily grid and save yearly feather shard

    Returns
    -------
    bool
        True if any data were produced for the year.
    """
    pbar.set_description(f"Year {year} [Load MODIS]")
    modis.load_year(year)
    pbar.set_description(f"Year {year} [Processing]")

    year_output = Config.DATA_INTERIM / f'sif_aggregated_{year}.feather'
    all_chunks = []
    batch_data = []

    for sif_file in year_files:
        try:
            df = process_single_file_raw(sif_file, expected_year=year)
            manifest.append({
                "file_path": str(sif_file),
                "filename": sif_file.name,
                "parsed_date": df["date"].iloc[0] if not df.empty else pd.NaT,
                "expected_year": year,
                "status": "parsed",
                "message": "",
            })
            if not df.empty:
                batch_data.append(df)
        except Exception as exc:
            manifest.append({
                "file_path": str(sif_file),
                "filename": sif_file.name,
                "parsed_date": pd.NaT,
                "expected_year": year,
                "status": "rejected_unreadable",
                "message": str(exc),
            })
        pbar.update(1)

        if len(batch_data) >= 50:  # small sub-batch for memory safety
            filtered = modis.filter_batch(pd.concat(batch_data, ignore_index=True))
            if not filtered.empty:
                all_chunks.append(aggregate_dataframe(filtered))
            batch_data = []

    if batch_data:
        filtered = modis.filter_batch(pd.concat(batch_data, ignore_index=True))
        if not filtered.empty:
            all_chunks.append(aggregate_dataframe(filtered))

    if all_chunks:
        # Re-aggregate year level
        full_year = pd.concat(all_chunks, ignore_index=True)
        final_grp = full_year.groupby(["date", "latitude", "longitude", "lat_id", "lon_id", "region_flags"])
        final_y = final_grp.agg({
            'sif_740nm': 'mean', 'sif_757nm': 'mean', 'sif_771nm': 'mean',
            'sif_stress_index': 'mean', 'count': 'sum'
        }).reset_index()
        final_y.to_feather(year_output)
        return True
    return False

def main():
    clean_previous_artifacts()
    Config.REPORTS_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)

    yearly_files = {}
    total_files = 0
    for year in range(START_YEAR, END_YEAR + 1):
        files = sorted(INPUT_DIR_OCO2.glob(f'oco2_LtSIF_{str(year)[-2:]}*.nc4'))
        if files:
            yearly_files[year] = files
            total_files += len(files)

    if total_files == 0:
        raise RuntimeError(f"No OCO-2 files found in {INPUT_DIR_OCO2} for years {START_YEAR}-{END_YEAR}")

    manifest = []
    modis_handler = ModisHandler(Config.FILE_MODIS_PARQUET)
    with tqdm(total=total_files, unit="file") as pbar:
        for year in sorted(yearly_files.keys()):
            process_year(yearly_files[year], year, modis_handler, pbar, manifest)

    # Persist manifest
    manifest_df = pd.DataFrame(
        manifest,
        columns=["file_path", "filename", "parsed_date", "expected_year", "status", "message"],
    )
    manifest_path = Config.REPORTS_INTEGRITY_DIR / "oco2_input_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    n_rejected = (manifest_df["status"] != "parsed").sum()
    print(f"\n[INFO] OCO-2 manifest saved: {manifest_path}")
    print(f"[INFO] Files: {len(manifest_df)} total, {(manifest_df['status'] == 'parsed').sum()} parsed, {n_rejected} rejected")

    if n_rejected > 0:
        rejected = manifest_df[manifest_df["status"] != "parsed"]
        print("[ERROR] Rejected OCO-2 files detected:")
        print(rejected.to_string(index=False))
        raise RuntimeError(f"{n_rejected} OCO-2 files were rejected; see {manifest_path}")

    print("\n[INFO] Final Merge...")
    files = sorted(Config.DATA_INTERIM.glob("sif_aggregated_*.feather"))
    if files:
        final_table = pa.concat_tables([feather.read_table(f) for f in files])
        feather.write_feather(final_table, Config.FILE_SIF_FINAL)
        for f in files:
            f.unlink()
        print(f"[SUCCESS] Saved {final_table.num_rows:,} rows with Region Flags.")
    else:
        raise RuntimeError("No yearly SIF shards were produced.")


if __name__ == "__main__":
    main()