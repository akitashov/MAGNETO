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

def process_single_file_raw(sif_file):
    """
    Parse one OCO-2 LtSIF NetCDF file into a row-wise DataFrame of valid observations.

    Processing
    ----------
    - Reads lat/lon and SIF bands
    - Applies quality mask (GoodOrBad / Quality_Flag)
    - Snaps lat/lon to the target 0.5° grid
    - Extracts date from filename

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns:
        date, latitude, longitude, lat_id, lon_id, sif_740nm, sif_757nm, sif_771nm
        or None if file is invalid/unreadable.
    """
    try:
        with xr.open_dataset(sif_file) as ds:
            if 'Latitude' in ds: lat, lon = ds['Latitude'].values, ds['Longitude'].values
            else: return None
            
            sif_740 = ds['Daily_SIF_740nm'].values.flatten()
            sif_757 = ds['Daily_SIF_757nm'].values.flatten()
            sif_771 = ds['Daily_SIF_771nm'].values.flatten()
            
            if 'SimplyGoodOrBadQualityFlag' in ds: q_mask = (ds['SimplyGoodOrBadQualityFlag'].values.flatten() == 0)
            elif 'Quality_Flag' in ds: q_mask = (ds['Quality_Flag'].values.flatten() == 0)
            else: return None

            valid_idx = np.where(q_mask)[0]
            if len(valid_idx) == 0: return None
            
            try:
                date_str = sif_file.name.split('_')[2] 
                date = pd.to_datetime(date_str, format='%y%m%d')
            except: return None

            lat_vals = lat.flatten()[valid_idx]
            lon_vals = lon.flatten()[valid_idx]
            
            # Snap
            res = Config.TARGET_RESOLUTION
            lat_s = (np.round(lat_vals / res) * res).astype(np.float32)
            lon_s = (np.round(lon_vals / res) * res).astype(np.float32)

            return pd.DataFrame({
                "date": date,
                "latitude": lat_s,
                "longitude": lon_s,
                "lat_id": (lat_s * 100).round().astype(np.int16),
                "lon_id": (lon_s * 100).round().astype(np.int16),
                "sif_740nm": sif_740[valid_idx],
                "sif_757nm": sif_757[valid_idx],
                "sif_771nm": sif_771[valid_idx],
            })

    except: return None

def process_year(year_files, year, modis, pbar):
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
        df = process_single_file_raw(sif_file)
        pbar.update(1)
        if df is not None:
            batch_data.append(df)
            if len(batch_data) >= 50: # small sub-batch for memory safety
                filtered = modis.filter_batch(pd.concat(batch_data))
                if not filtered.empty: all_chunks.append(aggregate_dataframe(filtered))
                batch_data = []
                
    if batch_data:
        filtered = modis.filter_batch(pd.concat(batch_data))
        if not filtered.empty: all_chunks.append(aggregate_dataframe(filtered))

    if all_chunks:
        # Re-aggregate year level
        full_year = pd.concat(all_chunks)
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
    yearly_files = {}
    total_files = 0
    for year in range(START_YEAR, END_YEAR + 1):
        files = sorted(INPUT_DIR_OCO2.glob(f'oco2_LtSIF_{str(year)[-2:]}*.nc4'))
        if files:
            yearly_files[year] = files
            total_files += len(files)

    if total_files == 0: return

    modis_handler = ModisHandler(Config.FILE_MODIS_PARQUET)
    with tqdm(total=total_files, unit="file") as pbar:
        for year in sorted(yearly_files.keys()):
            process_year(yearly_files[year], year, modis_handler, pbar)

    print("\n[INFO] Final Merge...")
    files = sorted(Config.DATA_INTERIM.glob("sif_aggregated_*.feather"))
    if files:
        final_table = pa.concat_tables([feather.read_table(f) for f in files])
        feather.write_feather(final_table, Config.FILE_SIF_FINAL)
        for f in files:
            f.unlink()
        print(f"[SUCCESS] Saved {final_table.num_rows:,} rows with Region Flags.")


if __name__ == "__main__":
    main()