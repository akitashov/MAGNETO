#!/usr/bin/env python3
"""
ERA5 Temperature Data Processing Pipeline with Weather Context Generation.

Logic:
1.  Iterates through ERA5 NetCDF files (Hourly resolution).
2.  Implements an **Hourly Buffer Strategy**: Maintains the tail of the previous 
    file in memory (in hourly resolution) to ensure temporal continuity across 
    file boundaries and precise daily aggregation.
3.  Aggregates Hourly data to Daily Means (single pass aggregation).
4.  Regrids/Interpolates data to a 0.5 degree target grid.
5.  Calculates a 10-day Rolling Average ('temp_ma10') to define the 
    short-term environmental context (Acute Weather Condition). 10-day is a "Scene" 
    on which the "Actor" (multiday magnetic disturbances) act.
6.  Saves the result as a compressed Parquet file.

OUTPUT FILE DESCRIPTION:
------------------------
Format: .parquet (binary dataframe)
Granularity: Daily
Columns structure:
1.  date          (timestamp): Date of observation.
2.  latitude      (float32):   Grid latitude (-90 to 90).
3.  longitude     (float32):   Grid longitude (-180 to 180).
4.  lat_id        (int16):     Integer latitude ID (lat * 100).
5.  lon_id        (int16):     Integer longitude ID (lon * 100).
6.  temperature_c (float32):   Air temperature at 2m in Celsius (Daily Mean).
7.  temp_ma10     (float32):   10-day Moving Average of temperature.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
import os
from datetime import timedelta
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters and paths."""
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    INPUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'ERA5'
    OUTPUT_DIR = PROJECT_ROOT / 'data' / 'interim'
    OUTPUT_FILE = OUTPUT_DIR / 'era5_temperature_daily_0.5deg.parquet'
    
    # Target Grid Resolution
    TARGET_RESOLUTION = 0.5
    CONVERT_TO_CELSIUS = True
    FILE_PATTERN = "era5_2m_temperature_*.nc"
    
    # Weather Context Window (10 Days)
    # Used to characterize the acute environmental condition.
    CONTEXT_WINDOW = 10
    
    # Buffer size (Hours)
    # Needs to cover the Context Window + margin to ensure valid rolling calcs.
    # 11 days * 24 hours
    BUFFER_DAYS = 11

# ==============================================================================
# PROCESSING UTILITIES
# ==============================================================================

def load_hourly_nc_regridded(file_path: Path) -> pd.DataFrame:
    """
    Loads NetCDF, Regrids to 0.5 deg, returns HOURLY DataFrame.
    Does NOT resample to daily yet.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # 1. Coordinate Standardization
            if 'valid_time' in ds:
                ds = ds.rename({'valid_time': 'time'})
            
            # 2. Drop metadata vars
            for drop_coord in ['number', 'expver', 'surface']:
                if drop_coord in ds.coords:
                    ds = ds.drop_vars(drop_coord)
            
            # 3. Handle Longitude (-180..180)
            if ds.longitude.max() > 180:
                ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                ds = ds.sortby('longitude')
            
            # 4. Interpolate to Target Grid (Hourly level)
            new_lat = np.arange(-90, 90.1, Config.TARGET_RESOLUTION)
            new_lon = np.arange(-180, 180.1, Config.TARGET_RESOLUTION)
            
            ds_coarse = ds.interp(
                latitude=new_lat,
                longitude=new_lon,
                method='linear'
            )
            
            # 5. Convert to Celsius
            if Config.CONVERT_TO_CELSIUS:
                ds_coarse['t2m'] = ds_coarse['t2m'] - 273.15
            
            # 6. Convert to DataFrame
            # Note: This loads the entire hourly grid into memory.
            df = ds_coarse['t2m'].to_dataframe().reset_index()
            
            # 7. Rename & Optimize
            df = df.rename(columns={
                'time': 'datetime', 
                't2m': 'temperature_c',
                'lat': 'latitude', 
                'lon': 'longitude'
            })
            
            df = df.dropna(subset=['temperature_c'])
            
            for col in ['latitude', 'longitude', 'temperature_c']:
                df[col] = df[col].astype('float32')
            
            # Generate IDs for sorting/grouping
            df['lat_id'] = (df['latitude'] * 100).round().astype('int16')
            df['lon_id'] = (df['longitude'] * 100).round().astype('int16')
            
            # Sort: Grid -> Time
            df = df.sort_values(by=['lat_id', 'lon_id', 'datetime'])
            
            return df
            
    except Exception as e:
        print(f"[ERROR] processing {file_path.name}: {e}")
        return pd.DataFrame()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n[INFO] ERA5 ETL: HOURLY BUFFERING & WEATHER CONTEXT")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy: Precise Hourly Buffering -> Daily Aggregation")
    print("-" * 60)
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove old file to ensure clean write
    if Config.OUTPUT_FILE.exists():
        print(f"[WARN] Removing existing file: {Config.OUTPUT_FILE.name}")
        os.remove(Config.OUTPUT_FILE)

    files = sorted(Config.INPUT_DIR.glob(Config.FILE_PATTERN))
    if not files:
        print("[ERROR] No NetCDF files found.")
        return

    # Buffer stores HOURLY data (Tail of previous file)
    hourly_buffer = pd.DataFrame()
    
    total_daily_rows = 0

    with tqdm(files, unit="file") as pbar:
        for file_path in pbar:
            pbar.set_description(f"Proc: {file_path.name[:15]}...")
            
            # 1. Load Hourly Data (Current File)
            df_hourly_current = load_hourly_nc_regridded(file_path)
            if df_hourly_current.empty: continue
            
            # Track time range of current file (to slice output later)
            current_start_dt = df_hourly_current['datetime'].min()
            
            # 2. Concatenate with Buffer (Hourly + Hourly)
            if not hourly_buffer.empty:
                # Combine buffer and current
                df_hourly_combined = pd.concat([hourly_buffer, df_hourly_current], ignore_index=True)
                # Ensure sort after concat
                df_hourly_combined = df_hourly_combined.sort_values(by=['lat_id', 'lon_id', 'datetime'])
            else:
                df_hourly_combined = df_hourly_current

            # 3. SINGLE AGGREGATION: Hourly -> Daily
            daily_agg = (
                df_hourly_combined.set_index('datetime')
                .groupby(['lat_id', 'lon_id', 'latitude', 'longitude'])['temperature_c'] 
                .resample('1D')
                .mean()
                .reset_index()
            )
            
            # Rename resampled time column back to 'date'
            daily_agg = daily_agg.rename(columns={'datetime': 'date'})

            # 4. Calculate Weather Context (MA10)
            roll_col = f'temp_ma{Config.CONTEXT_WINDOW}'
            
            # Only original indexing to be left
            daily_agg[roll_col] = (
                daily_agg.groupby(['lat_id', 'lon_id'])['temperature_c']
                .rolling(window=Config.CONTEXT_WINDOW, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True) 
                .astype('float32')
            )

            # 5. Extract Valid Data
            # We save only dates >= start of the current file.
            # Dates from the buffer were already saved in the previous iteration.
            save_start_date = pd.to_datetime(current_start_dt.date())
            df_to_save = daily_agg[daily_agg['date'] >= save_start_date].copy()
            
            # 6. Prepare Buffer for Next Iteration (Hourly Tail)
            # We keep the last N days of the *combined hourly* data.
            last_timestamp = df_hourly_combined['datetime'].max()
            buffer_cutoff = last_timestamp - timedelta(days=Config.BUFFER_DAYS)
            
            hourly_buffer = df_hourly_combined[df_hourly_combined['datetime'] > buffer_cutoff].copy()
            
            # Cleanup memory
            del df_hourly_current, df_hourly_combined, daily_agg
            gc.collect()
            
            # 7. Write to Disk
            if not df_to_save.empty:
                write_mode = 'append' if Config.OUTPUT_FILE.exists() else 'create'
                if write_mode == 'create':
                    df_to_save.to_parquet(Config.OUTPUT_FILE, engine='fastparquet', index=False)
                else:
                    df_to_save.to_parquet(Config.OUTPUT_FILE, engine='fastparquet', index=False, append=True)
                
                total_daily_rows += len(df_to_save)
            
            del df_to_save
            gc.collect()

    print("-" * 60)
    if Config.OUTPUT_FILE.exists():
        size_mb = Config.OUTPUT_FILE.stat().st_size / 1024**2
        print(f"[SUCCESS] Saved to {Config.OUTPUT_FILE.name}")
        print(f"Total Daily Rows: {total_daily_rows:,}")
        print(f"Final File Size:  {size_mb:.2f} MB")
        print(f"Context Column:   temp_ma{Config.CONTEXT_WINDOW}")
    else:
        print("[ERROR] No data saved.")

from datetime import datetime
if __name__ == "__main__":
    main()