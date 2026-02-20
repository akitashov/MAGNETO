"""
ERA5 Environmental Data Processing Pipeline: Direct Streaming & GPU Acceleration.

Logic:
1.  **Direct Streaming (CPU)**: Iterates through raw NetCDF files year-by-year. 
    Applies physics (VPD, PAR calculation) and regridding in memory without creating 
    intermediate disk shards.
2.  **Hybrid Memory Management (Chunking)**: 
    Processes data in **60-day chunks** with a **100-day overlap buffer** (kept in RAM 
    from the previous year iteration) to handle rolling window calculations without 
    memory explosion.
3.  **GPU Acceleration (CuPy)**:
    Replaces slow Pandas rolling windows with vectorized GPU Matrix operations:
    - Uploads (Time, Space) matrices to VRAM.
    - Uses `cp.cumsum` to compute all rolling windows instantly.
    - Offloads results back to CPU immediately.
4.  **Context Generation**: Calculates Rolling Averages for all metrics ('*_ma[1..90]') 
    to define the environmental context.
5.  **Streamed Output**: Results are appended to temporary parquet files chunk-by-chunk, 
    then merged into the final dataset.

CONFIGURATION:
--------------
Paths and Constants are imported from `_Common.py`.

OUTPUT FILE DESCRIPTION:
------------------------
File: data/interim/era5_env_daily.parquet
Format: .parquet (binary dataframe)
Granularity: Daily
Columns structure:
1.  date              (timestamp): Date of observation.
2.  latitude          (float32):   Grid latitude (-90 to 90).
3.  longitude         (float32):   Grid longitude (-180 to 180).
4.  lat_id            (int16):     Integer latitude ID (lat * 100).
5.  lon_id            (int16):     Integer longitude ID (lon * 100).
6.  temp_c            (float32):   Air temperature at 2m in Celsius.
7.  vpd               (float32):   Vapor Pressure Deficit (kPa).
8.  par               (float32):   Photosynthetically Active Radiation (W/m²).
9.  tcc               (float32):   Total Cloud Cover (0-1 fraction).
10. [var]_ma[N]       (float32):   Moving Averages for all variables (Context).
"""


import xarray as xr
import pandas as pd
import numpy as np
import cupy as cp
import warnings
import gc
import shutil
from pathlib import Path
from tqdm import tqdm
from _Common import Config

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR = Config.DATA_RAW / "ERA5"
OUTPUT_DIR = Config.DATA_INTERIM / "era5_gpu_chunks"
OUTPUT_FILE = Config.FILE_ERA5_PARQUET

METRICS = ['temp_c', 'vpd', 'par', 'tcc']
CHUNK_DAYS = 60      
BUFFER_DAYS = 100    

# ==============================================================================
# 1. PHYSICS & EXTRACTION (CPU)
# ==============================================================================

def calculate_vpd_numpy(t2m_k, d2m_k):
    t_c = t2m_k - 273.15
    td_c = d2m_k - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    return np.maximum(es - ea, 0.0)

def fast_regrid(ds_daily):
    try:
        new_lat = np.arange(-90, 90.1, Config.TARGET_RESOLUTION)
        new_lon = np.arange(-180, 180.1, Config.TARGET_RESOLUTION)
        return ds_daily.interp(latitude=new_lat, longitude=new_lon, method='linear')
    except Exception as e:
        print(f"Regrid error: {e}")
        return ds_daily

def load_and_process_year_netcdf(year):
    """
    Reads raw NetCDF for a full year, applies physics, regrids, and returns a DataFrame.
    """
    month_dfs = []
    # print(f"  > Loading NetCDF for {year}...")
    
    for month in range(1, 13):
        try:
            datasets = {}
            valid_month = True
            
            for v_name, v_info in Config.ERA5_VAR_MAP.items():
                fpath = INPUT_DIR / v_info['pattern'].format(year, month)
                if not fpath.exists(): 
                    valid_month = False; break
                
                ds = xr.open_dataset(fpath, chunks={})
                if 'valid_time' in ds: ds = ds.rename({'valid_time': 'time'})
                if 'expver' in ds.dims: ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                ds = ds.drop_vars(['number', 'surface'], errors='ignore')
                
                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
                
                var_key = list(ds.data_vars)[0]
                datasets[v_name] = ds.rename({var_key: v_name})

            if not valid_month: continue

            # Daily Aggregation
            times = datasets['t2m'].time
            days = np.unique(times.dt.day)
            
            # Pre-load month to memory to speed up day loop? No, too big. 
            # Slice day-by-day is safer for RAM.
            
            month_buffer = []
            for day in days:
                day_str = f"{year}-{month:02d}-{day:02d}"
                try:
                    slices = {v: ds[v].sel(time=day_str).load() for v, ds in datasets.items()}
                    
                    vpd = calculate_vpd_numpy(slices['t2m'].values, slices['d2m'].values)
                    par = (slices['ssrd'].values / Config.ERA5_SECONDS_PER_HOUR) * Config.ERA5_PAR_FRACTION_OF_SSRD
                    temp_c = slices['t2m'].values - 273.15
                    
                    ds_hour = xr.Dataset({
                        'temp_c': (('time', 'latitude', 'longitude'), temp_c),
                        'vpd': (('time', 'latitude', 'longitude'), vpd),
                        'par': (('time', 'latitude', 'longitude'), par),
                        'tcc': (('time', 'latitude', 'longitude'), slices['tcc'].values)
                    }, coords=slices['t2m'].coords)
                    
                    # Mean -> Regrid -> DataFrame
                    ds_daily = fast_regrid(ds_hour.mean(dim='time'))
                    df = ds_daily.to_dataframe().reset_index()
                    
                    # Optimization: Drop heavy indices, keep minimal
                    df['date'] = pd.Timestamp(day_str)
                    df['lat_id'] = (df['latitude'] * 100).round().astype('int16')
                    df['lon_id'] = (df['longitude'] * 100).round().astype('int16')
                    
                    for c in METRICS: df[c] = df[c].astype('float32')
                    month_buffer.append(df[['date', 'lat_id', 'lon_id'] + METRICS])
                    
                except Exception: continue
            
            for ds in datasets.values(): ds.close()
            if month_buffer: month_dfs.append(pd.concat(month_buffer, ignore_index=True))
            
        except Exception as e:
            print(f"[WARN] Error {year}-{month}: {e}")
            continue

    if not month_dfs: return None
    return pd.concat(month_dfs, ignore_index=True)

# ==============================================================================
# 2. GPU CALCULATION
# ==============================================================================

def gpu_rolling_calc(time_chunk_data, windows):
    total_len, space_dim = time_chunk_data.shape
    gpu_data = cp.array(time_chunk_data, dtype=cp.float32)
    
    padding = cp.zeros((1, space_dim), dtype=cp.float32)
    gpu_S = cp.cumsum(cp.concatenate([padding, gpu_data], axis=0), axis=0)
    gpu_C = cp.arange(total_len + 1, dtype=cp.float32)[:, None] 
    
    del gpu_data, padding
    results = {}
    
    for w in windows:
        S_upper = gpu_S[1:]
        C_upper = gpu_C[1:]
        
        if w >= total_len:
             S_lower = cp.zeros((total_len, space_dim), dtype=cp.float32)
             C_lower = cp.zeros((total_len, 1), dtype=cp.float32)
        else:
             S_part = gpu_S[:total_len - w]
             zeros_S = cp.zeros((w, space_dim), dtype=cp.float32)
             S_lower = cp.concatenate([zeros_S, S_part], axis=0)
             
             C_part = gpu_C[:total_len - w]
             zeros_C = cp.zeros((w, 1), dtype=cp.float32)
             C_lower = cp.concatenate([zeros_C, C_part], axis=0)

        valid_mean = (S_upper - S_lower) / (C_upper - C_lower)
        results[w] = cp.asnumpy(valid_mean)
        del valid_mean, S_lower, C_lower
        
    del gpu_S, gpu_C
    cp.get_default_memory_pool().free_all_blocks()
    return results

def process_year_on_gpu(df_year, df_prev, year):
    """
    Takes the current year DF and (optionally) the previous year DF from RAM.
    Computes rolling windows and saves to disk.
    """
    year_output_file = OUTPUT_DIR / f"processed_{year}.parquet"
    if year_output_file.exists(): year_output_file.unlink()

    # Create Buffer
    if df_prev is not None:
        buffer = df_prev[df_prev['date'] > (df_prev['date'].max() - pd.Timedelta(days=BUFFER_DAYS))]
        df_full = pd.concat([buffer, df_year], ignore_index=True)
        del buffer
    else:
        df_full = df_year

    df_full = df_full.sort_values(['lat_id', 'lon_id', 'date'])

    # Grid Metadata
    lat_ids = np.sort(df_full['lat_id'].unique())
    lon_ids = np.sort(df_full['lon_id'].unique())
    dates_full = np.sort(df_full['date'].unique())
    space_dim = len(lat_ids) * len(lon_ids)
    n_time = len(dates_full)

    # Reshape for GPU
    raw_arrays = {}
    for m in METRICS:
        raw_arrays[m] = df_full[m].values.reshape(space_dim, n_time).T

    # We drop the dataframe wrapper logic to save memory, keeping only raw_arrays for GPU processing.
    # Note: df_year persists in the main loop (passed as argument), so we don't need to return it.

    # Chunking
    target_mask = pd.to_datetime(dates_full).year == year
    target_indices = np.where(target_mask)[0]
    chunks = np.array_split(target_indices, np.ceil(len(target_indices) / CHUNK_DAYS))

    # Output Grid
    xx, yy = np.meshgrid(lon_ids, lat_ids)
    lats_flat = yy.flatten()
    lons_flat = xx.flatten()

    for idx_list in tqdm(chunks, desc=f"GPU Calc {year}", leave=False):
        abs_start = idx_list[0]
        abs_end = idx_list[-1] + 1
        calc_start = max(0, abs_start - BUFFER_DAYS)
        
        chunk_len = len(idx_list)
        chunk_dates = np.repeat(dates_full[idx_list], space_dim)
        
        chunk_df = pd.DataFrame({
            'date': chunk_dates,
            'lat_id': np.tile(lats_flat, chunk_len),
            'lon_id': np.tile(lons_flat, chunk_len)
        })
        chunk_df['date'] = pd.to_datetime(chunk_df['date']).astype('datetime64[ns]')

        for m in METRICS:
            data_slice = raw_arrays[m][calc_start : abs_end, :]
            gpu_res = gpu_rolling_calc(data_slice, Config.MA_WINDOWS)
            
            raw_offset = abs_start - calc_start
            chunk_df[m] = data_slice[raw_offset:, :].flatten().astype('float32')
            
            for w, res_arr in gpu_res.items():
                chunk_df[f"{m}_ma{w}"] = res_arr[raw_offset:, :].flatten().astype('float32')

        if not year_output_file.exists():
            chunk_df.to_parquet(year_output_file, engine='fastparquet', index=False)
        else:
            chunk_df.to_parquet(year_output_file, engine='fastparquet', index=False, append=True)
        
        del chunk_df
        gc.collect()
        
    del raw_arrays
    gc.collect()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"[INFO] Starting ERA5 Direct Streaming Pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    prev_year_df = None
    years = range(Config.ERA5_START_YEAR, Config.ERA5_END_YEAR_EXCLUSIVE)
    
    for year in tqdm(years, desc="Total Progress"):
        # 1. CPU Heavy: Load & Process NetCDF
        # print(f"\n[CPU] Loading {year}...")
        curr_year_df = load_and_process_year_netcdf(year)
        
        if curr_year_df is None or curr_year_df.empty:
            print(f"[WARN] No data for {year}")
            prev_year_df = None # Reset buffer chain
            continue
            
        # 2. GPU Heavy: Calculate Windows (using prev_year_df from RAM)
        process_year_on_gpu(curr_year_df, prev_year_df, year)
        
        # 3. Rotate Buffer
        # We keep curr_year_df in RAM for the next iteration
        del prev_year_df
        prev_year_df = curr_year_df
        gc.collect()

    print("\n[MERGE] Finalizing...")
    if OUTPUT_FILE.exists(): OUTPUT_FILE.unlink()
    
    chunks = sorted(list(OUTPUT_DIR.glob("processed_*.parquet")))
    for i, chunk in enumerate(tqdm(chunks, desc="Merging")):
        df = pd.read_parquet(chunk)
        if i == 0:
            df.to_parquet(OUTPUT_FILE, engine='fastparquet', index=False)
        else:
            df.to_parquet(OUTPUT_FILE, engine='fastparquet', index=False, append=True)
        del df; gc.collect()

    print(f"[SUCCESS] Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()