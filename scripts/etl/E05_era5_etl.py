#!/usr/bin/env python3
"""E05_era5_etl.py — ERA5 environmental upstream ETL.

Reads hourly ERA5 NetCDF files for 2 m temperature, 2 m dewpoint temperature,
surface solar radiation downwards and total cloud cover, computes daily means,
VPD and PAR, regrids to the canonical 0.5° grid, calculates shifted rolling
windows, and writes ``data/interim/era5_env_daily.parquet``.

Rolling-window arithmetic is performed on the GPU with CuPy, matching the
legacy v1 ERA5 ETL implementation.  Per-year matrices keep VRAM usage bounded.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Allow imports of shared utilities from scripts/ regardless of where this
# ETL script is executed.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (
    PARQUET_ENGINE,
    PROJECT_ROOT,
    setup_dirs,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml

_PIPELINE_CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
_PIPELINE_CFG = yaml.safe_load(_PIPELINE_CFG_PATH.read_text(encoding="utf-8"))
_ECFG = _PIPELINE_CFG.get("era5", {})

INPUT_DIR = PROJECT_ROOT / _ECFG.get("input_dir", "data/raw/ERA5")
OUTPUT_PATH = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_era5"]
CANDIDATE_PATH = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_candidate.parquet")
START_YEAR = int(_ECFG.get("start_year", 2014))
END_YEAR_EXCLUSIVE = int(_ECFG.get("end_year_exclusive", 2025))
TARGET_RESOLUTION = float(_ECFG.get("target_resolution", 0.5))

# Candidate ↔ canonical promotion helper.
import _etl_promote as _promote

VAR_MAP = {
    "t2m":  "era5_2m_temperature_{}_{:02d}.nc",
    "d2m":  "era5_2m_dewpoint_temperature_{}_{:02d}.nc",
    "ssrd": "era5_surface_solar_radiation_downwards_{}_{:02d}.nc",
    "tcc":  "era5_total_cloud_cover_{}_{:02d}.nc",
}

METRICS = ["temp_c", "vpd", "par", "tcc"]
PAR_FRACTION_OF_SSRD = 0.45
SECONDS_PER_HOUR = 3600.0

# Full window set used by downstream environmental-driver analyses.
MA_WINDOWS = list(range(1, 29)) + [30, 40, 50, 60, 75, 90]

CHUNK_DAYS = 60
BUFFER_DAYS = 100
OUTPUT_CHUNK_DIR = PROJECT_ROOT / "data" / "interim" / "era5_gpu_chunks"


def _calculate_vpd(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """Vapor pressure deficit in kPa from temperature and dewpoint in Kelvin."""
    t_c = t2m_k - 273.15
    td_c = d2m_k - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    return np.maximum(es - ea, 0.0)


def _regrid_to_target(ds_daily: xr.Dataset) -> xr.Dataset:
    """Linear interpolation to the canonical 0.5° grid."""
    new_lat = np.arange(-90, 90.0 + TARGET_RESOLUTION / 2, TARGET_RESOLUTION)
    new_lon = np.arange(-180, 180.0 + TARGET_RESOLUTION / 2, TARGET_RESOLUTION)
    return ds_daily.interp(latitude=new_lat, longitude=new_lon, method="linear")


def _process_year(year: int) -> pd.DataFrame | None:
    """Load one year of ERA5 files, aggregate to daily, regrid, return DataFrame."""
    month_dfs = []

    for month in range(1, 13):
        datasets: dict[str, xr.Dataset] = {}
        valid_month = True

        for v_name, pattern in VAR_MAP.items():
            fpath = INPUT_DIR / pattern.format(year, month)
            if not fpath.exists():
                valid_month = False
                break

            ds = xr.open_dataset(fpath)
            if "valid_time" in ds:
                ds = ds.rename({"valid_time": "time"})
            if "expver" in ds.dims:
                ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            ds = ds.drop_vars(["number", "surface"], errors="ignore")

            if float(ds.longitude.max()) > 180:
                ds = ds.assign_coords(
                    longitude=(((ds.longitude + 180) % 360) - 180)
                ).sortby("longitude")

            var_key = list(ds.data_vars)[0]
            datasets[v_name] = ds.rename({var_key: v_name})

        if not valid_month:
            for ds in datasets.values():
                ds.close()
            continue

        try:
            times = datasets["t2m"].time
            days = np.unique(times.dt.day.values)
        except Exception:
            for ds in datasets.values():
                ds.close()
            continue

        month_buffer = []
        for day in days:
            day_str = f"{year}-{month:02d}-{day:02d}"
            try:
                slices = {
                    v: ds.sel(time=day_str).load()
                    for v, ds in datasets.items()
                }

                vpd = _calculate_vpd(
                    slices["t2m"].values,
                    slices["d2m"].values,
                )
                par = (
                    slices["ssrd"].values / SECONDS_PER_HOUR
                ) * PAR_FRACTION_OF_SSRD
                temp_c = slices["t2m"].values - 273.15

                ds_hour = xr.Dataset(
                    {
                        "temp_c": (("time", "latitude", "longitude"), temp_c),
                        "vpd": (("time", "latitude", "longitude"), vpd),
                        "par": (("time", "latitude", "longitude"), par),
                        "tcc": (("time", "latitude", "longitude"), slices["tcc"].values),
                    },
                    coords=slices["t2m"].coords,
                )

                ds_daily = _regrid_to_target(ds_hour.mean(dim="time"))
                df = ds_daily.to_dataframe().reset_index()
                df["date"] = pd.Timestamp(day_str)
                df["lat_id"] = (df["latitude"] * 100).round().astype("int16")
                df["lon_id"] = (df["longitude"] * 100).round().astype("int16")

                for c in METRICS:
                    df[c] = df[c].astype("float32")

                month_buffer.append(df[["date", "lat_id", "lon_id", "latitude", "longitude"] + METRICS])
            except Exception:
                continue

        for ds in datasets.values():
            ds.close()

        if month_buffer:
            month_dfs.append(pd.concat(month_buffer, ignore_index=True))

    if not month_dfs:
        return None
    return pd.concat(month_dfs, ignore_index=True)


def _gpu_rolling_calc(time_chunk_data: np.ndarray, windows: list[int]) -> dict[int, np.ndarray]:
    """Shift(1) + rolling mean for all windows using CuPy cumsum."""
    total_len, space_dim = time_chunk_data.shape
    gpu_data = cp.array(time_chunk_data, dtype=cp.float32)

    # Shift by 1 day (no leakage): first row becomes NaN.
    shifted = cp.concatenate([cp.full((1, space_dim), cp.nan, dtype=cp.float32), gpu_data[:-1, :]], axis=0)
    padding = cp.zeros((1, space_dim), dtype=cp.float32)
    gpu_S = cp.cumsum(cp.concatenate([padding, shifted], axis=0), axis=0)
    gpu_C = cp.arange(total_len + 1, dtype=cp.float32)[:, None]

    del padding, shifted, gpu_data
    results = {}

    for w in windows:
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

        valid_mean = (gpu_S[1:] - S_lower) / (gpu_C[1:] - C_lower)
        results[w] = cp.asnumpy(valid_mean)
        del valid_mean, S_lower, C_lower

    del gpu_S, gpu_C
    cp.get_default_memory_pool().free_all_blocks()
    return results


def _process_year_on_gpu(df_year: pd.DataFrame, df_prev: pd.DataFrame | None, year: int) -> None:
    """Compute rolling windows for one year using a buffer from the previous year."""
    year_output_file = OUTPUT_CHUNK_DIR / f"processed_{year}.parquet"
    if year_output_file.exists():
        year_output_file.unlink()

    if df_prev is not None:
        buffer = df_prev[df_prev["date"] > (df_prev["date"].max() - pd.Timedelta(days=BUFFER_DAYS))]
        df_full = pd.concat([buffer, df_year], ignore_index=True)
        del buffer
    else:
        df_full = df_year

    df_full = df_full.sort_values(["lat_id", "lon_id", "date"])

    lat_ids = np.sort(df_full["lat_id"].unique())
    lon_ids = np.sort(df_full["lon_id"].unique())
    dates_full = np.sort(df_full["date"].unique())
    space_dim = len(lat_ids) * len(lon_ids)
    n_time = len(dates_full)

    # Build (time, space) arrays for each metric.
    raw_arrays = {}
    for m in METRICS:
        arr = df_full[m].values.reshape(space_dim, n_time).T
        raw_arrays[m] = arr

    target_mask = pd.to_datetime(dates_full).year == year
    target_indices = np.where(target_mask)[0]
    chunks = np.array_split(target_indices, np.ceil(len(target_indices) / CHUNK_DAYS).astype(int))

    xx, yy = np.meshgrid(lon_ids, lat_ids)
    lats_flat = yy.flatten()
    lons_flat = xx.flatten()

    for idx_list in tqdm(chunks, desc=f"GPU {year}", leave=False):
        abs_start = int(idx_list[0])
        abs_end = int(idx_list[-1]) + 1
        calc_start = max(0, abs_start - BUFFER_DAYS)

        chunk_len = len(idx_list)
        chunk_dates = np.repeat(dates_full[idx_list], space_dim)

        chunk_df = pd.DataFrame({
            "date": chunk_dates,
            "lat_id": np.tile(lats_flat, chunk_len),
            "lon_id": np.tile(lons_flat, chunk_len),
        })
        chunk_df["date"] = pd.to_datetime(chunk_df["date"]).astype("datetime64[ns]")

        for m in METRICS:
            data_slice = raw_arrays[m][calc_start:abs_end, :]
            gpu_res = _gpu_rolling_calc(data_slice, MA_WINDOWS)

            raw_offset = abs_start - calc_start
            chunk_df[m] = data_slice[raw_offset:, :].flatten().astype("float32")

            for w, res_arr in gpu_res.items():
                chunk_df[f"{m}_ma{w}"] = res_arr[raw_offset:, :].flatten().astype("float32")

        if not year_output_file.exists():
            chunk_df.to_parquet(year_output_file, engine=PARQUET_ENGINE, index=False)
        else:
            chunk_df.to_parquet(year_output_file, engine=PARQUET_ENGINE, index=False, append=True)

        del chunk_df
        gc.collect()

    del raw_arrays
    gc.collect()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MAGNETO ERA5 ETL")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output before writing",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("MAGNETO — ERA5 Environmental ETL")
    print("=" * 64)
    setup_dirs()

    if args.clean:
        for p in [OUTPUT_PATH, CANDIDATE_PATH]:
            if p.exists():
                p.unlink()
                print(f"[INFO] Removed existing output: {p}")

    if args.clean and OUTPUT_CHUNK_DIR.exists():
        shutil.rmtree(OUTPUT_CHUNK_DIR)
    OUTPUT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(START_YEAR, END_YEAR_EXCLUSIVE))
    total_rows = 0

    prev_year_df = None
    for year in tqdm(years, desc="ERA5 years"):
        curr_year_df = _process_year(year)
        if curr_year_df is None or curr_year_df.empty:
            print(f"[WARN] No data for {year}")
            prev_year_df = None
            continue

        _process_year_on_gpu(curr_year_df, prev_year_df, year)
        total_rows += len(curr_year_df)

        del prev_year_df
        prev_year_df = curr_year_df
        gc.collect()

    print("\n[MERGE] Finalizing candidate...")
    if CANDIDATE_PATH.exists():
        CANDIDATE_PATH.unlink()

    chunks = sorted(list(OUTPUT_CHUNK_DIR.glob("processed_*.parquet")))
    for i, chunk in enumerate(tqdm(chunks, desc="Merging")):
        df = pd.read_parquet(chunk, engine=PARQUET_ENGINE)
        if i == 0:
            df.to_parquet(CANDIDATE_PATH, engine=PARQUET_ENGINE, index=False)
        else:
            df.to_parquet(CANDIDATE_PATH, engine=PARQUET_ENGINE, index=False, append=True)
        del df
        gc.collect()

    spec = {
        "name": "era5_env_daily",
        "fmt": "parquet",
        "required": ["date", "lat_id", "lon_id", "temp_c", "temp_c_ma10"],
        "key_cols": ["date", "lat_id", "lon_id"],
        "metric_cols": ["temp_c", "temp_c_ma10", "vpd", "par"],
        "grid_cols": ["lat_id", "lon_id"],
    }
    report = _promote.compare_and_promote(OUTPUT_PATH, CANDIDATE_PATH, spec)
    if not report["passed"]:
        print("[STOP] ERA5 candidate differs from canonical. Promotion blocked; operator review required.")
        return 2

    print(f"\n{'='*64}")
    print(f"ERA5 ETL SUMMARY")
    print(f"  Years processed: {len(years)}")
    print(f"  Total rows:      {total_rows:,}")
    print(f"  Output:          {OUTPUT_PATH}")
    print(f"[OK] E05_era5_etl complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
