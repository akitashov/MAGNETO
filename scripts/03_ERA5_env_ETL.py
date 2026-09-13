"""
ERA5 Environmental Data Processing Pipeline.

Reads hourly ERA5 NetCDF files for 2014-2024, validates inputs, regrids to the
canonical 0.5° global grid, computes daily environmental variables and GPU-based
NaN-safe lagged rolling windows, and writes a single Parquet output.
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from tqdm import tqdm

from _Common import Config

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR = Config.ERA5_INPUT_DIR
OUTPUT_FILE = Config.FILE_ERA5_PARQUET
TMP_OUTPUT_FILE = Config.DATA_INTERIM / "era5_env_daily.parquet.tmp"
REJECTED_CSV = Config.REPORTS_INTEGRITY_DIR / "era5_rejected_items.csv"
MISSINGNESS_CSV = Config.REPORTS_INTEGRITY_DIR / "era5_missingness.csv"
MANIFEST_CSV = Config.REPORTS_INTEGRITY_DIR / "era5_input_manifest.csv"

METRICS = ["temp_c", "vpd", "par", "tcc"]

# Explicit, checkable mapping from short names to source NetCDF variable names.
# ERA5 CDS downloads for these variables use the same standard names.
ERA5_SOURCE_VARS = {
    "t2m": "t2m",
    "d2m": "d2m",
    "ssrd": "ssrd",
    "tcc": "tcc",
}

SPATIAL_CHUNK_CELLS = 5_000
TIME_CHUNK_DAYS = 500
MAX_ROLLING_WINDOW = max(Config.MA_WINDOWS)

NLAT = Config.CANONICAL_LAT_N_CELLS
NLON = Config.CANONICAL_LON_N_CELLS
N_CELLS = NLAT * NLON

N_CHUNKS = (N_CELLS + SPATIAL_CHUNK_CELLS - 1) // SPATIAL_CHUNK_CELLS
CHUNK_SIZES = [SPATIAL_CHUNK_CELLS] * N_CHUNKS
CHUNK_SIZES[-1] = N_CELLS - (N_CHUNKS - 1) * SPATIAL_CHUNK_CELLS

EXPECTED_DATES = pd.date_range("2014-01-01", "2024-12-31", freq="D")


# ==============================================================================
# DIAGNOSTIC LEDGER
# ==============================================================================

def reject(
    ledger: list[dict[str, Any]],
    year: int,
    month: int,
    day: int | None,
    variable: str,
    stage: str,
    message: str,
) -> None:
    """Record a rejected item and continue; the caller raises after writing."""
    ledger.append(
        {
            "year": year,
            "month": month,
            "day": day if day is not None else "",
            "variable": variable,
            "stage": stage,
            "message": str(message),
        }
    )


def write_rejected_csv(ledger: list[dict[str, Any]]) -> None:
    """Persist the diagnostic ledger."""
    if not ledger:
        return
    Config.REPORTS_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        ledger,
        columns=["year", "month", "day", "variable", "stage", "message"],
    ).to_csv(REJECTED_CSV, index=False)


def fail(ledger: list[dict[str, Any]], message: str) -> None:
    """Write the ledger and exit with a clear error."""
    write_rejected_csv(ledger)
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
# NETCDF LOADING & VALIDATION
# ==============================================================================

def normalize_longitude_xr(ds: xr.Dataset) -> xr.Dataset:
    """Wrap the longitude coordinate into [-180, 180) and sort ascending."""
    if "longitude" not in ds.coords:
        raise ValueError("Dataset has no longitude coordinate")
    lon = ds.coords["longitude"]
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(longitude=lon_norm)
    ds = ds.sortby("longitude")
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    return ds


def load_and_validate(
    fpath: Path,
    short_name: str,
    year: int,
    month: int,
    ledger: list[dict[str, Any]],
    stack: ExitStack,
) -> xr.Dataset:
    """
    Open a single ERA5 NetCDF file via the provided ExitStack, validate year/month
    coverage, handle expver, drop unwanted variables, normalize longitude, and
    rename the source variable to the canonical short name.
    """
    ds = stack.enter_context(xr.open_dataset(fpath))

    # Time coordinate
    if "time" not in ds.coords:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        else:
            reject(ledger, year, month, None, short_name, "load", "missing time coordinate")
            raise ValueError(f"{fpath.name}: neither 'time' nor 'valid_time' coordinate found")

    # Year/month coverage
    years = ds["time"].dt.year.values
    months = ds["time"].dt.month.values
    if not (np.all(years == year) and np.all(months == month)):
        reject(
            ledger,
            year,
            month,
            None,
            short_name,
            "load",
            f"time coverage mismatch: expected {year}-{month:02d}, got unique years {np.unique(years)}, months {np.unique(months)}",
        )
        raise ValueError(f"{fpath.name}: time coverage does not match {year}-{month:02d}")

    # expver dimension (ERA5 download mixes forecast/analysis)
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # Drop nuisance variables if present
    ds = ds.drop_vars(["number", "surface"], errors="ignore")

    # Ensure canonical coordinate names
    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        reject(ledger, year, month, None, short_name, "load", "missing latitude/longitude coordinates")
        raise ValueError(f"{fpath.name}: missing latitude/longitude coordinates")

    # Normalize longitude before any regridding
    ds = normalize_longitude_xr(ds)

    # Checkable variable renaming
    src_var = ERA5_SOURCE_VARS.get(short_name)
    if src_var not in ds.data_vars:
        if short_name in ds.data_vars:
            src_var = short_name
        else:
            available = list(ds.data_vars.keys())
            reject(
                ledger,
                year,
                month,
                None,
                short_name,
                "load",
                f"expected source variable {src_var!r} not found; available: {available}",
            )
            raise KeyError(f"{fpath.name}: expected variable {src_var!r}, found {available}")

    if src_var != short_name:
        ds = ds.rename({src_var: short_name})

    return ds


def validate_shared_coordinates(datasets: dict[str, xr.Dataset]) -> None:
    """Ensure all four variables share identical time/lat/lon after normalization."""
    names = list(datasets.keys())
    ref = datasets[names[0]]

    ref_time = np.asarray(ref["time"].values)
    ref_lat = np.asarray(ref["latitude"].values, dtype=np.float64)
    ref_lon = np.asarray(ref["longitude"].values, dtype=np.float64)

    for name in names[1:]:
        ds = datasets[name]
        if not np.array_equal(np.asarray(ds["time"].values), ref_time):
            raise ValueError(f"Coordinate mismatch: {name} time differs from {names[0]}")
        if not np.allclose(np.asarray(ds["latitude"].values, dtype=np.float64), ref_lat):
            raise ValueError(f"Coordinate mismatch: {name} latitude differs from {names[0]}")
        if not np.allclose(np.asarray(ds["longitude"].values, dtype=np.float64), ref_lon):
            raise ValueError(f"Coordinate mismatch: {name} longitude differs from {names[0]}")


# ==============================================================================
# PHYSICS & REGRIDDING
# ==============================================================================

def calculate_vpd(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """Vapor pressure deficit (kPa) from 2m temperature and 2m dewpoint."""
    t_c = t2m_k - 273.15
    td_c = d2m_k - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    return np.maximum(es - ea, 0.0)


def regrid_to_canonical(ds: xr.Dataset) -> xr.Dataset:
    """Linearly interpolate a Dataset onto the canonical 0.5° grid."""
    target_lat = Config.canonical_lat_centers()
    target_lon = Config.canonical_lon_centers()
    try:
        return ds.interp(latitude=target_lat, longitude=target_lon, method="linear")
    except Exception as exc:
        src_shape = {str(k): int(ds.sizes[k]) for k in ds.dims}
        src_lat = np.asarray(ds["latitude"].values)
        src_lon = np.asarray(ds["longitude"].values)
        raise RuntimeError(
            f"Regridding failed. Source shape={src_shape}, "
            f"source lat range=[{src_lat.min():.4f}, {src_lat.max():.4f}], "
            f"source lon range=[{src_lon.min():.4f}, {src_lon.max():.4f}], "
            f"target lat n={len(target_lat)}, target lon n={len(target_lon)}."
        ) from exc


def compute_chunk_ids(df: pd.DataFrame) -> np.ndarray:
    """Assign each canonical grid cell to a spatial processing chunk."""
    lat_id = df["lat_id"].to_numpy(dtype=np.int32)
    lon_id = df["lon_id"].to_numpy(dtype=np.int32)

    lat_idx = (lat_id + 9000 - 25) // 50
    lon_idx = (lon_id + 18000 - 25) // 50

    lat_idx = np.clip(lat_idx, 0, NLAT - 1)
    lon_idx = np.clip(lon_idx, 0, NLON - 1)

    linear = lat_idx * NLON + lon_idx
    return linear // SPATIAL_CHUNK_CELLS


# ==============================================================================
# DAILY AGGREGATION
# ==============================================================================

def process_month(
    year: int,
    month: int,
    ledger: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load, validate, aggregate and regrid one month of ERA5 data."""
    fpaths: dict[str, Path] = {}
    for short_name, vinfo in Config.ERA5_VAR_MAP.items():
        try:
            fpaths[short_name] = Config.resolve_era5_file(
                INPUT_DIR, vinfo["pattern"], year, month
            )
        except FileNotFoundError as exc:
            reject(ledger, year, month, None, short_name, "resolve", str(exc))
            raise
        except ValueError as exc:
            reject(ledger, year, month, None, short_name, "resolve", str(exc))
            raise

    with ExitStack() as stack:
        datasets: dict[str, xr.Dataset] = {}
        for short_name, fpath in fpaths.items():
            datasets[short_name] = load_and_validate(
                fpath, short_name, year, month, ledger, stack
            )

        validate_shared_coordinates(datasets)

        ref_ds = datasets[next(iter(datasets))]
        try:
            times = pd.to_datetime(ref_ds["time"].values)
        except Exception as exc:
            reject(ledger, year, month, None, "all", "time_parsing", str(exc))
            raise

        days = pd.DatetimeIndex(times).normalize().unique().sort_values()
        day_dfs: list[pd.DataFrame] = []

        for day in days:
            day_str = day.strftime("%Y-%m-%d")
            try:
                slices: dict[str, np.ndarray] = {}
                for short_name in Config.ERA5_VAR_MAP:
                    arr = datasets[short_name][short_name].sel(time=day_str).load().values
                    slices[short_name] = arr

                t2m = slices["t2m"]
                d2m = slices["d2m"]
                ssrd = slices["ssrd"]
                tcc = slices["tcc"]

                expected_shape = (len(ref_ds.sel(time=day_str)["time"]),) + t2m.shape[1:]
                if not (t2m.shape == d2m.shape == ssrd.shape == tcc.shape == expected_shape):
                    raise ValueError(
                        f"Inconsistent shapes on {day_str}: "
                        f"t2m={t2m.shape}, d2m={d2m.shape}, ssrd={ssrd.shape}, tcc={tcc.shape}"
                    )

                temp_c = (t2m - 273.15).astype(np.float32)
                vpd = calculate_vpd(t2m, d2m).astype(np.float32)
                par = (ssrd / Config.ERA5_SECONDS_PER_HOUR * Config.ERA5_PAR_FRACTION_OF_SSRD).astype(np.float32)
                tcc_arr = tcc.astype(np.float32)

                hour_times = ref_ds.sel(time=day_str)["time"].values
                ds_hour = xr.Dataset(
                    {
                        "temp_c": (["time", "latitude", "longitude"], temp_c),
                        "vpd": (["time", "latitude", "longitude"], vpd),
                        "par": (["time", "latitude", "longitude"], par),
                        "tcc": (["time", "latitude", "longitude"], tcc_arr),
                    },
                    coords={
                        "time": hour_times,
                        "latitude": ref_ds["latitude"].values,
                        "longitude": ref_ds["longitude"].values,
                    },
                )

                ds_day = ds_hour.mean(dim="time", skipna=True)
                ds_day_regrid = regrid_to_canonical(ds_day)

                df = ds_day_regrid.to_dataframe().reset_index()
                df["date"] = pd.Timestamp(day)
                lat_ids, lon_ids = Config.latlon_to_ids(
                    df["latitude"].to_numpy(), df["longitude"].to_numpy()
                )
                df["lat_id"] = lat_ids
                df["lon_id"] = lon_ids
                for col in METRICS:
                    df[col] = df[col].astype(np.float32)

                df = df[["date", "latitude", "longitude", "lat_id", "lon_id"] + METRICS]
                day_dfs.append(df)

            except Exception as exc:
                reject(
                    ledger,
                    year,
                    month,
                    day.day,
                    "all",
                    "daily_aggregation",
                    f"{day_str}: {exc}",
                )
                raise

    if not day_dfs:
        raise RuntimeError(f"No daily data produced for {year}-{month:02d}")

    return pd.concat(day_dfs, ignore_index=True)


# ==============================================================================
# PARQUET APPEND HELPER
# ==============================================================================

def append_to_parquet(path: Path, df: pd.DataFrame) -> None:
    """Append a DataFrame to a Parquet file using fastparquet."""
    path = Path(path)
    if path.exists():
        df.to_parquet(path, engine="fastparquet", index=False, append=True)
    else:
        df.to_parquet(path, engine="fastparquet", index=False)


# ==============================================================================
# BASE DATASET CONSTRUCTION
# ==============================================================================

def build_base_shards(
    run_id: str,
    ledger: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    """Aggregate daily data and write spatial base shards for rolling."""
    shard_dir = Config.DATA_INTERIM / f"era5_gpu_chunks_{run_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"base_chunk_{cid:04d}.parquet" for cid in range(N_CHUNKS)]

    date_total_counts: dict[pd.Timestamp, int] = defaultdict(int)
    chunk_first_seen: list[bool] = [False] * N_CHUNKS

    for year in range(Config.ERA5_START_YEAR, Config.ERA5_END_YEAR_EXCLUSIVE):
        for month in tqdm(
            range(1, 13),
            desc=f"ERA5 base {year}",
            leave=False,
        ):
            month_df = process_month(year, month, ledger)

            # Per-chunk write
            chunk_ids = compute_chunk_ids(month_df)
            month_df["_chunk_id"] = chunk_ids
            grouped = month_df.groupby("_chunk_id", sort=True)
            for cid, group in grouped:
                group = group.drop(columns=["_chunk_id"], errors="ignore")
                group = group.sort_values(["date", "lat_id", "lon_id"]).reset_index(drop=True)

                expected = CHUNK_SIZES[cid]
                # Uniqueness + completeness per date within this chunk
                for date_val, sub in group.groupby("date"):
                    if len(sub) != expected:
                        raise RuntimeError(
                            f"Chunk {cid} date {date_val.date()} has {len(sub)} rows, expected {expected}"
                        )
                    if sub[["lat_id", "lon_id"]].duplicated().any():
                        raise RuntimeError(
                            f"Chunk {cid} date {date_val.date()} contains duplicate (lat_id, lon_id)"
                        )

                if not chunk_first_seen[cid]:
                    chunk_first_seen[cid] = True

                append_to_parquet(shard_paths[cid], group)

            # Update global per-date cell counts
            for date_val, cnt in month_df.groupby("date").size().items():
                date_total_counts[pd.Timestamp(date_val)] += int(cnt)

            del month_df, grouped
            gc.collect()

    # Calendar / spatial validation on the base dataset
    observed_dates = set(date_total_counts.keys())
    if observed_dates != set(EXPECTED_DATES):
        missing = sorted(set(EXPECTED_DATES) - observed_dates)
        extra = sorted(observed_dates - set(EXPECTED_DATES))
        raise RuntimeError(
            f"Date set mismatch. Missing: {missing[:5]}... Extra: {extra[:5]}..."
        )

    bad_counts = [
        (d, c) for d, c in date_total_counts.items() if c != N_CELLS
    ]
    if bad_counts:
        raise RuntimeError(
            f"Spatial completeness failed for {len(bad_counts)} dates (expected {N_CELLS}): {bad_counts[:5]}"
        )

    # Year / month / calendar-correct day checks
    for year in range(Config.ERA5_START_YEAR, Config.ERA5_END_YEAR_EXCLUSIVE):
        year_dates = EXPECTED_DATES[EXPECTED_DATES.year == year]
        if len(year_dates) not in (365, 366):
            raise RuntimeError(f"Year {year} has {len(year_dates)} days")
        if year_dates.month.nunique() != 12:
            raise RuntimeError(f"Year {year} is missing months")

    if len(EXPECTED_DATES) != 4018:
        raise RuntimeError(f"Expected 4018 days, got {len(EXPECTED_DATES)}")

    # Consecutive-day check
    diffs = np.diff(EXPECTED_DATES.values.astype("datetime64[D]").astype(np.int32))
    if not np.all(diffs == 1):
        raise RuntimeError("Expected date range is not strictly consecutive")

    return shard_dir, shard_paths


# ==============================================================================
# GPU ROLLING WINDOWS
# ==============================================================================

def gpu_nanmean_lag(arr: cp.ndarray, window: int) -> cp.ndarray:
    """
    NaN-safe lagged rolling mean on GPU.

    Returns the mean of the previous `window` observations excluding the current
    time step. NaN values contribute 0 to the sum and 0 to the count, so a single
    NaN does not poison subsequent cells.
    """
    if arr.ndim != 2:
        raise ValueError("gpu_nanmean_lag expects a 2-D array (time, cells)")
    t, c = arr.shape

    data = cp.nan_to_num(arr, nan=0.0).astype(cp.float32)
    mask = cp.isfinite(arr).astype(cp.float32)

    pad = cp.zeros((window, c), dtype=cp.float32)
    data_p = cp.concatenate([pad, data], axis=0)
    mask_p = cp.concatenate([pad, mask], axis=0)

    s = cp.cumsum(data_p, axis=0)
    n = cp.cumsum(mask_p, axis=0)

    sum_win = s[window : window + t] - s[:t]
    count_win = n[window : window + t] - n[:t]

    mean = sum_win / cp.maximum(count_win, 1.0)
    mean[count_win == 0] = cp.nan

    del data, mask, pad, data_p, mask_p, s, n, sum_win, count_win
    return mean.astype(cp.float32)


def init_missingness() -> dict[str, dict[Any, dict[str, int]]]:
    missingness: dict[str, dict[Any, dict[str, int]]] = {}
    for metric in METRICS:
        missingness[metric] = {"base": {"n_total": 0, "n_nan_before": 0, "n_nan_after": 0}}
        for w in Config.MA_WINDOWS:
            missingness[metric][w] = {"n_total": 0, "n_nan_before": 0, "n_nan_after": 0}
    return missingness


def process_base_shard(
    base_path: Path,
    final_path: Path,
    missingness: dict[str, dict[Any, dict[str, int]]],
) -> None:
    """Read one spatial base shard and append rolled rows to the final tmp file."""
    df = pd.read_parquet(base_path, engine="fastparquet")
    df = df.sort_values(["date", "lat_id", "lon_id"]).reset_index(drop=True)

    dates = pd.to_datetime(df["date"].unique())
    n_dates = len(dates)
    n_cells = CHUNK_SIZES[int(base_path.stem.split("_")[-1])]

    if len(df) != n_dates * n_cells:
        raise RuntimeError(
            f"Shard {base_path.name}: {len(df)} rows, expected {n_dates * n_cells}"
        )

    # Cell metadata (first date block is representative)
    cell_meta = df[["latitude", "longitude", "lat_id", "lon_id"]].iloc[:n_cells].reset_index(drop=True)

    n_time_chunks = int(np.ceil(n_dates / TIME_CHUNK_DAYS))

    for tc in range(n_time_chunks):
        t_start = tc * TIME_CHUNK_DAYS
        t_end = min(t_start + TIME_CHUNK_DAYS, n_dates)
        buf_start = max(0, t_start - MAX_ROLLING_WINDOW)

        start_row = buf_start * n_cells
        end_row = t_end * n_cells
        sub = df.iloc[start_row:end_row].copy()
        t_buf = t_end - buf_start
        t_target = t_end - t_start

        out_df = pd.DataFrame(
            {
                "date": np.repeat(np.asarray(dates[t_start:t_end]), n_cells),
                "latitude": np.tile(cell_meta["latitude"].to_numpy(), t_target).astype(np.float32),
                "longitude": np.tile(cell_meta["longitude"].to_numpy(), t_target).astype(np.float32),
                "lat_id": np.tile(cell_meta["lat_id"].to_numpy(), t_target).astype(np.int16),
                "lon_id": np.tile(cell_meta["lon_id"].to_numpy(), t_target).astype(np.int16),
            }
        )

        for metric in METRICS:
            vals = sub[metric].to_numpy(dtype=np.float32).reshape(t_buf, n_cells)
            gpu_arr = cp.array(vals)
            target_arr = gpu_arr[t_start - buf_start :, :]

            # Base missingness
            base_nan = int(cp.isnan(target_arr).sum().get())
            missingness[metric]["base"]["n_total"] += int(target_arr.size)
            missingness[metric]["base"]["n_nan_before"] += base_nan
            missingness[metric]["base"]["n_nan_after"] += base_nan
            out_df[metric] = cp.asnumpy(target_arr).astype(np.float32).ravel()

            for w in Config.MA_WINDOWS:
                rolled = gpu_nanmean_lag(gpu_arr, w)
                rolled_target = rolled[t_start - buf_start :, :]

                missingness[metric][w]["n_total"] += int(rolled_target.size)
                missingness[metric][w]["n_nan_before"] += base_nan
                missingness[metric][w]["n_nan_after"] += int(cp.isnan(rolled_target).sum().get())
                out_df[f"{metric}_ma{w}"] = cp.asnumpy(rolled_target).astype(np.float32).ravel()

                del rolled, rolled_target

            del gpu_arr, target_arr, vals
            cp.get_default_memory_pool().free_all_blocks()

        append_to_parquet(final_path, out_df)
        del out_df, sub
        gc.collect()

    del df


def write_missingness_csv(missingness: dict[str, dict[Any, dict[str, int]]]) -> None:
    """Write the missingness summary report."""
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for key, rec in missingness[metric].items():
            window = "" if key == "base" else f"ma{key}"
            n_total = rec["n_total"]
            nnb = rec["n_nan_before"]
            nna = rec["n_nan_after"]
            rows.append(
                {
                    "metric": metric,
                    "window": window,
                    "n_total": n_total,
                    "n_nan_before": nnb,
                    "n_nan_after": nna,
                    "frac_nan_before": nnb / n_total if n_total else np.nan,
                    "frac_nan_after": nna / n_total if n_total else np.nan,
                }
            )
    Config.REPORTS_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "metric",
            "window",
            "n_total",
            "n_nan_before",
            "n_nan_after",
            "frac_nan_before",
            "frac_nan_after",
        ],
    ).to_csv(MISSINGNESS_CSV, index=False)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    ledger: list[dict[str, Any]] = []

    try:
        # 1. Manifest
        Config.REPORTS_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)
        manifest = Config.build_era5_manifest(INPUT_DIR, MANIFEST_CSV)
        bad_manifest = manifest[manifest["status"] != "ok"]
        if not bad_manifest.empty:
            print("[ERROR] ERA5 input manifest contains non-ok rows:", file=sys.stderr)
            print(bad_manifest.to_string(index=False), file=sys.stderr)
            sys.exit(1)

        ok_summary = manifest["status"].value_counts().to_dict()
        print(f"[INFO] ERA5 manifest status summary: {ok_summary}")

        run_id = os.environ.get("MAGNETO_RUN_ID") or Config.make_run_id()
        print(f"[INFO] Run ID: {run_id}")

        # 2. Build or reuse base daily shards
        shard_dir = Config.DATA_INTERIM / f"era5_gpu_chunks_{run_id}"
        if os.environ.get("MAGNETO_ERA5_SKIP_BASE", "").lower() in ("1", "true", "yes") and shard_dir.exists():
            shard_paths = sorted(shard_dir.glob("base_chunk_*.parquet"))
            if not shard_paths:
                raise RuntimeError(f"MAGNETO_ERA5_SKIP_BASE set but no shards found in {shard_dir}")
            print(f"[INFO] Reusing existing base shards from {shard_dir}")
        else:
            shard_dir, shard_paths = build_base_shards(run_id, ledger)
            print(f"[INFO] Base shards written to {shard_dir}")

        # 3. Rolling windows
        missingness = init_missingness()
        if TMP_OUTPUT_FILE.exists():
            TMP_OUTPUT_FILE.unlink()

        for shard_path in tqdm(shard_paths, desc="Rolling windows"):
            process_base_shard(shard_path, TMP_OUTPUT_FILE, missingness)

        write_missingness_csv(missingness)
        print(f"[INFO] Missingness report written to {MISSINGNESS_CSV}")

        # 4. Validate final tmp file
        expected_rows = len(EXPECTED_DATES) * N_CELLS
        meta = pq.read_metadata(str(TMP_OUTPUT_FILE))
        actual_rows = meta.num_rows
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Final tmp file row count mismatch: expected {expected_rows}, got {actual_rows}"
            )

        # 5. Atomic rename
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(TMP_OUTPUT_FILE), str(OUTPUT_FILE))

        n_unique_cells = N_CELLS
        print(
            f"[SUCCESS] run_id={run_id}, rows={actual_rows}, "
            f"dates={EXPECTED_DATES[0].date()} to {EXPECTED_DATES[-1].date()}, "
            f"unique_cells={n_unique_cells}, output={OUTPUT_FILE}"
        )

    except Exception as exc:
        fail(ledger, f"ERA5 ETL failed: {exc}")


if __name__ == "__main__":
    main()
