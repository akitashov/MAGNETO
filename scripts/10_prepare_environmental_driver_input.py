#!/usr/bin/env python3
"""
10_prepare_environmental_driver_input.py — Build the unified input table for
the environmental-driver matrix and multidriver screens.

This is a thin adapter: it joins the current validated residuals with the
already-produced OMNI and ERA5 outputs. It does NOT recalculate residuals,
reload raw ERA5, or recompute temperature classes.

Output:
    data/processed/environmental_driver_input.parquet

Columns include the residual, temperature bin, scenario flags, SII windows,
PAR windows, VPD windows, and (if present) TCC windows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Make project helpers importable whether this script is run directly or via
# the pipeline runner.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _Common import (
    PROJECT_ROOT,
    FILE_ERA5,
    FILE_QC,
    FILE_OMNI,
    PARQUET_ENGINE,
    atomic_write,
)
from supplementary_checks._supplementary_checks_common import (
    load_harmonic_residuals,
    decode_region_flag,
)


# ── Configuration ───────────────────────────────────────────────────────────
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "environmental_driver_input.parquet"
ENV_CACHE_PATH = PROJECT_ROOT / "data" / "interim" / "env_driver_qc_cells.parquet"

DEFAULT_WINDOWS = list(range(1, 29))  # 1–28 days
DEFAULT_TARGET = "sif_771nm"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare unified environmental-driver input table"
    )
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="Moving-average windows to retain for PAR/VPD/TCC/SII (default: 1..28)",
    )
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="SIF target column name (default: sif_771nm)",
    )
    p.add_argument(
        "--include-tcc",
        action="store_true",
        help="Include TCC (total cloud cover) windows if available",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached ERA5 extract even if it appears valid",
    )
    return p.parse_args(argv)


def _env_columns(prefix: str, windows: Iterable[int]) -> list[str]:
    """Return ['par_ma1', ..., 'par_ma28'] etc."""
    return [f"{prefix}_ma{w}" for w in windows]


def _cache_valid(cache_path: Path, required_cols: list[str],
                 qc_keys: set[tuple[Any, int, int]],
                 date_min: pd.Timestamp, date_max: pd.Timestamp) -> bool:
    """Check that the cached extract covers the required columns, cells, and dates."""
    if not cache_path.exists():
        return False
    try:
        info = pq.ParquetFile(cache_path)
        names = info.schema.names
    except Exception as e:
        print(f"  [WARN] Cannot read env-driver cache: {e}")
        return False

    missing = [c for c in required_cols if c not in names]
    if missing:
        print(f"  [WARN] Env-driver cache missing columns: {missing}")
        return False

    try:
        df = pd.read_parquet(
            cache_path, engine=PARQUET_ENGINE,
            columns=["date", "lat_id", "lon_id"],
        )
    except Exception as e:
        print(f"  [WARN] Cannot read env-driver cache keys: {e}")
        return False

    df["date"] = pd.to_datetime(df["date"]).dt.date
    if df.duplicated(["date", "lat_id", "lon_id"]).any():
        print("  [WARN] Env-driver cache contains duplicated keys")
        return False

    cache_keys = set(
        (row["date"], int(row["lat_id"]), int(row["lon_id"]))
        for _, row in df.iterrows()
    )
    if not qc_keys.issubset(cache_keys):
        missing_keys = len(qc_keys - cache_keys)
        print(f"  [WARN] Env-driver cache missing {missing_keys:,} QC date×cell keys")
        return False

    return True


def _build_env_cache(windows: list[int], qc_keys: set[tuple[Any, int, int]],
                     date_min: pd.Timestamp, date_max: pd.Timestamp,
                     include_tcc: bool, force: bool) -> pd.DataFrame:
    """Extract PAR/VPD/TCC windows for exact QC date×cell keys from ERA5 parquet."""
    required_env_prefixes = ["par", "vpd"]
    # TCC is optional; include only when explicitly requested and present.
    if include_tcc:
        try:
            era5_schema = pq.ParquetFile(FILE_ERA5).schema.names
            if any(c.startswith("tcc_ma") for c in era5_schema):
                required_env_prefixes.append("tcc")
        except Exception as e:
            print(f"  [WARN] Cannot inspect ERA5 schema: {e}")

    required_cols = ["date", "lat_id", "lon_id"]
    for prefix in required_env_prefixes:
        required_cols.extend(_env_columns(prefix, windows))

    if not force and _cache_valid(ENV_CACHE_PATH, required_cols, qc_keys, date_min, date_max):
        print(f"\n[2/4] Env-driver cache exists and is valid: {ENV_CACHE_PATH}")
        cache = pd.read_parquet(ENV_CACHE_PATH, engine=PARQUET_ENGINE)
        cache["date"] = pd.to_datetime(cache["date"])
        return cache

    if ENV_CACHE_PATH.exists():
        print("\n[2/4] Env-driver cache invalid — rebuilding")
        ENV_CACHE_PATH.unlink()
    else:
        print("\n[2/4] Building env-driver cache from ERA5 (chunked scan) …")

    pf = pq.ParquetFile(str(FILE_ERA5))
    n_total = pf.metadata.num_rows
    print(f"  ERA5 total rows: {n_total:,}")
    print(f"  ERA5 row groups: {pf.metadata.num_row_groups}")
    print(f"  QC date×cell keys: {len(qc_keys):,}")

    # Read only columns that exist.
    available_cols = [c for c in required_cols if c in pf.schema.names]
    missing_cols = [c for c in required_cols if c not in pf.schema.names]
    if missing_cols:
        print(f"  [WARN] Columns not found in ERA5 and will be omitted: {missing_cols}")

    chunks: list[pd.DataFrame] = []
    n_matched = 0
    # Smaller batches keep peak memory low on memory-constrained systems.
    batch_size = 1_000_000
    n_batches = max(1, (n_total // batch_size) + 1)

    for batch in tqdm(
        pf.iter_batches(columns=available_cols, batch_size=batch_size),
        total=n_batches,
        desc="  scanning ERA5",
    ):
        # Build date×cell tuples using datetime.date for fast set membership.
        date_arr = pd.to_datetime(batch.column("date").to_numpy()).date
        lat_arr = batch.column("lat_id").to_numpy()
        lon_arr = batch.column("lon_id").to_numpy()
        mask = np.zeros(len(batch), dtype=bool)
        for i in range(len(batch)):
            if (date_arr[i], int(lat_arr[i]), int(lon_arr[i])) in qc_keys:
                mask[i] = True
        if not mask.any():
            continue
        df_chunk = batch.to_pandas().loc[mask].copy()
        df_chunk["date"] = pd.to_datetime(df_chunk["date"])
        n_matched += len(df_chunk)
        chunks.append(df_chunk)
        # Periodically free memory if many small chunks accumulate.
        if len(chunks) >= 50:
            chunks = [pd.concat(chunks, ignore_index=True)]

    print(f"  Matched: {n_matched:,} ERA5 rows")
    cache = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=available_cols)
    )
    atomic_write(cache, ENV_CACHE_PATH)
    print(
        f"  Cache written: {ENV_CACHE_PATH} "
        f"({ENV_CACHE_PATH.stat().st_size / 1e6:.1f} MB)"
    )
    return cache


def _load_omni_windows(windows: list[int]) -> pd.DataFrame:
    """Load the OMNI feather and keep only the SII windows we need."""
    cols = ["date"] + [f"sii_mean_ma{w}" for w in windows]
    omni = pd.read_feather(FILE_OMNI, columns=cols)
    omni["date"] = pd.to_datetime(omni["date"])
    return omni


def main(argv=None) -> int:
    args = parse_args(argv)
    windows = sorted(args.windows)

    print("=" * 64)
    print("MAGNETO — Prepare environmental-driver input")
    print("=" * 64)
    print(f"Target: {args.target}")
    print(f"Windows: {windows[0]}..{windows[-1]} days")

    # ── Load validated harmonic residuals ────────────────────────────────
    print("\n[1/4] Loading harmonic residuals …")
    df = load_harmonic_residuals(args.target)
    print(f"  Residual rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Cells: {df[['lat_id', 'lon_id']].drop_duplicates().shape[0]:,}")

    df["cell_id"] = df["lat_id"].astype(str) + "_" + df["lon_id"].astype(str)
    df["is_SAA"] = decode_region_flag(df["region_flags"], "SAA")

    # ── Build / load env-driver cache ────────────────────────────────────
    # Restrict the ERA5 extract to the exact date × cell keys required by the
    # residual sample. This keeps the cache small (a few million rows) instead of
    # accumulating every ERA5 date for each QC cell.
    qc_keys = set(
        (row["date"].date() if hasattr(row["date"], "date") else row["date"],
         int(row["lat_id"]), int(row["lon_id"]))
        for _, row in df[["date", "lat_id", "lon_id"]].drop_duplicates().iterrows()
    )
    env_cache = _build_env_cache(
        windows,
        qc_keys,
        df["date"].min(),
        df["date"].max(),
        args.include_tcc,
        args.force,
    )

    # ── Join residuals with environmental drivers ────────────────────────
    print("\n[3/4] Joining residuals with PAR/VPD/TCC …")
    before = len(df)
    df = df.merge(
        env_cache,
        on=["date", "lat_id", "lon_id"],
        how="left",
    )
    print(f"  Before join: {before:,}  →  After: {len(df):,}")

    # ── Join SII windows ─────────────────────────────────────────────────
    print("\n[4/4] Joining SII windows …")
    omni = _load_omni_windows(windows)
    before = len(df)
    df = df.merge(omni, on="date", how="left")
    print(f"  Before join: {before:,}  →  After: {len(df):,}")

    # ── Select and order final columns ───────────────────────────────────
    keep = [
        "date",
        "year",
        "lat_id",
        "lon_id",
        "cell_id",
        "residual",
        "temp_bin_label",
        "is_vegetated",
        "is_strict_low_lai",
        "is_Sahara",
        "is_SAA",
        "lai_quartile",
    ] + [f"sii_mean_ma{w}" for w in windows]

    for prefix in ["par", "vpd", "tcc"]:
        cols = _env_columns(prefix, windows)
        keep.extend([c for c in cols if c in df.columns])

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    print(f"\n  Final columns: {len(df.columns)}")
    print(f"  Final rows: {len(df):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(df, OUTPUT_PATH)
    print(f"\n[OK] Written: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
