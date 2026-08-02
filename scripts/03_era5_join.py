#!/usr/bin/env python3
"""
03_era5_join.py — ERA5 temperature extraction for QC cells.

Joins the canonical 10-day moving-average temperature (temp_c_ma10) to the QC
dataset and assigns temperature classes. Uses a cached per-cell ERA5 extract for
memory safety.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
from tqdm import tqdm
from _Common import *

ERA5_CACHE = DATA_INTERIM / "era5_qc_cells.parquet"


def _cache_valid(path: Path, cell_keys: set, date_min: pd.Timestamp, date_max: pd.Timestamp) -> bool:
    """Validate existing ERA5 cache against current QC cell list and date range."""
    try:
        df = pd.read_parquet(path, engine=PARQUET_ENGINE, columns=ERA5_COLS_NEEDED)
    except Exception as e:
        print(f"  [WARN] Cannot read cache: {e}")
        return False

    if list(df.columns) != ERA5_COLS_NEEDED:
        print(f"  [WARN] Cache schema mismatch: {list(df.columns)}")
        return False

    if df.duplicated(["date", "lat_id", "lon_id"]).any():
        print("  [WARN] Cache contains duplicated keys")
        return False

    df["date"] = pd.to_datetime(df["date"])
    cache_cells = set(zip(df["lat_id"].astype(int), df["lon_id"].astype(int)))
    if not cell_keys.issubset(cache_cells):
        missing = len(cell_keys - cache_cells)
        print(f"  [WARN] Cache missing {missing:,} QC cells")
        return False

    if df["date"].min() > date_min or df["date"].max() < date_max:
        print(f"  [WARN] Cache date range {df['date'].min().date()} → {df['date'].max().date()} does not cover QC")
        return False

    return True


def main() -> int:
    print("=" * 64)
    print("MAGNETO — ERA5 Temperature Join")
    print("=" * 64)
    setup_dirs()

    # ── Load QC ───────────────────────────────────────────────────────
    print("\n[1/4] Loading QC dataset …")
    df_qc = pd.read_parquet(FILE_QC, engine=PARQUET_ENGINE)
    df_qc["date"] = pd.to_datetime(df_qc["date"])
    print(f"  QC rows: {len(df_qc):,}")
    print(f"  QC date range: {df_qc['date'].min().date()} → {df_qc['date'].max().date()}")
    print(f"  QC cells: {df_qc[['lat_id','lon_id']].drop_duplicates().shape[0]:,}")

    qc_cells = df_qc[["lat_id", "lon_id"]].drop_duplicates()
    cell_keys = set((int(row["lat_id"]), int(row["lon_id"])) for _, row in qc_cells.iterrows())
    print(f"  Cell keys: {len(cell_keys):,}")

    # ── Load/Cache ERA5 ───────────────────────────────────────────────
    date_min = df_qc["date"].min()
    date_max = df_qc["date"].max()
    cache_usable = (
        ERA5_CACHE.exists() and
        _cache_valid(ERA5_CACHE, cell_keys, date_min, date_max)
    )

    if cache_usable:
        print(f"\n[2/4] ERA5 cache exists and is valid: {ERA5_CACHE}")
        print(f"  Size: {ERA5_CACHE.stat().st_size / 1e6:.1f} MB")
        df_e = pd.read_parquet(ERA5_CACHE, engine=PARQUET_ENGINE)
        df_e["date"] = pd.to_datetime(df_e["date"])
        print(f"  Rows: {len(df_e):,}")
    else:
        if ERA5_CACHE.exists():
            print(f"\n[2/4] ERA5 cache invalid — rebuilding: {ERA5_CACHE}")
            ERA5_CACHE.unlink()
        else:
            print("\n[2/4] Scanning ERA5 file (chunked) …")
        print("\n[2/4] Scanning ERA5 file (chunked) …")
        pf = pq.ParquetFile(str(FILE_ERA5))
        n_total = pf.metadata.num_rows
        print(f"  ERA5 total rows: {n_total:,}")
        print(f"  ERA5 row groups: {pf.metadata.num_row_groups}")

        chunks = []
        n_matched = 0
        batch_size = 5_000_000
        n_batches = (n_total // batch_size) + 1

        for batch in tqdm(
            pf.iter_batches(columns=ERA5_COLS_NEEDED, batch_size=batch_size),
            total=n_batches, desc="  scanning ERA5",
        ):
            lat_arr = batch.column("lat_id").to_numpy()
            lon_arr = batch.column("lon_id").to_numpy()
            mask = np.zeros(len(batch), dtype=bool)
            for i in range(len(batch)):
                if (int(lat_arr[i]), int(lon_arr[i])) in cell_keys:
                    mask[i] = True
            if not mask.any():
                continue
            df_chunk = batch.to_pandas().loc[mask].copy()
            df_chunk["date"] = pd.to_datetime(df_chunk["date"])
            n_matched += len(df_chunk)
            chunks.append(df_chunk)

        print(f"  Matched: {n_matched:,} ERA5 rows")
        df_e = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=ERA5_COLS_NEEDED + ["date"])
        atomic_write(df_e, ERA5_CACHE)
        print(f"  Cache written: {ERA5_CACHE} ({ERA5_CACHE.stat().st_size / 1e6:.1f} MB)")

    # ── Join with QC ──────────────────────────────────────────────────
    print("\n[3/4] Joining temperature with QC …")
    before = len(df_qc)
    df_qc = df_qc.merge(
        df_e[["date", "lat_id", "lon_id", ERA5_TEMP_COL]],
        on=["date", "lat_id", "lon_id"],
        how="left",
    )
    print(f"  Before join: {before:,}  →  After: {len(df_qc):,}")
    n_temp = df_qc[ERA5_TEMP_COL].notna().sum()
    print(f"  Rows with temperature: {n_temp:,} ({100*n_temp/len(df_qc):.1f}%)")

    # ── Assign temperature bins ───────────────────────────────────────
    print("\n[4/4] Assigning temperature classes …")
    if n_temp > 0:
        mask = df_qc[ERA5_TEMP_COL].notna()
        tb = bin_temperature(df_qc.loc[mask, ERA5_TEMP_COL])
        df_qc["temp_bin_id"] = np.nan
        df_qc["temp_bin_label"] = pd.NA
        df_qc.loc[mask, "temp_bin_id"] = tb["temp_bin_id"].values
        df_qc.loc[mask, "temp_bin_label"] = tb["temp_bin_label"].values

        print("\n  Temperature class distribution:")
        for label in TEMP_LABELS:
            n = (df_qc["temp_bin_label"] == label).sum()
            if n > 0:
                print(f"    {label:15s}: {n:>10,} ({100*n/len(df_qc):.1f}%)")

    atomic_write(df_qc, FILE_QC)
    print(f"\n  Updated QC: {FILE_QC} ({len(df_qc):,} rows)")
    print(f"\n[OK] 03_era5_join complete — {n_temp:,} rows with temperature")
    return 0


if __name__ == "__main__":
    sys.exit(main())
