#!/usr/bin/env python3
"""E03_sif_etl.py — OCO-2 SIF upstream ETL.

Reads OCO-2 LtSIF Lite NetCDF files, applies retrieval-quality masks, snaps
observations to the canonical 0.5° analysis grid, merges with the
conservatively regridded MODIS environmental data, assigns region flags, and
aggregates to daily mean SIF per cell.

MODIS matching uses central grid-cell lookup by default.  If the central match
rate falls below the configured threshold, the script stops and reports the
unmatched observations so the operator can decide whether a nearest-cell
fallback is justified.  Automatic 3×3 spatial dilation is disabled by default.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from datetime import datetime
from enum import IntFlag
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import xarray as xr
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (  # noqa: E402
    PARQUET_ENGINE,
    PROJECT_ROOT,
    setup_dirs,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml

_PIPELINE_CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
_PIPELINE_CFG = yaml.safe_load(_PIPELINE_CFG_PATH.read_text(encoding="utf-8"))
_OCFG = _PIPELINE_CFG["oco2"]

INPUT_DIR_OCO2 = PROJECT_ROOT / _OCFG["input_dir"]
FILE_GLOB = _OCFG["file_glob"]
START_YEAR = int(_OCFG["start_year"])
END_YEAR = int(_OCFG["end_year"])
TARGET_RESOLUTION = float(_OCFG["target_resolution"])
SIF_MIN_THRESHOLD = float(_OCFG["sif_min_threshold"])
FILES_SUBBATCH = int(_OCFG["files_subbatch"])
FILTERS = {k: float(v) for k, v in _OCFG["filters"].items()}

_MODIS_DILATION_MODE = _OCFG["modis_dilation"]["mode"]
_MODIS_DILATION_RADIUS = int(_OCFG["modis_dilation"]["radius"])
_CENTRAL_MATCH_MIN_RATE = float(_OCFG.get("central_match_min_rate", 0.99))

_RCFG = _OCFG["region_flags"]
_BIT_DEFS = {k: int(v) for k, v in _RCFG["bits"].items()}
_GEOBOXES = _RCFG["boxes"]
_POLAR_LAT_ABS = float(_RCFG["polar_lat_abs"])
_LAI_VEGETATION_THRESHOLD = float(_RCFG["lai_vegetation_threshold"])

FILE_MODIS_PARQUET = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_modis"]
FILE_SIF_FINAL = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_sif"]
FILE_SIF_CANDIDATE = FILE_SIF_FINAL.with_name(FILE_SIF_FINAL.stem + "_candidate.feather")
DATA_INTERIM = PROJECT_ROOT / _PIPELINE_CFG["paths"]["data_interim"]

import _etl_promote as _promote


class RegionFlag(IntFlag):
    """Bitwise region classification used during SIF aggregation."""

    NONE = 0
    SAA = _BIT_DEFS["SAA"]
    CONTROL_NORTH = _BIT_DEFS["CONTROL_NORTH"]
    POLAR = _BIT_DEFS["POLAR"]
    SAHARA = _BIT_DEFS["SAHARA"]
    LOW_LAI = _BIT_DEFS["LOW_LAI"]
    HIGH_LAI = _BIT_DEFS["HIGH_LAI"]


def _in_box(
    lats: np.ndarray,
    lons: np.ndarray,
    box: dict[str, float],
) -> np.ndarray:
    """Return boolean mask for points inside a lat/lon box (inclusive bounds)."""
    return (
        (lats >= box["lat_min"])
        & (lats <= box["lat_max"])
        & (lons >= box["lon_min"])
        & (lons <= box["lon_max"])
    )


def compute_geo_flags(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute geographic region flags for arrays of lat/lon."""
    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    flags = np.full(lats.shape, int(RegionFlag.NONE), dtype=np.int16)

    flags[_in_box(lats, lons, _GEOBOXES["SAA"])] |= int(RegionFlag.SAA)
    flags[_in_box(lats, lons, _GEOBOXES["CONTROL_NA"])] |= int(RegionFlag.CONTROL_NORTH)
    flags[_in_box(lats, lons, _GEOBOXES["CONTROL_EUASIA"])] |= int(RegionFlag.CONTROL_NORTH)
    flags[np.abs(lats) > _POLAR_LAT_ABS] |= int(RegionFlag.POLAR)
    flags[_in_box(lats, lons, _GEOBOXES["SAHARA"])] |= int(RegionFlag.SAHARA)

    return flags


def apply_lai_flags(flags: np.ndarray, lai: np.ndarray) -> np.ndarray:
    """Add LOW_LAI / HIGH_LAI flags based on MODIS LAI."""
    flags = np.asarray(flags, dtype=np.int16).copy()
    lai = np.asarray(lai, dtype=np.float32)
    mask_high = np.isfinite(lai) & (lai >= _LAI_VEGETATION_THRESHOLD)
    mask_low = np.isfinite(lai) & (lai < _LAI_VEGETATION_THRESHOLD)
    flags[mask_high] |= int(RegionFlag.HIGH_LAI)
    flags[mask_low] |= int(RegionFlag.LOW_LAI)
    return flags


def _dilation_shifts() -> list[tuple[int, int]]:
    """Return (d_lat_id, d_lon_id) shifts for MODIS spatial dilation."""
    step = int(round(TARGET_RESOLUTION * 100))
    shifts = [(0, 0)]
    if _MODIS_DILATION_MODE == "none" or _MODIS_DILATION_RADIUS <= 0:
        return shifts

    offs = [k * step for k in range(-_MODIS_DILATION_RADIUS, _MODIS_DILATION_RADIUS + 1)]

    if _MODIS_DILATION_MODE == "cross":
        for d in offs:
            shifts.append((d, 0))
            shifts.append((0, d))
    elif _MODIS_DILATION_MODE == "3x3":
        for dlat in offs:
            for dlon in offs:
                if dlat == 0 and dlon == 0:
                    continue
                shifts.append((dlat, dlon))
    return shifts


class ModisHandler:
    """Year-scoped MODIS accessor with optional spatial dilation."""

    def __init__(self, parquet_path: Path):
        self.path = parquet_path
        self.data: pd.DataFrame | None = None
        self.match_diagnostics: list[dict] = []
        self.unmatched_cell_counts: dict[tuple[int, int, int], int] = {}
        self._current_year: int | None = None

    def load_year(self, year: int) -> None:
        self._current_year = year
        try:
            filters = [
                ("date", ">=", pd.Timestamp(f"{year}-01-01")),
                ("date", "<=", pd.Timestamp(f"{year}-12-31")),
            ]
            cols = ["date", "lat_id", "lon_id", "cloud_fraction", "aerosol_fraction", "lai"]
            df_real = pd.read_parquet(
                self.path,
                engine=PARQUET_ENGINE,
                columns=cols,
                filters=filters,
            )

            shifts = _dilation_shifts()
            if len(shifts) == 1:
                self.data = df_real
            else:
                expanded = []
                for d_lat, d_lon in shifts:
                    df_s = df_real.copy()
                    df_s["orig_lat_id"] = df_s["lat_id"]
                    df_s["orig_lon_id"] = df_s["lon_id"]
                    if d_lat != 0:
                        df_s["lat_id"] = df_s["lat_id"] + d_lat
                    if d_lon != 0:
                        df_s["lon_id"] = df_s["lon_id"] + d_lon
                    expanded.append(df_s)
                # With explicit dilation enabled, resolve duplicates by choosing the
                # closest offset to the original cell (shift (0,0) first), then by
                # smallest Manhattan distance.  This removes the old order-dependent
                # keep="first" behaviour.
                stacked = pd.concat(expanded, ignore_index=True)
                stacked["_dilation_rank"] = (
                    (stacked["lat_id"] - stacked["orig_lat_id"]).abs()
                    + (stacked["lon_id"] - stacked["orig_lon_id"]).abs()
                )
                self.data = (
                    stacked.sort_values("_dilation_rank")
                    .drop_duplicates(subset=["date", "lat_id", "lon_id"], keep="first")
                    .drop(columns=["_dilation_rank", "orig_lat_id", "orig_lon_id"])
                )
        except Exception as e:
            print(f"[ERROR] MODIS load for {year}: {e}")
            self.data = pd.DataFrame()
            raise

    def _modis_period_date(self, target_dates: pd.Series) -> pd.Series:
        """Map daily dates to MODIS 8-day compositing period start dates."""
        years = target_dates.dt.year
        start_of_years = pd.to_datetime(years.astype(str) + "-01-01")
        doy = target_dates.dt.dayofyear - 1
        period_start_doy = (doy // 8) * 8
        return start_of_years + pd.to_timedelta(period_start_doy, unit="D")

    def filter_batch(self, sif_df: pd.DataFrame) -> pd.DataFrame:
        """Merge SIF batch with MODIS and apply MODIS-based validity filters."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=sif_df.columns)

        sif_merged = sif_df.copy()
        sif_merged["modis_date_key"] = self._modis_period_date(sif_merged["date"])

        merged = pd.merge(
            sif_merged,
            self.data,
            left_on=["modis_date_key", "lat_id", "lon_id"],
            right_on=["date", "lat_id", "lon_id"],
            how="left",
            suffixes=("", "_modis"),
        )

        has_modis = (
            merged["cloud_fraction"].notna()
            | merged["aerosol_fraction"].notna()
            | merged["lai"].notna()
        )

        # Record central-match diagnostics for this batch.
        self.match_diagnostics.append({
            "year": self._current_year,
            "batch_sif_observations": int(len(sif_df)),
            "batch_central_matches": int(has_modis.sum()),
        })

        # Accumulate unmatched SIF observations by cell for geographic diagnosis.
        unmatched = merged.loc[~has_modis, ["lat_id", "lon_id"]].dropna()
        if not unmatched.empty and self._current_year is not None:
            counts = unmatched.groupby(["lat_id", "lon_id"]).size()
            for (lat_id, lon_id), n in counts.items():
                key = (int(self._current_year), int(lat_id), int(lon_id))
                self.unmatched_cell_counts[key] = self.unmatched_cell_counts.get(key, 0) + int(n)

        temp_df = merged.copy()
        temp_df["cloud_fraction"] = temp_df["cloud_fraction"].fillna(999)
        temp_df["aerosol_fraction"] = temp_df["aerosol_fraction"].fillna(999)
        temp_df["lai"] = temp_df["lai"].fillna(-1)

        mask_cloud = temp_df["cloud_fraction"] < FILTERS["cloud_max"]
        mask_aerosol = temp_df["aerosol_fraction"] < FILTERS["aerosol_max"]
        mask_lai = temp_df["lai"] >= FILTERS["lai_min"]

        filtered_df = merged[has_modis & mask_cloud & mask_aerosol & mask_lai].copy()
        if "quality_flag_x" in filtered_df.columns:
            filtered_df = filtered_df.rename(columns={"quality_flag_x": "quality_flag"})
        return filtered_df

    def report_match_rate(self) -> tuple[float, pd.DataFrame]:
        """Return the overall central match rate and a per-year summary."""
        if not self.match_diagnostics:
            return 0.0, pd.DataFrame()
        diag = pd.DataFrame(self.match_diagnostics)
        summary = (
            diag.groupby("year")
            .agg({
                "batch_sif_observations": "sum",
                "batch_central_matches": "sum",
            })
            .reset_index()
        )
        summary["central_match_rate"] = (
            summary["batch_central_matches"] / summary["batch_sif_observations"]
        )
        total_match = int(summary["batch_central_matches"].sum())
        total_obs = int(summary["batch_sif_observations"].sum())
        overall_rate = total_match / total_obs if total_obs else 0.0
        return overall_rate, summary

    def write_unmatched_diagnostics(self, output_path: Path) -> pd.DataFrame:
        """Write aggregated counts of SIF observations with no central MODIS match."""
        if not self.unmatched_cell_counts:
            df = pd.DataFrame(columns=["year", "lat_id", "lon_id", "unmatched_observations"])
        else:
            records = [
                {"year": y, "lat_id": la, "lon_id": lo, "unmatched_observations": n}
                for (y, la, lo), n in self.unmatched_cell_counts.items()
            ]
            df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine=PARQUET_ENGINE, index=False)
        return df


def _parse_date_from_filename(sif_file: Path) -> pd.Timestamp | None:
    """Extract the observation date from an OCO-2 LtSIF filename."""
    try:
        date_str = sif_file.name.split("_")[2]
        return pd.to_datetime(date_str, format="%y%m%d")
    except Exception:
        return None


def _parse_build_tag(sif_file: Path) -> str:
    """Return the OCO-2 product-build tag from the filename for tie-breaking."""
    parts = sif_file.name.split("_")
    if len(parts) >= 4:
        return parts[3]
    return ""


def deduplicate_files_by_date(files: list[Path]) -> list[Path]:
    """Keep exactly one file per observation date.

    If multiple product builds exist for the same date, the lexicographically
    largest build tag is chosen.  This rule is deterministic and reproducible.
    """
    by_date: dict[pd.Timestamp, list[Path]] = {}
    for f in files:
        d = _parse_date_from_filename(f)
        if d is not None:
            by_date.setdefault(d, []).append(f)

    selected: list[Path] = []
    for d in sorted(by_date):
        builds = by_date[d]
        if len(builds) == 1:
            selected.append(builds[0])
        else:
            chosen = max(builds, key=lambda p: _parse_build_tag(p))
            selected.append(chosen)
            print(f"[DEDUP] {d.date()}: kept {chosen.name} out of {len(builds)} builds")
    return selected


def process_single_file_raw(sif_file: Path) -> pd.DataFrame | None:
    """Parse one OCO-2 LtSIF NetCDF into a DataFrame of valid observations."""
    try:
        with xr.open_dataset(sif_file) as ds:
            if "Latitude" not in ds:
                return None
            lat = ds["Latitude"].values
            lon = ds["Longitude"].values

            sif_740 = ds["Daily_SIF_740nm"].values.flatten()
            sif_757 = ds["Daily_SIF_757nm"].values.flatten()
            sif_771 = ds["Daily_SIF_771nm"].values.flatten()

            if "SimplyGoodOrBadQualityFlag" in ds:
                q_mask = ds["SimplyGoodOrBadQualityFlag"].values.flatten() == 0
            elif "Quality_Flag" in ds:
                q_mask = ds["Quality_Flag"].values.flatten() == 0
            else:
                return None

            valid_idx = np.where(q_mask)[0]
            if len(valid_idx) == 0:
                return None

            date = _parse_date_from_filename(sif_file)
            if date is None:
                return None

            lat_vals = lat.flatten()[valid_idx]
            lon_vals = lon.flatten()[valid_idx]

            res = TARGET_RESOLUTION
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
    except Exception as e:
        print(f"[WARN] Could not process {sif_file}: {e}")
        return None


def aggregate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute region flags and aggregate raw SIF soundings to sums/counts."""
    flags = compute_geo_flags(df["latitude"].values, df["longitude"].values)
    if "lai" in df.columns:
        flags = apply_lai_flags(flags, df["lai"].values)
    df = df.copy()
    df["region_flags"] = flags.astype(np.int16)

    group_keys = ["date", "lat_id", "lon_id", "latitude", "longitude", "region_flags"]
    aggregated = (
        df.groupby(group_keys)
        .agg({
            "sif_740nm": "sum",
            "sif_757nm": "sum",
            "sif_771nm": "sum",
        })
        .reset_index()
    )
    count_df = df.groupby(group_keys).size().reset_index(name="count")
    aggregated = pd.merge(aggregated, count_df, on=group_keys)
    return aggregated


def _finalize_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sums/counts to means and compute the stress index."""
    df = df.copy()
    count = df["count"].astype(np.float64)
    for col in ["sif_740nm", "sif_757nm", "sif_771nm"]:
        df[col] = (df[col] / count).astype(np.float32)

    df["sif_stress_index"] = np.where(
        df["sif_771nm"] > SIF_MIN_THRESHOLD,
        df["sif_757nm"] / df["sif_771nm"],
        np.nan,
    ).astype(np.float32)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def process_year(
    year_files: Iterable[Path],
    year: int,
    modis: ModisHandler,
    pbar: tqdm,
) -> bool:
    """Process all SIF files for one year and write a yearly feather shard."""
    pbar.set_description(f"Year {year} [MODIS]")
    modis.load_year(year)
    pbar.set_description(f"Year {year} [OCO-2]")

    year_output = DATA_INTERIM / f"sif_aggregated_{year}.feather"
    all_chunks = []
    batch_data = []

    for sif_file in year_files:
        df = process_single_file_raw(sif_file)
        pbar.update(1)
        if df is not None:
            batch_data.append(df)
            if len(batch_data) >= FILES_SUBBATCH:
                filtered = modis.filter_batch(pd.concat(batch_data, ignore_index=True))
                if not filtered.empty:
                    all_chunks.append(aggregate_dataframe(filtered))
                batch_data = []

    if batch_data:
        filtered = modis.filter_batch(pd.concat(batch_data, ignore_index=True))
        if not filtered.empty:
            all_chunks.append(aggregate_dataframe(filtered))

    if all_chunks:
        full_year = pd.concat(all_chunks, ignore_index=True)
        final = (
            full_year.groupby(["date", "latitude", "longitude", "lat_id", "lon_id", "region_flags"])
            .agg({
                "sif_740nm": "sum",
                "sif_757nm": "sum",
                "sif_771nm": "sum",
                "count": "sum",
            })
            .reset_index()
        )
        final = _finalize_aggregates(final)
        final.to_feather(year_output)
        return True
    return False


def _clean_previous_artifacts() -> None:
    """Remove intermediate and final outputs from previous runs."""
    FILE_SIF_FINAL.unlink(missing_ok=True)
    FILE_SIF_CANDIDATE.unlink(missing_ok=True)
    for f in DATA_INTERIM.glob("sif_aggregated_*.feather"):
        f.unlink()
    model_dir = DATA_INTERIM / "SIF_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing sif_aggregated.feather and yearly shards before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed and exit.",
    )
    parser.add_argument(
        "--modis-path",
        type=Path,
        default=FILE_MODIS_PARQUET,
        help="Path to the MODIS parquet file to use (default: canonical)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_dirs()

    print("=" * 64)
    print("MAGNETO — OCO-2 SIF Upstream ETL")
    print("=" * 64)
    print(f"Input directory: {INPUT_DIR_OCO2}")
    print(f"Output file:     {FILE_SIF_FINAL}")
    print(f"Candidate file:  {FILE_SIF_CANDIDATE}")
    print(f"MODIS file:      {args.modis_path}")
    print(f"Target grid:     {TARGET_RESOLUTION}°")
    print(f"Years:           {START_YEAR}–{END_YEAR}")
    print(f"MODIS dilation:  {_MODIS_DILATION_MODE}")
    print(f"Central match min rate: {_CENTRAL_MATCH_MIN_RATE:.2%}")

    if not args.modis_path.exists():
        print(f"[FAIL] MODIS parquet not found: {args.modis_path}")
        return 1

    raw_files = sorted(INPUT_DIR_OCO2.glob(FILE_GLOB))
    deduped_files = deduplicate_files_by_date(raw_files)

    yearly_files: dict[int, list[Path]] = {}
    total_files = 0
    for f in deduped_files:
        year = _parse_date_from_filename(f).year
        if START_YEAR <= year <= END_YEAR:
            yearly_files.setdefault(year, []).append(f)
            total_files += 1

    print(f"OCO-2 files found: {len(raw_files):,}; after deduplication: {total_files:,}")

    if args.dry_run:
        for year, files in sorted(yearly_files.items()):
            print(f"  {year}: {len(files)} files")
        return 0

    if total_files == 0:
        print("[WARN] No OCO-2 files found; nothing to do.")
        return 0

    if args.clean:
        print("\n[INFO] Cleaning previous artifacts …")
        _clean_previous_artifacts()

    modis_handler = ModisHandler(args.modis_path)
    with tqdm(total=total_files, unit="file") as pbar:
        for year in sorted(yearly_files.keys()):
            process_year(yearly_files[year], year, modis_handler, pbar)

    overall_rate, match_summary = modis_handler.report_match_rate()
    print("\n[INFO] MODIS central-match diagnostics:")
    print(match_summary.to_string(index=False))
    print(f"Overall central match rate: {overall_rate:.4%}")

    if overall_rate < _CENTRAL_MATCH_MIN_RATE:
        diag_path = (
            PROJECT_ROOT
            / "results"
            / "etl_promotion_audits"
            / "sif_modis_unmatched_cells.parquet"
        )
        unmatched_df = modis_handler.write_unmatched_diagnostics(diag_path)
        print(
            f"\n[STOP] Central MODIS match rate {overall_rate:.4%} is below the "
            f"configured minimum {_CENTRAL_MATCH_MIN_RATE:.2%}."
        )
        print(f"       Unmatched-cell diagnostics written to {diag_path}")
        print(f"       Unique unmatched (year, cell) combinations: {len(unmatched_df):,}")
        print("       Review the diagnostics above before enabling any nearest-cell fallback.")
        return 3

    print("\n[INFO] Merging yearly shards …")
    shard_files = sorted(DATA_INTERIM.glob("sif_aggregated_*.feather"))
    if not shard_files:
        print("[FAIL] No yearly shards were produced.")
        return 1

    final_table = feather.read_table(str(shard_files[0]))
    for shard in shard_files[1:]:
        final_table = pa.concat_tables([final_table, feather.read_table(str(shard))])

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    feather.write_feather(final_table, str(FILE_SIF_CANDIDATE))

    for shard in shard_files:
        shard.unlink()

    print(f"[OK] Saved candidate {final_table.num_rows:,} rows to {FILE_SIF_CANDIDATE}")

    spec = {
        "name": "sif_aggregated",
        "fmt": "feather",
        "required": ["date", "lat_id", "lon_id", "sif_771nm", "region_flags"],
        "key_cols": ["date", "lat_id", "lon_id"],
        "metric_cols": ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"],
        "grid_cols": ["lat_id", "lon_id"],
    }
    report = _promote.compare_and_promote(FILE_SIF_FINAL, FILE_SIF_CANDIDATE, spec)
    if not report["passed"]:
        print("[STOP] SIF candidate differs from canonical. Promotion blocked; operator review required.")
        return 2
    print(f"[OK] E03_sif_etl complete — canonical promoted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
