#!/usr/bin/env python3
"""E03b_sif_hierarchical_fallback.py — hierarchical SIF–MODIS matching audit.

Reads OCO-2 Lite SIF files and the conservatively regridded candidate MODIS
product, applies the hierarchical fallback rules requested for the MAGNETO
candidate audit, and writes:

* ``data/interim/sif_aggregated_candidate.feather`` — primary hierarchical-match
  candidate;
* ``data/interim/sif_aggregated_central_only_candidate.feather`` — central-only
  sensitivity candidate;
* ``results/etl_promotion_audits/sif_hierarchical_fallback_diagnostics.json``;
* ``results/etl_promotion_audits/sif_hierarchical_fallback_diagnostics.md``.

The script does **not** run downstream analysis (QC, harmonic, spline,
surrogates, bootstrap, tables or figures).

Fallback hierarchy
------------------
A. ``central_exact``: same MODIS composite date and same 0.5° cell.
B. ``temporal_same_cell``: same cell, nearest available MODIS composite date
   within ±8 days; earlier date wins ties.
C. ``spatial_same_date``: same date, nearest valid MODIS cell within 0.5°;
   orthogonal neighbours (distance 0.5°) are tried before diagonals
   (distance ≈0.707°).
D. Combined spatiotemporal fallback is **not** applied automatically.
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from datetime import timedelta
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

from _Common import PARQUET_ENGINE, PROJECT_ROOT, atomic_write  # noqa: E402

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

_RCFG = _OCFG["region_flags"]
_BIT_DEFS = {k: int(v) for k, v in _RCFG["bits"].items()}
_GEOBOXES = _RCFG["boxes"]
_POLAR_LAT_ABS = float(_RCFG["polar_lat_abs"])
_LAI_VEGETATION_THRESHOLD = float(_RCFG["lai_vegetation_threshold"])

FILE_MODIS_PARQUET = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_modis"]
FILE_SIF_FINAL = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_sif"]
DATA_INTERIM = PROJECT_ROOT / _PIPELINE_CFG["paths"]["data_interim"]

# Primary analytical input: central-only MODIS matches.
# Hierarchical candidate is retained as a sensitivity input only.
FILE_SIF_CENTRAL = DATA_INTERIM / "sif_aggregated_central_only_candidate.feather"
FILE_SIF_HIERARCHICAL = DATA_INTERIM / "sif_aggregated_hierarchical_candidate.feather"
REPORT_DIR = PROJECT_ROOT / "results" / "etl_promotion_audits"

# Spatial neighbour offsets in lat_id/lon_id units (0.01° per unit).
ORTHOGONAL_NEIGHBOURS = [
    (50, 0, 0.50, "N"),
    (-50, 0, 0.50, "S"),
    (0, 50, 0.50, "E"),
    (0, -50, 0.50, "W"),
]
DIAGONAL_NEIGHBOURS = [
    (50, 50, 0.7071067811865476, "NE"),
    (50, -50, 0.7071067811865476, "NW"),
    (-50, 50, 0.7071067811865476, "SE"),
    (-50, -50, 0.7071067811865476, "SW"),
]


class RegionFlag(IntFlag):
    """Bitwise region classification used during SIF aggregation."""

    NONE = 0
    SAA = _BIT_DEFS["SAA"]
    CONTROL_NORTH = _BIT_DEFS["CONTROL_NORTH"]
    POLAR = _BIT_DEFS["POLAR"]
    SAHARA = _BIT_DEFS["SAHARA"]
    LOW_LAI = _BIT_DEFS["LOW_LAI"]
    HIGH_LAI = _BIT_DEFS["HIGH_LAI"]


def _in_box(lats: np.ndarray, lons: np.ndarray, box: dict[str, float]) -> np.ndarray:
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


def _parse_date_from_filename(sif_file: Path) -> pd.Timestamp | None:
    try:
        date_str = sif_file.name.split("_")[2]
        return pd.to_datetime(date_str, format="%y%m%d")
    except Exception:
        return None


def _parse_build_tag(sif_file: Path) -> str:
    parts = sif_file.name.split("_")
    if len(parts) >= 4:
        return parts[3]
    return ""


def deduplicate_files_by_date(files: list[Path]) -> list[Path]:
    """Keep exactly one file per observation date, largest build tag tie-break."""
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


class ModisIndex:
    """Memory-efficient lookup structures for hierarchical SIF–MODIS matching.

    Stores the full MODIS candidate as sorted NumPy arrays and uses int64
    composite keys for binary-search lookup.  This avoids the heavy memory
    overhead of a pandas MultiIndex and of per-date Python sets.
    """

    # Encoding: lat_id/lon_id are centidegree integers in [-9000, 9000] and
    # [-18000, 18000].  Offset them to unsigned 16-bit values before packing.
    _LAT_OFFSET = 18_000
    _LON_OFFSET = 36_000

    def __init__(self, parquet_path: Path, year: int | None = None):
        self.path = parquet_path
        self.year = year
        self.keys: np.ndarray = np.array([], dtype=np.int64)
        self.cloud: np.ndarray = np.array([], dtype=np.float32)
        self.aerosol: np.ndarray = np.array([], dtype=np.float32)
        self.lai: np.ndarray = np.array([], dtype=np.float32)
        self.quality_flag: np.ndarray = np.array([], dtype=np.float32)
        self.cell_dates: dict[tuple[int, int], np.ndarray] = {}
        self.date_cells: dict[int, np.ndarray] = {}
        self._load()

    @staticmethod
    def _encode_cell(lat_id: int, lon_id: int) -> int:
        lat_u = int(lat_id) + ModisIndex._LAT_OFFSET
        lon_u = int(lon_id) + ModisIndex._LON_OFFSET
        return (lat_u << 16) | lon_u

    @staticmethod
    def _decode_cell(cell_key: int) -> tuple[int, int]:
        return (int(cell_key >> 16) - ModisIndex._LAT_OFFSET,
                int(cell_key & 0xFFFF) - ModisIndex._LON_OFFSET)

    @staticmethod
    def _make_key(date_ord: int, lat_id: int, lon_id: int) -> int:
        return (int(date_ord) << 32) | ModisIndex._encode_cell(lat_id, lon_id)

    def _load(self) -> None:
        print(f"[INFO] Loading MODIS index from {self.path}")
        if self.year is not None:
            print(f"[INFO] Restricting MODIS index to years {self.year - 1}-{self.year + 1}")
        cols = ["date", "lat_id", "lon_id", "cloud_fraction", "aerosol_fraction", "lai", "quality_flag"]

        if self.year is None:
            df = pd.read_parquet(self.path, engine=PARQUET_ENGINE, columns=cols)
        else:
            # Load only the target year plus adjacent years to cover temporal
            # fallback at year boundaries while keeping memory low.
            start = pd.Timestamp(f"{self.year - 1}-01-01")
            end = pd.Timestamp(f"{self.year + 1}-12-31")
            filters = [
                [("date", ">=", start), ("date", "<=", end)],
            ]
            df = pd.read_parquet(
                self.path, engine=PARQUET_ENGINE, columns=cols, filters=filters
            )

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["date_ord"] = df["date"].apply(lambda d: d.toordinal()).astype(np.int32)

        has_any = (
            df["cloud_fraction"].notna()
            | df["aerosol_fraction"].notna()
            | df["lai"].notna()
        )
        df = df[has_any].copy()

        # Ensure native types before moving to NumPy.
        df["lat_id"] = df["lat_id"].astype(np.int16)
        df["lon_id"] = df["lon_id"].astype(np.int16)
        df["cloud_fraction"] = df["cloud_fraction"].astype(np.float32)
        df["aerosol_fraction"] = df["aerosol_fraction"].astype(np.float32)
        df["lai"] = df["lai"].astype(np.float32)
        df["quality_flag"] = df["quality_flag"].astype(np.float32)

        lat_cells = df["lat_id"].astype(np.int64).values + self._LAT_OFFSET
        lon_cells = df["lon_id"].astype(np.int64).values + self._LON_OFFSET
        df["cell_key"] = (lat_cells << 16) | lon_cells
        df["key"] = (df["date_ord"].astype(np.int64).values << 32) | df["cell_key"].values
        df.sort_values("key", inplace=True)

        self.keys = df["key"].values.astype(np.int64)
        self.cloud = df["cloud_fraction"].values
        self.aerosol = df["aerosol_fraction"].values
        self.lai = df["lai"].values
        self.quality_flag = df["quality_flag"].values

        # Compact per-cell date lists (int32 ordinals).
        for (lat_id, lon_id), grp in df.groupby(["lat_id", "lon_id"]):
            self.cell_dates[(int(lat_id), int(lon_id))] = np.sort(
                grp["date_ord"].unique().astype(np.int32)
            )

        # Compact per-date cell arrays (int64 cell keys, sorted).
        for date_ord, grp in df.groupby("date_ord"):
            self.date_cells[int(date_ord)] = np.sort(
                grp["cell_key"].unique().astype(np.int64)
            )

        # Release the temporary dataframe.
        del df
        gc.collect()

        print(f"[INFO] MODIS index ready: {len(self.keys):,} rows, "
              f"{len(self.cell_dates):,} cells, {len(self.date_cells):,} dates")

    def _lookup(self, date_ord: int, lat_id: int, lon_id: int) -> dict[str, float] | None:
        key = self._make_key(date_ord, lat_id, lon_id)
        idx = np.searchsorted(self.keys, key)
        if idx < len(self.keys) and self.keys[idx] == key:
            return {
                "cloud_fraction": float(self.cloud[idx]),
                "aerosol_fraction": float(self.aerosol[idx]),
                "lai": float(self.lai[idx]),
                "quality_flag": float(self.quality_flag[idx]),
            }
        return None

    def lookup(self, date: pd.Timestamp, lat_id: int, lon_id: int) -> dict[str, float] | None:
        """Return the MODIS row for an exact (date, cell) or None."""
        return self._lookup(pd.Timestamp(date).toordinal(), int(lat_id), int(lon_id))

    def temporal_fallback(
        self,
        target_date: pd.Timestamp,
        lat_id: int,
        lon_id: int,
        max_offset_days: int = 8,
    ) -> tuple[dict[str, float] | None, int]:
        """Find nearest available MODIS row for the same cell within max_offset_days."""
        dates = self.cell_dates.get((int(lat_id), int(lon_id)))
        if dates is None or len(dates) == 0:
            return None, 0

        target_ord = pd.Timestamp(target_date).toordinal()
        day_offsets = dates.astype(np.int64) - target_ord
        mask = (np.abs(day_offsets) > 0) & (np.abs(day_offsets) <= max_offset_days)
        if not mask.any():
            return None, 0

        valid_offsets = day_offsets[mask]
        valid_dates = dates[mask]
        min_abs = int(np.abs(valid_offsets).min())
        candidates = valid_dates[np.abs(valid_offsets) == min_abs]
        # Earlier date wins ties.
        chosen_ord = int(candidates.min())
        offset = chosen_ord - target_ord
        row = self._lookup(chosen_ord, lat_id, lon_id)
        return row, offset

    def spatial_fallback(
        self,
        target_date: pd.Timestamp,
        lat_id: int,
        lon_id: int,
    ) -> tuple[dict[str, float] | None, int, int, float, str]:
        """Find nearest valid MODIS cell for the same date. Orthogonals before diagonals."""
        date_ord = pd.Timestamp(target_date).toordinal()
        cells = self.date_cells.get(date_ord)
        if cells is None or len(cells) == 0:
            return None, 0, 0, 0.0, ""

        def _check(dlat: int, dlon: int, dist: float, direction: str):
            key = self._encode_cell(lat_id + dlat, lon_id + dlon)
            idx = np.searchsorted(cells, key)
            if idx < len(cells) and cells[idx] == key:
                source_lat, source_lon = self._decode_cell(key)
                row = self._lookup(date_ord, source_lat, source_lon)
                if row is not None:
                    return row, dlat, dlon, dist, direction
            return None

        for dlat, dlon, dist, direction in ORTHOGONAL_NEIGHBOURS:
            res = _check(dlat, dlon, dist, direction)
            if res is not None:
                return res

        for dlat, dlon, dist, direction in DIAGONAL_NEIGHBOURS:
            res = _check(dlat, dlon, dist, direction)
            if res is not None:
                return res

        return None, 0, 0, 0.0, ""


def _modis_period_date(target_dates: pd.Series) -> pd.Series:
    """Map daily dates to MODIS 8-day compositing period start dates.

    Matches the logic in E01_modis_etl.py: period starts at doy 1, 9, 17, ...
    """
    years = target_dates.dt.year
    start_of_years = pd.to_datetime(years.astype(str) + "-01-01")
    doy = target_dates.dt.dayofyear - 1
    period_start_doy = (doy // 8) * 8
    return start_of_years + pd.to_timedelta(period_start_doy, unit="D")


def aggregate_raw_soundings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw SIF soundings to daily sums/counts per cell."""
    flags = compute_geo_flags(df["latitude"].values, df["longitude"].values)
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
    aggregated["modis_date_key"] = _modis_period_date(aggregated["date"])
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


def apply_modis_and_fallback(
    agg: pd.DataFrame,
    modis: ModisIndex,
) -> pd.DataFrame:
    """Apply hierarchical fallback to each aggregate row and attach metadata."""
    n = len(agg)
    modes = np.full(n, "unmatched", dtype=object)
    temporal_offsets = np.zeros(n, dtype=np.int16)
    spatial_offsets_lat = np.zeros(n, dtype=np.int16)
    spatial_offsets_lon = np.zeros(n, dtype=np.int16)
    spatial_distances = np.zeros(n, dtype=np.float32)
    source_dates = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    source_lat_ids = np.zeros(n, dtype=np.int16)
    source_lon_ids = np.zeros(n, dtype=np.int16)
    cloud = np.full(n, np.nan, dtype=np.float32)
    aerosol = np.full(n, np.nan, dtype=np.float32)
    lai = np.full(n, np.nan, dtype=np.float32)
    quality_flag = np.full(n, np.nan, dtype=np.float32)

    for i in tqdm(range(n), desc="SIF–MODIS fallback", unit="cell"):
        target_date = agg.iloc[i]["modis_date_key"]
        lat_id = int(agg.iloc[i]["lat_id"])
        lon_id = int(agg.iloc[i]["lon_id"])

        # A. central_exact
        row = modis.lookup(target_date, lat_id, lon_id)
        if row is not None:
            modes[i] = "central_exact"
            source_dates[i] = target_date
            source_lat_ids[i] = lat_id
            source_lon_ids[i] = lon_id
            cloud[i] = row["cloud_fraction"]
            aerosol[i] = row["aerosol_fraction"]
            lai[i] = row["lai"]
            quality_flag[i] = row.get("quality_flag", np.nan)
            continue

        # B. temporal_same_cell
        row, offset = modis.temporal_fallback(target_date, lat_id, lon_id)
        if row is not None:
            modes[i] = "temporal_same_cell"
            temporal_offsets[i] = offset
            source_dates[i] = target_date + timedelta(days=int(offset))
            source_lat_ids[i] = lat_id
            source_lon_ids[i] = lon_id
            cloud[i] = row["cloud_fraction"]
            aerosol[i] = row["aerosol_fraction"]
            lai[i] = row["lai"]
            quality_flag[i] = row.get("quality_flag", np.nan)
            continue

        # C. spatial_same_date
        row, dlat, dlon, dist, direction = modis.spatial_fallback(target_date, lat_id, lon_id)
        if row is not None:
            modes[i] = "spatial_same_date"
            spatial_offsets_lat[i] = dlat
            spatial_offsets_lon[i] = dlon
            spatial_distances[i] = dist
            source_dates[i] = target_date
            source_lat_ids[i] = lat_id + dlat
            source_lon_ids[i] = lon_id + dlon
            cloud[i] = row["cloud_fraction"]
            aerosol[i] = row["aerosol_fraction"]
            lai[i] = row["lai"]
            quality_flag[i] = row.get("quality_flag", np.nan)
            continue

    agg = agg.copy()
    agg["modis_match_mode"] = modes
    agg["temporal_offset_days"] = temporal_offsets
    agg["spatial_offset_lat_id"] = spatial_offsets_lat
    agg["spatial_offset_lon_id"] = spatial_offsets_lon
    agg["spatial_distance"] = spatial_distances
    agg["source_modis_date"] = source_dates
    agg["source_modis_lat_id"] = source_lat_ids
    agg["source_modis_lon_id"] = source_lon_ids
    agg["cloud_fraction"] = cloud
    agg["aerosol_fraction"] = aerosol
    agg["lai"] = lai
    agg["quality_flag"] = quality_flag
    return agg


def _apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MODIS-based QC filters and LAI flags."""
    df = df.copy()
    df["region_flags"] = apply_lai_flags(df["region_flags"].values, df["lai"].values)

    has_modis = df["cloud_fraction"].notna() | df["aerosol_fraction"].notna() | df["lai"].notna()
    mask_cloud = df["cloud_fraction"].fillna(999) < FILTERS["cloud_max"]
    mask_aerosol = df["aerosol_fraction"].fillna(999) < FILTERS["aerosol_max"]
    mask_lai = df["lai"].fillna(-1) >= FILTERS["lai_min"]
    return df[has_modis & mask_cloud & mask_aerosol & mask_lai].copy()


def process_year(
    year_files: Iterable[Path],
    year: int,
    modis: ModisIndex,
    pbar: tqdm,
) -> pd.DataFrame | None:
    """Process all SIF files for one year and return aggregated rows with fallback."""
    pbar.set_description(f"Year {year} [OCO-2]")
    all_chunks = []
    batch_data = []

    for sif_file in year_files:
        df = process_single_file_raw(sif_file)
        pbar.update(1)
        if df is not None:
            batch_data.append(df)
            if len(batch_data) >= FILES_SUBBATCH:
                all_chunks.append(aggregate_raw_soundings(pd.concat(batch_data, ignore_index=True)))
                batch_data = []

    if batch_data:
        all_chunks.append(aggregate_raw_soundings(pd.concat(batch_data, ignore_index=True)))

    if not all_chunks:
        return None

    full_year = pd.concat(all_chunks, ignore_index=True)
    # Aggregate again in case a cell spanned multiple batches.
    full_year = (
        full_year.groupby(["date", "lat_id", "lon_id", "latitude", "longitude", "modis_date_key"])
        .agg({
            "sif_740nm": "sum",
            "sif_757nm": "sum",
            "sif_771nm": "sum",
            "count": "sum",
            "region_flags": "first",
        })
        .reset_index()
    )

    pbar.set_description(f"Year {year} [fallback]")
    full_year = apply_modis_and_fallback(full_year, modis)
    return full_year


def _scenario_label(flags: pd.Series) -> pd.Series:
    """Assign a readable scenario label from region flags."""
    vals = flags.astype(int)
    out = pd.Series(index=flags.index, dtype=object)
    out[(vals & int(RegionFlag.POLAR)).astype(bool)] = "polar"
    out[(vals & int(RegionFlag.SAHARA)).astype(bool)] = "sahara"
    out[(vals & int(RegionFlag.SAA)).astype(bool) & out.isna()] = "saa"
    out[(vals & int(RegionFlag.CONTROL_NORTH)).astype(bool) & out.isna()] = "control_north"
    out[(vals & int(RegionFlag.LOW_LAI)).astype(bool) & out.isna()] = "low_lai"
    out[(vals & int(RegionFlag.HIGH_LAI)).astype(bool) & out.isna()] = "high_lai"
    out[out.isna()] = "other"
    return out


def _numeric_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def _weighted_and_cell_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return sounding (count-weighted) and cell counts for a DataFrame."""
    return {
        "soundings": int(df["count"].sum()),
        "cells": int(len(df)),
    }


def _apply_filters_stepwise(df: pd.DataFrame) -> list[dict]:
    """Apply cloud, aerosol and LAI thresholds sequentially and record counts."""
    steps = []
    steps.append({"step": "matched_pre_qc", **_weighted_and_cell_counts(df)})

    cloud_ok = df["cloud_fraction"].notna() & (df["cloud_fraction"] < FILTERS["cloud_max"])
    after_cloud = df[cloud_ok]
    steps.append({"step": "after_cloud_filter", **_weighted_and_cell_counts(after_cloud)})

    aerosol_ok = after_cloud["aerosol_fraction"].notna() & (after_cloud["aerosol_fraction"] < FILTERS["aerosol_max"])
    after_aerosol = after_cloud[aerosol_ok]
    steps.append({"step": "after_aerosol_filter", **_weighted_and_cell_counts(after_aerosol)})

    lai_ok = after_aerosol["lai"].notna() & (after_aerosol["lai"] >= FILTERS["lai_min"])
    after_lai = after_aerosol[lai_ok]
    steps.append({"step": "after_lai_filter", **_weighted_and_cell_counts(after_lai)})

    return steps, after_lai


def build_diagnostics(
    full_hierarchical: pd.DataFrame,
    central_only_qc: pd.DataFrame,
    hierarchical_qc: pd.DataFrame,
    raw_unmatched_path: Path | None,
) -> dict:
    """Build diagnostic summary from the full pre-QC fallback output.

    ``full_hierarchical`` must contain all aggregate rows, including unmatched.
    QC losses are computed independently so that the unmatched fraction is not
    hidden by the QC filter.
    """
    full = full_hierarchical.copy()
    full["weight"] = full["count"]

    total_weight = int(full["weight"].sum())
    total_cells = int(len(full))

    # Pre-QC match-mode breakdown.
    mode_weights = (
        full.groupby("modis_match_mode")["weight"].sum()
        .reindex(["central_exact", "temporal_same_cell", "spatial_same_date", "unmatched"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    mode_cells = (
        full["modis_match_mode"]
        .value_counts()
        .reindex(["central_exact", "temporal_same_cell", "spatial_same_date", "unmatched"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    # QC flow on matched rows (central_exact + temporal + spatial).
    matched = full[full["modis_match_mode"] != "unmatched"].copy()
    hier_flow, _ = _apply_filters_stepwise(matched)
    cen_flow, _ = _apply_filters_stepwise(
        full[full["modis_match_mode"] == "central_exact"].copy()
    )

    # Final retained counts from the actual QC outputs (should match the last step).
    final_counts = {
        "hierarchical": _weighted_and_cell_counts(hierarchical_qc),
        "central_only": _weighted_and_cell_counts(central_only_qc),
    }

    # Yearly breakdown (pre-QC).
    full["year"] = pd.to_datetime(full["date"]).dt.year
    yearly = []
    for year, grp in full.groupby("year"):
        w = grp["weight"].sum()
        row = {
            "year": int(year),
            "total_soundings": int(w),
            "total_cells": int(len(grp)),
        }
        for mode in ["central_exact", "temporal_same_cell", "spatial_same_date", "unmatched"]:
            mw = grp[grp["modis_match_mode"] == mode]["weight"].sum()
            mc = (grp["modis_match_mode"] == mode).sum()
            row[f"{mode}_soundings"] = int(mw)
            row[f"{mode}_cells"] = int(mc)
            row[f"{mode}_fraction_soundings"] = float(mw / w) if w else 0.0
            row[f"{mode}_fraction_cells"] = float(mc / len(grp)) if len(grp) else 0.0
        yearly.append(row)

    # Scenario breakdown (pre-QC).
    full["scenario"] = _scenario_label(full["region_flags"])
    scenario = []
    for scen, grp in full.groupby("scenario"):
        w = grp["weight"].sum()
        scenario.append({
            "scenario": scen,
            "total_soundings": int(w),
            "total_cells": int(len(grp)),
            "central_exact_fraction_soundings": float(
                grp[grp["modis_match_mode"] == "central_exact"]["weight"].sum() / w
            ) if w else 0.0,
            "unmatched_fraction_soundings": float(
                grp[grp["modis_match_mode"] == "unmatched"]["weight"].sum() / w
            ) if w else 0.0,
        })

    # MODIS variable distributions by match mode (pre-QC matched rows).
    dists = {}
    for mode in ["central_exact", "temporal_same_cell", "spatial_same_date"]:
        sub = full[full["modis_match_mode"] == mode]
        dists[mode] = {
            "lai": _numeric_summary(sub["lai"]),
            "cloud_fraction": _numeric_summary(sub["cloud_fraction"]),
            "aerosol_fraction": _numeric_summary(sub["aerosol_fraction"]),
            "quality_flag": _numeric_summary(sub["quality_flag"]),
        }

    # Geographic unmatched fraction map (0.5° cells pooled to 5° grid).
    full["lat_bin"] = (full["latitude"] / 5).astype(int) * 5
    full["lon_bin"] = (full["longitude"] / 5).astype(int) * 5
    full["unmatched_weight"] = np.where(
        full["modis_match_mode"] == "unmatched", full["weight"], 0
    )
    geo = (
        full.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            total_soundings=("weight", "sum"),
            unmatched_soundings=("unmatched_weight", "sum"),
        )
    )
    geo["total_soundings"] = geo["total_soundings"].astype(int)
    geo["unmatched_soundings"] = geo["unmatched_soundings"].astype(int)
    geo["unmatched_fraction"] = geo["unmatched_soundings"] / geo["total_soundings"]
    geo = geo.sort_values("unmatched_fraction", ascending=False)

    report = {
        "total_aggregate_rows": total_cells,
        "total_soundings": total_weight,
        "mode_soundings": mode_weights,
        "mode_fractions_soundings": {k: float(v / total_weight) for k, v in mode_weights.items()},
        "mode_cells": mode_cells,
        "mode_fractions_cells": {k: float(v / total_cells) for k, v in mode_cells.items()},
        "qc_flow_soundings": {
            "hierarchical": {s["step"]: s["soundings"] for s in hier_flow},
            "central_only": {s["step"]: s["soundings"] for s in cen_flow},
        },
        "qc_flow_cells": {
            "hierarchical": {s["step"]: s["cells"] for s in hier_flow},
            "central_only": {s["step"]: s["cells"] for s in cen_flow},
        },
        "final_retained_counts": final_counts,
        "central_only_rows": final_counts["central_only"]["cells"],
        "central_only_soundings": final_counts["central_only"]["soundings"],
        "hierarchical_rows": final_counts["hierarchical"]["cells"],
        "hierarchical_soundings": final_counts["hierarchical"]["soundings"],
        "yearly": yearly,
        "scenario": scenario,
        "distributions_by_mode": dists,
        "geographic_unmatched_top": geo.head(20).to_dict(orient="records"),
    }

    if raw_unmatched_path and raw_unmatched_path.exists():
        raw_unmatched = pd.read_parquet(raw_unmatched_path)
        report["raw_unmatched_soundings"] = int(raw_unmatched["unmatched_observations"].sum())
    return report


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# MAGNETO hierarchical SIF–MODIS fallback audit",
        "",
        "## Overall match summary",
        "",
        f"- Total aggregate rows (date × 0.5° cell): {report['total_aggregate_rows']:,}",
        f"- Total SIF soundings represented: {report['total_soundings']:,}",
        "",
        "### Sounding-level match fractions (before MODIS QC)",
        "",
        "| Mode | Soundings | Fraction |",
        "|------|----------:|---------:|",
    ]
    for mode, frac in report["mode_fractions_soundings"].items():
        n = report["mode_soundings"][mode]
        lines.append(f"| {mode} | {n:,} | {frac:.4%} |")

    lines.extend([
        "",
        "### Cell-level match fractions (before MODIS QC)",
        "",
        "| Mode | Cells | Fraction |",
        "|------|------:|---------:|",
    ])
    for mode, frac in report["mode_fractions_cells"].items():
        n = report["mode_cells"][mode]
        lines.append(f"| {mode} | {n:,} | {frac:.4%} |")

    lines.extend([
        "",
        "## QC flow (matched rows only)",
        "",
        "Counts are shown as **soundings (cells)**.",
        "",
        "| Stage | Hierarchical | Central-only |",
        "|-------|-------------:|-------------:|",
    ])
    hier_flow_s = report["qc_flow_soundings"]["hierarchical"]
    cen_flow_s = report["qc_flow_soundings"]["central_only"]
    hier_flow_c = report["qc_flow_cells"]["hierarchical"]
    cen_flow_c = report["qc_flow_cells"]["central_only"]
    for step in ["matched_pre_qc", "after_cloud_filter", "after_aerosol_filter", "after_lai_filter"]:
        lines.append(
            f"| {step} | {hier_flow_s[step]:,} ({hier_flow_c[step]:,}) | "
            f"{cen_flow_s[step]:,} ({cen_flow_c[step]:,}) |"
        )

    lines.extend([
        "",
        "### Final retained candidate counts",
        "",
        f"- Hierarchical candidate: {report['hierarchical_rows']:,} rows, {report['hierarchical_soundings']:,} soundings",
        f"- Central-only candidate: {report['central_only_rows']:,} rows, {report['central_only_soundings']:,} soundings",
        "",
        "## Yearly breakdown",
        "",
        "| Year | Soundings | Cells | Central % | Temporal % | Spatial % | Unmatched % |",
        "|------|----------:|------:|----------:|-----------:|----------:|------------:|",
    ])
    for row in report["yearly"]:
        lines.append(
            f"| {row['year']} | {row['total_soundings']:,} | {row['total_cells']:,} | "
            f"{row['central_exact_fraction_soundings']:.2%} | "
            f"{row['temporal_same_cell_fraction_soundings']:.2%} | "
            f"{row['spatial_same_date_fraction_soundings']:.2%} | "
            f"{row['unmatched_fraction_soundings']:.2%} |"
        )

    lines.extend([
        "",
        "## Scenario-level unmatched fraction",
        "",
        "| Scenario | Soundings | Unmatched % |",
        "|----------|----------:|------------:|",
    ])
    for row in report["scenario"]:
        lines.append(
            f"| {row['scenario']} | {row['total_soundings']:,} | {row['unmatched_fraction_soundings']:.2%} |"
        )

    lines.extend([
        "",
        "## MODIS variable distributions by match mode",
        "",
    ])
    for mode, dists in report["distributions_by_mode"].items():
        lines.extend([f"### {mode}", "", "| Variable | N | Mean | Min | Median | Max |", "|----------|---|------|-----|--------|-----|"])
        for var, stats in dists.items():
            if stats["n"] == 0:
                lines.append(f"| {var} | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {var} | {stats['n']:,} | {stats['mean']:.4g} | {stats['min']:.4g} | "
                    f"{stats['q50']:.4g} | {stats['max']:.4g} |"
                )
        lines.append("")

    lines.extend([
        "## Geographic hotspots of unmatched soundings (5° bins)",
        "",
        "| Lat bin | Lon bin | Total soundings | Unmatched soundings | Fraction |",
        "|---------|---------|----------------:|--------------------:|---------:|",
    ])
    for row in report["geographic_unmatched_top"]:
        lines.append(
            f"| {row['lat_bin']} | {row['lon_bin']} | {int(row['total_soundings']):,} | "
            f"{int(row['unmatched_soundings']):,} | {row['unmatched_fraction']:.2%} |"
        )

    lines.extend([
        "",
        "## Conclusions",
        "",
        "- Combined spatiotemporal fallback was **not** applied automatically.",
        "- If the remaining unmatched fraction is unacceptable, a combined fallback",
        "  (nearest cell within ±8 days) can be evaluated as a separate sensitivity.",
        "- A full downstream recalculation is required if either candidate is promoted.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modis-path",
        type=Path,
        default=FILE_MODIS_PARQUET,
        help="Path to the MODIS parquet file to use",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help="First OCO-2 year to process (for testing)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help="Last OCO-2 year to process (inclusive, for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 64)
    print("MAGNETO — SIF hierarchical fallback audit")
    print("=" * 64)
    print(f"MODIS file: {args.modis_path}")
    print(f"Output hierarchical: {FILE_SIF_HIERARCHICAL}")
    print(f"Output central-only: {FILE_SIF_CENTRAL}")

    if not args.modis_path.exists():
        print(f"[FAIL] MODIS parquet not found: {args.modis_path}")
        return 1

    raw_files = sorted(INPUT_DIR_OCO2.glob(FILE_GLOB))
    deduped_files = deduplicate_files_by_date(raw_files)

    start_year = max(args.start_year, START_YEAR)
    end_year = min(args.end_year, END_YEAR)

    yearly_files: dict[int, list[Path]] = {}
    total_files = 0
    for f in deduped_files:
        year = _parse_date_from_filename(f).year
        if start_year <= year <= end_year:
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

    year_chunks = []
    with tqdm(total=total_files, unit="file") as pbar:
        for year in sorted(yearly_files.keys()):
            # Build a year-specific MODIS index so that memory stays bounded.
            modis_index = ModisIndex(args.modis_path, year=year)
            chunk = process_year(yearly_files[year], year, modis_index, pbar)
            if chunk is not None:
                year_chunks.append(chunk)
            del modis_index
            gc.collect()

    if not year_chunks:
        print("[FAIL] No SIF rows were produced.")
        return 1

    full_hierarchical = pd.concat(year_chunks, ignore_index=True)

    # Central-only candidate: rows with central_exact match only, then QC.
    central_only_raw = full_hierarchical[full_hierarchical["modis_match_mode"] == "central_exact"].copy()
    central_only_qc = _apply_qc(central_only_raw)

    # Hierarchical candidate: all matched rows (A+B+C), then QC.
    hierarchical_matched = full_hierarchical[full_hierarchical["modis_match_mode"] != "unmatched"].copy()
    hierarchical_qc = _apply_qc(hierarchical_matched)

    # Finalize means and stress index.
    central_only_qc = _finalize_aggregates(central_only_qc)
    hierarchical_qc = _finalize_aggregates(hierarchical_qc)

    # Diagnostics must run before dropping MODIS helper columns.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_unmatched_path = REPORT_DIR / "sif_modis_unmatched_cells.parquet"
    diag = build_diagnostics(full_hierarchical, central_only_qc, hierarchical_qc, raw_unmatched_path)

    # Drop helper columns not needed for downstream, but retain the matched
    # MODIS variables so that 01_build_qc.py does not perform an independent
    # nearest-date rematch.
    keep_cols = [
        "date", "lat_id", "lon_id", "latitude", "longitude", "region_flags",
        "sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index", "count",
        "modis_match_mode", "lai", "cloud_fraction", "aerosol_fraction", "quality_flag",
        "temporal_offset_days",
        "spatial_offset_lat_id", "spatial_offset_lon_id", "spatial_distance",
        "source_modis_date", "source_modis_lat_id", "source_modis_lon_id",
    ]
    hierarchical_qc = hierarchical_qc[[c for c in keep_cols if c in hierarchical_qc.columns]]
    # For central-only candidate the provenance columns are already populated
    # with source == target, offsets == 0, and mode == central_exact.  Retain
    # them so that downstream QC can assert key consistency and avoid any
    # implicit rematch.
    central_keep = [c for c in keep_cols if c in central_only_qc.columns]
    central_only_qc = central_only_qc[central_keep]

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    hierarchical_qc.to_feather(FILE_SIF_HIERARCHICAL)
    central_only_qc.to_feather(FILE_SIF_CENTRAL)

    print(f"[OK] Hierarchical candidate: {len(hierarchical_qc):,} rows, {int(hierarchical_qc['count'].sum()):,} soundings")
    print(f"[OK] Central-only candidate: {len(central_only_qc):,} rows, {int(central_only_qc['count'].sum()):,} soundings")

    diag_path_json = REPORT_DIR / "sif_hierarchical_fallback_diagnostics.json"
    diag_path_md = REPORT_DIR / "sif_hierarchical_fallback_diagnostics.md"
    atomic_write(diag, diag_path_json, fmt="json")
    _write_markdown(diag, diag_path_md)
    print(f"[OK] Diagnostics written to {diag_path_json} and {diag_path_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
