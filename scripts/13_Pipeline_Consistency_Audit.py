#!/usr/bin/env python3
"""
MAGNETO Pipeline Consistency Audit

Comprehensive integrity audit of pipeline interim outputs. Validates each
interim file that exists, reports per-check status/details/measured values,
saves JSON and Markdown reports, and exits non-zero if any critical check fails.

Critical checks are the ERA5 checks; their failure causes exit code 1.
"""

from __future__ import annotations

import calendar
import gc
import json
import re
import sys
from collections import Counter
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

from _Common import Config


class PipelineIntegrityAudit:
    """Run consistency checks over MAGNETO interim outputs."""

    ANALYSIS_START = pd.Timestamp("2014-01-01")
    ANALYSIS_END = pd.Timestamp("2024-12-31")

    # Expected unique daily date count for 2014-01-01 .. 2024-12-31 inclusive.
    EXPECTED_ERA5_UNIQUE_DATES = 4018  # 8*365 + 3*366 (leap years 2016, 2020, 2024)

    def __init__(self) -> None:
        self.results: dict[str, Any] = {
            "metadata": {
                "analysis_period": {
                    "start": self.ANALYSIS_START.strftime("%Y-%m-%d"),
                    "end": self.ANALYSIS_END.strftime("%Y-%m-%d"),
                },
                "expected_era5_dates": self.EXPECTED_ERA5_UNIQUE_DATES,
            },
            "files": {},
        }
        self.critical_failures: list[str] = []

        self.files = {
            "OMNI2": Config.FILE_OMNI_FEATHER,
            "MODIS": Config.FILE_MODIS_PARQUET,
            "ERA5": Config.FILE_ERA5_PARQUET,
            "SIF": Config.FILE_SIF_FINAL,
        }

    # -----------------------------------------------------------------------
    # Low-level IO helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _read_columns(
        path: Path,
        columns: list[str] | None,
        memory_map: bool = True,
    ) -> pd.DataFrame:
        """Read selected columns from a Parquet or Feather file."""
        path = Path(path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".parquet":
                table = pq.read_table(path, columns=columns)
                return table.to_pandas()
            if suffix == ".feather":
                table = feather.read_table(path, columns=columns, memory_map=memory_map)
                return table.to_pandas()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc
        raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _column_names(path: Path) -> list[str]:
        """Return column names without loading data."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pq.read_schema(path).names
        if suffix == ".feather":
            return feather.read_table(path).schema.names
        raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _to_date_series(values: pd.Series) -> pd.Series:
        """Convert a date/time column to timezone-naive datetime."""
        s = pd.to_datetime(values, errors="coerce")
        if s.dt.tz is not None:
            s = s.dt.tz_convert(None)
        return s

    @staticmethod
    def _era5_nan_fractions_from_missingness(
        base_cols: list[str], rolling_cols: list[str]
    ) -> dict[str, float]:
        """Map ETL missingness report to per-column NaN fractions."""
        nan_fractions: dict[str, float] = {}
        missingness_path = Config.REPORTS_INTEGRITY_DIR / "era5_missingness.csv"
        if missingness_path.exists():
            df = pd.read_csv(missingness_path)
            for _, row in df.iterrows():
                metric = row["metric"]
                window = row["window"]
                col = f"{metric}_ma{window}" if window else metric
                if col in base_cols or col in rolling_cols:
                    nan_fractions[col] = float(row.get("frac_nan_after", np.nan))
        # Any column not in the report gets NaN so the check is explicit.
        for col in base_cols + rolling_cols:
            nan_fractions.setdefault(col, float("nan"))
        return nan_fractions

    @staticmethod
    def _stream_inf_counts(path: Path, target_cols: list[str]) -> dict[str, int]:
        """Stream each target column in batches and count infinities."""
        inf_counts: dict[str, int] = {}
        pf = pq.ParquetFile(path)
        for col in target_cols:
            total_inf = 0
            for batch in pf.iter_batches(columns=[col], batch_size=1_000_000):
                arr = batch.column(0).to_numpy()
                total_inf += int(np.isinf(arr).sum())
                del batch, arr
            inf_counts[col] = total_inf
        return inf_counts

    @staticmethod
    def _stream_date_stats(path: Path, date_col: str) -> dict[str, Any]:
        """Stream the ERA5 date column and return lightweight stats."""
        print(f"[AUDIT] Streaming ERA5 date column '{date_col}'...")
        unique_dates: set[pd.Timestamp] = set()
        total_rows = 0
        dmin: pd.Timestamp | None = None
        dmax: pd.Timestamp | None = None
        rows_with_2025 = 0

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=[date_col], batch_size=5_000_000):
            s = pd.to_datetime(batch.column(0).to_pandas(), errors="coerce")
            if len(s) == 0:
                continue
            total_rows += len(s)
            batch_min, batch_max = s.min(), s.max()
            if dmin is None or (pd.notna(batch_min) and batch_min < dmin):
                dmin = batch_min
            if dmax is None or (pd.notna(batch_max) and batch_max > dmax):
                dmax = batch_max
            rows_with_2025 += int((s.dt.year == 2025).sum())
            unique_dates.update(s.dt.floor("D").unique())

        unique_dates_sorted = pd.Series(sorted(unique_dates)).reset_index(drop=True)
        return {
            "unique_dates": unique_dates_sorted,
            "n_unique": int(unique_dates_sorted.shape[0]),
            "dmin": dmin,
            "dmax": dmax,
            "total_rows": total_rows,
            "rows_with_2025": rows_with_2025,
        }

    @staticmethod
    def _stream_era5_spatial_audit(
        path: Path,
        date_col: str,
        unique_dates: pd.Series,
        sample_dates: list[pd.Timestamp],
    ) -> dict[str, Any]:
        """Stream (date, lat_id, lon_id) into a dense counter matrix.

        The canonical 0.5° grid has 360 × 720 = 259_200 cells and the audit
        period has 4_018 dates, so a uint32 matrix of that size (~4 GB) is
        enough to count every (cell, date) occurrence exactly. This avoids
        holding the 1B-row key space in pandas and works regardless of the
        parquet sort order.
        """
        n_dates = len(unique_dates)
        n_cells = 259_200  # canonical 0.5° global grid
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        print(f"[AUDIT] Allocating ERA5 counter matrix ({n_cells:,} cells × {n_dates:,} dates)...")
        counter = np.zeros((n_cells, n_dates), dtype=np.uint32)

        print("[AUDIT] Streaming ERA5 spatial keys into counter matrix...")
        pf = pq.ParquetFile(path)
        total_rows = 0
        off_grid_rows = 0
        bad_date_rows = 0

        for batch in pf.iter_batches(
            columns=[date_col, "lat_id", "lon_id"],
            batch_size=2_000_000,
        ):
            df = batch.to_pandas()
            dates = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
            lats = df["lat_id"].to_numpy(dtype=np.int64)
            lons = df["lon_id"].to_numpy(dtype=np.int64)

            # Canonical grid membership.
            lat_ok = (lats >= -8975) & (lats <= 8975) & ((lats % 50) == 25)
            lon_ok = (lons >= -17975) & (lons <= 17975) & ((lons % 50) == 25)
            on_grid = lat_ok & lon_ok
            valid_date = dates.notna()
            keep = on_grid & valid_date

            total_rows += len(df)
            off_grid_rows += int((~on_grid).sum())
            bad_date_rows += int((~valid_date).sum())

            if not keep.any():
                continue

            dates = dates[keep]
            lats = lats[keep]
            lons = lons[keep]

            date_idx = dates.map(date_to_idx).fillna(-1).to_numpy(dtype=np.int64)
            keep2 = date_idx >= 0
            if not keep2.all():
                bad_date_rows += int((~keep2).sum())
                date_idx = date_idx[keep2]
                lats = lats[keep2]
                lons = lons[keep2]

            lat_idx = (lats + 8975) // 50
            lon_idx = (lons + 17975) // 50
            cell_idx = lat_idx * 720 + lon_idx

            np.add.at(counter, (cell_idx, date_idx), 1)

        occupied_pairs = int(np.sum(counter > 0))
        duplicate_count = int(np.sum(counter)) - occupied_pairs
        cells_per_date = np.sum(counter > 0, axis=0)
        dates_per_cell = np.sum(counter > 0, axis=1)

        # Reconstruct (lat_id, lon_id) sets for sample dates from the matrix.
        sample_sets: dict[pd.Timestamp, set[tuple[int, int]]] = {}
        for sd in sample_dates:
            d_idx = date_to_idx[sd]
            cell_indices = np.where(counter[:, d_idx] > 0)[0]
            lat_idx = cell_indices // 720
            lon_idx = cell_indices % 720
            lats = (lat_idx * 50 - 8975).astype(int)
            lons = (lon_idx * 50 - 17975).astype(int)
            sample_sets[sd] = set(zip(lats.tolist(), lons.tolist()))

        print(f"[AUDIT] Counter matrix populated. total_rows={total_rows:,}, "
              f"duplicate_count={duplicate_count:,}, off_grid_rows={off_grid_rows:,}")
        return {
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "cells_per_date": cells_per_date,
            "dates_per_cell": dates_per_cell,
            "sample_sets": sample_sets,
            "off_grid_rows": off_grid_rows,
            "bad_date_rows": bad_date_rows,
            "n_cells_observed": int(np.sum(dates_per_cell > 0)),
        }

    @staticmethod
    def _pythonize(value: Any) -> Any:
        """Convert numpy/pandas scalars to plain Python types for JSON."""
        if isinstance(value, dict):
            return {k: PipelineIntegrityAudit._pythonize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PipelineIntegrityAudit._pythonize(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, Date):
            return value.strftime("%Y-%m-%d")
        if pd.isna(value):
            return None
        return value

    # -----------------------------------------------------------------------
    # Check bookkeeping
    # -----------------------------------------------------------------------
    def _add_check(
        self,
        file_results: dict[str, Any],
        name: str,
        passed: bool,
        details: str = "",
        measured: dict[str, Any] | None = None,
        critical: bool = False,
    ) -> bool:
        """Append a check result and record critical failures."""
        check = {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
        if measured is not None:
            check["measured"] = self._pythonize(measured)
        file_results["checks"].append(check)
        if critical and not passed:
            label = file_results.get("label", "unknown")
            self.critical_failures.append(f"{label}: {name}")
        return passed

    def _record_missing(self, label: str, path: Path) -> dict[str, Any]:
        """Record a missing file and return its results dict."""
        result = {
            "label": label,
            "path": str(path),
            "status": "missing",
            "checks": [],
        }
        self.results["files"][label] = result
        # Mark every expected check as failed so reports are explicit.
        self._add_check(result, "file_exists", False, f"File not found: {path}")
        return result

    # -----------------------------------------------------------------------
    # Generic helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _calendar_days_for_year(year: int) -> int:
        return 366 if calendar.isleap(year) else 365

    @staticmethod
    def _is_canonical_grid_id(lat_id: pd.Series, lon_id: pd.Series) -> pd.Series:
        """Return boolean mask of IDs that lie on the canonical 0.5° grid."""
        lat_ok = (
            (lat_id >= -8975)
            & (lat_id <= 8975)
            & ((lat_id % 50) == 25)
        )
        lon_ok = (
            (lon_id >= -17975)
            & (lon_id <= 17975)
            & ((lon_id % 50) == 25)
        )
        return lat_ok & lon_ok

    # -----------------------------------------------------------------------
    # ERA5 audit
    # -----------------------------------------------------------------------
    def audit_era5(self) -> None:
        label = "ERA5"
        path = self.files[label]
        result: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "status": "audited",
            "checks": [],
        }
        self.results["files"][label] = result

        if not path.exists():
            self._record_missing(label, path)
            return

        columns = self._column_names(path)
        date_col = "date" if "date" in columns else None
        if date_col is None:
            date_col = next((c for c in columns if "time" in c.lower()), None)

        if date_col is None:
            self._add_check(
                result,
                "has_date_column",
                False,
                "No date/time column found",
                {"columns": columns},
                critical=True,
            )
            return

        self._add_check(result, "has_date_column", True)

        required_id_cols = ["lat_id", "lon_id"]
        for col in required_id_cols:
            self._add_check(
                result,
                f"has_{col}",
                col in columns,
                f"{'Found' if col in columns else 'Missing'} column '{col}'",
                critical=True,
            )
        if not all(c in columns for c in required_id_cols):
            return

        # --- Dates (streamed, no full-column load) -------------------------
        date_stats = self._stream_date_stats(path, date_col)
        unique_dates = date_stats["unique_dates"]
        n_unique = date_stats["n_unique"]
        dmin = date_stats["dmin"]
        dmax = date_stats["dmax"]
        rows_with_2025 = date_stats["rows_with_2025"]

        self._add_check(
            result,
            "unique_dates_count",
            n_unique == self.EXPECTED_ERA5_UNIQUE_DATES,
            f"Expected {self.EXPECTED_ERA5_UNIQUE_DATES} unique dates, found {n_unique}",
            {"unique_dates": n_unique, "expected": self.EXPECTED_ERA5_UNIQUE_DATES},
            critical=True,
        )

        self._add_check(
            result,
            "min_date_exact",
            dmin == self.ANALYSIS_START,
            f"Min date is {dmin.date() if pd.notna(dmin) else None}, expected 2014-01-01",
            {"min_date": dmin},
            critical=True,
        )

        self._add_check(
            result,
            "max_date_exact",
            dmax == self.ANALYSIS_END,
            f"Max date is {dmax.date() if pd.notna(dmax) else None}, expected 2024-12-31",
            {"max_date": dmax},
            critical=True,
        )

        date_diffs = unique_dates.diff().dropna()
        all_one_day = (date_diffs == pd.Timedelta(days=1)).all()
        self._add_check(
            result,
            "dates_strictly_daily",
            all_one_day,
            "Dates are not strictly increasing with exactly 1-day gaps",
            {
                "first_gap": str(date_diffs.iloc[0]) if len(date_diffs) else None,
                "max_gap": str(date_diffs.max()) if len(date_diffs) else None,
            },
            critical=True,
        )

        # Calendar-correct days per year and all months present.
        years = sorted(unique_dates.dt.year.unique())
        bad_year_days = []
        missing_months = []
        for yr in years:
            yr_dates = unique_dates[unique_dates.dt.year == yr]
            expected_days = self._calendar_days_for_year(yr)
            if len(yr_dates) != expected_days:
                bad_year_days.append({
                    "year": int(yr),
                    "expected": expected_days,
                    "found": int(len(yr_dates)),
                })
            months_present = sorted(yr_dates.dt.month.unique())
            if months_present != list(range(1, 13)):
                missing_months.append({
                    "year": int(yr),
                    "missing": [m for m in range(1, 13) if m not in months_present],
                })

        self._add_check(
            result,
            "all_months_each_year",
            len(missing_months) == 0,
            f"{len(missing_months)} year(s) missing months",
            {"missing_months": missing_months} if missing_months else {},
            critical=True,
        )

        self._add_check(
            result,
            "calendar_correct_days_per_year",
            len(bad_year_days) == 0,
            f"{len(bad_year_days)} year(s) have wrong day count",
            {"bad_years": bad_year_days} if bad_year_days else {},
            critical=True,
        )

        self._add_check(
            result,
            "no_year_2025",
            rows_with_2025 == 0,
            f"Found {rows_with_2025} rows with year 2025",
            {"rows_with_year_2025": rows_with_2025},
            critical=True,
        )

        del date_stats, date_diffs
        gc.collect()

        # --- Spatial keys (counter matrix, ~4 GB, sort-order agnostic) ------
        first_date = unique_dates.iloc[0]
        last_date = unique_dates.iloc[-1]
        mid_idx = len(unique_dates) // 2
        mid_date = unique_dates.iloc[mid_idx]
        spatial = self._stream_era5_spatial_audit(
            path, date_col, unique_dates, [first_date, last_date, mid_date]
        )

        total_rows = spatial["total_rows"]
        duplicate_count = spatial["duplicate_count"]
        self._add_check(
            result,
            "date_lat_lon_unique",
            duplicate_count == 0,
            f"Found {duplicate_count} duplicate (date, lat_id, lon_id) rows",
            {"duplicate_rows": duplicate_count, "total_rows": total_rows},
            critical=True,
        )

        self._add_check(
            result,
            "canonical_grid_era5",
            spatial["off_grid_rows"] == 0,
            f"Found {spatial['off_grid_rows']} rows off the canonical 0.5° grid",
            {"off_grid_rows": spatial["off_grid_rows"], "total_rows": total_rows},
        )

        cells_per_date = pd.Series(spatial["cells_per_date"])
        n_cells_consistent = cells_per_date.nunique() == 1
        n_cells = int(cells_per_date.iloc[0]) if n_cells_consistent else None
        cells_min = int(cells_per_date.min())
        cells_max = int(cells_per_date.max())

        self._add_check(
            result,
            "same_spatial_cells_per_date",
            n_cells_consistent,
            f"Spatial cells per date vary: min={cells_min}, max={cells_max}",
            {"cells_per_date_min": cells_min, "cells_per_date_max": cells_max},
            critical=True,
        )

        dates_per_cell = pd.Series(spatial["dates_per_cell"])
        n_dates_per_cell_consistent = dates_per_cell.nunique() == 1
        n_dates_min = int(dates_per_cell.min())
        n_dates_max = int(dates_per_cell.max())
        expected_unique_cells = 259_200
        all_cells_complete = n_dates_min == n_unique and n_dates_max == n_unique
        self._add_check(
            result,
            "all_cells_have_all_dates",
            all_cells_complete,
            f"Dates per cell vary: min={n_dates_min}, max={n_dates_max}; "
            f"observed unique cells={spatial['n_cells_observed']}, expected={expected_unique_cells}",
            {
                "dates_per_cell_min": n_dates_min,
                "dates_per_cell_max": n_dates_max,
                "observed_unique_cells": spatial["n_cells_observed"],
                "expected_unique_cells": expected_unique_cells,
            },
            critical=True,
        )

        # Same key set across dates: compare first and last dates plus a sample.
        same_key_set = False
        sample_details: dict[str, Any] = {}
        if n_cells_consistent and n_cells is not None and n_cells > 0:
            first_set = spatial["sample_sets"].get(first_date, set())
            last_set = spatial["sample_sets"].get(last_date, set())
            mid_set = spatial["sample_sets"].get(mid_date, set())
            sample_details = {
                "first_date": first_date.strftime("%Y-%m-%d"),
                "last_date": last_date.strftime("%Y-%m-%d"),
                "mid_date": mid_date.strftime("%Y-%m-%d"),
                "set_size": n_cells,
                "first_last_equal": first_set == last_set,
                "first_mid_equal": first_set == mid_set,
            }
            same_key_set = (first_set == last_set) and (first_set == mid_set)
            if not same_key_set:
                sample_details["symmetric_diff_first_last"] = len(first_set ^ last_set)

        self._add_check(
            result,
            "same_spatial_key_set_per_date",
            same_key_set,
            "Spatial key set differs across sampled dates",
            sample_details,
            critical=True,
        )

        del spatial, cells_per_date, dates_per_cell
        gc.collect()

        # --- Base / rolling column integrity --------------------------------
        rolling_pattern = re.compile(r"_ma\d+$")
        rolling_cols = [c for c in columns if rolling_pattern.search(c)]
        base_cols = []
        for c in rolling_cols:
            base = c.rsplit("_ma", 1)[0]
            if base in columns and base not in base_cols:
                base_cols.append(base)
        target_cols = base_cols + rolling_cols

        self._add_check(
            result,
            "has_rolling_columns",
            len(rolling_cols) > 0,
            f"Found {len(rolling_cols)} rolling columns",
            {"rolling_columns": rolling_cols, "base_columns": base_cols},
        )

        if target_cols:
            # NaN fractions come from the ETL missingness report (avoids reading
            # the whole ERA5 file once per column).
            nan_fractions = self._era5_nan_fractions_from_missingness(
                base_cols, rolling_cols
            )

            # Infinity check: stream each target column in small batches so the
            # audit does not hold a full 1B-row column in memory.
            inf_counts = self._stream_inf_counts(path, target_cols)

            any_inf = any(v > 0 for v in inf_counts.values())
            self._add_check(
                result,
                "no_infinite_values",
                not any_inf,
                "Found infinite values in base/rolling columns",
                {"inf_counts": inf_counts},
                critical=True,
            )

            self._add_check(
                result,
                "nan_fractions",
                True,
                "NaN fraction recorded for base/rolling columns",
                {"nan_fractions": nan_fractions},
            )

    # -----------------------------------------------------------------------
    # OMNI audit
    # -----------------------------------------------------------------------
    def audit_omni(self) -> None:
        label = "OMNI2"
        path = self.files[label]
        result: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "status": "audited",
            "checks": [],
        }
        self.results["files"][label] = result

        if not path.exists():
            self._record_missing(label, path)
            return

        columns = self._column_names(path)
        date_col = "date" if "date" in columns else None
        if date_col is None:
            date_col = next((c for c in columns if "time" in c.lower()), None)

        if date_col is None:
            self._add_check(result, "has_date_column", False, "No date/time column found")
            return

        self._add_check(result, "has_date_column", True)

        # Expected base columns.
        base_means = ["sii_mean", "kp_mean", "f10_7_mean"]
        missing_base = [c for c in base_means if c not in columns]
        self._add_check(
            result,
            "expected_base_columns",
            len(missing_base) == 0,
            f"Missing base columns: {missing_base}" if missing_base else "All base columns present",
            {"expected": base_means, "missing": missing_base},
        )

        # MA / lag features.
        ma_cols = [c for c in columns if "_ma" in c]
        lag_cols = [c for c in columns if "_lag" in c]
        self._add_check(
            result,
            "has_ma_features",
            len(ma_cols) > 0,
            f"Found {len(ma_cols)} moving-average feature columns",
            {"ma_columns_sample": ma_cols[:20]},
        )
        self._add_check(
            result,
            "has_lag_features",
            len(lag_cols) > 0,
            f"Found {len(lag_cols)} lag feature columns",
            {"lag_columns_sample": lag_cols[:20]},
        )

        # Date range and duplicates.
        dates_df = self._read_columns(path, [date_col])
        dates = self._to_date_series(dates_df[date_col])
        dmin, dmax = dates.min(), dates.max()
        covers_period = (dmin <= self.ANALYSIS_START) and (dmax >= self.ANALYSIS_END)
        duplicate_dates = int(dates.duplicated().sum())

        self._add_check(
            result,
            "covers_analysis_period",
            covers_period,
            f"Date range {dmin.date() if pd.notna(dmin) else None} .. "
            f"{dmax.date() if pd.notna(dmax) else None} does not fully cover 2014-2024",
            {"min_date": dmin, "max_date": dmax},
        )

        self._add_check(
            result,
            "no_duplicate_dates",
            duplicate_dates == 0,
            f"Found {duplicate_dates} duplicate dates",
            {"duplicate_dates": duplicate_dates},
        )

        del dates_df, dates
        gc.collect()

        # Lag1 leak test: the ETL builds lag features as shift(l) followed by a
        # trailing smoothing window, so lag1(t) must equal the smoothed past
        # values of base(t), never base(t) itself.
        lag_smooth_w = getattr(Config, "OMNI_LAG_SMOOTH_WINDOW", 3)
        lag1_results: list[dict[str, Any]] = []
        for base in ("sii", "kp", "f10_7"):
            base_col = f"{base}_mean"
            lag1_col = f"{base_col}_lag1"
            if base_col not in columns or lag1_col not in columns:
                continue
            df = self._read_columns(path, [date_col, base_col, lag1_col])
            df[date_col] = self._to_date_series(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df["expected_lag1"] = (
                df[base_col]
                .shift(1)
                .rolling(window=lag_smooth_w, center=False, min_periods=1)
                .mean()
            )
            comparable = df[[lag1_col, "expected_lag1"]].dropna()
            if len(comparable) == 0:
                lag1_results.append({"base": base, "status": "no_comparable_rows"})
                continue
            match = np.isclose(
                comparable[lag1_col].astype(float).values,
                comparable["expected_lag1"].astype(float).values,
                equal_nan=True,
            )
            n_mismatch = int((~match).sum())
            same_day_leak = int(
                np.isclose(
                    df[lag1_col].astype(float).values,
                    df[base_col].astype(float).values,
                    equal_nan=True,
                ).sum()
            )
            lag1_results.append({
                "base": base,
                "compared_rows": int(len(comparable)),
                "mismatches": n_mismatch,
                "same_day_coincidences": same_day_leak,
                "leak_free": n_mismatch == 0,
            })
            del df, comparable
            gc.collect()

        if lag1_results:
            all_leak_free = all(r.get("leak_free", False) for r in lag1_results)
            self._add_check(
                result,
                "lag1_leak_test",
                all_leak_free,
                "lag1 features do not match expected shift+smooth values" if not all_leak_free else "lag1 shift+smooth verified",
                {"per_base": lag1_results, "smooth_window": lag_smooth_w},
            )
        else:
            self._add_check(
                result,
                "lag1_leak_test",
                False,
                "No lag1 columns found to test (expected e.g. sii_mean_lag1)",
                {"columns": columns},
            )

    # -----------------------------------------------------------------------
    # MODIS audit
    # -----------------------------------------------------------------------
    def audit_modis(self) -> None:
        label = "MODIS"
        path = self.files[label]
        result: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "status": "audited",
            "checks": [],
        }
        self.results["files"][label] = result

        if not path.exists():
            self._record_missing(label, path)
            return

        columns = self._column_names(path)
        date_col = "date" if "date" in columns else None
        if date_col is None:
            date_col = next((c for c in columns if "time" in c.lower()), None)

        if date_col is None:
            self._add_check(result, "has_date_column", False, "No date/time column found")
            return

        self._add_check(result, "has_date_column", True)

        required_cols = ["lat_id", "lon_id"]
        missing = [c for c in required_cols if c not in columns]
        self._add_check(
            result,
            "required_columns",
            len(missing) == 0,
            f"Missing columns: {missing}" if missing else "All required columns present",
            {"required": required_cols, "missing": missing},
        )
        if missing:
            return

        # Date range overlap.
        dates_df = self._read_columns(path, [date_col])
        dates = self._to_date_series(dates_df[date_col])
        dmin, dmax = dates.min(), dates.max()
        overlaps = (dmax >= self.ANALYSIS_START) and (dmin <= self.ANALYSIS_END)
        self._add_check(
            result,
            "date_range_overlaps_analysis_period",
            overlaps,
            f"Date range {dmin.date() if pd.notna(dmin) else None} .. "
            f"{dmax.date() if pd.notna(dmax) else None} does not overlap 2014-2024",
            {"min_date": dmin, "max_date": dmax},
        )
        del dates_df, dates
        gc.collect()

        # Grid check.
        grid_df = self._read_columns(path, ["lat_id", "lon_id"])
        on_grid = self._is_canonical_grid_id(grid_df["lat_id"], grid_df["lon_id"])
        n_off_grid = int((~on_grid).sum())
        self._add_check(
            result,
            "canonical_grid",
            n_off_grid == 0,
            f"Found {n_off_grid} rows with lat_id/lon_id off the canonical 0.5° grid",
            {
                "off_grid_rows": n_off_grid,
                "total_rows": int(len(grid_df)),
                "lat_id_min": int(grid_df["lat_id"].min()),
                "lat_id_max": int(grid_df["lat_id"].max()),
                "lon_id_min": int(grid_df["lon_id"].min()),
                "lon_id_max": int(grid_df["lon_id"].max()),
            },
        )
        del grid_df, on_grid
        gc.collect()

        # Duplicate keys.
        keys_df = self._read_columns(path, [date_col, "lat_id", "lon_id"])
        keys_df[date_col] = self._to_date_series(keys_df[date_col])
        dupes = int(keys_df.duplicated([date_col, "lat_id", "lon_id"]).sum())
        self._add_check(
            result,
            "no_duplicate_date_lat_lon",
            dupes == 0,
            f"Found {dupes} duplicate (date, lat_id, lon_id) rows",
            {"duplicate_rows": dupes, "total_rows": int(len(keys_df))},
        )
        del keys_df
        gc.collect()

    # -----------------------------------------------------------------------
    # SIF audit
    # -----------------------------------------------------------------------
    def audit_sif(self) -> None:
        label = "SIF"
        path = self.files[label]
        result: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "status": "audited",
            "checks": [],
        }
        self.results["files"][label] = result

        if not path.exists():
            self._record_missing(label, path)
            return

        columns = self._column_names(path)
        date_col = "date" if "date" in columns else None
        if date_col is None:
            date_col = next((c for c in columns if "time" in c.lower()), None)

        if date_col is None:
            self._add_check(result, "has_date_column", False, "No date/time column found")
            return

        self._add_check(result, "has_date_column", True)

        required_cols = ["lat_id", "lon_id"]
        missing = [c for c in required_cols if c not in columns]
        self._add_check(
            result,
            "required_columns",
            len(missing) == 0,
            f"Missing columns: {missing}" if missing else "All required columns present",
            {"required": required_cols, "missing": missing},
        )
        if missing:
            return

        # Date range overlap.
        dates_df = self._read_columns(path, [date_col])
        dates = self._to_date_series(dates_df[date_col])
        dmin, dmax = dates.min(), dates.max()
        overlaps = (dmax >= self.ANALYSIS_START) and (dmin <= self.ANALYSIS_END)
        self._add_check(
            result,
            "date_range_overlaps_analysis_period",
            overlaps,
            f"Date range {dmin.date() if pd.notna(dmin) else None} .. "
            f"{dmax.date() if pd.notna(dmax) else None} does not overlap 2014-2024",
            {"min_date": dmin, "max_date": dmax},
        )
        del dates_df, dates
        gc.collect()

        # Grid check.
        grid_df = self._read_columns(path, ["lat_id", "lon_id"])
        on_grid = self._is_canonical_grid_id(grid_df["lat_id"], grid_df["lon_id"])
        n_off_grid = int((~on_grid).sum())
        self._add_check(
            result,
            "canonical_grid",
            n_off_grid == 0,
            f"Found {n_off_grid} rows with lat_id/lon_id off the canonical 0.5° grid",
            {
                "off_grid_rows": n_off_grid,
                "total_rows": int(len(grid_df)),
                "lat_id_min": int(grid_df["lat_id"].min()),
                "lat_id_max": int(grid_df["lat_id"].max()),
                "lon_id_min": int(grid_df["lon_id"].min()),
                "lon_id_max": int(grid_df["lon_id"].max()),
            },
        )
        del grid_df, on_grid
        gc.collect()

        # Duplicate keys.
        keys_df = self._read_columns(path, [date_col, "lat_id", "lon_id"])
        keys_df[date_col] = self._to_date_series(keys_df[date_col])
        dupes = int(keys_df.duplicated([date_col, "lat_id", "lon_id"]).sum())
        self._add_check(
            result,
            "no_duplicate_date_lat_lon",
            dupes == 0,
            f"Found {dupes} duplicate (date, lat_id, lon_id) rows",
            {"duplicate_rows": dupes, "total_rows": int(len(keys_df))},
        )
        del keys_df
        gc.collect()

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    def _save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._pythonize(self.results), f, indent=2, ensure_ascii=False)

    def _save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "# MAGNETO Pipeline Integrity Audit Report",
            "",
            f"**Analysis period:** {self.ANALYSIS_START.date()} to {self.ANALYSIS_END.date()}",
            f"**Expected ERA5 unique dates:** {self.EXPECTED_ERA5_UNIQUE_DATES}",
            "",
            "## Summary",
            "",
        ]

        overall = "PASS" if not self.critical_failures else "FAIL"
        lines.append(f"**Overall critical status:** {overall}")
        if self.critical_failures:
            lines.append("")
            lines.append("**Failing critical checks:**")
            for failure in self.critical_failures:
                lines.append(f"- {failure}")
        lines.append("")

        for label, file_result in self.results["files"].items():
            lines.append(f"## {label}")
            lines.append(f"- **File:** `{file_result['path']}`")
            lines.append(f"- **Status:** {file_result['status']}")
            lines.append("")
            lines.append("| Check | Status | Details |")
            lines.append("|-------|--------|---------|")
            for check in file_result.get("checks", []):
                status = check["status"]
                details = check.get("details", "")
                measured = check.get("measured")
                if measured is not None:
                    measured_str = json.dumps(measured, ensure_ascii=False, default=str)
                    if len(measured_str) > 200:
                        measured_str = measured_str[:200] + "..."
                    details = f"{details} (measured: {measured_str})" if details else f"measured: {measured_str}"
                lines.append(f"| {check['name']} | {status} | {details} |")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _print_summary(self) -> None:
        n_files = len(self.results["files"])
        n_pass_files = sum(
            1 for fr in self.results["files"].values()
            if all(c["status"] == "PASS" for c in fr.get("checks", []))
        )
        print("[AUDIT] MAGNETO Pipeline Integrity Audit")
        print(f"[AUDIT] Files audited: {n_files}")
        print(f"[AUDIT] Files with all checks passing: {n_pass_files}")
        if self.critical_failures:
            print(f"[AUDIT] {len(self.critical_failures)} critical check(s) failed:")
            for failure in self.critical_failures:
                print(f"  - {failure}")
        else:
            print("[AUDIT] All critical checks passed.")

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self) -> int:
        print("[AUDIT] Starting pipeline integrity audit...")
        print("[AUDIT] Auditing OMNI2...")
        self.audit_omni()
        print("[AUDIT] Auditing MODIS...")
        self.audit_modis()
        print("[AUDIT] Auditing ERA5...")
        self.audit_era5()
        print("[AUDIT] Auditing SIF...")
        self.audit_sif()

        json_path = Config.REPORTS_INTEGRITY_DIR / "pipeline_integrity.json"
        md_path = Config.REPORTS_INTEGRITY_DIR / "pipeline_integrity.md"

        self._save_json(json_path)
        self._save_markdown(md_path)

        self._print_summary()
        print(f"[AUDIT] JSON report saved to {json_path}")
        print(f"[AUDIT] Markdown report saved to {md_path}")

        return 0 if not self.critical_failures else 1


if __name__ == "__main__":
    sys.exit(PipelineIntegrityAudit().run())
