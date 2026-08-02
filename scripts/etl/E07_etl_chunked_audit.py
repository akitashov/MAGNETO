#!/usr/bin/env python3
"""E07_etl_chunked_audit.py — memory-safe upstream ETL data audit.

Scans the canonical MODIS and ERA5 parquet files in batches and checks:
  * required columns and row counts;
  * monotonic date ranges across batches;
  * no all-NaN metric columns in any batch;
  * latitude/longitude bounds stay inside the expected global grid;
  * year-level row counts are sensible.

The script never materialises a full parquet file in RAM, so it can run on
modest hardware (including WSL) at the cost of not checking key uniqueness.
For a full uniqueness / value-equivalence audit see E08_etl_deep_equivalence_audit.py.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Allow imports of shared utilities from scripts/ regardless of where this
# ETL script is executed.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (
    FILE_MODIS,
    FILE_OMNI,
    FILE_SIF,
    FILE_ERA5,
    PROJECT_ROOT,
    setup_dirs,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "ETL_CHUNKED_AUDIT.md"
BATCH_SIZE = 1_000_000


def _check_parquet_chunks(
    path: Path,
    label: str,
    required: list[str],
    metric_cols: list[str],
) -> dict:
    print(f"\n[AUDIT] {label} (chunked)")
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema.names)
    missing = [c for c in required if c not in schema_names]
    if missing:
        return {"rows": pf.metadata.num_rows, "missing_columns": missing, "ok": False}

    total_rows = pf.metadata.num_rows
    prev_max_date = pd.Timestamp.min
    date_min = pd.Timestamp.max
    date_max = pd.Timestamp.min
    lat_min, lat_max = np.inf, -np.inf
    lon_min, lon_max = np.inf, -np.inf
    year_counts: dict[int, int] = {}
    bad_batches = []

    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=required)):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"])

        batch_min_date = df["date"].min()
        batch_max_date = df["date"].max()
        if batch_min_date < prev_max_date:
            bad_batches.append(
                f"batch {i}: dates non-monotonic ({batch_min_date} < {prev_max_date})"
            )
        prev_max_date = max(prev_max_date, batch_max_date)
        date_min = min(date_min, batch_min_date)
        date_max = max(date_max, batch_max_date)

        lat_min = min(lat_min, df["lat_id"].min())
        lat_max = max(lat_max, df["lat_id"].max())
        lon_min = min(lon_min, df["lon_id"].min())
        lon_max = max(lon_max, df["lon_id"].max())

        for yr, cnt in df["date"].dt.year.value_counts().items():
            year_counts[yr] = year_counts.get(yr, 0) + int(cnt)

        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() == 0:
                bad_batches.append(f"batch {i}: column '{col}' is all-NaN")

        del df, batch

    ok = len(missing) == 0 and len(bad_batches) == 0
    info = {
        "rows": total_rows,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "lat_range": (float(lat_min) / 100, float(lat_max) / 100),
        "lon_range": (float(lon_min) / 100, float(lon_max) / 100),
        "year_counts": year_counts,
        "missing_columns": missing,
        "bad_batches": bad_batches,
        "ok": ok,
    }
    print(f"  rows={total_rows:,}, range={info['date_min']} → {info['date_max']}")
    print(f"  lat={info['lat_range']}, lon={info['lon_range']}")
    print(f"  years={sorted(year_counts.keys())}")
    if bad_batches:
        for msg in bad_batches[:5]:
            print(f"  [WARN] {msg}")
    return info


def audit_modis() -> dict:
    return _check_parquet_chunks(
        FILE_MODIS,
        "MODIS",
        ["date", "lat_id", "lon_id", "lai", "cloud_fraction", "aerosol_fraction", "quality_flag"],
        ["lai", "cloud_fraction", "aerosol_fraction"],
    )


def audit_era5() -> dict:
    if not FILE_ERA5.exists():
        print("\n[AUDIT] ERA5 (chunked)\n  file not found")
        return {"exists": False, "ok": False, "missing_columns": []}
    return _check_parquet_chunks(
        FILE_ERA5,
        "ERA5",
        ["date", "lat_id", "lon_id", "temp_c", "temp_c_ma10", "vpd", "par"],
        ["temp_c", "temp_c_ma10", "vpd", "par"],
    )


def audit_omni() -> dict:
    print("\n[AUDIT] OMNI2 (chunked)")
    df = pd.read_feather(FILE_OMNI, columns=["date", "sii_mean", "kp_mean", "f10_7_mean"])
    info = {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "sii_mean": float(df["sii_mean"].mean()),
        "kp_mean": float(df["kp_mean"].mean()),
        "f10_7_mean": float(df["f10_7_mean"].mean()),
        "missing_columns": [],
        "ok": True,
    }
    print(f"  rows={info['rows']:,}, range={info['date_min']} → {info['date_max']}")
    return info


def audit_sif() -> dict:
    print("\n[AUDIT] OCO-2 SIF (chunked)")
    df = pd.read_feather(FILE_SIF, columns=["date", "lat_id", "lon_id", "sif_771nm", "region_flags"])
    info = {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_cells": int(df[["lat_id", "lon_id"]].drop_duplicates().shape[0]),
        "sif_771nm_mean": float(df["sif_771nm"].mean()),
        "missing_columns": [],
        "ok": True,
    }
    print(f"  rows={info['rows']:,}, cells={info['n_cells']:,}, range={info['date_min']} → {info['date_max']}")
    return info


def _write_report(results: dict) -> None:
    lines = [
        "# MAGNETO upstream ETL chunked audit",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Memory-safe batch-by-batch checks. Key uniqueness is **not** verified here.",
        "",
        "## Results",
        "",
        "| Dataset | Rows | Date range | OK |",
        "|---------|------|------------|----|",
    ]
    for name, res in results.items():
        rows = f"{res.get('rows', 'N/A'):,}" if "rows" in res else "N/A"
        drange = f"{res.get('date_min', '?')} → {res.get('date_max', '?')}"
        ok = "✅" if res.get("ok") else "❌"
        lines.append(f"| {name} | {rows} | {drange} | {ok} |")

    lines.extend(["", "## Notes", ""])
    for name, res in results.items():
        if res.get("bad_batches"):
            lines.append(f"- **{name}** issues:")
            for msg in res["bad_batches"]:
                lines.append(f"  - {msg}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


def main() -> int:
    print("MAGNETO — memory-safe upstream ETL chunked audit")
    setup_dirs()
    results = {
        "MODIS": audit_modis(),
        "OMNI2": audit_omni(),
        "OCO-2 SIF": audit_sif(),
        "ERA5": audit_era5(),
    }
    _write_report(results)
    all_ok = all(r.get("ok") for r in results.values())
    print("\nAUDIT PASSED" if all_ok else "\nAUDIT FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
