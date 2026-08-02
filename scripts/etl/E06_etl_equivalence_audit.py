#!/usr/bin/env python3
"""E06_etl_equivalence_audit.py — lightweight upstream ETL schema audit.

Performs cheap schema and metadata checks on the canonical intermediate files.
It deliberately does **not** load large parquet/feather files into memory so
that it can run on modest hardware.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

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

REPORT_PATH = PROJECT_ROOT / "reports" / "ETL_EQUIVALENCE_AUDIT.md"


def _schema_info(path: Path, required: list[str]) -> dict:
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    missing = [c for c in required if c not in names]
    return {
        "rows": pf.metadata.num_rows,
        "columns": list(names),
        "missing_columns": missing,
        "ok": len(missing) == 0,
    }


def audit_modis() -> dict:
    print("\n[AUDIT] MODIS")
    info = _schema_info(FILE_MODIS, ["date", "lat_id", "lon_id", "lai",
                                      "cloud_fraction", "aerosol_fraction",
                                      "quality_flag"])
    print(f"  rows={info['rows']:,}, missing={info['missing_columns']}")
    return info


def audit_omni() -> dict:
    print("\n[AUDIT] OMNI2")
    df = pd.read_feather(FILE_OMNI, columns=["date", "sii_mean", "kp_mean", "f10_7_mean"])
    info = {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "missing_columns": [],
        "ok": True,
    }
    print(f"  rows={info['rows']:,}, range={info['date_min']} → {info['date_max']}")
    return info


def audit_sif() -> dict:
    print("\n[AUDIT] OCO-2 SIF")
    df = pd.read_feather(FILE_SIF, columns=["date", "lat_id", "lon_id", "sif_771nm", "region_flags"])
    info = {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "missing_columns": [],
        "ok": True,
    }
    print(f"  rows={info['rows']:,}, range={info['date_min']} → {info['date_max']}")
    return info


def audit_era5() -> dict:
    print("\n[AUDIT] ERA5")
    if not FILE_ERA5.exists():
        print("  file not found — ERA5 ETL has not been run")
        return {"exists": False, "ok": False, "missing_columns": []}
    info = _schema_info(FILE_ERA5, ["date", "lat_id", "lon_id", "temp_c", "temp_c_ma10"])
    print(f"  rows={info['rows']:,}, missing={info['missing_columns']}")
    return info


def _write_report(results: dict) -> None:
    lines = [
        "# MAGNETO upstream ETL equivalence audit",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Lightweight schema/metadata checks only. Large files are not loaded into RAM.",
        "",
        "## Results",
        "",
        "| Dataset | Rows | Date range | Missing columns | OK |",
        "|---------|------|------------|-----------------|----|",
    ]
    for name, res in results.items():
        rows = f"{res.get('rows', 'N/A'):,}" if "rows" in res else "N/A"
        drange = f"{res.get('date_min', '?')} → {res.get('date_max', '?')}"
        missing = ", ".join(res.get("missing_columns", [])) or "none"
        ok = "✅" if res.get("ok") else "❌"
        lines.append(f"| {name} | {rows} | {drange} | {missing} | {ok} |")
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


def main() -> int:
    print("MAGNETO — lightweight upstream ETL equivalence audit")
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
