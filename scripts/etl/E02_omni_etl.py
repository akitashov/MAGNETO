#!/usr/bin/env python3
"""E02_omni_etl.py — OMNI2 upstream ETL.

Reads the hourly OMNI2 fixed-width archive, extracts Dst, Kp and F10.7,
applies missing-value handling, computes the Storm Intensity Index
(SII = -Dst), and aggregates to daily statistics.  Daily cumulative windows
and discrete lags are retained for compatibility with legacy downstream
stages that read ``sii_mean_ma*`` columns directly.

Output: ``data/interim/omni_biosphere_features.feather``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow imports of shared utilities from scripts/ regardless of where this
# ETL script is executed.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (
    PROJECT_ROOT,
    atomic_write,
    setup_dirs,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml

_PIPELINE_CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
_PIPELINE_CFG = yaml.safe_load(_PIPELINE_CFG_PATH.read_text(encoding="utf-8"))
_OCFG = _PIPELINE_CFG.get("omni2", {})

INPUT_ZIP = PROJECT_ROOT / _OCFG.get("input_zip", "data/raw/omni2_all_years.zip")
OUTPUT_PATH = PROJECT_ROOT / _PIPELINE_CFG["inputs"]["file_omni"]
CANDIDATE_PATH = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_candidate.feather")

# Kp scale note.  OMNI2 stores Kp as integer 0–90, i.e. in units of 0.1 × Kp.
# The values written to the canonical feather are kept on the raw OMNI2 scale
# unless the configuration explicitly requests a conversion.  Downstream tables
# and figures must report that kp_mean is in 0.1-Kp units (or divide by 10).
KP_SCALE = float(_OCFG.get("kp_scale", 0.1))
APPLY_KP_CONVERSION = bool(_OCFG.get("apply_kp_conversion", False))

# Candidate ↔ canonical promotion helper.
import _etl_promote as _promote


# Windows historically produced by the v1 OMNI ETL.  The preferred analytic
# windows (21, 28 d) are rebuilt downstream by ``magneto_lib.compute_sii_windows``;
# the full set is kept here only for legacy compatibility.
MA_WINDOWS = list(range(1, 29)) + [30, 40, 50, 60, 75, 90]
DISCRETE_LAGS = [1, 2] + list(range(3, 91, 3))

# Aggregations match the existing ``omni_biosphere_features.feather`` schema.
AGGREGATIONS = {
    "sii": ["mean", "max", "std"],
    "kp": ["mean", "max"],
    "f10_7": ["mean"],
}

# Legacy column-position parser.  Indices are zero-based OMNI2 hourly fields.
KP_COL = 38
DST_COL = 40
F10_7_COL = 50

DST_MIN_VALID = -2000


def _parse_line(line: str) -> dict | None:
    """Parse one OMNI2 hourly fixed-width record."""
    parts = line.split()
    if len(parts) < 51:
        return None

    try:
        year = int(parts[0])
        day = int(parts[1])
        hour = int(parts[2])
    except Exception:
        return None

    if not (1900 <= year <= 2100):
        return None

    try:
        kp = int(parts[KP_COL])
        if kp == 99:
            kp = np.nan
    except Exception:
        kp = np.nan

    try:
        dst = int(parts[DST_COL])
        if dst == 99999:
            dst = np.nan
    except Exception:
        dst = np.nan

    try:
        f10 = float(parts[F10_7_COL])
        if f10 == 999.9:
            f10 = np.nan
    except Exception:
        f10 = np.nan

    return {
        "year": year,
        "day": day,
        "hour": hour,
        "dst": dst,
        "kp": kp,
        "f10_7": f10,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MAGNETO OMNI2 ETL")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output before writing",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("MAGNETO — OMNI2 ETL")
    print("=" * 64)
    setup_dirs()

    if not INPUT_ZIP.exists():
        raise FileNotFoundError(f"OMNI2 ZIP not found: {INPUT_ZIP}")

    if args.clean:
        for p in [OUTPUT_PATH, CANDIDATE_PATH]:
            if p.exists():
                p.unlink()
                print(f"[INFO] Removed existing output: {p}")

    records: list[dict] = []

    with zipfile.ZipFile(INPUT_ZIP, "r") as z:
        inner_name = max(z.namelist(), key=lambda x: z.getinfo(x).file_size)
        print(f"[INFO] Reading {inner_name} from {INPUT_ZIP}")

        with z.open(inner_name, "r") as f:
            pbar = tqdm(unit="lines", desc="OMNI2 parsing")
            for raw in f:
                try:
                    line = raw.decode("utf-8", errors="ignore").rstrip("\n\r")
                except Exception:
                    pbar.update(1)
                    continue

                rec = _parse_line(line)
                if rec is not None:
                    records.append(rec)
                pbar.update(1)
            pbar.close()

    if not records:
        raise RuntimeError("No valid OMNI2 records parsed.")

    df = pd.DataFrame.from_records(records)
    df["datetime"] = pd.to_datetime(
        df["year"].astype(str)
        + df["day"].astype(str).str.zfill(3)
        + df["hour"].astype(str).str.zfill(2),
        format="%Y%j%H",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.drop(columns=["year", "day", "hour"])

    # SII = -Dst; keep NaN Dst as NaN SII.
    df = df[df["dst"].isna() | (df["dst"] > DST_MIN_VALID)].copy()
    df["sii"] = -df["dst"]

    # Daily aggregation.
    daily = df.resample("D").agg(AGGREGATIONS)
    daily.columns = ["_".join(col).strip("_") for col in daily.columns.to_flat_index()]

    # Kp scale documentation / optional conversion.
    kp_cols = [c for c in daily.columns if c.startswith("kp_")]
    if APPLY_KP_CONVERSION:
        print(f"[INFO] Converting Kp columns by factor {KP_SCALE} (raw OMNI2 → Kp units)")
        for c in kp_cols:
            daily[c] = daily[c] * KP_SCALE
    else:
        print(f"[INFO] Kp columns kept on raw OMNI2 scale ({KP_SCALE}-Kp units); "
              f"divide by {1.0 / KP_SCALE:.0f} to obtain ordinary Kp")

    # Feature engineering: shifted rolling windows (dose) and discrete lags.
    num_cols = daily.select_dtypes(include=[np.number]).columns.tolist()

    # 3-day centered smoothing for lag generation, as in the legacy pipeline.
    smoothed = daily[num_cols].rolling(window=3, center=True, min_periods=1).mean()

    engineered: dict[str, pd.Series] = {}
    for col in num_cols:
        for w in MA_WINDOWS:
            engineered[f"{col}_ma{w}"] = daily[col].shift(1).rolling(window=w, min_periods=1).mean()
        for l in DISCRETE_LAGS:
            engineered[f"{col}_lag{l}"] = smoothed[col].shift(l)
        engineered[f"{col}_diff"] = daily[col].diff()

    daily = pd.concat([daily, pd.DataFrame(engineered)], axis=1)
    daily.index.name = "date"
    daily = daily.reset_index()

    print(f"[INFO] OMNI2 date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"[INFO] Output columns: {len(daily.columns)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_feather(CANDIDATE_PATH)

    spec = {
        "name": "omni_biosphere_features",
        "fmt": "feather",
        "required": ["date", "sii_mean", "kp_mean", "f10_7_mean"],
        "key_cols": ["date"],
        "metric_cols": ["sii_mean", "kp_mean", "f10_7_mean"],
        "grid_cols": [],
    }
    report = _promote.compare_and_promote(OUTPUT_PATH, CANDIDATE_PATH, spec)
    if not report["passed"]:
        print("[STOP] OMNI candidate differs from canonical. Promotion blocked; operator review required.")
        return 2

    print(f"\n{'='*64}")
    print(f"OMNI2 ETL SUMMARY")
    print(f"  Daily rows: {len(daily):,}")
    print(f"  Output:     {OUTPUT_PATH}")
    print(f"[OK] E02_omni_etl complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
