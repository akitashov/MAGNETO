#!/usr/bin/env python3
"""
OMNI2 Data Processing Pipeline: Extraction of Magnetic (SII, Kp) and Solar (F10.7) Indices.

Core steps:
1) Read OMNI2 hourly records from ZIP (.dat inside).
2) Parse (year, DOY, hour), extract Kp, Dst, F10.7 with missing-value handling.
3) Transform: SII = -Dst (Storm Intensity Index).
4) Aggregate to daily statistics (mean/max/std as configured).
5) Feature engineering:
   - Moving averages (dose): shift(1) + rolling mean (to represent cumulative effects).
   - Discrete lags (signal/presensing): optional smoothing + shift(l).
6) Save to Feather format.

CONFIGURATION:
--------------
Paths and Constants are imported from `_Common.py`.

OUTPUT FILE DESCRIPTION:
-------------------------------
The final file (omni2_daily.feather) contains daily resolution features:

1. Temporal Index:
   - 'date': Timestamp (Index).

2. Raw Daily Aggregates (suffix: _mean, _max, _std):
   - 'sii_mean', 'sii_max', 'sii_std': Aggregated Storm Intensity Index.
   - 'kp_mean', 'kp_max': Planetary K-index.
   - 'f10_7_mean': Solar radio flux at 10.7 cm.

3. Cumulative Dose Features (suffix: _ma{window}):
   - 'sii_mean_ma1', ..., 'sii_mean_ma90': Moving averages of daily mean SII.
   - 'kp_mean_ma1', ..., 'f10_7_mean_ma1', etc.: Rolling averages for all variables 
     based on Config.MA_WINDOWS (1, 3, 5, 7, 10, 14, 21, 28, 45, 60, 90 days).
   - All MA features are shifted by 1 day to ensure no temporal leakage (predicting today 
     based on strictly previous data).

4. Discrete Lag Features (suffix: _lag{l}):
   - 'sii_mean_lag1', ..., 'sii_mean_lag7': Discrete daily lags based on 
     Config.DISCRETE_LAGS (1, 2, 3, 4, 5, 6, 7 days).

5. Diagnostic Features:
   - '{var}_diff': First-order differences for stationarity checks.


"""

from __future__ import annotations

import zipfile
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from _Common import Config

warnings.filterwarnings("ignore")


# ----------------------------
# Defaults (overridden via Config.*)
# ----------------------------
DEFAULT_OMNI_AGGREGATIONS = {
    "sii": ["mean", "max", "std"],
    "kp": ["mean", "max"],
    "f10_7": ["mean"],
}

DEFAULT_DST_MIN_VALID = -2000          # filter extremely invalid Dst outliers
DEFAULT_LAG_SMOOTH_WINDOW = 3          # smoothing window for discrete lags
DEFAULT_LAG_SMOOTH_CENTER = True       # centered smoothing


def parse_omni2_dat_line(line: str) -> dict | None:
    """
    Parse a single fixed-format OMNI2 record line.

    Parameters:
        line (str): Raw text line from OMNI2 .dat file.

    Returns:
        dict | None:
            Dictionary with parsed fields:
            - year (int)
            - day (int): Day of year
            - hour (int): Hour (0–23)
            - dst (float | NaN): Dst index
            - kp (float | NaN): Kp index
            - f10_7 (float | NaN): Solar radio flux

            Returns None if the line is malformed or contains invalid metadata.
    """
    try:
        parts = line.split()
        if len(parts) < 51:
            return None

        year = int(parts[0])
        day = int(parts[1])   # day of year
        hour = int(parts[2])

        if not (1900 <= year <= 2100):
            return None

        kp = int(parts[38])
        if kp == 99:
            kp = np.nan

        dst = int(parts[40])
        if dst == 99999:
            dst = np.nan

        f10 = float(parts[50])
        if f10 == 999.9:
            f10 = np.nan

        return {"year": year, "day": day, "hour": hour, "dst": dst, "kp": kp, "f10_7": f10}
    except Exception:
        return None


def main() -> None:
    """
    Run OMNI2 ETL pipeline.

    Steps:
    - Read OMNI2 ZIP archive.
    - Parse hourly records.
    - Compute SII = -Dst.
    - Aggregate to daily statistics.
    - Generate moving averages and lag features.
    - Save result to Feather file.

    Output:
        Config.FILE_OMNI_FEATHER
        Columns:
            date, sii_*, kp_*, f10_7_* and derived lag/MA features.
    """
    if not Config.OMNI_RAW_ZIP.exists():
        raise FileNotFoundError(f"ZIP file not found: {Config.OMNI_RAW_ZIP}")

    # Pull config overrides if present
    aggregations = getattr(Config, "OMNI_AGGREGATIONS", DEFAULT_OMNI_AGGREGATIONS)
    dst_min_valid = getattr(Config, "DST_MIN_VALID", DEFAULT_DST_MIN_VALID)
    lag_smooth_w = int(getattr(Config, "OMNI_LAG_SMOOTH_WINDOW", DEFAULT_LAG_SMOOTH_WINDOW))
    lag_smooth_center = bool(getattr(Config, "OMNI_LAG_SMOOTH_CENTER", DEFAULT_LAG_SMOOTH_CENTER))

    records: list[dict] = []

    with zipfile.ZipFile(Config.OMNI_RAW_ZIP, "r") as z:
        # Choose the largest file inside zip 
        inner_name = max(z.namelist(), key=lambda x: z.getinfo(x).file_size)

        with z.open(inner_name, "r") as f:
            pbar = tqdm(unit="lines", desc="OMNI2: reading & parsing")
            for raw in f:
                try:
                    line = raw.decode("utf-8", errors="ignore").rstrip("\n\r")
                except Exception:
                    pbar.update(1)
                    continue

                rec = parse_omni2_dat_line(line)
                if rec is not None:
                    records.append(rec)

                pbar.update(1)

            pbar.close()

    if not records:
        raise RuntimeError("No valid OMNI2 records parsed (records list is empty).")

    df = pd.DataFrame.from_records(records)

    # Create datetime index from (year, DOY, hour)
    df["datetime"] = pd.to_datetime(
        df["year"].astype(str)
        + df["day"].astype(str).str.zfill(3)
        + df["hour"].astype(str).str.zfill(2),
        format="%Y%j%H",
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.drop(columns=["year", "day", "hour"])

    # Transform SII
    # Filter obvious Dst parsing glitches (extremely low values)
    df = df[df["dst"].isna() | (df["dst"] > dst_min_valid)].copy()
    df["sii"] = -df["dst"]

    # Aggregate daily
    daily = df.resample("D").agg(aggregations)

    # Flatten MultiIndex columns: (var, stat) -> var_stat
    daily.columns = ["_".join(col).strip("_") for col in daily.columns.to_flat_index()]

    # Feature engineering
    num_cols = daily.select_dtypes(include=[np.number]).columns.tolist()

    # Precompute smoothed version for lag generation (optional)
    # Note: we smooth BEFORE shifting to reduce aliasing artifacts in lags.
    if lag_smooth_w >= 2:
        smoothed = daily[num_cols].rolling(window=lag_smooth_w, center=lag_smooth_center, min_periods=1).mean()
    else:
        smoothed = daily[num_cols]

    for col in num_cols:
        # Moving averages (dose) - shift(1) means "use yesterday and before" (no leakage)
        for w in Config.MA_WINDOWS:
            daily[f"{col}_ma{w}"] = daily[col].shift(1).rolling(window=w, min_periods=1).mean()

        # Discrete lags (signal/presensing)
        for l in Config.DISCRETE_LAGS:
            daily[f"{col}_lag{l}"] = smoothed[col].shift(l)

        # First difference (mostly diagnostic; can be useful downstream)
        daily[f"{col}_diff"] = daily[col].diff()

    # Save
    Config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    daily.index.name = "date"
    daily.reset_index().to_feather(Config.FILE_OMNI_FEATHER)


if __name__ == "__main__":
    main()
