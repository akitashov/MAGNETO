#!/usr/bin/env python3
"""
12_environmental_driver_matrix_summary.py — Neutral summary of the driver matrix.

Two modes:

1. Fixed-SII mode (default):
   For a fixed SII window (default 28 days), summarise the distribution of the
   SII partial coefficient across all PAR × VPD combinations within each
   scenario × temperature bin.

2. By-SII-window mode (--by-sii-window):
   For every SII window, summarise the distribution of β_SII across all
   PAR × VPD combinations within each scenario × temperature × SII window group.

No "best model" is selected. The goal is to show whether the sign and
approximate magnitude of β_SII are robust to the choice of PAR and VPD windows.

Output:
    results/environmental_driver_matrix_summary.csv
    results/environmental_driver_matrix_full_summary.csv (when --by-sii-window)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _Common import PROJECT_ROOT, PARQUET_ENGINE, atomic_write


DEFAULT_INPUT = PROJECT_ROOT / "results" / "environmental_driver_matrix.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "environmental_driver_matrix_summary.csv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Summarise environmental-driver matrix")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input Parquet matrix (default: results/environmental_driver_matrix.parquet).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: results/environmental_driver_matrix_summary.csv).",
    )
    p.add_argument(
        "--sii-window",
        type=int,
        default=28,
        help="SII window for fixed-SII mode (default: 28).",
    )
    p.add_argument(
        "--by-sii-window",
        action="store_true",
        help="Produce a row per scenario × temperature × SII window.",
    )
    return p.parse_args(argv)


def _beta_stats(series: pd.Series) -> dict:
    """Return neutral distribution statistics for beta_sii."""
    return {
        "median_beta_sii": float(series.median()),
        "q25_beta_sii": float(np.quantile(series, 0.25)),
        "q75_beta_sii": float(np.quantile(series, 0.75)),
        "p05_beta_sii": float(np.quantile(series, 0.05)),
        "p95_beta_sii": float(np.quantile(series, 0.95)),
        "min_beta_sii": float(series.min()),
        "max_beta_sii": float(series.max()),
        "share_beta_sii_negative": float((series < 0).mean()),
        "n_models": int(series.size),
    }


def _n_stats(series: pd.Series) -> dict:
    return {
        "median_n": float(series.median()),
        "min_n": int(series.min()),
        "max_n": int(series.max()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = args.input if args.input is not None else DEFAULT_INPUT
    output_path = args.output if args.output is not None else DEFAULT_OUTPUT

    print("=" * 64)
    print("MAGNETO — Environmental-driver matrix summary")
    print("=" * 64)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'by SII window' if args.by_sii_window else f'fixed SII window {args.sii_window}'}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input matrix not found:\n{input_path}\n"
            "Run 11_environmental_driver_matrix_gpu.py first."
        )

    df = pd.read_parquet(input_path, engine=PARQUET_ENGINE)
    print(f"Loaded matrix: {len(df):,} rows")

    if not args.by_sii_window:
        df = df[df["sii_window"] == args.sii_window].copy()
        if df.empty:
            raise ValueError(f"No rows for SII window = {args.sii_window}")
        group_keys = ["target", "method", "scenario", "temp_bin_label"]
    else:
        group_keys = ["target", "method", "scenario", "temp_bin_label", "sii_window"]

    beta_summary = (
        df.groupby(group_keys)["beta_sii"]
        .agg(
            median_beta_sii="median",
            q25_beta_sii=lambda x: float(np.quantile(x, 0.25)),
            q75_beta_sii=lambda x: float(np.quantile(x, 0.75)),
            p05_beta_sii=lambda x: float(np.quantile(x, 0.05)),
            p95_beta_sii=lambda x: float(np.quantile(x, 0.95)),
            min_beta_sii="min",
            max_beta_sii="max",
            share_beta_sii_negative=lambda x: float((x < 0).mean()),
            n_models="size",
        )
        .reset_index()
    )
    n_summary = (
        df.groupby(group_keys)["n"]
        .agg(
            median_n="median",
            min_n="min",
            max_n="max",
        )
        .reset_index()
    )
    summary = beta_summary.merge(n_summary, on=group_keys)

    cols = group_keys + [
        "n_models",
        "median_beta_sii",
        "q25_beta_sii",
        "q75_beta_sii",
        "p05_beta_sii",
        "p95_beta_sii",
        "share_beta_sii_negative",
        "min_beta_sii",
        "max_beta_sii",
        "median_n",
        "min_n",
        "max_n",
    ]
    summary = summary[cols].sort_values(group_keys).reset_index(drop=True)

    atomic_write(summary, output_path, fmt="csv")
    print(f"\n[OK] Written: {output_path} ({len(summary):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
