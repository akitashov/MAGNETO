#!/usr/bin/env python3
"""
13_environmental_driver_matrix_full_audit.py — Completeness and duplicate audit
for the full environmental-driver matrix cube.

Reads the combined Parquet produced by 11_environmental_driver_matrix_gpu.py and
verifies that every scenario × temperature × SII window × PAR window × VPD window
combination is present exactly once.

Outputs:
    results/environmental_driver_matrix_full_audit.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _Common import PROJECT_ROOT, PARQUET_ENGINE, full_sha256

DEFAULT_INPUT = PROJECT_ROOT / "results" / "environmental_driver_matrix.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "environmental_driver_matrix_full_audit.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit full environmental-driver matrix")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input Parquet matrix (default: results/environmental_driver_matrix_full.parquet).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output audit JSON (default: results/environmental_driver_matrix_full_audit.json).",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["full_sample", "persistently_vegetated"],
    )
    p.add_argument(
        "--temperature-bins",
        nargs="+",
        default=["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"],
    )
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(range(1, 29)),
    )
    p.add_argument(
        "--expected-missing",
        type=str,
        default=None,
        help='JSON list of [scenario, temp_bin] pairs expected to be empty (e.g. \'[["Sahara","Frozen"]]\').',
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = args.input if args.input is not None else DEFAULT_INPUT
    output_path = args.output if args.output is not None else DEFAULT_OUTPUT

    print("=" * 64)
    print("MAGNETO — Full environmental-driver matrix audit")
    print("=" * 64)
    print(f"Input:  {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input matrix not found:\n{input_path}")

    df = pd.read_parquet(input_path, engine=PARQUET_ENGINE)
    print(f"Loaded matrix: {len(df):,} rows")

    key = ["scenario", "temp_bin_label", "sii_window", "par_window", "vpd_window"]
    for col in key:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    expected = set(
        itertools.product(
            args.scenarios,
            args.temperature_bins,
            args.windows,
            args.windows,
            args.windows,
        )
    )
    observed = set(zip(*[df[c] for c in key]))

    missing = sorted(expected - observed)

    # Some scenario × temperature combinations are scientifically empty (e.g. Sahara/Frozen).
    expected_missing_pairs: set[tuple[str, str]] = set()
    if args.expected_missing:
        try:
            parsed = json.loads(args.expected_missing)
            expected_missing_pairs = {tuple(p) for p in parsed}
        except Exception as exc:
            raise ValueError(f"Invalid --expected-missing JSON: {exc}")

    unexpected_missing = [m for m in missing if (m[0], m[1]) not in expected_missing_pairs]
    duplicates = df[df.duplicated(key, keep=False)]

    # Basic numerical sanity checks.
    bad_beta = df[
        ~df["beta_sii"].between(-1e6, 1e6)
        | ~df["beta_par"].between(-1e6, 1e6)
        | ~df["beta_vpd"].between(-1e6, 1e6)
    ]
    bad_p = df[
        (df["p_sii"] < 0) | (df["p_sii"] > 1)
        | (df["p_par"] < 0) | (df["p_par"] > 1)
        | (df["p_vpd"] < 0) | (df["p_vpd"] > 1)
    ]
    bad_n = df[(df["n"] < 0) | (df["n_eff"] < 0)]

    try:
        input_rel = str(input_path.relative_to(PROJECT_ROOT))
    except ValueError:
        input_rel = str(input_path.resolve())

    audit = {
        "stage": "13_environmental_driver_matrix_full_audit.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": input_rel,
        "input_sha256": full_sha256(input_path),
        "observed_rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
        "expected_missing_rows": int(len(missing) - len(unexpected_missing)),
        "unexpected_missing_rows": int(len(unexpected_missing)),
        "duplicate_rows": int(len(duplicates)),
        "n_beta_out_of_range": int(len(bad_beta)),
        "n_p_out_of_range": int(len(bad_p)),
        "n_n_out_of_range": int(len(bad_n)),
        "scenarios": sorted(df["scenario"].unique().tolist()),
        "temperature_bins": sorted(df["temp_bin_label"].unique().tolist()),
        "sii_windows": sorted(df["sii_window"].unique().tolist()),
        "par_windows": sorted(df["par_window"].unique().tolist()),
        "vpd_windows": sorted(df["vpd_window"].unique().tolist()),
        "expected_missing_pairs": sorted(expected_missing_pairs),
        "missing_combinations": missing[:100],  # cap for readability
        "unexpected_missing_combinations": unexpected_missing[:100],
        "duplicate_keys": duplicates[key].drop_duplicates().sort_values(key).head(100).to_dict(
            orient="records"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    print(f"Expected rows: {audit['expected_rows']:,}")
    print(f"Observed rows: {audit['observed_rows']:,}")
    print(f"Missing:       {audit['missing_rows']:,}")
    print(f"  Expected:    {audit['expected_missing_rows']:,}")
    print(f"  Unexpected:  {audit['unexpected_missing_rows']:,}")
    print(f"Duplicates:    {audit['duplicate_rows']:,}")
    print(f"Bad beta:      {audit['n_beta_out_of_range']:,}")
    print(f"Bad p-values:  {audit['n_p_out_of_range']:,}")
    print(f"Bad N:         {audit['n_n_out_of_range']:,}")
    print(f"\n[OK] Written: {output_path}")

    passed = not unexpected_missing and duplicates.empty and bad_beta.empty and bad_p.empty and bad_n.empty
    print("\nAudit result: PASSED" if passed else "\nAudit result: ISSUES FOUND")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
