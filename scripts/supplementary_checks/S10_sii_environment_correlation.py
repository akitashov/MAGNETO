#!/usr/bin/env python3
"""
S10_sii_environment_correlation.py — SII vs environmental-driver correlations.

Adaptation of v1/10_SII_PAR_Correlation.py. For selected SII windows it computes,
inside each scenario and temperature bin, the Spearman correlation between SII
and PAR, VPD, and (when present) TCC. This is a diagnostic check: it shows how
much calendar structure SII shares with the environmental drivers that the
multivariate matrix later controls for.

Output:
    results/supplementary_checks/sii_environment_correlation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Parent of the supplementary_checks directory is the scripts directory.
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _Common import PROJECT_ROOT, PARQUET_ENGINE, TEMP_LABELS, atomic_write


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "environmental_driver_input.parquet"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "sii_environment_correlation.csv"
)

WINDOWS = [1, 7, 14, 21, 28]
ENV_PREFIXES = ["par", "vpd"]
MIN_OBS = 50

SCENARIOS = {
    "full_sample": lambda df: pd.Series(True, index=df.index),
    "persistently_vegetated": lambda df: df["is_vegetated"],
    "control_strict_low_lai": lambda df: df["is_strict_low_lai"],
    "Sahara": lambda df: df["is_Sahara"],
    "SAA": lambda df: df["is_SAA"],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SII vs environment correlation check")
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=WINDOWS,
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
    )
    return p.parse_args(argv)


def _neff_factor(x: np.ndarray, y: np.ndarray) -> float:
    """Pyper & Peterman (1998) correction factor from lag-1 autocorrelations."""
    def lag1(a):
        valid = a[np.isfinite(a)]
        if len(valid) < 5:
            return 0.0
        return np.corrcoef(valid[:-1], valid[1:])[0, 1]

    r1_x = np.clip(lag1(x), -0.99, 0.99)
    r1_y = np.clip(lag1(y), -0.99, 0.99)
    prod = r1_x * r1_y
    return (1 - prod) / (1 + prod)


def _p_from_rho(rho: float, n_eff: float) -> float:
    if abs(rho) >= 1.0:
        return 0.0
    if n_eff <= 2:
        return 1.0
    t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho ** 2))
    return float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_eff - 2)))


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 64)
    print("MAGNETO supplementary checks — SII vs environment correlation")
    print("=" * 64)
    print(f"Windows: {args.windows}")
    print(f"Scenarios: {args.scenarios}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{INPUT_PATH}\n"
            "Run 10_prepare_environmental_driver_input.py first."
        )

    df = pd.read_parquet(INPUT_PATH, engine=PARQUET_ENGINE)
    # Discover which environmental prefixes are actually present.
    prefixes = [p for p in ENV_PREFIXES if any(c.startswith(f"{p}_ma") for c in df.columns)]
    if "tcc" in df.columns or any(c.startswith("tcc_ma") for c in df.columns):
        prefixes.append("tcc")
    print(f"Environmental drivers present: {prefixes}")

    rows = []
    for scenario in tqdm(args.scenarios, desc="Scenarios"):
        scen_df = df[SCENARIOS[scenario](df)].copy()
        if scen_df.empty:
            continue

        for temp in TEMP_LABELS:
            bin_df = scen_df[scen_df["temp_bin_label"] == temp]
            if len(bin_df) < MIN_OBS:
                continue

            for w in args.windows:
                sii_col = f"sii_mean_ma{w}"
                if sii_col not in bin_df.columns:
                    continue
                for prefix in prefixes:
                    env_col = f"{prefix}_ma{w}"
                    if env_col not in bin_df.columns:
                        continue

                    pair = bin_df[[sii_col, env_col]].dropna()
                    if len(pair) < MIN_OBS:
                        continue

                    x = pair[sii_col].values.astype(float)
                    y = pair[env_col].values.astype(float)

                    rho, _ = stats.spearmanr(x, y)
                    n_raw = len(pair)
                    neff_f = _neff_factor(x, y)
                    n_eff = max(2, n_raw * neff_f)
                    p_val = _p_from_rho(rho, n_eff)

                    rows.append({
                        "scenario": scenario,
                        "temp_bin_label": temp,
                        "window": w,
                        "driver": prefix,
                        "sii_col": sii_col,
                        "env_col": env_col,
                        "rho": float(rho),
                        "n_obs": n_raw,
                        "n_eff": float(n_eff),
                        "p_value": p_val,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[WARN] No correlations met the minimum-observation threshold")
        return 0

    out = out.sort_values(["scenario", "temp_bin_label", "driver", "window"]).reset_index(drop=True)
    atomic_write(out, OUTPUT_PATH, fmt="csv")
    print(f"\n[OK] Written: {OUTPUT_PATH} ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
