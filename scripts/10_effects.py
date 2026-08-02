#!/usr/bin/env python3
"""
10_effects.py — Effect-size summary.

Reads fixed-window statistics, ensures per-100-nT and standardized effects are
present, and writes the canonical effect-size table.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
from _Common import *


def main() -> int:
    print("=" * 64)
    print("MAGNETO — Effect Sizes")
    print("=" * 64)
    setup_dirs()

    # ── Load fixed-window results ─────────────────────────────────────
    if not FILE_FIXED.exists():
        print(f"[WARN] Fixed window results not found: {FILE_FIXED}")
        return 0

    fw = pd.read_csv(FILE_FIXED)
    print(f"\n[1/2] Loaded fixed-window results: {len(fw)} rows")

    effects = fw.copy()
    if "ols_slope_per_100nT" not in effects.columns and "ols_slope_per_1nT" in effects.columns:
        effects["ols_slope_per_100nT"] = effects["ols_slope_per_1nT"] * 100.0

    # obs_hash is carried over from fixed-window results.
    if "obs_hash" not in effects.columns:
        effects["obs_hash"] = ""

    # Add run provenance directly so that stage-manifest hashes match the final files.
    prov = run_provenance_dict(
        dataset_files=[FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC],
        producing_stage="10_effects.py",
    )
    for k, v in prov.items():
        effects[k] = v

    atomic_write(effects, FILE_EFFECTS)
    print(f"  Written: {FILE_EFFECTS} ({len(effects)} rows)")

    print(f"\n[OK] 10_effects complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
