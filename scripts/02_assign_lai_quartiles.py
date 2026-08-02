#!/usr/bin/env python3
"""
02_assign_lai_quartiles.py — LAI cell statistics, quartiles, and control membership.

Uses the FULL MODIS LAI series (not just SIF-coincident dates). Computes per-cell
statistics, assigns static global quartiles, and defines functional and geographic
control memberships. All thresholds are fixed in config/pipeline.yaml before any
SII-SIF correlation is inspected.
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from _Common import *
from magneto_lib import compute_lai_cell_stats, assign_functional_controls, assign_geographic_controls, load_geographic_regions


def main() -> int:
    print("=" * 64)
    print("MAGNETO — LAI Quartiles & Controls")
    print("=" * 64)
    setup_dirs()

    # ── Load FULL MODIS (all dates) ───────────────────────────────────
    print("\n[1/5] Loading full MODIS LAI series …")
    modis = pd.read_parquet(FILE_MODIS, engine=PARQUET_ENGINE,
                            columns=["date", "lat_id", "lon_id", "latitude", "longitude", "lai"])
    n_raw = len(modis)
    print(f"  MODIS rows: {n_raw:,}")
    print(f"  MODIS date range: {modis['date'].min()} → {modis['date'].max()}")
    print(f"  MODIS unique cells: {modis[['lat_id','lon_id']].drop_duplicates().shape[0]:,}")

    modis = modis[modis["lai"].notna() & np.isfinite(modis["lai"])].copy()
    n_lai = len(modis)
    print(f"  Rows with finite LAI: {n_lai:,} ({100*n_lai/n_raw:.1f}%)")

    # ── Per-cell statistics ───────────────────────────────────────────
    print("\n[2/5] Computing per-cell LAI statistics …")
    cells = compute_lai_cell_stats(modis)
    cells["mean_lon"] = modis.groupby(["lat_id", "lon_id"])["longitude"].mean().values
    n_cells = len(cells)
    print(f"  Cells with LAI data: {n_cells:,}")
    print(f"  Median LAI range: [{cells.median_lai.min():.4f}, {cells.median_lai.max():.4f}]")

    print("\n  LAI distribution (per-cell median):")
    for lo, hi, label in [(0, 0.05, "<0.05"), (0.05, 0.15, "0.05-0.15"),
                           (0.15, 0.5, "0.15-0.5"), (0.5, 1.0, "0.5-1.0"),
                           (1.0, 2.0, "1.0-2.0"), (2.0, 100, ">2.0")]:
        n = ((cells.median_lai >= lo) & (cells.median_lai < hi)).sum()
        print(f"    LAI {label:12s}: {n:>8,} cells ({100*n/n_cells:.1f}%)")

    # ── Assign quartiles ──────────────────────────────────────────────
    print("\n[3/5] Assigning LAI quartiles …")
    ranked = cells["median_lai"].rank(method="first")
    cells["lai_quartile"] = pd.qcut(ranked, q=LAI_N_QUARTILES,
                                     labels=[1, 2, 3, 4]).cat.codes + 1

    bounds = {}
    for q in sorted(cells["lai_quartile"].unique()):
        s = cells.loc[cells["lai_quartile"] == q, "median_lai"]
        q_label = f"Q{int(q)}"
        bounds[q_label] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "n_cells": int(len(s)),
        }
        print(f"  {q_label}: LAI [{bounds[q_label]['min']:.4f}, {bounds[q_label]['max']:.4f}]  "
              f"median={bounds[q_label]['median']:.4f}  n={bounds[q_label]['n_cells']:,}")

    for q in range(1, LAI_N_QUARTILES + 1):
        assert q in cells["lai_quartile"].values, f"Missing quartile {q}"

    # ── Assign functional and geographic controls ─────────────────────
    print("\n[4/5] Assigning control memberships …")
    regions = load_geographic_regions()
    cells = assign_functional_controls(cells)
    cells = assign_geographic_controls(cells, regions)

    n_strict_low_lai = int(cells["is_strict_low_lai"].sum())
    n_veg = int(cells["is_vegetated"].sum())
    print(f"  Strict low-LAI control: {n_strict_low_lai:,} cells")
    print(f"  Functional vegetated:   {n_veg:,} cells")
    for name in regions:
        n = int(cells[f"is_{name}"].sum())
        print(f"  Geographic {name}: {n:,} cells")

    # ── Write outputs ─────────────────────────────────────────────────
    print("\n[5/5] Writing outputs …")
    atomic_write(cells, FILE_LAI_CELLS)
    print(f"  {FILE_LAI_CELLS}  ({len(cells):,} rows)")

    atomic_write(bounds, FILE_LAI_BOUNDS)
    print(f"  {FILE_LAI_BOUNDS}")

    members = cells[["lat_id", "lon_id", "lai_quartile",
                     "is_strict_low_lai", "is_vegetated"] +
                    [f"is_{name}" for name in regions]].copy()
    atomic_write(members, FILE_LAI_MEMBERS)
    print(f"  {FILE_LAI_MEMBERS}  ({len(members):,} rows)")

    # Also write as explicit control-membership file for provenance.
    atomic_write(members, FILE_CONTROL_MEMBERS)
    print(f"  {FILE_CONTROL_MEMBERS}  ({len(members):,} rows)")

    # Control diagnostics
    control_diag = {
        "functional": {
            "strict_low_lai": {
                "criteria": f"median_lai <= {CONTROL_STRICT_LOW_LAI_MEDIAN_LAI_MAX} & q90_lai <= {CONTROL_STRICT_LOW_LAI_Q90_LAI_MAX} & n_lai >= {CONTROL_STRICT_LOW_LAI_MIN_LAI_OBS}",
                "n_cells": n_strict_low_lai,
            },
            "vegetated": {
                "criteria": f"median_lai >= {CONTROL_Vegetated_MEDIAN_LAI_MIN} & q10_lai >= {CONTROL_Vegetated_Q10_LAI_MIN} & n_lai >= {CONTROL_Vegetated_MIN_LAI_OBS}",
                "n_cells": n_veg,
            },
        },
        "geographic": {name: int(cells[f"is_{name}"].sum()) for name in regions},
    }
    atomic_write(control_diag, FILE_CONTROL_DEFINITIONS)
    print(f"  {FILE_CONTROL_DEFINITIONS}")

    print(f"\n[OK] 02_assign_lai_quartiles complete — {n_cells:,} cells, "
          f"{LAI_N_QUARTILES} quartiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
