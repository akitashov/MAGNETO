#!/usr/bin/env python3
"""
00_preflight.py — MAGNETO Pre-flight Validation.

Checks canonical inputs, environment, and dependencies. No outputs written.
"""
from __future__ import annotations
import sys, platform
from datetime import datetime, timezone
from _Common import *


def _check_exists(path, label):
    if not path.exists():
        print(f"[FAIL] {label} not found: {path}")
        return False
    sz = path.stat().st_size / 1e6
    print(f"[OK]   {label}: {path}  ({sz:.1f} MB)")
    return True


def validate_sif():
    print("\n─── SIF aggregated ───")
    if not _check_exists(FILE_SIF, "SIF"):
        return False
    df = pd.read_feather(FILE_SIF)
    required = ["date", "latitude", "longitude", "lat_id", "lon_id",
                "region_flags", "sif_771nm"]
    missing = set(required) - set(df.columns)
    if missing:
        print(f"[FAIL] SIF missing columns: {missing}")
        return False
    print(f"[OK]   SIF schema: {len(df.columns)} cols, shape={df.shape}")
    n_finite = np.isfinite(df["sif_771nm"]).sum()
    print(f"[INFO] SIF 771 nm finite: {n_finite:,}/{len(df):,}")
    rf = df["region_flags"].values.astype(np.int64)
    n_land = np.sum(((rf & 16) | (rf & 32)) != 0)
    print(f"[INFO] Land observations: {n_land:,}/{len(df):,}")
    return True


def validate_modis():
    print("\n─── MODIS extract ───")
    if not _check_exists(FILE_MODIS, "MODIS"):
        return False
    df = pd.read_parquet(FILE_MODIS, engine=PARQUET_ENGINE)
    required = ["date", "latitude", "longitude", "lai",
                "cloud_fraction", "aerosol_fraction", "lat_id", "lon_id"]
    missing = set(required) - set(df.columns)
    if missing:
        print(f"[FAIL] MODIS missing: {missing}")
        return False
    print(f"[OK]   MODIS schema: {len(df.columns)} cols, shape={df.shape}")
    return True


def validate_omni():
    print("\n─── OMNI biosphere ───")
    if not _check_exists(FILE_OMNI, "OMNI"):
        return False
    df = pd.read_feather(FILE_OMNI)
    for col in ["date", SII_RAW_COL]:
        if col not in df.columns:
            print(f"[FAIL] OMNI missing: {col}")
            return False
    print(f"[OK]   OMNI schema: {len(df.columns)} cols, shape={df.shape}")
    return True


def validate_era5():
    print("\n─── ERA5 environment ───")
    if not _check_exists(FILE_ERA5, "ERA5"):
        return False
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(FILE_ERA5))
    cols = [f.name for f in pf.schema_arrow]
    needed = ["date", "lat_id", "lon_id", ERA5_TEMP_COL]
    missing = [c for c in needed if c not in cols]
    if missing:
        print(f"[FAIL] ERA5 missing: {missing}")
        return False
    print(f"[OK]   ERA5 schema: {len(cols)} cols, {pf.metadata.num_rows:,} rows")
    return True


def validate_mcd12c1():
    """Check that the MCD12C1 land-cover HDF source is available for stage 01b."""
    print("\n─── MCD12C1 land-cover source ───")
    search_dir = PROJECT_ROOT / "data" / "raw" / "MCD12C1" / "2022" / "001"
    if not search_dir.exists():
        print(f"[FAIL] Source directory not found: {search_dir}")
        return False
    hdf_files = sorted(search_dir.glob("*.hdf"))
    if not hdf_files:
        print(f"[FAIL] No HDF files found in {search_dir}")
        return False
    print(f"[OK]   Found {len(hdf_files)} HDF file(s) in {search_dir}")
    for p in hdf_files:
        print(f"       {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    return True


def print_environment():
    print("\n─── Environment ───")
    print(f"  Python:       {platform.python_version()}")
    print(f"  Host:         {platform.node()}")
    print(f"  Timestamp:    {datetime.now(timezone.utc).isoformat()}Z")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT.resolve()}")
    print(f"  Config hash:  {config_hash()[:16]}")
    for pkg in ["yaml", "patsy", "scipy", "statsmodels", "pyarrow"]:
        try:
            __import__(pkg)
            print(f"  package {pkg}: available")
        except ImportError:
            print(f"  package {pkg}: MISSING")


def main() -> int:
    print("=" * 64)
    print("MAGNETO — Pre-flight Validation")
    print("=" * 64)
    print_environment()
    setup_dirs()

    results = {
        "SIF": validate_sif(),
        "MODIS": validate_modis(),
        "OMNI": validate_omni(),
        "ERA5": validate_era5(),
        "MCD12C1": validate_mcd12c1(),
    }
    print("\n" + "=" * 64)
    print("VALIDATION SUMMARY")
    for name, ok in results.items():
        print(f"  {name:10s}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
