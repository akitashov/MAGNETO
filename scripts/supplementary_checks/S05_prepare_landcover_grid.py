#!/usr/bin/env python3
"""
S05_prepare_landcover_grid.py — Aggregate a stable global land-cover product onto
the project analysis grid.

The script expects a NetCDF4 file (configurable in config/supplementary_checks.yaml) with
variables:
    lat   : source latitude centres
    lon   : source longitude centres
    land_cover : integer IGBP class codes (1-17)

Each project grid cell is taken from the QC dataset (lat_id, lon_id), with
lat = lat_id/100 and lon = lon_id/100.  The cell half-width is inferred from
unique coordinate spacing in the QC cells, so the aggregation matches the
project's 0.5° × 0.5° analysis grid when the QC cells are on that grid.  The
dominant land-cover class is assigned when its share of valid source pixels
inside the cell is at least `landcover.majority_threshold`; otherwise the cell
is labelled "Mixed/Other Vegetation".
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _supplementary_checks_common import (
    SUPPLEMENTARY_CHECKS_DATA,
    SUPPLEMENTARY_CHECKS_RESULTS,
    make_supplementary_checks_dirs,
    atomic_write,
    attach_provenance,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare land-cover grid")
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--restrict-to-qc-cells", action="store_true", default=True)
    return p.parse_args(argv)


def _source_missing_message(cfg: dict) -> str:
    src = cfg["landcover"]["source_path"]
    return (
        f"\n[SKIP] Land-cover source file not found: {src}\n"
        "\nThis script requires a stable global land-cover classification on a\n"
        "regular lat/lon grid. The default configuration expects a NetCDF4 file\n"
        "with variables 'lat', 'lon', and 'land_cover' (integer IGBP classes).\n"
        "\nSuggested source: MODIS MCD12C1 Version 6.1 IGBP classification,\n"
        "e.g. MCD12C1.A2023001.061.*.hdf re-projected to 0.05-degree NetCDF.\n"
        "Place the file at the path above and re-run S05.\n"
    )


def _read_netcdf_landcover(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (lat, lon, land_cover_2d, fill_value)."""
    try:
        import xarray as xr
    except ImportError:
        xr = None

    def _to_int_array(arr: np.ndarray, fill: float) -> np.ndarray:
        arr = np.asarray(arr)
        # Replace masked/NaN values with the fill value before casting to int.
        if arr.dtype.kind == "f":
            arr = np.where(np.isnan(arr), fill, arr)
        elif hasattr(arr, "mask"):
            arr = arr.filled(fill)
        return arr.astype(int)

    if xr is not None:
        ds = xr.open_dataset(path)
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
        lc = ds["land_cover"].values
        if lc.ndim == 3:
            lc = lc[0]
        fill = ds["land_cover"].attrs.get("_FillValue", np.nan)
        if np.isnan(fill):
            fill = -1
        return lat, lon, _to_int_array(lc, fill), float(fill)

    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        lat = ds.variables["lat"][:].astype(float)
        lon = ds.variables["lon"][:].astype(float)
        lc_var = ds.variables["land_cover"]
        lc_raw = lc_var[:]
        fill = getattr(lc_var, "_FillValue", -1)
        if fill is None or np.isnan(fill):
            fill = -1
        lc = _to_int_array(lc_raw, fill)
        if lc.ndim == 3:
            lc = lc[0]
        return lat, lon, lc, float(fill)


def _validate_georeferencing(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    lc: np.ndarray,
    fill: float,
    igbp_map: dict,
) -> list[dict]:
    """Check known locations against expected dominant land-cover classes."""
    # (lat, lon, expected mapped dominant class or classes)
    checks = [
        (-3.0, -60.0, {"Forest"}, "Amazon"),
        (23.0, 5.0, {"Barren"}, "Sahara"),
        (50.0, 14.0, {"Forest", "Cropland"}, "Central Europe"),
    ]
    results = []
    for lat0, lon0, expected, name in checks:
        lat_mask = (src_lat >= lat0 - 0.5) & (src_lat <= lat0 + 0.5)
        lon_mask = (src_lon >= lon0 - 0.5) & (src_lon <= lon0 + 0.5)
        if not (lat_mask.any() and lon_mask.any()):
            results.append({"name": name, "ok": False, "reason": "no source pixels"})
            continue
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        sub = lc[np.ix_(lat_idx, lon_idx)]
        valid = sub[(sub != fill) & (sub >= 1) & (sub <= 17)]
        if valid.size == 0:
            results.append({"name": name, "ok": False, "reason": "all NoData"})
            continue
        dominant_igbp = int(pd.Series(valid).mode().iloc[0])
        dominant_mapped = igbp_map.get(dominant_igbp, "Mixed/Other Vegetation")
        ok = dominant_mapped in expected
        results.append({
            "name": name,
            "ok": ok,
            "dominant_igbp": dominant_igbp,
            "dominant_mapped": dominant_mapped,
            "expected": "/".join(sorted(expected)),
            "reason": "" if ok else f"expected {expected}, got {dominant_mapped}",
        })
    return results


def _project_grid_cells(restrict: bool = True) -> pd.DataFrame:
    """Return target project grid cells as (lat_id, lon_id, lat, lon)."""
    if restrict:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _Common import FILE_QC
        import pyarrow.parquet as pq
        qc = pq.read_table(FILE_QC).to_pandas()
        cells = qc[["lat_id", "lon_id"]].drop_duplicates()
        cells["lat"] = cells["lat_id"] / 100.0
        cells["lon"] = cells["lon_id"] / 100.0
        return cells

    lats = np.arange(-60, 75.5, 0.5)
    lons = np.arange(-180, 180, 0.5)
    grid = np.array([(la, lo) for la in lats for lo in lons])
    df = pd.DataFrame({"lat": grid[:, 0], "lon": grid[:, 1]})
    df["lat_id"] = (df["lat"] * 100).astype(int)
    df["lon_id"] = (df["lon"] * 100).astype(int)
    return df[["lat_id", "lon_id", "lat", "lon"]]


def _cell_half_width(cells: pd.DataFrame) -> float:
    """Infer the project grid cell half-width from unique coordinate spacing."""
    lat_steps = cells["lat"].drop_duplicates().sort_values().diff().dropna()
    lon_steps = cells["lon"].drop_duplicates().sort_values().diff().dropna()
    step = min(float(lat_steps.min()), float(lon_steps.min()))
    if not np.isfinite(step) or step <= 0:
        return 0.5
    return step / 2.0


def _aggregate_cell(
    lat: float,
    lon: float,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    lc: np.ndarray,
    fill: float,
    igbp_map: dict,
    majority_threshold: float,
    half_width: float = 0.5,
) -> dict:
    """Aggregate source land-cover pixels into one project cell."""
    lat_mask = (src_lat >= lat - half_width) & (src_lat <= lat + half_width)
    lon_mask = (src_lon >= lon - half_width) & (src_lon <= lon + half_width)
    if not (lat_mask.any() and lon_mask.any()):
        return {"land_cover_class": "No_Data", "dominant_fraction": np.nan, "n_pixels": 0}

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    sub = lc[np.ix_(lat_idx, lon_idx)]
    valid = sub[(sub != fill) & (sub >= 1) & (sub <= 17)]
    if valid.size == 0:
        return {"land_cover_class": "No_Data", "dominant_fraction": np.nan, "n_pixels": 0}

    # Map IGBP codes to target classes.
    mapped = np.array([igbp_map.get(int(v), "Mixed/Other Vegetation") for v in valid])
    classes, counts = np.unique(mapped, return_counts=True)
    total = counts.sum()
    dominant_idx = int(counts.argmax())
    dominant_class = classes[dominant_idx]
    dominant_count = counts[dominant_idx]
    dominant_fraction = dominant_count / total

    assigned_class = (
        dominant_class
        if dominant_fraction >= majority_threshold
        else "Mixed/Other Vegetation"
    )
    result = {
        "land_cover_class": assigned_class,
        "dominant_fraction": dominant_fraction,
        "n_pixels": int(total),
    }
    for cls, cnt in zip(classes, counts):
        result[f"fraction_{cls}"] = cnt / total
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    print("=" * 64)
    print("MAGNETO supplementary checks — Prepare land-cover grid")
    print("=" * 64)

    source_path = args.source if args.source else Path(cfg["landcover"]["source_path"])
    source_path = source_path.expanduser()
    if not source_path.is_absolute():
        source_path = Path(__file__).resolve().parents[2] / source_path

    if not source_path.exists():
        print(_source_missing_message(cfg))
        # Write a small placeholder diagnostics file so downstream scripts can
        # detect the missing grid unambiguously.
        placeholder = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["landcover_diagnostics_csv"]
        atomic_write(
            pd.DataFrame({
                "status": ["source_missing"],
                "expected_source": [str(source_path)],
            }),
            placeholder,
        )
        return 0

    igbp_map = cfg["landcover"]["igbp_map"]
    majority_threshold = cfg["landcover"]["majority_threshold"]
    source_year = cfg["landcover"].get("source_year", "unknown")
    source_product = cfg["landcover"].get("source_product", "unknown")

    print(f"[1/4] Reading land-cover source: {source_path}")
    src_lat, src_lon, lc, fill = _read_netcdf_landcover(source_path)
    print(f"  Source shape: {lc.shape}, fill={fill}")

    # Ensure ascending source coordinates (the canonical MCD12C1 CMG orientation
    # has row 0 at the north pole; hdf2netcdf.py reverses this, but we keep the
    # reader robust to either order).
    if src_lat[-1] < src_lat[0]:
        src_lat = src_lat[::-1]
        lc = lc[::-1, :]
    if src_lon[-1] < src_lon[0]:
        src_lon = src_lon[::-1]
        lc = lc[:, ::-1]

    print("[2/4] Validating georeferencing ...")
    val = _validate_georeferencing(src_lat, src_lon, lc, fill, igbp_map)
    for v in val:
        status = "OK" if v["ok"] else "FAIL"
        detail = f"igbp={v.get('dominant_igbp')} mapped={v.get('dominant_mapped')} expected={v.get('expected')}"
        if not v["ok"]:
            detail += f" ({v.get('reason')})"
        print(f"  {status}: {v['name']} ({detail})")
    if not all(v["ok"] for v in val):
        print("\n[ERROR] Georeferencing check failed. The source NetCDF may have an incorrect latitude assignment.")
        return 1

    print("[3/4] Building target grid ...")
    target = _project_grid_cells(restrict=args.restrict_to_qc_cells)
    half_width = _cell_half_width(target)
    print(f"  Target cells: {len(target):,}")
    print(f"  Inferred cell half-width: {half_width}°")

    print("[4/4] Aggregating land-cover classes ...")
    rows = []
    for _, row in target.iterrows():
        agg = _aggregate_cell(
            row["lat"], row["lon"], src_lat, src_lon, lc, fill,
            igbp_map, majority_threshold, half_width=half_width,
        )
        rows.append({
            "lat_id": row["lat_id"],
            "lon_id": row["lon_id"],
            "lat": row["lat"],
            "lon": row["lon"],
            **agg,
        })

    df = pd.DataFrame(rows)
    out_grid = SUPPLEMENTARY_CHECKS_DATA / cfg["outputs"]["landcover_grid_parquet"]
    df_out = df[[
        "lat_id", "lon_id", "lat", "lon", "land_cover_class",
        "dominant_fraction", "n_pixels",
    ]]
    atomic_write(df_out, out_grid)
    print(f"\n  Written: {out_grid}")

    # Diagnostics.
    diag = (
        df[df["land_cover_class"] != "No_Data"]
        .groupby("land_cover_class", as_index=False)
        .agg(n_cells=("lat_id", "count"))
    )
    total_cells = int((df["land_cover_class"] != "No_Data").sum())
    diag["fraction_of_cells"] = diag["n_cells"] / total_cells if total_cells else np.nan
    diag = diag.sort_values("n_cells", ascending=False)

    out_diag = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["landcover_diagnostics_csv"]
    diag["source_product"] = source_product
    diag["source_year"] = source_year
    diag["note"] = (
        "Land-cover class is the dominant 2022 MCD12C1 IGBP class per 0.5° "
        "project cell; it is a snapshot, not evidence of stable vegetation type "
        "throughout the 2014-2024 study period."
    )
    diag = attach_provenance(
        diag,
        stage="S05_prepare_landcover_grid.py",
        dataset_files=[source_path],
    )
    atomic_write(diag, out_diag)
    print(f"  Written: {out_diag}")
    print(diag[["land_cover_class", "n_cells", "fraction_of_cells"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
