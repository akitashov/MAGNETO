#!/usr/bin/env python3
"""
Supplementary Figure S1. Spatial coverage and control regions.

Purpose
-------
Show the spatial distribution of the 0.5°×0.5° analytical grid, biological samples,
and geographic controls used in the MAGNETO project.

Panels
------
A. Full analytical coverage (all unique lat_id/lon_id cells).
B. Persistently vegetated cells (final audited definition:
   median LAI ≥ 1.0, 10th-percentile LAI ≥ 0.30, ≥ 10 valid LAI observations).
C. Barren land-cover class (dominant 2022 MCD12C1 land-cover class == Barren,
   matching the pooled `landcover_Barren` analysis).
D. Sahara control (geographic barren control defined by the Sahara bounding box).
E. South Atlantic Anomaly control (geographic control defined by the SAA bounding
   box).
F. Dominant land-cover class for cells in the analysis grid, colored by the
   six aggregated land-cover classes used in the main analysis.

Inputs
------
data/processed/environmental_driver_input.parquet
    Columns include: lat_id, lon_id, is_vegetated, is_Sahara, is_SAA,
    temp_bin_label.
data/interim/supplementary_checks/landcover_cell_grid.parquet
    Columns: lat_id, lon_id, lat, lon, land_cover_class, dominant_fraction,
    n_pixels.
results/supplementary_checks/landcover_analysis.csv
    Pooled land-cover analysis; used to verify the Barren land-cover cell count.
results/geographic_control_diagnostics.csv
    Sahara diagnostic counts.
results/supplementary_checks/saa_control.csv
    SAA diagnostic counts.

Outputs
-------
reports/figures/figureS1_spatial_coverage_controls.png
reports/figures/figureS1_spatial_coverage_controls_source.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

ENV_DRIVER_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "environmental_driver_input.parquet"
)
LANDCOVER_GRID_INPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "supplementary_checks"
    / "landcover_cell_grid.parquet"
)
LANDCOVER_ANALYSIS_CSV = (
    PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis.csv"
)
GEOGRAPHIC_CONTROL_DIAGNOSTICS_CSV = (
    PROJECT_ROOT / "results" / "geographic_control_diagnostics.csv"
)
SAA_CONTROL_CSV = (
    PROJECT_ROOT / "results" / "supplementary_checks" / "saa_control.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS1_spatial_coverage_controls.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS1_spatial_coverage_controls.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS1_spatial_coverage_controls_source.csv"


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
    }
)

# Colorblind-friendly single-color choices for binary panels.
PANEL_COLORS = {
    "A": "#0072B2",  # blue
    "B": "#009E73",  # green
    "C": "#E69F00",  # orange
    "D": "#CC79A7",  # pink
    "E": "#56B4E9",  # light blue
}

# Six aggregated land-cover classes and a colorblind-friendly palette.
# Order places barren first, then increasingly vegetated classes, ending with
# forest, matching the convention used elsewhere in the project.
LC_ORDER = [
    "Barren",
    "Shrubland/Savanna",
    "Savanna",
    "Grassland",
    "Cropland",
    "Forest",
]

LC_COLORS = {
    "Barren": "#999999",
    "Shrubland/Savanna": "#8c6d31",
    "Savanna": "#d8a400",
    "Grassland": "#66a61e",
    "Cropland": "#c28e0e",
    "Forest": "#1b7837",
}

LC_LABELS = {
    "Barren": "Barren",
    "Shrubland/Savanna": "Shrubland/\nSavanna",
    "Savanna": "Savanna",
    "Grassland": "Grassland",
    "Cropland": "Cropland",
    "Forest": "Forest",
}

# Classes that are not part of the six aggregated classes are excluded from
# panel F, consistent with the land-cover analysis pipeline.
LC_EXCLUDED = {
    "No_Data",
    "Water",
    "Urban",
    "Snow/Ice",
    "Wetlands",
    "Mixed/Other Vegetation",
}

FIGSIZE = (11.5, 4.9)
DPI = 400
MARKER_SIZE = 6
GLOBAL_EXTENT = (-180.0, 180.0, -60.0, 75.0)


# =============================================================================
# CARTOPY / COASTLINE HANDLING
# =============================================================================

CARTOPY_AVAILABLE = False
CARTOPY = None
CCRS = None
FEATURE = None


def _import_cartopy() -> bool:
    """Import cartopy if installed; otherwise return False."""
    global CARTOPY_AVAILABLE, CARTOPY, CCRS, FEATURE
    if CARTOPY_AVAILABLE:
        return True
    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        CARTOPY = cartopy
        CCRS = ccrs
        FEATURE = cfeature
        CARTOPY_AVAILABLE = True
        return True
    except ImportError:
        return False


def _natural_earth_coastline_available(resolution: str = "110m") -> bool:
    """Return True if the Natural Earth coastline shapefile is already local.

    This check deliberately avoids triggering any cartopy download.
    """
    if not CARTOPY_AVAILABLE:
        return False
    data_dir = Path(CARTOPY.config.get("data_dir", CARTOPY.config["repo_data_dir"]))
    shape_path = (
        data_dir
        / "shapefiles"
        / "natural_earth"
        / "physical"
        / f"ne_{resolution}_coastline.shp"
    )
    return shape_path.exists()


def _add_coastlines(ax) -> None:
    """Add Natural Earth coastlines if locally available; otherwise warn."""
    if not _natural_earth_coastline_available("110m"):
        warnings.warn(
            "Natural Earth 110m coastline data is not available locally; "
            "coastlines will not be drawn. No basemaps will be downloaded.",
            UserWarning,
            stacklevel=2,
        )
        return
    ax.coastlines(resolution="110m", color="#333333", linewidth=0.5)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_environmental_cells(path: Path) -> pd.DataFrame:
    """Return unique analytical grid cells with derived lat/lon and flags."""
    df = pd.read_parquet(path)
    required = {"lat_id", "lon_id", "is_vegetated", "is_Sahara", "is_SAA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in environmental_driver_input: {missing}")

    cells = (
        df.groupby(["lat_id", "lon_id"], as_index=False)
        .agg(
            is_vegetated=("is_vegetated", "any"),
            is_Sahara=("is_Sahara", "any"),
            is_SAA=("is_SAA", "any"),
            temp_bin_label=("temp_bin_label", lambda s: ";".join(sorted(s.dropna().astype(str).unique()))),
        )
    )
    cells["lat"] = cells["lat_id"] / 100.0
    cells["lon"] = cells["lon_id"] / 100.0

    # Verify that the persistent-vegetation mask matches the audited control
    # definition (median LAI ≥ 1.0, q10 LAI ≥ 0.30, n_LAI ≥ 10).
    diagnostics_path = PROJECT_ROOT / "results" / "control_diagnostics.csv"
    if diagnostics_path.exists():
        diag = pd.read_csv(diagnostics_path)
        veg_diag = diag[diag["control"] == "Persistently_Vegetated"]
        if not veg_diag.empty:
            expected_cells = int(veg_diag.iloc[0]["n_analysis_cells_harmonic"])
            expected_obs = int(veg_diag.iloc[0]["n_analysis_obs_harmonic"])
            actual_cells = int(cells["is_vegetated"].sum())
            actual_obs = int(df["is_vegetated"].sum())
            if actual_cells != expected_cells or actual_obs != expected_obs:
                raise ValueError(
                    "Persistently vegetated mask in environmental_driver_input "
                    "does not match the audited control definition.\n"
                    f"Expected cells/obs: {expected_cells:,} / {expected_obs:,}\n"
                    f"Actual cells/obs:   {actual_cells:,} / {actual_obs:,}\n"
                    "Stop and inspect the vegetated definition rather than silently relabelling."
                )

    # Verify Sahara and SAA counts against their analytical diagnostics.
    if GEOGRAPHIC_CONTROL_DIAGNOSTICS_CSV.exists():
        geo_diag = pd.read_csv(GEOGRAPHIC_CONTROL_DIAGNOSTICS_CSV)
        sahara_diag = geo_diag[geo_diag["region"] == "Sahara"]
        if not sahara_diag.empty:
            expected_sahara = int(sahara_diag.iloc[0]["n_analysis_cells_harmonic"])
            actual_sahara = int(cells["is_Sahara"].sum())
            if actual_sahara != expected_sahara:
                raise ValueError(
                    "Sahara mask in environmental_driver_input does not match "
                    "the audited geographic control diagnostics.\n"
                    f"Expected cells: {expected_sahara:,}\n"
                    f"Actual cells:   {actual_sahara:,}\n"
                    "Stop and inspect the Sahara definition rather than silently relabelling."
                )

    if SAA_CONTROL_CSV.exists():
        saa_diag = pd.read_csv(SAA_CONTROL_CSV)
        saa_pooled = saa_diag[saa_diag["sample_type"] == "control_SAA"]
        if not saa_pooled.empty:
            expected_saa = int(saa_pooled["n_cells"].max())
            actual_saa = int(cells["is_SAA"].sum())
            if actual_saa != expected_saa:
                raise ValueError(
                    "SAA mask in environmental_driver_input does not match "
                    "the audited SAA control analysis.\n"
                    f"Expected cells: {expected_saa:,}\n"
                    f"Actual cells:   {actual_saa:,}\n"
                    "Stop and inspect the SAA definition rather than silently relabelling."
                )

    return cells


def load_barren_landcover_cells(analysis_cell_ids: set[tuple[int, int]]) -> pd.DataFrame:
    """Return the exact Barren land-cover cells used in the pooled analysis.

    Cells are taken from the dominant 2022 MCD12C1 land-cover grid, restricted
    to the six aggregated classes, and then intersected with the analytical
    0.5°×0.5° grid. The resulting count is verified against
    results/supplementary_checks/landcover_analysis.csv (sample_type
    == "landcover_Barren").
    """
    if not LANDCOVER_GRID_INPUT.exists():
        raise FileNotFoundError(f"Land-cover grid not found: {LANDCOVER_GRID_INPUT}")

    lc = pd.read_parquet(LANDCOVER_GRID_INPUT)
    required = {"lat_id", "lon_id", "lat", "lon", "land_cover_class"}
    missing = required - set(lc.columns)
    if missing:
        raise ValueError(f"Missing columns in landcover_cell_grid: {missing}")

    lc = lc.dropna(subset=["lat_id", "lon_id", "land_cover_class"]).copy()
    lc["lat_id"] = lc["lat_id"].astype(int)
    lc["lon_id"] = lc["lon_id"].astype(int)
    lc = lc.loc[~lc["land_cover_class"].isin(LC_EXCLUDED)].copy()

    barren = lc.loc[lc["land_cover_class"] == "Barren"].copy()
    barren = barren.loc[
        barren.apply(lambda r: (int(r["lat_id"]), int(r["lon_id"])) in analysis_cell_ids, axis=1)
    ].copy()

    if LANDCOVER_ANALYSIS_CSV.exists():
        la = pd.read_csv(LANDCOVER_ANALYSIS_CSV)
        pooled_barren = la[la["sample_type"] == "landcover_Barren"]
        if not pooled_barren.empty:
            expected_cells = int(pooled_barren["n_cells"].max())
            actual_cells = len(barren)
            if actual_cells != expected_cells:
                raise ValueError(
                    "Barren land-cover cell count does not match the pooled "
                    "landcover_Barren analysis.\n"
                    f"Expected cells: {expected_cells:,}\n"
                    f"Actual cells:   {actual_cells:,}\n"
                    "Stop and inspect the land-cover definition rather than silently relabelling."
                )

    return barren[["lat_id", "lon_id", "lat", "lon"]].copy()


def load_landcover_grid(path: Path) -> pd.DataFrame:
    """Return land-cover grid, keeping only the six aggregated classes."""
    df = pd.read_parquet(path)
    required = {"lat_id", "lon_id", "lat", "lon", "land_cover_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in landcover_cell_grid: {missing}")

    df = df.dropna(subset=["lat_id", "lon_id", "land_cover_class"]).copy()
    df["lat_id"] = df["lat_id"].astype(int)
    df["lon_id"] = df["lon_id"].astype(int)
    df = df.loc[~df["land_cover_class"].isin(LC_EXCLUDED)].copy()
    return df


# =============================================================================
# PLOTTING
# =============================================================================


def _plot_panel(ax, lons: np.ndarray, lats: np.ndarray, color: str, title: str) -> None:
    """Scatter cells on a global PlateCarree axis."""
    ax.set_extent(GLOBAL_EXTENT, crs=CCRS.PlateCarree())
    ax.scatter(
        lons,
        lats,
        s=MARKER_SIZE,
        c=color,
        marker="s",
        edgecolors="none",
        transform=CCRS.PlateCarree(),
        rasterized=True,
    )
    _add_coastlines(ax)
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _plot_landcover_panel(ax, df: pd.DataFrame, title: str) -> list:
    """Plot panel F with categorical land-cover colors and return legend handles."""
    ax.set_extent(GLOBAL_EXTENT, crs=CCRS.PlateCarree())
    handles = []
    for i, lc in enumerate(LC_ORDER):
        sub = df.loc[df["land_cover_class"] == lc]
        if sub.empty:
            continue
        color = LC_COLORS.get(lc, "#333333")
        ax.scatter(
            sub["lon"].values,
            sub["lat"].values,
            s=MARKER_SIZE,
            c=color,
            marker="s",
            edgecolors="none",
            transform=CCRS.PlateCarree(),
            rasterized=True,
            zorder=i,
        )
        handles.append(matplotlib.patches.Patch(color=color, label=LC_LABELS.get(lc, lc)))
    _add_coastlines(ax)
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return handles


def make_figure(
    cells: pd.DataFrame,
    barren_cells: pd.DataFrame,
    landcover: pd.DataFrame,
) -> tuple[plt.Figure, list]:
    """Create the 2×3 figure and return the figure plus legend handles for F."""
    fig = plt.figure(figsize=FIGSIZE)

    if CARTOPY_AVAILABLE:
        subplot_kw = {"projection": CCRS.PlateCarree()}
    else:
        subplot_kw = {}

    axes = fig.subplots(
        2,
        3,
        subplot_kw=subplot_kw,
        gridspec_kw={"hspace": 0.06, "wspace": 0.08},
    )
    axes_flat = axes.flatten()

    panels = [
        ("A", cells, "full_coverage", "A. Full analytical coverage", PANEL_COLORS["A"]),
        ("B", cells.loc[cells["is_vegetated"]], "vegetated", "B. Persistently vegetated", PANEL_COLORS["B"]),
        ("C", barren_cells, "landcover_barren", "C. Barren land-cover class", PANEL_COLORS["C"]),
        ("D", cells.loc[cells["is_Sahara"]], "Sahara", "D. Sahara control", PANEL_COLORS["D"]),
        ("E", cells.loc[cells["is_SAA"]], "SAA", "E. South Atlantic Anomaly control", PANEL_COLORS["E"]),
    ]

    for ax, (letter, sub, _, title, color) in zip(axes_flat, panels):
        _plot_panel(ax, sub["lon"].values, sub["lat"].values, color, title)

    # Panel F
    ax_f = axes_flat[5]
    legend_handles = _plot_landcover_panel(
        ax_f, landcover, "F. Dominant land-cover class"
    )

    # Panel labels A–F are already in titles. Add a shared legend for panel F
    # below the figure rather than inside the map panel.
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            title="Land-cover class",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=6,
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    return fig, legend_handles


# =============================================================================
# SOURCE CSV
# =============================================================================


def build_source_csv(
    cells: pd.DataFrame,
    barren_cells: pd.DataFrame,
    landcover: pd.DataFrame,
) -> pd.DataFrame:
    """Build the source-data CSV for the figure."""
    records = []
    figure_id = "figureS1_spatial_coverage_controls"
    env_source = str(ENV_DRIVER_INPUT.relative_to(PROJECT_ROOT))
    lc_source = str(LANDCOVER_GRID_INPUT.relative_to(PROJECT_ROOT))

    def add_panel(panel_id: str, display_order: int, sub: pd.DataFrame, flag: str, source: str):
        for _, row in sub.iterrows():
            records.append(
                {
                    "figure_id": figure_id,
                    "panel": panel_id,
                    "display_order": display_order,
                    "source_file": source,
                    "lat_id": int(row["lat_id"]),
                    "lon_id": int(row["lon_id"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "flag_or_class": flag,
                    "temp_bin_label": row.get("temp_bin_label", ""),
                }
            )

    add_panel("A", 1, cells, "full_coverage", env_source)
    add_panel("B", 2, cells.loc[cells["is_vegetated"]], "is_vegetated", env_source)
    add_panel("C", 3, barren_cells, "landcover_barren", lc_source)
    add_panel("D", 4, cells.loc[cells["is_Sahara"]], "is_Sahara", env_source)
    add_panel("E", 5, cells.loc[cells["is_SAA"]], "is_SAA", env_source)

    for _, row in landcover.iterrows():
        records.append(
            {
                "figure_id": figure_id,
                "panel": "F",
                "display_order": 6,
                "source_file": lc_source,
                "lat_id": int(row["lat_id"]),
                "lon_id": int(row["lon_id"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "flag_or_class": row["land_cover_class"],
                "temp_bin_label": "",
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# MAIN
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    print("=" * 64)
    print("Supplementary Figure S1 — Spatial coverage and control regions")
    print("=" * 64)

    # Input paths
    print(f"Input 1: {ENV_DRIVER_INPUT}")
    print(f"Input 2: {LANDCOVER_GRID_INPUT}")

    if not ENV_DRIVER_INPUT.exists():
        print(f"ERROR: input not found: {ENV_DRIVER_INPUT}", file=sys.stderr)
        return 1
    if not LANDCOVER_GRID_INPUT.exists():
        print(f"ERROR: input not found: {LANDCOVER_GRID_INPUT}", file=sys.stderr)
        return 1

    # Load data
    print("\nLoading environmental driver cells ...")
    cells = load_environmental_cells(ENV_DRIVER_INPUT)
    print(f"  Total unique cells: {len(cells):,}")

    print("Loading land-cover grid ...")
    landcover = load_landcover_grid(LANDCOVER_GRID_INPUT)
    print(f"  Land-cover cells (six aggregated classes): {len(landcover):,}")

    # Restrict landcover to cells present in the analytical grid
    analysis_ids = set(zip(cells["lat_id"], cells["lon_id"]))
    landcover = landcover.loc[
        landcover.apply(lambda r: (int(r["lat_id"]), int(r["lon_id"])) in analysis_ids, axis=1)
    ].copy()
    print(f"  After joining to analysis grid: {len(landcover):,}")

    # Barren land-cover cells come from the dominant MCD12C1 class, not the
    # legacy is_barren geographic flag.
    print("Loading Barren land-cover class cells ...")
    barren_cells = load_barren_landcover_cells(analysis_ids)
    print(f"  Barren land-cover cells: {len(barren_cells):,}")

    # Counts per panel
    print("\nPanel counts:")
    print(f"  A. Full analytical coverage:      {len(cells):,}")
    print(f"  B. Persistently vegetated:        {cells['is_vegetated'].sum():,}")
    print(f"  C. Barren land-cover class:       {len(barren_cells):,}")
    print(f"  D. Sahara control:                {cells['is_Sahara'].sum():,}")
    print(f"  E. SAA control:                   {cells['is_SAA'].sum():,}")
    print(f"  F. Dominant land-cover:           {len(landcover):,}")
    for lc in LC_ORDER:
        n = (landcover["land_cover_class"] == lc).sum()
        print(f"      - {lc}: {n:,}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cartopy availability
    has_cartopy = _import_cartopy()
    if has_cartopy:
        print("\nCartopy is available; using PlateCarree projection.")
        coastline_ok = _natural_earth_coastline_available("110m")
        print(f"Natural Earth 110m coastline local: {coastline_ok}")
    else:
        print("\nWARNING: cartopy is not installed; plotting as plain scatter.", file=sys.stderr)

    # Build figure
    print("\nBuilding figure ...")
    fig, legend_handles = make_figure(cells, barren_cells, landcover)

    # Source CSV (saved before figure so the text exporter can read it)
    print("Building source CSV ...")
    source_df = build_source_csv(cells, barren_cells, landcover)
    source_df.to_csv(OUTPUT_SOURCE, index=False)
    print(f"Saved source CSV -> {OUTPUT_SOURCE}")

    # Save outputs
    print(f"Saving PNG -> {OUTPUT_PNG}")
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.05)

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)

    print(f"Saving PDF -> {OUTPUT_PDF}")
    plt.close(fig)

    # Summary
    print("\nOutputs:")
    for path in (OUTPUT_PNG, OUTPUT_SOURCE):
        size = path.stat().st_size
        print(f"  {path} ({size:,} bytes)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
