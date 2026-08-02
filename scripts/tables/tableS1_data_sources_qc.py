#!/usr/bin/env python3
"""Table S1. Data sources, spatial and temporal resolution, preprocessing and QC.

This table is assembled from project documentation and QC manifests; no raw-data
ETL is rerun.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tables._table_utils import load_csv, project_root, write_table


def build_table_s1() -> tuple[pd.DataFrame, list[str]]:
    qc = load_csv(project_root() / "results" / "qc_flow.csv")

    # Pull the current pooled-full harmonic 28-day cell count from fixed-window results
    # so the table stays synchronized with the analytical sample.
    fw = load_csv(project_root() / "results" / "fixed_window_results.csv")
    fw_pooled = fw[
        (fw["sample_type"] == "pooled_full")
        & (fw["method"] == "harmonic")
        & (fw["sii_window_days"] == 28)
        & (fw["temperature_class"] == "Pooled")
    ]
    n_cells = int(fw_pooled.iloc[0]["n_cells"]) if not fw_pooled.empty else None
    n_obs = int(fw_pooled.iloc[0]["n_obs"]) if not fw_pooled.empty else None
    cells_str = f"{n_cells:,}" if n_cells is not None else "current analytical sample"
    obs_str = f"{n_obs:,}" if n_obs is not None else "current analytical sample"

    data = [
        {
            "Data source": "OCO-2 SIF Lite",
            "Variables": "SIF 757 nm, SIF 771 nm, SIF 740 nm (provider-derived)",
            "Spatial resolution": "1.3 km × 2.25 km soundings",
            "Temporal resolution": "16-day revisit",
            "Coverage": f"{cells_str} 0.5° land-grid cells in the final analytical sample, Sep 2014–Dec 2024",
            "Preprocessing / QC": "Land-only retrievals; cloud fraction ≤ 60 %; aerosol fraction ≤ 50 %; latitudes [−60°, 75°]; aggregated to 0.5° × 0.5° daily means.",
        },
        {
            "Data source": "OMNI2 (NASA SPDF)",
            "Variables": "Dst, Kp (0.1-Kp units), F10.7 cm flux",
            "Spatial resolution": "Global indices",
            "Temporal resolution": "Dst hourly, Kp 3-hourly, F10.7 daily; aligned to daily means for analysis",
            "Coverage": "Sep 2014–Dec 2024 (daily means)",
            "Preprocessing / QC": "Dst converted to the Storm Intensity Index (SII = −Dst); SII exposure is a 28-day trailing mean (shifted 1 day to avoid leakage). Kp and F10.7 are comparator indices.",
        },
        {
            "Data source": "ERA5 (Copernicus CDS)",
            "Variables": "PAR, VPD, 2-m temperature",
            "Spatial resolution": "0.25° reanalysis",
            "Temporal resolution": "Hourly; daily means",
            "Coverage": "Sep 2014–Dec 2024",
            "Preprocessing / QC": "Regridded to 0.5° × 0.5° daily means; PAR estimated as 0.45 × SSRD / 3600 s (W m⁻²); VPD from 2-m dewpoint and temperature.",
        },
        {
            "Data source": "MODIS/Terra MOD15A2H Collection 6.1",
            "Variables": "Leaf area index",
            "Spatial resolution": "500 m",
            "Temporal resolution": "8-day",
            "Coverage": "Sep 2014–Dec 2024",
            "Preprocessing / QC": "Regridded to 0.5° × 0.5°; cloud/aerosol diagnostics retained; aggregated to cell medians/quartiles for persistent-vegetation thresholds and LAI-strata diagnostics.",
        },
        {
            "Data source": "MCD12C1 2022",
            "Variables": "Dominant land-cover class (IGBP)",
            "Spatial resolution": "0.05°",
            "Temporal resolution": "Annual, 2022 classification",
            "Coverage": "Global land",
            "Preprocessing / QC": "Aggregated to six classes (Barren, Shrubland/Savanna, Savanna, Grassland, Cropland, Forest) on the 0.5° analysis grid.",
        },
    ]

    df = pd.DataFrame(data)

    final_rows = len(qc)
    retained = qc.iloc[-1]["pct_remaining"]
    notes = [
        f"Final QC flow retains {retained:.2f} % of merged SIF×MODIS observations ({final_rows} filtering steps).",
        f"The final analytical sample contains {cells_str} 0.5° land-grid cells and {obs_str} observations (see results/fixed_window_results.csv, pooled_full, harmonic, 28-day window).",
        "SIF 740 nm is provider-derived as 0.75 × (SIF 757 nm + 1.5 × SIF 771 nm), consistent with the OCO-2 SIF Lite product documentation.",
        "SII exposure is a trailing mean, not a cumulative total or integral.",
        "See reports/SIF_QC_AUDIT.md and results/qc_flow.csv for full QC details.",
    ]
    return df, notes


def main() -> int:
    df, notes = build_table_s1()
    write_table(df, "tableS1_data_sources_qc", "Table S1. Data sources, spatial and temporal resolution, preprocessing and QC", notes=notes, landscape=True)
    print(f"[OK] Table S1: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
