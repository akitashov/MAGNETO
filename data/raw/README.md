# Raw Data Sources

This directory is intended to store the raw input datasets required for the MAGNETO pipeline.
Due to size constraints, the actual data files are not included in the repository.

Please download the datasets from the official sources listed below and place them in the corresponding subdirectories.

## Directory Structure
Ensure the data is organized as follows:

```text
data/raw/
├── omni2_all_years.zip       # OMNI2 Data (NASA SPDF)
├── MODIS/                    # MODIS LAI/FPAR (MCD15A2H)
│   └── *.nc
├── ERA5/                     # ERA5 Reanalysis (ECMWF/Copernicus)
│   └── *.nc
└── OCO2/                     # OCO-2 SIF Lite Files (NASA GES DISC)
    └── *.nc4
```

## 1. OMNI2 (Space Weather)
* **Source:** [NASA SPDF OMNIWeb](https://omniweb.gsfc.nasa.gov/)
* **File:** `omni2_all_years.zip` (ASCII format)
* **Variables:** Dst, Kp, F10.7, Solar Wind parameters.

## 2. MODIS (Vegetation)
* **Source:** [LP DAAC (NASA Earthdata)](https://lpdaac.usgs.gov/products/mcd15a2h006/)
* **Product:** MCD15A2H (Leaf Area Index / FPAR) 8-Day L4 Global 500m.
* **Format:** NetCDF (converted from HDF) or HDF-EOS.
* **Variables:** `Lai_500m`, `Fpar_500m`, `FparLai_QC`.

## 3. ERA5 (Climate Context)
* **Source:** [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
* **Product:** ERA5 hourly data on single levels.
* **Variables:**
    * 2m temperature (`t2m`)
    * 2m dewpoint temperature (`d2m`) -> used for VPD
    * Surface solar radiation downwards (`ssrd`) -> used for PAR
    * Total cloud cover (`tcc`)
* **Resolution:** 0.25° x 0.25°, aggregated to daily.

## 4. OCO-2 (Fluorescence)
* **Source:** [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_SIF_10r/summary)
* **Product:** OCO-2 Level 2 Bias-Corrected SIF Lite (v10 or v11).
* **Format:** NetCDF4 (`.nc4`).
* **Variables:** `SIF_740`, `SIF_757`, `SIF_771`, `SIF_Relative_Azimuth`, etc.
