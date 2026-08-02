#!/usr/bin/env python3
"""
Convert MCD12C1 HDF to NetCDF with proper integer types and comprehensive validation

-------------------------------------------------------------------------------
DATA DESCRIPTION FOR SCIENTIFIC PUBLICATIONS
-------------------------------------------------------------------------------

Data Source
-----------
We used the MCD12C1 Version 6.1 land cover product provided by the NASA Land
Processes Distributed Active Archive Center (LP DAAC). This product contains
global land cover classifications derived from Moderate Resolution Imaging
Spectroradiometer (MODIS) data from both the Terra and Aqua satellites.

Product:        MCD12C1 Version 6.1
Data Set ID:    MCD12C1.A2022001.061.2023244164746.hdf
Year:           2022
Spatial Res.:   0.05° (~5.6 km at equator)
Grid:           3600 rows × 7200 columns (global)
Projection:     Geographic (Latitude/Longitude)
Classification: IGBP (International Geosphere-Biosphere Programme) – 17 classes
Data Type:      Land Cover Type 1 (Majority Class)
Source URL:     https://ladsweb.modaps.eosdis.nasa.gov/

Processing Workflow
-------------------
The original HDF-EOS file was processed using scripts/etl/E04_mcd12c1_hdf2netcdf.py to create a
compressed NetCDF file suitable for the MAGNETO project analysis grid.

Step 1: Data Extraction
    The variable 'Majority_Land_Cover_Type_1' was extracted from the HDF file.
    This variable contains the dominant IGBP class (1-17) for each 0.05° pixel.
    The original data type is 32-bit floating point with NaN values representing
    ocean/NoData areas.

Step 2: Data Cleaning
    All NaN (Not a Number) values were replaced with 0 (NoData/fill value).
    Values outside the valid range (1-17) were set to 0.
    The data were rounded to nearest integer and converted to 8-bit unsigned
    integer format (uint8).

Step 3: Coordinate System
    Latitude coordinates:  -89.975° to 89.975° with 0.05° step (3600 points)
    Longitude coordinates: -179.975° to 179.975° with 0.05° step (7200 points)
    This matches the standard MCD12C1 global grid.

Step 4: Compression
    The output file uses NetCDF4 format with zlib compression (level 5).
    Compression reduced file size from 1,186 MB to 1.08 MB (99.9% reduction).

Step 5: Validation
    Each cell contains a single integer value (0-16) representing the land
    cover class. Class 0 = NoData (ocean or missing data). Classes 1-16
    correspond to IGBP land cover types (see IGBP_CLASSES below).

Output File Format
------------------
Filename:   MCD12C1.A2022001.061.2023244164746.nc
Format:     NetCDF4 with zlib compression

NetCDF Structure:
    Dimensions:
        lat = 3600
        lon = 7200

    Variables:
        land_cover (lat, lon)
            dtype: uint8
            _FillValue: 0
            units: class
            valid_range: [1, 17]
            long_name: IGBP Land Cover Classification

        lat (lat)
            units: degrees_north
            standard_name: latitude

        lon (lon)
            units: degrees_east
            standard_name: longitude

    Global Attributes:
        title: MCD12C1 Land Cover Type 1 (IGBP)
        product: MCD12C1
        version: 061
        source: MCD12C1.A2022001.061.2023244164746.hdf
        references: Friedl, M., Sulla-Menashe, D. (2022)

IGBP Land Cover Classification
-------------------------------
Class  IGBP Type                           Description
-----  ----------------------------------  ------------------------------------
1      Evergreen Needleleaf Forest         Forests with >60% evergreen needleleaf
2      Evergreen Broadleaf Forest          Forests with >60% evergreen broadleaf
3      Deciduous Needleleaf Forest         Forests with >60% deciduous needleleaf
4      Deciduous Broadleaf Forest          Forests with >60% deciduous broadleaf
5      Mixed Forests                       Neither deciduous nor evergreen >60%
6      Closed Shrublands                   Shrubs >60% cover, >2m height
7      Open Shrublands                     Shrubs 10-60% cover, >2m height
8      Woody Savannas                      Trees 30-60%, herbaceous understory
9      Savannas                            Trees 10-30%, herbaceous understory
10     Grasslands                          Herbaceous >10% cover
11     Permanent Wetlands                  Inundated >9 months/year
12     Croplands                           >60% cropland
13     Urban and Built-up                  >30% impervious surface
14     Cropland/Natural Vegetation Mosaic  Croplands 40-60%
15     Snow and Ice                        Permanent snow/ice
16     Barren or Sparsely Vegetated        <10% vegetation
17     Water                               Water bodies >60%

Data Statistics (after processing)
----------------------------------
Total pixels:           25,920,000
Valid pixels (data):    8,372,565 (32.3%)
NoData (ocean/missing): 17,547,435 (67.7%)
Number of classes:      16 (all except 17)

Top 5 classes by area:
    15: Snow and Ice                   2,607,832 (31.15%)
    10: Grasslands                     1,379,067 (16.47%)
    16: Barren or Sparsely Vegetated     813,661 ( 9.72%)
    7:  Open Shrublands                  718,386 ( 8.58%)
    9:  Savannas                         709,865 ( 8.48%)

Citation
--------
When using these data in publications, please cite:

Data Product:
    Friedl, M., Sulla-Menashe, D. (2022). MCD12C1 MODIS/Terra+Aqua Land Cover
    Type Yearly L3 Global 0.05Deg CMG V061. NASA EOSDIS Land Processes DAAC.
    https://doi.org/10.5067/MODIS/MCD12C1.061

Processing Script:
    The conversion script scripts/etl/E04_mcd12c1_hdf2netcdf.py is available in the project repository
    at data/raw/MCD12C1/2022/001/.

-------------------------------------------------------------------------------
"""

import xarray as xr
from pathlib import Path
import numpy as np
import time
import os

# IGBP class names dictionary for reference
IGBP_CLASSES = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Water"
}


def validate_output_file(nc_file):
    """
    Validate the created NetCDF file and print detailed statistics.

    Parameters:
    -----------
    nc_file : Path
        Path to the NetCDF file to validate

    Returns:
    --------
    ds : xarray.Dataset
        The opened dataset
    """
    print("\n" + "="*60)
    print("OUTPUT FILE VALIDATION")
    print("="*60)

    # Open the file
    ds = xr.open_dataset(nc_file)

    # Basic information
    print(f"\n📁 File: {nc_file.name}")
    print(f"   Size: {nc_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Data structure
    print(f"\n📊 Data structure:")
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Coordinates: {list(ds.coords)}")
    print(f"   Attributes: {list(ds.attrs.keys())}")

    # Information about the land_cover variable
    lc = ds['land_cover']
    print(f"\n🌍 Variable 'land_cover':")
    print(f"   Shape: {lc.shape}")
    print(f"   Data type: {lc.dtype}")
    print(f"   Memory size: {lc.nbytes / (1024 * 1024):.2f} MB")

    # Value statistics - handle NaN properly
    values = lc.values

    # Check for NaN values
    nan_mask = np.isnan(values)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"\n⚠️ Found {nan_count:,} NaN values")
        values_clean = values.copy()
        values_clean[nan_mask] = 0
    else:
        values_clean = values

    unique_vals = np.unique(values_clean)
    print(f"\n📈 Value statistics:")
    print(f"   Unique values: {len(unique_vals)}")
    print(f"   Range: {values_clean.min()} - {values_clean.max()}")

    # Count NoData
    fill_value = lc.attrs.get('_FillValue', 0)
    nodata_count = np.sum(values_clean == fill_value)
    total_pixels = values.size
    print(f"   NoData (value {fill_value}): {nodata_count:,} pixels ({nodata_count/total_pixels*100:.2f}%)")

    # Class distribution (only valid data)
    valid_mask = values_clean != fill_value
    valid_values = values_clean[valid_mask]
    if len(valid_values) > 0:
        # Convert to integers for statistics
        valid_values_int = valid_values.astype(int)
        valid_classes, counts = np.unique(valid_values_int, return_counts=True)
        print(f"\n📊 Land cover class distribution (IGBP):")
        print(f"   Total valid pixels: {len(valid_values):,}")

        # Show top 5 classes
        print(f"\n   Top 5 classes by area:")
        sorted_indices = np.argsort(counts)[::-1]
        for i in range(min(5, len(valid_classes))):
            idx = sorted_indices[i]
            cls = valid_classes[idx]
            count = counts[idx]
            percentage = count / len(valid_values) * 100
            name = IGBP_CLASSES.get(int(cls), f"Class {int(cls)}")
            print(f"     {int(cls)}: {name:<30} {count:>10,} ({percentage:>5.2f}%)")

    # Metadata check
    print(f"\n📝 Metadata:")
    print(f"   _FillValue: {lc.attrs.get('_FillValue', 'not specified')}")
    print(f"   valid_range: {lc.attrs.get('valid_range', 'not specified')}")
    print(f"   units: {lc.attrs.get('units', 'not specified')}")

    # Coordinate check
    print(f"\n🗺️ Coordinates:")
    print(f"   Latitude: {ds.lat.values.min():.2f}° - {ds.lat.values.max():.2f}° (step: {ds.lat.values[1] - ds.lat.values[0]:.3f}°)")
    print(f"   Longitude: {ds.lon.values.min():.2f}° - {ds.lon.values.max():.2f}° (step: {ds.lon.values[1] - ds.lon.values[0]:.3f}°)")

    # Check for integer type
    if np.issubdtype(lc.dtype, np.integer):
        print(f"\n✅ Data stored in integer format")
    else:
        print(f"\n⚠️ WARNING: Data type is {lc.dtype}, expected integer!")

    # Check for NaN
    if np.any(np.isnan(values)):
        print(f"\n⚠️ WARNING: NaN values found in original data!")
    else:
        print(f"\n✅ No NaN values found")

    print("\n" + "="*60)

    return ds


def convert_hdf_to_nc(hdf_file=None, output_file=None):
    """
    Convert MCD12C1 HDF file to NetCDF format with integer types.

    If hdf_file is None, uses the first .hdf file in current directory.

    Parameters:
    -----------
    hdf_file : str or Path, optional
        Path to input HDF file
    output_file : str or Path, optional
        Path to output NetCDF file

    Returns:
    --------
    output_file : Path
        Path to the created NetCDF file
    """

    # Find HDF file.  Default to the MAGNETO raw-data location; fall back to the
    # current directory only when no project path is available.
    if hdf_file is None:
        project_raw = Path(__file__).resolve().parents[2] / "data" / "raw" / "MCD12C1" / "2022" / "001"
        hdf_files = list(project_raw.glob('*.hdf'))
        if not hdf_files:
            hdf_files = list(Path('.').glob('*.hdf'))
        if not hdf_files:
            print("Error: No HDF files found in project raw directory or current directory!")
            return
        hdf_path = hdf_files[0]
        print(f"Found file: {hdf_path}")
    else:
        hdf_path = Path(hdf_file)

    if not hdf_path.exists():
        print(f"File not found: {hdf_path}")
        return

    if output_file is None:
        project_interim = Path(__file__).resolve().parents[2] / "data" / "interim"
        output_file = project_interim / hdf_path.with_suffix('.nc').name

    print(f"Reading: {hdf_path}")
    start_time = time.time()

    # Open HDF
    ds = xr.open_dataset(hdf_path, engine='netcdf4')

    # Use Majority_Land_Cover_Type_1
    lc_data = ds['Majority_Land_Cover_Type_1']
    print(f"Original shape: {lc_data.shape}, dtype: {lc_data.dtype}")

    # Create coordinates (0.05° grid).  MCD12C1 CMG stores the first row at the
    # northernmost latitude, so we reverse axis 0 to obtain ascending latitudes
    # (-89.975° at row 0 to +89.975° at row -1).
    lat = np.linspace(-89.975, 89.975, lc_data.shape[0])
    lon = np.linspace(-179.975, 179.975, lc_data.shape[1])

    # Convert data to integers
    print("Converting data...")
    lc_values = lc_data.values.copy()[::-1, :]

    # Check and replace NaN values
    nan_mask = np.isnan(lc_values)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        print(f"  Found {nan_count:,} NaN values")
        print(f"  Replacing with {0}")
        lc_values[nan_mask] = 0

    # Also check for inf/-inf
    inf_mask = np.isinf(lc_values)
    if inf_mask.any():
        inf_count = inf_mask.sum()
        print(f"  Found {inf_count:,} Inf values")
        lc_values[inf_mask] = 0

    # Check for values outside valid range (0-17)
    out_of_range = (lc_values < 0) | (lc_values > 17)
    if out_of_range.any():
        out_count = out_of_range.sum()
        print(f"  Found {out_count:,} values outside 0-17 range")
        lc_values[out_of_range] = 0

    # Round to integers and convert type
    lc_values = np.round(lc_values).astype(np.uint8)

    unique_vals = np.unique(lc_values)
    print(f"  Unique values after conversion: {unique_vals}")
    print(f"  NoData count (0): {np.sum(lc_values == 0):,}")

    # Create dataset with explicit integer type
    output_ds = xr.Dataset()

    # Add the land_cover variable with explicit uint8 type
    output_ds['land_cover'] = xr.DataArray(
        lc_values,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        attrs={
            "_FillValue": 0,
            "units": "class",
            "long_name": "IGBP Land Cover Classification",
            "valid_range": (1, 17),
            "description": "17-class IGBP classification scheme (0 = NoData/FillValue)",
            "source_variable": "Majority_Land_Cover_Type_1",
        }
    )

    # Add global attributes with full metadata
    output_ds.attrs = {
        "title": "MCD12C1 Land Cover Type 1 (IGBP)",
        "product": "MCD12C1",
        "version": "061",
        "source": str(hdf_path.name),
        "description": "Majority Land Cover Type 1 from MCD12C1 (integer)",
        "references": "Friedl, M., Sulla-Menashe, D. (2022). MCD12C1 MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 0.05Deg CMG V061. NASA EOSDIS Land Processes DAAC. https://doi.org/10.5067/MODIS/MCD12C1.061",
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conversion_script": "scripts/etl/E04_mcd12c1_hdf2netcdf.py",
        "spatial_resolution": "0.05_degree",
        "grid_description": "3600 x 7200 global grid",
        "latitude_range": "-89.975 to 89.975",
        "longitude_range": "-179.975 to 179.975",
    }

    # Save with compression
    print(f"\nSaving: {output_file}")
    output_ds.to_netcdf(
        output_file,
        engine='netcdf4',
        encoding={
            'land_cover': {
                'zlib': True,
                'complevel': 5,
                'shuffle': True,
                'dtype': 'uint8',
            }
        }
    )

    elapsed_time = time.time() - start_time

    # Show file sizes
    original_size = hdf_path.stat().st_size / (1024 * 1024)
    new_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\n📊 Conversion results:")
    print(f"  Original HDF:   {original_size:.1f} MB")
    print(f"  Compressed NetCDF:  {new_size:.1f} MB")
    print(f"  Space saving:   {(1 - new_size/original_size) * 100:.1f}%")
    print(f"  Time:           {elapsed_time:.1f} seconds")

    # Validate the created file
    validate_output_file(output_file)

    print("\n" + "="*60)
    print("CITATION INFORMATION")
    print("="*60)
    print("When using these data in publications, please cite:")
    print()
    print("  Friedl, M., Sulla-Menashe, D. (2022). MCD12C1 MODIS/Terra+Aqua")
    print("  Land Cover Type Yearly L3 Global 0.05Deg CMG V061. NASA EOSDIS")
    print("  Land Processes DAAC. https://doi.org/10.5067/MODIS/MCD12C1.061")
    print()
    print("  Processing script: scripts/etl/E04_mcd12c1_hdf2netcdf.py")
    print("="*60)

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MCD12C1 HDF land-cover file to compressed NetCDF"
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Path to input HDF file (default: first .hdf in data/raw/MCD12C1/2022/001)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path to output NetCDF file (default: data/interim/<input>.nc)",
    )
    args = parser.parse_args()
    convert_hdf_to_nc(hdf_file=args.input, output_file=args.output)
