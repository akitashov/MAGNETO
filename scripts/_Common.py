"""
MAGNETO: Common Configuration and Shared Logic.

This module is the central orchestration hub for the MAGNETO pipeline (Magnetosphere–Atmosphere–Geosphere 
Interaction Network & Ecological Trend Observer). It standardizes data paths, physical constants, 
geographic masking, and statistical parameters to ensure consistency across all processing stages.

CORE COMPONENTS:
----------------
1. Bitwise Geographic Tagging (RegionFlag):
   Implements an IntFlag-based system to categorize grid cells. This allows for overlapping 
   classifications (e.g., a cell tagged as both HIGH_LAI and SAA), enabling complex 
   scenario filtering using bitwise AND/OR/XOR operations.

2. Centralized Pipeline Configuration (Config):
   - Path Management: Defines a strict hierarchy for raw, interim, and processed data.
   - Scenario Definitions: Maps experimental names (e.g., 'Global_High_LAI') to specific 
     required and forbidden bit-flags.
   - Temperature Stratification: Standardizes physiological and quantile-based temperature bins 
     (Cold to Extreme Heat) used to decouple geomagnetic signals from thermal confounders.
   - Statistical Windows: Sets integration ranges for SII (1-90 days) and environmental 
     controls (PAR, VPD).
   - Resource Limits: Configures batch sizes for GPU-accelerated Matrix Search (CuPy/OLS).
   - Visualization Styles: Standardizes colormaps (viridis/plasma/YlOrBr) and p-value 
     significance levels for plotting.

BITWISE REGION DEFINITIONS:
---------------------------
- SAA (1): South Atlantic Anomaly; identifies cells prone to instrumental particle noise.
- CONTROL_NORTH (2): Magnetically stable regions in North America and Eurasia.
- POLAR (4): Auroral zones (>60° N/S); high ionospheric interference.
- SAHARA (8): Bare soil control; used as a negative biophysical baseline.
- LOW_LAI (16): Areas with Leaf Area Index < 0.5 (sparsely vegetated).
- HIGH_LAI (32): Areas with Leaf Area Index >= 0.5 (actively photosynthesizing).

LOGIC HELPERS:
--------------
- scenario_mask(): Vectorized NumPy implementation to filter datasets based on the 
  Config.SCENARIO_MASKS requirements (required vs. forbidden bits).

This module must be imported by all ETL, Analysis, and Visualization scripts to 
maintain the integrity of the MAGNETO analytical framework.
"""

from __future__ import annotations

from pathlib import Path
from enum import IntFlag
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


class RegionFlag(IntFlag):
    """
    Bitwise region classification used across the pipeline.

    Flags combine:
    - Static geographic masks (by lat/lon)
    - Dynamic environmental masks (e.g., LAI-based)

    Values are designed to be OR-combined and stored as int16 (region_flags).
    """
    NONE = 0

    # Geographic Regions (static by coordinates)
    SAA = 1
    CONTROL_NORTH = 2
    POLAR = 4
    SAHARA = 8

    # Dynamic/Environmental (assigned based on data)
    LOW_LAI = 16
    HIGH_LAI = 32


class Config:
    """
    Single source of truth for pipeline configuration.

    Contains:
    - Paths to raw/interim/results artifacts
    - Constant parameters (grid resolution, windows, thresholds)
    - Scenario definitions and masks
    - Shared helpers used by multiple pipeline steps
    """
    # --------------------------------------------------------------------------
    # META (optional; safe to ignore)
    # --------------------------------------------------------------------------
    PIPELINE_TAG: Optional[str] = None

    # --------------------------------------------------------------------------
    # PATHS
    # --------------------------------------------------------------------------
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
    RESULTS_DIR = PROJECT_ROOT / "results"
    REPORTS_ROOT = PROJECT_ROOT / "reports"
    META_ANALYSIS_DIR = REPORTS_ROOT / "meta-statistics"

    # Input Files
    OMNI_RAW_ZIP = DATA_RAW / "omni2_all_years.zip"
    OMNI_RAW_DAT = DATA_RAW / "omni2_all_years.dat"

    # Interim Files
    FILE_OMNI_FEATHER = DATA_INTERIM / "omni_biosphere_features.feather"
    FILE_MODIS_PARQUET = DATA_INTERIM / "modis_extract.parquet"
    FILE_ERA5_PARQUET = DATA_INTERIM / "era5_env_daily.parquet"
    FILE_SIF_FINAL = DATA_INTERIM / "sif_aggregated.feather"

    DIR_SIF_MODEL = DATA_INTERIM / "SIF_model"

    # --------------------------------------------------------------------------
    # CONSTANTS
    # --------------------------------------------------------------------------
    TARGET_RESOLUTION = 0.5

    # LAI masking
    LAI_VEGETATION_THRESHOLD = 0.15
    LAI_ABSOLUTE_MIN = 0.0  # used to remove ocean/no-data after MODIS merge presence checks

    # Cumulative windows (Biological Dose)
    MA_WINDOWS = list(range(1, 29)) + [30, 40, 50, 60, 75, 90]

    # Discrete lags (Information/Pre-sensing)
    # 1-2 days: acute reaction
    # 3-90 days: 3-day grid to match 3-day smoothing window
    DISCRETE_LAGS = [1, 2] + list(range(3, 91, 3))

    # Environmental context window (ERA5 rolling mean)
    CONTEXT_WINDOW_DAYS = 10

    # Canonical column names (so scripts don't hardcode "ma10" everywhere)
    TEMP_CONTEXT_COL = f"temp_c_ma{CONTEXT_WINDOW_DAYS}"
    PAR_CONTEXT_COL = f"par_ma{CONTEXT_WINDOW_DAYS}"
    VPD_CONTEXT_COL = f"vpd_ma{CONTEXT_WINDOW_DAYS}"
    TCC_CONTEXT_COL = f"tcc_ma{CONTEXT_WINDOW_DAYS}"

    # --------------------------------------------------------------------------
    # OMNI2 (Space Weather) ETL CONFIG
    # --------------------------------------------------------------------------
    # Daily aggregations after reading hourly OMNI2 records
    OMNI_AGGREGATIONS = {
        "sii": ["mean", "max", "std"],
        "kp": ["mean", "max"],
        "f10_7": ["mean"],
    }

    # Filter obvious Dst parsing glitches / invalid outliers.
    # Keep NaN Dst (handled downstream), but drop extremely low values.
    DST_MIN_VALID = -2000

    # Discrete-lag smoothing before shifting (anti-aliasing).
    # Set window=1 to disable smoothing.
    OMNI_LAG_SMOOTH_WINDOW = 3
    OMNI_LAG_SMOOTH_CENTER = True

    # Optional: store which daily columns are expected after aggregation
    OMNI_BASE_VARS = ["sii", "kp", "f10_7"]

    # --------------------------------------------------------------------------
    # MODIS ETL CONFIG
    # --------------------------------------------------------------------------
    MODIS_INPUT_DIR = DATA_RAW / 'MODIS'
    MODIS_FILE_PATTERN = "*.nc"

    # Mapping from NetCDF var names -> standardized column names
    MODIS_VAR_MAPPING = {
        'lai': 'lai',
        'primary_qualityflag': 'quality_flag',
        'cloudfraction': 'cloud_fraction',
        'aerosolfraction': 'aerosol_fraction'
    }

    # Snap MODIS timestamps to period start (8-day products)
    MODIS_PERIOD_DAYS = 8
    MODIS_PERIOD_LAST_DOY = 361  # last valid 8-day bucket start day-of-year

    # IO / housekeeping
    MODIS_PARQUET_ENGINE = "fastparquet"  # keep as-is for append mode
    MODIS_GC_EVERY_N_FILES = 10
    MODIS_VERBOSE_ERRORS = False

    # --------------------------------------------------------------------------
    # ERA5 ETL CONFIG
    # --------------------------------------------------------------------------
    ERA5_INPUT_DIR = DATA_RAW / "ERA5"
    ERA5_TEMP_SHARDS_DIR = DATA_INTERIM / "era5_temp_shards"

    # Years to process (inclusive range start, exclusive end)
    ERA5_START_YEAR = 2014
    ERA5_END_YEAR_EXCLUSIVE = 2025

    # File patterns per variable (year, month)
    ERA5_VAR_MAP = {
        "t2m":  {"pattern": "era5_2m_temperature_{}_{}.nc"},
        "d2m":  {"pattern": "era5_2m_dewpoint_temperature_{}_{}.nc"},
        "ssrd": {"pattern": "era5_surface_solar_radiation_downwards_{}_{}.nc"},
        "tcc":  {"pattern": "era5_total_cloud_cover_{}_{}.nc"},
    }

    ERA5_PAR_FRACTION_OF_SSRD = 0.45
    ERA5_SECONDS_PER_HOUR = 3600.0

    ERA5_CONTEXT_EXTRA_BUFFER_DAYS = 5  # keep a few extra days beyond window

    # --------------------------------------------------------------------------
    # OCO-2 SIF ETL CONFIG
    # --------------------------------------------------------------------------
    OCO2_INPUT_DIR = DATA_RAW / "OCO2"
    OCO2_START_YEAR = 2014
    OCO2_END_YEAR_INCLUSIVE = 2025
    OCO2_FILE_GLOB_FMT = "oco2_LtSIF_{yy}*.nc4"  # yy = last two digits of year

    # SIF index safety threshold (avoid division by ~0)
    SIF_MIN_THRESHOLD = 0.001

    # Batch/IO knobs (performance)
    OCO2_FILES_SUBBATCH = 50  # number of daily files to accumulate before MODIS merge+aggregate


    # ==============================================================================
    # SIF ETL (OCO-2) CONFIG
    # ==============================================================================

    # MODIS columns required by SIF ETL
    MODIS_COLS_FOR_SIF = [
        "date", "lat_id", "lon_id",
        "cloud_fraction", "aerosol_fraction",
        "lai"
    ]

    # MODIS spatial dilation (to reduce join loss due to slight grid mismatch)
    # lat_id/lon_id are in "deg*100", so 0.5 degree == 50
    MODIS_DILATION_STEP = 50

    # 4-neighborhood
    MODIS_DILATION_SHIFTS = [
        (0, 0),
        (+MODIS_DILATION_STEP, 0),
        (-MODIS_DILATION_STEP, 0),
        (0, +MODIS_DILATION_STEP),
        (0, -MODIS_DILATION_STEP),
    ]

    # Filtering thresholds used during SIF<->MODIS join
    SIF_FILTERS = {
        "cloud_max": 60.0,
        "aerosol_max": 50.0,
        "lai_min": 0.01,
    }

    # Whether to keep quality_flag column after applying SIF quality mask
    KEEP_SIF_QUALITY_FLAG = False


    # --------------------------------------------------------------------------
    # MODIS FILTERING (Sky/atmosphere + land presence)
    # --------------------------------------------------------------------------
    # These are "analysis-defining" thresholds -> keep centralized
    MODIS_FILTERS = {
        "cloud_max": 60.0,      # percent (0..100), as in your current pipeline
        "aerosol_max": 50.0,    # percent (0..100) or unitless proxy; keep consistent with MODIS source
        "lai_min": LAI_VEGETATION_THRESHOLD,
    }

    # --------------------------------------------------------------------------
    # MODIS SPATIAL DILATION (to reduce lost matches due to grid/key mismatches)
    # --------------------------------------------------------------------------
    # mode: "none" | "cross" | "3x3"
    MODIS_DILATION_MODE = "3x3"
    MODIS_DILATION_RADIUS = 1  # 1 => neighbors at +/- 1 grid step

    # --------------------------------------------------------------------------
    # SIF ANOMALIES / SEASONAL MODEL CONFIG (Step 05)
    # --------------------------------------------------------------------------
    SIF_MODEL_MIN_OBSERVATIONS = 40
    SIF_MODEL_USE_TREND = True          # include alpha_t * t
    SIF_MODEL_PERIOD_DAYS = 365.25      # seasonal period
    SIF_MODEL_USE_LSTSQ = True          # more stable than solve() for near-singular X
    
    SIF_MODEL_N_WORKERS = None

    @staticmethod
    def modis_dilation_step_ids() -> int:
        """
        Grid step in ID units (lat_id/lon_id are lat/lon * 100).
        If TARGET_RESOLUTION = 0.5°, step_ids = 50.
        """
        return int(round(Config.TARGET_RESOLUTION * 100))

    @staticmethod
    def modis_dilation_shifts() -> list[tuple[int, int]]:
        """
        Returns a list of (d_lat_id, d_lon_id) shifts to apply for dilation.
        Always includes (0,0).
        """
        mode = getattr(Config, "MODIS_DILATION_MODE", "none").lower()
        radius = int(getattr(Config, "MODIS_DILATION_RADIUS", 1))
        step = Config.modis_dilation_step_ids()

        shifts = [(0, 0)]
        if mode == "none" or radius <= 0:
            return shifts

        # Build offsets in step_ids
        offs = [k * step for k in range(-radius, radius + 1)]

        if mode == "cross":
            for d in offs:
                shifts.append((d, 0))
                shifts.append((0, d))

        elif mode == "3x3":
            for dlat in offs:
                for dlon in offs:
                    shifts.append((dlat, dlon))

        else:
            raise ValueError(f"Unknown MODIS_DILATION_MODE: {mode}")

        # Deduplicate while preserving order
        out = []
        seen = set()
        for s in shifts:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    # --------------------------------------------------------------------------
    # TEMPERATURE BINNING
    # --------------------------------------------------------------------------
    # Two supported schemes:
    # 1) "physio": fixed physiological regimes (interpretable; recommended)
    # 2) "quantile": equal-frequency bins (robustness / sensitivity analyses)
    TEMP_BIN_SCHEME = "physio"  # "physio" or "quantile"

    # Physio bins (5)
    TEMP_BINS_PHYSIO = [-np.inf, 7, 15, 25, 30, np.inf]
    TEMP_LABELS_PHYSIO = ["Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]

    # Quantile bins (15) - for robustness; labels are integer bin indices by default
    TEMP_NUM_QBINS = 15


    # --------------------------------------------------------------------------
    # STEP 06: SPEARMAN CONFIG
    # --------------------------------------------------------------------------
    SPEARMAN_TARGETS = ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"]
    SPEARMAN_MIN_SAMPLES_PER_BIN = 50
    # Effective sample size correction: max lag for autocorrelation sum
    NEFF_MAX_LAG = 60  # 30..90 разумно; можно начать с 60

    @staticmethod
    def bin_temperature(
            temp_series: pd.Series,
            scheme: Optional[str] = None,
            *,
            physio_bins: Optional[Iterable[float]] = None,
            physio_labels: Optional[Iterable[str]] = None,
            qbins: Optional[int] = None,
        ) -> pd.DataFrame:
        """
        Assign temperature bins to observations.

        Parameters:
            temp_series (pd.Series): Temperature values.
            scheme (str | None): 'physio' or 'quantile'.

        Returns:
            pd.DataFrame:
                Columns:
                - temp_bin_id (float)
                - temp_bin_label (string)

                NaN for missing values.
        """
        scheme = (scheme or Config.TEMP_BIN_SCHEME).lower()

        if scheme == "physio":
            bins = list(physio_bins) if physio_bins is not None else Config.TEMP_BINS_PHYSIO
            labels = list(physio_labels) if physio_labels is not None else Config.TEMP_LABELS_PHYSIO

            cat = pd.cut(temp_series, bins=bins, labels=labels, include_lowest=True)
            # codes: -1 for NaN
            ids = cat.cat.codes.astype("float32").replace(-1, np.nan)
            lab = cat.astype("string")

            return pd.DataFrame({"temp_bin_id": ids, "temp_bin_label": lab}, index=temp_series.index)

        if scheme == "quantile":
            n = int(qbins or Config.TEMP_NUM_QBINS)  
            cat = pd.qcut(temp_series, n, duplicates="drop")  
            ids = cat.cat.codes.astype("float32").replace(-1, np.nan)
            lab = cat.astype("string")  # interval -> "(a, b]"

            return pd.DataFrame({"temp_bin_id": ids, "temp_bin_label": lab}, index=temp_series.index)

        raise ValueError(f"Unknown temperature binning scheme: {scheme}")


    # --------------------------------------------------------------------------
    # GEOGRAPHY: bounding boxes centralized here
    # --------------------------------------------------------------------------
    # Use (min_lat, max_lat, min_lon, max_lon)
    GEOBOX_SAA = (-50.0, 0.0, -90.0, 10.0)
    GEOBOX_SAHARA = (15.0, 30.0, -15.0, 35.0)

    GEOBOX_CONTROL_NA = (25.0, 60.0, -130.0, -60.0)
    GEOBOX_CONTROL_EUASIA = (25.0, 60.0, -10.0, 140.0)

    POLAR_LAT_ABS = 60.0

    @staticmethod
    def _in_box(lat: np.ndarray, lon: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Vectorized bounding-box membership test.

        Parameters
        ----------
        lat, lon : np.ndarray
            Latitude/longitude arrays of the same shape.
        box : tuple(min_lat, max_lat, min_lon, max_lon)

        Returns
        -------
        mask : np.ndarray (bool)
            True for points inside the box (inclusive boundaries).
        """
        min_lat, max_lat, min_lon, max_lon = box
        return (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)

    @staticmethod
    def get_geo_flag(lat: float, lon: float) -> int:
        """
        Scalar flagging: useful for single-point checks.
        Prefer get_geo_flags_vectorized() for arrays.
        """
        flag = RegionFlag.NONE

        if Config._in_box(np.array([lat]), np.array([lon]), Config.GEOBOX_SAA)[0]:
            flag |= RegionFlag.SAA

        is_na = Config._in_box(np.array([lat]), np.array([lon]), Config.GEOBOX_CONTROL_NA)[0]
        is_eu = Config._in_box(np.array([lat]), np.array([lon]), Config.GEOBOX_CONTROL_EUASIA)[0]
        if is_na or is_eu:
            flag |= RegionFlag.CONTROL_NORTH

        if abs(lat) > Config.POLAR_LAT_ABS:
            flag |= RegionFlag.POLAR

        if Config._in_box(np.array([lat]), np.array([lon]), Config.GEOBOX_SAHARA)[0]:
            flag |= RegionFlag.SAHARA

        return int(flag)

    # --------------------------------------------------------------------------
    # STEP 07: Mechanism check (SII vs ENV)
    # --------------------------------------------------------------------------
    MECH_WINDOWS_TO_CHECK = [1, 3, 7, 10, 14, 21, 28]            # SII MA windows to test
    MECH_ENV_VARS_BASE = ["par", "vpd", "tcc"]  # base env vars; targets will be f"{var}_ma{w}"

    MECH_MIN_SAMPLES_PER_BIN = 100

    # Scenarios to use specifically in Step 07
    MECH_SCENARIOS = ["Global_High_LAI", "Control_North", "Sahara_Barren"]

    @staticmethod
    def get_geo_flags_vectorized(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Vectorized geographic flag computation for arrays.

        Parameters
        ----------
        lats, lons : np.ndarray
            Arrays of latitude/longitude (same length/shape).

        Returns
        -------
        flags : np.ndarray (int16)
        """
        lats = np.asarray(lats, dtype=np.float32)
        lons = np.asarray(lons, dtype=np.float32)

        flags = np.full(lats.shape, RegionFlag.NONE, dtype=np.int16)

        mask_saa = Config._in_box(lats, lons, Config.GEOBOX_SAA)
        flags[mask_saa] |= RegionFlag.SAA

        mask_na = Config._in_box(lats, lons, Config.GEOBOX_CONTROL_NA)
        mask_eu = Config._in_box(lats, lons, Config.GEOBOX_CONTROL_EUASIA)
        flags[mask_na | mask_eu] |= RegionFlag.CONTROL_NORTH

        flags[np.abs(lats) > Config.POLAR_LAT_ABS] |= RegionFlag.POLAR

        mask_sahara = Config._in_box(lats, lons, Config.GEOBOX_SAHARA)
        flags[mask_sahara] |= RegionFlag.SAHARA

        return flags.astype(np.int16)

    @staticmethod
    def apply_lai_flags(flags: np.ndarray, lai: np.ndarray) -> np.ndarray:
        """
        Adds LOW_LAI / HIGH_LAI flags based on LAI threshold.
        Assumes lai is aligned with flags.
        """
        flags = np.asarray(flags, dtype=np.int16).copy()
        lai = np.asarray(lai, dtype=np.float32)

        # Treat NaN as "unknown" - caller should decide filtering.
        mask_high = np.isfinite(lai) & (lai >= Config.LAI_VEGETATION_THRESHOLD)
        mask_low = np.isfinite(lai) & (lai < Config.LAI_VEGETATION_THRESHOLD)

        flags[mask_high] |= RegionFlag.HIGH_LAI
        flags[mask_low] |= RegionFlag.LOW_LAI

        return flags.astype(np.int16)

    # --------------------------------------------------------------------------
    # SCENARIOS: centralized definitions
    # --------------------------------------------------------------------------
    # Bitmask-based scenarios (cleaner than lambdas scattered across scripts)
    # Each scenario is a tuple: (required_bits, forbidden_bits)
    SCENARIO_MASKS: Dict[str, Tuple[int, int]] = {
        "Global_High_LAI": (int(RegionFlag.HIGH_LAI), 0),
        "SAA_High_LAI": (int(RegionFlag.SAA | RegionFlag.HIGH_LAI), 0),
        "Control_North": (int(RegionFlag.CONTROL_NORTH | RegionFlag.HIGH_LAI), 0),
        "Sahara_Barren": (int(RegionFlag.SAHARA | RegionFlag.LOW_LAI), 0),
    }
    
    # --------------------------------------------------------------------------
    # STEP 10: GPU Matrix Search (Exhaustive)
    # --------------------------------------------------------------------------
    MATRIX_SII_RANGE = list(range(1, 29))   # 1..28
    # MATRIX_ENV_RANGE = list(range(1, 8))    # 1..7?
    MATRIX_ENV_RANGE = MATRIX_SII_RANGE    # to avoid alleged asymmetry

    MATRIX_TARGETS = ["sif_740nm", 
        "sif_stress_index", 
        "sif_757nm", 
        "sif_771nm"]
    MATRIX_SMOOTHING_WINDOW = 3

    # Scenarios (names must exist in Config.SCENARIO_MASKS / Config.scenario_mask)
    MATRIX_SCENARIOS = ["Control_North", 
        "Global_High_LAI", 
        "SAA_High_LAI", 
        "Sahara_Barren"]

    # Temperature bins to process (labels from Config.TEMP_LABELS) or None for all
    MATRIX_TARGET_TEMP_BINS = ["Cold", 
        "Cool", 
        "Optimum", 
        "Warm_Stress", 
        "Extreme_Heat"]

    # Streaming / memory
    MATRIX_LOCATIONS_PER_CHUNK = 100
    MATRIX_MAX_SAMPLES_PER_GPU_BATCH = 20000

    # GPU device index
    MATRIX_GPU_DEVICE = 0

    @staticmethod
    def scenario_mask(flags: np.ndarray, scenario_name: str) -> np.ndarray:
        """
        Returns boolean mask for a scenario based on bit requirements.
        """
        if scenario_name not in Config.SCENARIO_MASKS:
            raise KeyError(f"Unknown scenario: {scenario_name}")

        required, forbidden = Config.SCENARIO_MASKS[scenario_name]
        f = flags.astype(np.int64)

        ok_required = (f & required) == required
        ok_forbidden = ((f & forbidden) == 0) if forbidden else np.ones_like(ok_required, dtype=bool)

        return ok_required & ok_forbidden


    # --------------------------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------------------------

    TEMP_RANGES = {
        0: "≤ 10",
        1: "10–19",
        2: "19–26",
        3: "26–31",
        4: "> 31"
    }

    # --- P-VALUE CONFIGURATION ---
    P_VALUE_LEVELS = [
        (1e-16, 1.0),
        (1e-8, 0.8),
        (1e-4,  0.6),
        (1e-2,  0.4),
        (0.05,  0.15),
        ]

    P_FLOOR = 1e-16  # or 1e-300 if you prefer purely numerical safety

    @staticmethod
    def sanitize_p(p: np.ndarray, floor: float = 1e-16) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.where(np.isfinite(p) & (p > 0), p, floor)
    
