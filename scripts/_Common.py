"""
MAGNETO canonical configuration loader.

All analysis parameters come from config/pipeline.yaml (the single active
configuration). This module exposes them as flat, uppercase module constants so
that existing stage scripts can continue to import with `from _Common import *`.
"""
from __future__ import annotations
import os, json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import yaml

# ── Project root ─────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "scripts":
    PROJECT_ROOT = Path(os.environ.get("MAGNETO_ROOT", _THIS_FILE.parent.parent))
else:
    PROJECT_ROOT = Path(os.environ.get("MAGNETO_ROOT", _THIS_FILE.parent.parent.parent))

# ── Load canonical YAML config ───────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG: dict[str, Any] = yaml.safe_load(_f)


def _load_scenario_registry() -> dict[str, dict]:
    """Load the canonical scenario registry from supplementary_checks.yaml."""
    reg_path = PROJECT_ROOT / "config" / "supplementary_checks.yaml"
    with open(reg_path, "r", encoding="utf-8") as _f:
        reg_cfg: dict[str, Any] = yaml.safe_load(_f)
    return {
        entry["scenario_id"]: entry
        for entry in reg_cfg.get("scenario_registry", [])
    }


SCENARIO_REGISTRY: dict[str, dict] = _load_scenario_registry()


def _as_path(value: Any) -> Any:
    if isinstance(value, str):
        return PROJECT_ROOT / value
    return value


def _resolve_scalar(value: Any) -> Any:
    if value == "-Infinity":
        return -np.inf
    if value == "Infinity":
        return np.inf
    return value


# ── Directory layout ─────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / CONFIG["paths"]["data_raw"]
DATA_INTERIM = PROJECT_ROOT / CONFIG["paths"]["data_interim"]
DATA_PROCESSED = PROJECT_ROOT / CONFIG["paths"]["data_processed"]
RESULTS = PROJECT_ROOT / CONFIG["paths"]["results"]
REPORTS = PROJECT_ROOT / CONFIG["paths"]["reports"]
LOGS = PROJECT_ROOT / CONFIG["paths"]["logs"]

# ── Canonical ETL inputs (read-only) ─────────────────────────────────
FILE_SIF = PROJECT_ROOT / CONFIG["inputs"]["file_sif"]
FILE_OMNI = PROJECT_ROOT / CONFIG["inputs"]["file_omni"]
FILE_ERA5 = PROJECT_ROOT / CONFIG["inputs"]["file_era5"]
FILE_MODIS = PROJECT_ROOT / CONFIG["inputs"]["file_modis"]

# ── Intermediate outputs ─────────────────────────────────────────────
_io = CONFIG["interim_outputs"]
FILE_QC = _as_path(_io["file_qc"])
FILE_LAI_CELLS = _as_path(_io["file_lai_cells"])
FILE_LAI_BOUNDS = _as_path(_io["file_lai_bounds"])
FILE_LAI_MEMBERS = _as_path(_io["file_lai_members"])
FILE_CONTROL_MEMBERS = _as_path(_io["file_control_members"])

# ── Processed analytical datasets ────────────────────────────────────
_po = CONFIG["processed_outputs"]
FILE_RES_HARMONIC = _as_path(_po["file_harmonic_analysis"])
FILE_RES_CYCLIC = _as_path(_po["file_cyclic_spline_analysis"])
FILE_MATCHED_HC = _as_path(_po["file_matched_harmonic_spline"])
# Legacy aliases used by some older stages/tests
FILE_RES_DOY = DATA_INTERIM / "residuals_doy_mean_pm1.parquet"
FILE_MATCHED_3WAY = DATA_INTERIM / "matched_three_methods.parquet"

# ── Result files ─────────────────────────────────────────────────────
_ro = CONFIG["results_outputs"]
FILE_FIXED = _as_path(_ro["file_fixed"])
FILE_SURROGATES = _as_path(_ro["file_surrogates"])
FILE_SURROGATE_NULL = _as_path(_ro["file_surrogate_null"])
FILE_EFFECTS = _as_path(_ro["file_effects"])
FILE_QC_FLOW = _as_path(_ro["file_qc_flow"])
FILE_CONTROL_DEFINITIONS = _as_path(_ro["file_control_definitions"])
FILE_CONTROL_DIAGNOSTICS = _as_path(_ro["file_control_diagnostics"])
FILE_GEOGRAPHIC_CONTROL_DIAGNOSTICS = _as_path(_ro["file_geographic_control_diagnostics"])
FILE_LAI_QUARTILE_DIAGNOSTICS = _as_path(_ro["file_lai_quartile_diagnostics"])
FILE_DOY_COVERAGE = _as_path(_ro["file_doy_coverage"])
FILE_SIF_VALUE_DIAGNOSTICS = _as_path(_ro["file_sif_value_diagnostics"])
FILE_SMOKE_SUBSET_DIAGNOSTICS = _as_path(_ro["file_smoke_subset_diagnostics"])
FILE_OUTPUT_MANIFEST = _as_path(_ro["file_output_manifest"])
FILE_OUTPUT_MANIFEST_CSV = _as_path(_ro["file_output_manifest_csv"])

# ── Targets and windows ──────────────────────────────────────────────
TARGETS = list(CONFIG["analysis"]["targets"])
SII_WINDOWS = [int(w) for w in CONFIG["analysis"]["sii_windows"]]
PRIMARY_SII_WINDOW = int(CONFIG["analysis"]["primary_sii_window"])
SECONDARY_SII_WINDOW = int(CONFIG["analysis"]["secondary_sii_window"])
RESIDUAL_METHODS = list(CONFIG["analysis"]["residual_methods"])
LAI_N_QUARTILES = int(CONFIG["analysis"]["lai_n_quartiles"])

# ── QC thresholds ────────────────────────────────────────────────────
_qc = CONFIG["qc"]
QC_CLOUD_MAX = float(_qc["cloud_max"])
QC_AEROSOL_MAX = float(_qc["aerosol_max"])
QC_SIF_MIN = _resolve_scalar(_qc["sif_min"])
QC_SIF_MIN_SENSITIVITY = _resolve_scalar(_qc["sensitivity_sif_min"])
QC_LAT_MIN = float(_qc["lat_min"])
QC_LAT_MAX = float(_qc["lat_max"])

# ── Analysis-area filter ─────────────────────────────────────────────
# The upstream SIF ETL already supplies a land-only dataset; here we only
# exclude POLAR retrievals from the analysis area.
_lm = CONFIG["analysis_area_filter"]
LAND_KEEP_FLAGS = tuple(int(f) for f in _lm["keep_flags"])
LAND_EXCLUDE_FLAGS = tuple(int(f) for f in _lm["exclude_flags"])

# ── LAI quartiles ────────────────────────────────────────────────────
_lai = CONFIG["lai"]
LAI_USE_FULL_MODIS_SERIES = bool(_lai["use_full_modis_series"])
LAI_QUARTILE_POPULATION_FLAGS = tuple(int(f) for f in _lai["quartile_population_flags"])

# ── Functional control thresholds ────────────────────────────────────
# NOTE: historical files may use the column name ``is_barren``; that flag
# should be interpreted as the strict low-LAI control (``is_strict_low_lai``)
# defined by the thresholds below.
_fc = CONFIG["controls"]["functional"]
CONTROL_STRICT_LOW_LAI = _fc["strict_low_lai"]["name"]
CONTROL_STRICT_LOW_LAI_MEDIAN_LAI_MAX = float(_fc["strict_low_lai"]["median_lai_max"])
CONTROL_STRICT_LOW_LAI_Q90_LAI_MAX = float(_fc["strict_low_lai"]["q90_lai_max"])
CONTROL_STRICT_LOW_LAI_MIN_LAI_OBS = int(_fc["strict_low_lai"]["min_lai_obs"])
CONTROL_Vegetated = _fc["vegetated"]["name"]
CONTROL_Vegetated_MEDIAN_LAI_MIN = float(_fc["vegetated"]["median_lai_min"])
CONTROL_Vegetated_Q10_LAI_MIN = float(_fc["vegetated"]["q10_lai_min"])
CONTROL_Vegetated_MIN_LAI_OBS = int(_fc["vegetated"]["min_lai_obs"])

# ── Temperature (ERA5) ───────────────────────────────────────────────
_te = CONFIG["temperature"]
ERA5_TEMP_COL = _te["era5_temp_col"]
ERA5_COLS_NEEDED = ["date", "lat_id", "lon_id", ERA5_TEMP_COL]
TEMP_BINS = np.array([_resolve_scalar(v) for v in _te["bins"]], dtype=float)
TEMP_LABELS = list(_te["labels"])

# ── Detrending ───────────────────────────────────────────────────────
_de = CONFIG["detrending"]
HARMONIC_MIN_OBS = int(_de["harmonic_min_obs"])
HARMONIC_MIN_YEARS = int(_de["harmonic_min_years"])
CYCLIC_SPLINE_DF = int(_de["cyclic_spline_df"])
CYCLIC_SPLINE_MIN_OBS = int(_de["cyclic_spline_min_obs"])
CYCLIC_SPLINE_MIN_YEARS = int(_de["cyclic_spline_min_years"])

# ── DOY (retained only for feasibility report) ───────────────────────
DOY_PM1_MIN_DONORS = 3
DOY_PM1_MIN_CELL_OBS = 40
DOY_PM1_WINDOW = 1

# ── SII exposure ─────────────────────────────────────────────────────
_si = CONFIG["sii"]
SII_RAW_COL = _si["raw_col"]
SII_SHIFT_BEFORE_ROLL = bool(_si["shift_before_roll"])
SII_MIN_PERIODS = _si["min_periods"]
if SII_MIN_PERIODS is not None:
    SII_MIN_PERIODS = int(SII_MIN_PERIODS)

# ── Surrogates ───────────────────────────────────────────────────────
_su = CONFIG["surrogates"]
SURROGATE_N = int(_su["n"])
SURROGATE_SMOKE_N = int(_su["smoke_n"])
SURROGATE_SEED = int(_su["seed"])
SURROGATE_CIRC_SHIFT_MIN = int(_su["circ_shift_min"])
SURROGATE_BLOCK_SIZE = int(_su["block_size"])
SURROGATE_MODES = list(_su["modes"])

# ── Smoke-test subset ────────────────────────────────────────────────
_sm = CONFIG["smoke"]
SMOKE_YEARS = [int(y) for y in _sm["years"]]
SMOKE_CELL_SAMPLE_SIZE = int(_sm["cell_sample_size"])
SMOKE_DETERMINISTIC_SEED = int(_sm["deterministic_seed"])
SMOKE_MIN_YEARS_PER_CELL = int(_sm["min_years_per_cell"])

# ── Performance ──────────────────────────────────────────────────────
PARQUET_ENGINE = CONFIG["performance"]["parquet_engine"]


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════

def setup_dirs(smoke: bool = False):
    """Create all output directories."""
    dirs = [DATA_INTERIM, DATA_PROCESSED, RESULTS, REPORTS, LOGS]
    if smoke:
        dirs = [LOGS / "smoke" / "data", LOGS / "smoke" / "results", LOGS / "smoke" / "reports", LOGS / "smoke" / "logs"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def bin_temperature(temp: pd.Series) -> pd.DataFrame:
    """Assign temperature bins. -1°C → 'Cold' (not 'Frozen')."""
    cat = pd.cut(temp, bins=TEMP_BINS, labels=TEMP_LABELS, include_lowest=True)
    ids = cat.cat.codes.astype("float32").replace(-1, np.nan)
    lab = cat.astype("string")
    # Fix boundary: -1.0°C should be 'Cold', not 'Frozen'
    frozen = (temp == -1.0) & (lab == "Frozen")
    if frozen.any():
        cold_id = float(TEMP_LABELS.index("Cold"))
        ids = ids.where(~frozen, cold_id)
        lab = lab.where(~frozen, "Cold")
    return pd.DataFrame({"temp_bin_id": ids, "temp_bin_label": lab}, index=temp.index)


def atomic_write(df_or_dict, path: Path, fmt: str = None, **kwargs):
    """Write atomically: tmp → validate → rename. Supports DataFrame, dict, list."""
    if fmt is None:
        fmt = path.suffix.lstrip(".")
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if fmt == "csv":
            d = pd.DataFrame(df_or_dict if isinstance(df_or_dict, (list, dict)) else df_or_dict)
            d.to_csv(tmp, index=False, **kwargs)
        elif fmt == "parquet":
            df_or_dict.to_parquet(tmp, engine=PARQUET_ENGINE, index=False, **kwargs)
        elif fmt == "json":
            with open(tmp, "w") as f:
                json.dump(df_or_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        if tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, path)
        else:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Atomic write produced empty file: {path}")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def full_sha256(path: Path, fast_large_threshold_bytes: int = 1024 * 1024 * 1024) -> str:
    """Return full 64-char hex SHA-256 of file (chunked, memory-safe).

    For files larger than `fast_large_threshold_bytes` a fast metadata digest
    (size + mtime) is used instead of reading the entire file. This keeps
    checkpoint validation practical for very large raw inputs while still
    detecting replacements.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > fast_large_threshold_bytes:
        mtime = path.stat().st_mtime
        return hashlib.sha256(f"fast_digest:{size}:{mtime:.9f}".encode()).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def obs_hash(date, lat_id, lon_id) -> str:
    """Deterministic observation key hash."""
    return hashlib.sha256(f"{date}_{lat_id}_{lon_id}".encode()).hexdigest()


def config_hash() -> str:
    """SHA-256 of the canonical config file."""
    return full_sha256(CONFIG_PATH)


# ── Run metadata ───────────────────────────────────────────────────────
RUN_METADATA_PATH = RESULTS / "run_metadata.json"


def load_run_metadata(path: Path = RUN_METADATA_PATH) -> dict:
    """Load run metadata written by the runner."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def current_run_id() -> str:
    return load_run_metadata().get("run_id", "")


def current_run_type() -> str:
    return load_run_metadata().get("run_type", "")


def current_analysis_run_id() -> str:
    return load_run_metadata().get("analysis_run_id", "")


def current_execution_session_id() -> str:
    return load_run_metadata().get("execution_session_id", "")


def run_provenance_dict(dataset_files: list = None, producing_stage: str = "",
                        metadata_path: Path = RUN_METADATA_PATH) -> dict:
    """Return a dictionary of provenance columns for result tables."""
    meta = load_run_metadata(metadata_path)
    d = {
        "analysis_run_id": meta.get("analysis_run_id", ""),
        "execution_session_id": meta.get("execution_session_id", ""),
        "run_type": meta.get("run_type", ""),
        "git_commit": meta.get("git_commit", ""),
        "config_hash": meta.get("config_hash", ""),
        "input_manifest_hash": meta.get("input_manifest_hash", ""),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "producing_stage": producing_stage,
    }
    if dataset_files:
        h = hashlib.sha256()
        for f in dataset_files:
            if isinstance(f, (str, Path)) and Path(f).exists():
                h.update(full_sha256(Path(f)).encode())
        d["analysis_dataset_hash"] = h.hexdigest()
    return d


# ── Climatological-index helpers (shared by cyclic spline and DOY code) ─

def mmdd_to_clim_index(doy, year):
    """Convert (doy, year) to climatological index 1..365, excluding Feb 29.

    Jan 31 = 31, Feb 1 = 32, …, Dec 31 = 365.  Feb 29 → NaN.
    Accepts scalar ints or pandas Series (aligned).
    """
    date = pd.to_datetime({"year": year, "month": 1, "day": 1}) + pd.to_timedelta(doy - 1, unit="D")
    month = date.dt.month
    day = date.dt.day
    is_feb29 = (month == 2) & (day == 29)
    cum_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    idx = cum_days[month.values - 1] + day.values
    return pd.Series(np.where(is_feb29, np.nan, idx), index=doy.index)


def clim_index_to_mmdd(idx):
    """Convert climatological index 1..365 back to 'MM-DD' string."""
    idx = int(idx)
    if not (1 <= idx <= 365):
        return ""
    cum_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    for m in range(12):
        upper = cum_days[m + 1] if m < 11 else 365
        if idx <= upper:
            day = idx - cum_days[m]
            return f"{m + 1:02d}-{day:02d}"
    return ""


def build_climatological_neighbor_table(windows: list[int], period: int = 365) -> pd.DataFrame:
    """Return (clim_idx, donor_clim_idx, window_half_width) table for circular windows.

    The distance between two climatological indices is cyclic on ``period`` days:
    day 1 and day ``period`` are neighbours when ``window >= 1``.
    """
    rows = []
    for w in windows:
        for c in range(1, period + 1):
            for delta in range(-w, w + 1):
                nc = ((c - 1 + delta) % period) + 1
                rows.append((c, nc, w))
    return pd.DataFrame(rows, columns=["clim_idx", "donor_clim_idx", "window_half_width"])
