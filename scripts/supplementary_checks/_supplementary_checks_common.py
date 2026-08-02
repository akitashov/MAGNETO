"""
MAGNETO supplementary_checks analysis helpers.

Common loading, exposure-building, statistics, provenance, and figure utilities
for the S01...S09 scripts. This module is deliberately separate from the core
pipeline so that it can evolve without touching validated main results.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

# Make the parent scripts directory importable for _Common and magneto_lib.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (
    PROJECT_ROOT,
    CONFIG_PATH,
    PARQUET_ENGINE,
    FILE_QC,
    FILE_RES_HARMONIC,
    FILE_RES_CYCLIC,
    FILE_MATCHED_HC,
    FILE_OMNI,
    FILE_FIXED,
    FILE_SURROGATES,
    FILE_OUTPUT_MANIFEST,
    SII_WINDOWS,
    SII_SHIFT_BEFORE_ROLL,
    SII_MIN_PERIODS,
    atomic_write,
    full_sha256,
    run_provenance_dict as _base_provenance,
)
from magneto_lib import compute_stats, run_fixed_window, run_surrogates, build_daily_sii

# ── Revision configuration ─────────────────────────────────────────────
SUPPLEMENTARY_CHECKS_CONFIG_PATH = PROJECT_ROOT / "config" / "supplementary_checks.yaml"


def load_supplementary_checks_config() -> dict[str, Any]:
    with open(SUPPLEMENTARY_CHECKS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_REV_CFG = load_supplementary_checks_config()


def _supplementary_checks_paths() -> dict[str, Path]:
    p = _REV_CFG["paths"]
    return {
        "results": PROJECT_ROOT / p["results_supplementary_checks"],
        "reports": PROJECT_ROOT / p["reports_supplementary_checks"],
        "figures": PROJECT_ROOT / p["figures_supplementary_checks"],
        "data": PROJECT_ROOT / p["data_supplementary_checks"],
    }


SUPPLEMENTARY_CHECKS_RESULTS = _supplementary_checks_paths()["results"]
SUPPLEMENTARY_CHECKS_REPORTS = _supplementary_checks_paths()["reports"]
SUPPLEMENTARY_CHECKS_FIGURES = _supplementary_checks_paths()["figures"]
SUPPLEMENTARY_CHECKS_DATA = _supplementary_checks_paths()["data"]


def make_supplementary_checks_dirs() -> None:
    for d in (SUPPLEMENTARY_CHECKS_RESULTS, SUPPLEMENTARY_CHECKS_REPORTS, SUPPLEMENTARY_CHECKS_FIGURES, SUPPLEMENTARY_CHECKS_DATA):
        d.mkdir(parents=True, exist_ok=True)


# ── Provenance ─────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            )
            .strip()
        )
    except Exception:
        return ""


def supplementary_checks_config_hash() -> str:
    h = hashlib.sha256()
    for path in (CONFIG_PATH, SUPPLEMENTARY_CHECKS_CONFIG_PATH):
        h.update(full_sha256(path).encode())
    return h.hexdigest()


def supplementary_checks_provenance(
    stage: str,
    dataset_files: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Return provenance columns for supplementary checks result tables."""
    prov = _base_provenance(
        dataset_files=dataset_files,
        producing_stage=stage,
    )
    prov["supplementary_checks_config_hash"] = supplementary_checks_config_hash()
    prov["supplementary_checks_git_commit"] = git_commit()
    return prov


def attach_provenance(df: pd.DataFrame, stage: str,
                      dataset_files: list[Path | str] | None = None) -> pd.DataFrame:
    for k, v in supplementary_checks_provenance(stage, dataset_files).items():
        df[k] = v
    return df


# ── Residual loaders ───────────────────────────────────────────────────

def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, engine=PARQUET_ENGINE, columns=columns)


def _add_region_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Merge region_flags from the QC table onto a residual DataFrame."""
    qc = _read_parquet(FILE_QC, columns=["date", "lat_id", "lon_id", "region_flags"])
    qc["date"] = pd.to_datetime(qc["date"])
    return df.merge(qc, on=["date", "lat_id", "lon_id"], how="left")


def load_harmonic_residuals(outcome: str = "sif_771nm") -> pd.DataFrame:
    cols = [
        "date", "year", "lat_id", "lon_id", "observed_sif", "residual",
        "temp_bin_label", "lai_quartile", "is_strict_low_lai", "is_vegetated", "is_Sahara",
        "pass_flag",
    ]
    df = _read_parquet(FILE_RES_HARMONIC, columns=cols)
    df = df[df["pass_flag"] == True].copy()  # noqa: E712
    df["method"] = "harmonic"
    df = df.rename(columns={"observed_sif": outcome})
    df["date"] = pd.to_datetime(df["date"])
    return _add_region_flags(df)


def load_cyclic_residuals(outcome: str = "sif_771nm") -> pd.DataFrame:
    cols = [
        "date", "year", "lat_id", "lon_id", "observed_sif", "residual",
        "temp_bin_label", "lai_quartile", "is_strict_low_lai", "is_vegetated", "is_Sahara",
        "fit_status",
    ]
    df = _read_parquet(FILE_RES_CYCLIC, columns=cols)
    df = df[df["fit_status"] == "ok"].copy()
    df["method"] = "cyclic_spline"
    df = df.rename(columns={"observed_sif": outcome})
    df["date"] = pd.to_datetime(df["date"])
    return _add_region_flags(df)


def load_matched_residuals(outcome: str = "sif_771nm") -> pd.DataFrame:
    cols = [
        "date", "year", "lat_id", "lon_id",
        "observed_sif_harmonic", "residual_harmonic",
        "observed_sif_cyclic_spline", "residual_cyclic_spline",
        "temp_bin_label", "lai_quartile", "is_strict_low_lai", "is_vegetated", "is_Sahara",
    ]
    df = _read_parquet(FILE_MATCHED_HC, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return _add_region_flags(df)


def load_residual_dataset(
    method: str,
    outcome: str = "sif_771nm",
) -> pd.DataFrame:
    """Load a valid residual dataset for one detrending method."""
    if method == "harmonic":
        return load_harmonic_residuals(outcome)
    if method == "cyclic_spline":
        return load_cyclic_residuals(outcome)
    raise ValueError(f"Unknown method: {method}")


def load_all_residuals(outcome: str = "sif_771nm") -> dict[str, pd.DataFrame]:
    """Return {'harmonic': df, 'cyclic_spline': df, 'matched': df_matched}."""
    return {
        "harmonic": load_harmonic_residuals(outcome),
        "cyclic_spline": load_cyclic_residuals(outcome),
        "matched": load_matched_residuals(outcome),
    }


# ── Exposure builders ──────────────────────────────────────────────────

def build_windowed_exposure(
    omni_df: pd.DataFrame,
    value_col: str,
    windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    """shift(1) -> rolling.mean for an arbitrary daily exposure variable."""
    df = omni_df[["date", value_col]].copy()
    shifted = df[value_col].shift(1) if SII_SHIFT_BEFORE_ROLL else df[value_col]
    out = df[["date"]].copy()
    min_p = SII_MIN_PERIODS
    for w in windows:
        mp = w if min_p is None else min_p
        out[f"{prefix}_{w}d"] = shifted.rolling(window=w, min_periods=mp).mean()
    return out


def build_sii_exposure(windows: list[int]) -> pd.DataFrame:
    """Build SII exposure windows using the canonical OMNI daily series."""
    omni = build_daily_sii()
    return build_windowed_exposure(omni, "sii_mean", windows, "sii")


def build_omni_exposure(
    value_col: str,
    windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    """Build exposure windows for an OMNI variable (e.g. kp_mean, f10_7_mean)."""
    omni = pd.read_feather(FILE_OMNI, columns=["date", value_col])
    omni["date"] = pd.to_datetime(omni["date"])
    omni = omni.sort_values("date").reset_index(drop=True)
    full_dates = pd.date_range(omni["date"].min(), omni["date"].max(), freq="D")
    full = pd.DataFrame({"date": full_dates})
    full = full.merge(omni, on="date", how="left")
    return build_windowed_exposure(full, value_col, windows, prefix)


def attach_exposure(df: pd.DataFrame, exposure_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(exposure_df, on="date", how="left")


# ── Strata / statistics ────────────────────────────────────────────────

def build_supplementary_checks_strata(
    df: pd.DataFrame,
    sample_type: str,
    method: str,
    residual_col: str,
    windows: list[int] | None = None,
    group_cols: list[str] | None = None,
    min_n: int = 10,
) -> list[dict]:
    """Build strata compatible with run_fixed_window / run_surrogates.

    group_cols can include 'temp_bin_label', 'lai_quartile', etc.
    """
    if windows is None:
        windows = SII_WINDOWS
    if group_cols:
        df = df.dropna(subset=group_cols)
        if df.empty:
            return []
        groups = df.groupby(group_cols, sort=True)
    else:
        groups = [(None, df)]

    strata = []
    for keys, sub in groups:
        if isinstance(keys, tuple):
            key_dict = dict(zip(group_cols, keys))
        elif keys is None:
            key_dict = {}
        else:
            key_dict = {group_cols[0]: keys}
        for w in windows:
            sii_col = f"sii_{w}d"
            if sii_col not in sub.columns:
                continue
            valid = sub[[residual_col, sii_col]].dropna()
            if len(valid) < min_n:
                continue
            strata.append({
                "sample_type": sample_type,
                "method": method,
                "residual_col": residual_col,
                "sii_col": sii_col,
                "sii_window_days": w,
                **key_dict,
                "data": sub,
            })
    return strata


def cell_cluster_bootstrap_rho(
    df: pd.DataFrame,
    residual_col: str,
    sii_col: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Return a cell-cluster bootstrap confidence interval for Spearman rho.

    Observations are grouped by (lat_id, lon_id) cell.  Cells are resampled with
    replacement; within each resampled cell all observations are kept.  The CI
    is returned as the symmetric ``ci`` quantiles of the bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    needed = [residual_col, sii_col, "lat_id", "lon_id"]
    valid = df[needed].dropna()
    if len(valid) < 10:
        return {"boot_rho_median": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan}

    cells = valid[["lat_id", "lon_id"]].drop_duplicates().values
    cell_idx = valid[["lat_id", "lon_id"]].apply(tuple, axis=1).values
    cell_map = {tuple(c): i for i, c in enumerate(cells)}
    labels = np.array([cell_map[t] for t in cell_idx])
    n_cells = len(cells)

    x = valid[residual_col].values.astype(float)
    y = valid[sii_col].values.astype(float)

    rhos = []
    for _ in range(n_boot):
        sampled_cells = rng.integers(0, n_cells, size=n_cells)
        keep = np.isin(labels, sampled_cells)
        if keep.sum() < 10:
            continue
        # Within a resampled cell, preserve all observations (cell-cluster
        # resampling).  Observations from cells drawn multiple times appear
        # multiple times.
        ix = np.where(keep)[0]
        boot_rho = float(compute_stats(x[ix], y[ix])["spearman_rho"])
        rhos.append(boot_rho)

    if len(rhos) < n_boot // 2:
        return {"boot_rho_median": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan}

    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    return {
        "boot_rho_median": float(np.median(rhos)),
        "boot_ci_lo": float(np.quantile(rhos, lo)),
        "boot_ci_hi": float(np.quantile(rhos, hi)),
    }


def run_supplementary_checks_fixed_window(
    strata: list[dict],
    bootstrap: bool = False,
    n_boot: int = 1000,
    boot_seed: int | None = None,
) -> pd.DataFrame:
    """Compute fixed-window statistics for supplementary checks strata.

    Skips the per-stratum observation-key hash by default; it is expensive for
    large exploratory scans and is not required for supplementary checks outputs.
    """
    rows = []
    n_total = len(strata)
    for i, st in enumerate(strata, start=1):
        df = st["data"]
        residual_col = st["residual_col"]
        sii_col = st["sii_col"]
        needed = [residual_col, sii_col, "lat_id", "lon_id"]
        if not all(c in df.columns for c in needed):
            continue
        valid = df[needed].dropna()
        if len(valid) < 10:
            continue
        stats = compute_stats(valid[residual_col].values, valid[sii_col].values)
        n_cells = valid[["lat_id", "lon_id"]].drop_duplicates().shape[0]
        n_years = df["year"].nunique() if "year" in df.columns else np.nan
        row = {
            "sample_type": st["sample_type"],
            "method": st["method"],
            "residual_column": residual_col,
            "sii_window": sii_col,
            "sii_window_days": st.get("sii_window_days", int(sii_col.split("_")[1].rstrip("d"))),
            "n_cells": n_cells,
            "n_years": n_years,
            "obs_hash": "",
        }
        for k in st:
            if k not in {"sample_type", "method", "residual_col", "sii_col", "sii_window_days", "data"}:
                row[k] = st[k]
        row.update(stats)

        if bootstrap:
            boot_seed_i = boot_seed + i if boot_seed is not None else None
            row.update(cell_cluster_bootstrap_rho(
                df, residual_col, sii_col, n_boot=n_boot, seed=boot_seed_i
            ))

        rows.append(row)
        if i % 50 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] computed {len(rows)} strata so far")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_supplementary_checks_surrogates(
    strata: list[dict],
    n_surr: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper around magneto_lib.run_surrogates."""
    return run_surrogates(strata, n_surr=n_surr, seed=seed)


# ── Main-result loaders for figures ────────────────────────────────────

def load_main_fixed_window_results() -> pd.DataFrame:
    return pd.read_csv(FILE_FIXED)


def load_main_surrogate_summary() -> pd.DataFrame:
    return pd.read_csv(FILE_SURROGATES)


def load_output_manifest() -> dict[str, Any]:
    if FILE_OUTPUT_MANIFEST.exists():
        return json.loads(FILE_OUTPUT_MANIFEST.read_text())
    return {}


# ── Generic figure helpers ─────────────────────────────────────────────

def savefig(path: Path, fig=None, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    tmp = path.with_suffix(".tmp" + path.suffix)
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Empty figure written to {path}")


def scenario_label(sample_type: str, group_name: str, temperature_class: str = "") -> str:
    """Human-readable scenario label for figures."""
    if sample_type == "pooled_full":
        return "Global pooled"
    if sample_type.startswith("control_"):
        return group_name.replace("_", " ")
    if sample_type == "temperature":
        return temperature_class
    return sample_type


# ── Land-cover helpers ─────────────────────────────────────────────────

def load_landcover_cell_grid() -> pd.DataFrame | None:
    """Load the prepared 1-degree land-cover cell grid if it exists."""
    path = SUPPLEMENTARY_CHECKS_DATA / _REV_CFG["outputs"]["landcover_grid_parquet"]
    if not path.exists():
        return None
    return pd.read_parquet(path, engine=PARQUET_ENGINE)


def decode_region_flag(region_flags: pd.Series, flag_name: str) -> pd.Series:
    """Decode a boolean mask from the QC region_flags bitmask."""
    bits = {
        "SAA": 1, "CONTROL_NORTH": 2, "POLAR": 4, "SAHARA": 8,
        "LOW_LAI": 16, "HIGH_LAI": 32, "DESERT": 64, "REFERENCE_VEGETATED": 128,
    }
    bit = bits.get(flag_name)
    if bit is None:
        raise ValueError(f"Unknown region flag: {flag_name}")
    return (region_flags.astype(int) & bit) != 0


def dataset_hash_of(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(full_sha256(p).encode())
    return h.hexdigest()
