"""
Unit tests for the supplementary checks analysis layer.

These tests do not require the full datasets to be present. They exercise the
helper functions in scripts/supplementary_checks/_supplementary_checks_common.py and the parameter logic
of the S01-S09 scripts. Integration tests that need the actual SIF residuals are
marked and skipped when the inputs are absent.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUPPLEMENTARY_CHECKS_DIR = Path(__file__).resolve().parents[1] / "supplementary_checks"
if str(SUPPLEMENTARY_CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPLEMENTARY_CHECKS_DIR))
if str(SUPPLEMENTARY_CHECKS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SUPPLEMENTARY_CHECKS_DIR.parent))

from _supplementary_checks_common import (
    build_windowed_exposure,
    compute_stats,
    load_supplementary_checks_config,
)
from S05_prepare_landcover_grid import _aggregate_cell
from magneto_lib import compute_sii_windows, valid_circular_offsets


# ── Exposure-window tests ──────────────────────────────────────────────

def test_observation_day_excluded_from_window():
    """shift(1) means the current day's value is not included in its own window."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    sii = np.arange(1, 11, dtype=float)
    df = pd.DataFrame({"date": dates, "sii_mean": sii})
    windows = build_windowed_exposure(df, "sii_mean", [3], "sii")
    # With min_periods == window, the first valid window ends on day 4 (index 3)
    # and averages the three *previous* days (1, 2, 3).
    assert np.isnan(windows["sii_3d"].iloc[0])
    assert np.isnan(windows["sii_3d"].iloc[1])
    assert np.isnan(windows["sii_3d"].iloc[2])
    assert windows["sii_3d"].iloc[3] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert windows["sii_3d"].iloc[4] == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)


def test_compute_sii_windows_matches_revision_helper():
    """Canonical compute_sii_windows and build_windowed_exposure agree for SII."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({"date": dates, "sii_mean": np.arange(20, dtype=float)})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_feb29"] = False
    canon = compute_sii_windows(df, "sii_mean", [7, 14])
    helper = build_windowed_exposure(df, "sii_mean", [7, 14], "sii")
    merged = canon.merge(helper, on="date")
    assert np.allclose(merged["sii_7d_x"], merged["sii_7d_y"], equal_nan=True)
    assert np.allclose(merged["sii_14d_x"], merged["sii_14d_y"], equal_nan=True)


def test_same_sample_for_sii_and_alternative_exposure():
    """Dropna on the union of exposure columns guarantees identical n_obs."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    sii = np.arange(30, dtype=float)
    alt = np.where(np.arange(30) % 5 == 0, np.nan, np.arange(30, dtype=float))
    df = pd.DataFrame({"date": dates, "sii_mean": sii, "alt_mean": alt})
    exp = build_windowed_exposure(df, "sii_mean", [7], "sii")
    exp = exp.merge(build_windowed_exposure(df, "alt_mean", [7], "alt"), on="date")
    merged = pd.DataFrame({"date": dates, "residual": np.random.randn(30)}).merge(exp, on="date")
    union = merged.dropna(subset=["residual", "sii_7d", "alt_7d"])
    assert union["sii_7d"].notna().all()
    assert union["alt_7d"].notna().all()
    assert len(union) == merged[["sii_7d", "alt_7d"]].dropna().shape[0]


# ── Surrogate offset tests ─────────────────────────────────────────────

def test_circular_offsets_forbid_modulo_year_aliases():
    """For a 3680-day series, offset 3315 (= n-365) must be forbidden."""
    n = 3680
    offsets = valid_circular_offsets(n, min_shift=30)
    assert (n - 365) not in offsets
    assert (365) not in offsets
    assert (n - 730) not in offsets
    assert (730) not in offsets
    # All offsets are in [1, n-1].
    assert all(1 <= o < n for o in offsets)


# ── Land-cover aggregation tests ───────────────────────────────────────

def test_aggregate_cell_majority_assignment():
    """A cell dominated by forest pixels is assigned Forest."""
    src_lat = np.linspace(-0.45, 0.45, 10)
    src_lon = np.linspace(-0.45, 0.45, 10)
    lc = np.full((10, 10), 1, dtype=int)  # evergreen needleleaf forest
    cfg = load_supplementary_checks_config()
    igbp_map = cfg["landcover"]["igbp_map"]
    result = _aggregate_cell(
        lat=0.0, lon=0.0,
        src_lat=src_lat, src_lon=src_lon, lc=lc, fill=-1,
        igbp_map=igbp_map, majority_threshold=0.5,
    )
    assert result["land_cover_class"] == "Forest"
    assert result["dominant_fraction"] == pytest.approx(1.0)


def test_aggregate_cell_mixed_when_below_threshold():
    """A 50/50 split falls below the majority threshold."""
    src_lat = np.linspace(-0.45, 0.45, 10)
    src_lon = np.linspace(-0.45, 0.45, 10)
    lc = np.full((10, 10), 5, dtype=int)
    lc[:, 5:] = 10  # grassland
    cfg = load_supplementary_checks_config()
    result = _aggregate_cell(
        lat=0.0, lon=0.0,
        src_lat=src_lat, src_lon=src_lon, lc=lc, fill=-1,
        igbp_map=cfg["landcover"]["igbp_map"],
        majority_threshold=0.6,
    )
    assert result["land_cover_class"] == "Mixed/Other Vegetation"


def test_landcover_cell_unambiguous():
    """Each synthetic cell receives exactly one class label."""
    src_lat = np.linspace(-0.45, 0.45, 8)
    src_lon = np.linspace(-0.45, 0.45, 8)
    lc = np.full((8, 8), 12, dtype=int)  # cropland
    cfg = load_supplementary_checks_config()
    result = _aggregate_cell(
        lat=0.0, lon=0.0,
        src_lat=src_lat, src_lon=src_lon, lc=lc, fill=-1,
        igbp_map=cfg["landcover"]["igbp_map"],
        majority_threshold=0.5,
    )
    assert result["land_cover_class"] == "Cropland"
    assert isinstance(result["land_cover_class"], str)


# ── Statistics sanity tests ────────────────────────────────────────────

def test_compute_stats_on_perfect_negative_correlation():
    x = np.arange(10, dtype=float)
    y = -x
    stats = compute_stats(y, x)
    assert stats["spearman_rho"] == pytest.approx(-1.0, abs=1e-9)
    assert stats["n_obs"] == 10


# ── Config loading test ────────────────────────────────────────────────

def test_revision_config_loads():
    cfg = load_supplementary_checks_config()
    assert "windows" in cfg
    assert "fixed" in cfg["windows"]
    assert "landcover" in cfg
    assert "igbp_map" in cfg["landcover"]
