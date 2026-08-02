#!/usr/bin/env python3
"""
Focused tests for S11_driver_r2_shapley.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_LONG = PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_long.csv"
FULL_MODELS = PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_models.csv"
FULL_BY_WINDOW = PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_by_window.csv"
FULL_JSON = PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_metadata.json"
SMOKE_LONG = PROJECT_ROOT / "results" / "supplementary_checks" / "smoke" / "driver_r2_shapley_long.csv"
SMOKE_MODELS = PROJECT_ROOT / "results" / "supplementary_checks" / "smoke" / "driver_r2_shapley_models.csv"

TOL = 1e-10


@pytest.mark.skipif(not FULL_LONG.exists(), reason="S11 full long table not present")
def test_full_long_schema_and_dimensions():
    df = pd.read_csv(FULL_LONG)
    required = [
        "target", "method", "scenario", "temp_bin_label", "window_days", "driver",
        "feature_column", "shapley_r2", "shapley_r2_percent", "share_of_explained_r2",
        "r2_full", "r2_full_percent", "beta_raw", "beta_standardized", "coefficient_sign",
        "n_obs", "n_cells", "eligible", "eligibility_reason", "calculation_status",
    ]
    assert all(c in df.columns for c in required)
    assert len(df) > 0, "S11 long table is empty"
    assert set(df["driver"].unique()).issubset({"SII", "PAR", "VPD"})
    assert df["window_days"].dtype.kind in "iu"
    assert df["window_days"].min() >= 1
    assert df["window_days"].max() <= 28
    # Long rows must be an integer multiple of the number of drivers present.
    n_drivers = df["driver"].nunique()
    assert len(df) % n_drivers == 0, "Long rows are not a multiple of driver count"


@pytest.mark.skipif(not FULL_MODELS.exists(), reason="S11 full model table not present")
def test_full_models_dimensions():
    df = pd.read_csv(FULL_MODELS)
    assert len(df) > 0, "S11 models table is empty"
    subset_cols = ["r2_sii", "r2_par", "r2_vpd", "r2_sii_par", "r2_sii_vpd", "r2_par_vpd", "r2_full"]
    assert all(c in df.columns for c in subset_cols)
    long = pd.read_csv(FULL_LONG) if FULL_LONG.exists() else None
    if long is not None:
        n_drivers = long["driver"].nunique()
        assert len(df) * n_drivers == len(long), "Model rows do not match long rows / driver_count"


@pytest.mark.skipif(not FULL_LONG.exists(), reason="S11 full long table not present")
def test_no_duplicate_keys():
    df = pd.read_csv(FULL_LONG)
    key = ["scenario", "temp_bin_label", "window_days", "driver"]
    assert not df.duplicated(key, keep=False).any(), "Duplicate keys found"


@pytest.mark.skipif(not FULL_LONG.exists(), reason="S11 full long table not present")
def test_shapley_r2_nonnegative_and_additive():
    df = pd.read_csv(FULL_LONG)
    eligible = df[df["eligible"]]
    assert (eligible["shapley_r2"] >= -TOL).all(), "Negative Shapley R² found"
    grouped = eligible.groupby(["scenario", "temp_bin_label", "window_days"])["shapley_r2"].sum()
    full_r2 = eligible.groupby(["scenario", "temp_bin_label", "window_days"])["r2_full"].first()
    assert np.allclose(grouped.values, full_r2.values, atol=TOL), "Shapley sum != r2_full"


@pytest.mark.skipif(not FULL_LONG.exists(), reason="S11 full long table not present")
def test_share_of_explained_r2_sums_to_one():
    df = pd.read_csv(FULL_LONG)
    eligible = df[df["eligible"]]
    grouped = eligible.groupby(["scenario", "temp_bin_label", "window_days"])["share_of_explained_r2"].sum()
    # Where r2_full > epsilon, shares should sum to 1
    full_r2 = eligible.groupby(["scenario", "temp_bin_label", "window_days"])["r2_full"].first()
    mask = full_r2 > TOL
    assert np.allclose(grouped[mask].values, 1.0, atol=TOL), "Shares do not sum to 1"


@pytest.mark.skipif(not FULL_MODELS.exists(), reason="S11 full model table not present")
def test_r2_in_valid_range():
    df = pd.read_csv(FULL_MODELS)
    eligible = df[df["eligible"]]
    cols = ["r2_sii", "r2_par", "r2_vpd", "r2_sii_par", "r2_sii_vpd", "r2_par_vpd", "r2_full"]
    for c in cols:
        finite = eligible[c].dropna()
        assert finite.between(-TOL, 1 + TOL).all(), f"{c} outside [0,1] within tolerance"


@pytest.mark.skipif(not FULL_MODELS.exists(), reason="S11 full model table not present")
def test_decomposition_error_small():
    df = pd.read_csv(FULL_MODELS)
    eligible = df[df["eligible"]]
    assert (eligible["decomposition_error"] <= TOL).all(), "Decomposition error exceeds tolerance"


@pytest.mark.skipif(not FULL_LONG.exists(), reason="S11 full long table not present")
def test_ineligible_groups_retained():
    df = pd.read_csv(FULL_LONG)
    ineligible = df[~df["eligible"]]
    assert len(ineligible) > 0, "Expected ineligible rows"
    assert ineligible["shapley_r2"].isna().all(), "Ineligible rows should have NA shapley_r2"


@pytest.mark.skipif(not FULL_MODELS.exists(), reason="S11 full model table not present")
def test_same_n_obs_for_all_subset_models():
    df = pd.read_csv(FULL_MODELS)
    eligible = df[df["eligible"]]
    assert (eligible["n_obs"] > 0).all(), "Eligible model has zero observations"


@pytest.mark.skipif(not SMOKE_LONG.exists(), reason="S11 smoke output not present")
def test_smoke_separate_directory():
    assert SMOKE_LONG.exists()
    assert FULL_LONG.exists()
    assert SMOKE_LONG != FULL_LONG


@pytest.mark.skipif(not SMOKE_LONG.exists(), reason="S11 smoke output not present")
def test_smoke_dimensions():
    long = pd.read_csv(SMOKE_LONG)
    models = pd.read_csv(SMOKE_MODELS)
    assert len(long) == 15, f"Expected 15 smoke long rows, got {len(long)}"
    assert len(models) == 5, f"Expected 5 smoke model rows, got {len(models)}"
    assert set(long["scenario"].unique()) == {"full_sample"}
    assert set(long["temp_bin_label"].unique()) == {"Cool"}
    assert set(long["window_days"].unique()) == {1, 7, 14, 21, 28}


@pytest.mark.skipif(not FULL_BY_WINDOW.exists(), reason="S12 aggregate not present")
def test_by_window_additive_decomposition():
    df = pd.read_csv(FULL_BY_WINDOW)
    eligible = df[df["eligible"]]
    shapley_sum = eligible["sii_shapley_r2"] + eligible["par_shapley_r2"] + eligible["vpd_shapley_r2"]
    assert np.allclose(shapley_sum.values, eligible["r2_full"].values, atol=TOL), "S12 Shapley sum != r2_full"


def test_cpu_gpu_consistency():
    """Compare CPU and GPU Shapley results on a small sample if CuPy is available."""
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        pytest.skip("CuPy not available")

    sys_path_was = list(sys.path)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "supplementary_checks"))
    try:
        from S11_driver_r2_shapley import _fit_all_subsets, _shapley_values
    finally:
        sys.path[:] = sys_path_was

    rng = np.random.default_rng(42)
    n = 500
    y = rng.standard_normal(n)
    x = rng.standard_normal((n, 3))
    # Add strong correlation to create nontrivial Shapley values
    x[:, 1] += 0.5 * x[:, 0]
    x[:, 2] -= 0.3 * x[:, 0]

    fit_cpu = _fit_all_subsets(x, y, "cpu")
    fit_gpu = _fit_all_subsets(x, y, "gpu")

    for subset in fit_cpu["r2"]:
        assert abs(fit_cpu["r2"][subset] - fit_gpu["r2"][subset]) < 1e-10, f"R² mismatch for {subset}"

    shap_cpu = _shapley_values(fit_cpu["r2"])
    shap_gpu = _shapley_values(fit_gpu["r2"])
    for d in ["SII", "PAR", "VPD"]:
        assert abs(shap_cpu[d] - shap_gpu[d]) < 1e-10, f"Shapley mismatch for {d}"
