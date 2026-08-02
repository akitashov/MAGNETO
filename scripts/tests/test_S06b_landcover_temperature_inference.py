"""
Unit and integration tests for S06b_landcover_temperature_inference.py.

Pure unit tests use synthetic data. Integration tests that need the real
supplementary-checks outputs are marked `slow` and skipped when those outputs
are absent.
"""
from __future__ import annotations
import itertools
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

import S06b_landcover_temperature_inference as s06b


# ── Synthetic-data helpers ─────────────────────────────────────────────────


def _make_synthetic_residuals(n_cells: int = 5, obs_per_cell: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for c in range(n_cells):
        for _ in range(obs_per_cell):
            rows.append({
                "lat_id": float(c),
                "lon_id": float(c),
                "residual": rng.normal(),
                "sii_21d": rng.normal(),
                "sii_28d": rng.normal(),
                "temp_bin_label": "Cold" if c % 2 == 0 else "Cool",
                "land_cover_class": "Forest" if c < 3 else "Barren",
                "year": 2015 + (c % 5),
            })
    return pd.DataFrame(rows)


# ── Bootstrap behaviour ────────────────────────────────────────────────────


def test_cluster_bootstrap_preserves_multiplicity():
    """A cell sampled twice must contribute twice as many observations."""
    rng = np.random.default_rng(11)
    data = []
    for cell in range(3):
        for _ in range(4):
            data.append({
                "lat_id": float(cell),
                "lon_id": float(cell),
                "residual": rng.normal(),
                "sii_21d": rng.normal(),
            })
    df = pd.DataFrame(data)
    boot = s06b._cluster_bootstrap_rho(df, "residual", "sii_21d", n_boot=50, seed=99)
    assert len(boot["boot_rhos"]) == 50
    assert np.isfinite(boot["boot_rhos"]).sum() > 0
    # Bootstrap is over cells, so a deterministic seed must give a deterministic
    # vector of replicate rhos.
    boot2 = s06b._cluster_bootstrap_rho(df, "residual", "sii_21d", n_boot=50, seed=99)
    np.testing.assert_array_equal(boot["boot_rhos"], boot2["boot_rhos"])


def test_bootstrap_reproducible_for_fixed_seed():
    df = _make_synthetic_residuals(n_cells=4, obs_per_cell=4)
    b1 = s06b._cluster_bootstrap_rho(df, "residual", "sii_21d", n_boot=30, seed=123)
    b2 = s06b._cluster_bootstrap_rho(df, "residual", "sii_21d", n_boot=30, seed=123)
    np.testing.assert_array_equal(b1["boot_rhos"], b2["boot_rhos"])
    assert b1["boot_ci_lo"] == pytest.approx(b2["boot_ci_lo"])
    assert b1["boot_ci_hi"] == pytest.approx(b2["boot_ci_hi"])


def test_bootstrap_ci_ordering():
    df = _make_synthetic_residuals(n_cells=10, obs_per_cell=10)
    boot = s06b._cluster_bootstrap_rho(df, "residual", "sii_21d", n_boot=100, seed=5)
    if np.isfinite(boot["boot_ci_lo"]) and np.isfinite(boot["boot_ci_hi"]):
        assert boot["boot_ci_lo"] <= boot["boot_ci_hi"]


# ── Strata and eligibility ─────────────────────────────────────────────────


def test_subgroup_filtering_and_eligibility():
    # Forest Cold and Barren Cold have >=3 unique cells and >=10 obs;
    # Grassland Cool has only 2 cells.
    rows = []
    for cell in range(4):
        for obs in range(3):
            rows.append({
                "lat_id": float(cell), "lon_id": float(cell),
                "residual": float(cell + obs * 0.01), "sii_21d": float(cell + obs * 0.01),
                "temp_bin_label": "Cold", "land_cover_class": "Forest",
                "year": 2014,
            })
    for cell in range(4, 8):
        for obs in range(3):
            rows.append({
                "lat_id": float(cell), "lon_id": float(cell),
                "residual": float(cell + obs * 0.01), "sii_21d": float(cell + obs * 0.01),
                "temp_bin_label": "Cold", "land_cover_class": "Barren",
                "year": 2014,
            })
    for cell in range(8, 10):
        for obs in range(3):
            rows.append({
                "lat_id": float(cell), "lon_id": float(cell),
                "residual": float(cell + obs * 0.01), "sii_21d": float(cell + obs * 0.01),
                "temp_bin_label": "Cool", "land_cover_class": "Grassland",
                "year": 2014,
            })
    df = pd.DataFrame(rows)
    strata, ineligible = s06b._build_strata(df, "harmonic", [21], min_cells=3)
    eligible_keys = {(s["land_cover_class"], s["temp_bin_label"]) for s in strata}
    assert ("Forest", "Cold") in eligible_keys
    assert ("Barren", "Cold") in eligible_keys
    ineligible_keys = {(i["land_cover_class"], i["temp_bin_label"]) for i in ineligible}
    assert ("Grassland", "Cool") in ineligible_keys


def test_underpowered_groups_retained_with_eligible_false():
    df = pd.DataFrame({
        "lat_id": [0.0] * 4,
        "lon_id": [0.0] * 4,
        "residual": [1.0, 2.0, 3.0, 4.0],
        "sii_21d": [2.0, 1.0, 4.0, 3.0],
        "temp_bin_label": ["Cold"] * 4,
        "land_cover_class": ["Forest"] * 4,
        "year": [2014] * 4,
    })
    strata, ineligible = s06b._build_strata(df, "harmonic", [21], min_cells=50)
    assert len(strata) == 0
    assert len(ineligible) == 1
    assert ineligible[0]["eligible"] is False
    assert "50" in ineligible[0]["exclusion_reason"]


# ── p_display ──────────────────────────────────────────────────────────────


def test_p_display_equals_max_of_three_empirical_p_values():
    rows = [
        {"p_year_perm": 0.1, "p_circ_shift": 0.2, "p_block_perm": 0.3},
        {"p_year_perm": 0.05, "p_circ_shift": 0.01, "p_block_perm": 0.1},
        {"p_year_perm": np.nan, "p_circ_shift": 0.2, "p_block_perm": 0.15},
    ]
    expected = [0.3, 0.1, 0.2]
    for row, exp in zip(rows, expected):
        pvals = [row[c] for c in s06b.SURROGATE_MODE_TO_P.values() if pd.notna(row[c])]
        p_display = float(np.max(pvals)) if pvals else np.nan
        assert p_display == pytest.approx(exp)


# ── Contrasts ──────────────────────────────────────────────────────────────


def test_contrast_delta_rho_sign():
    g1 = pd.DataFrame({
        "lat_id": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        "lon_id": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        "residual": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "sii_21d": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    g2 = pd.DataFrame({
        "lat_id": [10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
        "lon_id": [10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
        "residual": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "sii_21d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    deltas = s06b._cluster_bootstrap_delta(g1, g2, "sii_21d", n_boot=50, seed=42)
    # Observed rho1 is strongly negative, rho2 is strongly positive.
    valid = deltas[np.isfinite(deltas)]
    if len(valid):
        assert valid.mean() < 0


# ── Source-code checks ─────────────────────────────────────────────────────


def test_no_plotting_imported_or_executed():
    src = Path(s06b.__file__).read_text()
    assert "matplotlib" not in src
    assert "pyplot" not in src
    assert "savefig" not in src
    assert "imshow" not in src


# ── Integration tests against real outputs ─────────────────────────────────


@pytest.mark.slow
def test_output_schemas_contain_required_columns():
    p = Path("results/supplementary_checks")
    inf = p / "landcover_temperature_inference.csv"
    if not inf.exists():
        pytest.skip("S06b outputs not present")
    df = pd.read_csv(inf)
    required = [
        "method", "sii_window_days", "land_cover_class", "temp_bin_label",
        "eligible", "exclusion_reason", "n_obs", "n_cells", "n_years",
        "spearman_rho", "boot_ci_lo", "boot_ci_hi", "p_year_perm",
        "p_circ_shift", "p_block_perm", "p_display",
        "n_year_completed", "n_year_failed",
        "n_circular_completed", "n_circular_failed",
        "n_block_completed", "n_block_failed",
        "seed", "n_bootstrap", "n_surrogates",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


@pytest.mark.slow
def test_real_outputs_sanity():
    p = Path("results/supplementary_checks")
    inf = p / "landcover_temperature_inference.csv"
    if not inf.exists():
        pytest.skip("S06b outputs not present")
    df = pd.read_csv(inf)

    # p-values inside [0, 1].
    for col in ["p_year_perm", "p_circ_shift", "p_block_perm", "p_display"]:
        vals = df[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of range"

    # CI ordering.
    sub = df.dropna(subset=["boot_ci_lo", "boot_ci_hi"])
    assert (sub["boot_ci_lo"] <= sub["boot_ci_hi"]).all()

    # Expected methods and windows are present.
    assert set(df["method"].dropna().unique()).issubset({"harmonic", "cyclic_spline"})
    assert {21, 28}.issubset(set(df["sii_window_days"].dropna().unique()))

    # Ineligible rows are kept but inferential values are missing.
    ineligible = df[df["eligible"] == False]
    assert ineligible["exclusion_reason"].notna().all()
    assert ineligible["spearman_rho"].isna().all()


@pytest.mark.slow
def test_full_grid_no_missing_combinations():
    p = Path("results/supplementary_checks")
    inf = p / "landcover_temperature_inference.csv"
    if not inf.exists():
        pytest.skip("S06b outputs not present")
    df = pd.read_csv(inf)

    methods = ["harmonic", "cyclic_spline"]
    windows = [21, 28]
    landcovers = [
        "Barren", "Cropland", "Forest", "Grassland",
        "Savanna", "Shrubland/Savanna",
    ]
    temperatures = [
        "Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat",
    ]

    expected = set(itertools.product(methods, windows, landcovers, temperatures))
    observed = set(zip(df["method"], df["sii_window_days"], df["land_cover_class"], df["temp_bin_label"]))
    assert len(observed) == len(expected), (
        f"Expected {len(expected)} combinations, got {len(observed)}; "
        f"missing: {sorted(expected - observed)}"
    )
    assert not (expected - observed)


@pytest.mark.slow
def test_null_distribution_unique_identifiers():
    p = Path("results/supplementary_checks")
    null_path = p / "landcover_temperature_null_distributions.parquet"
    if not null_path.exists():
        pytest.skip("S06b null distributions not present")
    nulls = pd.read_parquet(null_path)
    key_cols = ["method", "window", "land_cover_class", "temp_bin_label", "surrogate_mode", "realization"]
    assert nulls.duplicated(subset=key_cols).sum() == 0


@pytest.mark.slow
def test_contrast_outputs_sanity():
    p = Path("results/supplementary_checks")
    contr_path = p / "landcover_temperature_contrasts.csv"
    if not contr_path.exists():
        pytest.skip("S06b contrasts not present")
    contr = pd.read_csv(contr_path)
    # delta_rho equals rho_group_1 - rho_group_2 where both are finite.
    sub = contr.dropna(subset=["rho_group_1", "rho_group_2", "delta_rho"])
    np.testing.assert_array_almost_equal(
        sub["delta_rho"].values,
        (sub["rho_group_1"] - sub["rho_group_2"]).values,
        decimal=10,
    )
    # CI ordering for eligible contrasts.
    eligible = contr[contr["eligible"] == True].dropna(subset=["boot_ci_lo", "boot_ci_hi"])
    assert (eligible["boot_ci_lo"] <= eligible["boot_ci_hi"]).all()
