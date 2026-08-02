#!/usr/bin/env python3
"""
MAGNETO pipeline test suite.

Covers configuration, data transformations, detrending correctness,
exposure construction, and surrogate logic.
"""
import sys, os, json, tempfile, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _Common import *

SCRIPT_DIR = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd
import pytest

# Import magneto_lib helpers
import magneto_lib as mlib


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

def test_target_is_771_only():
    assert TARGETS == ["sif_771nm"]


def test_sii_windows_21_28():
    assert SII_WINDOWS == [21, 28]


def test_residual_methods():
    assert RESIDUAL_METHODS == ["harmonic", "cyclic_spline"]


def test_lai_quartiles():
    assert LAI_N_QUARTILES == 4


def test_cyclic_spline_df():
    assert CYCLIC_SPLINE_DF == 8


def test_surrogate_defaults():
    assert SURROGATE_N == 1000
    assert SURROGATE_SMOKE_N == 10
    assert SURROGATE_SEED == 20260717


def test_temp_boundaries():
    s = pd.Series([-1.0, -0.9, 7.0, 7.1, 15.0, 15.1])
    tb = bin_temperature(s)
    assert tb.loc[0, "temp_bin_label"] == "Cold"
    assert tb.loc[1, "temp_bin_label"] == "Cold"
    assert tb.loc[2, "temp_bin_label"] == "Cold"
    assert tb.loc[3, "temp_bin_label"] == "Cool"
    assert tb.loc[4, "temp_bin_label"] == "Cool"
    assert tb.loc[5, "temp_bin_label"] == "Optimum"


def test_config_hash_stable():
    h1 = config_hash()
    h2 = config_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_paths_unversioned():
    for p in [FILE_QC, FILE_LAI_CELLS, FILE_RES_HARMONIC, FILE_RES_CYCLIC,
              FILE_MATCHED_HC, FILE_FIXED, FILE_SURROGATES]:
        path_str = str(p)
        assert "interim_v" not in path_str
        assert "results_v" not in path_str
        assert "v4" not in path_str
        assert "v5" not in path_str


# ═══════════════════════════════════════════════════════════════════════
# Analysis-area filter (SIF is already land-only from upstream ETL)
# ═══════════════════════════════════════════════════════════════════════

def test_analysis_area_filter_keeps_land_flags():
    assert 16 in LAND_KEEP_FLAGS
    assert 32 in LAND_KEEP_FLAGS


def test_analysis_area_filter_excludes_polar():
    assert 4 in LAND_EXCLUDE_FLAGS


# ═══════════════════════════════════════════════════════════════════════
# DOY / climatological helpers
# ═══════════════════════════════════════════════════════════════════════

def test_mmdd_to_clim_index():
    s = pd.Series([59, 60, 61, 365, 366])
    y = pd.Series([2020, 2020, 2020, 2020, 2020])
    idx = mmdd_to_clim_index(s, y)
    assert idx.iloc[0] == 59.0
    assert pd.isna(idx.iloc[1])
    assert idx.iloc[2] == 60.0
    assert idx.iloc[3] == 364.0
    assert idx.iloc[4] == 365.0


def test_mmdd_to_clim_index_leap_dec31_preserved():
    """31 December of a leap year must map to climatological day 365."""
    s = pd.Series([366])
    y = pd.Series([2020])
    idx = mmdd_to_clim_index(s, y)
    assert idx.iloc[0] == 365.0


def test_climatological_neighbor_table_year_boundary():
    """Day 1 and day 365 must be neighbours across the year boundary."""
    table = build_climatological_neighbor_table([1])
    d1 = set(table[(table["clim_idx"] == 1) & (table["window_half_width"] == 1)]["donor_clim_idx"])
    d365 = set(table[(table["clim_idx"] == 365) & (table["window_half_width"] == 1)]["donor_clim_idx"])
    assert d1 == {365, 1, 2}
    assert d365 == {364, 365, 1}


def test_donor_years_exclude_target_year():
    """The target year must not count as a donor year."""
    from _Common import build_climatological_neighbor_table
    presence = pd.DataFrame({
        "lat_id": [1] * 5, "lon_id": [1] * 5,
        "year": [2018, 2019, 2020, 2021, 2022],
        "clim_idx": [100] * 5,
    })
    target = pd.DataFrame({"lat_id": [1], "lon_id": [1], "year": [2020], "clim_idx": [100]})
    neighbors = build_climatological_neighbor_table([1])
    target_exp = target.merge(neighbors, on="clim_idx")
    donor_pres = presence.rename(columns={"year": "donor_year", "clim_idx": "donor_clim_idx"})
    merged = target_exp.merge(donor_pres, on=["lat_id", "lon_id", "donor_clim_idx"], how="inner")
    merged = merged[merged["year"] != merged["donor_year"]]
    counts = merged.groupby(["lat_id", "lon_id", "year", "clim_idx", "window_half_width"])["donor_year"].nunique().reset_index(name="n_donor_years")
    assert counts[counts["window_half_width"] == 1]["n_donor_years"].iloc[0] == 4


def test_donor_years_unique_per_year():
    """Multiple observations from one donor year must contribute only one donor year."""
    from _Common import build_climatological_neighbor_table
    presence = pd.DataFrame({
        "lat_id": [1] * 6, "lon_id": [1] * 6,
        "year": [2018, 2018, 2019, 2020, 2021, 2022],
        "clim_idx": [100, 101, 100, 100, 100, 100],
    })
    target = pd.DataFrame({"lat_id": [1], "lon_id": [1], "year": [2020], "clim_idx": [100]})
    neighbors = build_climatological_neighbor_table([1])
    target_exp = target.merge(neighbors, on="clim_idx")
    donor_pres = presence.rename(columns={"year": "donor_year", "clim_idx": "donor_clim_idx"})
    merged = target_exp.merge(donor_pres, on=["lat_id", "lon_id", "donor_clim_idx"], how="inner")
    merged = merged[merged["year"] != merged["donor_year"]]
    counts = merged.groupby(["lat_id", "lon_id", "year", "clim_idx", "window_half_width"])["donor_year"].nunique().reset_index(name="n_donor_years")
    assert counts[counts["window_half_width"] == 1]["n_donor_years"].iloc[0] == 4


def test_climatological_phase_excludes_feb29():
    s = pd.Series([59, 60, 61])
    y = pd.Series([2020, 2020, 2020])
    phase = (mmdd_to_clim_index(s, y) - 1) / 365.0
    assert pd.isna(phase.iloc[1])
    assert 0.0 <= phase.iloc[0] < 1.0
    assert 0.0 <= phase.iloc[2] < 1.0


def test_mmdd_to_clim_index_same_calendar_date_all_years():
    """The same month/day must map to the same climatological index in every year."""
    leap_mar1 = mmdd_to_clim_index(pd.Series([61]), pd.Series([2020])).iloc[0]
    nonleap_mar1 = mmdd_to_clim_index(pd.Series([60]), pd.Series([2019])).iloc[0]
    assert leap_mar1 == nonleap_mar1 == 60.0
    dec31 = mmdd_to_clim_index(pd.Series([365, 366]), pd.Series([2019, 2020])).iloc
    assert dec31[0] == 365.0
    assert dec31[1] == 365.0


def test_no_test_artifacts_left_in_production_results():
    """Pytest must not write test diagnostics into production results/."""
    for root in [Path("results"), Path("data/processed")]:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file():
                assert "mismatch" not in f.name.lower(), f"test artifact left in production: {f}"
                assert not f.name.endswith(".tmp"), f"temporary file left in production: {f}"


# ═══════════════════════════════════════════════════════════════════════
# SII exposure
# ═══════════════════════════════════════════════════════════════════════

def test_sii_shift_before_roll():
    assert SII_SHIFT_BEFORE_ROLL is True


def test_sii_raw_col():
    assert SII_RAW_COL == "sii_mean"


def test_compute_sii_windows():
    dates = pd.date_range("2020-01-01", periods=40)
    sii = np.arange(40, dtype=float)
    df = pd.DataFrame({"date": dates, SII_RAW_COL: sii, "year": dates.year,
                       "month": dates.month, "day": dates.day, "is_feb29": False})
    out = mlib.compute_sii_windows(df, SII_RAW_COL, [3, 5])
    assert "sii_3d" in out.columns
    assert "sii_5d" in out.columns
    # first value after shift is NaN, second is NaN until window fills
    assert pd.isna(out["sii_3d"].iloc[0])
    # window mean starting at index 3 uses shifted values 0,1,2
    assert np.isclose(out["sii_3d"].iloc[3], np.mean([0, 1, 2]))


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

def test_compute_stats_perfect_line():
    x = np.arange(1, 21, dtype=float)
    y = 0.5 * x
    stats = mlib.compute_stats(y, x)
    assert np.isclose(stats["ols_slope_per_1nT"], 0.5)
    assert stats["ols_r_squared"] > 0.99
    assert stats["spearman_rho"] > 0.99


def test_empirical_p_plus_one():
    # Observed rho larger than all 10 nulls -> p = 1/11
    nulls = np.random.randn(10) * 0.1
    p = mlib.empirical_p(10.0, nulls)
    assert p == 1.0 / 11.0


# ═══════════════════════════════════════════════════════════════════════
# Functional controls
# ═══════════════════════════════════════════════════════════════════════

def test_functional_control_thresholds():
    cells = pd.DataFrame({
        "lat_id": [1, 2, 3, 4],
        "lon_id": [1, 2, 3, 4],
        "n_lai": [20, 20, 20, 20],
        "median_lai": [0.05, 0.05, 2.0, 2.0],
        "q10_lai": [0.0, 0.0, 0.5, 0.5],
        "q90_lai": [0.1, 0.5, 2.5, 2.5],
    })
    out = mlib.assign_functional_controls(cells)
    assert out.loc[0, "is_strict_low_lai"] == True   # median <=0.1, q90 <=0.3
    assert out.loc[1, "is_strict_low_lai"] == False  # q90 >0.3
    assert out.loc[2, "is_vegetated"] == True
    assert out.loc[3, "is_vegetated"] == True


# ═══════════════════════════════════════════════════════════════════════
# Harmonic detrending (synthetic)
# ═══════════════════════════════════════════════════════════════════════

def test_harmonic_detrending_no_leakage():
    # Four years of synthetic seasonal data for one cell
    dates = []
    y_true = []
    years = [2020, 2021, 2022, 2023]
    for yr in years:
        for doy in range(1, 366):
            dates.append(pd.Timestamp(f"{yr}-{1:02d}-{1:02d}") + pd.Timedelta(days=doy - 1))
            y_true.append(1.0 + 0.01 * yr + 0.5 * np.sin(2 * np.pi * doy / 365.25))
    df = pd.DataFrame({
        "date": dates,
        "year": [d.year for d in dates],
        "doy": [d.dayofyear for d in dates],
        "lat_id": 1,
        "lon_id": 1,
        "sif_771nm": y_true,
    })
    # Import harmonic fitting function
    import importlib.util
    spec = importlib.util.spec_from_file_location("harmonic", str(SCRIPT_DIR / "04_harmonic.py"))
    harmonic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harmonic)
    out = harmonic.fit_cell_harmonic(df)
    # All fits should pass and residuals should be near zero
    assert out["pass_flag"].all()
    assert np.allclose(out["residual"].dropna(), 0.0, atol=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Cyclic spline (synthetic)
# ═══════════════════════════════════════════════════════════════════════

def test_cyclic_spline_train_test_basis():
    # Synthetic data using the same climatological phase convention as the spline.
    # A cubic regression spline with df=8 approximates, but does not interpolate,
    # a sine wave, so we check that residuals are small and R^2 is near 1.
    from _Common import mmdd_to_clim_index
    dates = []
    y_true = []
    years = list(range(2015, 2025))
    for yr in years:
        for doy in range(1, 367):
            d = pd.Timestamp(f"{yr}-{1:02d}-{1:02d}") + pd.Timedelta(days=doy - 1)
            if d.month == 2 and d.day == 29:
                continue
            clim = mmdd_to_clim_index(pd.Series([doy], index=[0]), pd.Series([yr], index=[0])).iloc[0]
            phase = (clim - 1) / 365.0
            dates.append(d)
            y_true.append(1.0 + 0.01 * (yr + doy / 365.25) + 0.5 * np.sin(2 * np.pi * phase))
    df = pd.DataFrame({
        "date": dates,
        "year": [d.year for d in dates],
        "doy": [d.dayofyear for d in dates],
        "lat_id": 1,
        "lon_id": 1,
        "sif_771nm": y_true,
    })
    spec = importlib.util.spec_from_file_location("cyclic_spline", str(SCRIPT_DIR / "05_cyclic_spline.py"))
    cyc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyc)
    out = cyc.fit_cell_cyclic_spline(df)
    ok = out[out["fit_status"] == "ok"]
    assert len(ok) > 0
    assert ok["r_squared"].min() > 0.99
    assert ok["residual"].abs().max() < 0.02


def test_cyclic_spline_fixed_domain_and_boundary():
    """Fixed phase domain [0,1) must give continuous predictions across year boundary."""
    from _Common import mmdd_to_clim_index
    dates = []
    y_true = []
    years = list(range(2015, 2025))
    for yr in years:
        for doy in range(1, 367):
            d = pd.Timestamp(f"{yr}-{1:02d}-{1:02d}") + pd.Timedelta(days=doy - 1)
            if d.month == 2 and d.day == 29:
                continue
            clim = mmdd_to_clim_index(pd.Series([doy], index=[0]), pd.Series([yr], index=[0])).iloc[0]
            phase = (clim - 1) / 365.0
            dates.append(d)
            y_true.append(1.0 + 0.01 * (yr + doy / 365.25) + 0.5 * np.sin(2 * np.pi * phase))
    df = pd.DataFrame({
        "date": dates,
        "year": [d.year for d in dates],
        "doy": [d.dayofyear for d in dates],
        "lat_id": 1,
        "lon_id": 1,
        "sif_771nm": y_true,
    })
    spec = importlib.util.spec_from_file_location("cyclic_spline", str(SCRIPT_DIR / "05_cyclic_spline.py"))
    cyc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyc)
    out = cyc.fit_cell_cyclic_spline(df)
    ok = out[out["fit_status"] == "ok"]
    assert len(ok) > 0
    dec31 = ok[(ok["doy"] == 365) & (ok["year"] == 2023)]["expected_sif"].mean()
    jan1 = ok[(ok["doy"] == 1) & (ok["year"] == 2024)]["expected_sif"].mean()
    assert pd.notna(dec31) and pd.notna(jan1)
    assert abs(dec31 - jan1) < 0.05


def test_geographic_controls_only_sahara():
    from magneto_lib import load_geographic_regions
    regions = load_geographic_regions()
    assert "Sahara" in regions
    # Fictitious global controls removed
    assert "Desert_Barren" not in regions
    assert "Reference_Vegetated" not in regions


def test_matched_metadata_validation_catches_mismatch(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("matched", str(SCRIPT_DIR / "07_matched.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect any mismatch diagnostic to a temporary directory.
    mod.RESULTS = tmp_path
    # Suffix pairs from a merge with mismatched temperature labels should raise
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-07-01"]),
        "lat_id": [1], "lon_id": [1], "year": [2020],
        "temp_bin_label_x": ["Cold"],
        "temp_bin_label_y": ["Hot"],
    })
    with pytest.raises(ValueError, match="metadata mismatch"):
        mod._validate_metadata_equality(df)
    # The diagnostic file must be written to the temporary path, not to results/.
    assert (tmp_path / "matched_metadata_mismatches.txt").exists()


def test_pytest_does_not_create_production_mismatch_artifact():
    """Tests must not leave matched_metadata_mismatches.txt in production results."""
    assert not (Path("results") / "matched_metadata_mismatches.txt").exists()


@pytest.mark.skipif(not (Path("results") / "output_manifest.json").exists(), reason="output manifest not present")
def test_output_manifest_hashes_match_files():
    """SHA256 entries in the final output manifest must match actual files."""
    import json
    from _Common import full_sha256
    manifest = json.loads((Path("results") / "output_manifest.json").read_text())
    for item in manifest:
        p = Path(item["path"])
        if not p.exists():
            pytest.fail(f"Manifest path missing: {p}")
        actual = full_sha256(p)
        assert actual == item["sha256"], f"Hash mismatch for {p}"


@pytest.mark.parametrize("stage,outputs", [
    ("fixed_window", ["results/fixed_window_results.csv"]),
    ("surrogates", ["results/surrogate_summary.csv", "results/surrogate_null_distributions.parquet"]),
    ("effects", ["results/effect_sizes.csv"]),
])
def test_stage_manifest_hashes_match_files(stage, outputs):
    """Stage-manifest output hashes must match actual file hashes and the final manifest."""
    import json
    from _Common import full_sha256, PROJECT_ROOT
    manifest_path = Path("results") / "manifests" / f"{stage}.json"
    if not manifest_path.exists():
        pytest.skip(f"Stage manifest not found: {manifest_path}")
    stage_m = json.loads(manifest_path.read_text())
    out_hashes = stage_m.get("output_hashes", {})
    final_m = json.loads((Path("results") / "output_manifest.json").read_text()) if (Path("results") / "output_manifest.json").exists() else []
    final_by_path = {item["path"]: item["sha256"] for item in final_m}
    for op in outputs:
        p = PROJECT_ROOT / op
        if not p.exists():
            continue
        actual = full_sha256(p)
        assert out_hashes.get(op) == actual, f"Stage manifest hash mismatch for {op}"
        assert final_by_path.get(op) == actual, f"Final manifest hash mismatch for {op}"


# ═══════════════════════════════════════════════════════════════════════
# Atomic writes / integrity
# ═══════════════════════════════════════════════════════════════════════

def test_atomic_write_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        data = {"key": "value", "n": 42}
        atomic_write(data, tmp_path)
        assert tmp_path.exists()
        assert json.loads(tmp_path.read_text()) == data
    finally:
        tmp_path.unlink(missing_ok=True)


def test_full_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        tmp_path = Path(f.name)
    try:
        sha = full_sha256(tmp_path)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.slow
def test_all_scripts_parse():
    script_dir = Path(__file__).resolve().parent.parent
    for py_file in sorted(script_dir.glob("[0-9][0-9]_*.py")):
        with open(py_file, encoding="utf-8") as f:
            compile(f.read(), str(py_file), "exec")


# CLI parameter passing test: ensure surrogate script parses --n-surrogates
def test_surrogate_script_cli():
    spec = importlib.util.spec_from_file_location("surrogates", str(SCRIPT_DIR / "09_surrogates.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Default should be full N because no CLI args
    args = mod.parse_args([])
    assert args.n_surrogates is None


# ═══════════════════════════════════════════════════════════════════════
# Supplementary outcome provenance guards
# ═══════════════════════════════════════════════════════════════════════

def test_supplementary_sif_740_is_provider_derived():
    """SIF 740 nm must be classified as provider-derived, not direct."""
    spec = importlib.util.spec_from_file_location("supp", str(SCRIPT_DIR / "run_supplementary_outcomes.py"))
    supp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supp)
    assert supp.SUPPLEMENTARY_OUTCOMES["sif_740nm"]["type"] == "provider_derived"
    assert "provider" in supp.SUPPLEMENTARY_OUTCOMES["sif_740nm"]["type"].lower()
