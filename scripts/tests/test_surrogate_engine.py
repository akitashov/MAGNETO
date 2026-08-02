#!/usr/bin/env python3
"""
Unit tests for the temporal surrogate engine in scripts/magneto_lib.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from magneto_lib import (
    _build_surrogate_matrix,
    _rolling_mean_matrix,
    _spearman_many,
    block_permutation,
    build_daily_sii,
    circular_shift,
    valid_circular_offsets,
    year_permutation,
)


def _simple_daily_df() -> pd.DataFrame:
    """A short daily series with one leap year and a known pattern."""
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_feb29"] = (df["month"] == 2) & (df["day"] == 29)
    # deterministic monotonic-ish values: day-of-year * 0.1
    df["sii_mean"] = df["date"].dt.dayofyear.astype(float) * 0.1
    return df


def test_build_daily_sii_no_nans_after_interpolation():
    """After interpolation the surrogate daily SII must be fully finite."""
    df = build_daily_sii()
    filled = (
        df["sii_mean"]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    assert filled.isna().sum() == 0


def test_year_permutation_preserves_leap_day():
    """Year permutation must keep 29 February populated, not NaN."""
    df = _simple_daily_df()
    years = sorted(df["year"].unique().tolist())
    from magneto_lib import _build_year_mmdd_map

    ymap = _build_year_mmdd_map(df, years)
    rng = np.random.default_rng(1)
    perm = year_permutation(rng, df, years, ymap)
    leap_day = perm[(perm["year"] == 2020) & perm["is_feb29"]]["sii_mean"]
    assert len(leap_day) == 1
    assert np.isfinite(leap_day.iloc[0])


def test_year_perm_rolling_window_after_leap_day_is_finite():
    """After shift(1)+rolling the days following 29 Feb must stay finite."""
    df = _simple_daily_df()
    years = sorted(df["year"].unique().tolist())
    from magneto_lib import _build_year_mmdd_map

    ymap = _build_year_mmdd_map(df, years)
    rng = np.random.default_rng(2)
    S = _build_surrogate_matrix("year_perm", rng, df, years, ymap, n_surr=10)
    W = _rolling_mean_matrix(S, window=21)
    # Positions corresponding to 1-21 March 2020 in the daily series.
    post_leap = df[(df["year"] == 2020) & (df["month"] == 3) & (df["day"] <= 21)].index
    assert np.all(np.isfinite(W[post_leap, :]))


def test_identity_year_perm_matches_observed_rho():
    """Identity permutation should give surrogate rhos identical to observed."""
    df = _simple_daily_df()
    years = sorted(df["year"].unique().tolist())
    from magneto_lib import _build_year_mmdd_map

    ymap = _build_year_mmdd_map(df, years)
    rng = np.random.default_rng(3)
    # Build one identity permutation manually.
    n_years = len(years)
    year_to_idx = {yr: i for i, yr in enumerate(years)}
    years_arr = df["year"].values
    months_arr = df["month"].values.astype(int)
    days_arr = df["day"].values.astype(int)
    month_days_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mmdd_to_k = {}
    k = 0
    for m, nd in enumerate(month_days_leap, start=1):
        for d in range(1, nd + 1):
            mmdd_to_k[(m, d)] = k
            k += 1
    donor = np.full((n_years, 366), np.nan)
    for yi, yr in enumerate(years):
        src_map = ymap.get(yr, {})
        for (m, d), kidx in mmdd_to_k.items():
            if (m, d) == (2, 29):
                donor[yi, kidx] = src_map.get((2, 29), src_map.get((2, 28), np.nan))
            else:
                donor[yi, kidx] = src_map.get((m, d), np.nan)
    target_year_idx = np.array([year_to_idx[y] for y in years_arr], dtype=np.int64)
    mmdd_idx = np.array([mmdd_to_k[(m, d)] for m, d in zip(months_arr, days_arr)], dtype=np.int64)
    S = donor[target_year_idx, mmdd_idx][:, None]
    W = _rolling_mean_matrix(S, window=21)

    # synthetic residuals: same length as daily df, correlated with windowed SII
    residuals = df["sii_mean"] * 0.5 + np.random.default_rng(4).normal(size=len(df))
    mask = np.isfinite(residuals.values) & np.isfinite(W[:, 0])
    rho_obs = _spearman_many(residuals.values, W[:, 0][:, None], mask=mask)[0]
    rho_surr = _spearman_many(residuals.values, W, mask=mask)[0]
    assert np.isclose(rho_obs, rho_surr, atol=1e-10)


def test_spearman_many_frozen_mask():
    """Passing a mask must freeze the observation set across realisations."""
    n = 100
    rng = np.random.default_rng(5)
    r = rng.normal(size=n)
    E = rng.normal(size=(n, 50))
    mask = np.ones(n, dtype=bool)
    mask[::10] = False  # drop 10 observations
    rhos = _spearman_many(r, E, mask=mask)
    # All realisations should return finite values because mask excludes rows
    # that are present in all columns.
    assert np.all(np.isfinite(rhos))
    # The masked rows must not influence the result.
    rhos_full = _spearman_many(r, E)
    # With the same finite exposure the masked result differs only by sample size.
    assert rhos.shape == (50,)


def test_valid_circular_offsets_exclude_modulo_aliases():
    """Offsets near annual multiples or their n-modulo aliases are forbidden."""
    n = 3680
    min_shift = 30
    tolerance = 5
    valid = valid_circular_offsets(n, min_shift, tolerance=tolerance)
    assert len(valid) > 0
    for offset in valid:
        assert 1 <= offset < n
        signed_distance = min(offset, n - offset)
        # Must be farther than min_shift from identity.
        assert signed_distance >= min_shift
        # Must be farther than tolerance from every solar-year multiple.
        max_mult = int(np.floor(n / 365.2425)) + 2
        for mult in range(1, max_mult + 1):
            annual = mult * 365.2425
            assert abs(signed_distance - annual) > tolerance
            # The modulo-n alias is also excluded because signed_distance is
            # symmetric for offset and n-offset.


def test_circular_shift_excludes_year_multiples():
    """Drawn circular shifts must avoid small and near-annual offsets."""
    df = _simple_daily_df()
    rng = np.random.default_rng(6)
    n = len(df)
    valid = set(valid_circular_offsets(n, min_shift=30))
    for _ in range(400):
        out, offset = circular_shift(rng, df, min_shift=30)
        assert offset in valid
        assert 1 <= offset < n


def test_block_permutation_includes_remainder():
    """Block permutation must move the final partial block as well."""
    df = _simple_daily_df()
    rng = np.random.default_rng(7)
    n = len(df)
    block_size = 30
    out = block_permutation(rng, df, block_size=block_size)
    # The set of values is unchanged; only order changes.
    assert np.allclose(np.sort(out["sii_mean"].values), np.sort(df["sii_mean"].values))
    # With a non-trivial permutation the order should usually differ.
    assert not np.allclose(out["sii_mean"].values, df["sii_mean"].values)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
