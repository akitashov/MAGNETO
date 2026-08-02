"""
MAGNETO analysis library.

Shared functions for SII exposure, fixed-window statistics, temporal
surrogates, and control definitions. Used by stage scripts and tests.
"""
from __future__ import annotations
import calendar, os, sys, hashlib, numpy as np, pandas as pd
from pathlib import Path
from typing import Iterable
from scipy.stats import spearmanr, linregress, rankdata
from _Common import *

# Optional GPU acceleration for large surrogate correlation batches.
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False


def _cupy_rankdata_average(x: "cp.ndarray") -> "cp.ndarray":
    """Average-rank implementation on GPU using sorted unique values.

    Equivalent to scipy.stats.rankdata(x, method='average') for finite input.
    """
    unique_vals, inverse, counts = cp.unique(x, return_inverse=True, return_counts=True)
    cum_counts = cp.cumsum(counts)
    last_ranks = cum_counts
    first_ranks = last_ranks - counts + 1
    avg_ranks = (first_ranks + last_ranks) / 2.0
    return avg_ranks[inverse]


def _spearman_many_gpu(r: np.ndarray, E: np.ndarray,
                       mask: np.ndarray | None = None,
                       chunk_size: int = 25) -> np.ndarray:
    """GPU-accelerated Spearman correlation for many exposure columns."""
    # Allow environment override to cap per-chunk GPU memory on memory-constrained hosts.
    env_chunk = os.environ.get("MAGNETO_GPU_CHUNK_SIZE")
    if env_chunk:
        try:
            chunk_size = max(1, int(env_chunk))
        except ValueError:
            pass
    N = E.shape[1]
    rhos = np.full(N, np.nan, dtype=np.float64)
    if mask is None:
        base_mask = np.isfinite(r) & np.all(np.isfinite(E), axis=1)
    else:
        base_mask = mask
    n_base = int(base_mask.sum())
    if n_base < 10:
        return rhos

    r_cp = cp.asarray(r[base_mask])
    rx = _cupy_rankdata_average(r_cp)
    rx -= rx.mean()
    sx = cp.sqrt(cp.sum(rx * rx))
    if float(sx) == 0:
        return rhos

    E_masked = E[base_mask, :]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        E_chunk = cp.asarray(E_masked[:, start:end])
        Ry = cp.empty_like(E_chunk, dtype=cp.float64)
        for i in range(end - start):
            Ry[:, i] = _cupy_rankdata_average(E_chunk[:, i])
        Ry -= Ry.mean(axis=0)
        Sy = cp.sqrt(cp.sum(Ry * Ry, axis=0))
        rho_chunk = cp.sum(rx[:, None] * Ry, axis=0) / (sx * Sy)
        rhos[start:end] = cp.asnumpy(rho_chunk)
    return rhos


# ═══════════════════════════════════════════════════════════════════════
# SII exposure
# ═══════════════════════════════════════════════════════════════════════

def build_daily_sii() -> pd.DataFrame:
    """Build a continuous daily SII series from OMNI."""
    omni = pd.read_feather(FILE_OMNI, columns=["date", SII_RAW_COL])
    omni["date"] = pd.to_datetime(omni["date"])
    omni = omni.sort_values("date").reset_index(drop=True)
    full_dates = pd.date_range(omni["date"].min(), omni["date"].max(), freq="D")
    full_df = pd.DataFrame({"date": full_dates})
    full_df = full_df.merge(omni, on="date", how="left")
    full_df["year"] = full_df["date"].dt.year
    full_df["month"] = full_df["date"].dt.month
    full_df["day"] = full_df["date"].dt.day
    full_df["is_feb29"] = (full_df["month"] == 2) & (full_df["day"] == 29)
    return full_df


def compute_sii_windows(daily_df: pd.DataFrame, sii_col: str = SII_RAW_COL,
                        windows: list[int] = None) -> pd.DataFrame:
    """shift(1) then rolling.mean(w) for each window."""
    if windows is None:
        windows = SII_WINDOWS
    df = daily_df[["date", sii_col]].copy()
    shifted = df[sii_col].shift(1) if SII_SHIFT_BEFORE_ROLL else df[sii_col]
    out = df[["date"]].copy()
    min_p = SII_MIN_PERIODS
    for w in windows:
        mp = w if min_p is None else min_p
        out[f"sii_{w}d"] = shifted.rolling(window=w, min_periods=mp).mean()
    return out


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def compute_stats(residuals: np.ndarray, exposure: np.ndarray) -> dict:
    """Spearman rho + OLS statistics for a valid pair of arrays."""
    mask = _finite_mask(residuals, exposure)
    res = residuals[mask]
    exp = exposure[mask]
    n = int(mask.sum())

    rho, p_spearman = np.nan, np.nan
    if n >= 10:
        try:
            rho, p_spearman = spearmanr(res, exp)
        except Exception:
            pass

    slope = intercept = r_squared = p_ols = std_err = np.nan
    if n >= 10:
        try:
            slope, intercept, r_value, p_ols, std_err = linregress(exp, res)
            r_squared = r_value ** 2
        except Exception:
            pass

    sii_sd = float(np.std(exp, ddof=1)) if n > 1 else np.nan
    resid_sd = float(np.std(res, ddof=1)) if n > 1 else np.nan
    std_effect = float(slope * sii_sd) if np.isfinite(slope) and np.isfinite(sii_sd) else np.nan
    std_effect_rel = float(std_effect / resid_sd) if np.isfinite(std_effect) and resid_sd > 0 else np.nan

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "ols_slope_per_1nT": float(slope),
        "ols_slope_per_100nT": float(slope * 100.0) if np.isfinite(slope) else np.nan,
        "ols_intercept": float(intercept),
        "ols_r_squared": float(r_squared),
        "ols_p_value": float(p_ols),
        "ols_std_err": float(std_err),
        "sii_sd": sii_sd,
        "residual_sd": resid_sd,
        "std_effect_per_1sd_sii": std_effect,
        "std_effect_rel_residual_sd": std_effect_rel,
        "n_obs": n,
    }


# ═══════════════════════════════════════════════════════════════════════
# Surrogate transformations
# ═══════════════════════════════════════════════════════════════════════

def _build_year_mmdd_map(daily_df: pd.DataFrame, years: list[int]) -> dict:
    """Build a per-year month/day -> SII map, preserving 29 February for leap years."""
    sub = daily_df[["year", "month", "day", SII_RAW_COL]].copy()
    ymap = {}
    for yr, grp in sub.groupby("year"):
        src = dict(zip(
            zip(grp["month"].astype(int), grp["day"].astype(int)),
            grp[SII_RAW_COL]
        ))
        # For non-leap years 29 February does not exist; impute from 28 February
        # so leap-day targets always receive a finite donor value.
        if (2, 29) not in src:
            src[(2, 29)] = src.get((2, 28), np.nan)
        ymap[yr] = src
    for yr in years:
        ymap.setdefault(yr, {})
    return ymap


def year_permutation(rng: np.random.Generator, daily_df: pd.DataFrame,
                     years: list[int], year_mmdd_map: dict) -> pd.DataFrame:
    """Year permutation that keeps 29 February populated via its donor value."""
    valid_years = sorted(year_mmdd_map.keys())
    perm_years = rng.permutation(valid_years)
    yr_src = dict(zip(valid_years, perm_years))
    out = daily_df.copy()
    new_sii = out[SII_RAW_COL].copy()
    for target_yr in valid_years:
        src_yr = yr_src[target_yr]
        src_map = year_mmdd_map.get(src_yr, {})
        if not src_map:
            continue
        tmask = out["year"] == target_yr
        idxs = out.index[tmask]
        months = out.loc[idxs, "month"].astype(int).values
        days = out.loc[idxs, "day"].astype(int).values
        vals = np.array([
            src_map.get((m, d), np.nan)
            for m, d in zip(months, days)
        ], dtype=new_sii.dtype)
        new_sii.iloc[idxs] = vals
    out[SII_RAW_COL] = new_sii
    return out


def valid_circular_offsets(
    n: int,
    min_shift: int,
    annual_period: float = 365.2425,
    tolerance: int = 5,
) -> list[int]:
    """Return offsets in [1, n-1] whose signed distance from every annual
    multiple exceeds tolerance and whose modulo-n alias does the same.

    Because np.roll treats shift and shift-n as identical, we must forbid both
    an offset and its complement n-offset whenever either is near a solar-year
    multiple.
    """
    forbidden: set[int] = set()

    # Small absolute shifts in either direction (modulo n).
    for d in range(-min_shift + 1, min_shift):
        forbidden.add(d % n)

    # All near-integer solar-year multiples and their modulo-n mirrors.
    max_mult = int(np.floor(n / annual_period)) + 1
    for mult in range(1, max_mult + 1):
        centre = int(round(mult * annual_period))
        for delta in range(-tolerance, tolerance + 1):
            offset = (centre + delta) % n
            forbidden.add(offset)
            forbidden.add((-offset) % n)

    return [offset for offset in range(1, n) if offset not in forbidden]


def circular_shift(rng: np.random.Generator, daily_df: pd.DataFrame,
                   min_shift: int = SURROGATE_CIRC_SHIFT_MIN) -> tuple[pd.DataFrame, int]:
    """Return (shifted_df, chosen_offset).  chosen_offset is in [1, n-1]."""
    n = len(daily_df)
    valid = valid_circular_offsets(n, min_shift)
    offset = int(rng.choice(valid)) if valid else min_shift
    out = daily_df.copy()
    out[SII_RAW_COL] = np.roll(daily_df[SII_RAW_COL].values, offset)
    return out, offset


def block_permutation(rng: np.random.Generator, daily_df: pd.DataFrame,
                      block_size: int = SURROGATE_BLOCK_SIZE) -> pd.DataFrame:
    n = len(daily_df)
    sii_vals = daily_df[SII_RAW_COL].values.copy()
    n_full = n // block_size
    remainder = n % block_size
    blocks = [list(range(i * block_size, (i + 1) * block_size)) for i in range(n_full)]
    if remainder > 0:
        blocks.append(list(range(n_full * block_size, n)))
    perm_order = rng.permutation(len(blocks))
    new_sii = np.empty(n, dtype=sii_vals.dtype)
    pos = 0
    for bi in perm_order:
        blk = blocks[bi]
        new_sii[pos:pos + len(blk)] = sii_vals[blk]
        pos += len(blk)
    out = daily_df.copy()
    out[SII_RAW_COL] = new_sii
    return out


# ═══════════════════════════════════════════════════════════════════════
# Empirical p-value
# ═══════════════════════════════════════════════════════════════════════

def empirical_p(rho_obs: float, null_rhos: np.ndarray) -> float:
    """Two-sided empirical p-value with +1 correction."""
    valid = null_rhos[~np.isnan(null_rhos)]
    n = len(valid)
    if n == 0:
        return 1.0
    count_ge = np.sum(np.abs(valid) >= np.abs(rho_obs))
    return (1.0 + count_ge) / (n + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Strata builders for fixed-window and surrogate analyses
# ═══════════════════════════════════════════════════════════════════════

def build_strata(df: pd.DataFrame, sample_type: str, method: str,
                 residual_col: str, windows: list[int] = None,
                 group_cols: list[str] = None, min_n: int = 10) -> list[dict]:
    """Build analysis strata from a sample DataFrame.

    group_cols is a list of column names to stratify by.  If None, a single
    pooled stratum is returned.
    """
    if windows is None:
        windows = SII_WINDOWS
    if group_cols is None or not group_cols:
        groups = [(None, df)]
    else:
        # drop rows missing any grouping variable
        df = df.dropna(subset=group_cols)
        if len(df) == 0:
            return []
        groups = df.groupby(group_cols, sort=True)

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


def _observation_key_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of sorted observation keys in a stratum."""
    cols = [c for c in ["date", "lat_id", "lon_id"] if c in df.columns]
    if not cols:
        return ""
    sub = df[cols].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
    sub = sub.sort_values(["date", "lat_id", "lon_id"]).reset_index(drop=True)
    key_str = "\n".join("_".join(str(v) for v in r) for r in sub.values)
    return hashlib.sha256(key_str.encode()).hexdigest()


def run_fixed_window(strata: list[dict]) -> list[dict]:
    """Compute fixed-window statistics for a list of strata."""
    rows = []
    for st in strata:
        df = st["data"]
        residual_col = st["residual_col"]
        sii_col = st["sii_col"]
        needed = [residual_col, sii_col, "lat_id", "lon_id", "date"]
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
            "obs_hash": _observation_key_hash(valid),
        }
        # merge explicit grouping columns
        for k in st:
            if k not in {"sample_type", "method", "residual_col", "sii_col", "sii_window_days", "data"}:
                row[k] = st[k]
        row.update(stats)
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Surrogate engine
# ═══════════════════════════════════════════════════════════════════════

def _build_surrogate_matrix(mode: str, rng: np.random.Generator,
                            daily_sii: pd.DataFrame, years: list,
                            year_mmdd_map: dict, n_surr: int) -> np.ndarray:
    """Return (n_days, n_surr) array of surrogate daily SII values."""
    n = len(daily_sii)
    S = np.empty((n, n_surr), dtype=np.float64)
    sii_raw = daily_sii[SII_RAW_COL].values
    years_arr = daily_sii["year"].values
    months_arr = daily_sii["month"].values.astype(int)
    days_arr = daily_sii["day"].values.astype(int)
    is_feb29 = daily_sii["is_feb29"].values

    if mode == "year_perm":
        valid_years = sorted(year_mmdd_map.keys())
        n_years = len(valid_years)
        assert n_years > 0, "year permutation requires at least one source year"
        year_to_idx = {yr: i for i, yr in enumerate(valid_years)}

        # Fixed 1-366 climatological month/day index (29 February included).
        month_days_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        mmdd_to_k = {}
        k = 0
        for m, nd in enumerate(month_days_leap, start=1):
            for d in range(1, nd + 1):
                mmdd_to_k[(m, d)] = k
                k += 1

        # Donor array: (year, 366) SII values.  For non-leap donors 29 February
        # is imputed from 28 February so that leap-day target dates receive a
        # finite value and rolling windows after 29 February stay populated.
        donor = np.full((n_years, 366), np.nan, dtype=np.float64)
        for yi, yr in enumerate(valid_years):
            src_map = year_mmdd_map.get(yr, {})
            for (m, d), kidx in mmdd_to_k.items():
                if (m, d) == (2, 29):
                    donor[yi, kidx] = src_map.get((2, 29), src_map.get((2, 28), np.nan))
                else:
                    donor[yi, kidx] = src_map.get((m, d), np.nan)

        target_year_idx = np.array([year_to_idx[y] for y in years_arr], dtype=np.int64)
        mmdd_idx = np.array(
            [mmdd_to_k[(m, d)] for m, d in zip(months_arr, days_arr)],
            dtype=np.int64,
        )

        # Separate leap and non-leap source years.  A leap-year target day (in
        # particular 29 February) must receive a value from a leap donor year;
        # a non-leap target day must receive a value from a non-leap donor year.
        # Fixed points (year mapped to itself) are avoided where possible.
        leap_years = np.array([yr for yr in valid_years if calendar.isleap(yr)], dtype=np.int64)
        non_leap_years = np.array([yr for yr in valid_years if not calendar.isleap(yr)], dtype=np.int64)

        def _derangement(arr: np.ndarray, rng: np.random.Generator, max_trials: int = 1000) -> np.ndarray:
            if len(arr) <= 1:
                return arr.copy()
            for _ in range(max_trials):
                perm = rng.permutation(arr)
                if not np.any(perm == arr):
                    return perm
            return rng.permutation(arr)

        leap_src_to_perm = {yr: i for i, yr in enumerate(leap_years)}
        non_leap_src_to_perm = {yr: i for i, yr in enumerate(non_leap_years)}
        target_is_leap = np.array([calendar.isleap(int(y)) for y in years_arr], dtype=bool)

        for i in range(n_surr):
            leap_perm = _derangement(leap_years, rng)
            non_leap_perm = _derangement(non_leap_years, rng)
            src_years = np.empty(n, dtype=np.int64)
            src_years[target_is_leap] = leap_perm[np.array(
                [leap_src_to_perm[int(y)] for y in years_arr[target_is_leap]]
            )]
            src_years[~target_is_leap] = non_leap_perm[np.array(
                [non_leap_src_to_perm[int(y)] for y in years_arr[~target_is_leap]]
            )]
            src_yi = np.array([year_to_idx[int(y)] for y in src_years], dtype=np.int64)
            S[:, i] = donor[src_yi, mmdd_idx]
    elif mode == "circ_shift":
        valid = valid_circular_offsets(
            n=n,
            min_shift=SURROGATE_CIRC_SHIFT_MIN,
            annual_period=365.2425,
            tolerance=5,
        )
        if not valid:
            raise RuntimeError("No valid circular offsets remain after exclusions.")
        for i in range(n_surr):
            offset = int(rng.choice(valid))
            S[:, i] = np.roll(sii_raw, offset)
    elif mode == "block_perm":
        block_size = SURROGATE_BLOCK_SIZE
        n_full = n // block_size
        remainder = n % block_size
        blocks = [list(range(j * block_size, (j + 1) * block_size)) for j in range(n_full)]
        if remainder:
            blocks.append(list(range(n_full * block_size, n)))
        for i in range(n_surr):
            perm_order = rng.permutation(len(blocks))
            new_sii = np.empty(n, dtype=np.float64)
            pos = 0
            for bi in perm_order:
                blk = blocks[bi]
                new_sii[pos:pos + len(blk)] = sii_raw[blk]
                pos += len(blk)
            S[:, i] = new_sii
    else:
        raise ValueError(f"Unknown surrogate mode: {mode}")
    return S


def _rolling_mean_matrix(S: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """shift(1) then rolling mean along axis 0 for each surrogate column.

    The leading shift aligns surrogate windows with the observed SII windows
    produced by compute_sii_windows().
    """
    if min_periods is None:
        min_periods = window
    df = pd.DataFrame(S)
    shifted = df.shift(1) if SII_SHIFT_BEFORE_ROLL else df
    return shifted.rolling(window=window, min_periods=min_periods).mean().values


def _spearman_many(r: np.ndarray, E: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Vector-of-Spearman-rhos for residual vector r vs each column of E.

    Uses the GPU for large matrices when CuPy is available; falls back to the
    CPU implementation otherwise or for small inputs.

    Parameters
    ----------
    r : residual vector (n_obs,)
    E : exposure matrix (n_obs, n_surr)
    mask : optional bool vector (n_obs,).  If provided, only these observations
        are used for every surrogate realisation, guaranteeing a frozen
        observation set.  The caller must ensure the mask equals the observed-
        valid set (or a stricter whitelist).
    """
    use_mask = mask if mask is not None else (np.isfinite(r) & np.all(np.isfinite(E), axis=1))
    n_base = int(use_mask.sum())
    if n_base < 10:
        return np.full(E.shape[1], np.nan, dtype=np.float64)

    # GPU path: large observation count and many surrogate columns.
    if _CUPY_AVAILABLE and n_base >= 50000 and E.shape[1] >= 10:
        try:
            return _spearman_many_gpu(r, E, mask=mask)
        except Exception as exc:
            print(f"[WARN] GPU surrogate correlation failed ({exc}); falling back to CPU.")

    # CPU fallback (identical statistic).
    N = E.shape[1]
    rhos = np.full(N, np.nan, dtype=np.float64)
    x = r[use_mask]
    rx = rankdata(x, method="average")
    rx -= rx.mean()
    sx = np.sqrt(np.sum(rx * rx))
    if sx == 0:
        return rhos
    for i in range(N):
        y = E[use_mask, i]
        ry = rankdata(y, method="average")
        ry -= ry.mean()
        sy = np.sqrt(np.sum(ry * ry))
        if sy > 0:
            rhos[i] = np.sum(rx * ry) / (sx * sy)
    return rhos


def run_surrogates(strata: list[dict], n_surr: int, seed: int,
                   modes: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run temporal surrogates for all strata and return summary + nulls."""
    if modes is None:
        modes = SURROGATE_MODES

    # Study period is defined by the observations actually entering the analysis.
    all_obs_dates = pd.concat([st["data"]["date"] for st in strata])
    all_obs_dates = pd.to_datetime(all_obs_dates)
    study_min = all_obs_dates.min()
    study_max = all_obs_dates.max()
    study_years = sorted(all_obs_dates.dt.year.unique().tolist())
    print(f"[INFO] run_surrogates: study period {study_min.date()} -> {study_max.date()}, "
          f"source years {study_years[0]}..{study_years[-1]}")

    daily_sii_full = build_daily_sii()

    # Log gaps inside the study era before any interpolation.  The observed SII
    # windows are left unchanged; only the null-transformed daily series is
    # imputed here.
    # Gaps must be checked on the full daily series used to build both observed
    # and surrogate exposure windows, i.e. from the lookback start through the
    # last observation date.
    lookback_days = max(SII_WINDOWS) if SII_WINDOWS else 28
    filter_min = study_min - pd.Timedelta(days=lookback_days)
    analysis_daily_mask = (
        (daily_sii_full["date"] >= filter_min) &
        (daily_sii_full["date"] <= study_max)
    )
    n_analysis_gaps = int(
        daily_sii_full.loc[analysis_daily_mask, SII_RAW_COL].isna().sum()
    )
    if n_analysis_gaps:
        print(f"[WARNING] {n_analysis_gaps} gap(s) in analysis-era daily SII before interpolation")
    else:
        print("[INFO] No gaps in analysis-era daily SII before interpolation")

    # Fill gaps for the surrogate series.
    daily_sii_full[SII_RAW_COL] = (
        daily_sii_full[SII_RAW_COL]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    # Donor years for year permutation are strictly the study-era years.  Using
    # the full OMNI history as a donor pool would let surrogate SII leak from
    # decades outside the study period.
    source_daily = daily_sii_full[daily_sii_full["year"].isin(study_years)].copy()
    years = study_years
    year_mmdd_map = _build_year_mmdd_map(source_daily, years)

    # Restrict the target daily series to the analysis dates plus the lookback
    # required for rolling windows.  Circular and block permutations operate on
    # this filtered series; year permutation uses it only to align target dates.
    daily_sii = daily_sii_full[
        (daily_sii_full["date"] >= filter_min) & (daily_sii_full["date"] <= study_max)
    ].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    summary_rows = []
    null_rows = []

    # Map each stratum's observation dates to positions in the daily series.
    date_to_idx = {d: i for i, d in enumerate(daily_sii["date"])}
    stratum_idx = []
    for st in strata:
        idxs = np.array([date_to_idx[d] for d in st["data"]["date"].values], dtype=np.int64)
        stratum_idx.append(idxs)

    # Process surrogates in column-chunks to bound memory use.
    chunk_size = 250
    total = len(strata) * len(modes) * n_surr
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Surrogates", unit="realization")

    # Unique windows across all strata so rolling means are computed once per
    # (surrogate mode, window) rather than once per stratum.
    windows = sorted(set(
        st.get("sii_window_days", int(st["sii_col"].split("_")[1].rstrip("d")))
        for st in strata
    ))

    for mode in modes:
        print(f"[INFO] Building surrogate daily SII for mode={mode} (N={n_surr}) …")
        S = _build_surrogate_matrix(mode, rng, daily_sii, years, year_mmdd_map, n_surr)
        print(f"[INFO] Pre-computing rolling means for windows {windows} …")
        W_cache = {wd: _rolling_mean_matrix(S, wd) for wd in windows}

        for st, idxs in zip(strata, stratum_idx):
            sample_type = st["sample_type"]
            method = st["method"]
            residual_col = st["residual_col"]
            sii_col = st["sii_col"]
            window_days = st.get("sii_window_days", int(sii_col.split("_")[1].rstrip("d")))
            df_data = st["data"]
            base_key = {k: v for k, v in st.items()
                        if k not in {"sample_type", "method", "residual_col", "sii_col", "sii_window_days", "data"}}

            # observed rho and frozen observation whitelist
            r_all = df_data[residual_col].values
            e_obs = df_data[sii_col].values
            obs_mask = np.isfinite(r_all) & np.isfinite(e_obs)
            n_obs_used = int(obs_mask.sum())
            obs_valid = df_data.loc[obs_mask, [residual_col, sii_col]]
            rho_obs = compute_stats(obs_valid[residual_col].values, obs_valid[sii_col].values)["spearman_rho"]
            # Hash only the observations that actually enter both observed and null statistics.
            obs_hash = _observation_key_hash(df_data.loc[obs_mask])

            W = W_cache[window_days]
            nulls = np.empty(n_surr, dtype=np.float64)
            nulls[:] = np.nan
            failed = 0
            pos = 0
            while pos < n_surr:
                cs = min(chunk_size, n_surr - pos)
                E_chunk = W[idxs, pos:pos + cs]
                rhos_chunk = _spearman_many(r_all, E_chunk, mask=obs_mask)
                nulls[pos:pos + cs] = rhos_chunk
                failed += int(np.isnan(rhos_chunk).sum())
                pbar.update(cs)
                pos += cs

            for i, rho_s in enumerate(nulls):
                null_rows.append({
                    "sample_type": sample_type,
                    "method": method,
                    "surrogate_mode": mode,
                    "window": sii_col,
                    "realization": i,
                    "residual_col": residual_col,
                    "rho": rho_s,
                    **base_key,
                })

            valid_n = nulls[~np.isnan(nulls)]
            p_val = empirical_p(rho_obs, nulls)
            summary_rows.append({
                "sample_type": sample_type,
                "method": method,
                "surrogate_mode": mode,
                "sii_window": sii_col,
                "rho_obs": rho_obs,
                "p_value": p_val,
                "n_requested": n_surr,
                "n_completed": len(valid_n),
                "n_failed": failed,
                "n_obs_used": n_obs_used,
                "null_mean": float(valid_n.mean()) if len(valid_n) else np.nan,
                "null_sd": float(valid_n.std(ddof=1)) if len(valid_n) > 1 else np.nan,
                "null_q025": float(np.percentile(valid_n, 2.5)) if len(valid_n) else np.nan,
                "null_q05": float(np.percentile(valid_n, 5.0)) if len(valid_n) else np.nan,
                "null_q50": float(np.percentile(valid_n, 50.0)) if len(valid_n) else np.nan,
                "null_q95": float(np.percentile(valid_n, 95.0)) if len(valid_n) else np.nan,
                "null_q975": float(np.percentile(valid_n, 97.5)) if len(valid_n) else np.nan,
                "seed": seed,
                "obs_hash": obs_hash,
                **base_key,
            })

            # Release cached CuPy memory after each stratum to prevent the pool
            # from growing monotonically across large strata on memory-constrained hosts.
            if _CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

        # Release cached CuPy memory between surrogate modes to reduce VRAM spikes.
        if _CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    pbar.close()
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


# ═══════════════════════════════════════════════════════════════════════
# Functional and geographic controls
# ═══════════════════════════════════════════════════════════════════════

def compute_lai_cell_stats(modis_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell LAI statistics from the full MODIS series."""
    df = modis_df[["lat_id", "lon_id", "latitude", "lai"]].copy()
    df = df[df["lai"].notna() & np.isfinite(df["lai"])]
    cells = df.groupby(["lat_id", "lon_id"], as_index=False).agg(
        n_lai=("lai", "count"),
        median_lai=("lai", "median"),
        mean_lai=("lai", "mean"),
        std_lai=("lai", "std"),
        min_lai=("lai", "min"),
        max_lai=("lai", "max"),
        q10_lai=("lai", lambda x: x.quantile(0.10)),
        q25_lai=("lai", lambda x: x.quantile(0.25)),
        q75_lai=("lai", lambda x: x.quantile(0.75)),
        q90_lai=("lai", lambda x: x.quantile(0.90)),
        mean_lat=("latitude", "mean"),
    )
    return cells


def assign_functional_controls(cells: pd.DataFrame) -> pd.DataFrame:
    """Add functional strict low-LAI / vegetated control flags."""
    cells = cells.copy()
    cells["is_strict_low_lai"] = (
        (cells["n_lai"] >= CONTROL_STRICT_LOW_LAI_MIN_LAI_OBS) &
        (cells["median_lai"] <= CONTROL_STRICT_LOW_LAI_MEDIAN_LAI_MAX) &
        (cells["q90_lai"] <= CONTROL_STRICT_LOW_LAI_Q90_LAI_MAX)
    )
    cells["is_vegetated"] = (
        (cells["n_lai"] >= CONTROL_Vegetated_MIN_LAI_OBS) &
        (cells["median_lai"] >= CONTROL_Vegetated_MEDIAN_LAI_MIN) &
        (cells["q10_lai"] >= CONTROL_Vegetated_Q10_LAI_MIN)
    )
    return cells


def load_geographic_regions() -> dict:
    """Load region definitions from config/regions.yaml."""
    regions_path = PROJECT_ROOT / CONFIG["controls"]["geographic"]["config_path"]
    with open(regions_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("regions", {})


def assign_geographic_controls(cells: pd.DataFrame, regions: dict = None) -> pd.DataFrame:
    """Add geographic control flags from config/regions.yaml."""
    if regions is None:
        regions = load_geographic_regions()
    cells = cells.copy()
    for name, spec in regions.items():
        lat_lo, lat_hi = spec["lat_range"]
        lon_lo, lon_hi = spec["lon_range"]
        if lon_lo <= lon_hi:
            mask = (
                (cells["mean_lat"] >= lat_lo) & (cells["mean_lat"] <= lat_hi) &
                (cells["mean_lon"] >= lon_lo) & (cells["mean_lon"] <= lon_hi)
            )
        else:
            # wrap-around region (e.g. lon_range [-180, 180])
            mask = (
                (cells["mean_lat"] >= lat_lo) & (cells["mean_lat"] <= lat_hi) &
                ((cells["mean_lon"] >= lon_lo) | (cells["mean_lon"] <= lon_hi))
            )
        # apply forbidden flags if present
        exclude = spec.get("exclude_flags", [])
        if exclude:
            # bit mask check
            for flag_name in exclude:
                bit = globals().get(flag_name)
                if bit is not None:
                    # cells do not have region_flags; skip geographic exclusion by bit here
                    pass
        cells[f"is_{name}"] = mask
    return cells


def build_control_membership(cells: pd.DataFrame) -> pd.DataFrame:
    """Return a per-cell membership table for all controls."""
    cells = assign_functional_controls(cells)
    cells = assign_geographic_controls(cells)
    return cells
