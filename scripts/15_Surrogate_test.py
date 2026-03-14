#!/usr/bin/env python3
"""
15_Surrogate_Test.py

Lightweight surrogate test for the main MAGNETO result.

Idea
----
We do NOT rerun the full pipeline.
We reuse already prepared:
- OMNI daily features (Step 01)
- ERA5 daily context (Step 03)
- SIF residuals (Step 05)

For a small set of key target/scenario/temp-bin combinations, we:
1) compute the observed correlation spectrum rho(window)
2) generate circular-shift surrogates of the geomagnetic driver
3) recompute only rho(window) for the selected windows
4) compare observed summary metrics against surrogate distributions

Recommended interpretation:
- primary metric: signed_auc over a fixed window band
- secondary metric: max_abs_rho over the same band

Outputs
-------
results/surrogate_test/
    surrogate_spectrum_<target>_<scenario>_<bin>.csv
    surrogate_metrics_<target>_<scenario>_<bin>.csv
    surrogate_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.feather as feather
from scipy import stats

from tqdm.auto import tqdm

from _Common import Config

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIG
# ==============================================================================

SURROGATE_MODE = "permute_years"
MIN_YEAR_BLOCKS = 3

TARGETS = ["sif_stress_index", "sif_771nm"]
SCENARIOS = ["Global_High_LAI", "Control_North"]
TEMP_BIN_LABELS = ["Cold", "Cool"]
WINDOWS = list(range(1, 29)) + [30, 40, 50, 60, 75, 90]
OMNI_BASE = "sii_mean"
N_SURR = 500
MIN_N = 100
AUC_MIN_W = 20
AUC_MAX_W = 30
RANDOM_SEED = 12032026

# Where to save results
OUT_DIR = Config.RESULTS_DIR / "surrogate_test"


# ==============================================================================
# HELPERS
# ==============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_omni_cols(base: str, windows: list[int]) -> list[str]:
    return [f"{base}_ma{w}" for w in windows]


def fast_spearman_vector(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Vectorized Spearman correlation between y (N,) and X (N, M).
    Returns rho for each column of X.
    Assumes NaNs already removed.
    """
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y) != X.shape[0]:
        raise ValueError("y and X must have the same number of rows")

    # Rank-transform
    y_rank = stats.rankdata(y).astype(np.float64)
    # scipy.stats.rankdata supports axis in modern SciPy; fallback if needed
    try:
        X_rank = stats.rankdata(X, axis=0).astype(np.float64)
    except TypeError:
        X_rank = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])]).astype(np.float64)

    # Standardize
    y_rank -= y_rank.mean()
    X_rank -= X_rank.mean(axis=0)

    y_ss = np.sqrt(np.sum(y_rank ** 2))
    X_ss = np.sqrt(np.sum(X_rank ** 2, axis=0))

    denom = y_ss * X_ss
    denom[denom == 0] = np.nan

    rho = (y_rank @ X_rank) / denom
    return rho


def spectrum_metrics(rho: np.ndarray, windows: np.ndarray, auc_min_w: int, auc_max_w: int) -> dict:
    """
    Compute compact summary metrics from rho(window).
    """
    rho = np.asarray(rho, dtype=float)
    windows = np.asarray(windows, dtype=float)

    in_band = (windows >= auc_min_w) & (windows <= auc_max_w)
    if not np.any(in_band):
        raise ValueError("No windows fall inside the AUC band.")

    rho_band = rho[in_band]
    w_band = windows[in_band]

    # signed AUC over fixed biologically relevant range
    signed_auc = np.trapz(rho_band, x=w_band)

    # maximum absolute correlation in the same range
    max_abs_rho = np.nanmax(np.abs(rho_band))

    # optional: mean rho over the band
    mean_rho = np.nanmean(rho_band)

    return {
        "signed_auc": float(signed_auc),
        "max_abs_rho": float(max_abs_rho),
        "mean_rho": float(mean_rho),
    }


def monte_carlo_pvalue(obs: float, surr: np.ndarray, two_sided: bool = True) -> float:
    """
    Exact empirical Monte Carlo p-value with +1 correction.
    """
    surr = np.asarray(surr, dtype=float)
    if two_sided:
        count = np.sum(np.abs(surr) >= abs(obs))
    else:
        count = np.sum(surr >= obs)
    return (1.0 + count) / (len(surr) + 1.0)


def recompute_ma_columns(df: pd.DataFrame, base_col: str, windows: list[int]) -> pd.DataFrame:
    out = df[["date", base_col]].copy()
    out = out.sort_values("date").reset_index(drop=True)

    for w in windows:
        out[f"{base_col}_ma{w}"] = (
            out[base_col]
            .shift(1)
            .rolling(window=w, min_periods=w)
            .mean()
        )
    return out


def permute_years_base_series(
    omni_daily: pd.DataFrame,
    base_col: str,
    rng: np.random.Generator,
    min_year_blocks: int = 3,
) -> pd.DataFrame:
    """
    Permute whole calendar years of the raw daily SII series, then return a new
    daily series with the ORIGINAL date axis and permuted base_col values.
    """
    tmp = omni_daily[["date", base_col]].copy().sort_values("date").reset_index(drop=True)
    tmp["year"] = tmp["date"].dt.year

    years = np.array(sorted(tmp["year"].unique()))
    if len(years) < min_year_blocks:
        raise ValueError(f"Too few years for year permutation: {len(years)}")

    perm_years = rng.permutation(years)

    # avoid identity permutation
    tries = 0
    while np.array_equal(perm_years, years) and tries < 20:
        perm_years = rng.permutation(years)
        tries += 1

    pieces = []
    for y in perm_years:
        part = tmp.loc[tmp["year"] == y, [base_col]].copy()
        pieces.append(part)

    perm_values = pd.concat(pieces, ignore_index=True)[base_col].to_numpy()

    if len(perm_values) != len(tmp):
        raise RuntimeError("Permuted series length mismatch.")

    out = tmp[["date"]].copy()
    out[base_col] = perm_values
    return out

# ==============================================================================
# DATA LOADER
# ==============================================================================

class SurrogateAnalyzer:
    def __init__(self):
        ensure_dir(OUT_DIR)

        # OMNI already contains precomputed moving averages from Step 01
        self.omni_df = feather.read_feather(Config.FILE_OMNI_FEATHER)
        self.omni_df["date"] = pd.to_datetime(self.omni_df["date"]).dt.normalize()
        self.omni_daily = self.omni_df[["date", OMNI_BASE]].copy().sort_values("date").reset_index(drop=True)

        # Region flags / valid cells map
        self.df_flags = pd.read_feather(
            Config.FILE_SIF_FINAL,
            columns=["lat_id", "lon_id", "region_flags", "date"]
        )
        self.df_flags["date"] = pd.to_datetime(self.df_flags["date"]).dt.normalize()

        # ERA5 schema for temp context lookup
        self.era5_schema = pq.read_schema(Config.FILE_ERA5_PARQUET).names
        self.temp_col = self._find_temp_col()

    def _find_temp_col(self) -> str:
        candidates = [Config.TEMP_CONTEXT_COL, f"temp_c_ma{Config.CONTEXT_WINDOW_DAYS}"]
        for c in candidates:
            if c in self.era5_schema:
                return c
        raise ValueError("Temperature context column not found in ERA5 parquet.")

    def load_scenario_data(self, target: str, scenario_name: str, omni_cols: list[str]) -> pd.DataFrame | None:
        """
        Mirrors the Step-06 logic but loads only what is needed:
        residual + date/lat/lon + temperature context + selected OMNI columns.
        """
        print(f"[INFO] Loading target={target}, scenario={scenario_name}")

        # 1) scenario flags
        mask = Config.scenario_mask(self.df_flags["region_flags"].values, scenario_name)
        valid_flags = self.df_flags.loc[mask, ["date", "lat_id", "lon_id"]].copy()
        if valid_flags.empty:
            print(f"[WARN] No valid flags for scenario={scenario_name}")
            return None

        # 2) residuals
        res_path = Config.DIR_SIF_MODEL / f"sif_residuals_{target}.parquet"
        if not res_path.exists():
            print(f"[WARN] Missing residual file: {res_path}")
            return None

        df_res = pd.read_parquet(res_path)
        df_res["date"] = pd.to_datetime(df_res["date"]).dt.normalize()

        # Restrict to scenario universe early
        target_universe = pd.merge(
            df_res,
            valid_flags,
            on=["date", "lat_id", "lon_id"],
            how="inner"
        )
        del df_res, valid_flags
        gc.collect()

        if target_universe.empty:
            print(f"[WARN] No rows after merging residuals with scenario mask for {scenario_name}")
            return None

        # 3) stream ERA5 just for temp context
        accumulated = []
        parquet_file = pq.ParquetFile(Config.FILE_ERA5_PARQUET)
        cols_to_read = ["date", "lat_id", "lon_id", self.temp_col]

        for batch in parquet_file.iter_batches(batch_size=500_000, columns=cols_to_read):
            batch_df = batch.to_pandas()
            batch_df["date"] = pd.to_datetime(batch_df["date"]).dt.normalize()

            merged = pd.merge(
                target_universe,
                batch_df,
                on=["date", "lat_id", "lon_id"],
                how="inner"
            )
            if not merged.empty:
                accumulated.append(merged)

            del batch_df, merged

        del target_universe
        gc.collect()

        if not accumulated:
            print(f"[WARN] No rows after ERA5 merge for {scenario_name}")
            return None

        scen_df = pd.concat(accumulated, ignore_index=True)
        del accumulated
        gc.collect()

        # 4) merge only selected OMNI columns
        need_cols = ["date", OMNI_BASE] + omni_cols
        omni_small = self.omni_df[need_cols].copy()
        scen_df = pd.merge(scen_df, omni_small, on="date", how="inner")

        if scen_df.empty:
            print(f"[WARN] Empty after OMNI merge for {scenario_name}")
            return None

        # 5) temperature bins exactly as in main analysis
        tb = Config.bin_temperature(scen_df[self.temp_col])
        scen_df = scen_df.join(tb)

        return scen_df


# ==============================================================================
# SURROGATE CORE
# ==============================================================================

def analyze_one_combination(
    df: pd.DataFrame,
    target: str,
    scenario: str,
    temp_label: str,
    omni_cols: list[str],
    windows: list[int],
    n_surr: int,
    min_n: int,
    auc_min_w: int,
    auc_max_w: int,
    rng: np.random.Generator,
    base_col: str,
    min_year_blocks: int = 3,
) -> dict | None:
    """
    Runs surrogate test for one (target, scenario, temp_bin) combination.

    Surrogate logic:
    - take the raw daily geomagnetic base series (e.g. sii_mean)
    - permute whole years
    - recompute moving-average windows from the permuted base series
    - recompute rho(window) against the fixed SIF residual series
    """
    sub = df.loc[df["temp_bin_label"].astype(str) == str(temp_label)].copy()
    sub = sub.dropna(subset=["residual", base_col] + omni_cols)

    if len(sub) < min_n:
        print(f"[WARN] Too few samples: target={target}, scenario={scenario}, bin={temp_label}, n={len(sub)}")
        return None

    # Observed spectrum from the original data
    y = sub["residual"].to_numpy(dtype=float)
    X = sub[omni_cols].to_numpy(dtype=float)
    w = np.asarray(windows, dtype=float)

    obs_rho = fast_spearman_vector(y, X)
    obs_metrics = spectrum_metrics(obs_rho, w, auc_min_w, auc_max_w)

    # Base daily series for surrogate generation:
    # one value per date, sorted on the original date axis
    base_daily = (
        sub[["date", base_col]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    years_available = base_daily["date"].dt.year.nunique()
    if years_available < min_year_blocks:
        print(
            f"[WARN] Too few years for year permutation: "
            f"target={target}, scenario={scenario}, bin={temp_label}, years={years_available}"
        )
        return None

    surr_rows = []
    surr_max = []
    surr_auc = []
    surr_mean = []

    surr_iter = tqdm(
        range(1, n_surr + 1),
        desc=f"{target} | {scenario} | {temp_label}",
        unit="surr",
        leave=False,
        dynamic_ncols=True,
    )

    for s_idx in surr_iter:
        # 1) permute whole years of the raw daily base series
        perm_daily = permute_years_base_series(
            omni_daily=base_daily,
            base_col=base_col,
            rng=rng,
            min_year_blocks=min_year_blocks,
        )

        # 2) recompute moving averages from the permuted base series
        perm_ma = recompute_ma_columns(
            df=perm_daily,
            base_col=base_col,
            windows=windows,
        )

        # 3) merge surrogate OMNI windows onto the fixed residual series
        sub_s = pd.merge(
            sub[["date", "residual"]].copy(),
            perm_ma[["date"] + omni_cols],
            on="date",
            how="inner",
        ).dropna(subset=["residual"] + omni_cols)

        if len(sub_s) < min_n:
            continue

        y_s = sub_s["residual"].to_numpy(dtype=float)
        X_s = sub_s[omni_cols].to_numpy(dtype=float)

        surr_iter.set_postfix({
            "n": len(sub_s),
            "years": years_available,
        })

        rho_s = fast_spearman_vector(y_s, X_s)
        met_s = spectrum_metrics(rho_s, w, auc_min_w, auc_max_w)

        surr_max.append(met_s["max_abs_rho"])
        surr_auc.append(met_s["signed_auc"])
        surr_mean.append(met_s["mean_rho"])

        for win_i, win in enumerate(windows):
            surr_rows.append({
                "kind": "surrogate",
                "surrogate_id": s_idx,
                "shift_days": np.nan,
                "target": target,
                "scenario": scenario,
                "temp_bin_label": temp_label,
                "window": int(win),
                "rho": float(rho_s[win_i]),
            })

    if len(surr_auc) == 0:
        print(f"[WARN] No valid surrogate realizations: target={target}, scenario={scenario}, bin={temp_label}")
        return None

    surr_max = np.asarray(surr_max, dtype=float)
    surr_auc = np.asarray(surr_auc, dtype=float)
    surr_mean = np.asarray(surr_mean, dtype=float)

    obs_rows = [{
        "kind": "observed",
        "surrogate_id": 0,
        "shift_days": 0,
        "target": target,
        "scenario": scenario,
        "temp_bin_label": temp_label,
        "window": int(win),
        "rho": float(obs_rho[i]),
    } for i, win in enumerate(windows)]

    spectrum_df = pd.DataFrame(obs_rows + surr_rows)

    metrics_df = pd.DataFrame([
        {
            "target": target,
            "scenario": scenario,
            "temp_bin_label": temp_label,
            "n": int(len(sub)),
            "n_surrogates_used": int(len(surr_auc)),
            "auc_window_min": int(auc_min_w),
            "auc_window_max": int(auc_max_w),
            "metric": "signed_auc",
            "observed": float(obs_metrics["signed_auc"]),
            "surrogate_mean": float(np.mean(surr_auc)),
            "surrogate_std": float(np.std(surr_auc, ddof=1)) if len(surr_auc) > 1 else np.nan,
            "surrogate_q025": float(np.quantile(surr_auc, 0.025)),
            "surrogate_q500": float(np.quantile(surr_auc, 0.500)),
            "surrogate_q975": float(np.quantile(surr_auc, 0.975)),
            "p_empirical_two_sided": float(monte_carlo_pvalue(obs_metrics["signed_auc"], surr_auc, two_sided=True)),
        },
        {
            "target": target,
            "scenario": scenario,
            "temp_bin_label": temp_label,
            "n": int(len(sub)),
            "n_surrogates_used": int(len(surr_max)),
            "auc_window_min": int(auc_min_w),
            "auc_window_max": int(auc_max_w),
            "metric": "max_abs_rho",
            "observed": float(obs_metrics["max_abs_rho"]),
            "surrogate_mean": float(np.mean(surr_max)),
            "surrogate_std": float(np.std(surr_max, ddof=1)) if len(surr_max) > 1 else np.nan,
            "surrogate_q025": float(np.quantile(surr_max, 0.025)),
            "surrogate_q500": float(np.quantile(surr_max, 0.500)),
            "surrogate_q975": float(np.quantile(surr_max, 0.975)),
            "p_empirical_two_sided": float(monte_carlo_pvalue(obs_metrics["max_abs_rho"], surr_max, two_sided=False)),
        },
        {
            "target": target,
            "scenario": scenario,
            "temp_bin_label": temp_label,
            "n": int(len(sub)),
            "n_surrogates_used": int(len(surr_mean)),
            "auc_window_min": int(auc_min_w),
            "auc_window_max": int(auc_max_w),
            "metric": "mean_rho",
            "observed": float(obs_metrics["mean_rho"]),
            "surrogate_mean": float(np.mean(surr_mean)),
            "surrogate_std": float(np.std(surr_mean, ddof=1)) if len(surr_mean) > 1 else np.nan,
            "surrogate_q025": float(np.quantile(surr_mean, 0.025)),
            "surrogate_q500": float(np.quantile(surr_mean, 0.500)),
            "surrogate_q975": float(np.quantile(surr_mean, 0.975)),
            "p_empirical_two_sided": float(monte_carlo_pvalue(obs_metrics["mean_rho"], surr_mean, two_sided=True)),
        },
    ])

    stem = f"{target}_{scenario}_{temp_label}".replace(" ", "_")
    spectrum_df.to_csv(OUT_DIR / f"surrogate_spectrum_{stem}.csv", index=False)
    metrics_df.to_csv(OUT_DIR / f"surrogate_metrics_{stem}.csv", index=False)

    return {
        "summary_rows": metrics_df.to_dict(orient="records"),
        "spectrum_path": str(OUT_DIR / f"surrogate_spectrum_{stem}.csv"),
        "metrics_path": str(OUT_DIR / f"surrogate_metrics_{stem}.csv"),
        "n_used": int(len(sub)),
        "n_surrogates_used": int(len(surr_auc)),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ensure_dir(OUT_DIR)

    rng = np.random.default_rng(RANDOM_SEED)

    omni_cols = make_omni_cols(OMNI_BASE, WINDOWS)
    analyzer = SurrogateAnalyzer()

    # Check columns exist in OMNI
    missing = [c for c in omni_cols if c not in analyzer.omni_df.columns]
    if missing:
        raise ValueError(f"Missing OMNI columns: {missing}")

    all_summary_rows = []

    combinations = [
        (target, scenario, temp_label)
        for target in TARGETS
        for scenario in SCENARIOS
        for temp_label in TEMP_BIN_LABELS
    ]

    outer_pbar = tqdm(
        combinations,
        desc="Surrogate test",
        unit="combo",
        dynamic_ncols=True,
    )

    current_df_key = None
    current_df = None

    for target, scenario, temp_label in outer_pbar:
        outer_pbar.set_postfix({
            "target": target,
            "scenario": scenario,
            "bin": temp_label,
        })

        df_key = (target, scenario)

        # reload scenario dataframe only when target/scenario changes
        if current_df_key != df_key:
            if current_df is not None:
                del current_df
                gc.collect()

            current_df = analyzer.load_scenario_data(
                target=target,
                scenario_name=scenario,
                omni_cols=omni_cols,
            )
            current_df_key = df_key

        if current_df is None or current_df.empty:
            continue

        out = analyze_one_combination(
            df=current_df,
            target=target,
            scenario=scenario,
            temp_label=temp_label,
            omni_cols=omni_cols,
            windows=WINDOWS,
            n_surr=N_SURR,
            min_n=MIN_N,
            auc_min_w=AUC_MIN_W,
            auc_max_w=AUC_MAX_W,
            rng=rng,
            base_col=OMNI_BASE,
        )
        if out is not None:
            outer_pbar.set_postfix({
                "target": target,
                "scenario": scenario,
                "bin": temp_label,
                "n": out["n_used"],
            })
            all_summary_rows.extend(out["summary_rows"])

    if current_df is not None:
        del current_df
        gc.collect()

    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_df.to_csv(OUT_DIR / "surrogate_summary.csv", index=False)
        print(f"[OK] Wrote summary: {OUT_DIR / 'surrogate_summary.csv'}")
    else:
        print("[WARN] No surrogate outputs were produced.")


if __name__ == "__main__":
    main()

