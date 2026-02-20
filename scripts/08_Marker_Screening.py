#!/usr/bin/env python3
"""
Marker Screening (Spearman + Memory Diagnostics): SIF Residuals vs. SII and F10.7.
Designed as a formal triage step BEFORE the full GPU matrix search.

Goals:
1. Compare geomagnetic (SII=-Dst) vs solar activity (F10.7) markers on the same temporal grids.
2. Quantify "accumulation fingerprint": MA effects vs LAG effects (short-memory vs long-memory).
3. Provide a defensible, pre-registered-style criterion for selecting the primary space-weather marker
   used in the expensive multivariate Matrix Search stage (08_Matrix_Search_GPU.py).
4. Keep runtime manageable by restricting the tested OMNI2 features to:
   - SII family: sii_max and/or sii_mean (configurable)
   - F10.7 family: f10_7_mean
   and their derived lag/MA features only (no kp, no diffs unless explicitly enabled).

Methodology:
1. Loads SIF residuals (from 05_SIF_anomalies.py) and merges ERA5 temperature context and OMNI features.
2. Applies scenario masks:
   - Global_High_LAI, SAA_High_LAI, Control_North, Sahara_Barren (Config.SCENARIO_MASKS).
3. Stratifies by temperature bins using TEMP_CONTEXT_COL (Config.TEMP_CONTEXT_COL).
4. For each target × scenario × temp-bin:
   A) Univariate Spearman screening:
      - Computes Spearman rho for each window/lag feature of SII and F10.7.
      - Applies effective sample size correction N_eff (Chelton 1983; Pyper & Peterman 1998)
        using lag-1 autocorrelation of (Y, X) within the bin.
      - Reports Fisher-z and its magnitude: z = arctanh(rho).
      - Summarizes profiles by:
          * auc_abs_z_ma  = mean(|z|) over MA windows
          * auc_abs_z_lag = mean(|z|) over discrete lags
          * delta_acc     = mean(|z|)_MA - mean(|z|)_LAG
          * peak_feature  = window/lag with max |z|
   B) Cheap multivariate screening (same-window pairwise regression):
      - For each window w (MA) and lag l (LAG), fits a 2-predictor rank-Gaussian model:
            Y_rankz ~ SII_(w or l) + F10.7_(w or l)
        using closed-form correlation-system solution (no heavy OLS).
      - Reports t-stats for beta_SII and beta_F, with N_eff-based df (conservative).

Outputs:
- results/screening_profiles_{target}_{scenario}.csv
  Row per (bin, family, mode, window_type, window_value), containing rho, z, p_adj, n_eff, etc.
- results/screening_summary_{target}_{scenario}.csv
  Row per (bin, family), containing auc_abs_z_ma, auc_abs_z_lag, delta_acc, peak, etc.
- results/screening_pairwise_{target}_{scenario}.csv
  Row per (bin, mode, window_value), containing (t_sii, t_f10, beta_sii, beta_f10, n_eff_used, winner).

References:
- Chelton, D. B. (1983). Deep Sea Research, 30, 1083–1103.
- Pyper, B. J., & Peterman, R. M. (1998). Can. J. Fish. Aquat. Sci., 55, 2127–2140.
"""

import re
import gc
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from _Common import Config

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ERA5_ENV_FILE = Config.FILE_ERA5_PARQUET
TEMP_COLUMN = Config.TEMP_CONTEXT_COL

TARGET_VARIABLES = getattr(
    Config,
    "SPEARMAN_TARGETS",
    ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"],
)
MIN_SAMPLES = int(getattr(Config, "SPEARMAN_MIN_SAMPLES_PER_BIN", 50))
SCENARIO_NAMES = list(getattr(Config, "SCENARIO_MASKS", {}).keys())

# Which OMNI "base" vars to compare
SII_BASES = list(getattr(Config, "SCREEN_SII_BASES", ["sii_mean"]))
F10_BASES = list(getattr(Config, "SCREEN_F10_BASES", ["f10_7_mean"]))

# Whether to include *_diff features (usually not needed for screening)
INCLUDE_DIFFS = bool(getattr(Config, "SCREEN_INCLUDE_DIFFS", False))

# ------------------------------------------------------------------------------
# Helpers for parsing feature names into (mode, window_value)
# We assume features come from 01_Omni2_ETL naming:
#   <base>_ma{w} for MA windows
#   <base>_lag{l} for discrete lags
# ------------------------------------------------------------------------------

RE_MA = re.compile(r"_ma(\d+)$")
RE_LAG = re.compile(r"_lag(\d+)$")
RE_DIFF = re.compile(r"_diff$")


def _parse_window(feature: str) -> Tuple[str, int]:
    """
    Returns (mode, window_value) where mode in {"MA","LAG"}.
    Raises ValueError if feature does not match.
    """
    m = RE_MA.search(feature)
    if m:
        return "MA", int(m.group(1))
    m = RE_LAG.search(feature)
    if m:
        return "LAG", int(m.group(1))
    raise ValueError(feature)


# ==============================================================================
# STATISTICAL HELPERS
# ==============================================================================

def calculate_lag1_autocorr(series: np.ndarray) -> float:
    valid = series[np.isfinite(series)]
    if len(valid) < 5:
        return 0.0
    return float(np.corrcoef(valid[:-1], valid[1:])[0, 1])


def neff_factor_xy(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    r1_x = np.clip(calculate_lag1_autocorr(x), -0.99, 0.99)
    r1_y = np.clip(calculate_lag1_autocorr(y), -0.99, 0.99)
    prod = r1_x * r1_y
    factor = (1.0 - prod) / (1.0 + prod)
    return float(factor), float(r1_x), float(r1_y)


def spearman_with_neff(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns (rho, p_adj, n_eff, z) where z = arctanh(rho).
    """
    rho, _ = stats.spearmanr(x, y)
    rho = float(rho) if np.isfinite(rho) else np.nan

    factor, _, _ = neff_factor_xy(x, y)
    n_raw = len(x)
    n_eff = max(2.0, n_raw * factor)

    if not np.isfinite(rho):
        return np.nan, np.nan, float(n_eff), np.nan

    if abs(rho) >= 1.0:
        t_stat = np.inf
    else:
        t_stat = rho * np.sqrt((n_eff - 2.0) / (1.0 - rho * rho))

    p_adj = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1.0, n_eff - 2.0)))

    # Fisher z (for profile aggregation)
    if abs(rho) >= 1.0:
        z = np.sign(rho) * np.inf
    else:
        z = float(np.arctanh(rho))

    return rho, float(p_adj), float(n_eff), z


def rank_gaussian(x: np.ndarray) -> np.ndarray:
    """
    Rank-based Gaussianization (Rank-Z).
    Matches the spirit of the pipeline: robust to outliers and non-Gaussian tails.
    """
    x = np.asarray(x)
    n = x.size
    r = stats.rankdata(x, method="average")
    u = (r - 0.5) / n
    return stats.norm.ppf(u)


def two_predictor_beta_t(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, n_eff: float) -> Tuple[float, float, float, float]:
    """
    Closed-form standardized regression using correlation system:
        [1, r12] [b1] = [r1y]
        [r12,1] [b2]   [r2y]
    where variables are standardized (mean=0, std=1).

    Returns (beta1, beta2, t1, t2) using conservative df ~ n_eff - 3.
    """
    # Standardize
    def zscore(a):
        a = np.asarray(a, dtype=float)
        a = a - np.mean(a)
        s = np.std(a)
        return a / s if s > 0 else a * 0.0

    y = zscore(y)
    x1 = zscore(x1)
    x2 = zscore(x2)

    r12 = float(np.mean(x1 * x2))
    r1y = float(np.mean(x1 * y))
    r2y = float(np.mean(x2 * y))

    # Invert 2x2
    det = 1.0 - r12 * r12
    if det <= 1e-12:
        # near collinearity: return nan t-stats
        b1 = (r1y - r12 * r2y) / max(det, 1e-12)
        b2 = (r2y - r12 * r1y) / max(det, 1e-12)
        return float(b1), float(b2), np.nan, np.nan

    b1 = (r1y - r12 * r2y) / det
    b2 = (r2y - r12 * r1y) / det

    # Approx SE for standardized coefficients (conservative)
    df = max(1.0, n_eff - 3.0)
    # residual variance for standardized model:
    r2 = b1 * r1y + b2 * r2y
    sigma2 = max(1e-12, 1.0 - r2)
    # Var(b1) approx = sigma2 / df * 1/det
    se = np.sqrt(sigma2 / df * (1.0 / det))
    t1 = b1 / se
    t2 = b2 / se

    return float(b1), float(b2), float(t1), float(t2)


# ==============================================================================
# MAIN LOGIC
# ==============================================================================

@dataclass
class FeatureFamily:
    name: str
    bases: List[str]


class MarkerScreeningAnalyzer:
    def __init__(self):
        # load OMNI features once
        self.omni_df = pd.read_feather(Config.FILE_OMNI_FEATHER)
        self.omni_df["date"] = pd.to_datetime(self.omni_df["date"]).dt.normalize()

        # Select only features that belong to our families and are MA/LAG (and optionally diff)
        self.families = [
            FeatureFamily("SII", SII_BASES),
            FeatureFamily("F10.7", F10_BASES),
        ]
        self.family_features = self._select_family_features(self.omni_df)

        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _select_family_features(self, df_omni: pd.DataFrame) -> Dict[str, List[str]]:
        num_cols = df_omni.select_dtypes(include=[np.number]).columns.tolist()
        feats_by_family: Dict[str, List[str]] = {}

        for fam in self.families:
            feats = []
            for base in fam.bases:
                # MA/LAG features only
                base_re = re.compile(rf"^{re.escape(base)}_(ma\d+|lag\d+)$")
                for c in num_cols:
                    if base_re.match(c):
                        feats.append(c)
                if INCLUDE_DIFFS:
                    diff_re = re.compile(rf"^{re.escape(base)}_diff$")
                    for c in num_cols:
                        if diff_re.match(c):
                            feats.append(c)

            # sort features so MA and LAG are in numeric order
            def _sort_key(c: str):
                try:
                    mode, w = _parse_window(c)
                    return (0 if mode == "LAG" else 1, w, c)
                except Exception:
                    return (2, 10**9, c)

            feats_by_family[fam.name] = sorted(list(set(feats)), key=_sort_key)

        return feats_by_family

    def load_residuals_with_flags(self, var_name: str) -> pd.DataFrame:
        res_path = Config.DIR_SIF_MODEL / f"sif_residuals_{var_name}.parquet"
        df_res = pd.read_parquet(res_path)
        df_res["date"] = pd.to_datetime(df_res["date"]).dt.normalize()

        df_flags = pd.read_feather(
            Config.FILE_SIF_FINAL,
            columns=["date", "lat_id", "lon_id", "region_flags"],
        )
        df_flags["date"] = pd.to_datetime(df_flags["date"]).dt.normalize()

        return pd.merge(df_res, df_flags, on=["date", "lat_id", "lon_id"], how="inner")

    def merge_env_and_omni(self, df_res: pd.DataFrame) -> pd.DataFrame:
        years = sorted(df_res["date"].dt.year.unique())
        chunks = []

        for yr in tqdm(years, desc="  > Env Merge", leave=False):
            sif_yr = df_res[df_res["date"].dt.year == yr]
            if sif_yr.empty:
                continue

            try:
                env_yr = pd.read_parquet(
                    ERA5_ENV_FILE,
                    columns=["date", "lat_id", "lon_id", TEMP_COLUMN],
                    filters=[
                        ("date", ">=", pd.Timestamp(f"{yr}-01-01")),
                        ("date", "<=", pd.Timestamp(f"{yr}-12-31")),
                    ],
                )
                env_yr["date"] = pd.to_datetime(env_yr["date"]).dt.normalize()
                merged = pd.merge(sif_yr, env_yr, on=["date", "lat_id", "lon_id"], how="inner")
                chunks.append(merged)
            except Exception:
                continue

        if not chunks:
            return pd.DataFrame()

        full = pd.concat(chunks, ignore_index=True)
        full = pd.merge(full, self.omni_df, on="date", how="inner")
        return full

    def _summarize_profile(self, prof: pd.DataFrame) -> Dict:
        """
        Given per-window rows (with columns: mode, window_value, z),
        compute profile summaries for MA and LAG.
        """
        out = {}

        for mode in ["MA", "LAG"]:
            s = prof[prof["mode"] == mode].copy()
            s = s[np.isfinite(s["z"].values)]
            if s.empty:
                out[f"auc_abs_z_{mode.lower()}"] = np.nan
                out[f"peak_{mode.lower()}"] = None
                out[f"peak_abs_z_{mode.lower()}"] = np.nan
                continue

            s["abs_z"] = np.abs(s["z"].values)
            out[f"auc_abs_z_{mode.lower()}"] = float(s["abs_z"].mean())

            idx = s["abs_z"].idxmax()
            out[f"peak_{mode.lower()}"] = str(s.loc[idx, "feature"])
            out[f"peak_abs_z_{mode.lower()}"] = float(s.loc[idx, "abs_z"])

        # accumulation index: MA mean(|z|) - LAG mean(|z|)
        out["delta_acc"] = out.get("auc_abs_z_ma", np.nan) - out.get("auc_abs_z_lag", np.nan)
        return out

    def run_analysis(self):
        print("[START] Marker Screening (Spearman + Accumulation + SII vs F10.7)")

        target_pbar = tqdm(TARGET_VARIABLES, desc="Overall Targets")
        for target in target_pbar:
            target_pbar.set_postfix({"Current": target})

            try:
                df_res = self.load_residuals_with_flags(target)
                full_ds = self.merge_env_and_omni(df_res)
                if full_ds.empty:
                    print(f"  [WARN] No data after merges for {target}")
                    continue

                full_ds = full_ds.dropna(subset=[TEMP_COLUMN]).sort_values("date")

                # Scenario loop
                scenario_pbar = tqdm(SCENARIO_NAMES, desc=f"  Scenarios ({target})", leave=False)
                for scenario_name in scenario_pbar:
                    scenario_pbar.set_postfix({"Scen": scenario_name})

                    mask = Config.scenario_mask(full_ds["region_flags"].values, scenario_name)
                    scen = full_ds[mask].copy()
                    if len(scen) < MIN_SAMPLES * 5:
                        continue

                    # Temperature bins
                    try:
                        tb = Config.bin_temperature(scen[TEMP_COLUMN])
                        scen = scen.join(tb)
                    except Exception:
                        continue

                    # Outputs
                    profile_rows = []
                    summary_rows = []
                    pair_rows = []

                    # iterate bins
                    for b_id in sorted(scen["temp_bin_id"].dropna().unique()):
                        bin_df = scen[scen["temp_bin_id"] == b_id].copy()
                        if bin_df.empty:
                            continue
                        bin_label = (
                            str(bin_df["temp_bin_label"].dropna().iloc[0])
                            if bin_df["temp_bin_label"].notna().any()
                            else None
                        )

                        # Base series
                        valid_y = bin_df[["date", "residual"]].dropna()
                        if len(valid_y) < MIN_SAMPLES:
                            continue

                        # ----------------------------------------------------------
                        # A) Univariate profiles: Spearman for each family feature
                        # ----------------------------------------------------------
                        for fam in self.families:
                            feats = self.family_features.get(fam.name, [])
                            if not feats:
                                continue

                            prof = []
                            for feat in feats:
                                v = bin_df[["date", "residual", feat]].dropna()
                                if len(v) < MIN_SAMPLES:
                                    continue
                                v = v.sort_values("date")

                                try:
                                    mode, w = _parse_window(feat)
                                except Exception:
                                    # skip non-window features unless explicitly allowed
                                    continue

                                rho, p_adj, n_eff, z = spearman_with_neff(
                                    v["residual"].values,
                                    v[feat].values,
                                )

                                prof.append({
                                    "target": target,
                                    "scenario": scenario_name,
                                    "family": fam.name,
                                    "bin_id": int(b_id),
                                    "bin_label": bin_label,
                                    "temp_mean": float(bin_df[TEMP_COLUMN].mean()),
                                    "feature": feat,
                                    "mode": mode,
                                    "window_value": int(w),
                                    "n": int(len(v)),
                                    "n_eff": float(n_eff),
                                    "rho": float(rho),
                                    "z": float(z),
                                    "p_adj": float(p_adj),
                                })

                            if not prof:
                                continue

                            prof_df = pd.DataFrame(prof)

                            # store per-window profile rows
                            profile_rows.extend(prof)

                            # summary rows (AUC, delta_acc, peaks)
                            summ = self._summarize_profile(prof_df)
                            summary_rows.append({
                                "target": target,
                                "scenario": scenario_name,
                                "family": fam.name,
                                "bin_id": int(b_id),
                                "bin_label": bin_label,
                                "temp_mean": float(bin_df[TEMP_COLUMN].mean()),
                                "n_bin": int(len(valid_y)),
                                **summ,
                            })

                        # ----------------------------------------------------------
                        # B) Pairwise screening: same-window 2-predictor rankZ model
                        # ----------------------------------------------------------
                        # We compare SII vs F10.7 using matched windows (MA and LAG).
                        # Choose one SII base for this paired test (prefer sii_max if present).
                        sii_base = "sii_max" if "sii_max" in SII_BASES else SII_BASES[0]
                        f_base = F10_BASES[0]

                        # find intersection of available windows in OMNI for both bases
                        # For each mode separately.
                        for mode, windows in [("MA", getattr(Config, "MA_WINDOWS", Config.MA_WINDOWS)),
                                              ("LAG", getattr(Config, "DISCRETE_LAGS", Config.DISCRETE_LAGS))]:
                            for w in windows:
                                sii_feat = f"{sii_base}_{'ma' if mode=='MA' else 'lag'}{w}"
                                f_feat = f"{f_base}_{'ma' if mode=='MA' else 'lag'}{w}"
                                if sii_feat not in bin_df.columns or f_feat not in bin_df.columns:
                                    continue

                                v = bin_df[["date", "residual", sii_feat, f_feat]].dropna()
                                if len(v) < MIN_SAMPLES:
                                    continue
                                v = v.sort_values("date")

                                # Rank-Z
                                y_rg = rank_gaussian(v["residual"].values)
                                x1_rg = rank_gaussian(v[sii_feat].values)
                                x2_rg = rank_gaussian(v[f_feat].values)

                                # conservative N_eff: use min of pairwise neff(Y,X1) and neff(Y,X2)
                                _, _, n_eff1, _ = spearman_with_neff(v["residual"].values, v[sii_feat].values)
                                _, _, n_eff2, _ = spearman_with_neff(v["residual"].values, v[f_feat].values)
                                n_eff_use = float(max(5.0, min(n_eff1, n_eff2)))

                                b1, b2, t1, t2 = two_predictor_beta_t(
                                    y_rg, x1_rg, x2_rg, n_eff_use
                                )

                                winner = None
                                if np.isfinite(t1) and np.isfinite(t2):
                                    winner = "SII" if abs(t1) > abs(t2) else "F10.7"

                                pair_rows.append({
                                    "target": target,
                                    "scenario": scenario_name,
                                    "bin_id": int(b_id),
                                    "bin_label": bin_label,
                                    "temp_mean": float(bin_df[TEMP_COLUMN].mean()),
                                    "mode": mode,
                                    "window_value": int(w),
                                    "n": int(len(v)),
                                    "n_eff_used": float(n_eff_use),
                                    "sii_feature": sii_feat,
                                    "f10_feature": f_feat,
                                    "beta_sii": float(b1),
                                    "beta_f10": float(b2),
                                    "t_sii": float(t1) if np.isfinite(t1) else np.nan,
                                    "t_f10": float(t2) if np.isfinite(t2) else np.nan,
                                    "winner": winner,
                                })

                    # Write outputs per (target, scenario)
                    if profile_rows:
                        out = pd.DataFrame(profile_rows)
                        out_file = Config.RESULTS_DIR / f"screening_profiles_{target}_{scenario_name}.csv"
                        out.to_csv(out_file, index=False)

                    if summary_rows:
                        out = pd.DataFrame(summary_rows)
                        out_file = Config.RESULTS_DIR / f"screening_summary_{target}_{scenario_name}.csv"
                        out.to_csv(out_file, index=False)

                    if pair_rows:
                        out = pd.DataFrame(pair_rows)
                        out_file = Config.RESULTS_DIR / f"screening_pairwise_{target}_{scenario_name}.csv"
                        out.to_csv(out_file, index=False)

                # cleanup
                del df_res, full_ds
                gc.collect()

            except Exception as e:
                print(f"\n[ERROR] {target}: {e}")


if __name__ == "__main__":
    MarkerScreeningAnalyzer().run_analysis()
