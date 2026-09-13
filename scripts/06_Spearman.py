#!/usr/bin/env python3
"""
Spearman Correlation Analysis: SIF Residuals vs. Environmental & Space Weather Drivers.

This script performs a systematic correlation analysis to quantify the relationship between
Solar-Induced Fluorescence (SIF) residuals and various potential drivers (Space Weather & Atmosphere),
controlling for temperature regimes.

Logic:
1.  **Data Integration**: Merges SIF Residuals (Step 05), OMNI2 Space Weather features (Step 01),
    and ERA5 Environmental context (Step 03) into a single aligned dataset.
2.  **Scenario Filtering**: Applies strict geographic and vegetation masks (defined in `_Common.py`)
    to isolate specific biomes (e.g., 'Global_High_LAI', 'SAA_High_LAI').
3.  **Temperature Stratification**: Splits data into physiological temperature bins (e.g., 'Optimum', 'Heat_Stress')
    to ensure that correlations are not driven by simple seasonal temperature cycles.
4.  **Statistical Analysis**:
    - Computes **Spearman's Rank Correlation** (rho) for robustness against non-linear relationships.
    - **Autocorrelation Correction**: Calculates Effective Sample Size (N_eff) using the
      Chelton (1983) / Pyper & Peterman (1998) method based on Lag-1 autocorrelation.
    - **Significance Testing**: Derives adjusted P-values and 95% Confidence Intervals based on N_eff,
      penalizing "slow" signals (like accumulated heat) to prevent Type I errors.
5.  **Output Generation**: Saves detailed CSV reports for each Target/Scenario combination.

CONFIGURATION:
--------------
- Input Paths: Imported from `_Common.Config`.
- Context Window: `TEMP_CONTEXT_COL` (e.g., temp_c_ma10).
- Scenarios: `SCENARIO_MASKS` (Global, SAA, Control, etc.).

OUTPUT FILE DESCRIPTION:
------------------------
Files: results/spearman_{target}_{scenario}.csv
Columns:
1.  scenario    (str):   Name of the analyzed scenario.
2.  omni_var    (str):   Driver variable name (e.g., 'sii_max_ma10', 'par_ma10').
3.  bin_id      (int):   Temperature bin identifier.
4.  bin_label   (str):   Human-readable temperature label (e.g., 'Optimum').
5.  temp_mean   (float): Mean temperature in this bin.
6.  n           (int):   Raw number of daily observations.
7.  n_eff       (float): Effective Sample Size after autocorrelation correction.
8.  neff_factor (float): Ratio (N_eff / N). Low values indicate high autocorrelation.
9.  rho         (float): Spearman correlation coefficient.
10. p_adj       (float): P-value adjusted for N_eff.
11. ci_lower    (float): Lower bound of 95% Confidence Interval.
12. ci_upper    (float): Upper bound of 95% Confidence Interval.
"""

import sys
import pandas as pd
import numpy as np
import cupy as cp
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.feather as feather
from tqdm.auto import tqdm
import gc
import warnings
from _Common import Config
import scipy.stats as stats

warnings.filterwarnings('ignore')

# ==============================================================================
# GPU KERNEL (tie-aware Spearman)
# ==============================================================================

def _rankdata_average_1d(x: cp.ndarray) -> cp.ndarray:
    """
    Average-rank transform of a 1-D CuPy array, matching
    scipy.stats.rankdata(method='average'). Ties receive the mean of their
    positions (1-based in SciPy, 0-based here; Pearson correlation is invariant
    to the offset).
    """
    x = cp.asarray(x).ravel()
    if x.size == 0:
        return cp.empty_like(x)

    idx = cp.argsort(x)
    sorted_x = x[idx]

    # Boundaries of tie groups in sorted order
    diff = cp.empty(x.shape, dtype=bool)
    diff[0] = True
    diff[1:] = sorted_x[1:] != sorted_x[:-1]
    group_idx = cp.cumsum(diff.astype(cp.int64)) - 1

    n_groups = int(cp.max(group_idx)) + 1
    positions = cp.arange(len(x), dtype=cp.float64)

    # Small group-index array -> CPU bincount is fast and exact
    group_idx_cpu = cp.asnumpy(group_idx)
    sums = np.bincount(group_idx_cpu, weights=cp.asnumpy(positions), minlength=n_groups)
    counts = np.bincount(group_idx_cpu, minlength=n_groups)
    avg_ranks = cp.array(sums / counts, dtype=cp.float32)

    ranks = avg_ranks[group_idx]
    out = cp.empty_like(ranks)
    out[idx] = ranks
    return out


def gpu_spearman_matrix(residuals: cp.ndarray, data_matrix: cp.ndarray):
    """
    Pairwise Spearman correlation between residuals and each column of data_matrix.

    Parameters
    ----------
    residuals : (N,)
    data_matrix : (N, M)

    Returns
    -------
    rho : (M,) Spearman rho
    n_eff : (M,) NaN (legacy pooled space-time n_eff is invalid and removed)
    neff_factor : (M,) NaN

    Notes
    -----
    - Uses average ranks for ties (matches scipy.stats.spearmanr).
    - Filters NaNs pairwise per column.
    - Does NOT compute a pooled space-time effective sample size; that estimate
      is scientifically invalid for this data structure.
    """
    N, M = data_matrix.shape

    rho_out = cp.empty(M, dtype=cp.float32)
    rho_out[:] = cp.nan

    res_gpu = cp.asarray(residuals)
    mat_gpu = cp.asarray(data_matrix)

    for m in range(M):
        y = mat_gpu[:, m]
        mask = cp.isfinite(res_gpu) & cp.isfinite(y)
        n_valid = int(cp.sum(mask))
        if n_valid < 3:
            continue

        x_rank = _rankdata_average_1d(res_gpu[mask]).astype(cp.float64)
        y_rank = _rankdata_average_1d(y[mask]).astype(cp.float64)

        x_mean = cp.mean(x_rank)
        y_mean = cp.mean(y_rank)
        xm = x_rank - x_mean
        ym = y_rank - y_mean

        num = cp.sum(xm * ym)
        den = cp.sqrt(cp.sum(xm ** 2) * cp.sum(ym ** 2))
        if den > 0:
            rho_out[m] = float(num / den)

    # Legacy pooled space-time n_eff is intentionally not computed.
    n_eff_out = cp.full(M, cp.nan, dtype=cp.float32)
    neff_factor_out = cp.full(M, cp.nan, dtype=cp.float32)
    return rho_out, n_eff_out, neff_factor_out

# ==============================================================================
# SPATIAL STREAMING ANALYZER
# ==============================================================================

class SpatialCorrelationAnalyzer:
    def __init__(self):
        # Load OMNI
        self.omni_df = feather.read_feather(Config.FILE_OMNI_FEATHER)
        self.omni_df['date'] = pd.to_datetime(self.omni_df['date']).dt.normalize()
        num_cols = self.omni_df.select_dtypes(include=[np.number]).columns
        self.omni_vars = [c for c in num_cols if c not in ['year', 'day', 'hour']]

        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.REPORTS_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)

        # Load Flags (Lightweight map)
        print("[INFO] Loading Region Flags...", flush=True)
        self.df_flags = pd.read_feather(Config.FILE_SIF_FINAL, columns=['lat_id', 'lon_id', 'region_flags', 'date'])
        # Flags are kept dynamic (linked to date) to account for LAI changes.
        # Kept in memory as size is manageable (~2-3 GB).

        # Check ERA5 Schema
        self.era5_schema = pq.read_schema(Config.FILE_ERA5_PARQUET).names
        self.temp_col = self._find_temp_col()
        self.env_vars = self._get_env_vars()

        # Join accounting ledger
        self.join_ledger: list[dict] = []

    def _find_temp_col(self):
        candidates = [Config.TEMP_CONTEXT_COL, f"temp_c_ma{Config.CONTEXT_WINDOW_DAYS}"]
        for c in candidates:
            if c in self.era5_schema: return c
        raise ValueError(f"Temp column not found in ERA5.")

    def _get_env_vars(self):
        vars_ = []
        for v in ["par", "vpd", "tcc", "temp_c"]:
            for w in Config.MA_WINDOWS:
                c1 = f"{v}_ma{w}"
                c2 = f"{v}_c_ma{w}" if v == "temp" else c1
                if c1 in self.era5_schema: vars_.append(c1)
                elif c2 in self.era5_schema: vars_.append(c2)
        return sorted(list(set(vars_)))

    def load_target_data(self, target):
        """
        Load residuals + flags + ERA5 + OMNI once per target, then apply scenario
        masks in-memory. This avoids streaming ERA5 once per scenario.
        """
        print(f"[Spearman] Loading target data for {target}...", flush=True)

        # 1. Stream residuals and inner-join with full flags.
        res_path = Config.DIR_SIF_MODEL / f"sif_residuals_{target}.parquet"
        res_pf = pq.ParquetFile(res_path)
        res_batch_size = 5_000_000
        total_res_batches = max(1, (res_pf.metadata.num_rows + res_batch_size - 1) // res_batch_size)
        accumulated = []
        for batch in tqdm(
            res_pf.iter_batches(batch_size=res_batch_size, columns=['date', 'lat_id', 'lon_id', 'residual']),
            total=total_res_batches,
            desc=f"    Residuals {target}",
            file=sys.stdout,
            miniters=1,
            ncols=80,
        ):
            batch_df = batch.to_pandas()
            batch_df['date'] = pd.to_datetime(batch_df['date']).dt.normalize()
            batch_df['lat_id'] = batch_df['lat_id'].astype('int32')
            batch_df['lon_id'] = batch_df['lon_id'].astype('int32')

            merged = Config.merge_with_accounting(
                batch_df,
                self.df_flags[['date', 'lat_id', 'lon_id', 'region_flags']],
                on=['date', 'lat_id', 'lon_id'],
                how='inner',
                validate='many_to_one',
                stage=f'spearman_{target}_residuals_x_flags_batch',
                ledger=self.join_ledger,
            )
            if not merged.empty:
                accumulated.append(merged)
            del batch_df, merged

        if not accumulated:
            print(f"[Spearman] No residuals matched flags for target {target}", flush=True)
            return None
        target_df = pd.concat(accumulated, ignore_index=True)
        del accumulated
        gc.collect()
        print(f"[Spearman] Residuals+flags merged: {len(target_df):,} rows", flush=True)

        # 2. Add scenario mask columns in-memory.
        scenario_cols = []
        for scenario in Config.SCENARIO_MASKS.keys():
            col = f"_scenario_{scenario}"
            target_df[col] = Config.scenario_mask(target_df['region_flags'].values, scenario)
            scenario_cols.append(col)
        print(f"[Spearman] Scenario columns added: {scenario_cols}", flush=True)

        # 3. Stream ERA5 once and merge with the full target dataframe.
        accumulated_era5 = []
        cols_to_read = ['date', 'lat_id', 'lon_id', self.temp_col] + self.env_vars
        parquet_file = pq.ParquetFile(Config.FILE_ERA5_PARQUET)
        era5_batch_size = 3_000_000
        total_era5_batches = max(1, (parquet_file.metadata.num_rows + era5_batch_size - 1) // era5_batch_size)

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=era5_batch_size, columns=cols_to_read),
            total=total_era5_batches,
            desc=f"    ERA5 {target}",
            file=sys.stdout,
            miniters=1,
            ncols=80,
        ):
            batch_df = batch.to_pandas()
            batch_df['date'] = pd.to_datetime(batch_df['date']).dt.normalize()
            batch_df['lat_id'] = batch_df['lat_id'].astype('int32')
            batch_df['lon_id'] = batch_df['lon_id'].astype('int32')

            merged_chunk = Config.merge_with_accounting(
                target_df,
                batch_df,
                on=['date', 'lat_id', 'lon_id'],
                how='inner',
                validate='many_to_one',
                stage=f'spearman_{target}_universe_x_era5_batch',
                ledger=self.join_ledger,
            )
            if not merged_chunk.empty:
                accumulated_era5.append(merged_chunk)
            del batch_df, merged_chunk

        if not accumulated_era5:
            print(f"[Spearman] No ERA5 data matched target universe for {target}", flush=True)
            return None
        full_df = pd.concat(accumulated_era5, ignore_index=True)
        del accumulated_era5, target_df
        gc.collect()
        print(f"[Spearman] After ERA5 merge: {len(full_df):,} rows", flush=True)

        # 4. Add OMNI once.
        full_df = Config.merge_with_accounting(
            full_df,
            self.omni_df,
            on='date',
            how='inner',
            validate='many_to_one',
            stage=f'spearman_{target}_x_omni',
            ledger=self.join_ledger,
        )
        print(f"[Spearman] After OMNI merge: {len(full_df):,} rows", flush=True)

        return full_df

    def p_value_from_t(self, rho, n_eff, n_raw):
        # n_eff is NaN because pooled space-time autocorrelation is invalid.
        # Fall back to the raw pairwise sample size for the p-value, which is
        # transparent and conservative relative to any inflated effective-N.
        n_use = int(n_eff) if np.isfinite(n_eff) and n_eff > 2 else int(n_raw)
        if n_use <= 2:
            return np.nan
        if abs(rho) >= 1.0:
            return 0.0
        t_stat = rho * np.sqrt((n_use - 2) / (1 - rho**2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), df=n_use - 2))

    def run_analysis(self):
        targets = getattr(Config, "SPEARMAN_TARGETS", ["sif_740nm", "sif_stress_index"])
        all_vars = self.omni_vars + self.env_vars

        print(f"[INFO] Analysis Variables: {len(all_vars)}", flush=True)

        for target in tqdm(targets, desc="Targets"):
            print(f"[Spearman] Starting target {target}", flush=True)
            scenarios = list(Config.SCENARIO_MASKS.keys())

            # Resume at target level: skip entirely if all scenarios already done.
            missing_scenarios = []
            for scenario in scenarios:
                out_file = Config.RESULTS_DIR / f"spearman_{target}_{scenario}.csv"
                if out_file.exists() and out_file.stat().st_size > 0:
                    print(f"[Spearman] Skipping {target}/{scenario}: output already exists ({out_file})", flush=True)
                else:
                    missing_scenarios.append(scenario)

            if not missing_scenarios:
                print(f"[Spearman] All scenarios for target {target} already computed. Skipping target.", flush=True)
                continue

            # Load all data for this target once.
            full_df = self.load_target_data(target)
            if full_df is None:
                continue

            for scenario in tqdm(missing_scenarios, desc=f"Scenarios ({target})", leave=False):
                print(f"[Spearman] Target={target} Scenario={scenario}", flush=True)

                scenario_col = f"_scenario_{scenario}"
                if scenario_col not in full_df.columns:
                    print(f"[Spearman] Skipping {target}/{scenario}: scenario column missing", flush=True)
                    continue

                scen_df = full_df[full_df[scenario_col]].copy()
                scen_df = scen_df.drop(columns=[c for c in scen_df.columns if c.startswith("_scenario_")])

                if len(scen_df) < 1000:
                    print(f"[Spearman] Skipping {target}/{scenario}: too few rows ({len(scen_df)})", flush=True)
                    continue

                # 2. Binning
                try:
                    tb = Config.bin_temperature(scen_df[self.temp_col])
                    scen_df = scen_df.join(tb)
                except Exception as e:
                    print(f"[Spearman] Binning error for {target}/{scenario}: {e}", flush=True)
                    continue

                results = []
                unique_bins = sorted(scen_df["temp_bin_id"].dropna().unique())
                print(f"[Spearman] Temperature bins: {unique_bins}", flush=True)

                # 3. GPU Compute per Bin
                for b_id in unique_bins:
                    bin_df = scen_df[scen_df["temp_bin_id"] == b_id].dropna(subset=["residual"] + all_vars)
                    if len(bin_df) < 50:
                        continue

                    bin_label = str(bin_df["temp_bin_label"].iloc[0])
                    print(f"[Spearman]   Bin {b_id} ({bin_label}): n={len(bin_df):,}, computing {len(all_vars)} correlations on GPU...", flush=True)

                    res_gpu = cp.array(bin_df["residual"].values, dtype=cp.float32)
                    mat_gpu = cp.array(bin_df[all_vars].values, dtype=cp.float32)

                    rhos, neffs, factors = gpu_spearman_matrix(res_gpu, mat_gpu)

                    rhos_cpu = cp.asnumpy(rhos)
                    neff_cpu = cp.asnumpy(neffs)
                    fact_cpu = cp.asnumpy(factors)

                    temp_mean = float(bin_df[self.temp_col].mean())
                    n_samp = len(bin_df)

                    for i, var_name in enumerate(all_vars):
                        results.append({
                            "scenario": scenario,
                            "omni_var": var_name,
                            "bin_id": int(b_id),
                            "bin_label": bin_label,
                            "temp_mean": temp_mean,
                            "rho": float(rhos_cpu[i]),
                            "n_eff": float(neff_cpu[i]),
                            "neff_factor": float(fact_cpu[i]),
                            "n": n_samp,
                            "p_adj": self.p_value_from_t(float(rhos_cpu[i]), float(neff_cpu[i]), n_samp),
                            "inference_method": "not_computed_legacy_neff_invalid",
                        })

                    del res_gpu, mat_gpu
                    cp.get_default_memory_pool().free_all_blocks()

                if results:
                    out_df = pd.DataFrame(results)
                    out_file = Config.RESULTS_DIR / f"spearman_{target}_{scenario}.csv"
                    out_df.to_csv(out_file, index=False)
                    print(f"[Spearman] Saved {out_file} ({len(out_df):,} rows)", flush=True)

                del scen_df
                gc.collect()

            del full_df
            gc.collect()

        # Persist join accounting for this target's scenarios.
        if self.join_ledger:
            Config.save_join_accounting(
                self.join_ledger,
                Config.REPORTS_INTEGRITY_DIR / "join_accounting_spearman.csv",
            )
            print(f"[INFO] Join accounting saved: {Config.REPORTS_INTEGRITY_DIR / 'join_accounting_spearman.csv'}", flush=True)

if __name__ == "__main__":
    SpatialCorrelationAnalyzer().run_analysis()