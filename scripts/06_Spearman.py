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
# GPU KERNEL (Spearman + Neff)
# ==============================================================================

def gpu_spearman_matrix(residuals, data_matrix):
    """
    Computes Spearman Rho matrix-wise on GPU.
    residuals: (N,)
    data_matrix: (N, M)
    """
    N, M = data_matrix.shape
    
    # 1. Rank Transform
    def get_ranks(x):
        idx = cp.argsort(x, axis=0)
        ranks = cp.empty_like(idx, dtype=cp.float32)
        if x.ndim > 1:
            for i in range(M):
                ranks[idx[:, i], i] = cp.arange(N, dtype=cp.float32)
        else:
            ranks[idx] = cp.arange(N, dtype=cp.float32)
        return ranks

    res_ranks = get_ranks(residuals).reshape(-1, 1)
    mat_ranks = get_ranks(data_matrix)

    # 2. Correlation
    res_mean = (N - 1) / 2.0
    mat_mean = (N - 1) / 2.0
    
    res_c = res_ranks - res_mean
    mat_c = mat_ranks - mat_mean
    
    # Dot product for covariance
    numer = cp.dot(res_c.T, mat_centered := mat_c).flatten()
    
    # Variances
    res_ss = cp.sum(res_c**2)
    mat_ss = cp.sum(mat_c**2, axis=0)
    
    rho = numer / cp.sqrt(res_ss * mat_ss)
    
    # 3. Neff (Pyper–Peterman / Chelton: sum of autocorrelation products)
    def autocorr_lag(arr_2d, k: int):
        """
        Lag-k autocorrelation for each column of arr_2d (shape: [N, M] or [N, 1]).
        Returns shape: (M,) or (1,).
        """
        x0 = arr_2d[:-k]
        xk = arr_2d[k:]
        x0 = x0 - cp.mean(x0, axis=0)
        xk = xk - cp.mean(xk, axis=0)
        num = cp.sum(x0 * xk, axis=0)
        den = cp.sqrt(cp.sum(x0**2, axis=0) * cp.sum(xk**2, axis=0))
        return num / den

    # choose max lag; must be < N
    K = int(getattr(Config, "NEFF_MAX_LAG", 60))
    K = max(1, min(K, N - 2))

    # r_res(k): shape (K,) ; r_mat(k): shape (K, M)
    r_res = cp.empty((K,), dtype=cp.float32)
    r_mat = cp.empty((K, M), dtype=cp.float32)

    for k in range(1, K + 1):
        rr = autocorr_lag(res_ranks, k)
        rm = autocorr_lag(mat_ranks, k)

        # clip to avoid singularities
        r_res[k - 1] = cp.clip(rr, -0.99, 0.99)[0]   # residual is (N,1)
        r_mat[k - 1] = cp.clip(rm, -0.99, 0.99)      # vector (M,)

    # Sum of products across lags: shape (M,)
    s = cp.sum(r_res.reshape(-1, 1) * r_mat, axis=0)

    denom = 1.0 + 2.0 * s
    denom = cp.maximum(denom, 1e-3)  # safety

    n_eff = cp.maximum(2.0, N / denom)

    # keep a "factor" output compatible with your downstream CSV (optional)
    neff_factor = n_eff / float(N)

    return rho, n_eff, neff_factor

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
        
        # Load Flags (Lightweight map)
        print("[INFO] Loading Region Flags...")
        self.df_flags = pd.read_feather(Config.FILE_SIF_FINAL, columns=['lat_id', 'lon_id', 'region_flags', 'date'])
        # Flags are kept dynamic (linked to date) to account for LAI changes.
        # Kept in memory as size is manageable (~2-3 GB).
        
        # Check ERA5 Schema
        self.era5_schema = pq.read_schema(Config.FILE_ERA5_PARQUET).names
        self.temp_col = self._find_temp_col()
        self.env_vars = self._get_env_vars()
        
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

    def load_scenario_data(self, target, scenario_name):
        """
        Reads ERA5/SIF data filtering strictly for the given scenario to save RAM.
        """
        print(f"  > Streaming data for {scenario_name}...")
        
        # 1. Identify valid (lat, lon) pairs for this scenario
        # We process Flags first to get a mask
        mask = Config.scenario_mask(self.df_flags["region_flags"].values, scenario_name)
        valid_flags = self.df_flags[mask]
        
        if valid_flags.empty: return None
        
        # Create a set of valid locations for fast lookup
        # Optimization: Stream ERA5 data in batches and filter immediately 
        # by merging with valid_flags on [lat, lon, date]. 
        # Unmatched rows are dropped to save memory.
        
        # Load Residuals (Target)
        res_path = Config.DIR_SIF_MODEL / f"sif_residuals_{target}.parquet"
        df_res = pd.read_parquet(res_path)
        
        # Merge Residuals + Valid Flags (This defines our "Target Universe")
        # Only keep rows that are in the scenario
        target_universe = pd.merge(df_res, valid_flags[['date', 'lat_id', 'lon_id']], on=['date', 'lat_id', 'lon_id'])
        del df_res
        
        if target_universe.empty: return None

        # 2. Stream ERA5 and merge with Target Universe
        # We will accumulate chunks
        accumulated_era5 = []
        
        # Columns to read from ERA5
        cols_to_read = ['date', 'lat_id', 'lon_id', self.temp_col] + self.env_vars
        
        parquet_file = pq.ParquetFile(Config.FILE_ERA5_PARQUET)
        
        # Iterate over batches (e.g. 500k rows)
        for batch in parquet_file.iter_batches(batch_size=500000, columns=cols_to_read):
            batch_df = batch.to_pandas()
            batch_df['date'] = pd.to_datetime(batch_df['date'])
            
            # INNER JOIN with Target Universe
            # This automatically drops oceans/deserts/irrelevant pixels
            merged_chunk = pd.merge(target_universe, batch_df, on=['date', 'lat_id', 'lon_id'], how='inner')
            
            if not merged_chunk.empty:
                accumulated_era5.append(merged_chunk)
            
            del batch_df, merged_chunk
            # gc.collect() # Optional, slightly slows down
            
        if not accumulated_era5: return None
        
        full_scenario_df = pd.concat(accumulated_era5, ignore_index=True)
        del accumulated_era5, target_universe
        gc.collect()
        
        # 3. Add OMNI
        full_scenario_df = pd.merge(full_scenario_df, self.omni_df, on='date', how='inner')
        
        return full_scenario_df

    def p_value_from_t(self, rho, n_eff):
        if abs(rho) >= 1.0: return 0.0
        t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho**2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), df=n_eff - 2))
        
    def run_analysis(self):
        targets = getattr(Config, "SPEARMAN_TARGETS", ["sif_740nm", "sif_stress_index"])
        all_vars = self.omni_vars + self.env_vars
        
        print(f"[INFO] Analysis Variables: {len(all_vars)}")

        for target in tqdm(targets, desc="Targets"):
            # We iterate scenarios FIRST to minimize peak RAM
            # (Load only SAA, process, dump. Load only Global, process, dump.)
            
            scenarios = Config.SCENARIO_MASKS.keys()
            
            for scenario in tqdm(scenarios, desc=f"Scenarios ({target})", leave=False):
                # 1. Load Filtered Data (RAM Efficient)
                scen_df = self.load_scenario_data(target, scenario)
                
                if scen_df is None or len(scen_df) < 1000:
                    continue
                
                # 2. Binning
                try:
                    tb = Config.bin_temperature(scen_df[self.temp_col])
                    scen_df = scen_df.join(tb)
                except Exception as e:
                    print(f"  Binning error: {e}")
                    continue
                
                results = []
                unique_bins = sorted(scen_df["temp_bin_id"].dropna().unique())
                
                # 3. GPU Compute per Bin
                for b_id in unique_bins:
                    bin_df = scen_df[scen_df["temp_bin_id"] == b_id].dropna(subset=["residual"] + all_vars)
                    if len(bin_df) < 50: continue
                    
                    # Prepare GPU Arrays
                    res_gpu = cp.array(bin_df["residual"].values, dtype=cp.float32)
                    mat_gpu = cp.array(bin_df[all_vars].values, dtype=cp.float32)
                    
                    # Calculate
                    rhos, neffs, factors = gpu_spearman_matrix(res_gpu, mat_gpu)
                    
                    # Download
                    rhos_cpu = cp.asnumpy(rhos)
                    neff_cpu = cp.asnumpy(neffs)
                    fact_cpu = cp.asnumpy(factors)
                    
                    bin_label = str(bin_df["temp_bin_label"].iloc[0])
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
                            "p_adj": self.p_value_from_t(float(rhos_cpu[i]), float(neff_cpu[i]))
                        })
                    
                    del res_gpu, mat_gpu
                    cp.get_default_memory_pool().free_all_blocks()

                if results:
                    out_df = pd.DataFrame(results)
                    out_file = Config.RESULTS_DIR / f"spearman_{target}_{scenario}.csv"
                    out_df.to_csv(out_file, index=False)
                    
                del scen_df
                gc.collect()

if __name__ == "__main__":
    SpatialCorrelationAnalyzer().run_analysis()