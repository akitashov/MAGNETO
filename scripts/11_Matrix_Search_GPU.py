#!/usr/bin/env python3
"""
GPU-Accelerated Multivariate Matrix Search for MAGNETO.

This script performs an exhaustive, physically-grounded search for dependencies 
between Solar-Induced Fluorescence (SIF) residuals and Space Weather indices (SII), 
while rigorously controlling for immediate environmental drivers (PAR, VPD).

LOGIC:
1.  **Data Integration & Synchronization**:
    - Loads SIF Residuals (generated in Step 05) which represent vegetation anomalies 
      after removing seasonal cycles.
    - Filters data by target wavelength (e.g., 740nm, 757nm) and specific geographic 
      scenarios (e.g., 'Global_High_LAI', 'South_Atlantic_Anomaly').
    - Streams ERA5 environmental data (PAR, VPD, Temperature) using memory-efficient 
      batch processing to handle terabyte-scale datasets without OOM errors.
    - Merges these datasets with OMNI2 geomagnetic indices (SII) on a daily basis.

2.  **Multivariate Experimental Design**:
    - Unlike simple correlation analysis, this script constructs a full Multivariate 
      Ordinary Least Squares (OLS) model for every combination of time lags:
      
      $$ SIF_{resid}(t) = \beta_0 + \beta_{SII} \cdot SII(t-w_1) + \beta_{PAR} \cdot PAR(t-w_2) + \beta_{VPD} \cdot VPD(t-w_3) + \epsilon $$
      
    - This design isolates the *unique* partial contribution of SII ($\beta_{SII}$) 
      that cannot be explained by light (PAR) or water stress (VPD).

3.  **GPU Acceleration (CuPy)**:
    - The combinatorial space is massive (e.g., 30 SII windows × 30 PAR windows × 30 VPD windows = 27,000 regressions per bin).
    - The script utilizes CUDA kernels (via CuPy) to solve the Normal Equations $(X^T X)^{-1} X^T y$ 
      for thousands of models in parallel, reducing computation time from days to minutes.

4.  **Statistical Correction (Chelton Method)**:
    - Geophysical time series are often autocorrelated ("red noise"), which inflates standard significance tests.
    - The script calculates the Effective Sample Size ($N_{eff}$) based on the lag-1 autocorrelation 
      of the residuals (Pyper & Peterman, 1998).
    - T-statistics and P-values are adjusted using $N_{eff}$ degrees of freedom, providing 
      conservative and robust evidence for the paper.

CONFIGURATION:
--------------
- Targets: Config.SPEARMAN_TARGETS
- Scenarios: Config.SCENARIO_MASKS
- Windows: Config.MATRIX_SII_RANGE, Config.MATRIX_ENV_RANGE

OUTPUT FILE DESCRIPTION:
------------------------
Files are saved as: `results/matrix_search_{target}_{scenario}.csv`
Each row represents a specific model configuration. Columns include:
- `bin_label`: Temperature regime (e.g., 'Optimum', 'Heat_Stress').
- `sii_window`, `par_window`, `vpd_window`: Moving average window sizes (days) for each driver.
- `beta_sii`, `beta_par`, `beta_vpd`: Partial regression coefficients (slope).
- `t_sii`, `t_par`, `t_vpd`: T-statistics for the coefficients.
- `p_sii`, `p_par`, `p_vpd`: P-values (two-tailed), adjusted for N_eff.
- `n`: Raw sample size.
- `n_eff`: Effective sample size after autocorrelation correction.
- `rss`: Residual Sum of Squares (model fit quality).
"""


import pandas as pd
import numpy as np
import cupy as cp
import pyarrow.parquet as pq
import warnings
import gc
from scipy import stats
from pathlib import Path
from tqdm.auto import tqdm
from _Common import Config

warnings.filterwarnings('ignore')

# ==============================================================================
# GPU KERNEL
# ==============================================================================

def solve_ols_gpu(y_vec, X_mat):
    """
    Solves (X.T X)^-1 X.T y on GPU.
    Returns betas, t-stats, rss.
    """
    try:
        y_gpu = cp.array(y_vec, dtype=cp.float32)
        X_gpu = cp.array(X_mat, dtype=cp.float32)
        
        N, K = X_mat.shape
        
        xtx = X_gpu.T @ X_gpu
        xty = X_gpu.T @ y_gpu
        
        if cp.linalg.det(xtx) == 0:
            return None

        B = cp.linalg.solve(xtx, xty)
        
        residuals = y_gpu - (X_gpu @ B)
        rss = cp.sum(residuals**2)
        
        df = N - K
        if df <= 0: return None
        
        var_res = rss / df
        inv_xtx = cp.linalg.inv(xtx)
        se = cp.sqrt(cp.diag(inv_xtx) * var_res)
        t_stats = B / se
        
        return {
            'beta': cp.asnumpy(B),
            't': cp.asnumpy(t_stats),
            'rss': float(rss),
            'n': N
        }
    except Exception:
        return None

# ==============================================================================
# DATA ENGINE
# ==============================================================================

class MatrixSearchEngine:
    def __init__(self):
        # --- EXECUTION CONFIGURATION ---
        self.SKIP_EXISTING = True 
        # -------------------------------

        self.era5_path = Config.FILE_ERA5_PARQUET
        self.sif_path = Config.DIR_SIF_MODEL
        self.omni_path = Config.FILE_OMNI_FEATHER
        
        self.sii_wins = getattr(Config, 'MATRIX_SII_RANGE', list(range(1, 31))) 
        self.env_wins = getattr(Config, 'MATRIX_ENV_RANGE', list(range(1, 31)))
        
        if self.omni_path.exists():
            self.df_omni = pd.read_feather(self.omni_path)
        else:
            raise FileNotFoundError(f"OMNI file missing: {self.omni_path}")

    def _is_file_valid(self, path):
        """
        Checks if a result file is valid (exists, not empty, AND contains numeric data).
        Returns True only if the file is 'healthy'.
        """
        if not path.exists() or path.stat().st_size < 1000:
            return False
        
        try:
            # Peek into the file (first 50 rows)
            df = pd.read_csv(path, nrows=50)
            
            # Critical check: Does 't_sii' contain any valid numbers?
            if 't_sii' not in df.columns:
                return False
            
            # If all t_sii are NaN, the file is trash (failed regression)
            if df['t_sii'].isna().all():
                return False
                
            return True
        except Exception:
            return False

    def _get_era5_data_streamed(self, keys_df):
        """Reads ERA5 in batches to avoid OOM."""
        cols = ['date', 'lat_id', 'lon_id', 'temp_c_ma10']
        cols += [f'par_ma{w}' for w in self.env_wins]
        cols += [f'vpd_ma{w}' for w in self.env_wins]
        cols = list(set(cols))

        accumulated = []
        pf = pq.ParquetFile(self.era5_path)
        available_cols = [c for c in cols if c in pf.schema.names]
        
        keys_df = keys_df.copy()
        keys_df['date'] = pd.to_datetime(keys_df['date'])
        keys_df['lat_id'] = keys_df['lat_id'].astype('int16')
        keys_df['lon_id'] = keys_df['lon_id'].astype('int16')

        for batch in pf.iter_batches(batch_size=200000, columns=available_cols):
            b_df = batch.to_pandas()
            b_df['date'] = pd.to_datetime(b_df['date'])
            b_df['lat_id'] = b_df['lat_id'].astype('int16')
            b_df['lon_id'] = b_df['lon_id'].astype('int16')
            
            merged = pd.merge(keys_df, b_df, on=['date', 'lat_id', 'lon_id'], how='inner')
            if not merged.empty:
                accumulated.append(merged)
        
        if not accumulated:
            return pd.DataFrame()
            
        return pd.concat(accumulated, ignore_index=True)

    def run(self):
        mode_str = "RESUME (Smart Validation)" if self.SKIP_EXISTING else "OVERWRITE (Full Run)"
        print(f"[INFO] Starting Memory-Safe Multivariate Matrix Search. Mode: {mode_str}")
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        for target in Config.SPEARMAN_TARGETS:
            res_file = self.sif_path / f"sif_residuals_{target}.parquet"
            if not res_file.exists(): continue
            
            # --- TARGET LEVEL SMART SKIP ---
            if self.SKIP_EXISTING:
                all_valid = True
                for sc in Config.SCENARIO_MASKS.keys():
                    out_check = Config.RESULTS_DIR / f"matrix_search_{target}_{sc}.csv"
                    if not self._is_file_valid(out_check):
                        all_valid = False
                        break
                
                if all_valid:
                    print(f"[SKIP] Target {target} already completed and verified.")
                    continue
            # -------------------------------

            print(f"\n>>> Processing Target: {target}")
            df_res = pd.read_parquet(res_file)
            df_res['date'] = pd.to_datetime(df_res['date'])
            
            flags = pd.read_feather(Config.FILE_SIF_FINAL, columns=['lat_id', 'lon_id', 'region_flags', 'date'])
            flags['date'] = pd.to_datetime(flags['date'])
            
            for scenario in Config.SCENARIO_MASKS.keys():
                out_path = Config.RESULTS_DIR / f"matrix_search_{target}_{scenario}.csv"
                
                # --- SCENARIO LEVEL SMART SKIP ---
                if self.SKIP_EXISTING:
                    if self._is_file_valid(out_path):
                        print(f"  [SKIP] Scenario {scenario} exists (Verified).")
                        continue
                    elif out_path.exists():
                        print(f"  [RE-RUN] Scenario {scenario} found but corrupted (NaNs). Recalculating...")
                # ---------------------------------

                # 1. Masking
                mask = Config.scenario_mask(flags['region_flags'].values, scenario)
                scen_flags = flags[mask][['date', 'lat_id', 'lon_id']]
                
                if scen_flags.empty: continue
                
                # 2. Merge SIF + Geo
                skeleton = pd.merge(df_res, scen_flags, on=['date', 'lat_id', 'lon_id'])
                if skeleton.empty: continue
                
                # 3. Merge OMNI
                skeleton = pd.merge(skeleton, self.df_omni, on='date')
                
                # 4. Stream ERA5
                print(f"  Streaming ERA5 for {scenario} ({len(skeleton)} rows)...")
                keys = skeleton[['date', 'lat_id', 'lon_id']].drop_duplicates()
                df_env = self._get_era5_data_streamed(keys)
                
                if df_env.empty: 
                    print("    [WARN] No ERA5 data found.")
                    continue
                
                df_full = pd.merge(skeleton, df_env, on=['date', 'lat_id', 'lon_id'])
                
                # 5. Binning
                if 'temp_c_ma10' not in df_full.columns:
                    print("    [WARN] Temp column missing.")
                    continue
                    
                tb = Config.bin_temperature(df_full['temp_c_ma10'])
                df_full = df_full.join(tb)
                
                # 6. Grid Search
                unique_bins = sorted(df_full['temp_bin_label'].unique())
                results = []
                
                for b_label in unique_bins:
                    bin_df = df_full[df_full['temp_bin_label'] == b_label]
                    
                    # --- SANITIZE (Removing NaNs/Infs) ---
                    valid_mask = np.isfinite(bin_df['residual'])
                    bin_df = bin_df[valid_mask]
                    # -------------------------------------

                    if len(bin_df) < 50: continue
                    
                    y = bin_df['residual'].values
                    N = len(y)
                    
                    if N > 10:
                        r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
                        if np.isnan(r1): r1 = 0.5
                        n_eff = N * (1 - r1) / (1 + r1)
                    else:
                        n_eff = N
                    
                    avail_cols = set(bin_df.columns)
                    valid_sii = [w for w in self.sii_wins if f'sii_mean_ma{w}' in avail_cols]
                    valid_par = [w for w in self.env_wins if f'par_ma{w}' in avail_cols]
                    valid_vpd = [w for w in self.env_wins if f'vpd_ma{w}' in avail_cols]

                    sii_data = {w: bin_df[f'sii_mean_ma{w}'].values for w in valid_sii}
                    par_data = {w: bin_df[f'par_ma{w}'].values for w in valid_par}
                    vpd_data = {w: bin_df[f'vpd_ma{w}'].values for w in valid_vpd}
                    
                    ones = np.ones(N)

                    for w_sii in valid_sii:
                        v_sii = sii_data[w_sii]
                        for w_par in valid_par:
                            v_par = par_data[w_par]
                            for w_vpd in valid_vpd:
                                v_vpd = vpd_data[w_vpd]
                                
                                X = np.column_stack([v_sii, v_par, v_vpd, ones])
                                res = solve_ols_gpu(y, X)
                                
                                if res:
                                    dof = max(1, n_eff - 4)
                                    p_vals = 2 * (1 - stats.t.cdf(np.abs(res['t']), df=dof))
                                    
                                    results.append({
                                        "bin_label": b_label,
                                        "sii_window": w_sii,
                                        "par_window": w_par,
                                        "vpd_window": w_vpd,
                                        "n": N,
                                        "n_eff": n_eff,
                                        "rss": res['rss'],
                                        "beta_sii": res['beta'][0], "t_sii": res['t'][0], "p_sii": p_vals[0],
                                        "beta_par": res['beta'][1], "t_par": res['t'][1], "p_par": p_vals[1],
                                        "beta_vpd": res['beta'][2], "t_vpd": res['t'][2], "p_vpd": p_vals[2]
                                    })

                if results:
                    pd.DataFrame(results).to_csv(out_path, index=False)
                    print(f"    Saved {out_path.name} ({len(results)} models)")
            
            gc.collect()

if __name__ == "__main__":
    MatrixSearchEngine().run()