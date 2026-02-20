#!/usr/bin/env python3
"""
Mechanism Verification & Screening.

PART 1: Mechanism Check (Complex)
- Checks independence of Space Weather (SII) from Earth Weather (PAR, VPD, Temp).
- Uses Bin-wise analysis (Stratification by Temperature) to remove seasonal confounding.
- Uses Fisher's Z-transformation for correct aggregation.
- Fixes ERA5 date corruption on the fly.

PART 2: Global Screening (Simple)
- Checks fast correlation between SII and Global SIF to catch obvious signals.

Output:
- results/spearman_mechanism_SII_vs_ENV.csv (Deep check)
- results/screening_overview_global.csv (Fast check)
"""

import pandas as pd
import numpy as np
from scipy import stats
import pyarrow.parquet as pq
import pyarrow.feather as feather
import warnings
from tqdm.auto import tqdm
import gc
from _Common import Config

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ERA5_ENV_FILE = Config.FILE_ERA5_PARQUET
OMNI_FILE = Config.FILE_OMNI_FEATHER
SIF_FLAGS_FILE = Config.FILE_SIF_FINAL
TEMP_COLUMN = Config.TEMP_CONTEXT_COL # e.g. temp_c_ma10
# Windows to check for mechanism (subset of full range to save time)
WINDOWS_TO_CHECK = getattr(Config, 'MA_WINDOWS', [1, 5, 10, 30]) 
ENV_VARS_BASE = ["par", "vpd", "tcc"] # Variables to check against SII
MIN_SAMPLES = 100
MECH_SCENARIOS = list(Config.SCENARIO_MASKS.keys())

# ==============================================================================
# HELPERS
# ==============================================================================

def smart_date_conversion(series):
    """Safely converts numeric or object series to datetime with Million-Fix."""
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.max().year < 1980: series = series.astype(np.int64)
    
    if pd.api.types.is_numeric_dtype(series):
        max_val = series.max()
        if 1_000_000 < max_val < 5_000_000: 
            return pd.to_datetime(series * 1_000_000, unit='ms', origin='unix')
        elif max_val < 100_000: return pd.to_datetime(series, unit='D', origin='unix')
        elif max_val < 5e11: return pd.to_datetime(series, unit='s', origin='unix')
        elif max_val < 5e14: return pd.to_datetime(series, unit='ms', origin='unix')
        else: return pd.to_datetime(series, unit='ns', origin='unix')
    return pd.to_datetime(series)

def calculate_neff_factor(series_x, series_y):
    def lag1(s):
        valid = s[np.isfinite(s)]
        if len(valid) < 5: return 0.0
        return np.corrcoef(valid[:-1], valid[1:])[0, 1]
    r1_x = np.clip(lag1(series_x), -0.99, 0.99)
    r1_y = np.clip(lag1(series_y), -0.99, 0.99)
    prod = r1_x * r1_y
    return (1 - prod) / (1 + prod)

def calculate_stats(x, y, neff_factor):
    n_raw = len(x)
    rho, _ = stats.spearmanr(x, y)
    n_eff = max(2, n_raw * neff_factor)
    return rho, n_eff, n_raw

def p_value_from_t(rho, n_eff):
    if abs(rho) >= 1.0: return 0.0
    if n_eff <= 2: return 1.0
    t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho**2))
    return 2 * (1 - stats.t.cdf(abs(t_stat), df=n_eff - 2))

def get_temp_col(schema_names):
    if TEMP_COLUMN in schema_names: return TEMP_COLUMN
    for alt in ["temp_c", "t2m", "temp", "temp_c_ma10"]:
        if alt in schema_names: return alt
    return None

def stream_filtered_era5(target_cols, flags_df):
    accumulated = []
    pq_file = pq.ParquetFile(ERA5_ENV_FILE)
    
    available = [c for c in target_cols if c in pq_file.schema.names]
    batch_iter = pq_file.iter_batches(batch_size=2_000_000, columns=available)
    flags_keys = flags_df[['date', 'lat_id', 'lon_id']]
    
    for batch in tqdm(batch_iter, desc="Scanning ERA5", total=pq_file.num_row_groups, leave=False):
        batch_df = batch.to_pandas()
        batch_df['date'] = smart_date_conversion(batch_df['date']).dt.normalize()
        
        merged_batch = pd.merge(batch_df, flags_keys, on=['date', 'lat_id', 'lon_id'], how='inner')
        if not merged_batch.empty:
            accumulated.append(merged_batch)
        del batch_df, merged_batch
        
    if not accumulated: return pd.DataFrame()
    return pd.concat(accumulated, ignore_index=True)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. LOAD METADATA
    print("[INFO] Loading Metadata...")
    omni = feather.read_feather(OMNI_FILE)
    omni["date"] = pd.to_datetime(omni["date"]).dt.normalize()
    
    # Load SIF Flags (Used for filtering geography AND for global screening)
    pf_sif = feather.read_table(SIF_FLAGS_FILE)
    sif_cols = pf_sif.column_names
    cols_to_read = ["date", "lat_id", "lon_id", "region_flags"]
    cols_to_read += [c for c in Config.SPEARMAN_TARGETS if c in sif_cols]
    
    flags = pf_sif.select(cols_to_read).to_pandas()
    flags["date"] = pd.to_datetime(flags["date"]).dt.normalize()
    
    # --- PART A: GLOBAL SCREENING (SII vs SIF) ---
    print("\n[PART A] Running Global Screening (SII vs SIF)...")
    daily_sif = flags.groupby('date')[cols_to_read[4:]].mean().reset_index() # Skip geo cols
    merged_sif = pd.merge(daily_sif, omni, on='date', how='inner')
    
    screening_results = []
    sii_driver = 'sii_mean'
    
    for t in Config.SPEARMAN_TARGETS:
        if t in merged_sif.columns and sii_driver in merged_sif.columns:
            r, p = stats.spearmanr(merged_sif[sii_driver], merged_sif[t])
            screening_results.append({
                'target': t, 
                'spearman_r': r, 
                'p_value': p, 
                'n': len(merged_sif)
            })
            
    if screening_results:
        pd.DataFrame(screening_results).to_csv(Config.RESULTS_DIR / "screening_overview_global.csv", index=False)
        print("  Saved screening_overview_global.csv")

    # --- PART B: MECHANISM CHECK (SII vs ENV) ---
    print("\n[PART B] Starting Bin-wise Mechanism Check (SII vs ENV)...")
    schema = pq.read_schema(ERA5_ENV_FILE).names
    temp_col = get_temp_col(schema)
    
    if not temp_col:
        print("  [ERROR] Temp column missing. Skipping mechanism check.")
        return

    mech_results_raw = []
    
    # Batch windows to save RAM
    window_batches = [WINDOWS_TO_CHECK[i:i+5] for i in range(0, len(WINDOWS_TO_CHECK), 5)]
    
    for w_batch in window_batches:
        print(f"  > Processing windows: {w_batch}")
        cols_needed = ['date', 'lat_id', 'lon_id', temp_col]
        active_windows = []
        
        for w in w_batch:
            omni_driver = f"sii_mean_ma{w}"
            if omni_driver in omni.columns:
                w_env_cols = [f"{b}_ma{w}" for b in ENV_VARS_BASE if f"{b}_ma{w}" in schema]
                if w_env_cols:
                    cols_needed.extend(w_env_cols)
                    active_windows.append(w)
        
        cols_needed = list(set(cols_needed))
        
        # 1. Load ERA5
        filtered_era5 = stream_filtered_era5(cols_needed, flags)
        if filtered_era5.empty: continue
        
        # 2. Merge Context
        filtered_era5 = pd.merge(filtered_era5, flags[['date', 'lat_id', 'lon_id', 'region_flags']], on=['date', 'lat_id', 'lon_id'], how='left')
        omni_cols = ['date'] + [f"sii_mean_ma{w}" for w in active_windows]
        df_final = pd.merge(filtered_era5, omni[omni_cols], on='date', how='inner')
        
        del filtered_era5
        gc.collect()
        
        # 3. Calculate Stats
        for w in active_windows:
            omni_driver = f"sii_mean_ma{w}"
            
            for sc in MECH_SCENARIOS:
                mask = Config.scenario_mask(df_final["region_flags"].values, sc)
                sc_data = df_final[mask].copy()
                if len(sc_data) < MIN_SAMPLES: continue
                
                try:
                    sc_data = sc_data.join(Config.bin_temperature(sc_data[temp_col]))
                except: continue
                
                for b_id in sc_data["temp_bin_id"].unique():
                    if pd.isna(b_id): continue
                    bin_data = sc_data[sc_data["temp_bin_id"] == b_id]
                    if len(bin_data) < MIN_SAMPLES: continue
                    
                    for base in ENV_VARS_BASE:
                        env_target = f"{base}_ma{w}"
                        if env_target not in bin_data.columns: continue
                        
                        v = bin_data[[omni_driver, env_target]].dropna()
                        if len(v) < MIN_SAMPLES: continue
                        
                        neff_f = calculate_neff_factor(v[omni_driver].values, v[env_target].values)
                        rho, n_eff, n_raw = calculate_stats(v[omni_driver].values, v[env_target].values, neff_f)
                        
                        mech_results_raw.append({
                            "scenario": sc, "bin_id": int(b_id), 
                            "parameter_1": env_target, "window": w,
                            "rho": rho, "n_eff": n_eff
                        })
        del df_final
        gc.collect()

    # --- AGGREGATE RESULTS (Fisher Z) ---
    if mech_results_raw:
        df_res = pd.DataFrame(mech_results_raw)
        df_res['z'] = np.arctanh(df_res['rho'].clip(-0.99, 0.99))
        df_res['weight'] = np.maximum(df_res['n_eff'] - 3, 1)
        
        final_mech = []
        # Group by Scenario and Variable (Aggregating across bins and windows)
        # Note: Usually we aggregate across bins for a specific window.
        # Here we simplify for the report: Max correlation found? Or Mean?
        # Let's save the detailed aggregation per window/scenario.
        
        groups = df_res.groupby(['scenario', 'parameter_1', 'window'])
        for name, group in groups:
            sum_w = group['weight'].sum()
            mean_z = (group['z'] * group['weight']).sum() / sum_w
            mean_rho = np.tanh(mean_z)
            p_val = p_value_from_t(mean_rho, group['n_eff'].sum())
            
            final_mech.append({
                'scenario': name[0],
                'parameter_1': name[1], # e.g. vpd_ma10
                'window': name[2],
                'spearman_r': mean_rho, # Renamed for auditor
                'p_value': p_val,
                'n_eff_total': group['n_eff'].sum()
            })
            
        out_file = Config.RESULTS_DIR / 'spearman_mechanism_SII_vs_ENV.csv'
        pd.DataFrame(final_mech).to_csv(out_file, index=False)
        print(f"  [SUCCESS] Analysis complete. Saved mechanism check to {out_file.name}")
    else:
        print("  [WARN] No mechanism results generated.")

if __name__ == "__main__":
    main()