#!/usr/bin/env python3
"""
Meta-statistics Aggregator for MAGNETO: Synthesis of Global Response Patterns.

This script consolidates results from the high-resolution GPU Matrix Search (Step 11). 
It identifies peak geomagnetic signals across all physiological regimes and prepares 
harmonized datasets for final scientific visualization.

Core Logic:
1) Batch Processing: Iterates through all valid combinations of SIF targets 
   and geographic scenarios defined in _Common.Config.
2) Peak Signal Detection (Best Model Selection):
   For each physiological temperature bin (Cold to Extreme Heat), the script 
   scans thousands of OLS model results to find the "Best Model" defined by 
   the maximum absolute t-statistic of the SII coefficient (|t_sii|).
3) Driver Contextualization: Extracts corresponding coefficients for PAR and VPD 
   from the same "Best Model" to evaluate driver dominance.
4) Formatting: Transforms long-form data into wide-form layouts required by 
   Stage 5 visualization scripts (e.g., stacked bar charts).

OUTPUT FILES STRUCTURE (reports/meta-statistics/):
--------------------------------------------------
1. meta_summary_120rows.csv (Long Format - Detailed Synthesis):
   This file contains 120 rows (4 targets * 6 scenarios * 5 bins) representing 
   the "biological ceiling" of the geomagnetic effect:
   - 'target', 'scenario', 'bin_label': Grouping keys.
   - 't_sii', 'p_sii', 'beta_sii': Peak statistical metrics for the SII driver.
   - 'sii_window', 'par_window', 'vpd_window': The specific integration windows 
     where the strongest SII effect was detected.
   - 't_par', 't_vpd': The influence of climatic confounders in this specific 
     optimal window combination.
   - 'n_eff', 'n': Effective and raw sample sizes for the optimal model.

2. meta_summary_60combos_wide.csv (Wide Format - Visualization Ready):
   A pivoted version of the summary (focused on primary targets) designed 
   specifically for comparison and plotting:
   - 'metric_mask_bin': Combined key for unique experimental cells.
   - 't_sii_MA': Peak t-statistic for SII (representing cumulative effects).
   - 'p_sii_MA': Corresponding significance level.
   - 'beta_sii_MA': Effect size.
   - 't_par_MA', 't_vpd_MA': Climatic driver magnitudes for dominance analysis.
   - Note: The '_MA' suffix is used to distinguish these cumulative results 
     from discrete lag ('_LAG') models if added in future iterations.

These meta-statistics serve as the direct source for Figure 2 (Driver Contribution) 
and Table 1 of the manuscript.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from _Common import Config

# Create dir for reports
OUT_DIR = Config.META_ANALYSIS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_best_model_per_bin(df, target, scenario):
    """
    Finds the row with the strongest SII signal (max abs(t_sii)) for each temperature bin.
    """
    best_rows = []
    
    if 't_sii' not in df.columns:
        return []

    # Temp bin grouping
    # Just in case any 'bin_label' remains
    bin_col = 'bin_label' if 'bin_label' in df.columns else 'bin_id'
    
    if bin_col not in df.columns:
         return []

    for bin_val, group in df.groupby(bin_col):
        # Ignore bins with no valid t-score
        valid_t = group['t_sii'].dropna()
        if valid_t.empty:
            continue

        # Search for max abs t-score
        try:
            best_idx = valid_t.abs().idxmax()
            row = group.loc[best_idx]
            
            entry = {
                "metric": target,
                "bin": bin_val,
                "mask": scenario,
                "mode": "MA",  # Скрипт 11 сейчас работает в режиме Moving Average
                "sii_step": row.get('sii_window', np.nan),
                "par_step": row.get('par_window', np.nan),
                "vpd_step": row.get('vpd_window', np.nan),
                "beta_sii": row.get('beta_sii', np.nan),
                "t_sii": row.get('t_sii', np.nan),
                "p_sii": row.get('p_sii', np.nan),
                "n_eff": row.get('n_eff', np.nan),
                "n": row.get('n', np.nan),
                "t_par": row.get('t_par', np.nan),
                "t_vpd": row.get('t_vpd', np.nan)
            }
            best_rows.append(entry)
        except Exception:
            continue
        
    return best_rows

def main():
    print("[INFO] Starting Meta-Statistics Aggregation...")
    
    all_best_models = []
    
    # 1. shuffle all combinations
    targets = Config.SPEARMAN_TARGETS
    scenarios = Config.SCENARIO_MASKS.keys()
    
    found_count = 0
    missing_count = 0
    
    for target in tqdm(targets, desc="Targets"):
        for scenario in scenarios:
            # Scientific Matrix Search
            filename = f"matrix_search_{target}_{scenario}.csv"
            filepath = Config.RESULTS_DIR / filename
            
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if not df.empty:
                        best = get_best_model_per_bin(df, target, scenario)
                        all_best_models.extend(best)
                        found_count += 1
                    else:
                        print(f"  [WARN] File empty: {filename}")
                except Exception as e:
                    print(f"  [ERROR] Failed to read {filename}: {e}")
            else:
                missing_count += 1

    print(f"[INFO] Processed {found_count} files. Missing/Empty: {missing_count}")

    # 2. Create the DataFrame
    if not all_best_models:
        print("[WARNING] No data found! Creating empty skeleton.")
        cols = ["metric", "bin", "mask", "mode", "t_sii", "p_sii", "n_eff"]
        df_long = pd.DataFrame(columns=cols)
    else:
        df_long = pd.DataFrame(all_best_models)

    # 3. Produce a skeleton
    # Important for heatmaps geometry
    bins = list(Config.TEMP_LABELS_PHYSIO)
    modes = ["MA"] # no LAG yet?
    
    # Full index
    full_idx = pd.MultiIndex.from_product(
        [targets, bins, scenarios, modes],
        names=["metric", "bin", "mask", "mode"]
    ).to_frame(index=False)
    
    # Merge all the data along the index
    for c in ["metric", "bin", "mask", "mode"]:
        full_idx[c] = full_idx[c].astype(str)
        if not df_long.empty:
            df_long[c] = df_long[c].astype(str)
    
    final_long = pd.merge(full_idx, df_long, on=["metric", "bin", "mask", "mode"], how="left")
    
    # save long (120 lines)
    long_path = OUT_DIR / "meta_summary_120rows.csv"
    final_long.to_csv(long_path, index=False)
    print(f"  Saved Long Format: {long_path}")

    # 4. create wide  (60 lines)
    # Having only MA so far, but structure demands both _MA и _LAG
    # Creating fake _LAG (empty) и fill in _MA
    
    df_wide = final_long.copy()
    
    # Suffix formation
    # As pivot_table can be complex
    # but we have only MA
    
    cols_to_rename = {
        't_sii': 't_sii_MA',
        'p_sii': 'p_sii_MA',
        'beta_sii': 'beta_sii_MA',
        'n_eff': 'n_eff_MA',
        'sii_step': 'sii_step_MA',
        'par_step': 'par_step_MA',
        'vpd_step': 'vpd_step_MA',
        't_par': 't_par_MA',
        't_vpd': 't_vpd_MA',
        'n': 'n_MA'
    }
    
    # Check the cols
    existing_cols_to_rename = {k: v for k, v in cols_to_rename.items() if k in df_wide.columns}
    df_wide = df_wide.rename(columns=existing_cols_to_rename)
    
    # Add empty LAG cols for compatibility
    lag_cols = [c.replace('_MA', '_LAG') for c in existing_cols_to_rename.values()]
    for c in lag_cols:
        df_wide[c] = np.nan
        
    # remove 'mode', as it ia appended to cols
    if 'mode' in df_wide.columns:
        df_wide = df_wide.drop(columns=['mode'])
        
    wide_path = OUT_DIR / "meta_summary_60combos_wide.csv"
    df_wide.to_csv(wide_path, index=False)
    print(f"  Saved Wide Format: {wide_path}")

if __name__ == "__main__":
    main()