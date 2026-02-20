#!/usr/bin/env python3
"""
Aggregation and Meta-Analysis for Spearman Correlation Results.

This script collects individual correlation reports generated in Step 06 
and consolidates them into unified tables for global trend analysis and 
visualization (Stage 5).

Core Logic:
1) Scanning: Iterates through RESULTS_DIR for files matching 'spearman_*.csv'.
2) Target Identification: Uses a robust matching algorithm (based on Config.SPEARMAN_TARGETS) 
   to correctly extract target variables (e.g., 'sif_771nm', 'sif_stress_index') 
   from filenames, preventing substring collisions.
3) Consolidation: Merges all scenario-specific data into a single long-form master table.
4) Statistical Filtering: Aggregates results by target, temperature bin, and driver 
   to calculate global occurrence rates of significant correlations.

OUTPUT FILES STRUCTURE (reports/meta-statistics/):
--------------------------------------------------
1. spearman_overview_summary.csv (Detailed Master Table):
   - 'target': SIF wavelength or index (740nm, 757nm, 771nm, stress_index).
   - 'scenario': Geographic mask (e.g., Global_High_LAI, Control_North).
   - 'omni_var': Space weather or environmental driver (e.g., sii_mean_ma21).
   - 'bin_label': Physiological temperature bin (e.g., Cold, Optimum).
   - 'rho': Spearman's rank correlation coefficient.
   - 'p_adj': Significance value adjusted for effective sample size (N_eff).
   - 'n_eff': Effective sample size after autocorrelation correction.
   - 'n': Raw number of observations in the bin.

2. spearman_global_trends.csv (Consolidated Global Metrics):
   - 'target', 'bin_label', 'omni_var': Grouping keys.
   - 'mean_rho': Arithmetic mean of correlation across all scenarios.
   - 'median_rho': Robust average to mitigate scenario-specific outliers.
   - 'n_scenarios': Count of scenarios contributing to this specific bin/driver.
   - 'perc_sig': Percentage of scenarios where the correlation is significant (p_adj < 0.05).

The resulting tables serve as the primary input for Figure 1 and Figure 2 generation.
"""

import glob
import os
import pandas as pd
import numpy as np
from _Common import Config

def main():
    in_dir = Config.RESULTS_DIR
    out_dir = Config.META_ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_dir / "spearman_*.csv")))
    if not files:
        print(f"[WARN] No Spearman results found in {in_dir}")
        return

    all_data = []
    for f in files:
        # Extract target name from filename (between spearman_ and _scenario)
        basename = os.path.basename(f)
    
        # Robust target detection:
        # Check if filename starts with one of the known targets defined in Config.
        # Retrieved from _Common.py: ["sif_740nm", ..., "sif_stress_index"]
        known_targets = getattr(Config, "SPEARMAN_TARGETS", [])
        
        target = "unknown"
        # Sort by length (descending) to ensure specific matches (e.g., 'sif_stress_index')
        # are found before substrings (e.g., 'sif_stress' if it existed).
        for t in sorted(known_targets, key=len, reverse=True):
            if basename.startswith(f"spearman_{t}_"):
                target = t
                break
                
        # Remove the fallback logic that could create incorrect target names
        if target == "unknown":
            # Just report the error but don't try to guess the target
            print(f"[ERROR] Could not determine target from filename: {basename}")
            print(f"  Expected filename pattern: spearman_[TARGET]_[SCENARIO].csv")
            print(f"  Known targets: {known_targets}")
            continue

        df = pd.read_csv(f)
        df["target"] = target
        all_data.append(df)

    if not all_data:
        print("[ERROR] No valid data files found with correct target names")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # 1. Detailed Overview (All targets, scenarios, and bins)
    summary_path = out_dir / "spearman_overview_summary.csv"
    full_df.to_csv(summary_path, index=False)

    # 2. Global Trends (Mean Rho per target across scenarios)
    # Filter for significant results and group
    global_trends = (
        full_df.groupby(["target", "bin_label", "omni_var"])
        .agg(
            mean_rho=("rho", "mean"),
            median_rho=("rho", "median"),
            n_scenarios=("scenario", "count"),
            perc_sig=("p_adj", lambda x: (x < 0.05).mean() * 100)
        )
        .reset_index()
    )
    
    trends_path = out_dir / "spearman_global_trends.csv"
    global_trends.to_csv(trends_path, index=False)

    print(f"[SUCCESS] Spearman Meta-analysis saved to {out_dir}")

if __name__ == "__main__":
    main()