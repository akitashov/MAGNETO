#!/usr/bin/env python3
"""
Visualization Module: Correlation Line Plots (Batch Processing).
FIXED VERSION:
1. Dynamic detection of bin columns (fixes missing lines).
2. Plots one line per temperature bin found in the file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import re
import traceback
from pathlib import Path
import math

class Config:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    TARGETS = [
        'sif_740nm', 'sif_757nm', 'sif_771nm', 
        'sif_stress_index', 
    ]
    
    STATS = ['mean', 'min', 'max', 'std']
    
    PRETTY_NAMES = {
        'sif_740nm': r'SIF$_{740nm}$',
        'sif_757nm': r'SIF$_{757nm}$',
        'sif_771nm': r'SIF$_{771nm}$',
        'sif_stress_index': 'Stress Index',
    }
    
    # Regex to extract window size from row variable (e.g., sii_mean_ma10)
    VARIABLE_PATTERN = r'^sii_(?:mean|max|min|std)_ma\d+$'

    # Regex to find bin columns (e.g., rho_bin_1_min_to_1)
    BIN_COL_PATTERN = r'rho_bin_(\d+)_(.+)$'
    
    MAX_P_VALUE = 0.05
    MIN_RHO_THRESHOLD = 0.005        
    FIG_SIZE = (14, 9)
    CMAP_NAME = 'coolwarm' 
    LINE_WIDTH = 2.0
    POINT_SIZE_BASE = 40
    
    P_VALUE_LEVELS = [(0.001, 1.0), (0.05, 0.7), (0.1, 0.4), (1.0, 0.25)]
    LABEL_OFFSET = 15
    LEGEND_LOC = 'upper right'
    FONT_FAMILY = 'DejaVu Sans'
    AXIS_FONT_SIZE = 14
    TITLE_FONT_SIZE = 16

def format_p_value(val):
    if np.isnan(val): return ""
    if val == 0: return r"$p < 10^{-16}$"
    if val < 0.001:
        exponent = int(math.floor(math.log10(val)))
        mantissa = val / (10**exponent)
        return r"$p={:.1f} \cdot 10^{{{}}}$".format(mantissa, exponent)
    return f"$p={val:.3f}$"

def extract_window(val_str):
    match = re.search(r'ma(\d{1,3})', str(val_str))
    return int(match.group(1)) if match else None

def p_to_alpha(p_value):
    if np.isnan(p_value): return 0.0
    for threshold, alpha in Config.P_VALUE_LEVELS:
        if p_value <= threshold: return alpha
    return Config.P_VALUE_LEVELS[-1][1]

def check_2d_peak(grid, r, c, r_nb=1, c_nb=1):
    val = grid[r, c]
    if np.isnan(val): return False, False
    rows, cols = grid.shape
    r0, r1 = max(0, r - r_nb), min(rows, r + r_nb + 1)
    c0, c1 = max(0, c - c_nb), min(cols, c + c_nb + 1)
    sub = grid[r0:r1, c0:c1]
    valid = sub[~np.isnan(sub)]
    if len(valid) == 0: return False, False
    return ((val == np.max(valid)) and (val > 0), (val == np.min(valid)) and (val < 0))

def parse_bin_label(raw_desc):
    """Parses technical bin names into readable labels."""
    try:
        if 'min_to' in raw_desc:
            val = raw_desc.split('_')[-1]
            return f"≤ {val}°C"
        elif 'above' in raw_desc:
            val = raw_desc.split('_')[-1]
            return f"> {val}°C"
        elif '_to_' in raw_desc:
            parts = raw_desc.split('_to_')
            return f"{parts[0]} - {parts[1]}°C"
        else:
            return raw_desc.replace('_', ' ')
    except:
        return raw_desc

def get_bin_info(df_columns):
    """Dynamically finds all bin columns."""
    bins = {}
    for col in df_columns:
        match = re.search(Config.BIN_COL_PATTERN, col)
        if match:
            idx = int(match.group(1))
            desc = match.group(2)
            if idx not in bins:
                bins[idx] = {'desc': desc}
            bins[idx]['rho_col'] = f"rho_bin_{idx}_{desc}"
            bins[idx]['p_col'] = f"p_bin_{idx}_{desc}"
    
    sorted_indices = sorted(bins.keys())
    result = []
    for idx in sorted_indices:
        info = bins[idx]
        info['index'] = idx
        info['label'] = parse_bin_label(info['desc'])
        result.append(info)
    return result

def generate_line_plot(df_target, target_var, stat_type):
    pattern = f'sii_{stat_type}'
    df_plot = df_target[df_target['variable'].str.contains(pattern)].copy()
    
    if df_plot.empty:
        print(f"  [SKIP] No data for {pattern}")
        return

    # Prepare X-axis (Windows)
    df_plot['window_size'] = df_plot['variable'].apply(extract_window)
    df_plot = df_plot.dropna(subset=['window_size']).sort_values('window_size')
    unique_windows = df_plot['window_size'].unique()
    win_to_idx = {w: i for i, w in enumerate(unique_windows)}
    
    # Prepare Lines (Bins)
    bin_infos = get_bin_info(df_plot.columns)
    if not bin_infos:
        print("  [WARN] No bin columns found.")
        return

    n_wins = len(unique_windows)
    n_bins = len(bin_infos)
    
    # Grid: Rows=Windows, Cols=Bins
    rho_grid = np.full((n_wins, n_bins), np.nan)
    p_grid = np.full((n_wins, n_bins), np.nan)
    
    for _, row in df_plot.iterrows():
        w_idx = win_to_idx.get(row['window_size'])
        if w_idx is None: continue
        
        for b_idx, b_info in enumerate(bin_infos):
            rho_col = b_info['rho_col']
            p_col = b_info['p_col']
            
            if rho_col in row: rho_grid[w_idx, b_idx] = row[rho_col]
            if p_col in row: p_grid[w_idx, b_idx] = row[p_col]
            
    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    x_vals = unique_windows
    
    # Use index to pick color from colormap
    cmap = plt.get_cmap(Config.CMAP_NAME)
    norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
    
    # Plot each bin as a separate line
    for b_idx in range(n_bins):
        y_vals = rho_grid[:, b_idx] 
        p_vals = p_grid[:, b_idx]
        label = bin_infos[b_idx]['label']
        
        # Color based on bin index (Temperature gradient)
        c = cmap(norm(b_idx))
        
        # Determine alpha/size for points based on significance
        alphas = [p_to_alpha(p) for p in p_vals]
        sizes = [Config.POINT_SIZE_BASE * (1.5 if a > 0.8 else 1.0 if a > 0.3 else 0.5) for a in alphas]
        
        # Draw segments
        for i in range(len(x_vals) - 1):
            if np.isnan(y_vals[i]) or np.isnan(y_vals[i+1]): continue
            
            p_start = p_vals[i]
            p_end = p_vals[i+1]
            
            # Solid line if both points are significant
            is_sig = (p_start <= 0.05) and (p_end <= 0.05)
            line_style = '-' if is_sig else '--'
            
            # Transparency is average of points
            seg_alpha = (alphas[i] + alphas[i+1]) / 2
            # Boost alpha slightly for visibility of non-sig lines
            seg_alpha = max(seg_alpha, 0.3)
            
            ax.plot([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]], 
                    color=c, lw=Config.LINE_WIDTH, alpha=seg_alpha, ls=line_style, zorder=10)
        
        # Scatter significant points
        sig_indices = [idx for idx, p in enumerate(p_vals) if p <= 0.05]
        if sig_indices:
            ax.scatter(x_vals[sig_indices], y_vals[sig_indices], 
                       color=c, 
                       s=[sizes[idx] for idx in sig_indices], 
                       alpha=[alphas[idx] for idx in sig_indices], 
                       edgecolor='none', zorder=11)

    # Auto-scale Y
    valid_values = rho_grid[~np.isnan(rho_grid)]
    if len(valid_values) > 0:
        max_abs = np.max(np.abs(valid_values))
        limit = max(0.1, max_abs * 1.25)
        ax.set_ylim(-limit, limit)

    # Annotate Peaks
    for r in range(n_wins):
        for c in range(n_bins):
            val, pval = rho_grid[r, c], p_grid[r, c]
            if np.isnan(val): continue
            
            is_max, is_min = check_2d_peak(rho_grid, r, c)
            # Annotate if it is a local peak AND significant AND strong enough
            if (is_max or is_min) and (pval <= Config.MAX_P_VALUE) and (abs(val) >= Config.MIN_RHO_THRESHOLD):
                p_str = format_p_value(pval)
                txt = f"$\\rho$={val:.3f}\n{p_str}"
                
                xytext = (0, Config.LABEL_OFFSET) if val > 0 else (0, -Config.LABEL_OFFSET)
                ax.annotate(txt, (unique_windows[r], val), xytext=xytext,
                            textcoords='offset points', ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"), zorder=12)

    ax.axhline(0, color='black', lw=1.5, alpha=0.8)
    ax.set_xlabel('Smoothing Window Size (Days)', fontsize=Config.AXIS_FONT_SIZE)
    ax.set_xticks(unique_windows)
    ax.set_xticklabels(unique_windows.astype(int), fontsize=12, rotation=0)
    ax.set_ylabel(r'Spearman Correlation ($\rho$)', fontsize=Config.AXIS_FONT_SIZE)
    
    pretty_name = Config.PRETTY_NAMES.get(target_var, target_var)
    ax.set_title(f'Correlation: {pretty_name} vs SII {stat_type.upper()}', fontsize=Config.TITLE_FONT_SIZE)
    
    # Colorbar for Temperature
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Custom ticks for colorbar
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Temperature Bin', fontsize=12)
    # Set ticks in the middle of each color segment
    tick_locs = np.arange(n_bins)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([b['label'] for b in bin_infos])
    
    # Legend for Line Styles
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, ls='-', label='Sig. (p<0.05)'),
        Line2D([0], [0], color='black', lw=2, ls='--', alpha=0.5, label='Not Sig.')
    ]
    ax.legend(handles=legend_elements, loc=Config.LEGEND_LOC, title='Significance', frameon=True)
    
    output_file = Config.PROJECT_ROOT / "visualizations" / f"lines_{target_var}_sii_{stat_type}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {output_file.name}")

def process_target(target_var):
    input_file = Config.PROJECT_ROOT / "results" / f"spearman_{target_var}.parquet"
    if not input_file.exists():
        print(f"[WARN] Skipping {target_var}: File missing.")
        return

    print(f"--- Processing {target_var} ---")
    df = pd.read_parquet(input_file)
    if 'omni_variable' in df.columns:
        df = df.rename(columns={'omni_variable': 'variable'})

    for stat in Config.STATS:
        generate_line_plot(df, target_var, stat)

def main():
    print("Starting Line Visualization Pipeline (Dynamic Bins)...")
    for target in Config.TARGETS:
        try:
            process_target(target)
        except Exception as e:
            print(f"[ERROR] {target}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()