#!/usr/bin/env python3
"""
Figure S1 Generator: Attribution Analysis (Spearman rho).

**Description:**
This script generates a diagnostic visualization to disentangle the effects 
of solar activity (F10.7) and geomagnetic/ionospheric perturbations (SII) 
on SIF across various thermal regimes. It also exports the source data for 
the figure to a CSV file.

**Layout:**
- Single Panel: Focuses on the Global High LAI scenario to compare drivers.
- Primary Data (SII): Blue line representing the ionospheric response 
  for a selected thermal bin (default: 'Cold').
- Overlay (F10.7): Multiple lines representing the solar flux correlation 
  across all temperature regimes, enabling attribution of SIF variability.

**Visual Encoding:**
- Lines & Shading: Represent Spearman's rho and Fisher-transformed 95% CI.
- Discrete Temperature Colors: Solar lines (F10.7) use discrete colors from 
  the 'YlOrBr' colormap based on temperature bins.
- Significance (p-adj): Encoded via transparency and line style 
  (solid for p < 0.01, dashed otherwise) using thresholds from CommonConfig.
- The legend is positioned in the bottom-right corner with optical 
  compensation for high-contrast lines.

**Outputs:**
- PDF image: `reports/figures/supplementary/Fig_S1_Attribution.pdf`
- CSV data: `reports/figures/supplementary/Fig_S1_Attribution.csv`
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import re
from pathlib import Path
import math

# --- Pipeline Integration ---
try:
    from _Common import Config as GlobalConfig
except ImportError:
    print("[WARN] _Common not found. Using standalone paths.")
    class GlobalConfig:
        PROJECT_ROOT = Path(".")
        RESULTS_DIR = Path(".")
        # P_VALUE_LEVELS and P_FLOOR from _Common for consistency
        P_VALUE_LEVELS = [
            (1e-12, 1.0),
            (1e-7,  0.8),
            (1e-4,  0.6),
            (1e-2,  0.4),
            (0.5,   0.15),
        ]
        P_FLOOR = 1e-15
        TEMP_RANGES = {0: "0-10", 1: "10-20", 2: "20-26", 3: "26-32", 4: "32+"}

class Config:
    PROJECT_ROOT = GlobalConfig.PROJECT_ROOT
    INPUT_DIR = GlobalConfig.RESULTS_DIR
    OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "supplementary"
    
    # Analysis Settings
    SCENARIO = "Global_High_LAI"
    TARGET_VAR = "sif_771nm"
    OMNI_STAT = "mean"
    
    # Target bin for the SII line (Blue line)
    SII_BIN_LABEL = "Cold" 

    # Visual Style
    FIG_SIZE = (11, 7)
    CMAP_F107 = "YlOrBr"  # Keep YlOrBr for temperature lines
    LINE_WIDTH = 2.4
    CI_ALPHA_FACTOR = 0.25
    
    # Fonts
    FONT_SCALE = 1.6
    BASE_FONT = 11
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_FONT_SIZE = int(11 * FONT_SCALE)
    
    # Statistical thresholds from CommonConfig
    P_VALUE_LEVELS = GlobalConfig.P_VALUE_LEVELS
    P_FLOOR = GlobalConfig.P_FLOOR
    
    # Axis Limits - UPDATED to -0.2 to 0.2 as requested
    Y_LIM = (-0.2, 0.2)
    
    # Discrete temperature mapping (bin_id -> display name)
    TEMP_LABELS = {
        0: "Very Cold (0-10°C)",
        1: "Cold (10-20°C)",
        2: "Moderate (20-26°C)",
        3: "Warm (26-32°C)",
        4: "Hot (32°C+)"
    }
    
    # Fallback mapping if metadata is missing
    TEMP_PROXY = {0: 5.0, 1: 15.0, 2: 23.0, 3: 28.0, 4: 35.0}

def extract_window(val_str):
    match = re.search(r"ma(\d+)", str(val_str))
    return int(match.group(1)) if match else None

def p_to_alpha(p_val):
    if np.isnan(p_val): return 0.0
    # Ensure p_val is positive for threshold comparison
    p_val = max(p_val, Config.P_FLOOR)
    for threshold, alpha in Config.P_VALUE_LEVELS:
        if p_val <= threshold: return alpha
    return Config.P_VALUE_LEVELS[-1][1]

def calculate_fisher_ci(df):
    """Calculates 95% CI and handles missing metadata."""
    if 'p_adj' not in df.columns:
        df['p_adj'] = df.get('p_value', 1.0)
            
    if 'temp_mean' not in df.columns and 'bin_id' in df.columns:
        df['temp_mean'] = df['bin_id'].map(Config.TEMP_PROXY)

    if 'ci_lower' not in df.columns and 'rho' in df.columns:
        n = df.get('n_eff', df.get('n', 100))
        r = df['rho'].clip(-0.99, 0.99)
        z = np.arctanh(r)
        sigma = 1.0 / np.sqrt(np.maximum(n - 3, 1))
        df['ci_lower'] = np.tanh(z - 1.96 * sigma)
        df['ci_upper'] = np.tanh(z + 1.96 * sigma)
    return df

def plot_segmented_ci(ax, x, y_low, y_high, p_vals, color):
    # Ensure p_vals are positive for alpha calculation
    p_vals = np.asarray(p_vals, dtype=float)
    p_vals = np.where(np.isfinite(p_vals) & (p_vals > 0), p_vals, Config.P_FLOOR)
    
    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        line_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0
        ci_alpha = line_alpha * Config.CI_ALPHA_FACTOR
        if ci_alpha < 0.01: continue
        ax.fill_between([x[i], x[i+1]], [y_low[i], y_low[i+1]], [y_high[i], y_high[i+1]],
                        color=color, alpha=ci_alpha, edgecolor="none", zorder=0)

def plot_gradient_line(ax, x, y, p_vals, color):
    p_vals = np.asarray(p_vals, dtype=float)
    p_vals = np.where(np.isfinite(p_vals) & (p_vals > 0), p_vals, Config.P_FLOOR)
    
    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        seg_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0
        
        # solid if p < 0.01 else dashed (from Figure 1)
        p_mid = np.nanmean([p_vals[i], p_vals[i + 1]])
        ls = "-" if (np.isfinite(p_mid) and p_mid < 0.01) else "--"
        
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, alpha=seg_alpha,
                ls=ls, lw=Config.LINE_WIDTH, zorder=2)

def load_data(var_pattern: str) -> pd.DataFrame:
    csv_file = Config.INPUT_DIR / f"spearman_{Config.TARGET_VAR}_{Config.SCENARIO}.csv"
    if not csv_file.exists():
        print(f"[ERROR] Data file not found: {csv_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_file)
    df = df[df["omni_var"].astype(str).str.contains(var_pattern)].copy()
    if df.empty: return pd.DataFrame()

    df["window"] = df["omni_var"].apply(extract_window)
    df = df.dropna(subset=["window"]).sort_values("window")
    return calculate_fisher_ci(df)

def add_pvalue_colorbar(fig, cax):
    """Create p-value colorbar using thresholds from CommonConfig"""
    # Get the unique alpha thresholds and corresponding p-values
    p_thresholds = [t[0] for t in Config.P_VALUE_LEVELS if t[0] < 0.5]
    # Add a reasonable max
    p_thresholds.append(0.5)
    
    vmin = min(p_thresholds)
    vmax = 0.5
    
    cmap = plt.cm.Greys_r
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    label = "Transparency: adjusted p (log scale)"
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Set ticks at the p-value thresholds
    ticks = [t for t in p_thresholds if t <= vmax]
    cbar.set_ticks(ticks)

    def fmt(t):
        if t < 1e-1:
            # Format as 10^{-x} for small values
            exp = int(np.round(np.log10(t)))
            return rf"$10^{{{exp}}}$"
        return f"{t:g}"

    cbar.set_ticklabels([fmt(t) for t in ticks])
    return cbar

def main():
    print(f"[INFO] Generating Figure S1 (Attribution) for {Config.SCENARIO}...")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = int(Config.BASE_FONT * Config.FONT_SCALE)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    plt.subplots_adjust(right=0.85, left=0.12, top=0.9, bottom=0.12)

    export_list = [] # Collect data for export

    # 1. Plot F10.7 (Solar Flux) - All Temperature Bins with discrete colors
    df_f107 = load_data("f10_7_mean_ma")
    bin_temps = {}
    n_bins = 0
    
    if not df_f107.empty:
        bin_ids = sorted(df_f107["bin_id"].unique())
        n_bins = len(bin_ids)
        
        # Create discrete colormap from YlOrBr
        cmap = plt.get_cmap(Config.CMAP_F107)
        # Generate evenly spaced colors for the number of bins
        colors = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]

        for idx, b_id in enumerate(bin_ids):
            bin_data = df_f107[df_f107["bin_id"] == b_id].sort_values("window")
            if bin_data.empty: continue
            
            # Export data
            exp_data = bin_data.copy()
            exp_data['data_type'] = 'F10.7'
            exp_data['display_bin_id'] = b_id
            export_list.append(exp_data)

            current_temp = bin_data["temp_mean"].mean()
            bin_temps[b_id] = current_temp
            
            c = colors[idx]
            
            plot_segmented_ci(ax, bin_data["window"].to_numpy(), bin_data["ci_lower"].to_numpy(),
                              bin_data["ci_upper"].to_numpy(), bin_data["p_adj"].to_numpy(), color=c)
            plot_gradient_line(ax, bin_data["window"].to_numpy(), bin_data["rho"].to_numpy(),
                               bin_data["p_adj"].to_numpy(), color=c)

    # 2. Plot SII (Geomagnetic) - Selected Bin Only
    df_sii = load_data(f"sii_{Config.OMNI_STAT}_ma")
    if not df_sii.empty:
        # Search by label if available, else fallback to bin_id=1 (Cool)
        if 'bin_label' in df_sii.columns:
            bin_data = df_sii[df_sii['bin_label'] == Config.SII_BIN_LABEL].sort_values("window")
        else:
            target_id = 1 if 1 in df_sii["bin_id"].unique() else 0
            bin_data = df_sii[df_sii["bin_id"] == target_id].sort_values("window")
            
        if not bin_data.empty:
            # Export data
            exp_data = bin_data.copy()
            exp_data['data_type'] = f'SII_{Config.SII_BIN_LABEL}'
            export_list.append(exp_data)

            color = '#1f77b4' # Strong Blue
            plot_segmented_ci(ax, bin_data["window"].to_numpy(), bin_data["ci_lower"].to_numpy(),
                              bin_data["ci_upper"].to_numpy(), bin_data["p_adj"].to_numpy(), color=color)
            plot_gradient_line(ax, bin_data["window"].to_numpy(), bin_data["rho"].to_numpy(),
                               bin_data["p_adj"].to_numpy(), color=color)

    # 3. Formatting - UPDATED y-limits to -0.2, 0.2
    ax.axhline(0, color="black", lw=1.0, alpha=0.8)
    ax.set_ylim(Config.Y_LIM)  # Now using (-0.2, 0.2)
    ax.set_ylabel(r"Spearman Correlation ($\rho$)", fontsize=Config.AXIS_FONT_SIZE)
    ax.set_xlabel("Integration Window (Days)", fontsize=Config.AXIS_FONT_SIZE)
    ax.tick_params(labelsize=Config.TICK_FONT_SIZE)
    ax.grid(True, alpha=0.35)
    
    # Set y-axis ticks for better readability with new limits
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    # 4. Legends - Create discrete temperature legend entries
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3, label=f'SII ({Config.SII_BIN_LABEL})')
    ]
    
    # Add discrete temperature entries for F10.7
    if n_bins > 0:
        cmap = plt.get_cmap(Config.CMAP_F107)
        colors = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]
        bin_ids = sorted(df_f107["bin_id"].unique()) if not df_f107.empty else []
        
        for idx, b_id in enumerate(bin_ids):
            temp_label = Config.TEMP_LABELS.get(b_id, f"Bin {b_id}")
            legend_elements.append(
                Line2D([0], [0], color=colors[idx], lw=3, label=f'F10.7 {temp_label}')
            )
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, 
              ncol=2 if n_bins > 3 else 1)  # Use two columns if many bins

    # 5. p-value colorbar (using thresholds from CommonConfig)
    cax_p = ax.inset_axes([0.55, 0.15, 0.3, 0.05])
    add_pvalue_colorbar(fig, cax_p)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = Config.OUTPUT_DIR / "Fig_S1_Attribution.pdf"
    plt.savefig(out_file, bbox_inches="tight") 
    print(f"[SUCCESS] Saved: {out_file}")

    # Export Data
    if export_list:
        final_export_df = pd.concat(export_list, ignore_index=True)
        csv_file = Config.OUTPUT_DIR / "Fig_S1_Attribution.csv"
        final_export_df.to_csv(csv_file, index=False)
        print(f"[SUCCESS] Exported data: {csv_file}")

if __name__ == "__main__":
    main()