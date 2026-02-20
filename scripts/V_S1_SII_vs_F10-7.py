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
- Continuous Temperature Gradient: Solar lines (F10.7) are color-coded 
  using a smooth 'YlOrBr' (Yellow-Orange-Brown) scale based on mean temperature.
- Significance (p-adj): Encoded via transparency and line style. 
  The legend is positioned in the bottom-right corner with optical 
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

class Config:
    PROJECT_ROOT = GlobalConfig.PROJECT_ROOT
    INPUT_DIR = GlobalConfig.RESULTS_DIR
    OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
    
    # Analysis Settings
    SCENARIO = "Global_High_LAI" # UPDATED: Switched from Control_North
    TARGET_VAR = "sif_771nm"
    OMNI_STAT = "mean"
    
    # Target bin for the SII line (Blue line)
    SII_BIN_LABEL = "Cold" 

    # Visual Style
    FIG_SIZE = (11, 7)
    CMAP_F107 = "YlOrBr" # Yellow-Orange-Brown for Solar lines
    LINE_WIDTH = 2.4
    CI_ALPHA_FACTOR = 0.25
    
    # Fonts
    FONT_SCALE = 1.6
    BASE_FONT = 11
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_FONT_SIZE = int(11 * FONT_SCALE)
    
    # Statistical thresholds for opacity
    P_VALUE_LEVELS = [
        (1e-12, 1.0),
        (1e-7,  0.8),
        (1e-4,  0.6),
        (1e-2,  0.4),
        (0.5,   0.15),
    ]
    
    # Axis Limits (consistent with Fig 1 style)
    Y_LIM = (-0.12, 0.12)
    
    # Fallback mapping if metadata is missing
    TEMP_PROXY = {0: 5.0, 1: 15.0, 2: 23.0, 3: 28.0, 4: 35.0}

def extract_window(val_str):
    match = re.search(r"ma(\d+)", str(val_str))
    return int(match.group(1)) if match else None

def p_to_alpha(p_val):
    if np.isnan(p_val): return 0.0
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
    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        line_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0
        ci_alpha = line_alpha * Config.CI_ALPHA_FACTOR
        if ci_alpha < 0.01: continue
        ax.fill_between([x[i], x[i+1]], [y_low[i], y_low[i+1]], [y_high[i], y_high[i+1]],
                        color=color, alpha=ci_alpha, edgecolor="none", zorder=0)

def plot_gradient_line(ax, x, y, p_vals, color):
    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        seg_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0
        ls = "-" if seg_alpha > Config.P_VALUE_LEVELS[-2][1] else "--"
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

def main():
    print(f"[INFO] Generating Figure S1 (Attribution) for {Config.SCENARIO}...")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = int(Config.BASE_FONT * Config.FONT_SCALE)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    plt.subplots_adjust(right=0.85, left=0.12, top=0.9, bottom=0.12)

    export_list = [] # Collect data for export

    # 1. Plot F10.7 (Solar Flux) - All Temperature Bins
    df_f107 = load_data("f10_7_mean_ma")
    bin_temps = {}
    n_bins = 0
    norm_f107 = None # Will be initialized for colorbar
    
    if not df_f107.empty:
        t_min, t_max = df_f107["temp_mean"].min(), df_f107["temp_mean"].max()
        norm_f107 = mcolors.Normalize(vmin=t_min, vmax=t_max)
        
        bin_ids = sorted(df_f107["bin_id"].unique())
        n_bins = len(bin_ids) # UPDATE n_bins
        cmap = plt.get_cmap(Config.CMAP_F107)

        for b_id in bin_ids:
            bin_data = df_f107[df_f107["bin_id"] == b_id].sort_values("window")
            if bin_data.empty: continue
            
            # Export data
            exp_data = bin_data.copy()
            exp_data['data_type'] = 'F10.7'
            exp_data['display_bin_id'] = b_id
            export_list.append(exp_data)

            current_temp = bin_data["temp_mean"].mean()
            bin_temps[b_id] = current_temp # Save for potential labels
            
            c = cmap(norm_f107(current_temp))
            
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

    # 3. Formatting
    ax.axhline(0, color="black", lw=1.0, alpha=0.8)
    ax.set_ylim(Config.Y_LIM)
    ax.set_ylabel(r"Spearman Correlation ($\rho$)", fontsize=Config.AXIS_FONT_SIZE)
    ax.set_xlabel("Integration Window (Days)", fontsize=Config.AXIS_FONT_SIZE)
    ax.tick_params(labelsize=Config.TICK_FONT_SIZE)

    # 4. Legends
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3, label=f'Geomagnetic ({Config.SII_BIN_LABEL})'),
        Line2D([0], [0], color=plt.get_cmap(Config.CMAP_F107)(0.6), lw=3, label='Solar Flux F10.7 (All Regimes)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.95)

    # Significance Scale
    p_ax = fig.add_axes([0.68, 0.19, 0.15, 0.12]) 
    p_ax.set_axis_off()
    
    p_ax.text(0, 1.1, "Significance (p-adj)", fontsize=11, fontweight='bold', transform=p_ax.transAxes)
    
    for i, (thresh, alpha_val) in enumerate(Config.P_VALUE_LEVELS):
        y_pos = 0.85 - i * 0.22
        
        # Compensate for optical illusion at alpha=1.0 
        current_lw = Config.LINE_WIDTH + 0.4 if alpha_val == 1.0 else Config.LINE_WIDTH
        
        ls = '-' if alpha_val > 0.4 else '--'
        
        # Draw line
        p_ax.plot([0, 0.25], [y_pos, y_pos], 
                  color='black', 
                  lw=current_lw, 
                  alpha=alpha_val, 
                  ls=ls, 
                  transform=p_ax.transAxes)
        
        # Text label
        if thresh < 1e-3:
            label = rf"$10^{{{int(math.log10(thresh))}}}$" # Removed "<" for compactness in powers
        else:
            label = f"{thresh:g}"
            
        p_ax.text(0.35, y_pos, rf"$p <$ {label}", va='center', fontsize=10, transform=p_ax.transAxes)
            
    # 5. Colorbar for F10.7 (Fixed block)
    if n_bins > 0 and norm_f107 is not None:
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
        sm = ScalarMappable(cmap=plt.get_cmap(Config.CMAP_F107), norm=norm_f107)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("F10.7 Mean Temperature (°C)", fontsize=12)

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