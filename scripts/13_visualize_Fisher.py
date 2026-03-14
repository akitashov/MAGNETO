#!/usr/bin/env python3
"""
Visualization Module: Fisher Z-Difference Map for SIF 757nm vs 771nm
Methodology:
1. Convert Spearman rho to Fisher Z-score: z = 0.5 * ln((1+r)/(1-r))
2. Calculate Delta Z = Z_757 - Z_771
3. Visualize Delta Z to show differential sensitivity.

This approach statistically validates the difference in correlation strength
between two dependent variables, solving the "diagonal split" visual clutter problem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import traceback
from pathlib import Path

class Config:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # Target variables
    TARGET_A = 'sif_757nm'  # "Fluorescence + PSII congestion" indicator
    TARGET_B = 'sif_771nm'  # "PSI damage / Deep quenching" indicator
    
    # Statistic to visualize (from SII indices)
    # Options: 'mean', 'std', 'max', 'min'
    STAT = 'mean' 
    
    PRETTY_NAMES = {
        'sif_757nm': r'SIF$_{757nm}$',
        'sif_771nm': r'SIF$_{771nm}$',
    }

    # Regex patterns to parse column names
    # Expects columns like: rho_bin_5_16_to_20, p_bin_5_16_to_20
    BIN_COL_PATTERN = r'rho_bin_(\d+)_(.+)$'
    
    # Figure settings
    FIG_SIZE = (16, 10)
    
    # Colormap for Difference (Delta Z)
    # 'PuOr' (Purple-Orange) or 'RdBu' (Red-Blue) are standard for divergence
    # Positive (Red/Orange) = 757 > 771 (PSII signal dominates)
    # Negative (Blue/Purple) = 771 > 757 (PSI signal dominates)
    CMAP_DIFF = 'Greens'    
    FONT_FAMILY = 'DejaVu Sans'
    TITLE_SIZE = 16
    AXIS_LABEL_SIZE = 12
    TICK_SIZE = 10
    CELL_FONT_SIZE = 8
    MISSING_COLOR = '#eeeeee'

    # Output filename
    OUTPUT_FILENAME = f"fisher_diff_map_{STAT}.png"

def fisher_z_transform(r):
    """
    Apply Fisher r-to-z transformation.
    Stabilizes variance and allows valid subtraction of correlations.
    """
    # Clip values to avoid infinity at r=1.0 or r=-1.0
    r_clipped = np.clip(r, -0.999, 0.999)
    return np.arctanh(r_clipped)

def extract_window(val_str):
    """Extract moving average window size from variable name (e.g., 'sii_mean_ma25')"""
    match = re.search(r'ma(\d{1,3})', str(val_str))
    return int(match.group(1)) if match else None

def parse_bin_label(raw_desc):
    """Convert raw bin strings (e.g., '16_to_20') to readable labels"""
    try:
        if 'min_to' in raw_desc:
            val = raw_desc.split('_')[-1]
            return f"≤ {val}°C"
        elif 'above' in raw_desc:
            val = raw_desc.split('_')[-1]
            return f"> {val}°C"
        elif '_to_' in raw_desc:
            parts = raw_desc.split('_to_')
            return f"{parts[0]}–{parts[1]}°C"
        else:
            return raw_desc.replace('_', ' ')
    except:
        return raw_desc

def get_bin_structure(df_columns):
    """Identify bin columns and sort them"""
    bins = {}
    for col in df_columns:
        match = re.search(Config.BIN_COL_PATTERN, col)
        if match:
            idx = int(match.group(1))
            desc = match.group(2)
            if idx not in bins:
                bins[idx] = {'desc': desc, 'rho_col': col}
    
    # Sort by bin index
    sorted_bins = sorted(bins.values(), key=lambda x: int(x['rho_col'].split('_')[2]))
    
    # Add readable labels
    for b in sorted_bins:
        b['label'] = parse_bin_label(b['desc'])
        
    return sorted_bins

def load_and_pivot_data(target_var):
    """
    Loads parquet file and pivots it into a matrix:
    Rows: Temperature Bins
    Cols: Moving Average Windows
    Values: Spearman Rho
    """
    input_file = Config.PROJECT_ROOT / "results" / f"spearman_{target_var}.parquet"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    df = pd.read_parquet(input_file)
    if 'omni_variable' in df.columns:
        df = df.rename(columns={'omni_variable': 'variable'})

    # Filter for the specific statistic (e.g., only 'sii_mean')
    pattern = f'sii_{Config.STAT}'
    df = df[df['variable'].str.contains(pattern)].copy()
    
    if df.empty:
        raise ValueError(f"No data found for pattern '{pattern}' in {target_var}")

    # Extract window sizes
    df['window'] = df['variable'].apply(extract_window)
    df = df.dropna(subset=['window']).sort_values('window')
    
    # Get bin columns
    bins = get_bin_structure(df.columns)
    bin_labels = [b['label'] for b in bins]
    windows = sorted(df['window'].unique())
    
    # Create grid
    grid = np.full((len(bins), len(windows)), np.nan)
    
    # Fill grid
    win_map = {w: i for i, w in enumerate(windows)}
    
    for _, row in df.iterrows():
        w = row['window']
        if w in win_map:
            c_idx = win_map[w]
            for r_idx, b in enumerate(bins):
                if b['rho_col'] in row:
                    grid[r_idx, c_idx] = row[b['rho_col']]
                    
    return grid, windows, bin_labels

def plot_fisher_difference(grid_a, grid_b, windows, bin_labels):
    """
    Plots the heatmap of Delta Z = Z(A) - Z(B)
    """
    # 1. Fisher Transform
    z_a = fisher_z_transform(grid_a)
    z_b = fisher_z_transform(grid_b)
    
    # 2. Calculate Difference
    # Positive delta: A responds stronger (or less negatively) than B
    delta_z = z_a - z_b
    
    # Setup Figure
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE, layout="constrained")
    ax.set_facecolor(Config.MISSING_COLOR)
    
# 1. Находим реальный максимум данных
    max_val = np.nanmax(delta_z)
    # Находим минимум (скорее всего это будет > 0, но ставим 0 для логической точки отсчета "нет разницы")
    min_val = 0 
    
    # Добавляем небольшой отступ сверху, чтобы самый яркий цвет не упирался в потолок
    limit_max = max_val * 1.05 if not np.isnan(max_val) and max_val > 0 else 0.1
    
    # 2. Создаем нормализацию от 0 до Max
    norm = mcolors.Normalize(vmin=min_val, vmax=limit_max)
    
    # 3. Выбираем последовательную палитру (Зеленую)
    # Можно попробовать 'Greens', 'Viridis', 'YlGn' (Yellow-Green)
    current_cmap = 'Greens'
        
    # Plot Heatmap
    im = ax.imshow(delta_z, cmap=Config.CMAP_DIFF, norm=norm, aspect='auto', origin='lower')
    
    # Annotate cells (optional, showing Delta Z values)
    # Only show if the difference is "meaningful" (e.g. > 0.02) to avoid clutter
    threshold_text = max_val * 0.5 
        
    for r in range(delta_z.shape[0]):
        for c in range(delta_z.shape[1]):
            val = delta_z[r, c]
            if not np.isnan(val): # Показываем все значения или добавьте порог if abs(val) > 0.01
                # Если значение больше половины максимума (темный фон) -> белый текст
                txt_color = 'white' if val > threshold_text else 'black'
                
                # Опционально: можно не писать слишком маленькие значения, чтобы не засорять карту
                ax.text(c, r, f"{val:.2f}", 
                        ha='center', va='center', color=txt_color, 
                        fontsize=Config.CELL_FONT_SIZE)

    # Axis Labels
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(windows, fontsize=Config.TICK_SIZE)
    ax.set_xlabel("Smoothing Window Size (Days)", fontsize=Config.AXIS_LABEL_SIZE)
    
    ax.set_yticks(np.arange(len(bin_labels)))
    ax.set_yticklabels(bin_labels, fontsize=Config.TICK_SIZE)
    ax.set_ylabel("Temperature Bins", fontsize=Config.AXIS_LABEL_SIZE)
    
    # Title and Explanation
    name_a = Config.PRETTY_NAMES[Config.TARGET_A]
    name_b = Config.PRETTY_NAMES[Config.TARGET_B]
    
    title = f"Differential Sensitivity Map: $\Delta z = z({name_a}) - z({name_b})$"
    subtitle = (f"Statistic: {Config.STAT.upper()} | "
                f"Red: {name_a} dominates | "
                f"Blue: {name_b} dominates")
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=Config.TITLE_SIZE, pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(r'Difference in z-score ($\Delta z$)', fontsize=Config.AXIS_LABEL_SIZE)
    
    # Grid lines for readability
    ax.set_xticks(np.arange(len(windows) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(bin_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Save
    out_path = Config.PROJECT_ROOT / "visualizations" / Config.OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved Fisher Map to: {out_path}")
    print(f"          Comparison: {name_a} vs {name_b}")
    plt.close()

def main():
    print("--- Generating Fisher Z-Difference Map ---")
    try:
        # 1. Load Data
        print(f"Loading {Config.TARGET_A}...")
        grid_a, windows, bins = load_and_pivot_data(Config.TARGET_A)
        
        print(f"Loading {Config.TARGET_B}...")
        grid_b, _, _ = load_and_pivot_data(Config.TARGET_B)
        
        # 2. Check compatibility
        if grid_a.shape != grid_b.shape:
            print("[ERROR] Grids have different shapes. Check binning consistency.")
            return

        # 3. Plot
        plot_fisher_difference(grid_a, grid_b, windows, bins)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()