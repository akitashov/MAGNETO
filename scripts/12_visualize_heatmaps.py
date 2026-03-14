#!/usr/bin/env python3
"""
Visualization Module: Correlation Heatmaps (Batch Processing).
ENHANCED VERSION:
1. Dynamic detection of bin columns.
2. Axes Swapped: X-axis = Window Size, Y-axis = Temperature.
3. SMART COLOR SCHEMES: Different colormaps for all-positive/all-negative data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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

    VARIABLE_PATTERN = r'^sii_(?:mean|max|min|std)_ma\d+$'
    BIN_COL_PATTERN = r'rho_bin_(\d+)_(.+)$'
    
    P_VALUE_LEVELS = [(0.05, 1.0), (0.1, 0.5), (1.0, 0.25)]
    SHOW_TEXT_P_THRESHOLD = 0.05
    
    FIG_SIZE = (18, 10)
    
    # COLOR SCHEMES FOR DIFFERENT DATA DISTRIBUTIONS
    # Standard bidirectional data (-1 to 1)
    CMAP_BIDIRECTIONAL = 'coolwarm'  # blue (-) to red (+)
    
    # All positive data (0 to +max)
    CMAP_POSITIVE = 'YlOrRd'  # yellow (weak) to red (strong) - хорошая читаемость
    # Альтернативы: 'viridis', 'plasma', 'summer', 'hot'
    
    # All negative data (-max to 0)
    CMAP_NEGATIVE = 'PuBu'  # purple (weak) to blue (strong) - интуитивно для отрицательных
    # Альтернативы: 'Blues_r', 'cool', 'winter', 'bone_r'
    
    FONT_FAMILY = 'DejaVu Sans'
    TITLE_SIZE = 18
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 11
    CELL_FONT_SIZE = 8
    MISSING_COLOR = '#d9d9d9'

def format_p_value(val):
    if np.isnan(val): 
        return ""
    
    if val == 0 or val < 1e-16:
        return r"$p < 10^{-16}$"
    
    if val < 0.001:
        exponent = int(math.floor(math.log10(val)))
        mantissa = val / (10**exponent)
        return r"$p={:.1f}{{\cdot}}10^{{{}}}$".format(mantissa, exponent)
    
    return f"$p={val:.3f}$"

def extract_window(val_str):
    match = re.search(r'ma(\d{1,3})', str(val_str))
    return int(match.group(1)) if match else None

def calculate_alpha(p_val):
    if np.isnan(p_val): return 0.0
    for threshold, alpha in Config.P_VALUE_LEVELS:
        if p_val <= threshold: return alpha
    return Config.P_VALUE_LEVELS[-1][1]

def parse_bin_label(raw_desc):
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

def determine_colormap_scheme(rho_values):
    """
    Определяет, какую цветовую схему использовать на основе данных.
    Возвращает (cmap_name, norm) для построения графика.
    
    Стратегии:
    1. Если есть и положительные, и отрицательные значения -> бинарная схема
    2. Если все значения ≥ 0 -> шкала для положительных
    3. Если все значения ≤ 0 -> шкала для отрицательных
    """
    # Убираем NaN
    valid_vals = rho_values[~np.isnan(rho_values)]
    
    if len(valid_vals) == 0:
        # Нет данных - возвращаем стандартную
        return Config.CMAP_BIDIRECTIONAL, mcolors.Normalize(vmin=-1, vmax=1)
    
    min_val = np.min(valid_vals)
    max_val = np.max(valid_vals)
    
    # Определяем тип распределения
    if min_val >= 0:
        # Все значения неотрицательные
        # Используем 0 как нижнюю границу, max_val как верхнюю
        # Добавляем небольшой запас для визуализации
        upper_limit = max_val * 1.05 if max_val > 0 else 0.01
        norm = mcolors.Normalize(vmin=0, vmax=upper_limit)
        return Config.CMAP_POSITIVE, norm
        
    elif max_val <= 0:
        # Все значения неположительные
        # Используем min_val как нижнюю границу, 0 как верхнюю
        lower_limit = min_val * 1.05 if min_val < 0 else -0.01
        norm = mcolors.Normalize(vmin=lower_limit, vmax=0)
        return Config.CMAP_NEGATIVE, norm
        
    else:
        # Смешанные значения
        # Используем симметричную шкалу
        abs_limit = max(abs(min_val), abs(max_val)) * 1.05
        norm = mcolors.Normalize(vmin=-abs_limit, vmax=abs_limit)
        return Config.CMAP_BIDIRECTIONAL, norm

def generate_heatmap(df_target, target_var, stat_type):
    pattern = f'sii_{stat_type}'
    df_plot = df_target[df_target['variable'].str.contains(pattern)].copy()
    
    if df_plot.empty: 
        print(f"  [SKIP] No data for {pattern}")
        return

    # Extract Windows
    df_plot['window_size'] = df_plot['variable'].apply(extract_window)
    df_plot = df_plot.dropna(subset=['window_size'])
    df_plot['window_size'] = df_plot['window_size'].astype(int)
    
    df_plot = df_plot.sort_values('window_size')
    
    unique_windows = sorted(df_plot['window_size'].unique())
    win_to_idx = {w: i for i, w in enumerate(unique_windows)}
    
    bin_infos = get_bin_info(df_plot.columns)
    if not bin_infos:
        print("  [WARN] No bin columns found (check column naming format).")
        return

    bin_labels = [b['label'] for b in bin_infos]
    
    n_rows = len(bin_infos)
    n_cols = len(unique_windows)
    
    rho_grid = np.full((n_rows, n_cols), np.nan)
    p_grid = np.full((n_rows, n_cols), np.nan)
    
    for _, row in df_plot.iterrows():
        w_val = row['window_size']
        if w_val not in win_to_idx: continue
        col_idx = win_to_idx[w_val]
        
        for row_idx, bin_info in enumerate(bin_infos):
            rho_col = bin_info['rho_col']
            p_col = bin_info['p_col']
            
            if rho_col in row:
                rho_grid[row_idx, col_idx] = row[rho_col]
            if p_col in row:
                p_grid[row_idx, col_idx] = row[p_col]

    # --- SMART COLOR SCHEME SELECTION ---
    
    valid_rhos = rho_grid[~np.isnan(rho_grid)]
    if len(valid_rhos) == 0: 
        print("  [WARN] Grid is empty (all NaNs).")
        return
    
    # Определяем цветовую схему и нормализацию
    cmap_name, norm = determine_colormap_scheme(rho_grid)
    cmap = plt.get_cmap(cmap_name)
    
    # --- PLOTTING ---
    
    rgba_image = np.zeros((n_rows, n_cols, 4))
    
    for r in range(n_rows):
        for c in range(n_cols):
            rho, p = rho_grid[r, c], p_grid[r, c]
            if np.isnan(rho) or np.isnan(p):
                rgba_image[r, c] = (0,0,0,0) # Transparent
            else:
                color = cmap(norm(rho))
                rgba_image[r, c] = (color[0], color[1], color[2], calculate_alpha(p))

    plt.rcParams['font.family'] = Config.FONT_FAMILY
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE, layout="constrained")
    ax.set_facecolor(Config.MISSING_COLOR)
    
    im = ax.imshow(rgba_image, aspect='auto', origin='lower', interpolation='nearest')
    
    # X-Axis: Windows
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(unique_windows, fontsize=Config.TICK_SIZE)
    ax.set_xlabel("Smoothing Window Size (Days)", fontsize=Config.AXIS_LABEL_SIZE)
    
    # Y-Axis: Temperatures
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(bin_labels, fontsize=Config.TICK_SIZE)
    ax.set_ylabel("Temperature Bins", fontsize=Config.AXIS_LABEL_SIZE)
    
    # Minor grid
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    pretty_target = Config.PRETTY_NAMES.get(target_var, target_var)
    
    # Динамический заголовок с информацией о цветовой схеме
    if cmap_name == Config.CMAP_POSITIVE:
        scheme_info = "(all correlations ≥ 0)"
    elif cmap_name == Config.CMAP_NEGATIVE:
        scheme_info = "(all correlations ≤ 0)"
    else:
        scheme_info = "(mixed correlations)"
    
    ax.set_title(f"Heatmap: SII {stat_type.upper()} vs {pretty_target} {scheme_info}", 
                 fontsize=Config.TITLE_SIZE)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    
    # Динамическая подпись colorbar
    if cmap_name == Config.CMAP_POSITIVE:
        cbar_label = 'Spearman Correlation (ρ ≥ 0)'
    elif cmap_name == Config.CMAP_NEGATIVE:
        cbar_label = 'Spearman Correlation (ρ ≤ 0)'
    else:
        cbar_label = r'Spearman Correlation ($\rho$)'
    
    cbar.set_label(cbar_label, fontsize=Config.AXIS_LABEL_SIZE)
    
    # Text Annotations
    for r in range(n_rows):
        for c in range(n_cols):
            rho, p = rho_grid[r, c], p_grid[r, c]
            if not np.isnan(p) and p <= Config.SHOW_TEXT_P_THRESHOLD:
                # Динамическое определение цвета текста
                if not np.isnan(rho):
                    # Для монотонных шкал используем контрастный текст
                    if cmap_name == Config.CMAP_POSITIVE:
                        # Для YlOrRd: темный текст на светлых тонах, белый на темных
                        norm_rho = norm(rho) if hasattr(norm, '__call__') else (rho - norm.vmin) / (norm.vmax - norm.vmin)
                        txt_color = 'black' if norm_rho < 0.7 else 'white'
                    elif cmap_name == Config.CMAP_NEGATIVE:
                        # Для PuBu: темный текст на светлых тонах
                        norm_rho = norm(rho) if hasattr(norm, '__call__') else (rho - norm.vmin) / (norm.vmax - norm.vmin)
                        txt_color = 'black' if norm_rho > 0.3 else 'white'
                    else:
                        # Для coolwarm: как было
                        txt_color = 'white' if abs(rho) > 0.6 else 'black'
                else:
                    txt_color = 'black'
                
                p_str = format_p_value(p)
                label_text = f"$\\rho$={rho:.2f}\n{p_str}"
                ax.text(c, r, label_text, ha='center', va='center', 
                        color=txt_color, fontsize=Config.CELL_FONT_SIZE)
    
    # Legend
    legend_elements = [Patch(facecolor='gray', edgecolor='k', alpha=lvl[1], label=f'p ≤ {lvl[0]}') 
                       for lvl in Config.P_VALUE_LEVELS[:-1]]
    legend_elements.append(Patch(facecolor='gray', edgecolor='k', 
                                 alpha=Config.P_VALUE_LEVELS[-1][1], label='Not Sig.'))
    legend_elements.append(Patch(facecolor=Config.MISSING_COLOR, edgecolor='k', label='No Data (<50)'))
    
    # Добавляем информацию о цветовой схеме в легенду
    color_scheme_info = f"Color scheme: {cmap_name}"
    if cmap_name == Config.CMAP_POSITIVE:
        color_scheme_info += " (all positive)"
    elif cmap_name == Config.CMAP_NEGATIVE:
        color_scheme_info += " (all negative)"
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
              ncol=len(legend_elements), frameon=False, fontsize=Config.TICK_SIZE-2)

    output_file = Config.PROJECT_ROOT / "visualizations" / f"heatmap_{target_var}_sii_{stat_type}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  [OK] Saved: {output_file.name} (using {cmap_name} colormap)")

def process_target(target_var):
    input_file = Config.PROJECT_ROOT / "results" / f"spearman_{target_var}.parquet"
    if not input_file.exists():
        print(f"[WARN] Skipping {target_var}: File not found.")
        return

    print(f"--- Processing {target_var} ---")
    df = pd.read_parquet(input_file)
    if 'omni_variable' in df.columns:
        df = df.rename(columns={'omni_variable': 'variable'})

    for stat in Config.STATS:
        generate_heatmap(df, target_var, stat)

def main():
    print("Starting Batch Heatmap Visualization with Smart Color Schemes...")
    print(f"Color schemes configured:")
    print(f"  - Bidirectional (mixed): {Config.CMAP_BIDIRECTIONAL}")
    print(f"  - All positive: {Config.CMAP_POSITIVE}")
    print(f"  - All negative: {Config.CMAP_NEGATIVE}")
    
    for target in Config.TARGETS:
        try:
            process_target(target)
        except Exception as e:
            print(f"[ERROR] Failed {target}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()