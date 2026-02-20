#!/usr/bin/env python3
"""
Figure Generator: Stacked Driver Contribution.

**Description:**
This script creates a 2x3 grid of ordinal bar charts to visualize the stacked 
contribution of different drivers (SII, PAR, VPD) to the SIF stress index. 
It uses an earthy color palette and encodes significance levels via opacity.

**Method:**
Ordinal Bar Chart (touching bars) arranged in panels corresponding to 
temperature bins.

**Visual Encoding:**
- Colors: Earthy palette (SII=Red, PAR=Green, VPD=Blue).
- Bars: Ordinal (touching).
- Opacity: Represents p-value significance bins.

**Outputs:**
- PDF images: `reports/figures/fig_2_{variable}_{scenario}.pdf`
- CSV data: `reports/figures/fig_2_{variable}_{scenario}.csv`
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import re
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable

from _Common import Config as CommonConfig

class Config:
    PROJECT_ROOT = CommonConfig.PROJECT_ROOT
    INPUT_DIR = CommonConfig.META_ANALYSIS_DIR 
    OUTPUT_DIR = CommonConfig.REPORTS_ROOT / "figures" 

    # --- FONTS (match Fig.1) ---
    FONT_SCALE = 1.7
    BASE_FONT = 11
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_LABEL_SIZE = int(11 * FONT_SCALE)
    PANEL_LETTER_SIZE = int(18 * FONT_SCALE)

    CBAR_TICK_SIZE = int(9 * FONT_SCALE)   
    CBAR_LABEL_SIZE = int(11 * FONT_SCALE) 
    LEGEND_FONT_SIZE = CBAR_TICK_SIZE
    LEGEND_TITLE_SIZE = CBAR_LABEL_SIZE
     
    # Professional "Earthy" Palette
    COLORS = {
        'SII': '#B03A2E', # Deep Red
        'PAR': '#1D8348', # Deep Green
        'VPD': '#2874A6'  # Steel Blue
    }

    # --- TEMPERATURE BIN RANGES (for panel headers) ---
    TEMP_BIN_RANGES = {
        'Cold': '< 10 °C',
        'Cool': '10–19 °C',
        'Optimum': '19–26 °C',
        'Warm_Stress': '26–31 °C',
        'Extreme_Heat': '> 31 °C'
    }


def add_pvalue_colorbar(fig, cax, vmin):
    cmap = plt.cm.Greys_r

    # safety for LogNorm
    vmin = float(vmin)
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = 1e-15
    vmin = max(vmin, 1e-300)

    norm = mcolors.LogNorm(
            vmin=max(CommonConfig.P_VALUE_LEVELS[0][0], 1e-16),
            vmax=CommonConfig.P_VALUE_LEVELS[-1][0]
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

    FS = Config.CBAR_TICK_SIZE
    cbar.ax.tick_params(labelsize=FS)

    # ticks: decades, plus 0.05
    min_exp = int(np.floor(np.log10(vmin)))
    ticks = [10**e for e in range(min_exp, -1, 3)]  # every 3 decades up to 1e-1
    ticks = [t for t in ticks if (t >= vmin and t <= 1e-1)]
    if 0.05 not in ticks:
        ticks.append(0.05)
    ticks = sorted(set(ticks))

    cbar.set_ticks(ticks)

    def fmt(t):
        if t < 1e-1:
            return rf"$10^{{{int(np.log10(t))}}}$"
        return f"{t:g}"

    cbar.set_ticklabels([fmt(t) for t in ticks])
    return cbar

def p_to_alpha(p_val: float, vmin: float, vmax: float = 0.05) -> float:
    if pd.isna(p_val) or p_val <= 0:
        return 0.0
    p = float(p_val)
    p = min(max(p, vmin), vmax)
    # map log(p) -> [1 .. 0.15]
    lo, hi = np.log10(vmin), np.log10(vmax)
    t = (np.log10(p) - lo) / (hi - lo + 1e-12)  # 0 at vmin, 1 at vmax
    return float((1 - t) * 1.0 + t * 0.15)

def parse_window_features(feature_name: str) -> tuple[str, int]:
    m = re.search(r'ma(\d+)$', str(feature_name))
    if not m:
        return None, None
    window = int(m.group(1))
    feat_lower = str(feature_name).lower()
    if 'sii' in feat_lower:
        return 'SII', window
    if 'par' in feat_lower:
        return 'PAR', window
    if 'vpd' in feat_lower:
        return 'VPD', window
    return None, None

def generate_driver_contribution_figure(target_var: str, scenario: str = 'Control_North', df: pd.DataFrame | None = None):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = int(Config.BASE_FONT * Config.FONT_SCALE)
    input_path = Config.INPUT_DIR / "spearman_overview_summary.csv"
    if not input_path.exists():
        print(f"[ERROR] Source file not found: {input_path}")
        return

    # 1. Load and Preprocess
    if df is None:
        df = pd.read_csv(
            input_path,
            low_memory=False,
            dtype={"parameter_1": "string"}  # fix DtypeWarning (column index 1)
        )

    if target_var not in df['target'].unique():
        print(f"  [WARN] Target '{target_var}' not found in CSV. Skipping.")
        return
    else:
        target_lookup = target_var

    sub = df[(df['target'] == target_lookup) & (df['scenario'] == scenario)].copy()
    if sub.empty: 
        print(f"  [WARN] No data for {target_lookup} in scenario {scenario}. Skipping.")
        return

    sub['var_type'], sub['window'] = zip(*sub['omni_var'].apply(parse_window_features))
    sub = sub.dropna(subset=['var_type', 'window'])
    sub = sub[sub['var_type'].isin(['SII', 'PAR', 'VPD'])]

    # Determine p-range for adaptive colorbar scaling
    pvals_all = sub['p_adj'].to_numpy() # or p_fdr?
    pvals_all = pvals_all[np.isfinite(pvals_all) & (pvals_all > 0)]
    vmin = pvals_all.min() if pvals_all.size else 1e-15

    # Data collection for export
    export_rows = []

    # 2. Setup Grid (2 rows x 3 columns)
    bins_order = ['Cold', 'Cool', 'Optimum', 'Warm_Stress', 'Extreme_Heat']
    present_bins = set(sub['bin_label'].unique())

    # If nothing at all to plot (no overlap with known bins) -> skip
    if len(present_bins.intersection(bins_order)) == 0:
        print(f"  [WARN] No valid temperature bins found for {target_lookup}/{scenario}.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # 3. Iterate Bins IN FIXED ORDER (no shifting!)
    for i, bin_label in enumerate(bins_order):
        ax = axes_flat[i]

        bin_data = sub[sub['bin_label'] == bin_label].copy()
        if bin_data.empty:
            # Keep panel slot but mark as empty
            ax.text(0.5, 0.5, f"No data\n({bin_label})", ha='center', va='center',
                    transform=ax.transAxes, fontsize=Config.TICK_LABEL_SIZE, alpha=0.8)
            # --- Panel label + temperature range ---
            letter = chr(ord('A') + i)

            temp_range = Config.TEMP_BIN_RANGES.get(bin_label, bin_label)

            ax.text(
                0.97, 0.96,
                letter,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=Config.PANEL_LETTER_SIZE,
                fontweight='bold'
            )

            ax.text(
                0.97, 0.85,
                temp_range,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=int(Config.PANEL_LETTER_SIZE * 0.55),
                color='black',
                alpha=0.8
            )

            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', which='major', labelsize=Config.TICK_LABEL_SIZE)
            continue

        # --- ORDINAL MAPPING ---
        unique_windows = sorted(bin_data['window'].unique())
        window_to_idx = {w: i for i, w in enumerate(unique_windows)}
        full_indices = np.arange(len(unique_windows))

        def get_pivoted(col_val, fill_val):
            p = bin_data.pivot_table(index='window', columns='var_type', values=col_val, aggfunc='mean')
            p = p.reindex(unique_windows).fillna(fill_val)
            p.index = [window_to_idx[w] for w in p.index]
            return p.sort_index()

        piv_rho = get_pivoted('rho', 0).abs()
        piv_p = get_pivoted('p_adj', 1.0)

        drivers = ['SII', 'PAR', 'VPD']
        for d in drivers:
            if d not in piv_rho.columns: piv_rho[d] = 0
            if d not in piv_p.columns: piv_p[d] = 1.0

        rho_values = piv_rho[drivers].values
        cumulative_stack = np.cumsum(rho_values, axis=1)
        bases = np.zeros_like(cumulative_stack)
        bases[:, 1:] = cumulative_stack[:, :-1]

        for d_idx, driver in enumerate(drivers):
            heights = rho_values[:, d_idx]
            bottoms = bases[:, d_idx]
            p_vals = piv_p[driver].to_numpy(dtype=float)
            p_vals = np.where(np.isfinite(p_vals) & (p_vals > 0), p_vals, CommonConfig.P_FLOOR)

            base_rgb = mcolors.to_rgb(Config.COLORS[driver])
            rgba = np.zeros((len(full_indices), 4))
            rgba[:, :3] = base_rgb
            rgba[:, 3] = [p_to_alpha(p, vmin=vmin) for p in p_vals]


            ax.bar(full_indices, heights, bottom=bottoms,
                width=1.0, color=rgba, edgecolor=rgba,
                linewidth=0, align='edge')

            for w_idx, win_val in enumerate(unique_windows):
                export_rows.append({
                    'bin_label': bin_label,
                    'window': win_val,
                    'driver': driver,
                    'rho_abs': heights[w_idx],
                    'p_adj': p_vals[w_idx],
                    'alpha_used': rgba[w_idx, 3]
                })

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        def format_func(value, tick_number):
            idx = int(round(value))
            if 0 <= idx < len(unique_windows):
                return str(int(unique_windows[idx]))
            return ""
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.set_xlim(0, len(unique_windows))

        # --- Panel label + temperature range (always) ---
        letter = chr(ord('A') + i)
        temp_range = Config.TEMP_BIN_RANGES.get(bin_label, bin_label)

        ax.text(
            0.97, 0.96,
            letter,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=Config.PANEL_LETTER_SIZE,
            fontweight='bold'
        )
        ax.text(
            0.97, 0.85,
            temp_range,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=int(Config.PANEL_LETTER_SIZE * 0.55),
            color='black',
            alpha=0.8
        )

        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=Config.TICK_LABEL_SIZE)

        n_val = bin_data['n'].iloc[0] if not bin_data.empty else 0
        ax.text(0.97, 0.03, f'n = {int(n_val):,}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=Config.TICK_LABEL_SIZE, color='black')

    # 4. Handle the 6th subplot (Legend Panel)
    for j in range(len(bins_order), len(axes_flat)):  # i.e., from 5 to 5
        ax_leg = axes_flat[j]
        ax_leg.axis('off')
        if j == 5:
            FS = Config.CBAR_TICK_SIZE  

            # --- 1) Drivers ---
            ax_leg.text(
                0.10, 0.95, "Drivers",
                transform=ax_leg.transAxes,
                fontsize=FS,
                ha='left', va='top'
            )

            # Drivers Legend (no title)
            driver_handles = [
                Patch(facecolor=Config.COLORS['SII'], label='Geomagnetic (SII)'),
                Patch(facecolor=Config.COLORS['PAR'], label='Light (PAR)'),
                Patch(facecolor=Config.COLORS['VPD'], label='Water (VPD)')
            ]
            leg_drivers = ax_leg.legend(
                handles=driver_handles,
                loc='upper left',
                fontsize=FS,
                frameon=False,
                bbox_to_anchor=(0.10, 0.90),  # чуть ниже заголовка
                borderaxespad=0.0,
                handlelength=1.2,
                handletextpad=0.6,
                labelspacing=0.4
            )
            ax_leg.add_artist(leg_drivers)

            # --- 2) Transparency ---
            ax_leg.text(
                0.10, 0.50, "Transparency: adjusted p (log scale)",
                transform=ax_leg.transAxes,
                fontsize=FS,
                ha='left', va='top'
            )

            # --- 3) Colorbar ---
            cax_p = ax_leg.inset_axes([0.10, 0.36, 0.80, 0.07])
            add_pvalue_colorbar(fig, cax_p, vmin=vmin)


    # 5. Global Annotations
    fig.text(0.5, 0.02, 
            'Integration Window (Days)', 
            ha='center', 
            fontsize=Config.AXIS_FONT_SIZE+2
        )
    fig.text(0.035, 0.5,
            r'Stacked Magnitude of Associations (|$\rho$|)',
            va='center', rotation='vertical',
            fontsize=Config.AXIS_FONT_SIZE+2
        )

    for i in range(3):  # panels A, B, C
        ax = axes_flat[i]

        # Show tick labels
        ax.tick_params(
            axis='x',
            which='both',
            labelbottom=True
        )

    # 6. Save
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = target_var.replace("sif_", "")
    out_file = Config.OUTPUT_DIR / f"fig_2_{suffix}_{scenario}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file.name}")

    # Export Data
    if export_rows:
        out_csv = Config.OUTPUT_DIR / f"fig_2_{suffix}_{scenario}.csv"
        pd.DataFrame(export_rows).to_csv(out_csv, index=False)
        print(f"[SUCCESS] Exported data: {out_csv.name}")

def main():
    # 1. Targets (from config)
    targets = getattr(CommonConfig, "SPEARMAN_TARGETS", 
                      ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"])
    
    # 2. Scenarios (ALL from config, including SAA)
    scenarios = list(CommonConfig.SCENARIO_MASKS.keys())
    
    print(f"[INFO] Generating plots for {len(targets)} targets x {len(scenarios)} scenarios...")
    
    input_path = Config.INPUT_DIR / "spearman_overview_summary.csv"
    df = pd.read_csv(input_path, low_memory=False, dtype={"parameter_1": "string"})
    for target in targets:
        for sc in scenarios:
            try:
                generate_driver_contribution_figure(target, scenario=sc, df=df)
            except Exception as e:
                print(f"  [ERROR] Failed {target} - {sc}: {e}")

if __name__ == "__main__":
    main()