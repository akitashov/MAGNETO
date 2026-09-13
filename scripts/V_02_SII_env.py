#!/usr/bin/env python3
"""
Figure Generator: Stacked Driver Contribution (descriptive).

**Description:**
This script creates a 2x3 grid of ordinal bar charts to visualize the stacked
magnitude of associations between the SIF stress index and three drivers
(geomagnetic SII, PAR, VPD) across integration windows, per temperature bin.
It uses an earthy color palette and draws every bar with uniform opacity.

This is a descriptive comparison of association magnitudes: absolute Spearman
|rho| values are stacked per driver. No inferential significance encoding is
applied.

**Method:**
Ordinal Bar Chart (touching bars) arranged in panels corresponding to
temperature bins.

**Visual Encoding:**
- Colors: Earthy palette (SII=Red, PAR=Green, VPD=Blue).
- Bars: Ordinal (touching), uniform opacity (alpha=1.0).
- No p-value-based transparency, legends, or colorbars.

**Outputs:**
- PDF images: `reports/figures/fig_2_{variable}_{scenario}.pdf`
- CSV data: `reports/figures/fig_2_{variable}_{scenario}.csv`
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
from pathlib import Path
from matplotlib.patches import Patch

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

    LEGEND_FONT_SIZE = int(9 * FONT_SCALE)
    LEGEND_TITLE_SIZE = int(11 * FONT_SCALE)

    # Bar opacity is uniform across all bars (descriptive figure)
    BAR_ALPHA = 1.0

    # Professional "Earthy" Palette
    COLORS = {
        'SII': '#B03A2E', # Deep Red
        'PAR': '#1D8348', # Deep Green
        'VPD': '#2874A6'  # Steel Blue
    }

    # --- TEMPERATURE BIN RANGES (for panel headers) ---
    # Derived from Config to guarantee consistency with the canonical bins.
    TEMP_BIN_RANGES = {
        label: CommonConfig.TEMP_RANGES.get(i, label)
        for i, label in enumerate(CommonConfig.TEMP_LABELS_PHYSIO)
    }


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

            ax.bar(full_indices, heights, bottom=bottoms,
                width=1.0, color=Config.COLORS[driver],
                alpha=Config.BAR_ALPHA,
                edgecolor='none',
                linewidth=0, align='edge')

            for w_idx, win_val in enumerate(unique_windows):
                export_rows.append({
                    'bin_label': bin_label,
                    'window': win_val,
                    'driver': driver,
                    'rho_abs': heights[w_idx],
                    'p_adj': p_vals[w_idx],
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
            FS = Config.LEGEND_FONT_SIZE

            # --- Drivers ---
            ax_leg.text(
                0.10, 0.95, "Drivers",
                transform=ax_leg.transAxes,
                fontsize=Config.LEGEND_TITLE_SIZE,
                ha='left', va='top'
            )

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
                bbox_to_anchor=(0.10, 0.90),
                borderaxespad=0.0,
                handlelength=1.2,
                handletextpad=0.6,
                labelspacing=0.4
            )
            ax_leg.add_artist(leg_drivers)

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
