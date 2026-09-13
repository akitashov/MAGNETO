#!/usr/bin/env python3
"""
Figure 1 Generator: Descriptive correlation spectra for SIF indicators.

**Description:**
This script generates a multi-panel visualization of the Spearman correlation
between solar/ionospheric indices and SIF (Solar-Induced Fluorescence) across
different integration windows. It also exports the source data used for the
visualization to a CSV file.

The figure is purely descriptive: curves show Spearman's rho against the
integration window with uniform opacity and a fixed line style. No confidence
intervals and no p-value-derived transparency, line styles, or colorbars are
shown.

**Layout:**
- Panel A (Top): Global High LAI response (spans three columns).
- Panels B, C, D (Bottom): Regional/Control scenarios (one column each).
    - B: Northern Hemisphere Control.
    - C: South Atlantic Anomaly (SAA) High LAI.
    - D: Sahara Barren (Background control).
- A dedicated right-hand column holds the combined legend (F10.7 overlay and
  temperature bins), so it never overlaps the curves.

**Visual Encoding:**
- Color Mapping: Line colors correspond to discrete Mean Temperature bins.
- All SII curves use one fixed line style and uniform opacity.
- Overlays: Panel A includes a gold line for the Solar radio flux (F10.7)
  correlation in a representative thermal bin, drawn with the same fixed
  opacity and line style.

**Outputs:**
- PDF images: `reports/figures/fig_1_{variable}.pdf`
- CSV data: `reports/figures/fig_1_{variable}.csv`
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import re
from pathlib import Path
from _Common import Config as CommonConfig

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    PROJECT_ROOT = CommonConfig.PROJECT_ROOT
    INPUT_DIR = CommonConfig.RESULTS_DIR
    OUTPUT_DIR = CommonConfig.REPORTS_ROOT / "figures"

    PANEL_SCENARIOS = {
        "A": "Global_High_LAI",
        "B": "Control_North",
        "C": "SAA_High_LAI",
        "D": "Sahara_Barren",
    }

    TARGET_VAR = ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"]
    OMNI_STAT = "mean"

    # Visual style
    FIG_SIZE = (13.5, 8.8)
    CMAP_NAME = "coolwarm"
    LINE_WIDTH = 2.4
    LINE_ALPHA = 0.95

    # Fonts
    FONT_SCALE = 1.7
    BASE_FONT = 11
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_FONT_SIZE = int(11 * FONT_SCALE)
    PANEL_LETTER_SIZE = int(18 * FONT_SCALE)
    LEGEND_FONT_SIZE = int(9 * FONT_SCALE)

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_window(val_str):
    match = re.search(r"ma(\d+)", str(val_str))
    return int(match.group(1)) if match else None

def plot_bin_curve(ax, bin_data, color):
    """Plot a single descriptive rho-vs-window curve with fixed style."""
    ax.plot(
        bin_data["window"].to_numpy(),
        bin_data["rho"].to_numpy(),
        color=color,
        lw=Config.LINE_WIDTH,
        alpha=Config.LINE_ALPHA,
        ls="-",
        zorder=2,
    )

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_scenario_df(scenario: str, target_var: str) -> pd.DataFrame:
    csv_file = Config.INPUT_DIR / f"spearman_{target_var}_{scenario}.csv"
    if not csv_file.exists():
        print(f"[WARN] Missing: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)

    # Keep only SII MA windows (matches your original intent)
    pattern = f"sii_{Config.OMNI_STAT}_ma"
    df = df[df["omni_var"].astype(str).str.contains(pattern)].copy()
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["omni_var"].apply(extract_window)
    df = df.dropna(subset=["window"]).sort_values("window")
    return df

def load_f107_df(scenario: str, target_var: str) -> pd.DataFrame:
    csv_file = Config.INPUT_DIR / f"spearman_{target_var}_{scenario}.csv"
    if not csv_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    df = df[df["omni_var"].astype(str).str.contains("f10_7_mean_ma")].copy()
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["omni_var"].apply(extract_window)
    df = df.dropna(subset=["window"]).sort_values("window")
    return df

# ==============================================================================
# PLOTTING
# ==============================================================================

def draw_panel(ax, df):
    if df.empty:
        ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
        return (None, None, [], 0)

    bin_ids = sorted(df["bin_id"].unique())
    n_bins = len(bin_ids)
    cmap = plt.get_cmap(Config.CMAP_NAME)
    norm = mcolors.Normalize(vmin=0, vmax=max(n_bins - 1, 1))

    for i, b_id in enumerate(bin_ids):
        bin_data = df[df["bin_id"] == b_id].sort_values("window")
        if bin_data.empty:
            continue
        plot_bin_curve(ax, bin_data, cmap(norm(i)))

    ax.axhline(0, color="black", lw=1.0, alpha=0.8)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=Config.TICK_FONT_SIZE)
    return (cmap, norm, bin_ids, n_bins)

def process_variable(target_var):
    print(f"[INFO] Processing: {target_var}...")

    data = {}
    export_list = []

    for letter, scen in Config.PANEL_SCENARIOS.items():
        df = load_scenario_df(scen, target_var)
        data[letter] = df

        if not df.empty:
            df_export = df.copy()
            df_export["panel"] = letter
            df_export["scenario"] = scen
            export_list.append(df_export)

    fig = plt.figure(figsize=Config.FIG_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[2.2, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 0.72],
    )

    axA = fig.add_subplot(gs[0, :3])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])
    axLegend = fig.add_subplot(gs[:, 3])
    axLegend.axis("off")
    axes = {"A": axA, "B": axB, "C": axC, "D": axD}

    cmapA, normA, bin_idsA, n_binsA = draw_panel(axA, data["A"])
    draw_panel(axB, data["B"])
    draw_panel(axC, data["C"])
    draw_panel(axD, data["D"])

    # F10.7 Overlay (Panel A)
    f107_handle = None
    df_f107 = load_f107_df(Config.PANEL_SCENARIOS["A"], target_var)
    if not df_f107.empty:
        target_bin = 1
        if target_bin not in df_f107["bin_id"].unique():
            target_bin = 0
        bin_data = df_f107[df_f107["bin_id"] == target_bin].sort_values("window")
        if not bin_data.empty:
            axA.plot(
                bin_data["window"].to_numpy(),
                bin_data["rho"].to_numpy(),
                color="gold",
                lw=Config.LINE_WIDTH,
                alpha=Config.LINE_ALPHA,
                ls="-",
                zorder=2,
            )

            range_str = CommonConfig.TEMP_RANGES.get(target_bin, str(target_bin))
            f107_handle = Line2D(
                [0], [0],
                color="gold",
                lw=Config.LINE_WIDTH,
                label=f"F10.7, {range_str}"
            )

            f107_export = bin_data.copy()
            f107_export["panel"] = "A"
            f107_export["scenario"] = Config.PANEL_SCENARIOS["A"] + "_F10.7_Overlay"
            export_list.append(f107_export)

    # Legend (dedicated right-hand column)
    combined_handles = []
    if f107_handle:
        combined_handles.append(f107_handle)

    if cmapA and n_binsA > 0:
        for i in reversed(range(n_binsA)):
            color = cmapA(normA(i))
            b_id = bin_idsA[i]
            range_str = CommonConfig.TEMP_RANGES.get(b_id, f"Bin {b_id}")
            label = f"SII, {range_str}"
            combined_handles.append(Line2D([0], [0], color=color, lw=Config.LINE_WIDTH, label=label))

    if combined_handles:
        axLegend.legend(
            handles=combined_handles,
            loc="center left",
            fontsize=Config.LEGEND_FONT_SIZE,
            frameon=False,
        )

    # Global Y scaling
    all_rho = []
    for df in data.values():
        if "rho" in df.columns:
            all_rho.append(df["rho"].to_numpy())
    if all_rho:
        y = np.concatenate([arr[np.isfinite(arr)] for arr in all_rho])
        if len(y) > 0:
            max_val = np.max(np.abs(y))
            m = max(max_val * 1.15, 0.05)
            locator = mticker.MaxNLocator(nbins=5, symmetric=True)
            formatter = mticker.FormatStrFormatter("%.2f")
            for ax in axes.values():
                ax.set_ylim(-m, m)
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_major_formatter(formatter)

    # Axis labels
    axC.tick_params(labelleft=False)
    axD.tick_params(labelleft=False)
    axA.set_ylabel(r"Spearman Correlation ($\rho$)", fontsize=Config.AXIS_FONT_SIZE)

    for letter in ["C"]:
        axes[letter].set_xlabel("Integration Window (Days)", fontsize=Config.AXIS_FONT_SIZE)

    for letter, ax in axes.items():
        ax.text(0.01, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
                fontsize=Config.PANEL_LETTER_SIZE, fontweight="bold")

    # Save
    suffix = target_var.replace("sif_", "")
    out_file = Config.OUTPUT_DIR / f"fig_1_{suffix}.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved: {out_file}")

    # Export data
    if export_list:
        final_export_df = pd.concat(export_list, ignore_index=True)
        csv_file = Config.OUTPUT_DIR / f"fig_1_{suffix}.csv"
        final_export_df.to_csv(csv_file, index=False)
        print(f"[SUCCESS] Exported data: {csv_file}")

def main():
    print("[INFO] Starting Figure 1 Generation Sequence...")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = int(Config.BASE_FONT * Config.FONT_SCALE)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = Config.TARGET_VAR
    if isinstance(targets, str):
        targets = [targets]

    for var in targets:
        process_variable(var)

if __name__ == "__main__":
    main()
