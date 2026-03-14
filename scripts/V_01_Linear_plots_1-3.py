#!/usr/bin/env python3
"""
Figure 1 Generator: Cumulative Response (Spearman rho) for SIF indicators.

**Description:**
This script generates a multi-panel visualization of the Spearman correlation 
between solar/ionospheric indices and SIF (Solar-Induced Fluorescence) across 
different integration windows. It also exports the source data used for the 
visualization to a CSV file.
- Uses p_fdr (BH-adjusted) for transparency / line style.
- Keeps the original p_adj (Neff-adjusted) in the exported CSV as well.
- For each (scenario, target_var, bin_id), apply BH across all tested SII windows in that bin.
  (i.e., multiple comparisons are the windows within a given thermal bin and scenario.)

**Layout:**
- Panel A (Top): Global High LAI response (Full Width).
- Panels B, C, D (Bottom): Regional/Control scenarios (1/3 Width each).
    - B: Northern Hemisphere Control.
    - C: South Atlantic Anomaly (SAA) High LAI.
    - D: Sahara Barren (Background control).

**Visual Encoding:**
- Lines & Shading: Represent Spearman's rho and Fisher-transformed 95% CI.
- Color Mapping: Line colors correspond to discrete Mean Temperature bins.
- Alpha/Line Style: Significance (p-adj) is encoded via transparency and 
  line style (solid for p < 0.01, dashed otherwise).
- Overlays: Panel A includes a gold line for Solar radiation at 10.7cm correlation.
- Legend & Colorbar: Combined discrete legend for Temperature Bins/F10.7 and 
  an inset horizontal adjusted p-value colorbar in Panel A, placed in the lower right area.

**Outputs:**
- PDF images: `reports/figures/fig_1_{variable}.pdf`
- CSV data: `reports/figures/fig_1_{variable}.csv`
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
    if CommonConfig.LOWER_FIG_LETTERS:
        PANEL_SCENARIOS = {k.lower(): v for k,v in PANEL_SCENARIOS.items()}

    TARGET_VAR = ["sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index"]
    OMNI_STAT = "mean"

    # ---- NEW: FDR ----
    USE_FDR = True
    FDR_METHOD = "BH"     # (kept for readability; implementation is BH)
    FDR_Q = 0.05          # Typical reporting level; used only for legend text if you want

    # Visual style
    FIG_SIZE = (12, 9)
    CMAP_NAME = "coolwarm"
    LINE_WIDTH = 2.4
    CI_ALPHA_FACTOR = 0.25

    # Fonts
    FONT_SCALE = 1.7
    BASE_FONT = 11
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_FONT_SIZE = int(11 * FONT_SCALE)
    PANEL_LETTER_SIZE = int(18 * FONT_SCALE)
    CBAR_LABEL_SIZE = int(11 * FONT_SCALE)
    CBAR_TICK_SIZE = int(9 * FONT_SCALE)

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_window(val_str):
    match = re.search(r"ma(\d+)", str(val_str))
    return int(match.group(1)) if match else None

def p_to_alpha(p_val):
    if np.isnan(p_val):
        return 0.0
    for threshold, alpha in CommonConfig.P_VALUE_LEVELS:
        if p_val <= threshold:
            return alpha
    return CommonConfig.P_VALUE_LEVELS[-1][1]

def calculate_fisher_ci(df):
    """Autofills missing stats (CI, P-adj) if needed."""
    if "p_adj" not in df.columns:
        df["p_adj"] = df.get("p_value", 1.0)

    if "temp_mean" not in df.columns and "bin_id" in df.columns:
        df["temp_mean"] = df["bin_id"].map(Config.TEMP_PROXY)

    if "ci_lower" not in df.columns and "rho" in df.columns:
        n = df.get("n_eff", df.get("n", 100))
        r = df["rho"].clip(-0.99, 0.99)
        z = np.arctanh(r)
        sigma = 1.0 / np.sqrt(np.maximum(n - 3, 1))
        df["ci_lower"] = np.tanh(z - 1.96 * sigma)
        df["ci_upper"] = np.tanh(z + 1.96 * sigma)
    return df

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg adjusted p-values (q-values) for a 1D array.
    Returns array of same shape with NaNs preserved.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out

    p_ok = p[ok]
    m = p_ok.size

    order = np.argsort(p_ok)
    p_sorted = p_ok[order]

    # BH: q_i = min_{j>=i} (m/j)*p_(j)
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = (m / ranks) * p_sorted
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_ok = np.empty_like(p_ok)
    q_ok[order] = q_sorted

    out[ok] = q_ok
    return out

def add_fdr_by_bin(df: pd.DataFrame, p_col_in="p_adj", out_col="p_fdr") -> pd.DataFrame:
    """
    Apply BH FDR across windows within each bin_id (and within whatever slice is passed in).
    Assumes df already filtered to the feature family you want (e.g., only SII MA windows).
    """
    df = df.copy()
    df[out_col] = np.nan
    if df.empty:
        return df

    for b_id, g in df.groupby("bin_id"):
        q = bh_fdr(g[p_col_in].to_numpy())
        df.loc[g.index, out_col] = q
    return df

def pcol_to_use(df: pd.DataFrame) -> str:
    """
    Choose which p-value column to use for encoding.
    """
    if Config.USE_FDR:
        return "p_fdr" if "p_fdr" in df.columns else "p_adj"
    return "p_adj"

def plot_segmented_ci(ax, x, y_low, y_high, p_vals, color):
    p_vals = np.asarray(p_vals, dtype=float)
    p_vals = np.where(np.isfinite(p_vals) & (p_vals > 0), p_vals, CommonConfig.P_FLOOR)

    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        line_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0
        ci_alpha = line_alpha * Config.CI_ALPHA_FACTOR
        if ci_alpha < 0.01:
            continue
        ax.fill_between(
            [x[i], x[i + 1]],
            [y_low[i], y_low[i + 1]],
            [y_high[i], y_high[i + 1]],
            color=color,
            alpha=ci_alpha,
            edgecolor="none",
            zorder=0
        )

def plot_gradient_line(ax, x, y, p_vals, color):
    p_vals = np.asarray(p_vals, dtype=float)
    p_vals = np.where(np.isfinite(p_vals) & (p_vals > 0), p_vals, CommonConfig.P_FLOOR)

    point_alphas = [p_to_alpha(p) for p in p_vals]
    for i in range(len(x) - 1):
        seg_alpha = (point_alphas[i] + point_alphas[i + 1]) / 2.0

        # solid if p < 0.01 else dashed
        p_mid = np.nanmean([p_vals[i], p_vals[i + 1]])
        ls = "-" if (np.isfinite(p_mid) and p_mid < 0.01) else "--"

        ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                color=color, alpha=seg_alpha, ls=ls, lw=Config.LINE_WIDTH, zorder=2)

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
    df = calculate_fisher_ci(df)

    # ---- FDR across windows within each bin ----
    if Config.USE_FDR:
        df = add_fdr_by_bin(df, p_col_in="p_adj", out_col="p_fdr")

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
    df = calculate_fisher_ci(df)

    # Optional: apply FDR for the F10.7 curve too (within-bin across windows).
    # (Here it’s just one bin plotted; still multiple windows.)
    if Config.USE_FDR:
        df = add_fdr_by_bin(df, p_col_in="p_adj", out_col="p_fdr")

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

    pcol = pcol_to_use(df)

    for i, b_id in enumerate(bin_ids):
        bin_data = df[df["bin_id"] == b_id].sort_values("window")
        if bin_data.empty:
            continue

        c = cmap(norm(i))

        plot_segmented_ci(
            ax,
            bin_data["window"].to_numpy(),
            bin_data["ci_lower"].to_numpy(),
            bin_data["ci_upper"].to_numpy(),
            bin_data[pcol].to_numpy(),
            color=c
        )
        plot_gradient_line(
            ax,
            bin_data["window"].to_numpy(),
            bin_data["rho"].to_numpy(),
            bin_data[pcol].to_numpy(),
            color=c
        )

    ax.axhline(0, color="black", lw=1.0, alpha=0.8)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=Config.TICK_FONT_SIZE)
    return (cmap, norm, bin_ids, n_bins)

def add_pvalue_colorbar(fig, cax, vmin):
    cmap = plt.cm.Greys_r
    vmin = max(float(vmin), 1e-300)  # safety for LogNorm
    norm = mcolors.LogNorm(vmin=vmin, vmax=1e-1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    label = "Transparency: "
    label += "BH-FDR adjusted p (log scale)" if Config.USE_FDR else "adjusted p (log scale)"
    cbar.set_label(label, fontsize=Config.CBAR_TICK_SIZE)
    cbar.ax.tick_params(labelsize=Config.CBAR_TICK_SIZE)

    # Choose ticks based on vmin
    # Start at nearest decade below vmin
    min_exp = int(np.floor(np.log10(vmin)))
    ticks = [10**e for e in range(min_exp, -0, 3)]  # every 3 decades up to 1e0 (we'll clip)
    ticks = [t for t in ticks if (t >= vmin and t <= 1e-1)]
    # Always include 0.05 for readability
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
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2.2, 1.0])

    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])
    keys = list(Config.PANEL_SCENARIOS.keys())
    axes = {keys[0]: axA, keys[1]: axB, keys[2]: axC, keys[3]: axD}

    cmapA, normA, bin_idsA, n_binsA = draw_panel(axA, data[keys[0]])
    draw_panel(axB, data[keys[1]])
    draw_panel(axC, data[keys[2]])
    draw_panel(axD, data[keys[3]])

    # F10.7 Overlay (Panel A)
    f107_handle = None
    df_f107 = load_f107_df(Config.PANEL_SCENARIOS[keys[0]], target_var)
    if not df_f107.empty:
        target_bin = 1
        if target_bin not in df_f107["bin_id"].unique():
            target_bin = 0
        bin_data = df_f107[df_f107["bin_id"] == target_bin].sort_values("window")
        if not bin_data.empty:
            pcol = pcol_to_use(bin_data)
            plot_gradient_line(
                axA,
                bin_data["window"].to_numpy(),
                bin_data["rho"].to_numpy(),
                bin_data[pcol].to_numpy(),
                color="gold"
            )

            f107_handle = Line2D(
                [0], [0],
                color="gold",
                lw=Config.LINE_WIDTH,
                label=f"F10.7 Solar radiation at {CommonConfig.TEMP_RANGES.get(target_bin, str(target_bin))}"
            )

            f107_export = bin_data.copy()
            f107_export["panel"] = keys[0]
            f107_export["scenario"] = Config.PANEL_SCENARIOS[keys[0]] + "_F10.7_Overlay"
            export_list.append(f107_export)

    # Legend (Panel A)
    combined_handles = []
    if f107_handle:
        combined_handles.append(f107_handle)

    if cmapA and n_binsA > 0:
        for i in reversed(range(n_binsA)):
            color = cmapA(normA(i))
            b_id = bin_idsA[i]
            range_str = CommonConfig.TEMP_RANGES.get(b_id, f"Bin {b_id}")
            label = f"SII at {range_str} °C"
            combined_handles.append(Line2D([0], [0], color=color, lw=Config.LINE_WIDTH, label=label))

    if combined_handles:
        axA.legend(handles=combined_handles, loc="lower right",
                   fontsize=Config.CBAR_TICK_SIZE, framealpha=0.9)

    # Determine p-range for the figure (for consistent colorbar scaling)
    p_mins = []
    for df in data.values():
        if df is None or df.empty:
            continue
        pcol = pcol_to_use(df)
        if pcol in df.columns:
            vals = df[pcol].to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                p_mins.append(vals.min())

    vmin = min(p_mins) if p_mins else 1e-15

    # p-value colorbar (Panel A)
    cax_p = axA.inset_axes([0.1, 0.15, 0.3, 0.05])
    add_pvalue_colorbar(fig, cax_p, vmin)

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

    for letter in [keys[2]]:
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