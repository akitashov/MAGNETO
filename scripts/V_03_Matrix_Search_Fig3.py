#!/usr/bin/env python3
"""
V_03_v2_Scenarios.py (MAGNETO Pipeline Visualization)

Figure 3 generator: robustness landscapes of SII effect across geographic scenarios.

Key updates:
- Layout: 2x2 grid representing 4 scenarios for a FIXED target (e.g., 771nm).
- Metrics: All panels show absolute t-statistics (|t_sii|) using 'viridis' colormap.
- Axis cleanup: Temperature axis with range labels ONLY on left panels (A, C). 
  Right panels (B, D) have no Y-axis labels to reduce clutter.
- Spacing: Increased wspace and labelpad to prevent colorbar/label overlap.
- Colorbars: Single Viridis colorbar spanning both rows on the far right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

from _Common import Config as CommonConfig


# ==============================================================================
# CONFIG
# ==============================================================================

class Config:
    PROJECT_ROOT = getattr(CommonConfig, "PROJECT_ROOT", Path("."))
    INPUT_DIR = getattr(CommonConfig, "RESULTS_DIR", PROJECT_ROOT / "results")
    OUTPUT_DIR = getattr(CommonConfig, "REPORTS_ROOT", PROJECT_ROOT / "reports") / "figures"

    # Fixed target for scenario comparison
    FIXED_TARGET = "sif_771nm"

    # Scenarios for the 2x2 grid
    COMPOSITE_SCENARIOS: List[Tuple[str, str]] = [
        ("Global", "Global_High_LAI"),  # Panel A
        ("North", "Control_North"),     # Panel B
        ("SAA", "SAA_High_LAI"),        # Panel C
        ("Sahara", "Sahara_Barren"),    # Panel D
    ]

    BIN_ORDER: List[str] = getattr(
        CommonConfig,
        "TEMP_LABELS_PHYSIO",
        ["Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"],
    )

    COMPOSITE_FIG_SIZE = (16, 12)
    FONT_SCALE = 1.25
    AXIS_FONT_SIZE = int(13 * FONT_SCALE)
    TICK_FONT_SIZE = int(10.5 * FONT_SCALE)
    CBAR_LABEL_SIZE = int(10.5 * FONT_SCALE)
    CBAR_TICK_SIZE = int(9.5 * FONT_SCALE)
    TITLE_FONT_SIZE = int(14 * FONT_SCALE)
    PANEL_LABEL_FONT_SIZE = int(16 * FONT_SCALE)

    CMAP_ABS = "viridis"
    AGG_FUNC = "median"
    MIN_MODELS_PER_CELL = 10
    XTICK_STEP = 2

    # Scale settings
    USE_GLOBAL_SCALES = True       
    ABS_NORM = "linear"            
    ABS_POWER_GAMMA = 0.65         
    COLORBAR_NBINS = 9             

    PANEL_LABELS = ["a", "b", "c", "d"] if CommonConfig.LOWER_FIG_LETTERS else ["A", "B", "C", "D"]
    PANEL_LABEL_POS = (-0.02, 1.05)


# ==============================================================================
# DATA HELPERS
# ==============================================================================

def _get_temperature_range_labels() -> List[str]:
    """Get formatted temperature range labels from CommonConfig.TEMP_RANGES."""
    temp_ranges = getattr(CommonConfig, "TEMP_RANGES_PHYSIO", {})
    
    # Create formatted labels for each bin in BIN_ORDER
    range_labels = []
    for bin_label in Config.BIN_ORDER:
        if bin_label in temp_ranges:
            low, high = temp_ranges[bin_label]
            # Format the range string
            if low == -np.inf:
                range_labels.append(f"<{high}°C")
            elif high == np.inf:
                range_labels.append(f"≥{low}°C")
            else:
                range_labels.append(f"{low}-{high}°C")
        else:
            range_labels.append(bin_label.replace("_", " "))
    
    return range_labels

def _compute_physio_centers() -> Dict[str, float]:
    """Compute numeric centers for temperature bins from CommonConfig."""
    bins = getattr(CommonConfig, "TEMP_BINS_PHYSIO", [-np.inf, 7, 15, 25, 30, np.inf])
    labels = getattr(CommonConfig, "TEMP_LABELS_PHYSIO", Config.BIN_ORDER)
    finite = [b for b in bins if np.isfinite(b)]
    if len(finite) < 2: return {str(l): float(i) for i, l in enumerate(labels)}
    l_w, r_w = finite[1] - finite[0], finite[-1] - finite[-2]
    eff = list(bins)
    if not np.isfinite(eff[0]): eff[0] = finite[0] - l_w
    if not np.isfinite(eff[-1]): eff[-1] = finite[-1] + r_w
    return {str(labels[i]): float((eff[i] + eff[i+1]) / 2.0) for i in range(len(labels))}

BIN_CENTER = _compute_physio_centers()

def load_matrix_search(target: str, scenario: str) -> pd.DataFrame:
    """Load results from matrix search CSV."""
    path = Path(Config.INPUT_DIR) / f"matrix_search_{target}_{scenario}.csv"
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    return df.dropna(subset=["bin_label", "sii_window", "t_sii"])

def aggregate_strength(df: pd.DataFrame, *, use_abs: bool = True) -> pd.DataFrame:
    """Aggregate t-statistics into matrix format."""
    if df.empty: return pd.DataFrame()
    df["metric"] = df["t_sii"].abs() if use_abs else df["t_sii"]
    grouped = df.groupby(["bin_label", "sii_window"]).agg(
        strength=("metric", Config.AGG_FUNC), count=("metric", "size")
    ).reset_index()
    grouped.loc[grouped["count"] < Config.MIN_MODELS_PER_CELL, "strength"] = np.nan
    return grouped.pivot(index="bin_label", columns="sii_window", values="strength").reindex(Config.BIN_ORDER).sort_index(axis=1)

# ==============================================================================
# SCALE COMPUTATION
# ==============================================================================

def compute_global_abs_scale() -> float:
    """Compute global max for the fixed target across all 4 scenarios."""
    vmax = 0.0
    for _, scen_name in Config.COMPOSITE_SCENARIOS:
        df = load_matrix_search(Config.FIXED_TARGET, scen_name)
        mat = aggregate_strength(df, use_abs=True)
        if not mat.empty: vmax = max(vmax, float(np.nanmax(mat.to_numpy())))
    return vmax if vmax > 0 else 1.0

# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_composite_scenarios(
    mats: Dict[str, pd.DataFrame],
    *,
    target: str,
    out_path: Path,
    vmax_abs: float,
) -> None:
    """Render 2x2 composite figure comparing 4 scenarios for one target."""
    fig = plt.figure(figsize=Config.COMPOSITE_FIG_SIZE)
    # gs: 2 plot cols and a colormap col
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.03], hspace=0.2, wspace=0.1)
    
    # require single image for normalization
    im = None
    panel_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    
    # Get temperature range labels once for all panels
    temp_labels = _get_temperature_range_labels()
    
    for idx, (short_name, scen_name) in enumerate(Config.COMPOSITE_SCENARIOS):
        row, col = panel_map[idx]
        ax = fig.add_subplot(gs[row, col])
        
        mat = mats.get(scen_name, pd.DataFrame())
        if mat.empty:
            ax.text(0.5, 0.5, f"No Data\n{short_name}", ha="center", va="center", transform=ax.transAxes)
            continue
        
        vmin, vmax = 0.0, vmax_abs
        cmap = Config.CMAP_ABS
        norm = colors.PowerNorm(gamma=Config.ABS_POWER_GAMMA, vmin=vmin, vmax=vmax) if Config.ABS_NORM == "power" else None
        
        current_im = ax.imshow(mat.to_numpy(), aspect="auto", origin="lower", 
                               cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
        
        # Store im 
        if im is None:
            im = current_im

        # --- Y-AXIS (Temperature) ---
        if col == 0:  # Left column (A, C) - show labels
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(temp_labels, fontsize=Config.TICK_FONT_SIZE)
            # ax.set_ylabel("Temperature Range", fontsize=Config.AXIS_FONT_SIZE, labelpad=15)
        else:  # Right column (B, D) - no labels
            ax.set_yticks([])
            ax.set_ylabel("")  # Remove ylabel

        # --- X-AXIS ---
        windows = list(mat.columns)
        ax.set_xticks(range(0, len(windows), Config.XTICK_STEP))
        ax.set_xticklabels([str(int(windows[i])) for i in range(0, len(windows), Config.XTICK_STEP)],
                           fontsize=Config.TICK_FONT_SIZE)
        
        if row == 1: ax.set_xlabel("SII integration window (days)", fontsize=Config.AXIS_FONT_SIZE)
        
        ax.set_title(f"{short_name}", fontsize=Config.TITLE_FONT_SIZE, pad=10)
        ax.text(Config.PANEL_LABEL_POS[0], Config.PANEL_LABEL_POS[1], Config.PANEL_LABELS[idx],
                transform=ax.transAxes, fontsize=Config.PANEL_LABEL_FONT_SIZE, fontweight="bold", va="top", ha="right")

    # --- Single colormap ---
    if im is not None:
        cax = fig.add_subplot(gs[:, 2])  
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Median |t$_{SII}$|", fontsize=Config.CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=Config.CBAR_TICK_SIZE)
    
    # fig.suptitle(f"Target: {target.replace('sif_', '').replace('nm', ' nm')}", 
    #              fontsize=Config.PANEL_LABEL_FONT_SIZE, y=0.98)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- EXPORT DATA ---
    csv_dir = out_path.parent / (out_path.stem + "_data")
    csv_dir.mkdir(exist_ok=True)
    for scen_name, mat in mats.items():
        if not mat.empty:
            mat.to_csv(csv_dir / f"matrix_{scen_name}.csv")
    print(f"[SUCCESS] Saved figure: {out_path.name}")
    print(f"[SUCCESS] Exported matrices to: {csv_dir}")

def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    # Pass 1: Compute shared scale across all scenarios
    vmax_abs = compute_global_abs_scale()
    
    # Pass 2: Collect data for all 4 panels
    mats = {}
    for _, full_name in Config.COMPOSITE_SCENARIOS:
        df = load_matrix_search(Config.FIXED_TARGET, full_name)
        mats[full_name] = aggregate_strength(df, use_abs=True)
        
    # Generate figure
    out_file = Config.OUTPUT_DIR / f"fig_3_scenarios_{Config.FIXED_TARGET}.pdf"
    plot_composite_scenarios(mats, target=Config.FIXED_TARGET, out_path=out_file, vmax_abs=vmax_abs)

if __name__ == "__main__":
    main()