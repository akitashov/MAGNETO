#!/usr/bin/env python3
"""
Supplementary Figure S4: complete temperature and control profiles.

Extends Main Figure 2 by showing all temperature-stratified scenarios and both
detrending methods for the 28-day SII window.

Inputs
------
- results/supplementary_checks/temperature_scenarios_harmonic.csv
- results/supplementary_checks/temperature_scenarios_spline.csv

Outputs
-------
- reports/figures/figureS4_temperature_control_profiles.png  (400 dpi)
- reports/figures/figureS4_temperature_control_profiles.pdf
- reports/figures/figureS4_temperature_control_profiles_source.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_FILES = {
    "Harmonic": PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "temperature_scenarios_harmonic.csv",
    "Cyclic spline": PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "temperature_scenarios_spline.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS4_temperature_control_profiles.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS4_temperature_control_profiles.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS4_temperature_control_profiles_source.csv"


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

WINDOW_DAYS = 28

TEMP_ORDER = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]

TEMP_LABELS = {
    "Frozen": "Frozen",
    "Cold": "Cold",
    "Cool": "Cool",
    "Optimum": "Optimum",
    "Warm_Stress": "Warm Stress",
    "Extreme_Heat": "Extreme Heat",
}

# Top-row panels: Full sample + Persistently vegetated
TOP_SCENARIOS = ["pooled", "control_vegetated"]

# Bottom-row panels: Barren + Sahara + SAA
BOTTOM_SCENARIOS = ["landcover_barren", "Sahara", "SAA"]

SCENARIO_LABELS = {
    "pooled": "Full sample",
    "control_vegetated": "Persistently vegetated",
    "landcover_barren": "Barren land-cover class",
    "Sahara": "Sahara",
    "SAA": "SAA",
}

SCENARIO_COLORS = {
    "pooled": "#4055a8",          # blue
    "control_vegetated": "#2f7d4a",  # green
    "landcover_barren": "#b33e3e",   # red
    "Sahara": "#d19a24",             # gold
    "SAA": "#7b3fa0",                # purple
}

SCENARIO_LINESTYLES = {
    "pooled": "-",
    "control_vegetated": "--",
    "landcover_barren": "--",
    "Sahara": "-",
    "SAA": "-.",
}

PANEL_CONFIG = {
    "A": {"method": "Harmonic", "scenarios": TOP_SCENARIOS, "row": 0, "col": 0},
    "B": {"method": "Cyclic spline", "scenarios": TOP_SCENARIOS, "row": 0, "col": 1},
    "C": {"method": "Harmonic", "scenarios": BOTTOM_SCENARIOS, "row": 1, "col": 0},
    "D": {"method": "Cyclic spline", "scenarios": BOTTOM_SCENARIOS, "row": 1, "col": 1},
}

FIGSIZE = (10.5, 8.0)
LINE_WIDTH = 2.2
MARKER_SIZE = 5.5
CI_ALPHA = 0.22

Y_LABEL = r"Spearman $\rho$"
X_LABEL = "Temperature regime"


# =============================================================================
# HELPERS
# =============================================================================

def normalize_temperature(value: object) -> str:
    """Map common temperature labels to the canonical internal form."""
    text = str(value).strip()
    replacements = {
        "Warm Stress": "Warm_Stress",
        "Extreme Heat": "Extreme_Heat",
    }
    return replacements.get(text, text)


def load_and_standardize(path: Path, method_label: str) -> pd.DataFrame:
    """Load a temperature-scenario CSV and standardise columns."""
    if not path.exists():
        raise FileNotFoundError(f"Input table not found:\n{path}")

    raw = pd.read_csv(path)

    required = {
        "window_days",
        "scenario",
        "temp_class",
        "rho",
        "ci_low",
        "ci_high",
        "p_year",
        "p_circular",
        "p_block",
        "n_obs",
        "n_cells",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_file": path.name,
            "method": method_label,
            "window_days": pd.to_numeric(raw["window_days"], errors="coerce"),
            "scenario": raw["scenario"].astype(str).str.strip(),
            "temp_class": raw["temp_class"].map(normalize_temperature),
            "rho": pd.to_numeric(raw["rho"], errors="coerce"),
            "ci_low": pd.to_numeric(raw["ci_low"], errors="coerce"),
            "ci_high": pd.to_numeric(raw["ci_high"], errors="coerce"),
            "p_year": pd.to_numeric(raw["p_year"], errors="coerce"),
            "p_circular": pd.to_numeric(raw["p_circular"], errors="coerce"),
            "p_block": pd.to_numeric(raw["p_block"], errors="coerce"),
            "n_obs": pd.to_numeric(raw["n_obs"], errors="coerce"),
            "n_cells": pd.to_numeric(raw["n_cells"], errors="coerce"),
        }
    )

    out["p_display"] = out[["p_year", "p_circular", "p_block"]].max(axis=1, skipna=False)
    out["temp_class"] = pd.Categorical(
        out["temp_class"], categories=TEMP_ORDER, ordered=True
    )
    return out


def panel_for_row(method: str, scenario: str) -> str:
    """Return the panel letter for a given method/scenario combination."""
    for panel, cfg in PANEL_CONFIG.items():
        if cfg["method"] == method and scenario in cfg["scenarios"]:
            return panel
    return ""


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    panel: str,
    add_legend: bool,
) -> None:
    """Plot one 2×2 panel."""
    cfg = PANEL_CONFIG[panel]
    scenarios = cfg["scenarios"]
    x = np.arange(len(TEMP_ORDER), dtype=float)

    for display_order, scenario in enumerate(scenarios):
        d = (
            df[(df["scenario"] == scenario)]
            .set_index("temp_class")
            .reindex(TEMP_ORDER)
            .reset_index()
        )

        y = d["rho"].to_numpy(dtype=float)
        ci_low = d["ci_low"].to_numpy(dtype=float)
        ci_high = d["ci_high"].to_numpy(dtype=float)
        color = SCENARIO_COLORS[scenario]
        linestyle = SCENARIO_LINESTYLES[scenario]

        # Draw CI ribbon as a shaded band, one adjacent-pair segment at a time
        # so missing bins do not get connected.
        for i in range(len(x) - 1):
            segment_values = [
                y[i],
                y[i + 1],
                ci_low[i],
                ci_low[i + 1],
                ci_high[i],
                ci_high[i + 1],
            ]
            if not np.all(np.isfinite(segment_values)):
                continue
            ax.fill_between(
                x[i : i + 2],
                ci_low[i : i + 2],
                ci_high[i : i + 2],
                color=color,
                alpha=CI_ALPHA,
                linewidth=0,
                zorder=1,
            )

        # Draw profile line segments between adjacent bins only when both exist.
        for i in range(len(x) - 1):
            if not np.all(np.isfinite(y[i : i + 2])):
                continue
            ax.plot(
                x[i : i + 2],
                y[i : i + 2],
                color=color,
                linestyle=linestyle,
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
                zorder=3,
            )

        # Markers at available bins.
        for xi, yi in zip(x, y):
            if not np.isfinite(yi):
                continue
            ax.plot(
                xi,
                yi,
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                zorder=4,
            )

    ax.axhline(
        0,
        color="black",
        linewidth=0.9,
        alpha=0.75,
        zorder=0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([TEMP_LABELS[t] for t in TEMP_ORDER], rotation=30, ha="right")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if add_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=SCENARIO_COLORS[s],
                linestyle=SCENARIO_LINESTYLES[s],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=SCENARIO_COLORS[s],
                label=SCENARIO_LABELS[s],
            )
            for s in scenarios
        ]
        ax.legend(
            handles=handles,
            loc="best",
            frameon=True,
            framealpha=0.92,
            fontsize=9,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Input files:")
    for label, path in INPUT_FILES.items():
        print(f"  [{label}] {path}")

    frames = []
    for method_label, path in INPUT_FILES.items():
        frames.append(load_and_standardize(path, method_label))
    df = pd.concat(frames, ignore_index=True)

    df = df[df["window_days"] == WINDOW_DAYS].copy()

    all_scenarios = TOP_SCENARIOS + BOTTOM_SCENARIOS
    df = df[df["scenario"].isin(all_scenarios)].copy()

    df["panel"] = df.apply(
        lambda row: panel_for_row(row["method"], row["scenario"]), axis=1
    )
    df = df[df["panel"] != ""].copy()

    if df.empty:
        raise ValueError(
            f"No eligible rows remained for window={WINDOW_DAYS} days."
        )

    duplicated = df.duplicated(["method", "scenario", "temp_class"], keep=False)
    if duplicated.any():
        examples = df.loc[
            duplicated, ["method", "scenario", "temp_class"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicate method × scenario × temperature rows:\n"
            + examples.to_string(index=False)
        )

    print(f"\nSelected rows after filtering: {len(df)}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Window: {WINDOW_DAYS} days")

    # Build source CSV in a deterministic order.
    df["sample_type"] = df["scenario"]
    df["sample_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["display_order"] = df["scenario"].apply(
        lambda s: (TOP_SCENARIOS + BOTTOM_SCENARIOS).index(s)
    )
    df["temp_bin_label"] = df["temp_class"].astype(str)

    source_out = (
        df.sort_values(["panel", "display_order", "temp_class"])
        .reset_index(drop=True)
        .assign(figure_id="figureS4")[
            [
                "figure_id",
                "panel",
                "display_order",
                "source_file",
                "sample_type",
                "sample_label",
                "method",
                "temp_bin_label",
                "rho",
                "ci_low",
                "ci_high",
                "n_obs",
                "n_cells",
                "p_display",
            ]
        ]
        .copy()
    )

    source_out.to_csv(OUTPUT_SOURCE, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.edgecolor": "#333333",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)

    all_finite_values: list[np.ndarray] = []
    for panel in ["A", "B", "C", "D"]:
        cfg = PANEL_CONFIG[panel]
        ax = axes[cfg["row"], cfg["col"]]
        panel_df = df[df["panel"] == panel].copy()
        add_legend = cfg["col"] == 1  # legend on right-hand panels
        plot_panel(ax, panel_df, panel, add_legend=add_legend)

        # Panel letter to the left of the y-axis.
        ax.text(
            -0.18,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

        panel_vals = df[
            (df["panel"] == panel)
            & df[["rho", "ci_low", "ci_high"]].notna().any(axis=1)
        ][["rho", "ci_low", "ci_high"]].to_numpy(dtype=float)
        finite = panel_vals[np.isfinite(panel_vals)]
        if finite.size:
            all_finite_values.append(finite)

    if all_finite_values:
        all_values = np.concatenate(all_finite_values)
        bound = max(float(np.max(np.abs(all_values))) * 1.05, 0.03)
        for ax in axes.flat:
            ax.set_ylim(-bound, bound)

    # Column headers (methods).
    fig.text(0.27, 0.98, "Harmonic", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.77, 0.98, "Cyclic spline", ha="center", va="top", fontsize=12, fontweight="bold")

    # Row labels (sample groups).
    fig.text(0.02, 0.72, "Biological samples", ha="center", va="center", rotation="vertical", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.28, "Geographic controls", ha="center", va="center", rotation="vertical", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0.05, 0, 0.95, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source data: {OUTPUT_SOURCE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
