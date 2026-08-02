#!/usr/bin/env python3

"""
Supplementary Figure S8: land-cover composition and LAI.

Purpose
-------
Document sample composition differences among the dominant 2022 land-cover
classes. Panel A shows the temperature-regime composition of each land-cover
class (100 % stacked bars by observation count). Panel B shows a weighted
median plus inter-quartile-range error bar for the time-varying LAI distribution
within each class, computed from the pre-aggregated LAI-quartile composition
table.

Inputs
------
- results/supplementary_checks/landcover_temperature_composition.csv
- results/supplementary_checks/landcover_lai_composition.csv

Outputs
-------
- reports/figures/figureS8_landcover_lai_composition.png (400 dpi)
- reports/figures/figureS8_landcover_lai_composition_source.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_TEMP = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "landcover_temperature_composition.csv"
)
INPUT_LAI = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "landcover_lai_composition.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS8_landcover_lai_composition.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS8_landcover_lai_composition.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS8_landcover_lai_composition_source.csv"

FIGURE_ID = "figureS8_landcover_lai_composition"


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

LANDCOVER_ORDER = [
    "Barren",
    "Shrubland/Savanna",
    "Savanna",
    "Grassland",
    "Cropland",
    "Forest",
]

TEMP_ORDER = [
    "Frozen",
    "Cold",
    "Cool",
    "Optimum",
    "Warm_Stress",
    "Extreme_Heat",
]

TEMP_LABELS = {
    "Frozen": "Frozen",
    "Cold": "Cold",
    "Cool": "Cool",
    "Optimum": "Optimum",
    "Warm_Stress": "Warm stress",
    "Extreme_Heat": "Extreme heat",
}

# Colour map chosen for intuitive temperature progression.
TEMP_COLORS = {
    "Frozen": "#2166ac",
    "Cold": "#67a9cf",
    "Cool": "#b5d8e8",
    "Optimum": "#1a9850",
    "Warm_Stress": "#fdae61",
    "Extreme_Heat": "#d73027",
}

FIGSIZE = (10.5, 6.2)
BAR_HEIGHT = 0.62
POINT_SIZE = 5.5
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPSIZE = 3.5
ERRORBAR_CAPTHICK = 1.2


# =============================================================================
# HELPERS
# =============================================================================


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: float | list[float] = 0.5,
) -> np.ndarray | float:
    """
    Compute weighted quantiles of `values` using `weights`.

    Follows the weighted percentile definition used by NumPy's histogram
    logic: sort values, accumulate normalised weights, and linearly
    interpolate where the cumulative weight crosses the requested quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.size == 0 or weights.size == 0:
        raise ValueError("Cannot compute quantiles for empty input.")
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape.")

    quantiles = np.atleast_1d(quantiles)
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("Quantiles must lie in [0, 1].")

    # Remove zero-weight entries before sorting.
    positive = weights > 0
    if not np.any(positive):
        return np.full(quantiles.shape, np.nan)

    values = values[positive]
    weights = weights[positive]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0:
        return np.full(quantiles.shape, np.nan)

    cumulative /= total

    # Linear interpolation of the cumulative distribution.
    result = np.interp(quantiles, cumulative, values)
    return result if result.size > 1 else float(result[0])


def prepare_temperature_composition(path: Path) -> pd.DataFrame:
    """Load and normalise the land-cover × temperature composition table."""
    df = pd.read_csv(path)
    required = {"land_cover_class", "temp_bin_label", "n_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["land_cover_class"] = pd.Categorical(
        df["land_cover_class"],
        categories=LANDCOVER_ORDER,
        ordered=True,
    )
    df["temp_bin_label"] = pd.Categorical(
        df["temp_bin_label"],
        categories=TEMP_ORDER,
        ordered=True,
    )

    # Proportion of observations within each land-cover class.
    total_obs = df.groupby("land_cover_class", observed=True)["n_obs"].transform("sum")
    df["proportion_within_landcover"] = df["n_obs"] / total_obs

    return df.sort_values(["land_cover_class", "temp_bin_label"])


def prepare_lai_composition(path: Path) -> pd.DataFrame:
    """Load and normalise the land-cover × LAI-quartile composition table."""
    df = pd.read_csv(path)
    required = {
        "land_cover_class",
        "lai_quartile",
        "proportion_within_landcover",
        "median_lai",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["land_cover_class"] = pd.Categorical(
        df["land_cover_class"],
        categories=LANDCOVER_ORDER,
        ordered=True,
    )
    return df.sort_values(["land_cover_class", "lai_quartile"])


def compute_weighted_lai_summary(df_lai: pd.DataFrame) -> pd.DataFrame:
    """Weighted median LAI and 25th/75th percentiles per land-cover class."""
    summaries = []
    for lc in LANDCOVER_ORDER:
        sub = df_lai[df_lai["land_cover_class"] == lc]
        if sub.empty:
            summaries.append(
                {
                    "land_cover_class": lc,
                    "weighted_median_lai": np.nan,
                    "weighted_q25_lai": np.nan,
                    "weighted_q75_lai": np.nan,
                }
            )
            continue

        values = sub["median_lai"].to_numpy(dtype=float)
        weights = sub["proportion_within_landcover"].to_numpy(dtype=float)

        median, q25, q75 = weighted_quantile(values, weights, [0.5, 0.25, 0.75])
        summaries.append(
            {
                "land_cover_class": lc,
                "weighted_median_lai": float(median),
                "weighted_q25_lai": float(q25),
                "weighted_q75_lai": float(q75),
            }
        )

    out = pd.DataFrame(summaries)
    out["land_cover_class"] = pd.Categorical(
        out["land_cover_class"],
        categories=LANDCOVER_ORDER,
        ordered=True,
    )
    return out.sort_values("land_cover_class")


def build_source_table(
    df_temp: pd.DataFrame,
    df_lai_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the two-section source CSV required by the figure spec."""
    temp_source = df_temp[
        ["land_cover_class", "temp_bin_label", "n_obs", "proportion_within_landcover"]
    ].copy()
    temp_source["figure_id"] = FIGURE_ID
    temp_source["panel"] = "A"
    temp_source["display_order"] = temp_source["land_cover_class"].cat.codes
    temp_source["source_file"] = INPUT_TEMP.name

    lai_source = df_lai_summary[
        [
            "land_cover_class",
            "weighted_median_lai",
            "weighted_q25_lai",
            "weighted_q75_lai",
        ]
    ].copy()
    lai_source["figure_id"] = FIGURE_ID
    lai_source["panel"] = "B"
    lai_source["display_order"] = lai_source["land_cover_class"].cat.codes
    lai_source["source_file"] = INPUT_LAI.name

    composition_cols = [
        "figure_id",
        "panel",
        "display_order",
        "source_file",
        "land_cover_class",
        "temp_bin_label",
        "n_obs",
        "proportion_within_landcover",
    ]
    lai_cols = [
        "figure_id",
        "panel",
        "display_order",
        "source_file",
        "land_cover_class",
        "weighted_median_lai",
        "weighted_q25_lai",
        "weighted_q75_lai",
    ]

    temp_section = temp_source[composition_cols]
    lai_section = lai_source[lai_cols]
    combined = pd.concat([temp_section, lai_section], ignore_index=True)
    return combined


# =============================================================================
# PLOTTING
# =============================================================================


def draw_panel_a(ax: plt.Axes, df_temp: pd.DataFrame) -> None:
    """100 % stacked horizontal bars by temperature regime."""
    n_classes = len(LANDCOVER_ORDER)
    y_positions = np.arange(n_classes)

    # Pivot to (land-cover class) × (temperature bin).
    pivot = df_temp.pivot(
        index="land_cover_class",
        columns="temp_bin_label",
        values="proportion_within_landcover",
    ).reindex(LANDCOVER_ORDER)[TEMP_ORDER]

    left = np.zeros(n_classes)
    for temp_bin in TEMP_ORDER:
        values = pivot[temp_bin].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        ax.barh(
            y_positions,
            values * 100,
            left=left * 100,
            height=BAR_HEIGHT,
            color=TEMP_COLORS[temp_bin],
            label=TEMP_LABELS[temp_bin],
            edgecolor="white",
            linewidth=0.5,
        )
        left += values

    ax.set_yticks(y_positions)
    ax.set_yticklabels(LANDCOVER_ORDER)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion of observations (%)")
    ax.set_title(
        "Composition by temperature regime (2022 land cover)",
        fontsize=10.5,
        loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        fontsize=8.5,
        title="Temperature regime",
        title_fontsize=9,
        columnspacing=0.9,
    )


def draw_panel_b(ax: plt.Axes, df_lai_summary: pd.DataFrame) -> None:
    """Weighted median LAI with horizontal IQR error bars."""
    n_classes = len(LANDCOVER_ORDER)
    y_positions = np.arange(n_classes)

    summary = df_lai_summary.set_index("land_cover_class").reindex(LANDCOVER_ORDER)
    medians = summary["weighted_median_lai"].to_numpy(dtype=float)
    q25 = summary["weighted_q25_lai"].to_numpy(dtype=float)
    q75 = summary["weighted_q75_lai"].to_numpy(dtype=float)

    finite = np.isfinite(medians) & np.isfinite(q25) & np.isfinite(q75)
    xerr_low = np.where(finite, medians - q25, np.nan)
    xerr_high = np.where(finite, q75 - medians, np.nan)

    ax.errorbar(
        medians,
        y_positions,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#2c3e50",
        ecolor="#2c3e50",
        markersize=POINT_SIZE,
        linewidth=ERRORBAR_LINEWIDTH,
        capsize=ERRORBAR_CAPSIZE,
        capthick=ERRORBAR_CAPTHICK,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(LANDCOVER_ORDER)
    ax.set_xlabel("LAI (m² m⁻²)")
    ax.set_title(
        "LAI distribution by land-cover class (time-varying)",
        fontsize=10.5,
        loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Light vertical grid for readability.
    ax.grid(True, axis="x", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)

    # Expand x-axis slightly so the rightmost error cap is not clipped.
    if finite.any():
        x_min = float(np.nanmin(q25[finite]))
        x_max = float(np.nanmax(q75[finite]))
        pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
        ax.set_xlim(left=max(0, x_min - pad), right=x_max + pad)


def add_panel_letters(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Add bold A/B panel labels in the upper-left of each axis."""
    for label, ax in zip("AB", axes):
        ax.text(
            -0.18,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="right",
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    print(f"Input file (temperature): {INPUT_TEMP}")
    print(f"Input file (LAI): {INPUT_LAI}")

    df_temp = prepare_temperature_composition(INPUT_TEMP)
    df_lai = prepare_lai_composition(INPUT_LAI)
    df_lai_summary = compute_weighted_lai_summary(df_lai)

    print(f"Selected temperature rows: {len(df_temp)}")
    print(f"Selected LAI rows: {len(df_lai)}")
    print("Land-cover classes present:")
    for lc in LANDCOVER_ORDER:
        n_temp = (df_temp["land_cover_class"] == lc).sum()
        n_lai = (df_lai["land_cover_class"] == lc).sum()
        print(f"  - {lc}: {n_temp} temperature bins, {n_lai} LAI quartiles")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_table = build_source_table(df_temp, df_lai_summary)
    source_table.to_csv(OUTPUT_SOURCE, index=False)
    print(f"Saved source data: {OUTPUT_SOURCE}")

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

    fig, axes = plt.subplots(
        ncols=2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42},
    )

    draw_panel_a(axes[0], df_temp)
    draw_panel_b(axes[1], df_lai_summary)
    add_panel_letters(fig, axes)

    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure PNG: {OUTPUT_PNG}")
    print(f"Saved figure PDF: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
