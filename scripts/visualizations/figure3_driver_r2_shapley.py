#!/usr/bin/env python3
"""
Figure 3: absolute horizontal stacked bars for the 28-day Shapley/LMG
allocation of residual SIF variance explained by the joint SII + PAR + VPD
model.

Panel A shows the Frozen regime on its own x-axis because its explained
variance is much larger than the other regimes. Panel B shows the remaining
temperature regimes on an enlarged scale.

Bar length equals the full 28-day model R². Segments show the exact Shapley/LMG
allocation of that explained variance among SII, PAR, and VPD. Values inside
segments are shares of the explained R², not shares of total residual variance.

Input:
    results/supplementary_checks/driver_r2_shapley_by_window.csv

Outputs:
    reports/figures/figure3_driver_r2_shapley.png
    reports/figures/figure3_driver_r2_shapley.pdf
    reports/figures/figure3_driver_r2_shapley_source.csv

Intended location:
    scripts/visualizations/figure3_driver_r2_shapley.py
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "driver_r2_shapley_by_window.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figure3_driver_r2_shapley.png"
OUTPUT_PDF = OUTPUT_DIR / "figure3_driver_r2_shapley.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figure3_driver_r2_shapley_source.csv"


# =============================================================================
# ANALYSIS SELECTION
# =============================================================================

SCENARIO = "full_sample"
WINDOW_DAYS = 28

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
    "Warm_Stress": "Warm Stress",
    "Extreme_Heat": "Extreme Heat",
}

VALUE_COLUMNS = {
    "SII": "sii_shapley_percent",
    "PAR": "par_shapley_percent",
    "VPD": "vpd_shapley_percent",
}

BETA_COLUMNS = {
    "SII": "sii_beta_standardized",
    "PAR": "par_beta_standardized",
    "VPD": "vpd_beta_standardized",
}

R2_COLUMN = "r2_full_percent"

# Stack order chosen so the usually smaller SII component stays visible.
STACK_ORDER = ["PAR", "VPD", "SII"]
LEGEND_ORDER = ["SII", "PAR", "VPD"]

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DRIVER_COLORS = {
    "SII": DEFAULT_COLORS[0],
    "PAR": DEFAULT_COLORS[2 % len(DEFAULT_COLORS)],
    "VPD": DEFAULT_COLORS[1 % len(DEFAULT_COLORS)],
}

# Frozen is split onto its own x-axis because its explained variance is an
# order of magnitude larger than the other regimes.
BROKEN_AXIS_FROZEN = "Frozen"


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

FIGSIZE = (12.6, 5.6)
DPI = 400

BAR_HEIGHT = 0.72
BAR_ALPHA = 0.92

SHOW_SHARE_LABELS = True
SHOW_SIGNS = False           # can be turned on if desired
MIN_LABEL_WIDTH = 0.05       # percentage points
RIGHT_LABEL_PAD = 0.15

DECOMPOSITION_TOLERANCE = 1e-8


# =============================================================================
# HELPERS
# =============================================================================

def to_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def normalize_scenario(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "pooled": "full_sample",
        "full": "full_sample",
        "global": "full_sample",
    }
    return aliases.get(text, text)


def normalize_temperature(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Warm Stress": "Warm_Stress",
        "Extreme Heat": "Extreme_Heat",
    }
    return aliases.get(text, text)


def sign_symbol(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value > 0:
        return "+"
    if value < 0:
        return "−"
    return "0"


def load_source() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input table not found:\n{INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {
        "scenario",
        "temp_bin_label",
        "window_days",
        "eligible",
        "eligibility_reason",
        R2_COLUMN,
        *VALUE_COLUMNS.values(),
        *BETA_COLUMNS.values(),
        "n_obs",
        "n_cells",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    raw["_scenario"] = raw["scenario"].map(normalize_scenario)
    raw["_temperature"] = raw["temp_bin_label"].map(normalize_temperature)
    raw["_eligible"] = to_boolean(raw["eligible"])
    raw["_window"] = pd.to_numeric(raw["window_days"], errors="coerce")

    mask = (
        (raw["_scenario"] == normalize_scenario(SCENARIO))
        & (raw["_window"] == WINDOW_DAYS)
        & raw["_temperature"].isin(TEMP_ORDER)
    )

    selected = raw.loc[mask].copy()

    if selected.empty:
        raise ValueError(
            "No rows remained after filtering for:\n"
            f"  scenario={SCENARIO}\n"
            f"  window_days={WINDOW_DAYS}"
        )

    for column in [R2_COLUMN, *VALUE_COLUMNS.values(), *BETA_COLUMNS.values(), "n_obs", "n_cells"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected["temperature"] = pd.Categorical(
        selected["_temperature"],
        categories=TEMP_ORDER,
        ordered=True,
    )

    duplicated = selected.duplicated(["temperature"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate temperature rows remain after filtering.")

    selected = selected.sort_values("temperature").reset_index(drop=True)
    plot_df = selected.loc[selected["_eligible"]].copy()

    if len(plot_df) != len(TEMP_ORDER):
        warnings.warn(
            f"Expected {len(TEMP_ORDER)} eligible rows, found {len(plot_df)}."
        )

    # Validate exact decomposition on the plotting scale.
    summed = plot_df[list(VALUE_COLUMNS.values())].sum(axis=1)
    error = np.abs(summed - plot_df[R2_COLUMN])
    if (error > DECOMPOSITION_TOLERANCE * 100).any():
        bad = plot_df.loc[
            error > DECOMPOSITION_TOLERANCE * 100,
            ["temperature", *VALUE_COLUMNS.values(), R2_COLUMN],
        ]
        raise ValueError(
            "Shapley contributions do not sum to full-model R²:\n"
            + bad.to_string(index=False)
        )

    # Relative shares within explained R².
    for driver, column in VALUE_COLUMNS.items():
        plot_df[f"{driver.lower()}_share"] = np.where(
            plot_df[R2_COLUMN] > 0,
            100.0 * plot_df[column] / plot_df[R2_COLUMN],
            np.nan,
        )

    for driver, column in BETA_COLUMNS.items():
        plot_df[f"{driver.lower()}_sign"] = plot_df[column].map(sign_symbol)

    plot_df["temp_label"] = plot_df["temperature"].astype(str).map(TEMP_LABELS)

    return plot_df


def add_share_labels(ax: plt.Axes, row: pd.Series, y: float) -> None:
    left = 0.0
    for driver in STACK_ORDER:
        value = float(row[VALUE_COLUMNS[driver]])
        share = float(row[f"{driver.lower()}_share"])
        center = left + value / 2.0

        if np.isfinite(value) and value >= MIN_LABEL_WIDTH:
            ax.text(
                center,
                y,
                f"{driver} {share:.0f}%",
                ha="center",
                va="center",
                fontsize=8.0,
                color="black",
                zorder=5,
            )
        left += max(value, 0.0)


def _draw_bar_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    show_y_labels: bool = True,
    panel_letter: str | None = None,
) -> None:
    """Draw horizontal stacked bars for the provided regimes on the given axis."""
    y_positions = np.arange(len(df))
    left = np.zeros(len(df), dtype=float)

    for driver in STACK_ORDER:
        values = df[VALUE_COLUMNS[driver]].to_numpy(dtype=float)
        ax.barh(
            y_positions,
            values,
            left=left,
            height=BAR_HEIGHT,
            color=DRIVER_COLORS[driver],
            alpha=BAR_ALPHA,
            edgecolor="none",
            zorder=3,
            label=driver,
        )
        left = left + np.nan_to_num(values, nan=0.0)

    ax.set_yticks(y_positions)
    if show_y_labels:
        ytick_labels = [
            f"{row.temp_label}\n(R²={getattr(row, R2_COLUMN):.2f}%)"
            for row in df.itertuples(index=False)
        ]
        ax.set_yticklabels(ytick_labels)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()

    xmax = float(df[R2_COLUMN].max()) * 1.18
    ax.set_xlim(0, xmax)

    # Optional inside-segment share labels.
    if SHOW_SHARE_LABELS:
        for yi, (_, row) in zip(y_positions, df.iterrows()):
            add_share_labels(ax, row, yi)

    ax.grid(axis="x", linestyle="--", linewidth=0.75, alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if panel_letter is not None:
        ax.text(
            -0.12,
            0.98,
            panel_letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )


def draw_absolute_bars_split(ax_frozen: plt.Axes, ax_others: plt.Axes, df: pd.DataFrame) -> None:
    """Draw Frozen on its own axis and the remaining regimes on another."""
    frozen_df = df.loc[df["temperature"].astype(str) == BROKEN_AXIS_FROZEN].copy()
    others_df = df.loc[df["temperature"].astype(str) != BROKEN_AXIS_FROZEN].copy()

    if frozen_df.empty:
        raise ValueError("Frozen row not found in plotting data.")
    if others_df.empty:
        raise ValueError("No non-Frozen rows found in plotting data.")

    _draw_bar_panel(ax_frozen, frozen_df, show_y_labels=True, panel_letter="A")
    _draw_bar_panel(ax_others, others_df, show_y_labels=True, panel_letter="B")

    # Only the bottom axis keeps the x-axis label.
    ax_frozen.set_xlabel("")
    ax_others.set_xlabel(
        r"Residual SIF variance explained by the 28-day joint model "
        r"($R^2$, percentage points)"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    df = load_source()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_columns = [
        "scenario",
        "temp_bin_label",
        "temperature",
        "temp_label",
        "window_days",
        R2_COLUMN,
        *VALUE_COLUMNS.values(),
        "sii_share",
        "par_share",
        "vpd_share",
        *BETA_COLUMNS.values(),
        "sii_sign",
        "par_sign",
        "vpd_sign",
        "n_obs",
        "n_cells",
        "_eligible",
        "eligibility_reason",
    ]
    existing = [c for c in source_columns if c in df.columns]
    df[existing].to_csv(OUTPUT_SOURCE, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.4,
        }
    )

    # Split Frozen onto its own x-axis because its R² is much larger.
    fig, (ax_frozen, ax_others) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIGSIZE,
        sharex=False,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.18},
    )
    draw_absolute_bars_split(ax_frozen, ax_others, df)

    legend_handles = [
        Patch(
            facecolor=DRIVER_COLORS[driver],
            edgecolor="none",
            alpha=BAR_ALPHA,
            label=driver,
        )
        for driver in LEGEND_ORDER
    ]
    ax_others.legend(
        handles=legend_handles,
        labels=LEGEND_ORDER,
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    plt.close(fig)

    print(f"Input: {INPUT_CSV}")
    print(f"Scenario: {SCENARIO}")
    print(f"Window: {WINDOW_DAYS} days")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source data: {OUTPUT_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())