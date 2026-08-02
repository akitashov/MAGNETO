#!/usr/bin/env python3
"""
Supplementary Figure S7. Extended Shapley/LMG decomposition across all
temperature bins and both main vegetation scenarios.

Rendering only: reads
results/supplementary_checks/driver_r2_shapley_by_window.csv
for window_days=28 and the scenarios 'full_sample' and
'persistently_vegetated'.

Bars show exact Shapley/LMG allocations of the full-model R² for a 28-day
window; total length equals residual SIF variance explained by SII, PAR, and
VPD. Frozen bars are clipped on the main axis; full-scale insets are shown
above each panel.
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Paths -----------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "driver_r2_shapley_by_window.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS7_shapley_extended.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS7_shapley_extended.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS7_shapley_extended_source.csv"

FIGURE_ID = "S7"


# Selection -------------------------------------------------------------------
SCENARIOS = ["full_sample", "persistently_vegetated"]
PANEL_LABELS = {
    "full_sample": "Full sample",
    "persistently_vegetated": "Persistently vegetated",
}
PANEL_LETTERS = ["A", "B"]
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
    "Warm_Stress": "Warm stress",
    "Extreme_Heat": "Extreme heat",
}


# Variables -------------------------------------------------------------------
STACK_ORDER = ["SII", "PAR", "VPD"]
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

_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DRIVER_COLORS = {
    "SII": _cycle[0],
    "PAR": _cycle[2 % len(_cycle)],
    "VPD": _cycle[1 % len(_cycle)],
}


# Display ---------------------------------------------------------------------
FIGSIZE = (9.6, 5.8)
DPI = 400
BAR_HEIGHT = 0.65
BAR_ALPHA = 0.94
DECOMPOSITION_TOLERANCE = 1e-10
INSET_HEADROOM = 1.12
MAIN_HEADROOM = 1.05


# Helpers ---------------------------------------------------------------------
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


def load_source() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input table not found:\n{INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {
        "scenario",
        "temp_bin_label",
        "window_days",
        "sii_shapley_r2",
        "par_shapley_r2",
        "vpd_shapley_r2",
        "sii_shapley_percent",
        "par_shapley_percent",
        "vpd_shapley_percent",
        "r2_full",
        "r2_full_percent",
        "sii_beta_standardized",
        "par_beta_standardized",
        "vpd_beta_standardized",
        "n_obs",
        "n_cells",
        "eligible",
        "eligibility_reason",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    raw["_scenario"] = raw["scenario"].map(normalize_scenario)
    raw["temperature"] = raw["temp_bin_label"].map(normalize_temperature)
    raw["eligible"] = to_boolean(raw["eligible"])
    raw["window_days"] = pd.to_numeric(raw["window_days"], errors="coerce")

    mask = (
        raw["_scenario"].isin(SCENARIOS)
        & raw["temperature"].isin(TEMP_ORDER)
        & (raw["window_days"] == WINDOW_DAYS)
    )

    selected = raw.loc[mask].copy()

    if selected.empty:
        raise ValueError(
            "No rows remained after filtering for:\n"
            f"  scenarios={SCENARIOS}\n"
            f"  window={WINDOW_DAYS}"
        )

    numeric_columns = [
        "sii_shapley_r2",
        "par_shapley_r2",
        "vpd_shapley_r2",
        *VALUE_COLUMNS.values(),
        "r2_full",
        R2_COLUMN,
        *BETA_COLUMNS.values(),
        "n_obs",
        "n_cells",
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    duplicated = selected.duplicated(["_scenario", "temperature"], keep=False)
    if duplicated.any():
        examples = (
            selected.loc[duplicated, ["scenario", "temperature"]]
            .drop_duplicates()
            .head(20)
        )
        raise ValueError(
            "Duplicate scenario × temperature rows found:\n"
            + examples.to_string(index=False)
        )

    return selected.sort_values(
        ["_scenario", "temperature"],
        key=lambda col: (
            col.map({s: i for i, s in enumerate(SCENARIOS)})
            if col.name == "_scenario"
            else col.map({t: i for i, t in enumerate(TEMP_ORDER)})
        ),
    )


def validate_decomposition(df: pd.DataFrame) -> float:
    """Return the maximum absolute decomposition error for eligible rows."""
    eligible = df.loc[df["eligible"]].copy()
    if eligible.empty:
        return np.nan

    summed = (
        eligible["sii_shapley_r2"]
        + eligible["par_shapley_r2"]
        + eligible["vpd_shapley_r2"]
    )
    error = np.abs(summed - eligible["r2_full"])
    return float(error.max())


def nice_upper_limit(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0

    steps = [0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0]
    for step in steps:
        rounded = np.ceil(value / step) * step
        if rounded <= value * 1.25:
            return float(rounded)

    magnitude = 10 ** np.floor(np.log10(value))
    return float(np.ceil(value / magnitude) * magnitude)


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    scenario: str,
    panel_letter: str,
    main_xlim: float,
    frozen_xlim: float,
) -> None:
    # Temperature bins run top-to-bottom in scientific order.
    y_positions = np.arange(len(TEMP_ORDER))
    left = np.zeros(len(TEMP_ORDER), dtype=float)

    for driver in STACK_ORDER:
        values = np.array(
            [df.loc[temp, VALUE_COLUMNS[driver]] for temp in TEMP_ORDER],
            dtype=float,
        )
        values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)

        ax.barh(
            y_positions,
            values,
            left=left,
            height=BAR_HEIGHT,
            color=DRIVER_COLORS[driver],
            alpha=BAR_ALPHA,
            edgecolor="none",
            zorder=3,
        )

        # Segment labels for wide enough segments.
        for i, (v, l) in enumerate(zip(values, left)):
            if v > main_xlim * 0.06:
                ax.text(
                    l + v / 2,
                    y_positions[i],
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white",
                    fontweight="bold",
                )

        left += values

    # Total labels at the end of each bar.
    for i, temp in enumerate(TEMP_ORDER):
        total = left[i]
        if np.isfinite(total) and total > 0:
            # Place label just right of the bar; clipped bars still get a value.
            x_text = min(total + main_xlim * 0.015, main_xlim * 0.98)
            ax.text(
                x_text,
                y_positions[i],
                f"{total:.2f}",
                ha="left",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color="black",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([TEMP_LABELS[t] for t in TEMP_ORDER])
    ax.set_xlim(0.0, main_xlim)
    ax.invert_yaxis()

    ax.set_title(PANEL_LABELS[scenario], fontsize=11.0, fontweight="bold", pad=6)
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

    ax.set_xlabel(
        r"Residual SIF variance explained ($R^2$, percentage points)",
        fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.75, alpha=0.28, zorder=0)
    ax.set_axisbelow(True)

    # Inset for Frozen, which may exceed the main axis limit.
    frozen_total = df.loc["Frozen", R2_COLUMN]
    if np.isfinite(frozen_total) and frozen_total > main_xlim:
        inset = ax.inset_axes([0.58, 0.78, 0.38, 0.18])
        inset.barh(
            [0],
            [frozen_total],
            height=BAR_HEIGHT,
            color=DRIVER_COLORS["SII"],
            alpha=BAR_ALPHA,
            edgecolor="none",
            zorder=3,
        )
        left_frozen = np.array([0.0])
        for driver in STACK_ORDER[1:]:
            v = float(df.loc["Frozen", VALUE_COLUMNS[driver]])
            if np.isfinite(v):
                inset.barh(
                    [0],
                    [v],
                    left=left_frozen,
                    height=BAR_HEIGHT,
                    color=DRIVER_COLORS[driver],
                    alpha=BAR_ALPHA,
                    edgecolor="none",
                    zorder=3,
                )
                left_frozen += v

        inset.set_xlim(0.0, frozen_xlim)
        inset.set_ylim(-0.5, 0.5)
        inset.set_yticks([0])
        inset.set_yticklabels(["Frozen"], fontsize=8)
        inset.set_xticks(
            [0, np.round(frozen_xlim / 2, 1), np.round(frozen_xlim, 1)]
        )
        inset.tick_params(axis="x", labelsize=7)
        inset.set_title("Frozen (full scale)", fontsize=8, pad=3)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.28)
        inset.set_axisbelow(True)

        # Annotation linking the clipped bar to the inset.
        ax.annotate(
            "clipped",
            xy=(main_xlim, 0),
            xytext=(main_xlim * 0.92, 0.45),
            fontsize=7.5,
            color="0.35",
            ha="right",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color="0.5",
                lw=0.8,
                connectionstyle="arc3,rad=0.2",
            ),
        )


def build_source_table(source: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario in SCENARIOS:
        panel_df = source.loc[source["_scenario"] == scenario].copy()
        for display_order, row in enumerate(
            panel_df.itertuples(index=False), start=1
        ):
            records.append(
                {
                    "figure_id": f"Figure {FIGURE_ID}",
                    "panel": PANEL_LABELS[scenario],
                    "display_order": display_order,
                    "source_file": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
                    "scenario": scenario,
                    "temp_bin_label": row.temp_bin_label,
                    "window_days": int(row.window_days)
                    if pd.notna(row.window_days)
                    else None,
                    "sii_shapley_percent": row.sii_shapley_percent,
                    "par_shapley_percent": row.par_shapley_percent,
                    "vpd_shapley_percent": row.vpd_shapley_percent,
                    "r2_full_percent": row.r2_full_percent,
                    "sii_beta_standardized": row.sii_beta_standardized,
                    "par_beta_standardized": row.par_beta_standardized,
                    "vpd_beta_standardized": row.vpd_beta_standardized,
                    "n_obs": row.n_obs,
                    "n_cells": row.n_cells,
                    "eligible": row.eligible,
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    source = load_source()

    max_error = validate_decomposition(source)
    print(f"Input file: {INPUT_CSV}")
    print(f"Selected rows: {len(source)}")
    print(f"Scenarios: {source['scenario'].unique().tolist()}")
    print(f"Method: Shapley/LMG decomposition (rendering only)")
    print(f"Window: {WINDOW_DAYS} days")
    print(f"Max decomposition error: {max_error:.2e}")

    if max_error > DECOMPOSITION_TOLERANCE:
        warnings.warn(
            f"Max decomposition error {max_error:.2e} exceeds tolerance "
            f"{DECOMPOSITION_TOLERANCE:.0e}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_table = build_source_table(source)
    source_table.to_csv(OUTPUT_SOURCE, index=False)
    print(f"Saved source data: {OUTPUT_SOURCE}")

    # Prepare per-panel data indexed by temperature.
    panel_data = {}
    for scenario in SCENARIOS:
        df = source.loc[source["_scenario"] == scenario].set_index("temperature")
        missing = [t for t in TEMP_ORDER if t not in df.index]
        if missing:
            raise ValueError(
                f"Missing temperature bins for {scenario}: {missing}"
            )
        panel_data[scenario] = df

    # Axis limits: main axis uses the largest non-Frozen R² across panels.
    nonfrozen_max = 0.0
    frozen_max = 0.0
    for df in panel_data.values():
        for temp in TEMP_ORDER:
            total = df.loc[temp, R2_COLUMN]
            if not np.isfinite(total):
                continue
            if temp == "Frozen":
                frozen_max = max(frozen_max, total)
            else:
                nonfrozen_max = max(nonfrozen_max, total)

    main_xlim = nice_upper_limit(nonfrozen_max * MAIN_HEADROOM)
    frozen_xlim = nice_upper_limit(frozen_max * INSET_HEADROOM)

    # Plotting ----------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0})

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    for col, scenario in enumerate(SCENARIOS):
        draw_panel(
            ax=axes[0, col],
            df=panel_data[scenario],
            scenario=scenario,
            panel_letter=PANEL_LETTERS[col],
            main_xlim=main_xlim,
            frozen_xlim=frozen_xlim,
        )

    legend_handles = [
        Patch(
            facecolor=DRIVER_COLORS[driver],
            edgecolor="none",
            alpha=BAR_ALPHA,
            label=driver,
        )
        for driver in STACK_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        labels=STACK_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        title="Driver",
        title_fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.18,
        top=0.90,
        wspace=0.22,
    )

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    plt.close(fig)

    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved PDF: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
