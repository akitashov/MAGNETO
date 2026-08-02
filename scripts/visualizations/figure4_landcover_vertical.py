#!/usr/bin/env python3
"""
Figure 4. Land-cover-specific SII–SIF associations.

MAIN DISPLAY
------------
- Pooled land-cover classes only; landcover_temp_* rows are excluded.
- Harmonic detrending, 28-day SII window.
- Barren is shown first and Forest last.
- x-axis: land-cover class.
- y-axis: Spearman rho.
- Vertical stems connect each estimate to zero and ARE NOT confidence intervals.
- Opacity reflects p-values.
- For stable visual behavior, opacity is mapped using fixed bounds:
      p = 1e-40  -> darkest
      p = 1e-1   -> palest
- No separate p-scale inset is shown.
- Instead, each class is annotated with both n and p near the bottom.

MANUSCRIPT CAPTION TEXT
-----------------------
"Points show the 28-day Spearman association between SII and harmonic residual
SIF across land-cover classes. Vertical stems connect each estimate to zero
and are not confidence intervals. Opacity reflects the conservative temporal-
surrogate p-value. The sample size and p-value for each class are annotated
below the x-axis."

No explanatory footer is drawn inside the figure.

Inputs
------
results/supplementary_checks/landcover_analysis.csv
results/supplementary_checks/landcover_analysis_surrogates.csv

Outputs
-------
reports/figures/figure4_landcover_vertical.png
reports/figures/figure4_landcover_vertical.pdf
reports/figures/figure4_landcover_vertical_source.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MAIN_INPUT_CANDIDATES = [
    PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis.csv",
    PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis_results.csv",
    PROJECT_ROOT / "results" / "landcover_analysis.csv",
]

SURROGATE_INPUT_CANDIDATES = [
    PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis_surrogates.csv",
    PROJECT_ROOT / "results" / "landcover_analysis_surrogates.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figure4_landcover_vertical.png"
OUTPUT_PDF = OUTPUT_DIR / "figure4_landcover_vertical.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figure4_landcover_vertical_source.csv"


# =============================================================================
# SELECTION
# =============================================================================

METHOD = "harmonic"
WINDOW_DAYS = 28

CLASS_ORDER = [
    "Barren",
    "Shrubland/Savanna",
    "Savanna",
    "Grassland",
    "Cropland",
    "Forest",
]

CLASS_LABELS = {
    "Barren": "Barren",
    "Shrubland/Savanna": "Shrubland/\nSavanna",
    "Savanna": "Savanna",
    "Grassland": "Grassland",
    "Cropland": "Cropland",
    "Forest": "Forest",
}

CLASS_COLORS = {
    "Barren": "#9e9e9e",
    "Shrubland/Savanna": "#8c6d31",
    "Savanna": "#d8a400",
    "Grassland": "#66a61e",
    "Cropland": "#c28e0e",
    "Forest": "#1b7837",
}

X_POSITIONS = {
    "Barren": 0.0,
    "Shrubland/Savanna": 1.1,
    "Savanna": 2.1,
    "Grassland": 3.1,
    "Cropland": 4.1,
    "Forest": 5.1,
}


# =============================================================================
# DISPLAY
# =============================================================================

FIGSIZE = (8.0, 6.7)
AXES_BOX_ASPECT = 0.78
DPI = 400

POINT_SIZE = 120
POINT_EDGE_WIDTH = 0.9
STEM_LINE_WIDTH = 4.0
CAP_WIDTH = 0.16

ALPHA_MIN = 0.20
ALPHA_MAX = 0.98

# Fixed p-value bounds for opacity mapping.
P_ALPHA_DARKEST = 0.03
P_ALPHA_PALEST = 0.2

SHOW_RHO_LABELS = True
SHOW_N_AND_P = True
SHOW_EXACT_P_LABELS_ABOVE = False

ZERO_LINE_COLOR = "#666666"


# =============================================================================
# NORMALIZATION
# =============================================================================

def find_first_existing(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{description} not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_method(value: object) -> str:
    text = normalize_text(value)
    if "harmonic" in text:
        return "harmonic"
    if "spline" in text:
        return "cyclic_spline"
    return text


def class_from_sample_type(value: object) -> str | None:
    text = normalize_text(value)

    if text.startswith("landcover_temp_"):
        return None
    if not text.startswith("landcover_"):
        return None

    suffix = text.removeprefix("landcover_").replace("_", " ").strip()

    mapping = {
        "forest": "Forest",
        "cropland": "Cropland",
        "grassland": "Grassland",
        "savanna": "Savanna",
        "shrubland/savanna": "Shrubland/Savanna",
        "shrubland savanna": "Shrubland/Savanna",
        "shrubland": "Shrubland/Savanna",
        "barren": "Barren",
    }
    return mapping.get(suffix)


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def matches_window(series: pd.Series, window_days: int) -> pd.Series:
    """
    Match both numeric and canonical string window encodings.

    Examples accepted for a 28-day window:
        28
        "28"
        "28d"
        "sii_28d"
        "sii-28d"
    """
    numeric = pd.to_numeric(series, errors="coerce").eq(window_days)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
    text_match = (
        text.eq(str(window_days))
        | text.eq(f"{window_days}d")
        | text.eq(f"sii_{window_days}d")
        | text.eq(f"sii-{window_days}d")
    )

    return numeric | text_match


# =============================================================================
# MAIN EFFECT TABLE
# =============================================================================

def load_main_results() -> tuple[pd.DataFrame, Path]:
    path = find_first_existing(
        MAIN_INPUT_CANDIDATES,
        "Land-cover analysis table",
    )
    raw = pd.read_csv(path, low_memory=False)

    required = {
        "sample_type",
        "method",
        "sii_window_days",
        "spearman_rho",
        "spearman_p",
        "n_cells",
        "n_obs",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "Missing required columns in land-cover analysis:\n"
            + "\n".join(f"  - {column}" for column in missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(f"  - {column}" for column in raw.columns)
        )

    df = raw.copy()
    df["landcover"] = df["sample_type"].map(class_from_sample_type)
    df["_method"] = df["method"].map(normalize_method)
    df["_window"] = pd.to_numeric(df["sii_window_days"], errors="coerce")

    for column in ["spearman_rho", "spearman_p", "n_cells", "n_obs"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    selected = df.loc[
        df["landcover"].isin(CLASS_ORDER)
        & (df["_method"] == normalize_method(METHOD))
        & (df["_window"] == WINDOW_DAYS)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No pooled land-cover rows remained after filtering for "
            f"method={METHOD}, window={WINDOW_DAYS}."
        )

    duplicated = selected.duplicated(["landcover"], keep=False)
    if duplicated.any():
        detail = [
            column
            for column in [
                "sample_type",
                "method",
                "sii_window_days",
                "temp_bin_label",
                "outcome",
                "land_cover_class",
            ]
            if column in selected.columns
        ]
        raise ValueError(
            "More than one pooled row remains for a land-cover class:\n"
            + selected.loc[duplicated, detail].to_string(index=False)
        )

    selected["order"] = selected["landcover"].map(
        {name: index for index, name in enumerate(CLASS_ORDER)}
    )
    selected = selected.sort_values("order").reset_index(drop=True)

    return selected, path


# =============================================================================
# SURROGATE P-VALUE TABLE
# =============================================================================

def load_surrogate_p_values() -> tuple[pd.DataFrame | None, Path | None, str]:
    path = next(
        (candidate for candidate in SURROGATE_INPUT_CANDIDATES if candidate.exists()),
        None,
    )
    if path is None:
        return None, None, "nominal Spearman p (surrogate table absent)"

    raw = pd.read_csv(path, low_memory=False)

    sample_column = choose_column(
        raw,
        ["sample_type", "scenario", "group", "sample", "sample_group"],
    )
    if sample_column is None:
        warnings.warn(
            "Surrogate table found but sample/group column could not be resolved. "
            "Falling back to nominal Spearman p."
        )
        return None, path, "nominal Spearman p (surrogate merge failed)"

    work = raw.copy()
    work["landcover"] = work[sample_column].map(class_from_sample_type)

    method_column = choose_column(raw, ["method", "detrending_method"])
    if method_column is not None:
        work = work.loc[
            work[method_column].map(normalize_method) == normalize_method(METHOD)
        ]

    window_column = choose_column(
        raw,
        ["sii_window_days", "window_days", "sii_window", "window"],
    )
    if window_column is not None:
        work = work.loc[
            matches_window(work[window_column], WINDOW_DAYS)
        ].copy()

    work = work.loc[work["landcover"].isin(CLASS_ORDER)].copy()
    if work.empty:
        warnings.warn(
            "Surrogate table contains no matching pooled land-cover rows. "
            "Falling back to nominal Spearman p."
        )
        return None, path, "nominal Spearman p (no matching surrogate rows)"

    long_p_column = choose_column(
        work,
        ["p_display", "surrogate_p", "empirical_p", "p_empirical", "p_value", "p"],
    )

    if long_p_column is not None:
        work["_p"] = pd.to_numeric(work[long_p_column], errors="coerce")
        summary = (
            work.groupby("landcover", as_index=False, observed=True)["_p"]
            .max()
            .rename(columns={"_p": "p_display"})
        )
        if summary["p_display"].notna().any():
            return summary, path, "conservative temporal-surrogate p"

    exact_wide_candidates = [
        "p_year_perm",
        "p_year_permutation",
        "p_circ_shift",
        "p_circular_shift",
        "p_block_perm",
        "p_block_permutation",
        "year_permutation_p",
        "circular_shift_p",
        "block_permutation_p",
    ]
    wide_columns = [column for column in exact_wide_candidates if column in work.columns]

    if not wide_columns:
        wide_columns = [
            column
            for column in work.columns
            if (
                column.lower().startswith("p_")
                or column.lower().endswith("_p")
            )
            and any(
                token in column.lower()
                for token in ["year", "circ", "circular", "block", "surrogate"]
            )
        ]

    if wide_columns:
        numeric = work[wide_columns].apply(pd.to_numeric, errors="coerce")
        work["_p_conservative"] = numeric.max(axis=1, skipna=True)
        summary = (
            work.groupby("landcover", as_index=False, observed=True)["_p_conservative"]
            .max()
            .rename(columns={"_p_conservative": "p_display"})
        )
        if summary["p_display"].notna().any():
            return summary, path, "conservative temporal-surrogate p"

    warnings.warn(
        "Surrogate table was found, but no empirical p-value columns could be "
        "resolved. Falling back to nominal Spearman p.\n"
        f"Available columns: {list(raw.columns)}"
    )
    return None, path, "nominal Spearman p (surrogate p unresolved)"


# =============================================================================
# P-VALUE FORMATTING AND OPACITY
# =============================================================================

def p_to_alpha(p_value: float) -> float:
    """Fixed log-scale mapping from p to alpha."""
    if not np.isfinite(p_value):
        return 0.60

    p = float(p_value)
    if p <= 0:
        p = P_ALPHA_DARKEST

    p = float(np.clip(p, P_ALPHA_DARKEST, P_ALPHA_PALEST))
    score = -math.log10(p)
    score_min = -math.log10(P_ALPHA_PALEST)
    score_max = -math.log10(P_ALPHA_DARKEST)
    scaled = (score - score_min) / (score_max - score_min)

    return float(
        np.clip(
            ALPHA_MIN + scaled * (ALPHA_MAX - ALPHA_MIN),
            ALPHA_MIN,
            ALPHA_MAX,
        )
    )


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value <= 0:
        return "<0.001"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    if p_value < 0.01:
        return f"{p_value:.3f}"
    if p_value < 0.1:
        return f"{p_value:.3f}"
    return f"{p_value:.2f}"


# =============================================================================
# MERGE
# =============================================================================

def load_plot_data() -> tuple[pd.DataFrame, Path, Path | None, str]:
    main, main_path = load_main_results()
    surrogate, surrogate_path, p_source = load_surrogate_p_values()

    if surrogate is not None:
        merged = main.merge(
            surrogate,
            on="landcover",
            how="left",
            validate="one_to_one",
        )
        missing = merged["p_display"].isna()
        if missing.any():
            warnings.warn(
                "Missing surrogate p-values for some classes; nominal Spearman "
                "p-values were used for those rows:\n"
                + ", ".join(merged.loc[missing, "landcover"].astype(str))
            )
            merged.loc[missing, "p_display"] = merged.loc[missing, "spearman_p"]
    else:
        merged = main.copy()
        merged["p_display"] = merged["spearman_p"]

    merged["alpha"] = merged["p_display"].map(p_to_alpha)
    merged["p_label"] = merged["p_display"].map(format_p_value)
    merged["x"] = merged["landcover"].map(X_POSITIONS)
    merged["color"] = merged["landcover"].map(CLASS_COLORS)

    return merged, main_path, surrogate_path, p_source


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    df, main_path, surrogate_path, p_source = load_plot_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_columns = [
        "sample_type",
        "method",
        "sii_window_days",
        "temp_bin_label",
        "land_cover_class",
        "landcover",
        "spearman_rho",
        "spearman_p",
        "p_display",
        "p_label",
        "n_cells",
        "n_obs",
        "alpha",
    ]
    export_columns = [
        column for column in export_columns if column in df.columns
    ]
    df[export_columns].to_csv(OUTPUT_SOURCE, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.6,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_box_aspect(AXES_BOX_ASPECT)

    ax.axhline(
        0.0,
        color=ZERO_LINE_COLOR,
        linestyle="--",
        linewidth=1.15,
        alpha=0.85,
        zorder=1,
    )

    separator = (
        X_POSITIONS["Barren"] + X_POSITIONS["Shrubland/Savanna"]
    ) / 2.0
    ax.axvline(
        separator,
        color="#bdbdbd",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )
    ax.text(
        X_POSITIONS["Barren"],
        0.985,
        "control",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.7,
        alpha=0.76,
    )

    rho_min = float(df["spearman_rho"].min())
    rho_max = float(df["spearman_rho"].max())
    span = max(rho_max - rho_min, 0.01)

    lower_pad = 0.34 * span
    upper_pad = 0.25 * span
    label_pad = 0.035 * span

    for row in df.itertuples(index=False):
        ax.vlines(
            row.x,
            0.0,
            row.spearman_rho,
            color=row.color,
            alpha=float(row.alpha),
            linewidth=STEM_LINE_WIDTH,
            zorder=2,
        )

        ax.hlines(
            row.spearman_rho,
            row.x - CAP_WIDTH / 2.0,
            row.x + CAP_WIDTH / 2.0,
            color=row.color,
            alpha=float(row.alpha),
            linewidth=STEM_LINE_WIDTH * 0.8,
            zorder=2,
        )

        ax.scatter(
            [row.x],
            [row.spearman_rho],
            s=POINT_SIZE,
            color=row.color,
            alpha=float(row.alpha),
            edgecolors="white",
            linewidths=POINT_EDGE_WIDTH,
            zorder=3,
        )

        if SHOW_RHO_LABELS:
            va = "top" if row.spearman_rho < 0 else "bottom"
            offset = -label_pad if row.spearman_rho < 0 else label_pad
            ax.text(
                row.x,
                row.spearman_rho + offset,
                f"{row.spearman_rho:+.3f}",
                ha="center",
                va=va,
                fontsize=8.6,
            )

        if SHOW_N_AND_P:
            ax.text(
                row.x,
                rho_min - 0.14 * span,
                f"n cells={int(row.n_cells):,}\n"
                f"p={row.p_label}",
                ha="center",
                va="top",
                fontsize=8.0,
                alpha=0.86,
                linespacing=1.15,
            )

        if SHOW_EXACT_P_LABELS_ABOVE:
            ax.text(
                row.x,
                rho_max + 0.07 * span,
                f"p={row.p_label}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    present_classes = [
        name for name in CLASS_ORDER if name in set(df["landcover"])
    ]
    tick_positions = [X_POSITIONS[name] for name in present_classes]
    tick_labels = [CLASS_LABELS[name] for name in present_classes]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    for tick_label, class_name in zip(ax.get_xticklabels(), present_classes):
        tick_label.set_color(CLASS_COLORS[class_name])
        tick_label.set_fontweight("semibold")

    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_xlabel("Land-cover class")
    ax.set_ylim(rho_min - lower_pad, rho_max + upper_pad)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.75,
        alpha=0.25,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.95,
        bottom=0.16,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    print(f"Main input: {main_path}")
    print(f"Surrogate input: {surrogate_path}")
    print(f"p-value source: {p_source}")
    print(f"Opacity mapping fixed to p in [{P_ALPHA_DARKEST:.0e}, {P_ALPHA_PALEST:.0e}]")
    print("Displayed class p-values:")
    for row in df.itertuples(index=False):
        print(
            f"  {row.landcover}: p={row.p_label}, alpha={row.alpha:.3f}"
        )
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source data: {OUTPUT_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())