#!/usr/bin/env python3

"""
Temperature dependence of the SII-SIF association.

Rendering only:
- harmonic detrending
- one fixed SII window (default: 28 days)
- x-axis: temperature classes
- y-axis: Spearman rho
- lines: geographic/ecological scenarios
- shaded bands: precomputed 95% cell-cluster bootstrap intervals
- opacity: conservative surrogate p-value,
  p_display = max(p_year, p_circular, p_block)

Expected input:
    results/supplementary_checks/temperature_scenarios_harmonic.csv

The script accepts common alternative column names used by the MAGNETO
supplementary-check outputs. It does not recalculate correlations, confidence
intervals, or surrogate tests.
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

# Intended location:
# scripts/figures/figure2_temperature_geographic_profiles.py
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "temperature_scenarios_harmonic.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figure2_temperature_geographic_profiles.png"
OUTPUT_PDF = OUTPUT_DIR / "figure2_temperature_geographic_profiles.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figure2_temperature_geographic_profiles_source.csv"


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

METHOD = "harmonic"
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

# Remove a scenario here if the plot becomes crowded.
SCENARIO_ORDER = [
    "pooled",
    "control_vegetated",
    "Sahara",
]

SCENARIO_LABELS = {
    "pooled": "Full analysis sample",
    "control_vegetated": "Persistently vegetated areas",
    "Sahara": "Sahara geographic control",
}

SCENARIO_COLORS = {
    "pooled": "#4055a8",
    "control_vegetated": "#2f7d4a",
    "Sahara": "#d19a24",
}

SCENARIO_LINESTYLES = {
    "pooled": "-",
    "control_vegetated": "-",
    "Sahara": "-",
}

FIGSIZE = (9.6, 6.1)
LINE_WIDTH = 2.5
MARKER_SIZE = 6.2
CI_ALPHA_FACTOR = 0.24

Y_LABEL = r"Spearman $\rho$"
X_LABEL = "Temperature regime"

# A continuous opacity map on a log-p scale.
# p <= P_MIN is fully opaque; p >= P_MAX is faint.
P_MIN = 0.02
P_MAX = 0.20
ALPHA_MIN = 0.2
ALPHA_MAX = 1.00


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

COLUMN_ALIASES = {
    "method": ["method"],
    "window": ["window_days", "sii_window_days", "window"],
    "scenario": ["scenario", "sample", "sample_type"],
    "temperature": ["temp_class", "temp_bin_label", "temperature_class"],
    "rho": ["rho", "spearman_rho"],
    "ci_low": ["ci_low", "boot_ci_lo", "ci_lower"],
    "ci_high": ["ci_high", "boot_ci_hi", "ci_upper"],
    "p_display": ["p_display"],
    "p_year": ["p_year", "p_year_perm"],
    "p_circular": ["p_circular", "p_circ_shift"],
    "p_block": ["p_block", "p_block_perm"],
    "eligible": ["eligible"],
    "n_obs": ["n_obs", "n"],
}


def resolve_column(
    df: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
) -> str | None:
    """Return the first available column matching a logical field."""
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Could not resolve {logical_name!r}. "
            f"Tried: {COLUMN_ALIASES[logical_name]}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


# =============================================================================
# HELPERS
# =============================================================================

def normalize_method(value: object) -> str:
    text = str(value).strip().lower()
    return {
        "cyclic_spline": "spline",
        "cyclic spline": "spline",
    }.get(text, text)


def normalize_temperature(value: object) -> str:
    text = str(value).strip()
    replacements = {
        "Warm Stress": "Warm_Stress",
        "Extreme Heat": "Extreme_Heat",
    }
    return replacements.get(text, text)


def normalize_scenario(value: object) -> str:
    text = str(value).strip()
    replacements = {
        "Pooled": "pooled",
        "full": "pooled",
        "full_sample": "pooled",
        "vegetated": "control_vegetated",
        "persistently_vegetated": "control_vegetated",
        "barren": "landcover_barren",
        "Landcover_Barren": "landcover_barren",
        "sahara": "Sahara",
    }
    return replacements.get(text, text)


def p_to_alpha(p_value: float) -> float:
    """
    Map p to opacity continuously on a base-10 logarithmic scale.

    Values p <= P_MIN are rendered with ALPHA_MAX (opaque), and values
    p >= P_MAX are rendered with ALPHA_MIN (faint).  Between these
    thresholds opacity decreases linearly in log10(p) space.
    """
    if not np.isfinite(p_value):
        return ALPHA_MIN

    p = float(np.clip(p_value, P_MIN, P_MAX))
    log_position = (
        np.log10(P_MAX) - np.log10(p)
    ) / (
        np.log10(P_MAX) - np.log10(P_MIN)
    )

    return float(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * log_position)


def prepare_data(path: Path) -> pd.DataFrame:
    """Load and standardize the precomputed figure-source table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{path}\n"
            "Run the temperature-scenario analysis/export step first."
        )

    raw = pd.read_csv(path)

    columns = {
        logical: resolve_column(
            raw,
            logical,
            required=logical not in {"method", "p_display", "eligible"},
        )
        for logical in COLUMN_ALIASES
    }

    out = pd.DataFrame(
        {
            "window_days": pd.to_numeric(raw[columns["window"]], errors="coerce"),
            "scenario": raw[columns["scenario"]].map(normalize_scenario),
            "temp_class": raw[columns["temperature"]].map(normalize_temperature),
            "rho": pd.to_numeric(raw[columns["rho"]], errors="coerce"),
            "ci_low": pd.to_numeric(raw[columns["ci_low"]], errors="coerce"),
            "ci_high": pd.to_numeric(raw[columns["ci_high"]], errors="coerce"),
            "n_obs": pd.to_numeric(raw[columns["n_obs"]], errors="coerce"),
        }
    )

    if columns["method"] is not None:
        out["method"] = raw[columns["method"]].map(normalize_method)
    else:
        out["method"] = METHOD

    if columns["eligible"] is not None:
        eligible = raw[columns["eligible"]]
        if eligible.dtype == bool:
            out["eligible"] = eligible
        else:
            out["eligible"] = (
                eligible.astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"})
            )
    else:
        out["eligible"] = True

    if columns["p_display"] is not None:
        out["p_display"] = pd.to_numeric(
            raw[columns["p_display"]],
            errors="coerce",
        )
    else:
        p_year_col = resolve_column(raw, "p_year")
        p_circular_col = resolve_column(raw, "p_circular")
        p_block_col = resolve_column(raw, "p_block")

        p_table = pd.DataFrame(
            {
                "p_year": pd.to_numeric(raw[p_year_col], errors="coerce"),
                "p_circular": pd.to_numeric(
                    raw[p_circular_col],
                    errors="coerce",
                ),
                "p_block": pd.to_numeric(raw[p_block_col], errors="coerce"),
            }
        )
        # Conservative display statistic already defined by the analysis design.
        out["p_display"] = p_table.max(axis=1, skipna=False)

    out = out[
        (out["method"] == METHOD)
        & (out["window_days"] == WINDOW_DAYS)
        & out["eligible"]
    ].copy()

    out["temp_class"] = pd.Categorical(
        out["temp_class"],
        categories=TEMP_ORDER,
        ordered=True,
    )

    unexpected = sorted(
        set(out["scenario"].dropna()) - set(SCENARIO_ORDER)
    )
    if unexpected:
        warnings.warn(
            "Ignoring scenarios not listed in SCENARIO_ORDER: "
            + ", ".join(unexpected)
        )

    out = out[out["scenario"].isin(SCENARIO_ORDER)].copy()

    if out.empty:
        raise ValueError(
            f"No eligible rows remained for method={METHOD!r}, "
            f"window={WINDOW_DAYS} days."
        )

    duplicated = out.duplicated(["scenario", "temp_class"], keep=False)
    if duplicated.any():
        examples = out.loc[
            duplicated,
            ["scenario", "temp_class"],
        ].drop_duplicates()
        raise ValueError(
            "The source table contains duplicate scenario × temperature rows:\n"
            + examples.to_string(index=False)
        )

    return out.sort_values(["scenario", "temp_class"])


def draw_segmented_profile(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    p_values: np.ndarray,
    *,
    color: str,
    linestyle: str,
) -> None:
    """Draw CI ribbons and line segments with p-dependent opacity."""
    alpha = np.array([p_to_alpha(p) for p in p_values], dtype=float)

    # Confidence bands, one segment at a time so opacity can vary with p.
    for i in range(len(x) - 1):
        values = [
            y[i],
            y[i + 1],
            ci_low[i],
            ci_low[i + 1],
            ci_high[i],
            ci_high[i + 1],
        ]
        if not np.all(np.isfinite(values)):
            continue

        segment_alpha = float(np.mean(alpha[i:i + 2]))
        ax.fill_between(
            x[i:i + 2],
            ci_low[i:i + 2],
            ci_high[i:i + 2],
            color=color,
            alpha=segment_alpha * CI_ALPHA_FACTOR,
            linewidth=0,
            zorder=1,
        )

    # Profile line, also segmented for variable opacity.
    for i in range(len(x) - 1):
        if not np.all(np.isfinite(y[i:i + 2])):
            continue

        segment_alpha = float(np.mean(alpha[i:i + 2]))
        ax.plot(
            x[i:i + 2],
            y[i:i + 2],
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            alpha=segment_alpha,
            solid_capstyle="round",
            zorder=3,
        )

    # Points.
    for xi, yi, ai in zip(x, y, alpha):
        if not np.isfinite(yi):
            continue
        ax.plot(
            xi,
            yi,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=float(ai),
            zorder=4,
        )


def add_p_opacity_legend(ax: plt.Axes) -> None:
    """Add a compact inset scale explaining the p-dependent opacity.

    The representative p-values are derived from the current opacity
    constants so the legend stays meaningful when P_MIN/P_MAX are edited.
    """
    representative_p = [
        P_MIN / 5.0,                 # clipped to the opaque plateau
        P_MIN,                       # beginning of the graded range
        np.sqrt(P_MIN * P_MAX),      # geometric midpoint
        P_MAX,                       # end of the graded range
        1.0,                         # faint saturation
    ]
    labels = [f"{p:.3g}" for p in representative_p]

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.6,
            alpha=p_to_alpha(p),
            label=label,
        )
        for p, label in zip(representative_p, labels)
    ]

    legend = ax.legend(
        handles=handles,
        title=r"Opacity: $p_{\mathrm{display}}$",
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        ncol=len(handles),
        frameon=True,
        framealpha=0.92,
        fontsize=7.6,
        title_fontsize=8.2,
        handlelength=1.7,
        columnspacing=0.85,
        borderpad=0.55,
    )
    legend.get_frame().set_edgecolor("#cccccc")
    ax.add_artist(legend)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_data(INPUT_CSV)
    df.to_csv(OUTPUT_SOURCE, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(TEMP_ORDER), dtype=float)
    y_limits_source: list[np.ndarray] = []

    available_scenarios = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in set(df["scenario"])
    ]

    for scenario in available_scenarios:
        d = (
            df[df["scenario"] == scenario]
            .set_index("temp_class")
            .reindex(TEMP_ORDER)
            .reset_index()
        )

        y = d["rho"].to_numpy(dtype=float)
        ci_low = d["ci_low"].to_numpy(dtype=float)
        ci_high = d["ci_high"].to_numpy(dtype=float)
        p_values = d["p_display"].to_numpy(dtype=float)

        draw_segmented_profile(
            ax,
            x,
            y,
            ci_low,
            ci_high,
            p_values,
            color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario],
        )

        # Label the most significant point on this scenario curve.
        finite_mask = np.isfinite(p_values)
        if finite_mask.any():
            min_idx = int(np.nanargmin(p_values))
            label_p = float(p_values[min_idx])
            label_n = int(d["n_obs"].iloc[min_idx])
            # Stagger text boxes so labels for different scenarios do not overlap.
            text_offset = {
                "pooled": (12, -16),
                "control_vegetated": (-10, -18),
                "Sahara": (-10, 14),
            }.get(scenario, (8, 10))
            ax.annotate(
                f"min p={label_p:.3g}\nN={label_n:,}",
                xy=(x[min_idx], y[min_idx]),
                xytext=text_offset,
                textcoords="offset points",
                fontsize=7.5,
                alpha=1.0,
                color=SCENARIO_COLORS[scenario],
                ha="left" if text_offset[0] > 0 else "right",
                va="bottom" if text_offset[1] > 0 else "top",
                clip_on=False,
                zorder=5,
            )

        for values in (y, ci_low, ci_high):
            finite = values[np.isfinite(values)]
            if finite.size:
                y_limits_source.append(finite)

    ax.axhline(
        0,
        color="black",
        linewidth=1.0,
        alpha=0.75,
        zorder=0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TEMP_LABELS[temp] for temp in TEMP_ORDER],
        rotation=0,
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # Symmetric y-axis preserves visual comparability around rho = 0.
    if y_limits_source:
        all_values = np.concatenate(y_limits_source)
        bound = max(float(np.max(np.abs(all_values))) * 1.05, 0.03)
        ax.set_ylim(-bound, bound)

    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scenario_handles = [
        Line2D(
            [0],
            [0],
            color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=5.5,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in available_scenarios
    ]

    scenario_legend = ax.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        title_fontsize=9.5,
    )
    scenario_legend.get_frame().set_edgecolor("#cccccc")
    ax.add_artist(scenario_legend)

    add_p_opacity_legend(ax)

    # No figure title: the caption should carry method/window details.
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source data: {OUTPUT_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())