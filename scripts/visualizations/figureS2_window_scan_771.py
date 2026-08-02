#!/usr/bin/env python3
"""
Supplementary Figure S2. Descriptive SII accumulation-window scan (1–90 days)
for SIF 771 nm.

FINAL DISPLAY LOGIC
-------------------
The figure uses the pairwise-matched sample only. This is the appropriate
comparison for harmonic versus cyclic-spline detrending because both methods
are evaluated on exactly the same observations.

The pooled-full and pairwise-matched curves are nearly identical for the
current 0.5° grid run, so plotting all four curves creates redundant
overlapping lines and can make the method comparison visually misleading.
The exact sample sizes are read from the source CSV at render time.

Display:
- x-axis: SII accumulation window (days)
- y-axis: Spearman rho
- two curves:
    pairwise-matched harmonic
    pairwise-matched cyclic spline
- vertical reference lines at 21 and 28 days
- no optimum or best-window annotation

This is a rendering-only script.

Input:
    results/supplementary_checks/window_scan_771.csv

Outputs:
    reports/figures/figureS2_window_scan_771.png
    reports/figures/figureS2_window_scan_771.pdf
    reports/figures/figureS2_window_scan_771_source.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "supplementary_checks"
    / "window_scan_771.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS2_window_scan_771.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS2_window_scan_771.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS2_window_scan_771_source.csv"

SAMPLE_TYPE = "pairwise_matched"
METHOD_ORDER = ["harmonic", "cyclic_spline"]

METHOD_LABELS = {
    "harmonic": "Harmonic",
    "cyclic_spline": "Cyclic spline",
}

METHOD_LINESTYLES = {
    "harmonic": "-",
    "cyclic_spline": "--",
}

METHOD_LINEWIDTHS = {
    "harmonic": 2.6,
    "cyclic_spline": 2.4,
}

REFERENCE_WINDOWS = [21, 28]

FIGSIZE = (7.2, 5.8)
AXES_BOX_ASPECT = 0.78
DPI = 400


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {
        "sample_type",
        "method",
        "sii_window_days",
        "spearman_rho",
        "n_obs",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    df = df.copy()
    df["sample_type"] = df["sample_type"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip()
    df["sii_window_days"] = pd.to_numeric(
        df["sii_window_days"],
        errors="coerce",
    )
    df["spearman_rho"] = pd.to_numeric(
        df["spearman_rho"],
        errors="coerce",
    )
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")

    selected = df.loc[
        df["sample_type"].eq(SAMPLE_TYPE)
        & df["method"].isin(METHOD_ORDER)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No rows found for sample_type={SAMPLE_TYPE!r}."
        )

    duplicated = selected.duplicated(
        ["method", "sii_window_days"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError(
            "Duplicate method/window rows detected:\n"
            + selected.loc[
                duplicated,
                ["method", "sii_window_days"],
            ].to_string(index=False)
        )

    counts = selected.groupby("method")["sii_window_days"].nunique()
    for method in METHOD_ORDER:
        if method not in counts:
            raise ValueError(f"Method missing: {method}")
        if counts[method] != 90:
            raise ValueError(
                f"Expected 90 windows for {method}, found {counts[method]}."
            )

    n_obs_by_method = (
        selected.groupby("method")["n_obs"]
        .unique()
        .to_dict()
    )

    harmonic_n = n_obs_by_method["harmonic"]
    spline_n = n_obs_by_method["cyclic_spline"]

    if (
        len(harmonic_n) != 1
        or len(spline_n) != 1
        or harmonic_n[0] != spline_n[0]
    ):
        raise ValueError(
            "Pairwise-matched methods do not have identical n_obs:\n"
            f"harmonic={harmonic_n}, cyclic_spline={spline_n}"
        )

    return selected.sort_values(
        ["method", "sii_window_days"]
    ).reset_index(drop=True)


def compare_pooled_and_matched(all_data: pd.DataFrame) -> None:
    """
    Print the maximum pooled-full versus pairwise-matched rho difference.
    This is diagnostic only and does not alter the figure.
    """
    for method in METHOD_ORDER:
        subset = all_data.loc[
            all_data["method"].eq(method)
            & all_data["sample_type"].isin(
                ["pooled_full", "pairwise_matched"]
            ),
            [
                "sample_type",
                "sii_window_days",
                "spearman_rho",
            ],
        ].copy()

        pivot = subset.pivot(
            index="sii_window_days",
            columns="sample_type",
            values="spearman_rho",
        )

        required_columns = {"pooled_full", "pairwise_matched"}
        if not required_columns.issubset(pivot.columns):
            print(
                f"Could not compare pooled and matched for {method}: "
                "one sample type is missing."
            )
            continue

        difference = (
            pivot["pooled_full"]
            - pivot["pairwise_matched"]
        ).abs()

        print(
            f"{method}: maximum |pooled - matched| rho = "
            f"{difference.max():.12g}"
        )


def save_source(df: pd.DataFrame) -> None:
    columns = [
        "sample_type",
        "method",
        "residual_column",
        "sii_window",
        "sii_window_days",
        "spearman_rho",
        "spearman_p",
        "n_obs",
        "n_cells",
        "sample_label",
    ]
    columns = [
        column for column in columns if column in df.columns
    ]
    df[columns].to_csv(OUTPUT_SOURCE, index=False)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = pd.read_csv(INPUT_CSV, low_memory=False)
    for column in ["sii_window_days", "spearman_rho", "n_obs"]:
        all_data[column] = pd.to_numeric(
            all_data[column],
            errors="coerce",
        )

    df = load_data()
    save_source(df)
    compare_pooled_and_matched(all_data)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_box_aspect(AXES_BOX_ASPECT)

    for method in METHOD_ORDER:
        subset = df.loc[
            df["method"].eq(method)
        ].sort_values("sii_window_days")

        ax.plot(
            subset["sii_window_days"],
            subset["spearman_rho"],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=METHOD_LINEWIDTHS[method],
            label=METHOD_LABELS[method],
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

    for window in REFERENCE_WINDOWS:
        ax.axvline(
            window,
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )

    rho_min = float(df["spearman_rho"].min())
    rho_max = float(df["spearman_rho"].max())
    rho_span = max(rho_max - rho_min, 0.01)

    for window in REFERENCE_WINDOWS:
        ax.text(
            window,
            rho_max + 0.035 * rho_span,
            f"{window} d",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xlabel("SII accumulation window (days)")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_xlim(1, 90)
    ax.set_xticks([1, 7, 14, 21, 28, 42, 56, 70, 84, 90])
    ax.set_ylim(
        rho_min - 0.08 * rho_span,
        rho_max + 0.10 * rho_span,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=9.2,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.95,
        bottom=0.13,
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

    n_obs = int(df["n_obs"].iloc[0])

    print(f"Input: {INPUT_CSV}")
    print(f"Sample type: {SAMPLE_TYPE}")
    print(f"Common n_obs: {n_obs:,}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source: {OUTPUT_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())