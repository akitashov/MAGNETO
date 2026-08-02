#!/usr/bin/env python3
"""
Figure S6: stability of beta_SII across SII accumulation windows.

For every scenario × temperature regime, the panel shows how the median
beta_SII (solid line) and its inter-quartile range (shaded band) vary as the
SII window changes from 1 to 28 days. The statistics are taken across all
PAR × VPD combinations at each SII window.

Input:
    results/environmental_driver_matrix_full.parquet

Outputs:
    reports/figures/figureS6_environmental_driver_stability.png
    reports/figures/figureS6_environmental_driver_stability.pdf
    reports/figures/figureS6_environmental_driver_stability_source.csv
"""
from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_PATH = PROJECT_ROOT / "results" / "environmental_driver_matrix.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figureS6_environmental_driver_stability.png"
OUTPUT_PDF = OUTPUT_DIR / "figureS6_environmental_driver_stability.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "figureS6_environmental_driver_stability_source.csv"

TEMP_ORDER = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]
TEMP_LABELS = {
    "Frozen": "Frozen",
    "Cold": "Cold",
    "Cool": "Cool",
    "Optimum": "Optimum",
    "Warm_Stress": "Warm Stress",
    "Extreme_Heat": "Extreme Heat",
}

SCENARIO_COLORS = {
    "full_sample": "#1f77b4",
    "persistently_vegetated": "#2ca02c",
}

SCENARIO_LABELS = {
    "full_sample": "Full sample",
    "persistently_vegetated": "Persistently vegetated",
}

FIGSIZE = (12.0, 7.0)
DPI = 400


def load_source() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input matrix not found:\n{INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    df["sii_window"] = pd.to_numeric(df["sii_window"], errors="coerce")
    df["beta_sii"] = pd.to_numeric(df["beta_sii"], errors="coerce")

    summary = (
        df.groupby(["scenario", "temp_bin_label", "sii_window"])["beta_sii"]
        .agg(
            median="median",
            q25=lambda x: float(np.quantile(x, 0.25)),
            q75=lambda x: float(np.quantile(x, 0.75)),
            p05=lambda x: float(np.quantile(x, 0.05)),
            p95=lambda x: float(np.quantile(x, 0.95)),
            n_models="size",
        )
        .reset_index()
    )

    summary["temp_bin_label"] = pd.Categorical(
        summary["temp_bin_label"], categories=TEMP_ORDER, ordered=True
    )
    return summary.sort_values(["scenario", "temp_bin_label", "sii_window"]).reset_index(drop=True)


def main() -> int:
    print("Figure S6 — beta_SII stability across SII windows")
    df = load_source()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_SOURCE, index=False)

    scenarios = sorted(df["scenario"].unique())
    n_scenarios = len(scenarios)
    n_cols = 3
    n_rows = int(np.ceil(len(TEMP_ORDER) / n_cols))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for ax, temp in zip(axes_flat, TEMP_ORDER):
        for scenario in scenarios:
            sub = df[(df["scenario"] == scenario) & (df["temp_bin_label"] == temp)]
            if sub.empty:
                continue
            color = SCENARIO_COLORS.get(scenario, "black")
            ax.plot(
                sub["sii_window"],
                sub["median"],
                color=color,
                linewidth=1.6,
                label=SCENARIO_LABELS.get(scenario, scenario),
            )
            ax.fill_between(
                sub["sii_window"],
                sub["q25"],
                sub["q75"],
                color=color,
                alpha=0.18,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(TEMP_LABELS.get(temp, temp), fontsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(TEMP_ORDER):]:
        ax.set_visible(False)

    fig.supxlabel("SII accumulation window (days)", y=0.02, fontsize=11)
    fig.supylabel(r"SII partial coefficient $\beta_{\mathrm{SII}}$", x=0.02, fontsize=11)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)

    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.96])
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE, __doc__, OUTPUT_PNG)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved source: {OUTPUT_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
