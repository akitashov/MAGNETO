"""Supplementary Figure S3: temporal surrogate diagnostics.

Reads the prepared source table and draws observed Spearman rho against three
temporal null models (year permutation, circular shift, 30-day block permutation)
for the 28-day SII window, both harmonic and cyclic spline, full and persistently
vegetated samples.
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def format_p_value(p: float) -> str:
    """Format empirical p-value for annotation.

    Values below 0.001 are clipped to 0.001 for display; p<0.001 is never used.
    """
    if p < 0.001:
        return "p=0.001"
    return f"p={p:.3f}"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    source_path = project_root / "reports" / "figures" / "figureS3_surrogate_diagnostics_source.csv"
    png_path = project_root / "reports" / "figures" / "figureS3_surrogate_diagnostics.png"
    pdf_path = project_root / "reports" / "figures" / "figureS3_surrogate_diagnostics.pdf"

    print(f"Input source CSV: {source_path}")
    df = pd.read_csv(source_path)
    print(f"Selected rows: {len(df)}")
    print(f"Sample types: {sorted(df['sample_type'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Surrogate modes: {sorted(df['surrogate_mode'].unique())}")
    print(f"Windows: {sorted(df['sii_window'].unique())}")
    print(f"p-value source: empirical p_value from surrogate_summary.csv")

    # Validate expected structure: 3 panels × 4 rows = 12 rows.
    expected_rows = 12
    if len(df) != expected_rows:
        print(
            f"WARNING: expected {expected_rows} rows, got {len(df)}.",
            file=sys.stderr,
        )

    # Consistency check for p-values.
    # Allow a tiny tolerance for floating-point representation of exact minima.
    min_reportable = 1.0 / (df["n_completed"] + 1)
    inconsistent = df["p_value"] < (min_reportable - 1e-12)
    if inconsistent.any():
        n_bad = inconsistent.sum()
        print(
            f"WARNING: {n_bad} row(s) have p_value < 1/(n_completed+1).",
            file=sys.stderr,
        )

    # Panel ordering.
    panel_order = ["A", "B", "C"]
    panel_titles = {
        "A": "A   Year permutation",
        "B": "B   Circular shift",
        "C": "C   30-day block permutation",
    }
    surrogate_mode_by_panel = {
        "A": "year_perm",
        "B": "circ_shift",
        "C": "block_perm",
    }

    method_labels = {
        "harmonic": "Harmonic",
        "cyclic_spline": "Cyclic spline",
    }

    # Row labels.
    def row_label(row: pd.Series) -> str:
        return f"{row['sample_label']} — {method_labels[row['method']]}"

    # Set restrained journal style.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 400,
            "savefig.dpi": 400,
        }
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(6.5, 7.0),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )

    for ax, panel in zip(axes, panel_order):
        mode = surrogate_mode_by_panel[panel]
        panel_df = df[df["panel"] == panel].sort_values("display_order")

        y_positions = list(range(len(panel_df) - 1, -1, -1))
        labels = [row_label(row) for _, row in panel_df.iterrows()]

        for y, (_, row) in zip(y_positions, panel_df.iterrows()):
            null_q025 = row["null_q025"]
            null_median = row["null_median"]
            null_q975 = row["null_q975"]
            rho_obs = row["rho_obs"]
            p_value = row["p_value"]

            # Null 2.5th–97.5th percentile interval.
            ax.plot(
                [null_q025, null_q975],
                [y, y],
                color="#444444",
                linewidth=1.2,
                solid_capstyle="butt",
                zorder=1,
            )

            # Null median marker.
            ax.plot(
                null_median,
                y,
                marker="o",
                markersize=4,
                color="#444444",
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=2,
            )

            # Observed rho marker.
            ax.plot(
                rho_obs,
                y,
                marker="D",
                markersize=7,
                color="#c0392b",
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
                label="Observed" if y == y_positions[0] and panel == "A" else "",
            )

            # p-value annotation, placed just to the right of the observed marker
            # and the null interval so it does not overlap either.
            p_text = format_p_value(p_value)
            text_x = max(rho_obs, null_q975) + 0.006
            ax.text(
                text_x,
                y,
                p_text,
                va="center",
                ha="left",
                fontsize=8,
                color="#111111",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )

        ax.axvline(x=0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(panel_titles[panel], loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

    # Shared x-axis label on the bottom panel.
    axes[-1].set_xlabel("Spearman ρ", fontsize=10)

    # Determine a clean x-axis range that includes all intervals and annotations.
    x_min = min(df["null_q025"].min(), df["rho_obs"].min())
    x_max = max(df["null_q975"].max(), df["rho_obs"].max())
    pad = 0.03
    axes[-1].set_xlim(x_min - pad, x_max + pad + 0.03)

    # Shared legend for observed marker.
    handles, labels_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=1,
            frameon=False,
            fontsize=9,
        )

    fig.subplots_adjust(bottom=0.08)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)

    export_figure_text(fig, source_path, __doc__, png_path)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved source CSV: {source_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
