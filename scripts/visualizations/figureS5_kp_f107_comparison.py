"""Supplementary Figure S5: Kp and F10.7 comparisons.

Shows robustness to alternative geomagnetic index (Kp) and distinguishes the
SII effect from solar activity (F10.7). Uses pre-computed Spearman correlations
from the supplementary_checks pipeline stage; no new correlations are computed.
"""
from __future__ import annotations
from _figure_text_export import export_figure_text

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_KP = PROJECT_ROOT / "results" / "supplementary_checks" / "kp_comparison.csv"
INPUT_F107 = PROJECT_ROOT / "results" / "supplementary_checks" / "f107_comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_ID = "figureS5_kp_f107_comparison"
METHOD = "harmonic"
WINDOW_DAYS = 28

# Ecological temperature-bin order used consistently across the figure.
BIN_ORDER = ["Frozen", "Cold", "Cool", "Optimum", "Warm_Stress", "Extreme_Heat"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_filter(path: Path) -> pd.DataFrame:
    """Load a comparison CSV and keep only the primary method/window rows."""
    df = pd.read_csv(path)
    selected = df[(df["method"] == METHOD) & (df["sii_window_days"] == WINDOW_DAYS)].copy()
    return selected


def make_panel_data(
    df: pd.DataFrame,
    sii_samples: list[str],
    comparator_samples: list[str],
    comparator_name: str,
    source_file: str,
    panel: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sii_df, comparator_df) ready for plotting, plus an audit trail."""
    sii = df[df["sample_type"].isin(sii_samples)].copy()
    comp = df[df["sample_type"].isin(comparator_samples)].copy()

    # Distinguish pooled vs binned rows using non-null temp_bin_label.
    sii["is_pooled"] = sii["temp_bin_label"].isna() | (sii["temp_bin_label"] == "")
    comp["is_pooled"] = comp["temp_bin_label"].isna() | (comp["temp_bin_label"] == "")

    # Build plot categories: Pooled first (when present), then ordered bins.
    def prepare_plot(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        sub = sub.copy()
        sub["sample_label"] = label
        sub["comparator"] = comparator_name
        sub["panel"] = panel
        sub["source_file"] = source_file
        sub["figure_id"] = FIGURE_ID

        pooled = sub[sub["is_pooled"]].copy()
        binned = sub[~sub["is_pooled"]].copy()

        if not binned.empty:
            # Sort bins by the predefined ecological order.
            binned["bin_rank"] = binned["temp_bin_label"].map(
                {b: i for i, b in enumerate(BIN_ORDER)}
            )
            binned = binned.sort_values("bin_rank")

        # Concatenate with Pooled at the far left.
        combined = pd.concat([pooled, binned], ignore_index=True)
        combined["display_order"] = range(len(combined))
        return combined

    sii_prepared = prepare_plot(sii, "SII")
    comp_prepared = prepare_plot(comp, comparator_name)

    return sii_prepared, comp_prepared


def plot_panel(ax, sii_df: pd.DataFrame, comp_df: pd.DataFrame, title: str) -> None:
    """Draw one comparison panel on the supplied axis."""
    n = len(sii_df)
    x = range(n)
    offset = 0.15

    ax.plot(
        [i - offset for i in x],
        sii_df["spearman_rho"],
        marker="o",
        color="#1f77b4",
        label="SII",
        linewidth=1.2,
        markersize=5,
    )
    ax.plot(
        [i + offset for i in x],
        comp_df["spearman_rho"],
        marker="s",
        color="#ff7f0e",
        label=comp_df["comparator"].iloc[0],
        linewidth=1.2,
        markersize=5,
    )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["Pooled" if (pd.isna(lbl) or lbl == "") else lbl.replace("_", " ") for lbl in sii_df["temp_bin_label"]],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Spearman ρ")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Input files:\n  {INPUT_KP}\n  {INPUT_F107}")

    kp_df = load_and_filter(INPUT_KP)
    f107_df = load_and_filter(INPUT_F107)

    print(f"\nSelected rows: Kp={len(kp_df)}, F10.7={len(f107_df)}")
    print(f"Methods: {kp_df['method'].unique().tolist()} | Windows: {kp_df['sii_window_days'].unique().tolist()}")
    print(f"Kp sample types: {kp_df['sample_type'].unique().tolist()}")
    print(f"F10.7 sample types: {f107_df['sample_type'].unique().tolist()}")

    kp_sii, kp_comp = make_panel_data(
        kp_df,
        sii_samples=["pooled_sii", "temperature_sii"],
        comparator_samples=["pooled_kp", "temperature_kp"],
        comparator_name="Kp",
        source_file=str(INPUT_KP.relative_to(PROJECT_ROOT)),
        panel="A",
    )

    f107_sii, f107_comp = make_panel_data(
        f107_df,
        sii_samples=["pooled_sii", "temperature_sii"],
        comparator_samples=["pooled_f10_7", "temperature_f10_7"],
        comparator_name="F10.7",
        source_file=str(INPUT_F107.relative_to(PROJECT_ROOT)),
        panel="B",
    )

    print(f"\nMatched n_obs: Kp panel={kp_sii['n_obs'].sum():,}, F10.7 panel={f107_sii['n_obs'].sum():,}")

    # Combine source records for archival.
    source_cols = [
        "figure_id", "panel", "display_order", "source_file", "sample_type",
        "sample_label", "comparator", "method", "sii_window_days", "temp_bin_label",
        "spearman_rho", "n_obs", "n_cells", "exposure",
    ]
    source_df = pd.concat(
        [kp_sii[source_cols], kp_comp[source_cols], f107_sii[source_cols], f107_comp[source_cols]],
        ignore_index=True,
    )
    source_path = OUTPUT_DIR / f"{FIGURE_ID}_source.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_df.to_csv(source_path, index=False)

    # Determine shared y-axis limits across both panels.
    all_rho = pd.concat([kp_sii, kp_comp, f107_sii, f107_comp])["spearman_rho"]
    y_min, y_max = all_rho.min(), all_rho.max()
    margin = 0.05 * (y_max - y_min)
    y_lo = y_min - margin
    y_hi = y_max + margin

    # Plotting setup: restrained journal style.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    plot_panel(axes[0], kp_sii, kp_comp, "A   SII vs. Kp")
    plot_panel(axes[1], f107_sii, f107_comp, "B   SII vs. F10.7")

    for ax in axes:
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{FIGURE_ID}.png"
    pdf_path = OUTPUT_DIR / f"{FIGURE_ID}.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)

    export_figure_text(fig, source_path, __doc__, png_path)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    print(f"\nOutput paths:")
    print(f"  PNG: {png_path} ({png_path.stat().st_size:,} bytes)")
    print(f"  PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    print(f"  CSV: {source_path} ({source_path.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
