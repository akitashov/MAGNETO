#!/usr/bin/env python3
"""
V_Surrogate_Spectrum.py

Supplementary figure for surrogate analysis:
Panel A: observed rho(window) + surrogate median + 95% envelope
Panel B: surrogate distribution of signed AUC with observed value

Input:
    results/surrogate_test/surrogate_spectrum_<target>_<scenario>_<bin>.csv
    results/surrogate_test/surrogate_metrics_<target>_<scenario>_<bin>.csv

Output:
    reports/figures/supplementary/Fig_Surrogate_<target>_<scenario>_<bin>.pdf
    reports/figures/supplementary/Fig_Surrogate_<target>_<scenario>_<bin>.png
    reports/figures/supplementary/Fig_Surrogate_<target>_<scenario>_<bin>_plotdata.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _Common import Config as CommonConfig


class Config:
    PROJECT_ROOT = CommonConfig.PROJECT_ROOT
    INPUT_DIR = CommonConfig.RESULTS_DIR / "surrogate_test"
    OUTPUT_DIR = CommonConfig.REPORTS_ROOT / "figures" / "supplementary"

    TARGET = "sif_771nm"
    SCENARIO = "Global_High_LAI"
    TEMP_BIN = "Cold"
    METRIC = "signed_auc"

    FIG_SIZE = (13.5, 5.8)

    FONT_FAMILY = "DejaVu Sans"
    BASE_FONT = 12
    AXIS_FONT = 14
    TICK_FONT = 11
    LEGEND_FONT = 10
    TITLE_FONT = 12
    PANEL_FONT = 18

    OBS_COLOR = "#1f3b73"      # dark blue
    SURR_MEDIAN = "#555555"    # dark gray
    SURR_FILL = "#d9d9d9"      # light gray
    SURR_IQR = "#bdbdbd"       # darker gray
    HIST_COLOR = "#cfcfcf"     # histogram fill
    ZERO_LINE = "#666666"
    AUC_BAND = "#e8eef9"       # very light blue
    GRID_COLOR = "#d0d0d0"

    Y_LIM = (-0.22, 0.22)

    SAVE_DPI = 300


def make_stem(target: str, scenario: str, temp_bin: str) -> str:
    return f"{target}_{scenario}_{temp_bin}".replace(" ", "_")


def load_inputs(target: str, scenario: str, temp_bin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = make_stem(target, scenario, temp_bin)

    spectrum_path = Config.INPUT_DIR / f"surrogate_spectrum_{stem}.csv"
    metrics_path = Config.INPUT_DIR / f"surrogate_metrics_{stem}.csv"

    if not spectrum_path.exists():
        raise FileNotFoundError(f"Missing spectrum file: {spectrum_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    spectrum_df = pd.read_csv(spectrum_path)
    metrics_df = pd.read_csv(metrics_path)
    return spectrum_df, metrics_df


def build_spectrum_summary(spectrum_df: pd.DataFrame) -> pd.DataFrame:
    obs = (
        spectrum_df.loc[spectrum_df["kind"] == "observed", ["window", "rho"]]
        .rename(columns={"rho": "rho_obs"})
        .sort_values("window")
        .reset_index(drop=True)
    )

    surr = spectrum_df.loc[spectrum_df["kind"] == "surrogate", ["window", "rho"]].copy()

    surr_summary = (
        surr.groupby("window")["rho"]
        .agg(
            rho_surr_median="median",
            rho_surr_q025=lambda x: np.quantile(x, 0.025),
            rho_surr_q250=lambda x: np.quantile(x, 0.250),
            rho_surr_q750=lambda x: np.quantile(x, 0.750),
            rho_surr_q975=lambda x: np.quantile(x, 0.975),
        )
        .reset_index()
        .sort_values("window")
    )

    out = pd.merge(obs, surr_summary, on="window", how="inner")
    return out


def build_metric_distribution(
    spectrum_df: pd.DataFrame,
    windows: np.ndarray,
    auc_min_w: int,
    auc_max_w: int,
    metric_name: str,
) -> tuple[np.ndarray, float]:
    in_band = (windows >= auc_min_w) & (windows <= auc_max_w)
    if not np.any(in_band):
        raise ValueError("No windows inside requested AUC band.")

    obs = spectrum_df.loc[spectrum_df["kind"] == "observed"].sort_values("window")
    surr = spectrum_df.loc[spectrum_df["kind"] == "surrogate"].copy()

    obs_rho = obs["rho"].to_numpy(dtype=float)
    obs_w = obs["window"].to_numpy(dtype=float)

    if metric_name == "signed_auc":
        obs_metric = float(np.trapz(obs_rho[in_band], x=obs_w[in_band]))
    elif metric_name == "max_abs_rho":
        obs_metric = float(np.nanmax(np.abs(obs_rho[in_band])))
    elif metric_name == "mean_rho":
        obs_metric = float(np.nanmean(obs_rho[in_band]))
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    surr_metrics = []
    for s_id, grp in surr.groupby("surrogate_id"):
        grp = grp.sort_values("window")
        rho = grp["rho"].to_numpy(dtype=float)
        w = grp["window"].to_numpy(dtype=float)
        mask = (w >= auc_min_w) & (w <= auc_max_w)
        if not np.any(mask):
            continue

        if metric_name == "signed_auc":
            val = float(np.trapz(rho[mask], x=w[mask]))
        elif metric_name == "max_abs_rho":
            val = float(np.nanmax(np.abs(rho[mask])))
        else:
            val = float(np.nanmean(rho[mask]))

        surr_metrics.append(val)

    return np.asarray(surr_metrics, dtype=float), obs_metric


def monte_carlo_pvalue(obs: float, surr: np.ndarray, two_sided: bool = True) -> float:
    if two_sided:
        count = np.sum(np.abs(surr) >= abs(obs))
    else:
        count = np.sum(surr >= obs)
    return (1.0 + count) / (len(surr) + 1.0)


def plot_figure(
    spectrum_plot_df: pd.DataFrame,
    surr_metric: np.ndarray,
    obs_metric: float,
    target: str,
    scenario: str,
    temp_bin: str,
    metric_name: str,
    auc_min_w: int,
    auc_max_w: int,
) -> None:
    plt.rcParams["font.family"] = Config.FONT_FAMILY
    plt.rcParams["font.size"] = Config.BASE_FONT

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=Config.FIG_SIZE,
        gridspec_kw={"width_ratios": [1.7, 1.0]}
    )

    x = spectrum_plot_df["window"].to_numpy(dtype=float)
    y_obs = spectrum_plot_df["rho_obs"].to_numpy(dtype=float)
    y_med = spectrum_plot_df["rho_surr_median"].to_numpy(dtype=float)
    y_lo = spectrum_plot_df["rho_surr_q025"].to_numpy(dtype=float)
    y_hi = spectrum_plot_df["rho_surr_q975"].to_numpy(dtype=float)
    y_q25 = spectrum_plot_df["rho_surr_q250"].to_numpy(dtype=float)
    y_q75 = spectrum_plot_df["rho_surr_q750"].to_numpy(dtype=float)

    # Panel A
    ax1.axvspan(auc_min_w, auc_max_w, color=Config.AUC_BAND, alpha=0.75, zorder=0)

    ax1.fill_between(
        x, y_lo, y_hi,
        color=Config.SURR_FILL,
        alpha=0.85,
        linewidth=0,
        label="Surrogate 95% range",
        zorder=1
    )

    ax1.fill_between(
        x, y_q25, y_q75,
        color=Config.SURR_IQR,
        alpha=0.75,
        linewidth=0,
        label="Surrogate IQR",
        zorder=2
    )

    ax1.plot(
        x, y_med,
        color=Config.SURR_MEDIAN,
        linewidth=2.0,
        linestyle="--",
        label="Surrogate median",
        zorder=3
    )

    ax1.plot(
        x, y_obs,
        color=Config.OBS_COLOR,
        linewidth=2.8,
        label="Observed",
        zorder=4
    )

    ax1.axhline(0, color=Config.ZERO_LINE, linewidth=1.0)
    ax1.set_ylim(Config.Y_LIM)
    ax1.set_xlabel("Integration window (days)", fontsize=Config.AXIS_FONT)
    ax1.set_ylabel(r"Spearman correlation ($\rho$)", fontsize=Config.AXIS_FONT)
    ax1.tick_params(axis="both", labelsize=Config.TICK_FONT)
    ax1.grid(True, color=Config.GRID_COLOR, linewidth=0.8, alpha=0.6)

    ax1.text(
        0.01, 0.98, "A",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=Config.PANEL_FONT, fontweight="bold"
    )

    ax1.text(
        0.99, 0.98,
        f"{target} | {scenario} | {temp_bin}",
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=Config.TITLE_FONT
    )

    ax1.legend(
        loc="lower right",
        fontsize=Config.LEGEND_FONT,
        frameon=True,
        framealpha=0.95
    )

    # Panel B
    bins = min(24, max(12, int(np.sqrt(len(surr_metric)) * 1.8)))
    ax2.hist(
        surr_metric,
        bins=bins,
        color=Config.HIST_COLOR,
        edgecolor="white",
        linewidth=0.8
    )
    ax2.axvline(obs_metric, color=Config.OBS_COLOR, linewidth=2.8)

    two_sided = metric_name in {"signed_auc", "mean_rho"}
    p_emp = monte_carlo_pvalue(obs_metric, surr_metric, two_sided=two_sided)

    ax2.set_xlabel(f"{metric_name} ({auc_min_w}–{auc_max_w} d)", fontsize=Config.AXIS_FONT)
    ax2.set_ylabel("Count", fontsize=Config.AXIS_FONT)
    ax2.tick_params(axis="both", labelsize=Config.TICK_FONT)
    ax2.grid(True, axis="y", color=Config.GRID_COLOR, linewidth=0.8, alpha=0.6)

    ax2.text(
        0.01, 0.98, "B",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=Config.PANEL_FONT, fontweight="bold"
    )

    ax2.text(
        0.98, 0.98,
        f"Observed = {obs_metric:.3f}\n"
        f"Empirical p = {p_emp:.3f}\n"
        f"N surrogates = {len(surr_metric)}",
        transform=ax2.transAxes,
        ha="right", va="top",
        fontsize=Config.TITLE_FONT
    )

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = make_stem(target, scenario, temp_bin)

    out_pdf = Config.OUTPUT_DIR / f"Fig_Surrogate_{stem}.pdf"
    out_png = Config.OUTPUT_DIR / f"Fig_Surrogate_{stem}.png"
    out_csv = Config.OUTPUT_DIR / f"Fig_Surrogate_{stem}_plotdata.csv"

    fig.savefig(out_pdf, dpi=Config.SAVE_DPI, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png, dpi=Config.SAVE_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    export_df = spectrum_plot_df.copy()
    export_df["metric_name"] = metric_name
    export_df["metric_observed"] = obs_metric
    export_df["metric_surrogate_p"] = monte_carlo_pvalue(obs_metric, surr_metric, two_sided=two_sided)
    export_df.to_csv(out_csv, index=False)

    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_csv}")


def main():
    spectrum_df, metrics_df = load_inputs(
        target=Config.TARGET,
        scenario=Config.SCENARIO,
        temp_bin=Config.TEMP_BIN,
    )

    spectrum_plot_df = build_spectrum_summary(spectrum_df)

    if not {"auc_window_min", "auc_window_max"}.issubset(metrics_df.columns):
        raise ValueError("Metrics file does not contain auc_window_min / auc_window_max.")

    metric_row = metrics_df.loc[metrics_df["metric"] == Config.METRIC]
    if metric_row.empty:
        raise ValueError(f"Metric '{Config.METRIC}' not found in metrics file.")

    auc_min_w = int(metric_row["auc_window_min"].iloc[0])
    auc_max_w = int(metric_row["auc_window_max"].iloc[0])

    windows = spectrum_plot_df["window"].to_numpy(dtype=float)
    surr_metric, obs_metric = build_metric_distribution(
        spectrum_df=spectrum_df,
        windows=windows,
        auc_min_w=auc_min_w,
        auc_max_w=auc_max_w,
        metric_name=Config.METRIC,
    )

    plot_figure(
        spectrum_plot_df=spectrum_plot_df,
        surr_metric=surr_metric,
        obs_metric=obs_metric,
        target=Config.TARGET,
        scenario=Config.SCENARIO,
        temp_bin=Config.TEMP_BIN,
        metric_name=Config.METRIC,
        auc_min_w=auc_min_w,
        auc_max_w=auc_max_w,
    )


if __name__ == "__main__":
    main()