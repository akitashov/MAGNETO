#!/usr/bin/env python3
"""
S07_make_graphical_abstract_matrix.py — Graphical abstract matrix.

Rows: temperature classes (+ an overall pooled row).
Columns: analysis scenarios (global pooled, vegetated control, strict low-LAI
control, Sahara control, SAA control).
Panels: one per temporal-surrogate mode (year permutation, circular shift,
block permutation).

Hue encodes Spearman rho. Saturation/alpha encodes empirical p. Cells are never
hidden. Temperature estimates for control scenarios come from S04b; SAA
estimates come from S04 and S04b.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _Common import SCENARIO_REGISTRY
from _supplementary_checks_common import (
    load_main_fixed_window_results,
    load_main_surrogate_summary,
    SUPPLEMENTARY_CHECKS_RESULTS,
    SUPPLEMENTARY_CHECKS_FIGURES,
    make_supplementary_checks_dirs,
    atomic_write,
    savefig,
    load_supplementary_checks_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Graphical abstract matrix")
    p.add_argument("--method", default="harmonic")
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--figure", action="store_true", default=False,
                   help="Render PNG figure in addition to writing CSV matrix.")
    return p.parse_args(argv)


def _load_main_p(surr: pd.DataFrame, method: str, window: int,
                 sample_type: str, mode: str, temp: str | None = None) -> float:
    ss = surr[(surr["method"] == method) & (surr["sii_window"] == f"sii_{window}d")]
    ss = ss[(ss["surrogate_mode"] == mode) & (ss["sample_type"] == sample_type)]
    if temp is not None:
        if temp == "Pooled":
            ss = ss[ss["temp_bin_label"].isna()]
        else:
            ss = ss[ss["temp_bin_label"] == temp]
    if ss.empty:
        return np.nan
    return float(ss["p_value"].iloc[0])


def _load_main_rho(fixed: pd.DataFrame, sample_type: str,
                   temp: str | None = None) -> float:
    sub = fixed[fixed["sample_type"] == sample_type]
    if temp is not None:
        if temp == "Pooled":
            sub = sub[sub["temp_bin_label"].isna()]
        else:
            sub = sub[sub["temp_bin_label"] == temp]
    if sub.empty:
        return np.nan
    return float(sub["spearman_rho"].iloc[0])


def _load_supplementary_checks_temperature_profiles(method: str, window: int) -> pd.DataFrame | None:
    path = SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["sii_window_days"] == window)]
    return df if not df.empty else None


def _load_supplementary_checks_temperature_surrogates(method: str, window: int) -> pd.DataFrame | None:
    path = SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles_surrogates.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["sii_window"] == f"sii_{window}d")]
    return df if not df.empty else None


def _load_saa_pooled(method: str, window: int) -> dict | None:
    path = SUPPLEMENTARY_CHECKS_RESULTS / "saa_control.csv"
    surr_path = SUPPLEMENTARY_CHECKS_RESULTS / "saa_control_surrogates.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["sii_window_days"] == window) &
            (df["sample_type"] == "control_SAA")]
    if df.empty:
        return None
    rho = float(df["spearman_rho"].iloc[0])
    p_vals = {}
    if surr_path.exists():
        surr = pd.read_csv(surr_path)
        surr = surr[(surr["method"] == method) & (surr["sii_window"] == f"sii_{window}d") &
                    (surr["sample_type"] == "control_SAA")]
        for _, r in surr.iterrows():
            p_vals[r["surrogate_mode"]] = float(r["p_value"])
    return {"rho": rho, "p": p_vals}


def _load_saa_temperature(method: str, window: int) -> pd.DataFrame | None:
    path = SUPPLEMENTARY_CHECKS_RESULTS / "saa_control.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["sii_window_days"] == window) &
            (df["sample_type"] == "temperature_SAA")]
    return df if not df.empty else None


def build_matrix_df(method: str, window: int, cfg: dict) -> pd.DataFrame:
    fixed = load_main_fixed_window_results()
    fixed = fixed[(fixed["method"] == method) & (fixed["sii_window_days"] == window)]

    main_surr = load_main_surrogate_summary()
    main_surr = main_surr[(main_surr["method"] == method) &
                          (main_surr["sii_window"] == f"sii_{window}d")]

    rev_temp = _load_supplementary_checks_temperature_profiles(method, window)
    rev_temp_surr = _load_supplementary_checks_temperature_surrogates(method, window)
    saa_pooled = _load_saa_pooled(method, window)
    saa_temp = _load_saa_temperature(method, window)

    temp_classes = cfg["temperature"]["classes"]
    modes = cfg["surrogates"]["modes"]
    rows = []

    # ── Global pooled ──
    rows.append({
        "scenario": "Global pooled",
        "temperature_class": "Pooled",
        "rho": _load_main_rho(fixed, "pooled_full", "Pooled"),
        **{f"p_{mode}": _load_main_p(main_surr, method, window, "pooled_full", mode, "Pooled") for mode in modes},
    })
    for temp in temp_classes:
        rows.append({
            "scenario": "Global pooled",
            "temperature_class": temp,
            "rho": _load_main_rho(fixed, "temperature", temp),
            **{f"p_{mode}": _load_main_p(main_surr, method, window, "temperature", mode, temp) for mode in modes},
        })

    # ── Main controls: Vegetated, strict low-LAI, Sahara ──
    # Core fixed-window table uses lowercase control sample_type labels.
    main_controls = [s for s in cfg["scenarios"]["main_scenarios"] if s != "pooled_full"]
    for sample_type in main_controls:
        label = SCENARIO_REGISTRY.get(sample_type, {}).get("display_label", sample_type)
        rows.append({
            "scenario": label,
            "temperature_class": "Pooled",
            "rho": _load_main_rho(fixed, sample_type, "Pooled"),
            **{f"p_{mode}": _load_main_p(main_surr, method, window, sample_type, mode, "Pooled") for mode in modes},
        })
        # Temperature profile from S04b if available.
        if rev_temp is not None:
            st = f"temperature_{sample_type}"
            sub = rev_temp[rev_temp["sample_type"] == st]
            for temp in temp_classes:
                trow = sub[sub["temp_bin_label"] == temp]
                if trow.empty:
                    continue
                pvals = {}
                if rev_temp_surr is not None:
                    ss = rev_temp_surr[
                        (rev_temp_surr["sample_type"] == st) &
                        (rev_temp_surr["temp_bin_label"] == temp)
                    ]
                    for _, r in ss.iterrows():
                        pvals[r["surrogate_mode"]] = float(r["p_value"])
                rows.append({
                    "scenario": label,
                    "temperature_class": temp,
                    "rho": float(trow["spearman_rho"].iloc[0]),
                    **{f"p_{mode}": pvals.get(mode, np.nan) for mode in modes},
                })

    # ── SAA ──
    if saa_pooled is not None:
        rows.append({
            "scenario": "SAA",
            "temperature_class": "Pooled",
            "rho": saa_pooled["rho"],
            **{f"p_{mode}": saa_pooled["p"].get(mode, np.nan) for mode in modes},
        })
    if saa_temp is not None:
        for temp in temp_classes:
            trow = saa_temp[saa_temp["temp_bin_label"] == temp]
            if trow.empty:
                continue
            pvals = {}
            if rev_temp_surr is not None:
                ss = rev_temp_surr[
                    (rev_temp_surr["sample_type"] == "temperature_SAA") &
                    (rev_temp_surr["temp_bin_label"] == temp)
                ]
                for _, r in ss.iterrows():
                    pvals[r["surrogate_mode"]] = float(r["p_value"])
            rows.append({
                "scenario": "SAA",
                "temperature_class": temp,
                "rho": float(trow["spearman_rho"].iloc[0]),
                **{f"p_{mode}": pvals.get(mode, np.nan) for mode in modes},
            })

    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, cfg: dict, method: str, window: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    temp_order_raw = ["Pooled"] + cfg["temperature"]["classes"]
    temp_order_labels = [t.replace("_", " ") for t in temp_order_raw]
    main_control_labels = [
        SCENARIO_REGISTRY.get(s, {}).get("display_label", s)
        for s in cfg["scenarios"]["main_scenarios"] if s != "pooled_full"
    ]
    scenarios = ["Global pooled"] + main_control_labels
    if cfg["scenarios"].get("include_saa", True):
        scenarios.append("SAA")
    scenarios = [s for s in scenarios if s in df["scenario"].values]

    modes = cfg["surrogates"]["modes"]
    fig, axes = plt.subplots(
        1, len(modes), figsize=(4.8 * len(modes), 7.0), squeeze=False
    )

    rho_min = df["rho"].min()
    rho_max = df["rho"].max()
    vmax = max(abs(rho_min), abs(rho_max), 0.001)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r

    for ax, mode in zip(axes[0], modes):
        mat = np.full((len(temp_order_raw), len(scenarios)), np.nan)
        pmat = np.full((len(temp_order_raw), len(scenarios)), np.nan)
        for j, scen in enumerate(scenarios):
            sub = df[df["scenario"] == scen]
            for i, temp in enumerate(temp_order_raw):
                row = sub[sub["temperature_class"] == temp]
                if not row.empty:
                    mat[i, j] = row["rho"].iloc[0]
                    pmat[i, j] = row[f"p_{mode}"].iloc[0]

        # Build an RGBA image: colour encodes rho, alpha encodes empirical p.
        rgba = cmap(norm(mat))
        for i in range(len(temp_order_raw)):
            for j in range(len(scenarios)):
                if np.isnan(mat[i, j]):
                    rgba[i, j, 3] = 0.0
                    continue
                p = pmat[i, j]
                alpha = 0.5 if np.isnan(p) else max(0.25, 1.0 - p)
                rgba[i, j, 3] = alpha

        ax.imshow(rgba, aspect="auto")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(temp_order_raw)))
        ax.set_yticklabels(temp_order_labels, fontsize=9)
        ax.set_title(mode, fontsize=11)
        ax.tick_params(axis="both", length=0)

        for i in range(len(temp_order_raw)):
            for j in range(len(scenarios)):
                if np.isnan(mat[i, j]):
                    continue
                text = f"{mat[i, j]:.3f}"
                color = "white" if abs(mat[i, j]) > vmax * 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color,
                        fontsize=8)

    # Single shared colorbar for all panels; reserve right margin manually
    # because tight_layout does not handle an externally-added colorbar axis.
    fig.subplots_adjust(left=0.10, right=0.84, top=0.90, bottom=0.12)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.018, 0.70])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        label="Spearman rho",
    )

    fig.suptitle(
        f"SIF 771 nm association matrix ({method}, {window} days)", y=0.96
    )
    out_png = SUPPLEMENTARY_CHECKS_FIGURES / cfg["outputs"]["graphical_abstract_fig"]
    savefig(out_png, fig)
    print(f"  Figure: {out_png}")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    print("=" * 64)
    print("MAGNETO supplementary checks — Graphical abstract matrix")
    print("=" * 64)

    df = build_matrix_df(args.method, args.window, cfg)
    if df.empty:
        print("[WARN] No matrix data available")
        return 0

    out_csv = SUPPLEMENTARY_CHECKS_RESULTS / "graphical_abstract_matrix.csv"
    atomic_write(df, out_csv)
    print(f"  Written: {out_csv}")

    if args.figure:
        make_figure(df, cfg, args.method, args.window)

    return 0


if __name__ == "__main__":
    sys.exit(main())
