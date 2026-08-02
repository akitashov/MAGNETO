#!/usr/bin/env python3
"""Build manuscript-ready numerical summaries from validated Stage B outputs.

Reads current results and writes:
    results/reporting/MANUSCRIPT_NUMBERS_FINAL.md
    results/reporting/manuscript_numbers_final.csv

Does not compute new statistics; it only extracts and formats values already
produced by the approved analytical stages.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "reporting"
OUT_MD = OUT_DIR / "MANUSCRIPT_NUMBERS_FINAL.md"
OUT_CSV = OUT_DIR / "manuscript_numbers_final.csv"

RUN_ID = "20260730T061934Z"
COMMIT = "de1499e4749a3caea562453078d4e88447b791e1"


def load(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def fmt_num(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{int(x):,}"


def fmt_ci(low, high, decimals: int = 3) -> str:
    return f"[{fmt_num(low, decimals)}, {fmt_num(high, decimals)}]"


def fmt_p(p, empirical: bool = True) -> str:
    if pd.isna(p):
        return "NA"
    # Use exact value; floor at 0.001 only if requested empirical B=1000.
    if empirical and p < 0.001:
        return "0.001"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    md_lines: list[str] = [
        "# MANUSCRIPT NUMBERS FINAL",
        "",
        f"- **analysis_run_id**: `{RUN_ID}`",
        f"- **Git commit**: `{COMMIT}`",
        f"- **Generated**: {datetime.now(timezone.utc).isoformat()}Z",
        "",
        "## Primary Table 1 estimates",
        "",
    ]

    # Table 1
    t1 = load(PROJECT_ROOT / "reports" / "tables" / "table1_primary_results_source.csv", thousands=",")
    for _, row in t1.iterrows():
        n_obs = int(row["n observations"])
        n_cells = int(row["n cells"])
        label = f"{row['Sample']} | {row['Detrending method']} | {int(row['SII window (days)'])} d"
        records.append({
            "section": "table1",
            "label": label,
            "rho": row["Spearman ρ [95% CI]"],
            "effect_per_100_nT": row["Effect per 100 nT"],
            "n_obs": n_obs,
            "n_cells": n_cells,
            "max_empirical_p": row["Maximum empirical p"],
            "source_file": "reports/tables/table1_primary_results_source.csv",
        })
        md_lines.append(
            f"- **{label}**: rho = {row['Spearman ρ [95% CI]']}, "
            f"effect/100 nT = {row['Effect per 100 nT']}, "
            f"n = {n_obs:,} obs / {n_cells:,} cells, "
            f"max empirical p = {row['Maximum empirical p']}"
        )

    md_lines.extend(["", "## Bootstrap 95% CI (clustered by 0.5° cell)", ""])
    boot = load(PROJECT_ROOT / "results" / "supplementary_checks" / "table1_bootstrap_summary.csv")
    for _, row in boot.iterrows():
        label = f"{row['sample_label']} | {row['method']} | {int(row['sii_window_days'])} d"
        records.append({
            "section": "bootstrap_ci",
            "label": label,
            "rho_obs": fmt_num(row["observed_rho"]),
            "ci_95": fmt_ci(row["bootstrap_ci_low"], row["bootstrap_ci_high"]),
            "n_boot": int(row["n_bootstrap"]),
            "n_obs": int(row["n_observations"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/table1_bootstrap_summary.csv",
        })
        md_lines.append(
            f"- **{label}**: rho = {fmt_num(row['observed_rho'])}, "
            f"CI = {fmt_ci(row['bootstrap_ci_low'], row['bootstrap_ci_high'])}, "
            f"B = {int(row['n_bootstrap'])}, n = {int(row['n_observations']):,}"
        )

    md_lines.extend(["", "## Temporal-surrogate p-values (pooled harmonic 28 d)", ""])
    surr = load(PROJECT_ROOT / "results" / "surrogate_summary.csv")
    surr["sii_window"] = surr["sii_window"].astype(str)
    surr_sub = surr[
        (surr["sample_type"] == "pooled_full")
        & (surr["method"] == "harmonic")
        & (surr["sii_window"] == "sii_28d")
    ]
    max_p = None
    for _, row in surr_sub.iterrows():
        mode = row["surrogate_mode"]
        p = row["p_value"]
        records.append({
            "section": "surrogate_p",
            "label": f"pooled harmonic 28 d | {mode}",
            "rho_obs": fmt_num(row["rho_obs"]),
            "p_value": fmt_p(p, empirical=True),
            "n_surr": int(row["n_completed"]),
            "source_file": "results/surrogate_summary.csv",
        })
        md_lines.append(f"- **{mode}**: p = {fmt_p(p, empirical=True)} (rho = {fmt_num(row['rho_obs'])}, B = {int(row['n_completed'])})")
        if max_p is None or p > max_p:
            max_p = p
    if max_p is not None:
        records.append({
            "section": "surrogate_p",
            "label": "pooled harmonic 28 d | maximum empirical p",
            "p_value": fmt_p(max_p, empirical=True),
            "source_file": "results/surrogate_summary.csv",
        })
        md_lines.append(f"- **Maximum empirical p**: {fmt_p(max_p, empirical=True)}")

    md_lines.extend(["", "## Temperature-profile key results (harmonic 28 d, pooled sample)", ""])
    temp = load(PROJECT_ROOT / "results" / "supplementary_checks" / "temperature_scenarios_harmonic.csv")
    temp_sub = temp[(temp["window_days"] == 28) & (temp["scenario"] == "pooled")]
    for _, row in temp_sub.iterrows():
        records.append({
            "section": "temperature_profile",
            "label": f"pooled 28 d | {row['temp_class']}",
            "rho": fmt_num(row["rho"]),
            "ci_95": fmt_ci(row["ci_low"], row["ci_high"]),
            "n_obs": int(row["n_obs"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/temperature_scenarios_harmonic.csv",
        })
        md_lines.append(
            f"- **{row['temp_class']}**: rho = {fmt_num(row['rho'])}, "
            f"CI = {fmt_ci(row['ci_low'], row['ci_high'])}, "
            f"n = {int(row['n_obs']):,} / {int(row['n_cells']):,} cells"
        )

    md_lines.extend(["", "## Shapley/LMG R² decomposition (28 d, full sample)", ""])
    shap = load(PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_by_window.csv")
    shap_sub = shap[(shap["window_days"] == 28) & (shap["scenario"] == "full_sample")]
    for _, row in shap_sub.iterrows():
        records.append({
            "section": "shapley_28d_full_sample",
            "label": f"full sample 28 d | {row['temp_bin_label']}",
            "sii_r2_percent": fmt_num(row["sii_shapley_percent"], 4),
            "par_r2_percent": fmt_num(row["par_shapley_percent"], 4),
            "vpd_r2_percent": fmt_num(row["vpd_shapley_percent"], 4),
            "full_r2_percent": fmt_num(row["r2_full_percent"], 4),
            "n_obs": int(row["n_obs"]) if pd.notna(row["n_obs"]) else None,
            "n_cells": int(row["n_cells"]) if pd.notna(row["n_cells"]) else None,
            "source_file": "results/supplementary_checks/driver_r2_shapley_by_window.csv",
        })
        md_lines.append(
            f"- **{row['temp_bin_label']}**: SII = {fmt_num(row['sii_shapley_percent'], 4)}%, "
            f"PAR = {fmt_num(row['par_shapley_percent'], 4)}%, "
            f"VPD = {fmt_num(row['vpd_shapley_percent'], 4)}%; "
            f"full R² = {fmt_num(row['r2_full_percent'], 4)}%"
        )

    md_lines.extend(["", "## Land-cover class estimates (harmonic 28 d)", ""])
    lc = load(PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis.csv")
    lc_sub = lc[
        (lc["method"] == "harmonic")
        & (lc["sii_window"] == "sii_28d")
        & (~lc["sample_type"].str.contains("_temp_", na=False))
    ].copy()
    for _, row in lc_sub.iterrows():
        lc_label = row["land_cover_class"]
        records.append({
            "section": "landcover",
            "label": f"{lc_label} | harmonic 28 d",
            "rho": fmt_num(row["spearman_rho"]),
            "p_value": fmt_p(row["spearman_p"], empirical=False),
            "n_obs": int(row["n_obs"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/landcover_analysis.csv",
        })
        md_lines.append(
            f"- **{lc_label}**: rho = {fmt_num(row['spearman_rho'])}, "
            f"p = {fmt_p(row['spearman_p'], empirical=False)}, "
            f"n = {int(row['n_obs']):,} / {int(row['n_cells']):,} cells"
        )

    md_lines.extend(["", "## SAA control (harmonic 28 d)", ""])
    saa = load(PROJECT_ROOT / "results" / "supplementary_checks" / "saa_control.csv")
    saa_sub = saa[(saa["method"] == "harmonic") & (saa["sii_window"] == "sii_28d")]
    for _, row in saa_sub.iterrows():
        records.append({
            "section": "saa_control",
            "label": row["sample_type"],
            "rho": fmt_num(row["spearman_rho"]),
            "p_value": fmt_p(row["spearman_p"], empirical=False),
            "n_obs": int(row["n_obs"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/saa_control.csv",
        })
        md_lines.append(
            f"- **{row['sample_type']}**: rho = {fmt_num(row['spearman_rho'])}, "
            f"p = {fmt_p(row['spearman_p'], empirical=False)}, "
            f"n = {int(row['n_obs']):,} / {int(row['n_cells']):,} cells"
        )

    md_lines.extend(["", "## Kp and F10.7 sensitivity (pooled harmonic 28 d)", ""])
    kp = load(PROJECT_ROOT / "results" / "supplementary_checks" / "kp_comparison.csv")
    kp_sub = kp[(kp["sample_type"] == "pooled_kp") & (kp["method"] == "harmonic") & (kp["sii_window"] == "kp_28d")]
    for _, row in kp_sub.iterrows():
        records.append({
            "section": "kp_f107",
            "label": "Kp 28 d exposure | pooled harmonic",
            "rho": fmt_num(row["spearman_rho"]),
            "p_value": fmt_p(row["spearman_p"], empirical=False),
            "ols_slope_per_100_nT": fmt_num(row["ols_slope_per_100nT"], 4),
            "n_obs": int(row["n_obs"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/kp_comparison.csv",
        })
        md_lines.append(
            f"- **Kp 28 d exposure**: rho = {fmt_num(row['spearman_rho'])}, "
            f"p = {fmt_p(row['spearman_p'], empirical=False)}, "
            f"OLS/100 nT = {fmt_num(row['ols_slope_per_100nT'], 4)}, "
            f"n = {int(row['n_obs']):,} / {int(row['n_cells']):,} cells"
        )

    f107 = load(PROJECT_ROOT / "results" / "supplementary_checks" / "f107_comparison.csv")
    f107_sub = f107[
        (f107["sample_type"] == "pooled_f10_7")
        & (f107["method"] == "harmonic")
        & (f107["sii_window"] == "f10_7_28d")
    ]
    for _, row in f107_sub.iterrows():
        records.append({
            "section": "kp_f107",
            "label": "F10.7 28 d exposure | pooled harmonic",
            "rho": fmt_num(row["spearman_rho"]),
            "p_value": fmt_p(row["spearman_p"], empirical=False),
            "ols_slope_per_100_nT": fmt_num(row["ols_slope_per_100nT"], 6),
            "n_obs": int(row["n_obs"]),
            "n_cells": int(row["n_cells"]),
            "source_file": "results/supplementary_checks/f107_comparison.csv",
        })
        md_lines.append(
            f"- **F10.7 28 d exposure**: rho = {fmt_num(row['spearman_rho'])}, "
            f"p = {fmt_p(row['spearman_p'], empirical=False)}, "
            f"OLS/100 nT = {fmt_num(row['ols_slope_per_100nT'], 6)}, "
            f"n = {int(row['n_obs']):,} / {int(row['n_cells']):,} cells"
        )

    # Wavelength sensitivity placeholder
    wave_path = PROJECT_ROOT / "results" / "supplementary_outcomes_results.csv"
    if wave_path.exists():
        wave = load(wave_path)
        wave_sub = wave[
            (wave["sample_type"] == "outcome_specific_full")
            & (wave["outcome"].isin(["sif_757nm", "sif_740nm"]))
        ]
        md_lines.extend(["", "## SIF wavelength sensitivity (outcome-specific full sample)", ""])
        for _, row in wave_sub.iterrows():
            records.append({
                "section": "wavelength",
                "label": f"{row['outcome']} | {row['method']} | {int(row['sii_window_days'])} d",
                "rho": fmt_num(row["spearman_rho"]),
                "effect_per_1sd_sii": fmt_num(row.get("std_effect_per_1sd_sii"), 5),
                "n_obs": int(row["n_obs"]),
                "n_cells": int(row["n_cells"]),
                "source_file": "results/supplementary_outcomes_results.csv",
            })
            md_lines.append(
                f"- **{row['outcome']} {row['method']} {int(row['sii_window_days'])} d**: "
                f"rho = {fmt_num(row['spearman_rho'])}, n = {int(row['n_obs']):,}"
            )
    else:
        md_lines.extend([
            "",
            "## SIF wavelength sensitivity",
            "",
            "_Pending: run_supplementary_outcomes.py not yet complete._",
        ])

    md_lines.extend(["", "## Source-output checksums", ""])
    for src in sorted(set(r["source_file"] for r in records)):
        md_lines.append(f"- `{src}`")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_MD} ({len(df)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
