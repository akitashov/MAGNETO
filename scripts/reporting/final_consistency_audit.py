#!/usr/bin/env python3
"""Final numerical consistency audit for Stage B outputs.

Reads the final set of CSV, Markdown, and table outputs and checks that:
- every file carries the approved analysis_run_id and Git commit;
- primary numerical values (sample sizes, rho, CI, p-values) are internally
  consistent across Table 1, bootstrap summary, surrogate summary, and figures;
- no stale pre-grid-fix numbers or forbidden terminology remain in final
  manuscript-ready outputs;
- empirical p-values respect the plus-one convention for B = 1000.

Outputs:
    results/audits/FINAL_NUMERICAL_CONSISTENCY_REPORT.md
    results/audits/final_numerical_consistency_report.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _Common import PROJECT_ROOT, atomic_write  # noqa: E402

RUN_ID = "20260730T061934Z"
COMMIT = "de1499e4749a3caea562453078d4e88447b791e1"
PRIMARY_WINDOW = 28
SECONDARY_WINDOW = 21
GRID_LABEL = "0.5° × 0.5°"
PERIOD_LABEL = "September 2014–December 2024"
PRIMARY_TARGET = "sif_771nm"

OUT_DIR = PROJECT_ROOT / "results" / "audits"
OUT_MD = OUT_DIR / "FINAL_NUMERICAL_CONSISTENCY_REPORT.md"
OUT_JSON = OUT_DIR / "final_numerical_consistency_report.json"

# These phrases must not appear in final manuscript-facing outputs.
FORBIDDEN_PHRASES = [
    "Storm Index Integral",
    "cumulative sum",
    "cumulative mean",
    "28-day integral",
    "MCD15A2H",
    "MOD08",
    "MYD08",
]

OLD_SAMPLE_SIZES = ["456,254", "455,764", "456254", "455764"]

# Files that intentionally mention old values or terminology (change checklists,
# historical audit reports, the audit report itself) are excluded from the
# forbidden/stale scan.
EXCLUDED_FROM_TEXT_SCAN = {
    PROJECT_ROOT / "results" / "reporting" / "MANUSCRIPT_CHANGE_CHECKLIST.md",
    PROJECT_ROOT / "results" / "audits" / "STAGE_A_CLEAN_RUN_REPORT.md",
    OUT_MD,
    OUT_JSON,
}

CRITICAL = "CRITICAL"
MAJOR = "MAJOR"
MINOR = "MINOR"


def issue(level: str, category: str, message: str, file: str | None = None) -> dict:
    return {"level": level, "category": category, "message": message, "file": file}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _parse_rho_ci(cell: str) -> tuple[float | None, float | None, float | None]:
    """Parse strings like '−0.051 [−0.053, −0.050]' into (rho, low, high)."""
    if pd.isna(cell):
        return None, None, None
    cell = str(cell).replace("−", "-")
    m = re.match(r"\s*([-0-9.]+)\s*\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\]", cell)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def check_run_id_and_commit(issues_list: list[dict]) -> None:
    """Check that key metadata-bearing files reference the approved run."""
    files_to_check = [
        PROJECT_ROOT / "results" / "reporting" / "MANUSCRIPT_NUMBERS_FINAL.md",
        PROJECT_ROOT / "results" / "reporting" / "METHODS_CODE_FACTS_FINAL.md",
        PROJECT_ROOT / "results" / "reporting" / "MANUSCRIPT_CHANGE_CHECKLIST.md",
        PROJECT_ROOT / "results" / "audits" / "STAGE_B_LIVE_STATUS.md",
    ]
    for path in files_to_check:
        text = _read_text(path)
        if not text:
            issues_list.append(issue(MINOR, "metadata", f"File not found or empty: {path}", str(path)))
            continue
        if RUN_ID not in text:
            issues_list.append(issue(CRITICAL, "metadata", f"Missing analysis_run_id {RUN_ID} in {path}", str(path)))
        if COMMIT not in text:
            issues_list.append(issue(CRITICAL, "metadata", f"Missing Git commit {COMMIT} in {path}", str(path)))


def check_forbidden_and_stale_text(issues_list: list[dict]) -> None:
    """Scan final text outputs for forbidden phrases and stale sample sizes."""
    scan_paths = list((PROJECT_ROOT / "results" / "reporting").glob("*.md"))
    scan_paths += list((PROJECT_ROOT / "results" / "audits").glob("*.md"))
    scan_paths += list((PROJECT_ROOT / "reports" / "figures").glob("*_description.md"))
    scan_paths += list((PROJECT_ROOT / "reports" / "tables").glob("*.md"))

    for path in scan_paths:
        if path in EXCLUDED_FROM_TEXT_SCAN:
            continue
        text = _read_text(path)
        if not text:
            continue
        lower = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase.lower() in lower:
                issues_list.append(issue(MAJOR, "terminology", f"Forbidden phrase '{phrase}' found in {path}", str(path)))
        for old in OLD_SAMPLE_SIZES:
            if old in text:
                issues_list.append(issue(MAJOR, "stale_numbers", f"Stale sample size '{old}' found in {path}", str(path)))


def check_table_1_consistency(issues_list: list[dict]) -> dict:
    """Verify Table 1 against bootstrap source."""
    result: dict = {"table1_rows": 0, "bootstrap_rows": 0, "consistent": False}
    t1 = _read_csv(PROJECT_ROOT / "reports" / "tables" / "table1_primary_results_source.csv")
    if t1 is None:
        issues_list.append(issue(CRITICAL, "table1", "Table 1 source CSV not found"))
        return result
    result["table1_rows"] = len(t1)

    required = {"Sample", "Detrending method", "SII window (days)", "n cells", "n observations", "Spearman ρ [95% CI]", "Maximum empirical p"}
    missing = required - set(t1.columns)
    if missing:
        issues_list.append(issue(CRITICAL, "table1", f"Table 1 missing columns: {sorted(missing)}"))
        return result

    bootstrap = _read_csv(PROJECT_ROOT / "results" / "supplementary_checks" / "table1_bootstrap_summary.csv")
    if bootstrap is None:
        issues_list.append(issue(CRITICAL, "bootstrap", "Bootstrap summary CSV not found"))
        return result
    result["bootstrap_rows"] = len(bootstrap)

    def _map_sample_type(label: str) -> str:
        ll = label.lower()
        if "pairwise" in ll:
            return "pairwise_matched"
        if "vegetated" in ll:
            return "control_vegetated"
        if "sahara" in ll:
            return "control_sahara"
        if "strict" in ll and "lai" in ll:
            return "control_strict_low_lai"
        return "pooled_full"

    def _map_method(method: str) -> str:
        return method.lower().replace(" ", "_")

    for _, row in t1.iterrows():
        label = str(row.get("Sample", ""))
        method = str(row.get("Detrending method", ""))
        window = int(row.get("SII window (days)")) if not pd.isna(row.get("SII window (days)")) else None
        rho, ci_low, ci_high = _parse_rho_ci(row.get("Spearman ρ [95% CI]"))
        n_obs = row.get("n observations")
        n_cells = row.get("n cells")

        if rho is None or ci_low is None or ci_high is None:
            issues_list.append(issue(CRITICAL, "table1", f"Could not parse rho/CI for row '{label}'"))
            continue
        if ci_low > ci_high:
            issues_list.append(issue(CRITICAL, "table1", f"Inverted CI for '{label}': [{ci_low}, {ci_high}]"))
        if rho < -1 or rho > 1:
            issues_list.append(issue(CRITICAL, "table1", f"rho out of range for '{label}': {rho}"))

        # Match bootstrap by sample_type and method/window (case-insensitive
        # because some bootstrap summaries use control_Sahara vs control_sahara).
        sample_type = _map_sample_type(label)
        method_norm = _map_method(method)
        b_match = bootstrap[
            (bootstrap["sample_type"].str.lower() == sample_type.lower())
            & (bootstrap["method"].str.lower() == method_norm.lower())
            & (bootstrap["sii_window_days"] == window)
        ]
        if b_match.empty:
            issues_list.append(issue(MINOR, "table1", f"No bootstrap match for '{label}' {method} {window}d"))
            continue
        br = b_match.iloc[0]
        # Table 1 values are rounded to 3 decimals; allow a tolerance that
        # accounts for rounding (0.0005) plus a small safety margin.
        tol = 0.0015
        if not (pd.isna(br.get("observed_rho")) or abs(float(br["observed_rho"]) - rho) < tol):
            issues_list.append(issue(MAJOR, "table1", f"Table 1 rho {rho} differs from bootstrap {br['observed_rho']} for '{label}'"))
        if not (pd.isna(br.get("bootstrap_ci_low")) or abs(float(br["bootstrap_ci_low"]) - ci_low) < tol):
            issues_list.append(issue(MAJOR, "table1", f"Table 1 CI low {ci_low} differs from bootstrap {br['bootstrap_ci_low']} for '{label}'"))
        if not (pd.isna(br.get("bootstrap_ci_high")) or abs(float(br["bootstrap_ci_high"]) - ci_high) < tol):
            issues_list.append(issue(MAJOR, "table1", f"Table 1 CI high {ci_high} differs from bootstrap {br['bootstrap_ci_high']} for '{label}'"))

    # Pooled harmonic sign/range stop condition.
    pooled = t1[(t1["Sample"].str.contains("Full sample", case=False, na=False)) & (~t1["Sample"].str.contains("pairwise", case=False, na=False))]
    harmonic28 = pooled[(pooled["Detrending method"].str.contains("Harmonic", case=False, na=False)) & (pooled["SII window (days)"] == PRIMARY_WINDOW)]
    if not harmonic28.empty:
        r, _, _ = _parse_rho_ci(harmonic28.iloc[0]["Spearman ρ [95% CI]"])
        if r is not None:
            if r > 0:
                issues_list.append(issue(CRITICAL, "stop_condition", f"Pooled harmonic 28 d rho is positive: {r}"))
            if not (-0.070 <= r <= -0.035):
                issues_list.append(issue(CRITICAL, "stop_condition", f"Pooled harmonic 28 d rho {r} outside expected range [-0.070, -0.035]"))

    result["consistent"] = not any(i["level"] in (CRITICAL, MAJOR) and i["category"] in ("table1", "bootstrap") for i in issues_list)
    return result


def check_surrogate_p_values(issues_list: list[dict]) -> dict:
    """Check surrogate summary for plus-one p-values and consistency."""
    result: dict = {"surrogate_modes": [], "observed_rho": None, "max_p": None}
    surr = _read_csv(PROJECT_ROOT / "results" / "surrogate_summary.csv")
    if surr is None:
        issues_list.append(issue(CRITICAL, "surrogates", "surrogate_summary.csv not found"))
        return result

    modes = set()
    for _, row in surr.iterrows():
        mode = row.get("surrogate_mode")
        p = row.get("p_value")
        observed = row.get("rho_obs")
        if pd.isna(mode):
            continue
        modes.add(str(mode))
        sample_type = row.get("sample_type")
        method = row.get("method")
        window = row.get("sii_window")
        if sample_type == "pooled_full" and method == "harmonic" and window == f"sii_{PRIMARY_WINDOW}d" and not pd.isna(observed):
            result["observed_rho"] = float(observed)
        if not pd.isna(p):
            p = float(p)
            if p < 0 or p > 1:
                issues_list.append(issue(CRITICAL, "surrogates", f"p-value out of range for {mode}: {p}"))
            # Minimum possible plus-one p for B=1000 is 1/(1000+1) ≈ 0.000999.
            if 0 < p < 0.000999:
                issues_list.append(issue(MAJOR, "surrogates", f"p-value {p} for {mode} is below minimum possible plus-one p (B=1000)"))
            if p == 0:
                issues_list.append(issue(MAJOR, "surrogates", f"Exact zero p-value reported for {mode}; should be at least 0.001 with B=1000"))

    result["surrogate_modes"] = sorted(modes)
    # Maximum empirical p over pooled harmonic 28 d modes.
    if modes:
        sub = surr[(surr["sample_type"] == "pooled_full") & (surr["method"] == "harmonic") & (surr["sii_window"] == f"sii_{PRIMARY_WINDOW}d")]
        if not sub.empty:
            result["max_p"] = float(sub["p_value"].max())
    return result


def check_figures_exist(issues_list: list[dict]) -> dict:
    """Check that all final figures and source CSVs exist and are non-empty."""
    result: dict = {"figures": [], "missing": []}
    expected = {
        "figure1_sii_response_surrogates": ["png", "pdf", "_source.csv"],
        "figure2_temperature_geographic_profiles": ["png", "pdf", "_source.csv"],
        "figure3_driver_r2_shapley": ["png", "pdf", "_source.csv"],
        "figure4_landcover_vertical": ["png", "pdf", "_source.csv"],
        "figureS1_spatial_coverage_controls": ["png", "pdf", "_source.csv"],
        "figureS2_window_scan_771": ["png", "pdf", "_source.csv"],
        "figureS3_surrogate_diagnostics": ["png", "pdf", "_source.csv"],
        "figureS4_temperature_control_profiles": ["png", "pdf", "_source.csv"],
        "figureS5_kp_f107_comparison": ["png", "pdf", "_source.csv"],
        "figureS6_environmental_driver_stability": ["png", "pdf", "_source.csv"],
        "figureS7_shapley_extended": ["png", "pdf", "_source.csv"],
        "figureS8_landcover_lai_composition": ["png", "pdf", "_source.csv"],
    }
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    for stem, exts in expected.items():
        for ext in exts:
            path = fig_dir / f"{stem}{ext}" if ext.startswith("_") else fig_dir / f"{stem}.{ext}"
            if path.exists() and path.stat().st_size > 0:
                result["figures"].append(str(path.relative_to(PROJECT_ROOT)))
            else:
                result["missing"].append(str(path.relative_to(PROJECT_ROOT)))
                level = CRITICAL if stem.startswith("figure1") or stem.startswith("figure2") or stem.startswith("figure3") or stem.startswith("figure4") else MAJOR
                issues_list.append(issue(level, "figures", f"Missing or empty figure file: {path}", str(path)))
    return result


def check_main_figures_metadata(issues_list: list[dict]) -> None:
    """Ensure main figure descriptions cite the approved run and commit."""
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    for stem in ["figure1_sii_response_surrogates", "figure2_temperature_geographic_profiles", "figure3_driver_r2_shapley", "figure4_landcover_vertical"]:
        desc = fig_dir / f"{stem}_description.md"
        text = _read_text(desc)
        if not text:
            issues_list.append(issue(MAJOR, "figure_metadata", f"Figure description missing: {desc}", str(desc)))
            continue
        if RUN_ID not in text:
            issues_list.append(issue(MAJOR, "figure_metadata", f"Figure {stem} description missing run_id", str(desc)))
        if COMMIT not in text:
            issues_list.append(issue(MAJOR, "figure_metadata", f"Figure {stem} description missing commit", str(desc)))


def check_environmental_driver_outputs(issues_list: list[dict]) -> None:
    """Check driver matrix, summary, and Shapley outputs."""
    matrix = PROJECT_ROOT / "results" / "environmental_driver_matrix.parquet"
    if not matrix.exists():
        issues_list.append(issue(CRITICAL, "driver_matrix", "environmental_driver_matrix.parquet not found"))
    summary = PROJECT_ROOT / "results" / "environmental_driver_matrix_summary.csv"
    if not summary.exists():
        issues_list.append(issue(CRITICAL, "driver_matrix", "environmental_driver_matrix_summary.csv not found"))
    shapley = PROJECT_ROOT / "results" / "supplementary_checks" / "driver_r2_shapley_by_window.csv"
    if not shapley.exists():
        issues_list.append(issue(CRITICAL, "shapley", "driver_r2_shapley_by_window.csv not found"))
    else:
        df = _read_csv(shapley)
        if df is not None and "window_days" in df.columns:
            if (df["window_days"] == PRIMARY_WINDOW).sum() == 0:
                issues_list.append(issue(CRITICAL, "shapley", f"No Shapley rows for primary window {PRIMARY_WINDOW}"))


def check_supplementary_outputs(issues_list: list[dict]) -> None:
    """Check supplementary checks and tables."""
    files = [
        PROJECT_ROOT / "results" / "supplementary_checks" / "temperature_scenarios_harmonic.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "temperature_scenarios_spline.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_analysis.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "kp_comparison.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "f107_comparison.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "saa_control.csv",
        PROJECT_ROOT / "results" / "supplementary_checks" / "landcover_temperature_inference.csv",
    ]
    for path in files:
        if not path.exists():
            issues_list.append(issue(MAJOR, "supplementary", f"Supplementary output missing: {path}", str(path)))


def check_table_s10(issues_list: list[dict]) -> None:
    """Check Table S10 outputs if available."""
    path = PROJECT_ROOT / "reports" / "tables" / "tableS10_sif_wavelength.md"
    if not path.exists():
        issues_list.append(issue(MINOR, "table_s10", "Table S10 not yet built"))


def main() -> int:
    print("=" * 64)
    print("MAGNETO — Final numerical consistency audit")
    print("=" * 64)

    issues_list: list[dict] = []

    check_run_id_and_commit(issues_list)
    check_forbidden_and_stale_text(issues_list)
    table1_info = check_table_1_consistency(issues_list)
    surrogate_info = check_surrogate_p_values(issues_list)
    fig_info = check_figures_exist(issues_list)
    check_main_figures_metadata(issues_list)
    check_environmental_driver_outputs(issues_list)
    check_supplementary_outputs(issues_list)
    check_table_s10(issues_list)

    counts = {CRITICAL: 0, MAJOR: 0, MINOR: 0}
    for i in issues_list:
        counts[i["level"]] += 1

    summary = {
        "stage": "final_consistency_audit.py",
        "analysis_run_id": RUN_ID,
        "git_commit": COMMIT,
        "primary_window": PRIMARY_WINDOW,
        "secondary_window": SECONDARY_WINDOW,
        "grid": GRID_LABEL,
        "period": PERIOD_LABEL,
        "primary_target": PRIMARY_TARGET,
        "counts": counts,
        "table1": table1_info,
        "surrogates": surrogate_info,
        "figures": fig_info,
        "issues": issues_list,
        "passed": counts[CRITICAL] == 0 and counts[MAJOR] == 0,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(summary, OUT_JSON, fmt="json")

    md_lines = [
        "# FINAL NUMERICAL CONSISTENCY REPORT",
        "",
        f"- **analysis_run_id**: `{RUN_ID}`",
        f"- **Git commit**: `{COMMIT}`",
        f"- **primary window**: {PRIMARY_WINDOW} days",
        f"- **sensitivity window**: {SECONDARY_WINDOW} days",
        f"- **spatial grid**: {GRID_LABEL}",
        f"- **study period**: {PERIOD_LABEL}",
        f"- **primary target**: {PRIMARY_TARGET}",
        "",
        "## Summary",
        "",
        f"- CRITICAL: {counts[CRITICAL]}",
        f"- MAJOR: {counts[MAJOR]}",
        f"- MINOR: {counts[MINOR]}",
        f"- Passed: {'YES' if summary['passed'] else 'NO'}",
        "",
        f"- Table 1 rows: {table1_info['table1_rows']}",
        f"- Bootstrap rows: {table1_info['bootstrap_rows']}",
        f"- Surrogate modes: {', '.join(surrogate_info['surrogate_modes']) or 'N/A'}",
        f"- Observed pooled rho: {surrogate_info['observed_rho']}",
        f"- Max empirical p: {surrogate_info['max_p']}",
        f"- Figures present: {len(fig_info['figures'])}",
        f"- Figures missing: {len(fig_info['missing'])}",
        "",
        "## Issues",
        "",
    ]
    if not issues_list:
        md_lines.append("No consistency issues found.")
    else:
        for level in [CRITICAL, MAJOR, MINOR]:
            level_issues = [i for i in issues_list if i["level"] == level]
            if not level_issues:
                continue
            md_lines.append(f"### {level}")
            md_lines.append("")
            for i in level_issues:
                file_str = f" (`{i['file']}`)" if i.get("file") else ""
                md_lines.append(f"- **{i['category']}**: {i['message']}{file_str}")
            md_lines.append("")

    md_lines += [
        "",
        "## Stop-condition checks",
        "",
        f"- Pooled harmonic 28 d rho sign negative: {not any('positive' in i['message'] for i in issues_list if i['category'] == 'stop_condition')}",
        f"- Pooled harmonic 28 d rho within [-0.070, -0.035]: {not any('outside expected range' in i['message'] for i in issues_list if i['category'] == 'stop_condition')}",
        "",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Written: {OUT_JSON}")
    print(f"[OK] Written: {OUT_MD}")
    print(f"Audit result: {'PASSED' if summary['passed'] else 'ISSUES FOUND'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
