#!/usr/bin/env python3
"""Post-run audit for a clean Stage A downstream re-calculation.

Compares newly produced Stage A outputs against an archived baseline,
verifies run_id provenance, and reports key inferential summaries.

Outputs:
    results/audits/stage_a_clean_run_audit.json
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = RESULTS_DIR / "audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_DIR = PROJECT_ROOT / "archive" / "stage_a_post_grid_fix"

RUN_METADATA_PATH = RESULTS_DIR / "run_metadata.json"

OUTPUT_FILES: dict[str, Path] = {
    "global_qc": PROJECT_ROOT / "data" / "interim" / "global_qc.parquet",
    "harmonic_analysis": PROJECT_ROOT / "data" / "processed" / "harmonic_analysis.parquet",
    "cyclic_spline_analysis": PROJECT_ROOT / "data" / "processed" / "cyclic_spline_analysis.parquet",
    "harmonic_spline_matched": PROJECT_ROOT / "data" / "processed" / "harmonic_spline_matched.parquet",
    "fixed_window_results": PROJECT_ROOT / "results" / "fixed_window_results.csv",
    "lai_cell_summary": PROJECT_ROOT / "data" / "interim" / "lai_cell_summary.parquet",
    "control_membership": PROJECT_ROOT / "data" / "interim" / "control_membership.parquet",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def key_stats(df: pd.DataFrame, key_cols: list[str]) -> dict[str, Any]:
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        return {"error": f"missing key columns: {missing}"}
    keys = df[key_cols].drop_duplicates()
    return {
        "n_rows": int(len(df)),
        "n_unique_keys": int(len(keys)),
        "key_columns": key_cols,
    }


def residual_stats(df: pd.DataFrame, col: str = "residual") -> dict[str, Any]:
    if col not in df.columns:
        return {"error": f"column {col!r} not found"}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
    }


def compare_keys(current: pd.DataFrame, baseline: pd.DataFrame, key_cols: list[str]) -> dict[str, Any]:
    cur_keys = set(map(tuple, current[key_cols].drop_duplicates().values.tolist()))
    base_keys = set(map(tuple, baseline[key_cols].drop_duplicates().values.tolist()))
    overlap = cur_keys & base_keys
    added = cur_keys - base_keys
    removed = base_keys - cur_keys
    return {
        "current_n": len(cur_keys),
        "baseline_n": len(base_keys),
        "overlap_n": len(overlap),
        "added_n": len(added),
        "removed_n": len(removed),
        "overlap_frac_current": len(overlap) / len(cur_keys) if cur_keys else None,
        "overlap_frac_baseline": len(overlap) / len(base_keys) if base_keys else None,
    }


def compare_residuals(current: pd.DataFrame, baseline: pd.DataFrame, key_cols: list[str], residual_col: str) -> dict[str, Any]:
    cur = current[key_cols + [residual_col]].copy()
    base = baseline[key_cols + [residual_col]].copy()
    cur[residual_col] = pd.to_numeric(cur[residual_col], errors="coerce")
    base[residual_col] = pd.to_numeric(base[residual_col], errors="coerce")
    merged = cur.merge(base, on=key_cols, suffixes=("_cur", "_base"), how="inner")
    diff = (merged[f"{residual_col}_cur"] - merged[f"{residual_col}_base"]).dropna()
    return {
        "common_rows": int(len(merged)),
        "max_abs_diff": float(diff.abs().max()) if len(diff) else None,
        "mean_abs_diff": float(diff.abs().mean()) if len(diff) else None,
        "rmse": float(np.sqrt((diff ** 2).mean())) if len(diff) else None,
    }


def summarize_fixed_window(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    rows = []
    for sample_type in ["pooled_full", "pairwise_matched", "control_strict_low_lai", "control_vegetated", "control_Sahara"]:
        sub = df[df["sample_type"] == sample_type]
        for method in ["harmonic", "cyclic_spline"]:
            s2 = sub[(sub["method"] == method) & (sub["sii_window_days"] == 28)]
            s1 = sub[(sub["method"] == method) & (sub["sii_window_days"] == 21)]
            for s, wd in [(s2, 28), (s1, 21)]:
                if s.empty:
                    continue
                row = s.iloc[0]
                rows.append({
                    "sample_type": sample_type,
                    "method": method,
                    "window_days": wd,
                    "spearman_rho": float(row["spearman_rho"]),
                    "spearman_p": float(row["spearman_p"]),
                    "ols_slope_per_100nT": float(row["ols_slope_per_100nT"]),
                    "n_obs": int(row["n_obs"]),
                    "n_cells": int(row["n_cells"]),
                })
    return {"summary_rows": rows, "total_rows": int(len(df))}


def main() -> None:
    report: dict[str, Any] = {"status": "in_progress", "sections": {}}

    # Run metadata
    if RUN_METADATA_PATH.exists():
        with open(RUN_METADATA_PATH) as f:
            run_meta = json.load(f)
    else:
        run_meta = None
    report["run_metadata"] = run_meta
    run_id = run_meta.get("analysis_run_id") if run_meta else None

    # File provenance check
    provenance: dict[str, Any] = {}
    for name, path in OUTPUT_FILES.items():
        df = load_optional(path)
        if df is None:
            provenance[name] = {"exists": False}
            continue
        prov = {
            "exists": True,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "n_rows": int(len(df)),
            "run_id_match": None,
        }
        for col in ["analysis_run_id", "run_id"]:
            if col in df.columns:
                vals = df[col].dropna().unique()
                prov["run_id_match"] = bool(len(vals) == 1 and vals[0] == run_id)
                prov["observed_run_id"] = vals[0] if len(vals) == 1 else vals.tolist()
                break
        provenance[name] = prov
    report["provenance"] = provenance

    # QC comparison
    cur_qc = load_optional(OUTPUT_FILES["global_qc"])
    base_qc = load_optional(ARCHIVE_DIR / "global_qc.parquet")
    if cur_qc is not None and base_qc is not None:
        report["sections"]["global_qc"] = {
            "current": key_stats(cur_qc, ["date", "lat_id", "lon_id"]),
            "baseline": key_stats(base_qc, ["date", "lat_id", "lon_id"]),
            "key_comparison": compare_keys(cur_qc, base_qc, ["date", "lat_id", "lon_id"]),
        }

    # Harmonic comparison
    cur_har = load_optional(OUTPUT_FILES["harmonic_analysis"])
    base_har = load_optional(ARCHIVE_DIR / "harmonic_analysis.parquet")
    if cur_har is not None and base_har is not None:
        report["sections"]["harmonic_analysis"] = {
            "current": {**key_stats(cur_har, ["date", "lat_id", "lon_id"]), "residuals": residual_stats(cur_har, "residual")},
            "baseline": {**key_stats(base_har, ["date", "lat_id", "lon_id"]), "residuals": residual_stats(base_har, "residual")},
            "key_comparison": compare_keys(cur_har, base_har, ["date", "lat_id", "lon_id"]),
            "residual_comparison": compare_residuals(cur_har, base_har, ["date", "lat_id", "lon_id"], "residual"),
        }

    # Spline comparison
    cur_spl = load_optional(OUTPUT_FILES["cyclic_spline_analysis"])
    base_spl = load_optional(ARCHIVE_DIR / "cyclic_spline_analysis.parquet")
    if cur_spl is not None and base_spl is not None:
        report["sections"]["cyclic_spline_analysis"] = {
            "current": {**key_stats(cur_spl, ["date", "lat_id", "lon_id"]), "residuals": residual_stats(cur_spl, "residual")},
            "baseline": {**key_stats(base_spl, ["date", "lat_id", "lon_id"]), "residuals": residual_stats(base_spl, "residual")},
            "key_comparison": compare_keys(cur_spl, base_spl, ["date", "lat_id", "lon_id"]),
            "residual_comparison": compare_residuals(cur_spl, base_spl, ["date", "lat_id", "lon_id"], "residual"),
        }

    # Matched comparison
    cur_mat = load_optional(OUTPUT_FILES["harmonic_spline_matched"])
    base_mat = load_optional(ARCHIVE_DIR / "harmonic_spline_matched.parquet")
    if cur_mat is not None and base_mat is not None:
        report["sections"]["harmonic_spline_matched"] = {
            "current": key_stats(cur_mat, ["date", "lat_id", "lon_id"]),
            "baseline": key_stats(base_mat, ["date", "lat_id", "lon_id"]),
            "key_comparison": compare_keys(cur_mat, base_mat, ["date", "lat_id", "lon_id"]),
        }

    # Fixed-window summary
    cur_fw = load_optional(OUTPUT_FILES["fixed_window_results"])
    base_fw = load_optional(ARCHIVE_DIR / "fixed_window_results.csv")
    if cur_fw is not None and base_fw is not None:
        report["sections"]["fixed_window_results"] = {
            "current": summarize_fixed_window(OUTPUT_FILES["fixed_window_results"]),
            "baseline": summarize_fixed_window(ARCHIVE_DIR / "fixed_window_results.csv"),
        }

    report["status"] = "completed"
    out_path = AUDIT_DIR / "stage_a_clean_run_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Audit written: {out_path}")


if __name__ == "__main__":
    main()
