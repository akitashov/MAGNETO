#!/usr/bin/env python3
"""Audit candidate MODIS and SIF ETL outputs against the old canonical files.

This script is intentionally separate from the pipeline stages.  It reads only
summary columns where possible, writes a lightweight comparison report, and
does not modify canonical files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import PARQUET_ENGINE, PROJECT_ROOT, atomic_write  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "results" / "etl_promotion_audits"
UNMATCHED_PATH = REPORT_DIR / "sif_modis_unmatched_cells.parquet"
FALLBACK_DIAG_PATH = REPORT_DIR / "sif_hierarchical_fallback_diagnostics.json"


def _read_modis_summary(path: Path) -> pd.DataFrame:
    cols = ["date", "lat_id", "lon_id", "lai", "cloud_fraction", "aerosol_fraction"]
    return pd.read_parquet(path, engine=PARQUET_ENGINE, columns=cols)


def _read_sif_summary(path: Path) -> pd.DataFrame:
    cols = ["date", "lat_id", "lon_id", "latitude", "longitude",
            "sif_740nm", "sif_757nm", "sif_771nm", "sif_stress_index", "region_flags", "count",
            "modis_match_mode", "lai", "cloud_fraction", "aerosol_fraction",
            "source_modis_date", "source_modis_lat_id", "source_modis_lon_id"]
    df = pd.read_feather(path)
    return df[[c for c in cols if c in df.columns]]


def _numeric_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def _flag_counts(series: pd.Series) -> dict[str, int]:
    vals = series.dropna().astype(int)
    return {
        "LOW_LAI": int((vals & 16).astype(bool).sum()),
        "HIGH_LAI": int((vals & 32).astype(bool).sum()),
        "SAA": int((vals & 1).astype(bool).sum()),
        "SAHARA": int((vals & 8).astype(bool).sum()),
        "POLAR": int((vals & 4).astype(bool).sum()),
        "CONTROL_NORTH": int((vals & 2).astype(bool).sum()),
    }


def audit_modis(candidate: Path, canonical: Path) -> dict:
    print(f"[AUDIT] MODIS candidate: {candidate}")
    print(f"[AUDIT] MODIS canonical: {canonical}")
    cnd = _read_modis_summary(candidate)
    ref = _read_modis_summary(canonical)

    cnd_cells = cnd[["lat_id", "lon_id"]].drop_duplicates()
    ref_cells = ref[["lat_id", "lon_id"]].drop_duplicates()
    cnd["year"] = pd.to_datetime(cnd["date"]).dt.year
    ref["year"] = pd.to_datetime(ref["date"]).dt.year

    report = {
        "candidate_rows": int(len(cnd)),
        "canonical_rows": int(len(ref)),
        "candidate_unique_cells": int(len(cnd_cells)),
        "canonical_unique_cells": int(len(ref_cells)),
        "candidate_date_min": str(cnd["date"].min()),
        "candidate_date_max": str(cnd["date"].max()),
        "canonical_date_min": str(ref["date"].min()),
        "canonical_date_max": str(ref["date"].max()),
        "candidate_year_counts": cnd.groupby("year").size().to_dict(),
        "canonical_year_counts": ref.groupby("year").size().to_dict(),
        "candidate_cell_summary": {
            "lat_step": float(np.diff(np.sort(cnd["lat_id"].unique()))[np.diff(np.sort(cnd["lat_id"].unique())) > 0].min()),
            "lon_step": float(np.diff(np.sort(cnd["lon_id"].unique()))[np.diff(np.sort(cnd["lon_id"].unique())) > 0].min()),
        },
        "distributions": {
            "candidate": {
                "lai": _numeric_summary(cnd["lai"]),
                "cloud_fraction": _numeric_summary(cnd["cloud_fraction"]),
                "aerosol_fraction": _numeric_summary(cnd["aerosol_fraction"]),
            },
            "canonical": {
                "lai": _numeric_summary(ref["lai"]),
                "cloud_fraction": _numeric_summary(ref["cloud_fraction"]),
                "aerosol_fraction": _numeric_summary(ref["aerosol_fraction"]),
            },
        },
    }
    return report


def audit_sif(candidate: Path, canonical: Path) -> dict:
    print(f"[AUDIT] SIF candidate: {candidate}")
    print(f"[AUDIT] SIF canonical: {canonical}")
    cnd = _read_sif_summary(candidate)
    ref = _read_sif_summary(canonical)

    cnd["year"] = pd.to_datetime(cnd["date"]).dt.year
    ref["year"] = pd.to_datetime(ref["date"]).dt.year

    cnd_cells = cnd[["lat_id", "lon_id"]].drop_duplicates()
    ref_cells = ref[["lat_id", "lon_id"]].drop_duplicates()

    report = {
        "candidate_rows": int(len(cnd)),
        "canonical_rows": int(len(ref)),
        "candidate_unique_cells": int(len(cnd_cells)),
        "canonical_unique_cells": int(len(ref_cells)),
        "candidate_date_min": str(cnd["date"].min()),
        "candidate_date_max": str(cnd["date"].max()),
        "canonical_date_min": str(ref["date"].min()),
        "canonical_date_max": str(ref["date"].max()),
        "candidate_year_counts": cnd.groupby("year").size().to_dict(),
        "canonical_year_counts": ref.groupby("year").size().to_dict(),
        "candidate_flag_counts": _flag_counts(cnd["region_flags"]),
        "canonical_flag_counts": _flag_counts(ref["region_flags"]),
        "distributions": {
            "candidate": {
                "sif_740nm": _numeric_summary(cnd["sif_740nm"]),
                "sif_757nm": _numeric_summary(cnd["sif_757nm"]),
                "sif_771nm": _numeric_summary(cnd["sif_771nm"]),
                "sif_stress_index": _numeric_summary(cnd["sif_stress_index"]),
            },
            "canonical": {
                "sif_740nm": _numeric_summary(ref["sif_740nm"]),
                "sif_757nm": _numeric_summary(ref["sif_757nm"]),
                "sif_771nm": _numeric_summary(ref["sif_771nm"]),
                "sif_stress_index": _numeric_summary(ref["sif_stress_index"]),
            },
        },
    }
    return report


def audit_sif_candidates(central: Path, hierarchical: Path) -> dict:
    """Compare the central-only and hierarchical SIF candidates."""
    print(f"[AUDIT] Central-only candidate: {central}")
    print(f"[AUDIT] Hierarchical candidate: {hierarchical}")
    cen = _read_sif_summary(central)
    hier = _read_sif_summary(hierarchical)

    cen["weight"] = cen["count"]
    hier["weight"] = hier["count"]

    cen_soundings = int(cen["weight"].sum())
    hier_soundings = int(hier["weight"].sum())
    cen_cells = int(len(cen))
    hier_cells = int(len(hier))

    extra = hier[
        ~hier.set_index(["date", "lat_id", "lon_id"]).index.isin(
            cen.set_index(["date", "lat_id", "lon_id"]).index
        )
    ]
    extra_soundings = int(extra["weight"].sum())
    extra_cells = int(len(extra))

    return {
        "central_rows": cen_cells,
        "hierarchical_rows": hier_cells,
        "central_soundings": cen_soundings,
        "hierarchical_soundings": hier_soundings,
        "extra_hierarchical_rows": extra_cells,
        "extra_hierarchical_soundings": extra_soundings,
        "extra_fraction_soundings": float(extra_soundings / cen_soundings) if cen_soundings else 0.0,
    }


def audit_unmatched_after_masks(unmatched_path: Path) -> dict:
    """Compute unmatched soundings after analytical exclusion masks.

    Uses the raw unmatched-cell summary produced by E03b.  The main
    analytical exclusions are POLAR (|lat| > 60°) and the QC latitude
    band [-60°, 75°].  Their intersection keeps -60° ≤ lat ≤ 60°.
    """
    print(f"[AUDIT] Unmatched after analytical masks: {unmatched_path}")
    df = pd.read_parquet(unmatched_path)
    df["lat"] = df["lat_id"] / 100.0

    pre_qc = int(df["unmatched_observations"].sum())

    not_polar = df["lat"].abs() <= 60.0
    in_lat_band = (df["lat"] >= -60.0) & (df["lat"] <= 75.0)
    combined = not_polar & in_lat_band

    post_polar = int(df.loc[not_polar, "unmatched_observations"].sum())
    post_lat_band = int(df.loc[in_lat_band, "unmatched_observations"].sum())
    post_combined = int(df.loc[combined, "unmatched_observations"].sum())

    return {
        "pre_qc_unmatched_soundings": pre_qc,
        "post_polar_exclusion_unmatched_soundings": post_polar,
        "post_lat_band_unmatched_soundings": post_lat_band,
        "post_combined_mask_unmatched_soundings": post_combined,
        "pre_qc_unmatched_cells": int(len(df)),
        "post_combined_mask_unmatched_cells": int(combined.sum()),
    }


def audit_modis_source_key_consistency(
    sif_path: Path,
    modis_path: Path,
    mode: str,
) -> dict:
    """Check that every MODIS source key in the SIF candidate exists in MODIS."""
    print(f"[AUDIT] MODIS source-key consistency ({mode}): {sif_path}")
    sif = pd.read_feather(sif_path, columns=None)
    modis = pd.read_parquet(
        modis_path,
        engine=PARQUET_ENGINE,
        columns=["date", "lat_id", "lon_id"],
    )
    modis_keys = set(zip(pd.to_datetime(modis["date"]).dt.date, modis["lat_id"], modis["lon_id"]))

    key_cols = ["source_modis_date", "source_modis_lat_id", "source_modis_lon_id"]
    if all(c in sif.columns for c in key_cols):
        sif["source_date"] = pd.to_datetime(sif["source_modis_date"]).dt.date
        sif_keys = list(zip(sif["source_date"], sif["source_modis_lat_id"], sif["source_modis_lon_id"]))
    elif mode == "hierarchical":
        return {"error": "hierarchical source keys not found in SIF candidate"}
    else:
        sif["date_only"] = pd.to_datetime(sif["date"]).dt.date
        sif_keys = list(zip(sif["date_only"], sif["lat_id"], sif["lon_id"]))

    total = len(sif_keys)
    missing = sum(1 for k in sif_keys if k not in modis_keys)
    return {
        "mode": mode,
        "total_keys": total,
        "missing_keys": missing,
        "missing_fraction": float(missing / total) if total else 0.0,
    }


def _fmt_num(x) -> str:
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _year_table(counts: dict) -> str:
    rows = "\n".join(
        f"| {year} | {_fmt_num(counts.get(year, 0))} |"
        for year in sorted(counts)
    )
    return f"| Year | Rows |\n|------|------|\n{rows}"


def _dist_row(var: str, stats: dict) -> str:
    if stats.get("n", 0) == 0:
        return f"| {var} | — | — | — | — | — |"
    return (
        f"| {var} | {_fmt_num(stats['n'])} | {_fmt_num(stats['mean'])} | "
        f"{_fmt_num(stats['min'])} | {_fmt_num(stats['q50'])} | {_fmt_num(stats['max'])} |"
    )


def _dist_table(candidate_dist: dict, canonical_dist: dict) -> str:
    rows = []
    vars_ = sorted(set(candidate_dist) | set(canonical_dist))
    for var in vars_:
        rows.append(_dist_row(f"{var} (candidate)", candidate_dist.get(var, {})))
        rows.append(_dist_row(f"{var} (canonical)", canonical_dist.get(var, {})))
    return (
        "| Variable | N | Mean | Min | Median | Max |\n"
        "|----------|---|------|-----|--------|-----|\n" + "\n".join(rows)
    )


def _fallback_summary(path: Path) -> str:
    """Summarise hierarchical SIF-MODIS fallback diagnostics if available."""
    if not path.exists():
        return "No hierarchical fallback diagnostics available."
    with path.open("r", encoding="utf-8") as fh:
        diag = json.load(fh)

    total_weight = int(diag.get("total_soundings", 0))
    total_cells = int(diag.get("total_aggregate_rows", 0))
    mode_weights = diag.get("mode_soundings", {})
    mode_cells = diag.get("mode_cells", {})

    lines = [
        "### Hierarchical SIF–MODIS fallback",
        "",
        f"- Total matched soundings: {total_weight:,}",
        f"- Total matched cells: {total_cells:,}",
        "",
        "| Match mode | Soundings | % soundings | Cells | % cells |",
        "|------------|----------:|------------:|------:|--------:|",
    ]
    for mode in ["central_exact", "temporal_same_cell", "spatial_same_date", "unmatched"]:
        w = int(mode_weights.get(mode, 0))
        c = int(mode_cells.get(mode, 0))
        lines.append(
            f"| {mode} | {w:,} | "
            f"{100 * w / total_weight if total_weight else 0:.2f}% | "
            f"{c:,} | {100 * c / total_cells if total_cells else 0:.2f}% |"
        )

    yearly = diag.get("yearly", [])
    if yearly:
        lines.extend([
            "",
            "#### Yearly breakdown",
            "",
            "| Year | Soundings | Cells | Central % | Temporal % | Spatial % | Unmatched % |",
            "|------|----------:|------:|----------:|-----------:|----------:|------------:|",
        ])
        for row in yearly:
            lines.append(
                f"| {row['year']} | {int(row['total_soundings']):,} | "
                f"{int(row['total_cells']):,} | "
                f"{row['central_exact_fraction_soundings']:.2%} | "
                f"{row['temporal_same_cell_fraction_soundings']:.2%} | "
                f"{row['spatial_same_date_fraction_soundings']:.2%} | "
                f"{row.get('unmatched_fraction_soundings', 0):.2%} |"
            )

    scenario = diag.get("scenario", [])
    if scenario:
        lines.extend([
            "",
            "#### Unmatched fraction by scenario",
            "",
            "| Scenario | Soundings | Unmatched % |",
            "|----------|----------:|------------:|",
        ])
        for row in scenario:
            lines.append(
                f"| {row['scenario']} | {int(row['total_soundings']):,} | "
                f"{row['unmatched_fraction_soundings']:.2%} |"
            )

    lines.extend([
        "",
        "### QC flow and final retained counts",
        "",
        "Counts are **soundings (cells)**.",
        "",
        "| Stage | Hierarchical | Central-only |",
        "|-------|-------------:|-------------:|",
    ])
    hier_flow_s = diag.get("qc_flow_soundings", {}).get("hierarchical", {})
    cen_flow_s = diag.get("qc_flow_soundings", {}).get("central_only", {})
    hier_flow_c = diag.get("qc_flow_cells", {}).get("hierarchical", {})
    cen_flow_c = diag.get("qc_flow_cells", {}).get("central_only", {})
    for step in ["matched_pre_qc", "after_cloud_filter", "after_aerosol_filter", "after_lai_filter"]:
        lines.append(
            f"| {step} | {int(hier_flow_s.get(step, 0)):,} ({int(hier_flow_c.get(step, 0)):,}) | "
            f"{int(cen_flow_s.get(step, 0)):,} ({int(cen_flow_c.get(step, 0)):,}) |"
        )

    final = diag.get("final_retained_counts", {})
    lines.extend([
        "",
        "### Final candidate outputs",
        "",
        f"- Hierarchical retained: {int(final.get('hierarchical', {}).get('soundings', 0)):,} soundings, "
        f"{int(final.get('hierarchical', {}).get('cells', 0)):,} cells",
        f"- Central-only retained: {int(final.get('central_only', {}).get('soundings', 0)):,} soundings, "
        f"{int(final.get('central_only', {}).get('cells', 0)):,} cells",
    ])

    return "\n".join(lines)


def _unmatched_summary(path: Path, fallback_diag_path: Path) -> str:
    """Report on unmatched observations, using fallback diagnostics if present."""
    if fallback_diag_path.exists():
        with fallback_diag_path.open("r", encoding="utf-8") as fh:
            diag = json.load(fh)
        raw_unmatched = int(diag.get("raw_unmatched_soundings", 0))
        hierarchical_unmatched = int(diag.get("mode_soundings", {}).get("unmatched", 0))
        if hierarchical_unmatched == 0:
            return (
                "### MODIS central-match gap analysis\n\n"
                "The hierarchical SIF–MODIS fallback recovered all SIF soundings: "
                f"**0 unmatched** out of {_fmt_num(int(diag.get('total_soundings', 0)))} "
                "soundings.  The legacy unmatched-cell parquet "
                f"(`{path.name}`) is retained for reference but is now stale."
            )
        return (
            "### MODIS central-match gap analysis\n\n"
            f"- Hierarchical unmatched soundings: {_fmt_num(hierarchical_unmatched)}\n"
            f"- Raw central-only unmatched soundings: {_fmt_num(raw_unmatched)}"
        )

    if not path.exists():
        return "No unmatched-cell diagnostics available."
    df = pd.read_parquet(path)
    if df.empty:
        return "No unmatched SIF observations found."

    df["lat"] = df["lat_id"] / 100.0
    df["lon"] = df["lon_id"] / 100.0
    lat_bins = pd.cut(df["lat"], bins=np.arange(-60, 81, 10), right=False)
    lon_bins = pd.cut(df["lon"], bins=np.arange(-180, 181, 20), right=False)

    lat_summary = (
        df.groupby(lat_bins, observed=False)["unmatched_observations"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    lon_summary = (
        df.groupby(lon_bins, observed=False)["unmatched_observations"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    lat_rows = "\n".join(
        f"| {str(idx)} | {_fmt_num(int(v))} |"
        for idx, v in lat_summary.items()
    )
    lon_rows = "\n".join(
        f"| {str(idx)} | {_fmt_num(int(v))} |"
        for idx, v in lon_summary.items()
    )

    return f"""### MODIS central-match gap analysis

- Total unmatched SIF observations: {_fmt_num(int(df["unmatched_observations"].sum()))}
- Unique unmatched (year, cell) combinations: {_fmt_num(len(df))}
- Unique unmatched cells: {_fmt_num(int(df[["lat_id", "lon_id"]].drop_duplicates().shape[0]))}

These are SIF soundings for which the conservatively regridded MODIS product has
no valid LAI, cloud or aerosol value at the same (date, 0.5° cell).  They are not
observations that merely failed the cloud/aerosol/LAI quality filters.

#### Top 10 latitude bands by unmatched observations

| Latitude band | Unmatched observations |
|---------------|-----------------------:|
{lat_rows}

#### Top 10 longitude bands by unmatched observations

| Longitude band | Unmatched observations |
|----------------|-----------------------:|
{lon_rows}

#### Likely causes

1. **Genuine missing MODIS 8-day periods** in the raw ICDC archive (e.g. 2022-10-15
   appears absent, leaving a one-period gap across all cells).
2. **Persistent coastal / inland-water cells** where the source MODIS product is
   flagged as missing for the entire year.
3. **High-latitude cells** (e.g. 81–82° N in 2024) where the MODIS coverage is sparse.

A conditional nearest-cell fallback would recover some of these observations by
borrowing MODIS values from an adjacent 0.5° cell, but this introduces spatial
misalignment and should be evaluated against the scientific cost of dropping the
observations.
"""


def _candidate_comparison_md(report: dict) -> str:
    comp = report.get("sif_candidate_comparison", {})
    if not comp:
        return ""
    return f"""### Central-only vs hierarchical candidate

| Metric | Value |
|--------|------:|
| Central-only rows | {_fmt_num(comp.get('central_rows', 0))} |
| Hierarchical rows | {_fmt_num(comp.get('hierarchical_rows', 0))} |
| Central-only soundings | {_fmt_num(comp.get('central_soundings', 0))} |
| Hierarchical soundings | {_fmt_num(comp.get('hierarchical_soundings', 0))} |
| Extra hierarchical rows | {_fmt_num(comp.get('extra_hierarchical_rows', 0))} |
| Extra hierarchical soundings | {_fmt_num(comp.get('extra_hierarchical_soundings', 0))} |
| Extra fraction vs central | {comp.get('extra_fraction_soundings', 0):.4%} |
"""


def _key_consistency_md(report: dict) -> str:
    checks = report.get("modis_source_key_consistency", [])
    if not checks:
        return ""
    lines = [
        "### MODIS source-key consistency",
        "",
        "| Mode | Total keys | Missing | Missing % |",
        "|------|----------:|--------:|----------:|",
    ]
    for check in checks:
        if "error" in check:
            lines.append(f"| {check.get('mode', '?')} | — | — | {check['error']} |")
        else:
            lines.append(
                f"| {check['mode']} | {_fmt_num(check.get('total_keys', 0))} | "
                f"{_fmt_num(check.get('missing_keys', 0))} | "
                f"{check.get('missing_fraction', 0):.4%} |"
            )
    return "\n".join(lines)


def _unmatched_after_masks_md(report: dict) -> str:
    u = report.get("unmatched_after_masks", {})
    if not u:
        return ""
    return f"""### Unmatched after analytical exclusions

Pre-QC unmatched soundings include polar regions that are explicitly excluded
from the analysis area in `01_build_qc.py`.  After applying the POLAR exclusion
(|lat| > 60°) and the QC latitude band [-60°, 75°], only the overlap
(-60° ≤ lat ≤ 60°) remains relevant for inference.

| Mask | Unmatched soundings | Unmatched cells |
|------|--------------------:|----------------:|
| Pre-QC (all) | {_fmt_num(u.get('pre_qc_unmatched_soundings', 0))} | {_fmt_num(u.get('pre_qc_unmatched_cells', 0))} |
| After POLAR exclusion | {_fmt_num(u.get('post_polar_exclusion_unmatched_soundings', 0))} | — |
| After QC lat band | {_fmt_num(u.get('post_lat_band_unmatched_soundings', 0))} | — |
| After combined mask (-60° ≤ lat ≤ 60°) | {_fmt_num(u.get('post_combined_mask_unmatched_soundings', 0))} | {_fmt_num(u.get('post_combined_mask_unmatched_cells', 0))} |
"""


def _impact_assessment(report: dict) -> str:
    modis = report.get("modis", {})
    sif = report.get("sif", {})

    sif_rows_c = sif.get("candidate_rows", 0)
    sif_rows_r = sif.get("canonical_rows", 0)
    sif_cells_c = sif.get("candidate_unique_cells", 0)
    sif_cells_r = sif.get("canonical_unique_cells", 0)

    return f"""## Downstream impact assessment

Because the MODIS grid-registration correction intentionally changes the spatial
assignment of LAI, cloud and aerosol data, the candidate MODIS and SIF files are
expected to differ from the old canonical files.  The old canonical files used a
0.25°-offset snap (`np.round`) that placed MODIS cells on integer-degree centres;
the candidate uses conservative regridding onto the true 0.5° target grid.

### Expected changes

- MODIS candidate rows and unique cells may shift relative to the baseline.
- SIF observations after MODIS QC may gain or lose rows depending on whether the
  corrected MODIS cell passes cloud/aerosol/LAI filters.
- Region flags (LOW_LAI / HIGH_LAI) depend on LAI thresholds and may shift at
  arid margins.

### Candidate vs canonical counts

| Dataset | Candidate rows | Canonical rows | Candidate cells | Canonical cells |
|---------|---------------:|---------------:|----------------:|----------------:|
| MODIS   | {_fmt_num(modis.get('candidate_rows', 0))} | {_fmt_num(modis.get('canonical_rows', 0))} | {_fmt_num(modis.get('candidate_unique_cells', 0))} | {_fmt_num(modis.get('canonical_unique_cells', 0))} |
| SIF     | {_fmt_num(sif_rows_c)} | {_fmt_num(sif_rows_r)} | {_fmt_num(sif_cells_c)} | {_fmt_num(sif_cells_r)} |

### Stages requiring recalculation if candidate is accepted

If the candidate files are promoted after operator review, the following stages
must be rerun from scratch because their inputs change:

1. **01_build_qc.py** — merges SIF with corrected MODIS.
2. **02_assign_lai_quartiles.py** — LAI summaries and quartile boundaries depend on MODIS.
3. **02b_smoke_subset.py** — smoke-subset eligibility depends on QC.
4. **04_harmonic.py**, **05_cyclic_spline.py** — residuals depend on QC sample.
5. **07_matched.py** — matched samples depend on residuals.
6. **08_fixed_window.py**, **09_surrogates.py** — inference depends on residuals/SII.
7. **10_effects.py**, **10_prepare_environmental_driver_input.py**,
   **11_environmental_driver_matrix_gpu.py**, **12_environmental_driver_matrix_summary.py**.
8. All supplementary-check stages that read fixed-window / surrogate results.
9. All figure and table rendering scripts.

Stages that do **not** need recalculation:

- E02_omni_etl.py (non-spatial).
- E04_mcd12c1_hdf2netcdf.py and S05 land-cover assignment (MCD12C1 is independent).
- E05_era5_etl.py (ERA5 grid is unchanged).

### Resource estimate

A full clean analytical run from QC through figures/tables is expected to take
several hours and produce output files of roughly the same size as the current
results directory.  The exact duration depends on GPU availability for the
driver-matrix stage.

### Recommendation

Do **not** promote the candidate files or rerun downstream analysis until the
MODIS/SIF candidate audit has been reviewed.  Pay particular attention to:

- SIF row/cell counts and geographic distribution;
- LOW_LAI / HIGH_LAI counts, especially for the strict low-LAI control;
- Central MODIS match rate reported by E03_sif_etl.py;
- Any abrupt year-to-year shifts that are not explained by the grid correction.
"""


def _write_markdown(report: dict, path: Path) -> None:
    modis = report.get("modis", {})
    sif = report.get("sif", {})

    md = f"""# MAGNETO candidate ETL audit

Generated: {report.get('timestamp', '')}

## MODIS candidate vs canonical

- Candidate: `data/interim/modis_extract_candidate.parquet`
- Canonical: `data/interim/modis_extract.parquet`

### Summary

| Metric | Candidate | Canonical |
|--------|----------:|----------:|
| Rows | {_fmt_num(modis.get('candidate_rows', 0))} | {_fmt_num(modis.get('canonical_rows', 0))} |
| Unique cells | {_fmt_num(modis.get('candidate_unique_cells', 0))} | {_fmt_num(modis.get('canonical_unique_cells', 0))} |
| Date range | {modis.get('candidate_date_min', 'N/A')} → {modis.get('candidate_date_max', 'N/A')} | {modis.get('canonical_date_min', 'N/A')} → {modis.get('canonical_date_max', 'N/A')} |

### Year counts

{_year_table(modis.get('candidate_year_counts', {}))}

### Distribution summary

{_dist_table(modis.get('distributions', {}).get('candidate', {}), modis.get('distributions', {}).get('canonical', {}))}

## SIF central-only candidate vs canonical

- Candidate: `data/interim/sif_aggregated_central_only_candidate.feather`
- Canonical: `data/interim/sif_aggregated.feather`

### Summary

| Metric | Candidate | Canonical |
|--------|----------:|----------:|
| Rows | {_fmt_num(sif.get('candidate_rows', 0))} | {_fmt_num(sif.get('canonical_rows', 0))} |
| Unique cells | {_fmt_num(sif.get('candidate_unique_cells', 0))} | {_fmt_num(sif.get('canonical_unique_cells', 0))} |
| Date range | {sif.get('candidate_date_min', 'N/A')} → {sif.get('candidate_date_max', 'N/A')} | {sif.get('canonical_date_min', 'N/A')} → {sif.get('canonical_date_max', 'N/A')} |

### Region flag counts

| Flag | Candidate | Canonical |
|------|----------:|----------:|
| LOW_LAI | {_fmt_num(sif.get('candidate_flag_counts', {}).get('LOW_LAI', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('LOW_LAI', 0))} |
| HIGH_LAI | {_fmt_num(sif.get('candidate_flag_counts', {}).get('HIGH_LAI', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('HIGH_LAI', 0))} |
| SAA | {_fmt_num(sif.get('candidate_flag_counts', {}).get('SAA', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('SAA', 0))} |
| SAHARA | {_fmt_num(sif.get('candidate_flag_counts', {}).get('SAHARA', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('SAHARA', 0))} |
| POLAR | {_fmt_num(sif.get('candidate_flag_counts', {}).get('POLAR', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('POLAR', 0))} |
| CONTROL_NORTH | {_fmt_num(sif.get('candidate_flag_counts', {}).get('CONTROL_NORTH', 0))} | {_fmt_num(sif.get('canonical_flag_counts', {}).get('CONTROL_NORTH', 0))} |

### Year counts

{_year_table(sif.get('candidate_year_counts', {}))}

### Distribution summary

{_dist_table(sif.get('distributions', {}).get('candidate', {}), sif.get('distributions', {}).get('canonical', {}))}

{_fallback_summary(FALLBACK_DIAG_PATH)}

{_candidate_comparison_md(report)}

{_key_consistency_md(report)}

{_unmatched_after_masks_md(report)}

{_unmatched_summary(UNMATCHED_PATH, FALLBACK_DIAG_PATH)}

{_impact_assessment(report)}
"""
    path.write_text(md, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit candidate ETL outputs")
    parser.add_argument("--modis-candidate", type=Path,
                        default=PROJECT_ROOT / "data/interim/modis_extract_candidate.parquet")
    parser.add_argument("--modis-canonical", type=Path,
                        default=PROJECT_ROOT / "data/interim/modis_extract.parquet")
    parser.add_argument("--sif-candidate", type=Path,
                        default=PROJECT_ROOT / "data/interim/sif_aggregated_central_only_candidate.feather")
    parser.add_argument("--sif-hierarchical-candidate", type=Path,
                        default=PROJECT_ROOT / "data/interim/sif_aggregated_hierarchical_candidate.feather")
    parser.add_argument("--sif-canonical", type=Path,
                        default=PROJECT_ROOT / "data/interim/sif_aggregated.feather")
    args = parser.parse_args(argv)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"timestamp": pd.Timestamp.now().isoformat()}

    if args.modis_candidate.exists() and args.modis_canonical.exists():
        report["modis"] = audit_modis(args.modis_candidate, args.modis_canonical)
    else:
        print(f"[SKIP] MODIS audit: candidate={args.modis_candidate.exists()}, canonical={args.modis_canonical.exists()}")

    if args.sif_candidate.exists() and args.sif_canonical.exists():
        report["sif"] = audit_sif(args.sif_candidate, args.sif_canonical)
    else:
        print(f"[SKIP] SIF central-only audit: candidate={args.sif_candidate.exists()}, canonical={args.sif_canonical.exists()}")

    if args.sif_candidate.exists() and args.sif_hierarchical_candidate.exists():
        report["sif_candidate_comparison"] = audit_sif_candidates(
            args.sif_candidate, args.sif_hierarchical_candidate
        )
    else:
        print(f"[SKIP] SIF candidate comparison: central={args.sif_candidate.exists()}, hierarchical={args.sif_hierarchical_candidate.exists()}")

    if args.modis_candidate.exists():
        consistency = []
        if args.sif_candidate.exists():
            consistency.append(audit_modis_source_key_consistency(
                args.sif_candidate, args.modis_candidate, mode="central"
            ))
        if args.sif_hierarchical_candidate.exists():
            consistency.append(audit_modis_source_key_consistency(
                args.sif_hierarchical_candidate, args.modis_candidate, mode="hierarchical"
            ))
        if consistency:
            report["modis_source_key_consistency"] = consistency

    unmatched_path = REPORT_DIR / "sif_modis_unmatched_cells.parquet"
    if unmatched_path.exists():
        report["unmatched_after_masks"] = audit_unmatched_after_masks(unmatched_path)

    json_path = REPORT_DIR / "candidate_etl_audit.json"
    md_path = REPORT_DIR / "candidate_etl_audit.md"
    atomic_write(report, json_path, fmt="json")
    _write_markdown(report, md_path)
    print(f"[OK] Reports written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
