#!/usr/bin/env python3
"""
run_supplementary_outcomes.py — Lightweight supplementary outcome pipeline.

Evaluates the pooled SII-SIF association for two secondary outcomes:
  - SIF 757 nm (direct retrieval)
  - SIF 740 nm (provider-derived reference-wavelength estimate)

The SIF757/SIF771 stress ratio is not evaluated as a physiological outcome
(see Stage B instructions).

Uses the same fixed SII windows (21/28 days), detrending methods (harmonic,
corrected cyclic spline), and temporal-null procedures as the primary pipeline.
Does NOT add temperature/LAI stratification, functional/geographic controls, or
new detrending methods.

For each outcome it computes two samples:
  - outcome_specific_full: maximal valid sample for that outcome;
  - common_outcome_matched: strict intersection of keys where all three outcomes
    (771, 757, 740) have valid residuals under the same method.
"""
from __future__ import annotations
import sys, os, argparse, importlib.util, subprocess, hashlib, json, gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from _Common import (
    PROJECT_ROOT, config_hash, full_sha256, atomic_write, run_provenance_dict,
    current_analysis_run_id, RUN_METADATA_PATH,
    FILE_QC, FILE_LAI_MEMBERS, FILE_RES_HARMONIC, FILE_RES_CYCLIC, FILE_MATCHED_HC,
    FILE_FIXED, FILE_SURROGATES, PARQUET_ENGINE, SII_WINDOWS, SII_RAW_COL,
    SURROGATE_MODES, SURROGATE_SEED, SURROGATE_N, SURROGATE_SMOKE_N,
)
from magneto_lib import build_daily_sii, compute_sii_windows, build_strata, run_fixed_window, run_surrogates

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

SUPPLEMENTARY_RUN_METADATA = PROJECT_ROOT / "results" / "supplementary_run_metadata.json"

# Outcome definitions.
# SIF 740 nm is a provider-estimated reference-wavelength quantity derived from
# the 757 nm and 771 nm retrievals in the OCO-2/OCO-3 SIF Lite product.
# The stress indicator is a study-derived ratio with an explicit denominator
# admissibility rule (sif_771nm >= 0.001) recovered from the archived column.
SUPPLEMENTARY_OUTCOMES = {
    "sif_757nm": {"column": "sif_757nm", "type": "direct",
                  "formula": "sif_757nm", "denominator_rule": "none",
                  "sample_label": "outcome_specific_full"},
    "sif_740nm": {"column": "sif_740nm", "type": "provider_derived",
                  "formula": "0.75 * (sif_757nm + 1.5 * sif_771nm)",
                  "denominator_rule": "none",
                  "sample_label": "outcome_specific_full"},
}

OUTPUT_RESIDUALS_HARMONIC = PROJECT_ROOT / "data" / "processed" / "supplementary_harmonic_residuals.parquet"
OUTPUT_RESIDUALS_SPLINE = PROJECT_ROOT / "data" / "processed" / "supplementary_spline_residuals.parquet"
OUTPUT_COMMON_HARMONIC = PROJECT_ROOT / "data" / "processed" / "supplementary_common_harmonic.parquet"
OUTPUT_COMMON_SPLINE = PROJECT_ROOT / "data" / "processed" / "supplementary_common_spline.parquet"
OUTPUT_RESULTS = PROJECT_ROOT / "results" / "supplementary_outcomes_results.csv"
OUTPUT_SURROGATE_SUMMARY = PROJECT_ROOT / "results" / "supplementary_outcomes_surrogate_summary.csv"
OUTPUT_SURROGATE_NULL = PROJECT_ROOT / "results" / "supplementary_outcomes_null_distributions.parquet"
OUTPUT_AVAILABILITY = PROJECT_ROOT / "results" / "supplementary_outcomes_availability.csv"
OUTPUT_COMMON_DIAGNOSTICS = PROJECT_ROOT / "results" / "supplementary_common_sample_diagnostics.csv"
OUTPUT_MANIFEST_JSON = PROJECT_ROOT / "results" / "supplementary_outcomes_manifest.json"
OUTPUT_MANIFEST_CSV = PROJECT_ROOT / "results" / "supplementary_outcomes_manifest.csv"
OUTPUT_TABLE = PROJECT_ROOT / "results" / "table_supplementary_outcomes.csv"


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def _runner_hash() -> str:
    """SHA-256 of this runner script; used to invalidate cached residuals on code changes."""
    return full_sha256(Path(__file__).resolve())


def _outcome_def_hash() -> str:
    """Deterministic hash of the current outcome definitions."""
    return hashlib.sha256(json.dumps(SUPPLEMENTARY_OUTCOMES, sort_keys=True).encode()).hexdigest()


def _residual_path(outcome: str, method: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"supplementary_{method}_residuals_{outcome}.parquet"


def _load_old_residuals_for_reuse(resume: bool) -> dict:
    """Load per-outcome residual files from a previous run when resuming."""
    if not resume:
        return {}
    old = {}
    for outcome in SUPPLEMENTARY_OUTCOMES:
        for method in ["harmonic", "cyclic_spline"]:
            p = _residual_path(outcome, method)
            if p.exists():
                old[(outcome, method)] = pd.read_parquet(p, engine=PARQUET_ENGINE)
    return old


def _clear_gpu_cache() -> None:
    """Release cached CuPy memory back to the driver between heavy stages."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned = cp.get_default_pinned_memory_pool()
        pinned.free_all_blocks()
    except Exception:
        pass


def _load_old_results_for_reuse(resume: bool) -> dict:
    """Load old results/surrogate tables that may be reused when resuming."""
    if not resume:
        return {}
    old = {}
    for p, key in [
        (OUTPUT_RESULTS, "results"),
        (OUTPUT_SURROGATE_SUMMARY, "surr_summary"),
        (OUTPUT_SURROGATE_NULL, "surr_null"),
        (OUTPUT_AVAILABILITY, "availability"),
    ]:
        if p.exists():
            if p.suffix == ".csv":
                old[key] = pd.read_csv(p)
            elif p.suffix == ".parquet":
                old[key] = pd.read_parquet(p, engine=PARQUET_ENGINE)
    return old


def _load_detrending_modules() -> tuple:
    """Import the harmonic and cyclic-spline fitting modules."""
    spec_h = importlib.util.spec_from_file_location("harmonic_supp", str(SCRIPT_DIR / "04_harmonic.py"))
    mod_h = importlib.util.module_from_spec(spec_h)
    spec_h.loader.exec_module(mod_h)

    spec_c = importlib.util.spec_from_file_location("spline_supp", str(SCRIPT_DIR / "05_cyclic_spline.py"))
    mod_c = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(mod_c)
    return mod_h, mod_c


def _build_qc_for_outcome(qc: pd.DataFrame, outcome: str, stress_source: str = "recompute") -> pd.DataFrame:
    """Return QC rows valid for the given outcome.

    For the stress indicator the default is to recompute sif_757nm / sif_771nm
    and enforce the recovered denominator rule sif_771nm >= 0.001. The stored
    column is never used silently.
    """
    col = SUPPLEMENTARY_OUTCOMES[outcome]["column"]
    if outcome == "sif_stress_index":
        qc = qc.copy()
        recomputed = qc["sif_757nm"] / qc["sif_771nm"]
        # Apply the only recoverable admissibility rule.
        admissible = (
            qc["sif_757nm"].notna() & np.isfinite(qc["sif_757nm"]) &
            qc["sif_771nm"].notna() & np.isfinite(qc["sif_771nm"]) &
            (qc["sif_771nm"] >= 0.001)
        )
        if stress_source == "stored_verified":
            if col not in qc.columns:
                raise ValueError("--stress-source=stored_verified requested but "
                                 "sif_stress_index column is missing from QC")
            stored = qc[col]
            finite_both = admissible & stored.notna() & np.isfinite(stored)
            mismatch = (stored[finite_both] - recomputed[finite_both]).abs().max()
            if mismatch > 1e-12:
                raise ValueError(
                    f"--stress-source=stored_verified failed: stored stress column "
                    f"differs from recomputed ratio by {mismatch:.3e}. Use recompute."
                )
            missing_match = (stored.notna() == admissible).all()
            if not missing_match:
                raise ValueError(
                    "--stress-source=stored_verified failed: stored stress missingness "
                    "does not match the admissibility rule sif_771nm >= 0.001."
                )
            qc[col] = stored
        else:
            # Recompute: set stress to NaN where denominator rule is not satisfied.
            qc[col] = recomputed.where(admissible, np.nan)
        mask = qc[col].notna() & np.isfinite(qc[col])
    else:
        mask = qc[col].notna() & np.isfinite(qc[col])
    return qc[mask].copy()


def _fit_outcome(qc: pd.DataFrame, outcome: str, method: str, mod_h, mod_c) -> pd.DataFrame:
    """Fit leave-one-year-out model for one outcome and method."""
    col = SUPPLEMENTARY_OUTCOMES[outcome]["column"]
    fit_func = mod_h.fit_cell_harmonic if method == "harmonic" else mod_c.fit_cell_cyclic_spline

    cells = qc.groupby(["lat_id", "lon_id"])
    all_r = []
    for _, df_cell in tqdm(cells, total=cells.ngroups, desc=f"{method} {outcome}"):
        r = fit_func(df_cell, target=col)
        if len(r) > 0:
            r["outcome"] = outcome
            r["outcome_type"] = SUPPLEMENTARY_OUTCOMES[outcome]["type"]
            all_r.append(r)

    if not all_r:
        return pd.DataFrame()
    df_out = pd.concat(all_r, ignore_index=True)
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out["method"] = method
    return df_out


def _attach_sii_and_metadata(residuals: pd.DataFrame, sii_win: pd.DataFrame,
                             qc_meta: pd.DataFrame, run_meta_path: Path) -> pd.DataFrame:
    """Attach SII windows, temperature/LAI/control metadata, and run provenance."""
    residuals = residuals.merge(sii_win, on="date", how="left")

    meta_cols = ["date", "lat_id", "lon_id", "temp_bin_label", "lai_quartile",
                 "is_strict_low_lai", "is_vegetated", "is_Sahara"]
    meta_cols = [c for c in meta_cols if c in qc_meta.columns]
    if meta_cols:
        residuals = residuals.merge(qc_meta[meta_cols].drop_duplicates(),
                                    on=["date", "lat_id", "lon_id"], how="left")

    prov = run_provenance_dict(producing_stage="run_supplementary_outcomes.py",
                               metadata_path=run_meta_path)
    for k, v in prov.items():
        residuals[k] = v
    return residuals


def _valid_residual_mask(residuals: pd.DataFrame) -> pd.Series:
    """Method-aware valid-fit mask.

    Harmonic rows use pass_flag == True; cyclic-spline rows use fit_status == "ok".
    The combined long dataset carries both columns, so the mask must be method-specific.
    """
    mask = pd.Series(False, index=residuals.index)
    if "pass_flag" in residuals.columns:
        mask |= (residuals.get("method") == "harmonic") & (residuals["pass_flag"] == True)
    if "fit_status" in residuals.columns:
        mask |= (residuals.get("method") == "cyclic_spline") & (residuals["fit_status"] == "ok")
    # Fallback for rows with neither flag
    mask |= residuals["residual"].notna()
    return mask


def _build_common_sample(residuals: pd.DataFrame, required_outcomes: list) -> pd.DataFrame:
    """Return rows whose keys have valid residuals for all required outcomes."""
    valid = residuals[_valid_residual_mask(residuals)].copy()

    keys = valid[["date", "lat_id", "lon_id", "outcome"]].drop_duplicates()
    key_counts = keys.groupby(["date", "lat_id", "lon_id"]).size().reset_index(name="n_outcomes")
    common_keys = key_counts[key_counts["n_outcomes"] >= len(required_outcomes)]
    common = valid.merge(common_keys[["date", "lat_id", "lon_id"]], on=["date", "lat_id", "lon_id"], how="inner")
    return common


def _observation_key_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of observation keys for a sample."""
    cols = [c for c in ["date", "lat_id", "lon_id"] if c in df.columns]
    if not cols:
        return ""
    sub = df[cols].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
    sub = sub.drop_duplicates().sort_values(["date", "lat_id", "lon_id"]).reset_index(drop=True)
    key_str = "\n".join("_".join(str(v) for v in r) for r in sub.values)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _compute_stats_for_sample(residuals: pd.DataFrame, sample_type: str, method: str,
                              sii_window_cols: list) -> list:
    """Compute fixed-window stats for each outcome in a residual dataset."""
    valid = residuals[_valid_residual_mask(residuals)].copy()

    strata = build_strata(valid, sample_type, method, "residual",
                          group_cols=["outcome"], windows=SII_WINDOWS)
    rows = run_fixed_window(strata)
    return rows


def _import_main_sif771_pooled() -> tuple:
    """Import primary SIF 771 pooled-full fixed-window and surrogate results."""
    fixed = pd.read_csv(FILE_FIXED)
    fixed = fixed[fixed["sample_type"] == "pooled_full"].copy()
    fixed["outcome"] = "sif_771nm"
    fixed["outcome_type"] = "direct"
    fixed["sample_type"] = "outcome_specific_full"

    surr = pd.read_csv(FILE_SURROGATES)
    surr = surr[surr["sample_type"] == "pooled_full"].copy()
    surr["outcome"] = "sif_771nm"
    surr["outcome_type"] = "direct"
    surr["sample_type"] = "outcome_specific_full"
    return fixed, surr


def _replace_sif771_from_main(summary_df: pd.DataFrame,
                              null_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace supplementary SIF 771 outcome-specific-full rows with the final
    primary-run rows.  This guarantees that the supplementary table imports the
    same N=1000 (or N=10 smoke) null distribution that the main pipeline wrote,
    even if the main run finished after the supplementary run started.
    """
    if not FILE_SURROGATES.exists():
        return summary_df, null_df

    main_surr = pd.read_csv(FILE_SURROGATES)
    main_surr = main_surr[main_surr["sample_type"] == "pooled_full"].copy()
    if main_surr.empty:
        return summary_df, null_df
    main_surr["outcome"] = "sif_771nm"
    main_surr["outcome_type"] = "direct"
    main_surr["sample_type"] = "outcome_specific_full"

    # Drop any previously imported SIF 771 outcome-specific-full rows.
    mask = (summary_df["outcome"] == "sif_771nm") & (
        summary_df["sample_type"] == "outcome_specific_full"
    )
    n_drop = mask.sum()
    summary_df = summary_df[~mask].copy()

    # Align columns and append final main rows.
    main_surr = main_surr[[c for c in summary_df.columns if c in main_surr.columns]]
    for c in summary_df.columns:
        if c not in main_surr.columns:
            main_surr[c] = None
    main_surr = main_surr[summary_df.columns]
    summary_df = pd.concat([summary_df, main_surr], ignore_index=True)
    log(f"Replaced {n_drop} supplementary SIF 771 outcome-specific-full summary rows "
        f"with {len(main_surr)} primary-run rows.")

    # Update the null-distribution file in the same way.
    main_null_path = PROJECT_ROOT / "results" / "surrogate_null_distributions.parquet"
    if not main_null_path.exists() or null_df.empty:
        return summary_df, null_df

    main_null = pd.read_parquet(main_null_path)
    main_null = main_null[main_null["sample_type"] == "pooled_full"].copy()
    if main_null.empty:
        return summary_df, null_df
    main_null["outcome"] = "sif_771nm"
    main_null["outcome_type"] = "direct"
    main_null["sample_type"] = "outcome_specific_full"

    mask_null = (null_df["outcome"] == "sif_771nm") & (
        null_df["sample_type"] == "outcome_specific_full"
    )
    n_drop_null = mask_null.sum()
    null_df = null_df[~mask_null].copy()

    main_null = main_null[[c for c in null_df.columns if c in main_null.columns]]
    for c in null_df.columns:
        if c not in main_null.columns:
            main_null[c] = None
    main_null = main_null[null_df.columns]
    null_df = pd.concat([null_df, main_null], ignore_index=True)
    log(f"Replaced {n_drop_null} supplementary SIF 771 outcome-specific-full null rows "
        f"with {len(main_null)} primary-run rows.")

    return summary_df, null_df


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGNETO supplementary outcome pipeline")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-surrogates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--clean-generated", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=("Analysis run_id to record in provenance. Defaults to the run_id in "
              "results/run_metadata.json if present; otherwise a new UTC timestamp "
              "is generated."),
    )
    parser.add_argument(
        "--gpu-chunk-size",
        type=int,
        default=None,
        help=("CuPy Spearman chunk size (columns processed per GPU batch). Lower "
              "values reduce VRAM spikes on memory-constrained WSL hosts. If not "
              "set, the default in magneto_lib is used."),
    )
    parser.add_argument(
        "--stress-source",
        choices=["recompute", "stored_verified"],
        default="recompute",
        help=("How to obtain the stress indicator: 'recompute' (default) explicitly "
              "computes sif_757nm / sif_771nm with the recovered denominator rule "
              "sif_771nm >= 0.001; 'stored_verified' requires the archived "
              "sif_stress_index column to match the recomputed ratio exactly."),
    )
    return parser.parse_args(argv)


def clean_supplementary_generated() -> None:
    paths = [
        OUTPUT_RESIDUALS_HARMONIC, OUTPUT_RESIDUALS_SPLINE,
        OUTPUT_COMMON_HARMONIC, OUTPUT_COMMON_SPLINE,
        OUTPUT_RESULTS, OUTPUT_SURROGATE_SUMMARY, OUTPUT_SURROGATE_NULL,
        OUTPUT_AVAILABILITY, OUTPUT_COMMON_DIAGNOSTICS,
        OUTPUT_MANIFEST_JSON, OUTPUT_MANIFEST_CSV, OUTPUT_TABLE,
        SUPPLEMENTARY_RUN_METADATA,
    ]
    # Also remove per-outcome residual files.
    for outcome in SUPPLEMENTARY_OUTCOMES:
        for method in ["harmonic", "cyclic_spline"]:
            paths.append(_residual_path(outcome, method))
    for p in paths:
        if p.exists():
            p.unlink()
            log(f"Removed: {p}")


def main() -> int:
    args = parse_args()
    if args.project_root:
        os.environ["MAGNETO_ROOT"] = args.project_root
    if args.gpu_chunk_size is not None:
        os.environ["MAGNETO_GPU_CHUNK_SIZE"] = str(args.gpu_chunk_size)

    if args.clean_generated:
        clean_supplementary_generated()

    n_surr = args.n_surrogates
    if n_surr is None:
        n_surr = SURROGATE_SMOKE_N if args.smoke_test else SURROGATE_N

    session_start = datetime.now(timezone.utc)
    if args.run_id:
        analysis_run_id = args.run_id
    else:
        analysis_run_id = current_analysis_run_id()
        if not analysis_run_id:
            analysis_run_id = session_start.strftime("%Y%m%dT%H%M%SZ")
    execution_session_id = session_start.strftime("%Y%m%dT%H%M%SZ")

    input_files = {
        "data/interim/sif_aggregated.feather": PROJECT_ROOT / "data" / "interim" / "sif_aggregated.feather",
        "data/interim/omni_biosphere_features.feather": PROJECT_ROOT / "data" / "interim" / "omni_biosphere_features.feather",
        "data/interim/era5_env_daily.parquet": PROJECT_ROOT / "data" / "interim" / "era5_env_daily.parquet",
        "data/interim/modis_extract.parquet": PROJECT_ROOT / "data" / "interim" / "modis_extract.parquet",
    }
    input_hashes = {k: full_sha256(p) if p.exists() else None for k, p in input_files.items()}
    input_manifest_hash = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()

    run_meta = {
        "analysis_run_id": analysis_run_id,
        "execution_session_id": execution_session_id,
        "run_type": "supplementary_smoke" if args.smoke_test else "supplementary_full",
        "git_commit": _git_commit(),
        "config_hash": config_hash(),
        "input_manifest_hash": input_manifest_hash,
        "n_surrogates": n_surr,
        "stress_source": args.stress_source,
        "generation_timestamp": session_start.isoformat(),
        "outcomes": list(SUPPLEMENTARY_OUTCOMES.keys()),
        "execution_sessions": [{
            "execution_session_id": execution_session_id,
            "started_at": session_start.isoformat(),
            "type": "initial",
        }],
    }
    SUPPLEMENTARY_RUN_METADATA.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_RUN_METADATA.write_text(json.dumps(run_meta, indent=2))
    log(f"Supplementary pipeline started: analysis_run_id={analysis_run_id} "
        f"execution_session_id={execution_session_id} smoke={args.smoke_test} n_surrogates={n_surr}")

    # ── Load QC and membership ────────────────────────────────────────
    log("Loading QC and membership …")
    qc = pd.read_parquet(FILE_QC, engine=PARQUET_ENGINE)
    qc["date"] = pd.to_datetime(qc["date"])
    members = pd.read_parquet(FILE_LAI_MEMBERS, engine=PARQUET_ENGINE)
    qc = qc.merge(members, on=["lat_id", "lon_id"], how="left")

    if args.smoke_test:
        log("Applying smoke subset …")
        spec = importlib.util.spec_from_file_location("smoke_subset", str(SCRIPT_DIR / "02b_smoke_subset.py"))
        smoke_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_mod)
        qc, _ = smoke_mod.smoke_subset_qc(qc, members)
        qc["date"] = pd.to_datetime(qc["date"])
        qc["year"] = qc["date"].dt.year.astype("int32")
        qc["doy"] = qc["date"].dt.dayofyear.astype("int32")

    # ── Build SII exposure ────────────────────────────────────────────
    log("Building SII exposure …")
    sii_win = compute_sii_windows(build_daily_sii(), SII_RAW_COL, SII_WINDOWS)

    # ── Compute residuals for supplementary outcomes ──────────────────
    log("Loading detrending modules …")
    mod_h, mod_c = _load_detrending_modules()

    stress_source = args.stress_source
    old_residuals = _load_old_residuals_for_reuse(args.resume)
    residuals_by_outcome_method: dict[tuple[str, str], pd.DataFrame] = {}
    availability = []
    for outcome in SUPPLEMENTARY_OUTCOMES:
        qc_out = _build_qc_for_outcome(qc, outcome, stress_source=stress_source)
        log(f"Outcome {outcome}: {len(qc_out):,} valid QC rows")
        availability.append({
            "outcome": outcome,
            "outcome_type": SUPPLEMENTARY_OUTCOMES[outcome]["type"],
            "outcome_sample_label": SUPPLEMENTARY_OUTCOMES[outcome]["sample_label"],
            "formula": SUPPLEMENTARY_OUTCOMES[outcome]["formula"],
            "denominator_rule": SUPPLEMENTARY_OUTCOMES[outcome]["denominator_rule"],
            "raw_valid_observations": len(qc_out),
            "raw_valid_cells": qc_out[["lat_id", "lon_id"]].drop_duplicates().shape[0],
            "raw_valid_years": qc_out["year"].nunique(),
        })

        for method, mod in [("harmonic", mod_h), ("cyclic_spline", mod_c)]:
            key = (outcome, method)
            # Reuse numerically unchanged residuals for direct/provider outcomes.
            if args.resume and outcome in ("sif_757nm", "sif_740nm") and key in old_residuals:
                res = old_residuals[key].copy()
                res["outcome_type"] = SUPPLEMENTARY_OUTCOMES[outcome]["type"]
                log(f"Reusing {method} residuals for {outcome} ({len(res):,} rows)")
            else:
                res = _fit_outcome(qc_out, outcome, method, mod_h, mod_c)
                if len(res) == 0:
                    log(f"[WARN] No residuals for {outcome} {method}")
                    continue
                res = _attach_sii_and_metadata(res, sii_win, qc, SUPPLEMENTARY_RUN_METADATA)
            residuals_by_outcome_method[key] = res
            atomic_write(res, _residual_path(outcome, method))

    if not residuals_by_outcome_method:
        log("[FAIL] No residuals produced")
        return 1

    residuals_long = pd.concat(residuals_by_outcome_method.values(), ignore_index=True)
    del residuals_by_outcome_method, old_residuals
    gc.collect()

    # Split by method
    harm_full = residuals_long[residuals_long["method"] == "harmonic"].copy()
    spline_full = residuals_long[residuals_long["method"] == "cyclic_spline"].copy()

    # ── Build common-outcome matched samples ──────────────────────────
    log("Building common-outcome matched samples …")
    required = list(SUPPLEMENTARY_OUTCOMES.keys()) + ["sif_771nm"]

    # Add SIF 771 residuals to long dataset for common-sample construction
    sif771_harm = pd.read_parquet(FILE_RES_HARMONIC, engine=PARQUET_ENGINE)
    sif771_harm = sif771_harm[sif771_harm["pass_flag"] == True].copy() if "pass_flag" in sif771_harm.columns else sif771_harm
    sif771_harm["outcome"] = "sif_771nm"
    sif771_harm["outcome_type"] = "direct"
    sif771_harm["method"] = "harmonic"

    sif771_spline = pd.read_parquet(FILE_RES_CYCLIC, engine=PARQUET_ENGINE)
    sif771_spline = sif771_spline[sif771_spline["fit_status"] == "ok"].copy() if "fit_status" in sif771_spline.columns else sif771_spline
    sif771_spline["outcome"] = "sif_771nm"
    sif771_spline["outcome_type"] = "direct"
    sif771_spline["method"] = "cyclic_spline"

    # In smoke mode the supplementary residuals are restricted to the smoke subset,
    # so the common-outcome matched sample must be built inside the same key space.
    if args.smoke_test:
        smoke_keys = set(zip(qc["date"], qc["lat_id"], qc["lon_id"]))
        sif771_harm = sif771_harm[sif771_harm[["date", "lat_id", "lon_id"]].apply(
            lambda r: (r["date"], r["lat_id"], r["lon_id"]) in smoke_keys, axis=1
        )].copy()
        sif771_spline = sif771_spline[sif771_spline[["date", "lat_id", "lon_id"]].apply(
            lambda r: (r["date"], r["lat_id"], r["lon_id"]) in smoke_keys, axis=1
        )].copy()
        log(f"SIF 771 smoke-restricted keys: harmonic={len(sif771_harm):,}, spline={len(sif771_spline):,}")

    sif771_harm = _attach_sii_and_metadata(sif771_harm, sii_win, qc, SUPPLEMENTARY_RUN_METADATA)
    sif771_spline = _attach_sii_and_metadata(sif771_spline, sii_win, qc, SUPPLEMENTARY_RUN_METADATA)

    log("Valid residuals counts before common sample construction:")
    for method, src in [("harmonic", harm_full), ("cyclic_spline", spline_full)]:
        for out in src["outcome"].unique():
            sub = src[(src["outcome"] == out)]
            n_ok = _valid_residual_mask(sub).sum()
            log(f"  {method} {out}: {len(sub)} rows, {n_ok} ok")

    common_harm = _build_common_sample(pd.concat([harm_full, sif771_harm], ignore_index=True), required)
    common_spline = _build_common_sample(pd.concat([spline_full, sif771_spline], ignore_index=True), required)

    # Verify common-sample integrity
    for label, df_common in [("harmonic", common_harm), ("spline", common_spline)]:
        outcomes_present = df_common["outcome"].unique() if len(df_common) > 0 else []
        log(f"Common {label}: {len(df_common)} rows, outcomes={outcomes_present}")
        assert set(required).issubset(set(outcomes_present)), f"Common {label} sample missing outcomes"
        # Ensure keys are identical across outcomes
        key_sets = []
        for out in required:
            sub = df_common[df_common["outcome"] == out]
            key_sets.append(set(zip(sub["date"], sub["lat_id"], sub["lon_id"])))
        assert all(k == key_sets[0] for k in key_sets), f"Common {label} keys differ across outcomes"

    # Write residual datasets
    log("Writing residual datasets …")
    atomic_write(harm_full, OUTPUT_RESIDUALS_HARMONIC)
    atomic_write(spline_full, OUTPUT_RESIDUALS_SPLINE)
    atomic_write(common_harm, OUTPUT_COMMON_HARMONIC)
    atomic_write(common_spline, OUTPUT_COMMON_SPLINE)

    # ── Compute fixed-window statistics ───────────────────────────────
    log("Computing fixed-window statistics …")
    rows = []
    rows.extend(_compute_stats_for_sample(harm_full, "outcome_specific_full", "harmonic", [f"sii_{w}d" for w in SII_WINDOWS]))
    rows.extend(_compute_stats_for_sample(spline_full, "outcome_specific_full", "cyclic_spline", [f"sii_{w}d" for w in SII_WINDOWS]))
    rows.extend(_compute_stats_for_sample(common_harm, "common_outcome_matched", "harmonic", [f"sii_{w}d" for w in SII_WINDOWS]))
    rows.extend(_compute_stats_for_sample(common_spline, "common_outcome_matched", "cyclic_spline", [f"sii_{w}d" for w in SII_WINDOWS]))

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        log("[FAIL] No fixed-window statistics computed")
        return 1

    # Prepare pooled surrogate input early so we can release the large residuals_long
    # frame before availability/diagnostics work.
    pooled_full = residuals_long.copy()
    pooled_full["sample_type"] = "outcome_specific_full"
    del residuals_long
    gc.collect()
    _clear_gpu_cache()

    # Add outcome / outcome_type columns from residual_column? Actually rows have 'outcome' in group key.
    # Rename residual_column to outcome for clarity
    results_df = results_df.drop(columns=["residual_column"], errors="ignore")
    results_df["outcome_type"] = results_df["outcome"].map(
        lambda x: SUPPLEMENTARY_OUTCOMES[x]["type"] if x in SUPPLEMENTARY_OUTCOMES else "direct")
    results_df["denominator_rule"] = results_df["outcome"].map(
        lambda x: SUPPLEMENTARY_OUTCOMES[x]["denominator_rule"] if x in SUPPLEMENTARY_OUTCOMES else "none")
    results_df["outcome_sample_label"] = results_df.apply(
        lambda r: SUPPLEMENTARY_OUTCOMES[r["outcome"]]["sample_label"]
        if r["outcome"] in SUPPLEMENTARY_OUTCOMES and r["sample_type"] == "outcome_specific_full"
        else r["sample_type"],
        axis=1,
    )

    # Import primary SIF 771 outcome-specific-full results
    main_fixed, main_surr = _import_main_sif771_pooled()
    # Align columns
    for col in results_df.columns:
        if col not in main_fixed.columns:
            main_fixed[col] = np.nan
    main_fixed = main_fixed[results_df.columns]
    results_df = pd.concat([results_df, main_fixed], ignore_index=True)

    # Add provenance
    prov = run_provenance_dict(producing_stage="run_supplementary_outcomes.py",
                               metadata_path=SUPPLEMENTARY_RUN_METADATA)
    for k, v in prov.items():
        results_df[k] = v

    # Results are written after surrogate reuse so both tables share one run_id.
    results_rows = results_df.to_dict("records")
    _old_outputs = _load_old_results_for_reuse(args.resume)
    old_results_for_reuse = _old_outputs.get("results")
    old_surr_summary_for_reuse = _old_outputs.get("surr_summary")
    old_surr_null_for_reuse = _old_outputs.get("surr_null")

    # ── Availability diagnostics ──────────────────────────────────────
    avail = pd.DataFrame(availability)
    for method in ["harmonic", "cyclic_spline"]:
        col_valid = f"valid_{method}_observations"
        col_cells = f"valid_{method}_cells"
        avail[col_valid] = 0
        avail[col_cells] = 0
        src = harm_full if method == "harmonic" else spline_full
        for i, out in enumerate(avail["outcome"]):
            sub = src[src["outcome"] == out]
            valid_sub = sub[_valid_residual_mask(sub)]
            avail.at[i, col_valid] = len(valid_sub)
            avail.at[i, col_cells] = valid_sub[["lat_id", "lon_id"]].drop_duplicates().shape[0]

    # Common sample sizes
    avail["common_harmonic_observations"] = 0
    avail["common_spline_observations"] = 0
    for out in avail["outcome"]:
        avail.loc[avail["outcome"] == out, "common_harmonic_observations"] = len(common_harm[common_harm["outcome"] == out])
        avail.loc[avail["outcome"] == out, "common_spline_observations"] = len(common_spline[common_spline["outcome"] == out])

    # Overlap with SIF 771
    sif771_keys = set(zip(sif771_harm["date"], sif771_harm["lat_id"], sif771_harm["lon_id"]))
    avail["overlap_with_sif771_keys"] = 0
    avail["fraction_retained_in_common_sample"] = np.nan
    for i, out in enumerate(avail["outcome"]):
        sub = _build_qc_for_outcome(qc, out, stress_source=stress_source)
        out_keys = set(zip(sub["date"], sub["lat_id"], sub["lon_id"]))
        overlap = len(out_keys & sif771_keys)
        avail.at[i, "overlap_with_sif771_keys"] = overlap
        common_keys_harm = set(zip(common_harm[common_harm["outcome"] == out]["date"],
                                   common_harm[common_harm["outcome"] == out]["lat_id"],
                                   common_harm[common_harm["outcome"] == out]["lon_id"]))
        avail.at[i, "fraction_retained_in_common_sample"] = len(common_keys_harm) / len(out_keys) if out_keys else np.nan

    for k, v in prov.items():
        avail[k] = v
    atomic_write(avail, OUTPUT_AVAILABILITY)
    log(f"Wrote {OUTPUT_AVAILABILITY}")

    # Common sample diagnostics
    common_diag = []
    for label, df_common in [("harmonic", common_harm), ("spline", common_spline)]:
        for out in required:
            sub = df_common[df_common["outcome"] == out]
            out_type = "direct" if out == "sif_771nm" else SUPPLEMENTARY_OUTCOMES[out]["type"]
            out_rule = "none" if out == "sif_771nm" else SUPPLEMENTARY_OUTCOMES[out]["denominator_rule"]
            common_diag.append({
                "method": label,
                "outcome": out,
                "outcome_type": out_type,
                "denominator_rule": out_rule,
                "n_obs": len(sub),
                "n_cells": sub[["lat_id", "lon_id"]].drop_duplicates().shape[0],
                "n_years": sub["year"].nunique(),
                "obs_hash": _observation_key_hash(sub),
                "mean_residual": float(sub["residual"].mean()),
                "residual_sd": float(sub["residual"].std()),
            })
    common_diag_df = pd.DataFrame(common_diag)
    for k, v in prov.items():
        common_diag_df[k] = v
    atomic_write(common_diag_df, OUTPUT_COMMON_DIAGNOSTICS)
    log(f"Wrote {OUTPUT_COMMON_DIAGNOSTICS}")

    # Free frames that are no longer needed before the memory-heavy surrogate stage.
    del harm_full, spline_full, sif771_harm, sif771_spline, qc, members, sii_win
    gc.collect()
    _clear_gpu_cache()

    # ── Surrogates ────────────────────────────────────────────────────
    log("Running temporal surrogates …")
    # Surrogates only for pooled outcome-specific full and common matched samples.
    # Build strata directly from the three source frames to avoid a large
    # concatenated surrogate_input DataFrame that has been pushing WSL memory.
    common_harm["sample_type"] = "common_outcome_matched"
    common_spline["sample_type"] = "common_outcome_matched"

    # Keep only columns required for surrogate strata to reduce memory footprint.
    surrogate_keep_cols = ["date", "lat_id", "lon_id", "year", "outcome",
                           "sample_type", "method", "residual",
                           f"sii_{SII_WINDOWS[0]}d", f"sii_{SII_WINDOWS[1]}d"]
    surrogate_keep_cols = [c for c in surrogate_keep_cols
                           if c in pooled_full.columns]
    pooled_full = pooled_full[surrogate_keep_cols].copy()
    common_harm = common_harm[surrogate_keep_cols].copy()
    common_spline = common_spline[surrogate_keep_cols].copy()
    gc.collect()

    all_strata = []
    for src, stype in [
        (pooled_full, "outcome_specific_full"),
        (common_harm, "common_outcome_matched"),
        (common_spline, "common_outcome_matched"),
    ]:
        for method, df_sub in src.groupby("method", sort=True):
            all_strata.extend(build_strata(df_sub, stype, method, "residual",
                                           group_cols=["outcome"], windows=SII_WINDOWS))
        # Release the copy as soon as strata are built; the list still holds the
        # references it needs until run_surrogates finishes.
        del src
        gc.collect()
        _clear_gpu_cache()

    log(f"Built {len(all_strata)} strata for surrogate analysis")
    df_surr_summary, df_surr_null = run_surrogates(all_strata, n_surr=n_surr, seed=SURROGATE_SEED,
                                                   modes=SURROGATE_MODES)

    # Import primary SIF 771 outcome-specific-full surrogates
    main_surr_fixed = main_surr
    for col in df_surr_summary.columns:
        if col not in main_surr_fixed.columns:
            main_surr_fixed[col] = np.nan
    main_surr_fixed = main_surr_fixed[df_surr_summary.columns]
    df_surr_summary = pd.concat([df_surr_summary, main_surr_fixed], ignore_index=True)

    for k, v in prov.items():
        df_surr_summary[k] = v
        df_surr_null[k] = v

    # Reuse numerically unchanged SIF 757/740 outcome-specific rows.
    surr_summary_rows = df_surr_summary.to_dict("records")
    surr_null_rows = df_surr_null.to_dict("records")
    results_rows, surr_summary_rows, surr_null_rows = _reuse_unchanged_outcome_specific(
        results_rows, surr_summary_rows, surr_null_rows,
        old_results_for_reuse, old_surr_summary_for_reuse, old_surr_null_for_reuse,
        analysis_run_id, execution_session_id, prov)

    results_df = pd.DataFrame(results_rows)
    df_surr_summary = pd.DataFrame(surr_summary_rows)
    df_surr_null = pd.DataFrame(surr_null_rows)

    # Ensure SIF 771 outcome-specific-full rows match the final primary run.
    df_surr_summary, df_surr_null = _replace_sif771_from_main(df_surr_summary, df_surr_null)

    atomic_write(results_df, OUTPUT_RESULTS)
    log(f"Wrote {OUTPUT_RESULTS} ({len(results_df)} rows)")

    atomic_write(df_surr_summary, OUTPUT_SURROGATE_SUMMARY)
    atomic_write(df_surr_null, OUTPUT_SURROGATE_NULL)
    log(f"Wrote {OUTPUT_SURROGATE_SUMMARY} ({len(df_surr_summary)} rows)")
    log(f"Wrote {OUTPUT_SURROGATE_NULL} ({len(df_surr_null)} rows)")

    # ── Supplementary table for the article ───────────────────────────
    _write_table(results_df, df_surr_summary)

    # ── Manifests ─────────────────────────────────────────────────────
    _write_manifests(analysis_run_id, run_meta)
    _save_run_state(run_meta, args.smoke_test)

    log("Supplementary pipeline complete")
    return 0


def _reuse_unchanged_outcome_specific(rows: list, surr_summary_rows: list, surr_null_rows: list,
                                        old_results: pd.DataFrame | None,
                                        old_surr_summary: pd.DataFrame | None,
                                        old_surr_null: pd.DataFrame | None,
                                        analysis_run_id: str, execution_session_id: str,
                                        prov: dict) -> tuple:
    """Reuse only observed fixed-window results for SIF 757/740.

    The QC, detrending, and SII exposure for these direct/provider outcomes are
    unchanged, so the observed correlations/effects can safely be reused.
    Surrogate summaries and null distributions are NEVER reused: the surrogate
    engine is under active development and old null distributions are not
    numerically equivalent to freshly computed ones.
    """
    _ = old_surr_summary, old_surr_null  # deliberately ignored
    if old_results is None:
        return rows, surr_summary_rows, surr_null_rows

    outcomes_to_reuse = {"sif_757nm", "sif_740nm"}
    sample_type_to_reuse = "outcome_specific_full"

    def _update_meta(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for k, v in prov.items():
            df[k] = v
        df["analysis_run_id"] = analysis_run_id
        df["execution_session_id"] = execution_session_id
        df["run_type"] = "supplementary_full"
        df["outcome_type"] = df["outcome"].map(
            lambda x: SUPPLEMENTARY_OUTCOMES[x]["type"] if x in SUPPLEMENTARY_OUTCOMES else "direct")
        df["denominator_rule"] = "none"
        df["outcome_sample_label"] = "outcome_specific_full"
        return df

    # Results rows (observed fixed-window statistics only)
    old_res = old_results[
        old_results["outcome"].isin(outcomes_to_reuse) &
        (old_results["sample_type"] == sample_type_to_reuse)
    ].copy()
    if not old_res.empty:
        old_res = _update_meta(old_res)
        rows_df = pd.DataFrame(rows)
        rows_df = rows_df[
            ~(rows_df["outcome"].isin(outcomes_to_reuse) &
              (rows_df["sample_type"] == sample_type_to_reuse))
        ]
        rows_df = pd.concat([rows_df, old_res], ignore_index=True)
        rows = rows_df.to_dict("records")

    # Do NOT reuse surrogate summary or null rows.
    return rows, surr_summary_rows, surr_null_rows


def _write_table(results_df: pd.DataFrame, surr_df: pd.DataFrame) -> None:
    """Create a compact table for the supplementary materials."""
    # Pivot surrogate p-values: rows = outcome × sample_type, columns = method × window × mode
    surr_p = surr_df.pivot_table(
        index=["outcome", "sample_type"],
        columns=["method", "sii_window", "surrogate_mode"],
        values="p_value",
        aggfunc="first",
    ).reset_index()

    # Flatten MultiIndex columns
    flat_cols = []
    for col in surr_p.columns.values:
        if isinstance(col, tuple):
            flat = "_".join(str(c) for c in col if pd.notna(c) and str(c) not in ["", "nan"]).strip("_")
            flat_cols.append(flat)
        else:
            flat_cols.append(col)
    surr_p.columns = flat_cols

    rows = []
    for _, r in results_df.iterrows():
        row = {
            "outcome": r["outcome"],
            "outcome_type": r["outcome_type"],
            "outcome_sample_label": r["outcome_sample_label"],
            "sample_type": r["sample_type"],
            "denominator_rule": r["denominator_rule"],
            "method": r["method"],
            "sii_window_days": r["sii_window_days"],
            "rho": r["spearman_rho"],
            "effect_per_1sd_sii": r["std_effect_per_1sd_sii"],
            "n_obs": r["n_obs"],
            "n_cells": r["n_cells"],
        }
        for mode in SURROGATE_MODES:
            pcol = f"{r['method']}_{r['sii_window']}_{mode}"
            sp = surr_p[(surr_p["outcome"] == r["outcome"]) & (surr_p["sample_type"] == r["sample_type"])]
            if not sp.empty and pcol in sp.columns:
                row[f"p_{mode}"] = sp.iloc[0][pcol]
            else:
                row[f"p_{mode}"] = np.nan
        rows.append(row)
    table_df = pd.DataFrame(rows)
    prov = run_provenance_dict(producing_stage="run_supplementary_outcomes.py",
                               metadata_path=SUPPLEMENTARY_RUN_METADATA)
    for k, v in prov.items():
        table_df[k] = v
    atomic_write(table_df, OUTPUT_TABLE)
    log(f"Wrote {OUTPUT_TABLE}")


def _write_manifests(run_id: str, run_meta: dict) -> None:
    files = [
        OUTPUT_RESIDUALS_HARMONIC, OUTPUT_RESIDUALS_SPLINE,
        OUTPUT_COMMON_HARMONIC, OUTPUT_COMMON_SPLINE,
        OUTPUT_RESULTS, OUTPUT_SURROGATE_SUMMARY, OUTPUT_SURROGATE_NULL,
        OUTPUT_AVAILABILITY, OUTPUT_COMMON_DIAGNOSTICS, OUTPUT_TABLE,
        SUPPLEMENTARY_RUN_METADATA,
        PROJECT_ROOT / "results" / "supplementary_run_state.json",
        PROJECT_ROOT / "results" / "sif_740_provenance_check.csv",
        PROJECT_ROOT / "results" / "stress_indicator_availability_audit.csv",
        PROJECT_ROOT / "results" / "stress_denominator_diagnostics.csv",
        PROJECT_ROOT / "reports" / "SIF_740_PROVENANCE_AUDIT.md",
        PROJECT_ROOT / "reports" / "STRESS_INDICATOR_PROVENANCE_AUDIT.md",
        PROJECT_ROOT / "reports" / "STRESS_DENOMINATOR_STABILITY.md",
    ]
    # Per-outcome residual files
    for outcome in SUPPLEMENTARY_OUTCOMES:
        for method in ["harmonic", "cyclic_spline"]:
            files.append(_residual_path(outcome, method))
    manifest = []
    for f in files:
        if f.exists():
            manifest.append({
                "path": str(f.relative_to(PROJECT_ROOT)),
                "size_bytes": f.stat().st_size,
                "sha256": full_sha256(f),
            })
    manifest.append({
        "path": "config/pipeline.yaml",
        "size_bytes": (PROJECT_ROOT / "config" / "pipeline.yaml").stat().st_size,
        "sha256": config_hash(),
    })
    atomic_write(manifest, OUTPUT_MANIFEST_JSON)
    pd.DataFrame(manifest).to_csv(OUTPUT_MANIFEST_CSV, index=False)
    log(f"Wrote {OUTPUT_MANIFEST_JSON} ({len(manifest)} files)")


def _save_run_state(run_meta: dict, smoke: bool) -> None:
    """Persist lightweight hash state to enable future resume decisions."""
    state = {
        "analysis_run_id": run_meta["analysis_run_id"],
        "run_type": run_meta["run_type"],
        "smoke": smoke,
        "qc_hash": full_sha256(FILE_QC),
        "config_hash": run_meta["config_hash"],
        "runner_hash": _runner_hash(),
        "outcome_def_hash": _outcome_def_hash(),
        "stress_source": run_meta["stress_source"],
        "git_commit": run_meta["git_commit"],
        "residuals": {},
    }
    for outcome in SUPPLEMENTARY_OUTCOMES:
        for method in ["harmonic", "cyclic_spline"]:
            p = _residual_path(outcome, method)
            if p.exists():
                state["residuals"][f"{outcome}_{method}"] = {
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "sha256": full_sha256(p),
                }
    state_path = PROJECT_ROOT / "results" / "supplementary_run_state.json"
    atomic_write(state, state_path)
    log(f"Wrote {state_path}")


if __name__ == "__main__":
    sys.exit(main())
