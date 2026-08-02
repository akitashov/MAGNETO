#!/usr/bin/env python3
"""
MAGNETO Unified Pipeline Runner.

Single entry point for smoke and full pipeline runs. Enforces a fixed stage
order, validates checkpoints on resume, and writes per-stage manifests.
"""
from __future__ import annotations
import sys, os, subprocess, shutil, argparse, time, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from _Common import (
    PROJECT_ROOT, config_hash, full_sha256, atomic_write, RUN_METADATA_PATH
)

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


def _find_python() -> str:
    env_py = os.environ.get("MAGNETO_PYTHON")
    if env_py:
        return env_py
    for base in [os.environ.get("CONDA_PREFIX", ""), os.path.expanduser("~/miniconda3")]:
        if base:
            candidate = os.path.join(base, "envs", "magneto_gpu", "bin", "python3")
            if os.path.isfile(candidate):
                return candidate
            candidate = os.path.join(base, "bin", "python3")
            if os.path.isfile(candidate):
                return candidate
    return "python3"


PYTHON = _find_python()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MANIFEST_DIR = PROJECT_ROOT / "results" / "manifests"

ANALYSIS_RUN_ID = RUN_ID
EXECUTION_SESSION_ID = RUN_ID
EXECUTION_SESSIONS = []

# Stage order is fixed and dependency-respecting.
STAGES = [
    (1,  "preflight",            "00_preflight.py",            [],                                    []),
    (1.2,"modis_etl",            "etl/E01_modis_etl.py",            ["data/raw/MODIS"],
                                                                  ["data/interim/modis_extract.parquet"]),
    (1.3,"omni_etl",             "etl/E02_omni_etl.py",            ["data/raw/omni2_all_years.zip"],
                                                                  ["data/interim/omni_biosphere_features.feather"]),
    (1.5,"sif_etl",              "etl/E03_sif_etl.py",             ["data/raw/OCO2",
                                                                  "data/interim/modis_extract.parquet"],
                                                                  ["data/interim/sif_aggregated.feather"]),
    (1.6,"hdf2netcdf",           "etl/E04_mcd12c1_hdf2netcdf.py",  ["data/raw/MCD12C1/2022/001/*.hdf"],
                                                                  ["data/interim/MCD12C1*.nc"]),
    (1.7,"era5_etl",             "etl/E05_era5_etl.py",            ["data/raw/ERA5"],
                                                                  ["data/interim/era5_env_daily.parquet"]),
    (1.8,"etl_equivalence_audit","etl/E06_etl_equivalence_audit.py",[],
                                                                  ["reports/ETL_EQUIVALENCE_AUDIT.md"]),
    (1.85,"etl_chunked_audit",  "etl/E07_etl_chunked_audit.py",[],
                                                                  ["reports/ETL_CHUNKED_AUDIT.md"]),
    (1.9,"etl_deep_equivalence_audit","etl/E08_etl_deep_equivalence_audit.py",[],
                                                                  ["reports/ETL_DEEP_AUDIT.md"]),
    (2,  "qc",                   "01_build_qc.py",             ["data/interim/sif_aggregated.feather",
                                                             "data/interim/modis_extract.parquet"],
                                                             ["data/interim/global_qc.parquet",
                                                              "results/qc_flow.csv"]),
    (3,  "lai_cells_controls",   "02_assign_lai_quartiles.py", ["data/interim/modis_extract.parquet"],
                                                             ["data/interim/lai_cell_summary.parquet",
                                                              "data/interim/lai_quartile_boundaries.json",
                                                              "data/interim/lai_quartile_membership.parquet",
                                                              "data/interim/control_membership.parquet",
                                                              "results/control_definitions.json"]),
    (4,  "era5_join",            "03_era5_join.py",            ["data/interim/global_qc.parquet",
                                                             "data/interim/era5_env_daily.parquet"],
                                                             ["data/interim/global_qc.parquet"]),
    (4.5,"smoke_subset",         "02b_smoke_subset.py",        ["data/interim/global_qc.parquet",
                                                             "data/interim/lai_quartile_membership.parquet"],
                                                             ["data/interim/global_qc.parquet"]),
    (5,  "harmonic",             "04_harmonic.py",             ["data/interim/global_qc.parquet",
                                                             "data/interim/lai_quartile_membership.parquet"],
                                                             ["data/processed/harmonic_analysis.parquet"]),
    (6,  "cyclic_spline",        "05_cyclic_spline.py",        ["data/interim/global_qc.parquet",
                                                             "data/interim/lai_quartile_membership.parquet"],
                                                             ["data/processed/cyclic_spline_analysis.parquet"]),
    (7,  "matched",              "07_matched.py",              ["data/processed/harmonic_analysis.parquet",
                                                             "data/processed/cyclic_spline_analysis.parquet"],
                                                             ["data/processed/harmonic_spline_matched.parquet"]),
    (8,  "fixed_window",         "08_fixed_window.py",         ["data/processed/harmonic_analysis.parquet",
                                                             "data/processed/cyclic_spline_analysis.parquet",
                                                             "data/processed/harmonic_spline_matched.parquet",
                                                             "data/interim/omni_biosphere_features.feather"],
                                                             ["results/fixed_window_results.csv"]),
    (9,  "surrogates",           "09_surrogates.py",           ["data/processed/harmonic_analysis.parquet",
                                                             "data/processed/cyclic_spline_analysis.parquet",
                                                             "data/processed/harmonic_spline_matched.parquet",
                                                             "data/interim/omni_biosphere_features.feather"],
                                                             ["results/surrogate_summary.csv",
                                                              "results/surrogate_null_distributions.parquet"]),
    (9.1, "environmental_driver_input", "10_prepare_environmental_driver_input.py",
                                                             ["data/processed/harmonic_analysis.parquet",
                                                              "data/interim/era5_env_daily.parquet",
                                                              "data/interim/omni_biosphere_features.feather"],
                                                             ["data/processed/environmental_driver_input.parquet"]),
    (9.2, "environmental_driver_matrix", "11_environmental_driver_matrix_gpu.py",
                                                             ["data/processed/environmental_driver_input.parquet"],
                                                             ["results/environmental_driver_matrix.parquet",
                                                              "results/environmental_driver_matrix_audit.json"]),
    (9.3, "environmental_driver_summary", "12_environmental_driver_matrix_summary.py",
                                                             ["results/environmental_driver_matrix.parquet"],
                                                             ["results/environmental_driver_matrix_summary.csv"]),
    (10, "effects",              "10_effects.py",              ["results/fixed_window_results.csv",
                                                             "data/processed/harmonic_spline_matched.parquet"],
                                                             ["results/effect_sizes.csv"]),
    (11, "report",               "reports/11_report.py",       [],
                                                             ["reports/pipeline_report.md",
                                                              "results/output_manifest.json"]),
    (12, "audit_reports",        "reports/12_audit_reports.py",[],
                                                             ["reports/QC_AUDIT.md",
                                                              "reports/SIF_QC_AUDIT.md",
                                                              "reports/LAI_QUARTILE_AUDIT.md",
                                                              "reports/FUNCTIONAL_CONTROL_AUDIT.md",
                                                              "reports/GEOGRAPHIC_CONTROL_AUDIT.md",
                                                              "reports/ERA5_AUDIT.md",
                                                              "reports/SII_WINDOW_AUDIT.md",
                                                              "reports/CYCLIC_SPLINE_VALIDATION.md",
                                                              "reports/DOY_FEASIBILITY_REPORT.md",
                                                              "reports/SURROGATE_VALIDATION.md",
                                                              "reports/OUTPUT_INTEGRITY_REPORT.md",
                                                              "reports/METHODOLOGY.md",
                                                              "reports/ARTICLE_REVISION_SUMMARY.md"]),
    (13, "validate",             None,                         [],
                                                             ["results/fixed_window_results.csv",
                                                              "results/surrogate_summary.csv",
                                                              "results/output_manifest.json",
                                                              "reports/pipeline_report.md"]),
    (14, "supplementary_checks", "supplementary_checks/run_supplementary_checks.sh",
                                                             ["results/fixed_window_results.csv",
                                                              "results/surrogate_summary.csv"],
                                                             ["results/supplementary_checks/supplementary_checks_manifest.json"]),
]


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {msg}", flush=True)


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p


def _file_hashes(paths: list[str]) -> dict[str, str | None]:
    out = {}
    for p in paths:
        rp = _resolve(p)
        out[p] = full_sha256(rp) if rp.exists() else None
    return out


def _stage_manifest_path(stage_name: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return MANIFEST_DIR / f"{stage_name}.json"


def _load_manifest(stage_name: str) -> dict | None:
    p = _stage_manifest_path(stage_name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save_manifest(stage_name: str, script: str | None, inputs: list[str], outputs: list[str],
                   extra: dict | None = None, rc: int = 0) -> None:
    out_hashes = _file_hashes(outputs)
    manifest = {
        "run_id": RUN_ID,
        "analysis_run_id": ANALYSIS_RUN_ID,
        "execution_session_id": EXECUTION_SESSION_ID,
        "stage": stage_name,
        "script": script,
        "config_hash": config_hash(),
        "code_hash": full_sha256(SCRIPT_DIR / script) if script else None,
        "input_hashes": _file_hashes(inputs),
        "output_hashes": out_hashes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": rc,
    }
    if extra:
        manifest.update(extra)
    atomic_write(manifest, _stage_manifest_path(stage_name), fmt="json")


def _checkpoint_valid(stage_name: str, script: str | None, inputs: list[str],
                      n_surrogates: int | None) -> bool:
    """Resume only if config, code, inputs, run parameters and analysis run are unchanged."""
    m = _load_manifest(stage_name)
    if m is None:
        return False
    # A checkpoint belongs to the current analysis run.
    if m.get("analysis_run_id") and m.get("analysis_run_id") != ANALYSIS_RUN_ID:
        return False
    if m.get("config_hash") != config_hash():
        return False
    if script and m.get("code_hash") != full_sha256(SCRIPT_DIR / script):
        return False
    cur_inputs = _file_hashes(inputs)
    if cur_inputs != m.get("input_hashes"):
        return False
    # Surrogate checkpoint is only valid for the same N.
    if stage_name == "surrogates" and n_surrogates is not None:
        if m.get("n_surrogates") != n_surrogates:
            return False
    # All declared outputs must exist.
    for p in m.get("output_hashes", {}):
        if not _resolve(p).exists():
            return False
    return True


def _run_script(script_name: str, extra_args: list) -> int:
    script_path = SCRIPT_DIR / script_name
    if str(script_path).endswith(".sh"):
        cmd = ["bash", str(script_path)] + extra_args
    else:
        cmd = [PYTHON, "-u", str(script_path)] + extra_args
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def clean_generated():
    targets = [
        "data/interim/global_qc.parquet",
        "data/interim/lai_cell_summary.parquet",
        "data/interim/lai_quartile_boundaries.json",
        "data/interim/lai_quartile_membership.parquet",
        "data/interim/control_membership.parquet",
        "data/processed/harmonic_analysis.parquet",
        "data/processed/cyclic_spline_analysis.parquet",
        "data/processed/harmonic_spline_matched.parquet",
        "results/",
        "reports/pipeline_report.md",
        "logs/pipeline.log",
        "results/manifests/",
    ]
    for t in targets:
        p = PROJECT_ROOT / t
        if p.is_dir():
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink()
                    log(f"Removed: {f}")
        elif p.exists():
            p.unlink()
            log(f"Removed: {p}")


def main():
    global ANALYSIS_RUN_ID, EXECUTION_SESSION_ID, EXECUTION_SESSIONS

    parser = argparse.ArgumentParser(description="MAGNETO Pipeline Runner")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--stop-after-stage", type=int, default=999)
    parser.add_argument("--n-surrogates", type=int, default=None)
    parser.add_argument("--clean-generated", action="store_true")
    parser.add_argument(
        "--environmental-smoke",
        action="store_true",
        help="Run the environmental-driver matrix in smoke mode (tiny grid).",
    )
    parser.add_argument(
        "--environmental-full",
        action="store_true",
        help="Run the full environmental-driver matrix (SII/PAR/VPD 1-28).",
    )
    parser.add_argument(
        "--run-etl-chunked-audit",
        action="store_true",
        help="Run the memory-safe chunked ETL audit (E07).",
    )
    parser.add_argument(
        "--run-etl-deep-audit",
        action="store_true",
        help="Run the deep ETL equivalence audit (E08).",
    )
    parser.add_argument(
        "--save-etl-reference",
        action="store_true",
        help="Pass --save-reference to the deep ETL audit (E08).",
    )
    args = parser.parse_args()

    if args.clean_generated:
        clean_generated()

    n_surr = args.n_surrogates
    smoke = args.smoke_test
    if n_surr is None:
        n_surr = 10 if smoke else 1000

    if args.environmental_smoke and args.environmental_full:
        log("ERROR: --environmental-smoke and --environmental-full are mutually exclusive")
        return 1
    env_matrix_mode = "smoke" if args.environmental_smoke else ("full" if args.environmental_full else None)

    def _stage_skip_reason(name: str) -> str | None:
        """Return the reason a stage is disabled, or None if it should run."""
        if name in ("environmental_driver_matrix", "environmental_driver_summary"):
            if env_matrix_mode is None:
                return "no --environmental-smoke/--environmental-full flag"
        if name == "etl_chunked_audit" and not args.run_etl_chunked_audit:
            return "pass --run-etl-chunked-audit to enable"
        if name == "etl_deep_equivalence_audit" and not args.run_etl_deep_audit:
            return "pass --run-etl-deep-audit to enable"
        return None

    # ── Resolve analysis / execution session identifiers ──────────────
    session_start = datetime.now(timezone.utc)
    if args.resume and RUN_METADATA_PATH.exists():
        prior = json.loads(RUN_METADATA_PATH.read_text())
        ANALYSIS_RUN_ID = prior.get("analysis_run_id", RUN_ID)
        EXECUTION_SESSION_ID = RUN_ID
        EXECUTION_SESSIONS = prior.get("execution_sessions", [])
        EXECUTION_SESSIONS.append({
            "execution_session_id": EXECUTION_SESSION_ID,
            "started_at": session_start.isoformat(),
            "type": "resume",
            "continues_analysis_run_id": ANALYSIS_RUN_ID,
        })
        log(f"Resuming analysis_run_id={ANALYSIS_RUN_ID} with new execution_session_id={EXECUTION_SESSION_ID}")
    else:
        ANALYSIS_RUN_ID = RUN_ID
        EXECUTION_SESSION_ID = RUN_ID
        EXECUTION_SESSIONS = [{
            "execution_session_id": EXECUTION_SESSION_ID,
            "started_at": session_start.isoformat(),
            "type": "initial",
        }]

    # ── Write run metadata for downstream provenance ──────────────────
    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
        except Exception:
            return ""

    input_hashes = _file_hashes([
        "data/interim/sif_aggregated.feather",
        "data/interim/omni_biosphere_features.feather",
        "data/interim/era5_env_daily.parquet",
        "data/interim/modis_extract.parquet",
    ])
    input_manifest_hash = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True).encode()
    ).hexdigest()

    run_meta = {
        "analysis_run_id": ANALYSIS_RUN_ID,
        "execution_session_id": EXECUTION_SESSION_ID,
        "run_type": "smoke" if smoke else "full",
        "git_commit": _git_commit(),
        "config_hash": config_hash(),
        "input_manifest_hash": input_manifest_hash,
        "n_surrogates": n_surr,
        "generation_timestamp": session_start.isoformat(),
        "execution_sessions": EXECUTION_SESSIONS,
    }
    RUN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_METADATA_PATH.write_text(json.dumps(run_meta, indent=2))

    log(f"MAGNETO Pipeline started  analysis_run_id={ANALYSIS_RUN_ID}  execution_session_id={EXECUTION_SESSION_ID}  smoke={smoke}  n_surrogates={n_surr}")

    if args.dry_run:
        log("DRY RUN — pipeline plan:")
        for sn, name, script, inputs, outputs in STAGES:
            exists = _checkpoint_valid(name, script, inputs, n_surr) if args.resume else False
            reason = _stage_skip_reason(name)
            status = "skip" if (sn < args.start_stage or sn > args.stop_after_stage or reason) else "run"
            if exists:
                marker = "[VALID]"
            elif reason:
                marker = f"[{reason}]"
            else:
                marker = "[will run]"
            print(f"  [{status:>4}] {sn:>2}. {name:<22} {marker:<10}")
        return 0

    def stage_args(name):
        args = []
        if name in ("qc", "smoke_subset", "surrogates", "supplementary_checks") and smoke:
            args.append("--smoke-test")
        if name == "surrogates" and n_surr is not None:
            args.append(f"--n-surrogates={n_surr}")
        if name == "environmental_driver_matrix" and env_matrix_mode == "smoke":
            args.append("--smoke")
        if name == "etl_deep_equivalence_audit" and args.save_etl_reference:
            args.append("--save-reference")
        return args

    for sn, name, script, inputs, outputs in STAGES:
        if sn < args.start_stage:
            continue
        if sn > args.stop_after_stage:
            log(f"Stop after stage {args.stop_after_stage} reached")
            break

        skip_reason = _stage_skip_reason(name)
        if skip_reason:
            log(f"SKIP {name}: {skip_reason}")
            continue

        if args.resume and _checkpoint_valid(name, script, inputs, n_surr):
            log(f"SKIP {name}: valid checkpoint exists")
            continue

        log(f"STAGE {sn}: {name}")

        if script is None:
            if name == "validate":
                missing = [r for r in outputs if not _resolve(r).exists()]
                if missing:
                    log(f"VALIDATION FAILED — missing: {missing}")
                    return 1
                log("VALIDATION OK")
            continue

        rc = _run_script(script, stage_args(name))
        extra = {"n_surrogates": n_surr} if name == "surrogates" else {}
        if name == "environmental_driver_matrix" and env_matrix_mode is not None:
            extra["environmental_matrix_mode"] = env_matrix_mode
        _save_manifest(name, script, inputs, outputs, extra=extra, rc=rc)

        if rc != 0:
            log(f"FAILED: {name} (rc={rc}) — aborting")
            return rc

    log("Pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
