#!/usr/bin/env bash
# MAGNETO unified pipeline launcher.
#
# Usage:
#   bash pipeline.sh
#
# Stages can be skipped by commenting out the corresponding RUN_*=1 line at the
# top of this file. Logs are written to logs/<stage>.log with line-flushed
# pseudo-terminal capture so tqdm progress bars are visible.
#
# Requirements:
#   - conda environment "magneto_gpu" exists
#   - executed from the repository root

set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration: set to 1 to enable a stage ───────────────────────────────
RUN_PREFLIGHT=1
RUN_MODIS_ETL=0
RUN_OMNI_ETL=0
RUN_ERA5_ETL=0
RUN_SIF_ETL=0
RUN_QC=1
RUN_HDF2NETCDF=1
RUN_LAI_CELLS_CONTROLS=1
RUN_ERA5_JOIN=1
# RUN_SMOKE_SUBSET=0        # uncomment only for smoke testing
RUN_HARMONIC=1
RUN_CYCLIC_SPLINE=1
RUN_MATCHED=1
RUN_FIXED_WINDOW=1
RUN_SURROGATES=1
RUN_ENV_DRIVER_PREPARE=1
# Set to 1 to run the GPU matrix. The full 1-28 day grid is expensive;
# use RUN_SMOKE_SUBSET=1 together with RUN_ENV_DRIVER_MATRIX=1 for a tiny grid.
RUN_ENV_DRIVER_MATRIX=0
RUN_ENV_DRIVER_SUMMARY=0
RUN_EFFECTS=1
RUN_MAIN_REPORTS=1
RUN_SUPPLEMENTARY=1
RUN_SUPPLEMENTARY_REPORTS=1
RUN_SUPPLEMENTARY_CHECKS=1
RUN_ARCHIVES=1

# Optional ETL audits. E06 runs automatically after upstream ETL stages.
RUN_ETL_CHUNKED_AUDIT=0
RUN_ETL_DEEP_AUDIT=0

# ── Global parameters ───────────────────────────────────────────────────────
# Override N_SURROGATES to run a smoke test (e.g. 10) without editing the runner.
N_SURROGATES="${N_SURROGATES:-1000}"
SUP_N_SURROGATES="${SUP_N_SURROGATES:-1000}"
CONDA_ENV="${CONDA_ENV:-magneto_gpu}"

# ── Helper: run a Python stage inside a flushed pseudo-terminal ─────────────
run_stage() {
    local name="$1"
    shift
    local log="logs/${name}.log"
    mkdir -p logs
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting stage: ${name}"
    # script -q -f provides a PTY so tqdm renders and flushes to the log file.
    # conda run --no-capture-output streams stdout/stderr instead of buffering.
    script -q -f -c "PYTHONUNBUFFERED=1 conda run --no-capture-output -n ${CONDA_ENV} python -u $*" "${log}"
    local rc=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished stage: ${name} (rc=${rc})"
    return ${rc}
}

# ── Main pipeline ───────────────────────────────────────────────────────────
[[ "${RUN_PREFLIGHT}" == "1" ]] && run_stage "00_preflight" "scripts/00_preflight.py"
[[ "${RUN_MODIS_ETL}" == "1" ]] && run_stage "E01_modis_etl" "scripts/etl/E01_modis_etl.py --clean"
[[ "${RUN_OMNI_ETL}" == "1" ]] && run_stage "E02_omni_etl" "scripts/etl/E02_omni_etl.py --clean"
[[ "${RUN_SIF_ETL}" == "1" ]] && run_stage "E03_sif_etl" "scripts/etl/E03_sif_etl.py --clean"
[[ "${RUN_HDF2NETCDF}" == "1" ]] && run_stage "E04_mcd12c1_hdf2netcdf" "scripts/etl/E04_mcd12c1_hdf2netcdf.py"
[[ "${RUN_ERA5_ETL}" == "1" ]] && run_stage "E05_era5_etl" "scripts/etl/E05_era5_etl.py --clean"
[[ "${RUN_ERA5_ETL}" == "1" || "${RUN_MODIS_ETL}" == "1" || "${RUN_OMNI_ETL}" == "1" || "${RUN_SIF_ETL}" == "1" ]] && run_stage "E06_etl_equivalence_audit" "scripts/etl/E06_etl_equivalence_audit.py"
[[ "${RUN_ETL_CHUNKED_AUDIT}" == "1" ]] && run_stage "E07_etl_chunked_audit" "scripts/etl/E07_etl_chunked_audit.py"
[[ "${RUN_ETL_DEEP_AUDIT}" == "1" ]] && run_stage "E08_etl_deep_equivalence_audit" "scripts/etl/E08_etl_deep_equivalence_audit.py"
[[ "${RUN_QC}" == "1" ]] && run_stage "01_build_qc" "scripts/01_build_qc.py"
[[ "${RUN_LAI_CELLS_CONTROLS}" == "1" ]] && run_stage "02_lai_cells_controls" "scripts/02_assign_lai_quartiles.py"
[[ "${RUN_ERA5_JOIN}" == "1" ]] && run_stage "03_era5_join" "scripts/03_era5_join.py"
[[ "${RUN_SMOKE_SUBSET:-0}" == "1" ]] && run_stage "02b_smoke_subset" "scripts/02b_smoke_subset.py --smoke-test"
[[ "${RUN_HARMONIC}" == "1" ]] && run_stage "04_harmonic" "scripts/04_harmonic.py"
[[ "${RUN_CYCLIC_SPLINE}" == "1" ]] && run_stage "05_cyclic_spline" "scripts/05_cyclic_spline.py"
[[ "${RUN_MATCHED}" == "1" ]] && run_stage "07_matched" "scripts/07_matched.py"
[[ "${RUN_FIXED_WINDOW}" == "1" ]] && run_stage "08_fixed_window" "scripts/08_fixed_window.py"
[[ "${RUN_SURROGATES}" == "1" ]] && run_stage "09_surrogates" "scripts/09_surrogates.py --n-surrogates ${N_SURROGATES}"

# ── Environmental-driver matrix (restored from v1) ──────────────────────────
[[ "${RUN_ENV_DRIVER_PREPARE}" == "1" ]] && run_stage "10_prepare_environmental_driver_input" "scripts/10_prepare_environmental_driver_input.py"

if [[ "${RUN_ENV_DRIVER_MATRIX}" == "1" ]]; then
    if [[ "${RUN_SMOKE_SUBSET:-0}" == "1" ]]; then
        run_stage "11_environmental_driver_matrix_gpu" "scripts/11_environmental_driver_matrix_gpu.py --smoke"
    else
        run_stage "11_environmental_driver_matrix_gpu" "scripts/11_environmental_driver_matrix_gpu.py"
    fi
fi

[[ "${RUN_ENV_DRIVER_SUMMARY}" == "1" ]] && run_stage "12_environmental_driver_matrix_summary" "scripts/12_environmental_driver_matrix_summary.py"

[[ "${RUN_EFFECTS}" == "1" ]] && run_stage "10_effects" "scripts/10_effects.py"

# ── Main reports and audits ─────────────────────────────────────────────────
if [[ "${RUN_MAIN_REPORTS}" == "1" ]]; then
    run_stage "11_report" "scripts/reports/11_report.py"
    run_stage "12_audit_reports" "scripts/reports/12_audit_reports.py"
fi

# ── Supplementary outcomes sensitivity analysis ─────────────────────────────
if [[ "${RUN_SUPPLEMENTARY}" == "1" ]]; then
    run_stage "supplementary_outcomes" "scripts/run_supplementary_outcomes.py --n-surrogates ${SUP_N_SURROGATES} --clean-generated"
fi

# ── Supplementary reports and tables ────────────────────────────────────────
if [[ "${RUN_SUPPLEMENTARY_REPORTS}" == "1" ]]; then
    run_stage "supplementary_reports" "scripts/reports/generate_supplementary_reports.py"
    run_stage "supplementary_tables" "scripts/reports/format_supplementary_tables.py"
    run_stage "supplementary_provenance_audit" "scripts/tests/audit_supplementary_provenance.py"
fi

# ── Supplementary checks (sensitivity / robustness) ─────────────────────────
if [[ "${RUN_SUPPLEMENTARY_CHECKS}" == "1" ]]; then
    run_stage "supplementary_checks" "scripts/supplementary_checks/run_supplementary_checks.sh"
fi

# ── Final archives ──────────────────────────────────────────────────────────
[[ "${RUN_ARCHIVES}" == "1" ]] && run_stage "create_archives" "scripts/create_archives.py"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] All enabled stages completed."
