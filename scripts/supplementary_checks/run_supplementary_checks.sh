#!/usr/bin/env bash
# run_supplementary_checks.sh — sequential launcher for the MAGNETO supplementary checks layer.
#
# Each stage is independent and writes to results/supplementary_checks,
# reports/supplementary_checks, figures/supplementary_checks, and
# logs/supplementary_checks. Comment out any stage that is not needed.
#
# Usage:
#   bash scripts/supplementary_checks/run_supplementary_checks.sh
#
# To run only a subset, comment out the unwanted lines below.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Parse optional --smoke-test flag and forward it to stages that support it.
SMOKE_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--smoke-test" ]; then
        SMOKE_FLAG="--smoke"
        break
    fi
done

LOG_DIR="$ROOT_DIR/logs/supplementary_checks"
mkdir -p "$LOG_DIR"

# Use the project Conda environment. Adjust if your environment name differs.
RUNNER="conda run -n magneto_gpu python -u"

# ---------------------------------------------------------------------------
# Stage S01: SII exposure window scan for SIF 771 nm
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S01 — Window scan (SIF 771 nm)"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S01_window_scan_771.py \
    2>&1 | tee "$LOG_DIR/S01_window_scan_771.log"

# ---------------------------------------------------------------------------
# Stage S02: SII vs Kp comparison
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S02 — SII vs Kp comparison"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S02_kp_comparison.py \
    2>&1 | tee "$LOG_DIR/S02_kp_comparison.log"

# ---------------------------------------------------------------------------
# Stage S03: SII vs F10.7 comparison
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S03 — SII vs F10.7 comparison"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S03_f107_comparison.py \
    2>&1 | tee "$LOG_DIR/S03_f107_comparison.log"

# ---------------------------------------------------------------------------
# Stage S04: South Atlantic Anomaly control (with surrogates)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S04 — SAA control analysis"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S04_saa_control.py \
    --n-surrogates 1000 \
    2>&1 | tee "$LOG_DIR/S04_saa_control.log"

# ---------------------------------------------------------------------------
# Stage S05: Prepare land-cover grid
# ---------------------------------------------------------------------------
# This stage requires a global land-cover classification file configured in
# config/supplementary_checks.yaml (default: data/interim/MCD12C1*.nc).
echo "================================================================"
echo "S05 — Prepare land-cover grid"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S05_prepare_landcover_grid.py \
    2>&1 | tee "$LOG_DIR/S05_prepare_landcover_grid.log"

# ---------------------------------------------------------------------------
# Stage S06: Land-cover class analysis (with surrogates)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S06 — Land-cover class analysis"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S06_landcover_analysis.py \
    --n-surrogates 1000 \
    2>&1 | tee "$LOG_DIR/S06_landcover_analysis.log"

# ---------------------------------------------------------------------------
# Stage S06b: Land-cover × temperature inference
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S06b — Land-cover × temperature inference"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S06b_landcover_temperature_inference.py \
    $SMOKE_FLAG \
    ${SMOKE_FLAG:+--force} \
    ${SMOKE_FLAG:+--output-dir} ${SMOKE_FLAG:+"$ROOT_DIR/results/supplementary_checks/smoke"} \
    2>&1 | tee "$LOG_DIR/S06b_landcover_temperature_inference.log"

# ---------------------------------------------------------------------------
# Stage S04b: Temperature profiles for control scenarios (with surrogates)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S04b — Control temperature profiles"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S04b_control_temperature_profiles.py \
    --n-surrogates 1000 \
    2>&1 | tee "$LOG_DIR/S04b_control_temperature_profiles.log"

# ---------------------------------------------------------------------------
# Stage S07: Graphical abstract matrix
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S07 — Graphical abstract matrix"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S07_make_graphical_abstract_matrix.py \
    2>&1 | tee "$LOG_DIR/S07_graphical_abstract_matrix.log"

# ---------------------------------------------------------------------------
# Stage S08: Temperature scenario tables
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S08 — Temperature scenario tables"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S08_temperature_scenarios.py \
    2>&1 | tee "$LOG_DIR/S08_temperature_scenarios.log"

# ---------------------------------------------------------------------------
# Stage S10: SII vs environment correlation check
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S10 — SII vs environment correlation"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S10_sii_environment_correlation.py \
    2>&1 | tee "$LOG_DIR/S10_sii_environment_correlation.log"

# ---------------------------------------------------------------------------
# Stage S11: Shapley/LMG R² decomposition across SII, PAR, VPD
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S11 — Driver R² Shapley/LMG decomposition"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S11_driver_r2_shapley.py \
    $SMOKE_FLAG \
    ${SMOKE_FLAG:+--output-dir} ${SMOKE_FLAG:+"$ROOT_DIR/results/supplementary_checks/smoke"} \
    2>&1 | tee "$LOG_DIR/S11_driver_r2_shapley.log"

# ---------------------------------------------------------------------------
# Stage S12: Aggregate Shapley/LMG decomposition for plotting source
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S12 — Driver R² Shapley/LMG aggregate"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S12_driver_r2_shapley_aggregate.py \
    ${SMOKE_FLAG:+--smoke-test} \
    2>&1 | tee "$LOG_DIR/S12_driver_r2_shapley_aggregate.log"

# ---------------------------------------------------------------------------
# Stage S09b: Cluster-bootstrap confidence intervals for LAI-stratified associations
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S09b — LAI-quartile cluster bootstrap"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S09b_lai_quartile_bootstrap.py \
    2>&1 | tee "$LOG_DIR/S09b_lai_quartile_bootstrap.log"

# ---------------------------------------------------------------------------
# Stage S09c: Cluster-bootstrap confidence intervals for Table 1 primary rows
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S09c — Table 1 primary rows cluster bootstrap"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S09c_table1_bootstrap.py \
    2>&1 | tee "$LOG_DIR/S09c_table1_bootstrap.log"

# ---------------------------------------------------------------------------
# Stage S09: Collect outputs, manifest, README, consistency checks
# ---------------------------------------------------------------------------
echo "================================================================"
echo "S09 — Build supplementary checks outputs and manifest"
echo "================================================================"
PYTHONUNBUFFERED=1 $RUNNER scripts/supplementary_checks/S09_build_supplementary_checks_outputs.py \
    ${SMOKE_FLAG:+--smoke-test} \
    2>&1 | tee "$LOG_DIR/S09_build_supplementary_checks_outputs.log"

echo "================================================================"
echo "Supplementary checks pipeline complete."
echo "================================================================"
