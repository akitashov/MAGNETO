#!/bin/bash

export PYTHONUNBUFFERED=1

LOGFILE="pipeline.log"

# Redirect all output (both stdout and stderr) to both screen and logfile simultaneously,
# filtering out lines containing "%|" or "/it/s" (tqdm progress bars)
exec > >(tee -i >(sed -u '/%|/d; /it\/s/d' > "$LOGFILE")) 2>&1

printf '%0.s=' {1..38}; echo
echo "PIPELINE STARTED AT: $(date)"
echo "LOGGING TO FILE: $LOGFILE"
printf '%0.s=' {1..38}; echo

# Function for running steps with error checking
run_step() {
    printf '\n'
    printf '%0.s-' {1..38}; echo
    echo "[$(date +'%H:%M:%S')] EXECUTING: $1"
    printf '%0.s-' {1..38}; echo

    # Execute the script
    python "$1"

    # Check return code of last command ($?)
    if [ $? -ne 0 ]; then
        printf '\n'
        printf '%0.s!' {1..38}; echo
        echo "[ERROR] SCRIPT FAILED: $1"
        echo "SEE $LOGFILE"
        printf '%0.s!' {1..38}; echo
        exit 1
    else
        echo "[SUCCESS] SCRIPT COMPLETED SUCCESSFULLY: $1"
    fi
}


# --- FLOW OF SCRIPTS ---

# ETL and Preprocessing

run_step "01_Omni2_ETL.py"
run_step "02_MODIS_ETL.py"
run_step "03_ERA5_env_ETL.py"
run_step "04_SIF_ETL.py"
run_step "05_SIF_anomalies.py"

# Analytical Module

run_step "06_Spearman.py"            # Basic correlations
run_step "07_Spearman_aggregate.py"  # Spearman aggregation
run_step "08_Marker_Screening.py"    # Comparison: SII vs F10.7
run_step "09_Screening_aggregate.py" # Screening aggregation
run_step "10_SII_PAR_Correlation.py" # Independence check: SII vs ENV
run_step "11_Matrix_Search_GPU.py"   # Exhaustive search (Symmetric)
run_step "12_Meta_statistics.py"     # Matrix summary

# Checkups & Audits

run_step "13_Pipeline_Consistency_Audit.py"
run_step "14_Results_Sanity_Check.py"

printf '\n'
printf '%0.s=' {1..38}; echo
echo "PIPELINE COMPLETE: ALL STEPS FINISHED SUCCESSFULLY AT: $(date)"
printf '%0.s=' {1..38}; echo