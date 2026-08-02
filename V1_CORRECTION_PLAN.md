# V1 Correction Plan

## 1. Original grid-registration error

The original preprint pipeline used a MODIS grid-registration step that did
not consistently snap the high-resolution MODIS product to the intended
0.5° × 0.5° analysis grid. This affected cell IDs and downstream joins with
SIF and environmental data.

## 2. Affected ETL stages

- `02_MODIS_ETL.py` — MODIS aggregation and grid registration
- `04_SIF_ETL.py` — SIF-MODIS join keys
- `05_SIF_anomalies.py` — anomaly model grid assumptions

## 3. Downstream analyses

- Spearman correlations (`06_Spearman.py`, `07_Spearman_aggregate.py`)
- Marker screening (`08_Marker_Screening.py`, `09_Screening_aggregate.py`)
- SII–environment independence checks (`10_SII_PAR_Correlation.py`)
- GPU matrix search (`11_Matrix_Search_GPU.py`)
- Meta-statistics (`12_Meta_statistics.py`)
- Surrogate testing (`15_Surrogate_test.py`)

## 4. Outputs requiring recalculation

- All `data/interim/` canonical files
- All `data/processed/` analytical files
- All `results/` CSV and JSON summaries
- All preprint figures (`V_*.py` outputs)
- All preprint tables and meta-statistics reports

## 5. Figures and tables of the preprint

All figure- and table-generating scripts in `scripts/` must be re-run after
corrected ETL outputs are available.

## 6. External v1 output root

Use a dedicated external root such as `MAGNETO-data/v1/`. This root must be
separate from `MAGNETO-data/v2/`.

## 7. Clean-run requirement

The corrected v1 run must start from raw inputs with empty interim/processed
 directories. Partial outputs from the original run must not be reused.

## 8. New v1 analysis_run_id

Assign a new analysis_run_id that is not equal to the v2 run ID
(`20260730T061934Z`).

## 9. Input/output hashing

Record SHA256 hashes of:
- raw input archives
- ETL scripts
- analytical scripts
- generated interim and processed files

## 10. Tests

- Compile all scripts: `python -m compileall -q .`
- Smoke-test each ETL stage on a small spatial/temporal subset.
- Validate grid alignment and source-key matching.

## 11. Validation criteria

- MODIS grid cells align exactly with the 0.5° × 0.5° target.
- Source-key matching has zero unmatched cells after masks.
- Reproduced qualitative conclusions are stable or explicitly revised.
- All outputs are written under the v1 external root.

## 12. Criteria for creating v1.1.0-preprint-corrected

Create the tag only after:

1. Full clean recalculation from raw inputs.
2. Successful script compilation and smoke tests.
3. Validation of grid registration and source keys.
4. Regeneration of all affected figures and tables.
5. Updated preprint text and supplementary materials.
6. Provenance record with new analysis_run_id and hashes.
