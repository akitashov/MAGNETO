# MAGNETO — corrected bioRxiv preprint pipeline

This branch contains the corrected analytical pipeline associated with Version 4 of the bioRxiv preprint:

**Kitashov, A. V. (2026). _Temperature-dependent associations between cumulative geomagnetic disturbances and satellite-derived solar-induced fluorescence_. bioRxiv 2026.02.17.706448.**  
https://doi.org/10.1101/2026.02.17.706448

## Important status note

This branch is an **archival preprint-analysis branch**. It is not the analytical pipeline used for the final peer-reviewed publication.

The final analysis was independently revised, peer reviewed, and published as:

**Kitashov, A. V. (2026). _A Weak Temporal Association Between Multi-Week Geomagnetic Activity and Satellite-Derived Solar-Induced Chlorophyll Fluorescence Anomalies_. Biology 15, 1415.**  
https://doi.org/10.3390/biology15161415

For scientific interpretation and citation of the final results, the peer-reviewed article should be preferred. The publication pipeline is maintained separately on the repository's publication branch (`main`) and is archived under the tag `v2.0.0-publication`.

## Purpose of this branch

The original bioRxiv analysis was produced with an earlier analytical pipeline. A later audit identified implementation problems affecting parts of the preprint analysis. Version 4 of the preprint and this branch preserve the original analytical design while correcting the affected pipeline.

The corrected preprint branch includes, among other changes:

- corrected environmental-data ingestion and validation;
- corrected MODIS-to-analysis-grid registration;
- explicit spatial join keys;
- lagged trailing-window construction that excludes the observation day;
- tie-aware Spearman ranking;
- safeguards against incomplete or stale intermediate outputs;
- revised treatment of pooled space-time statistics: the former pooled effective-sample-size correction is not used for inferential claims;
- consistency and sanity-check stages for the corrected workflow.

The window-dependent pooled correlation and matrix-search analyses in the corrected preprint should therefore be interpreted as **descriptive analyses**, as stated in Version 4 of the manuscript.

## Relationship between repository branches

The two analytical lines in this repository should not be mixed:

- **`main` / `v2.0.0-publication`** — final peer-reviewed `Biology` analysis.
- **this branch / `v1.1.0-preprint-corrected`** — corrected archival pipeline for bioRxiv Version 4.

Outputs from one branch must not be reused as inputs to the other branch.

## Repository contents

```text
scripts/
    01_Omni2_ETL.py
    02_MODIS_ETL.py
    03_ERA5_env_ETL.py
    04_SIF_ETL.py
    05_SIF_anomalies.py
    06_Spearman.py
    07_Spearman_aggregate.py
    08_Marker_Screening.py
    09_Screening_aggregate.py
    10_SII_PAR_Correlation.py
    11_Matrix_Search_GPU.py
    12_Meta_statistics.py
    13_Pipeline_Consistency_Audit.py
    14_Results_Sanity_Check.py
    V_*.py
    _Common.py

config/
    pipeline.example.yaml

environment.yml
pipeline.sh
```

The numbered scripts form the analytical pipeline. Visualization scripts are kept separately under `scripts/V_*.py`.

Raw data, intermediate files, generated results, figures, reports, and logs are not distributed through Git.

## Software environment

The supplied Conda environment targets Python 3.11 and a CUDA-capable system.

```bash
conda env create -f environment.yml
conda activate magneto_gpu
```

The GPU matrix-search stage and some preprocessing operations depend on CUDA/CuPy/RAPIDS. The exact package specification in `environment.yml` is the reference environment for this archival branch.

## Input data

Expected raw inputs are placed under:

```text
data/raw/
├── omni2_all_years.zip
├── MODIS/
│   └── *.nc
├── ERA5/
│   └── *.nc
└── OCO2/
    └── oco2_LtSIF_*.nc4
```

The raw datasets themselves are not included in this repository. They are obtained from the original data providers described in the preprint.

The pipeline uses paths and analysis parameters defined in `scripts/_Common.py`. `config/pipeline.example.yaml` is a human-readable record of the intended configuration; the archived scripts do not use it as their primary runtime configuration.

## Running the analytical pipeline

`pipeline.sh` executes the numbered stages from `01_Omni2_ETL.py` through `14_Results_Sanity_Check.py`.

Because the archived shell wrapper invokes the Python files by basename, run it with `scripts/` as the working directory:

```bash
cd scripts
bash ../pipeline.sh
```

The stages are:

1. OMNI2 preprocessing
2. MODIS preprocessing
3. ERA5 environmental preprocessing
4. OCO-2 SIF preprocessing
5. SIF anomaly construction
6. Spearman window analysis
7. Spearman aggregation
8. SII/F10.7 marker screening
9. screening aggregation
10. SII–environment covariation diagnostics
11. GPU matrix search
12. matrix-search summary statistics
13. pipeline consistency audit
14. results sanity check

Run the complete sequence from clean intermediate/output directories when reproducing the corrected analysis. Do not reuse intermediate files created by a different branch or an earlier pipeline version.

## Outputs

The scripts write generated material under project-local directories such as:

```text
data/interim/
results/
reports/
logs/
```

These directories are runtime products and should remain outside version control.

## Reproducibility scope

This repository preserves the corrected **preprint analytical branch**, not the final journal workflow. Reproducing the numerical results also requires the corresponding raw source datasets and their versions.

The peer-reviewed article used a substantially revised analytical design, including a different inferential hierarchy. Its results should not be expected to be reproduced by this branch.

## Version references

- `v1.0.0-preprint-source` — source-only archival snapshot of the original preprint pipeline.
- `v1.1.0-preprint-corrected` — corrected bioRxiv Version 4 pipeline (this branch).
- `v2.0.0-publication` — final peer-reviewed publication pipeline.

Existing release tags are immutable historical references and should not be moved.

## Citation

For the corrected preprint:

> Kitashov, A. V. (2026). _Temperature-dependent associations between cumulative geomagnetic disturbances and satellite-derived solar-induced fluorescence_. bioRxiv 2026.02.17.706448. https://doi.org/10.1101/2026.02.17.706448

For the final scientific results, please cite:

> Kitashov, A. V. (2026). _A Weak Temporal Association Between Multi-Week Geomagnetic Activity and Satellite-Derived Solar-Induced Chlorophyll Fluorescence Anomalies_. Biology, 15, 1415. https://doi.org/10.3390/biology15161415

## License

See the repository `LICENSE` file.
