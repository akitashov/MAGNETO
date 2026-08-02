# MAGNETO v1 preprint pipeline

This branch preserves and corrects the analytical pipeline corresponding to
the original preprint.

## Status

The historical original preprint pipeline is preserved at commit
`96e9731c11e94f1064a83525246a747ea044e848`. That commit is kept in local
mirror, verified bundle, and immutable archive, but is **not** published as
a GitHub tag because it contains generated outputs and notebooks.

This branch is a source-only public release derived from the historical
commit. It is intended for a limited corrective revision of version 1. It is
**not** the pipeline corresponding to the revised journal manuscript.

- **Historical original commit:** `96e9731c11e94f1064a83525246a747ea044e848`
- **Source-only release tag:** `v1.0.0-preprint-source`
- **Future corrected tag:** `v1.1.0-preprint-corrected`

## Scope boundary

Version 1 preserves the analytical scope of the original preprint:

- 0.5° × 0.5° spatial grid
- Daily temporal resolution
- OMNI2 → MODIS → ERA5 → OCO-2 SIF ETL
- Spearman correlation, marker screening, GPU matrix search, meta-statistics,
  surrogate testing

The expanded analytical hierarchy of version 2 will not automatically be
backported.

## Planned corrections

1. Restoration of the intended 0.5° × 0.5° grid.
2. Corrected MODIS grid registration.
3. Explicit source-key matching.
4. Prevention of reuse of partial outputs.
5. Regeneration of affected analyses, tables, and figures.
6. Updated provenance and reproducibility records.

## Repository contents

This branch contains only source code and reproduction instructions:

- `scripts/` — v1 ETL, analysis, visualization, and utility scripts
- `config/` — portable configuration templates
- `README.md`, `LICENSE`, `.gitignore`
- `V1_CORRECTION_PLAN.md`
- environment specification

Generated data, results, figures, tables, reports, and logs are kept outside
Git.

## Installation

```bash
conda env create -f environment.yml
conda activate magneto_gpu
```

## Input data

Place raw datasets under `data/raw/`:

```text
data/raw/omni2_all_years.zip
data/raw/MODIS/*.nc
data/raw/ERA5/*.nc
data/raw/OCO2/*.nc4
```

## Running the pipeline

```bash
bash pipeline.sh
```

This executes the v1 stages in order from `01_Omni2_ETL.py` through
`14_Results_Sanity_Check.py`.

## Outputs

All outputs are written to a version-specific external root, for example:

```text
MAGNETO-data/v1/interim/
MAGNETO-data/v1/processed/
MAGNETO-data/v1/results/
MAGNETO-data/v1/figures/
MAGNETO-data/v1/logs/
```

The v1 output root must contain `/v1/`. Writing to a `/v2/` path is forbidden.

## Relation to version 2

Version 2 (`main`, tag `v2.0.0-publication`) is the revised journal pipeline.
Do not mix v1 and v2 outputs.

## Citation

Original preprint: *Cumulative geomagnetic disturbances modulate global
photosystem stoichiometry through temperature-dependent gating* (2026).
bioRxiv. DOI: 10.64898/2026.02.17.706448.

## License

See `LICENSE`.
