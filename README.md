# MAGNETO analysis pipeline

This repository contains two distinct analytical generations of the MAGNETO
(Magnetosphere–Atmosphere–Geosphere Interaction Network & Ecological Trend
Observer) pipeline.

## Version 2 — journal publication

The default `main` branch contains the revised pipeline corresponding to the
journal manuscript.

- **Branch:** `main`
- **Release tag:** `v2.0.0-publication` (points to the final clean
  source-code commit created after repository cleanup)
- **Analysis run:** `20260730T061934Z`
- **Historical analysis-core commit:** `de1499e4749a3caea562453078d4e88447b791e1`
- **Historical packaging commit:** `b318161d6a060b23ee95ace422a5395b3501a2a0`
- **Spatial grid:** 0.5° × 0.5°
- **Primary outcome:** OCO-2 SIF at 771 nm
- **Primary exposure:** 28-day trailing mean of the Dst-derived SII
- **Primary seasonal adjustment:** leave-one-year-out harmonic model
- **Sensitivity adjustment:** leave-one-year-out cyclic spline

## Version 1 — preprint pipeline

The `v1-preprint` branch preserves the source code of the analytical
generation used for the original preprint.

- **Historical original commit:** `96e9731c11e94f1064a83525246a747ea044e848`
  (kept in local mirror, bundle, and immutable archive; not published as a
  GitHub tag because it contains generated outputs)
- **Source-only release tag:** `v1.0.0-preprint-source`
- **Future corrected tag:** `v1.1.0-preprint-corrected`

Version 1 and Version 2 are intentionally maintained as separate analytical
generations. They differ in analytical scope, architecture, inferential
framework, and outputs. They are not interchangeable and must not be merged.

## Repository contents

This repository contains only source code and reproduction instructions:

- `scripts/` — reproducible analysis source code
- `config/` — portable configuration templates (`*.example.yaml`)
- `README.md`, `LICENSE`, `CITATION.cff`
- environment specification (`pyproject.toml`, `requirements-lock.txt`,
  optional `environment.yml` for Conda/CUDA users)
- smoke tests under `scripts/tests/`

Data, results, figures, tables, reports, logs, and manuscripts are kept
outside Git.

## Installation

The authoritative dependency specification is `pyproject.toml`.

For a reproducible environment:

```bash
pip install -r requirements-lock.txt
```

For development or custom installs:

```bash
pip install -e .
```

If you use Conda for CUDA/RAPIDS support:

```bash
conda env create -f environment.yml
conda activate magneto_gpu
pip install -r requirements-lock.txt
```

Python 3.11 is required. Create a local configuration file from the template:

```bash
cp config/pipeline.example.yaml config/pipeline.local.yaml
# edit config/pipeline.local.yaml if absolute paths are needed
```

## Input data

Place raw datasets under `data/raw/`:

| Dataset | Path | Source |
|---------|------|--------|
| OMNI2 | `data/raw/omni2_all_years.zip` | NASA SPDF / OMNIWeb |
| MODIS LAI | `data/raw/MODIS/*.nc` | NASA LP DAAC |
| ERA5 | `data/raw/ERA5/*.nc` | Copernicus CDS |
| OCO-2 SIF | `data/raw/OCO2/*.nc4` | NASA GES-DISC |

## Running the pipeline

### Smoke test (N = 10 surrogates)

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_pipeline.py \
    --smoke-test \
    --n-surrogates 10 \
    --clean-generated \
    2>&1 | tee logs/smoke_test.log
```

### Full run (N = 1000 surrogates)

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_pipeline.py \
    --n-surrogates 1000 \
    --clean-generated \
    2>&1 | tee logs/pipeline.log
```

### Resume a failed run

```bash
python -u scripts/run_pipeline.py --resume --n-surrogates 1000
```

## Tests

```bash
python -m compileall -q .
pytest scripts/tests/
```

## Output structure

The pipeline creates the following directories locally (all ignored by Git):

```text
data/interim/
data/processed/
results/
reports/
figures/
logs/
```

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
