# MAGNETO Version History

## v1 historical original commit

- Original preprint pipeline, including generated outputs and notebooks.
- Corresponds to bioRxiv preprint DOI 10.64898/2026.02.17.706448.
- Historical source commit: `96e9731c11e94f1064a83525246a747ea044e848`.
- Preserved in local mirror, verified bundle, and immutable archive.
- Not published as a GitHub tag because the commit contains data/results.

## v1.0.0-preprint-source

- Source-only public release of the preprint pipeline.
- Created after removing generated outputs from the historical commit.
- Tag message: "Source-only archival release of the pipeline corresponding
  to the original preprint. Historical source commit:
  96e9731c11e94f1064a83525246a747ea044e848."
- Immutable reference for the source code; no scientific changes.

## v1.1.0-preprint-corrected

- Future corrected release of the preprint pipeline.
- Will address the MODIS grid-registration issue and regenerate affected
  analyses, tables, and figures.
- Tagged only after full recalculation, validation, and preprint update.
- Not yet released.

## v2.0.0-publication

- Revised journal-manuscript pipeline.
- Tag `v2.0.0-publication` points to the final clean publication-code commit
  produced after repository cleanup.
- Historical commits:
  - Analysis-core: `de1499e4749a3caea562453078d4e88447b791e1`
  - Packaging: `b318161d6a060b23ee95ace422a5395b3501a2a0`
- Analysis run: `20260730T061934Z`.
- Key features:
  - Primary outcome: OCO-2 SIF 771 nm
  - Primary exposure: 28-day trailing mean of Dst-derived SII
  - Primary seasonal adjustment: leave-one-year-out harmonic model
  - Sensitivity adjustment: leave-one-year-out cyclic spline
  - Temporal surrogates (year permutation, circular shift, block permutation)
  - Cluster bootstrap and environmental driver analyses
