# MAGNETO Branch Relationship

| Manuscript | Branch | Historical original commit | Public source tag | Future corrected tag |
|------------|--------|---------------------------|-------------------|----------------------|
| Journal article | `main` | — | `v2.0.0-publication` | — |
| Preprint | `v1-preprint` | `96e9731c11e94f1064a83525246a747ea044e848` | `v1.0.0-preprint-source` | `v1.1.0-preprint-corrected` |

## Rules

- `main` and `v1-preprint` are separate analytical generations.
- Do **not** merge them into each other.
- Do not share writable output directories between versions.
- Do not treat them as interchangeable implementations.
- Do not automatically backport analytical stages from one version to the other.
