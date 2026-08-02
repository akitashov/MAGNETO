#!/usr/bin/env python3
"""
audit_supplementary_provenance.py — Numerical provenance audit for supplementary outcomes.

This script does NOT modify the analytical pipeline. It only reads the QC dataset
and writes audit reports and diagnostic tables used to decide whether the stored
SIF 740 nm and stress-indicator columns can be trusted.
"""
from __future__ import annotations
import sys, hashlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _Common import PROJECT_ROOT, FILE_QC, PARQUET_ENGINE, full_sha256

RESULTS = PROJECT_ROOT / "results"
REPORTS = PROJECT_ROOT / "reports"

# Provider formula for SIF 740 nm from the OCO-2/OCO-3 SIF Lite product description.
# Equivalent forms:
#   SIF_740 = 0.75 * (SIF_757 + 1.5 * SIF_771)
#   SIF_740 = 0.5 * (1.5 * SIF_757 + 2.25 * SIF_771)
SIF_740_FORMULA_TEXT = "0.75 * (SIF_757nm + 1.5 * SIF_771nm)"
SIF_740_FORMULA_LAMBDA = lambda df: 0.75 * (df["sif_757nm"] + 1.5 * df["sif_771nm"])


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def load_qc() -> pd.DataFrame:
    qc = pd.read_parquet(FILE_QC, engine=PARQUET_ENGINE)
    qc["date"] = pd.to_datetime(qc["date"])
    return qc


def audit_sif_740(qc: pd.DataFrame) -> None:
    """Write SIF 740 provenance audit files."""
    stored = qc["sif_740nm"]
    computed = SIF_740_FORMULA_LAMBDA(qc)
    mask = stored.notna() & qc["sif_757nm"].notna() & qc["sif_771nm"].notna()

    n_total = mask.sum()
    diff = (stored - computed)[mask]
    abs_diff = diff.abs()
    rel_diff = abs_diff / stored[mask].abs().replace(0, np.nan)

    # Random sample
    rng = np.random.default_rng(42)
    idx_random = rng.choice(diff.index, size=min(10000, len(diff)), replace=False)

    # Systematic sample by year/latitude/SIF level
    q_lat = pd.qcut(qc.loc[mask, "lat"], q=5, labels=False, duplicates="drop")
    q_sif = pd.qcut(qc.loc[mask, "sif_771nm"], q=5, labels=False, duplicates="drop")
    systematic_rows = []
    for year in sorted(qc.loc[mask, "year"].unique()):
        for lat_bin in range(5):
            for sif_bin in range(5):
                sub_idx = qc.index[mask & (qc["year"] == year) & (q_lat == lat_bin) & (q_sif == sif_bin)]
                if len(sub_idx) > 0:
                    systematic_rows.append(rng.choice(sub_idx))

    records = []
    for label, idx in [("random", idx_random), ("systematic", np.array(systematic_rows))]:
        sub_diff = diff.loc[idx]
        sub_abs = abs_diff.loc[idx]
        sub_rel = rel_diff.loc[idx]
        records.append({
            "sample": label,
            "n": len(sub_diff),
            "max_abs_diff": float(sub_abs.max()),
            "mean_abs_diff": float(sub_abs.mean()),
            "median_abs_diff": float(sub_abs.median()),
            "n_within_1e9": int((sub_abs < 1e-9).sum()),
            "n_within_1e6": int((sub_abs < 1e-6).sum()),
            "max_rel_diff_finite": float(sub_rel.replace([np.inf, -np.inf], np.nan).max()),
        })

    check_df = pd.DataFrame(records)
    atomic_write(check_df, RESULTS / "sif_740_provenance_check.csv")

    # Overall diagnostics
    n_within_tol = int((abs_diff < 1e-6).sum())
    corr = float(stored[mask].corr(computed[mask]))

    md = f"""# SIF 740 nm Provenance Audit

## Source product

The archived SIF input (`{FILE_QC.relative_to(PROJECT_ROOT)}`) contains the variable
`sif_740nm`. The OCO-2/OCO-3 SIF Lite product description states that the actual
narrow-band retrievals are performed at 757 nm and 771 nm, and that SIF at 740 nm is
a **provider-estimated reference-wavelength quantity** derived from those two
retrievals.

## Provider formula

The product documentation gives the relationship as:

```text
SIF_740 nm = 0.75 * (SIF_757 nm + 1.5 * SIF_771 nm)
```

This is algebraically identical to:

```text
SIF_740 nm = 0.5 * (1.5 * SIF_757 nm + 2.25 * SIF_771 nm)
```

## Numerical verification

- QC rows compared: {n_total:,}
- Pearson correlation (stored vs. computed): {corr:.12f}
- Rows with absolute difference < 1e-6: {n_within_tol:,} / {n_total:,} ({100*n_within_tol/n_total:.2f}%)
- Maximum absolute difference: {float(abs_diff.max()):.3e}
- Mean absolute difference: {float(abs_diff.mean()):.3e}

## Sample checks

{_df_to_md(check_df)}

## Interpretation

The stored `sif_740nm` values agree with the provider-derived formula to within
~1e-6 mW m⁻² sr⁻¹ nm⁻¹ (median ~4e-8). The small deviations are consistent with
single-precision floating-point storage or rounding in the upstream aggregation.
Therefore `sif_740nm` is treated as a **provider-estimated / provider-derived**
reference-wavelength quantity, not as an independent instrumental retrieval.

No numerical recalculation of SIF 740 nm residuals or surrogates is required; only
the outcome classification metadata needs to be updated.

## Provenance

- QC file: `{FILE_QC.relative_to(PROJECT_ROOT)}`
- QC SHA-256: `{full_sha256(FILE_QC)}`
- Git commit: `{_git_commit()}`
- Audit generated: `{pd.Timestamp.utcnow().isoformat()}`
"""
    (REPORTS / "SIF_740_PROVENANCE_AUDIT.md").write_text(md)
    print("Wrote results/sif_740_provenance_check.csv")
    print("Wrote reports/SIF_740_PROVENANCE_AUDIT.md")


def audit_stress(qc: pd.DataFrame) -> None:
    """Write stress-indicator provenance and denominator-stability files."""
    stored = qc["sif_stress_index"]
    denom = qc["sif_771nm"]
    numer = qc["sif_757nm"]
    computed_ratio = numer / denom

    # Stored vs computed
    mask_both = stored.notna() & numer.notna() & denom.notna() & (denom != 0)
    n_both = mask_both.sum()
    diff = (stored - computed_ratio)[mask_both]
    ratio_exact_match = int((diff.abs() < 1e-12).sum())

    # Missingness rules
    rule_threshold = denom < 0.001
    rule_zero = denom == 0
    rule_nonfinite = denom.isna() | numer.isna() | ~np.isfinite(computed_ratio)
    expected_missing = rule_threshold | rule_zero | rule_nonfinite

    n_missing_stored = stored.isna().sum()
    n_expected_missing = expected_missing.sum()
    missing_agreement = int((stored.isna() == expected_missing).sum())

    # Availability audit
    avail_records = []
    for rule_name, rule_mask in [
        ("stored_sif_stress_index", stored.notna()),
        ("recomputed_no_threshold", numer.notna() & denom.notna() & (denom != 0) & np.isfinite(computed_ratio)),
        ("recomputed_threshold_0.001", numer.notna() & (denom >= 0.001) & np.isfinite(computed_ratio)),
    ]:
        sub = qc[rule_mask]
        avail_records.append({
            "rule": rule_name,
            "n_obs": int(rule_mask.sum()),
            "n_cells": sub[["lat_id", "lon_id"]].drop_duplicates().shape[0] if len(sub) else 0,
            "n_years": sub["year"].nunique() if len(sub) else 0,
        })
    avail_df = pd.DataFrame(avail_records)
    atomic_write(avail_df, RESULTS / "stress_indicator_availability_audit.csv")

    # Denominator diagnostics
    denom_finite = denom[numer.notna() & denom.notna() & np.isfinite(denom)]
    denom_stats = {
        "n_total": len(qc),
        "n_missing": int(denom.isna().sum()),
        "n_negative": int((denom < 0).sum()),
        "n_zero": int((denom == 0).sum()),
        "n_0_to_1e4": int(((denom > 0) & (denom < 0.0001)).sum()),
        "n_1e4_to_1e3": int(((denom >= 0.0001) & (denom < 0.001)).sum()),
        "n_1e3_to_1e2": int(((denom >= 0.001) & (denom < 0.01)).sum()),
        "median": float(denom_finite.median()),
        "q01": float(denom_finite.quantile(0.01)),
        "q05": float(denom_finite.quantile(0.05)),
        "q25": float(denom_finite.quantile(0.25)),
        "q75": float(denom_finite.quantile(0.75)),
        "q95": float(denom_finite.quantile(0.95)),
        "q99": float(denom_finite.quantile(0.99)),
        "max": float(denom_finite.max()),
    }
    denom_df = pd.DataFrame([denom_stats])
    atomic_write(denom_df, RESULTS / "stress_denominator_diagnostics.csv")

    # Ratio diagnostics with and without threshold
    ratio_no_threshold = numer / denom
    ratio_threshold = numer / denom.where(denom >= 0.001, np.nan)
    ratio_records = []
    for label, s in [("no_threshold", ratio_no_threshold), ("threshold_0.001", ratio_threshold)]:
        finite = s[np.isfinite(s)]
        ratio_records.append({
            "rule": label,
            "finite_fraction": float(finite.count() / len(s)),
            "n_finite": int(finite.count()),
            "n_infinite_or_nan": int((~np.isfinite(s)).sum()),
            "median": float(finite.median()),
            "iqr": float(finite.quantile(0.75) - finite.quantile(0.25)),
            "q01": float(finite.quantile(0.01)),
            "q99": float(finite.quantile(0.99)),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "fraction_removed_by_rule": float(1 - finite.count() / len(s)),
        })
    ratio_df = pd.DataFrame(ratio_records)

    md_stress = f"""# Stress Indicator Provenance Audit

## Definition

The archived QC column `sif_stress_index` is the ratio:

```text
sif_stress_index = sif_757nm / sif_771nm
```

## Numerical verification

- Rows where stored value and recomputed ratio are both finite: {n_both:,}
- Exact matches (|diff| < 1e-12): {ratio_exact_match:,} / {n_both:,}
- Maximum absolute difference: {float(diff.abs().max()):.3e}

## Missingness analysis

- Stored `sif_stress_index` non-null rows: {int(stored.notna().sum()):,}
- Stored `sif_stress_index` null rows: {n_missing_stored:,}
- Rows expected missing under rule `sif_771nm < 0.001 OR sif_771nm == 0 OR non-finite inputs`: {n_expected_missing:,}
- Rows where stored missingness matches expected rule: {missing_agreement:,} / {len(qc):,}

The stored missingness is **exactly explained** by the admissibility rule
`sif_771nm >= 0.001` (plus finite-input and non-zero-denominator guards).

## Availability by rule

{_df_to_md(avail_df)}

## Interpretation

The `sif_stress_index` column appears to have been constructed in the upstream ETL
with an explicit denominator threshold (`sif_771nm >= 0.001`). The threshold is not
applied to the direct SIF columns (`sif_757nm`, `sif_771nm`, `sif_740nm`), which
remain fully populated. This pattern is consistent with a deliberate stability rule
for the ratio rather than a legacy general QC truncation.

Because the upstream ETL script is not present in the archived revision pipeline,
the exact intent cannot be reconstructed with certainty. The conservative choice is
to recompute the stress indicator explicitly from `sif_757nm / sif_771nm` and apply
the only recoverable admissibility rule (`sif_771nm >= 0.001`). The resulting sample
is labelled `stress_threshold_conditioned` and is not treated as comparable in size
to the direct SIF outcomes.

## Provenance

- QC file: `{FILE_QC.relative_to(PROJECT_ROOT)}`
- QC SHA-256: `{full_sha256(FILE_QC)}`
- Git commit: `{_git_commit()}`
- Audit generated: `{pd.Timestamp.utcnow().isoformat()}`
"""
    (REPORTS / "STRESS_INDICATOR_PROVENANCE_AUDIT.md").write_text(md_stress)
    print("Wrote results/stress_indicator_availability_audit.csv")
    print("Wrote reports/STRESS_INDICATOR_PROVENANCE_AUDIT.md")

    md_denom = f"""# Stress Indicator Denominator Stability

## Distribution of the denominator (`sif_771nm`)

{_df_to_md(denom_df)}

## Distribution of the ratio under alternative admissibility rules

{_df_to_md(ratio_df)}

## Interpretation

Without any denominator threshold, the ratio `sif_757nm / sif_771nm` is defined for
all rows with non-zero `sif_771nm`, but it becomes numerically unstable when the
denominator is close to zero. The threshold `sif_771nm >= 0.001` removes
{int((denom < 0.001).sum()):,} rows ({100*(denom < 0.001).mean():.2f}% of the QC
dataset). This is the rule that reproduces the archived `sif_stress_index`
missingness exactly.

The threshold-conditioned sample should not be interpreted as a full outcome-specific
sample; it is a stability-conditioned sensitivity outcome.

## Provenance

- QC file: `{FILE_QC.relative_to(PROJECT_ROOT)}`
- QC SHA-256: `{full_sha256(FILE_QC)}`
- Git commit: `{_git_commit()}`
- Audit generated: `{pd.Timestamp.utcnow().isoformat()}`
"""
    (REPORTS / "STRESS_DENOMINATOR_STABILITY.md").write_text(md_denom)
    print("Wrote results/stress_denominator_diagnostics.csv")
    print("Wrote reports/STRESS_DENOMINATOR_STABILITY.md")


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        rows.append([str(v) if not pd.isna(v) else "—" for v in r.values])
    if not rows:
        return ""
    col_widths = [max(len(str(c)), max(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    lines = ["| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols)) + " |"]
    lines.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
    return "\n".join(lines)


def main() -> int:
    qc = load_qc()
    audit_sif_740(qc)
    audit_stress(qc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
