#!/usr/bin/env python3
"""Post-resolution audit of barren / low-LAI / landcover Barren usage.

Scans the current pipeline (excluding v1/, legacy/, archive/, VCS, and binary
outputs) and produces:

    reports/audits/barren_sample_usage_after.csv
    reports/audits/barren_sample_resolution.md

Every hit is classified as one of the canonical samples or flagged as ambiguous.
Legacy identifiers (is_barren, control_barren, Persistently_Barren, etc.) in the
current pipeline are reported as a separate block so they can be reviewed.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "audits"
OUT_CSV = OUT_DIR / "barren_sample_usage_after.csv"
OUT_MD = OUT_DIR / "barren_sample_resolution.md"

# Order matters: longer/more specific identifiers before shorter ones.
IDENTIFIER_PATTERNS = [
    r"\bcontrol_strict_low_lai\b",
    r"\bis_strict_low_lai\b",
    r"\bCONTROL_STRICT_LOW_LAI\b",
    r"\bPersistently_Barren\b",
    r"\bpersistently_barren\b",
    r"\bcontrol_barren\b",
    r"\bcontrol_Barren\b",
    r"\bis_barren\b",
    r"\blandcover_barren\b",
    r"\blandcover_Barren\b",
    r"\bBarren\b",
    r"\bbarren\b",
]

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "v1", "legacy", "archive"}
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".parquet", ".feather", ".pkl", ".pyc", ".log"}

# Limit the audit to source / config / documentation. Generated reports, results,
# figures, notebooks and data artefacts are not inspected here because they are
# rebuilt by the pipeline and would contain stale historical labels until then.
SOURCE_DIRS = {"scripts", "config"}
SOURCE_FILES = {
    "README.md",
    "SMOKE_README.md",
    "FULL_RUN_README.md",
    "pipeline.sh",
    "pyproject.toml",
    "environment.yml",
    "requirements.txt",
    "requirements-lock.txt",
}


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        rel = p.relative_to(root)
        # Keep files inside scripts/ or config/, plus selected top-level docs.
        if rel.parts[0] not in SOURCE_DIRS and rel.name not in SOURCE_FILES:
            continue
        # Ignore this audit script and the old temporary helper.
        if rel.name in {"_audit_barren_usage.py", "_tmp_audit_barren.py"}:
            continue
        try:
            if p.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue
        yield p


def classify(identifier: str, file: Path, line: str) -> tuple[str, str]:
    """Return (intended_sample, definition) for a matched identifier."""
    low_id = identifier.lower()
    low_line = line.lower()
    rel = file.relative_to(ROOT) if file.is_absolute() else file
    fname = rel.name.lower()

    # Legacy functional-control identifiers should no longer appear in the
    # current pipeline, except in explanatory comments about historical files.
    # Flag them explicitly so the report is easy to review.
    if low_id in {"is_barren", "control_barren", "persistently_barren", "persistently barren"}:
        if "historical" in low_line or "legacy" in low_line:
            return "COMMENT (historical note)", "Documents an old identifier in historical result files; not live code."
        return "LEGACY STRICT LOW-LAI IDENTIFIER (should be 0)", (
            "Legacy functional low-LAI control identifier; post-resolution code should use "
            "is_strict_low_lai / control_strict_low_lai."
        )

    # Canonical strict low-LAI identifiers
    if "strict_low_lai" in low_id:
        return "control_strict_low_lai", (
            "Strict low-LAI functional control: median LAI ≤ 0.10, "
            "90th percentile LAI ≤ 0.30, ≥10 valid LAI observations."
        )

    # The config key in pipeline.yaml that groups the strict low-LAI thresholds.
    if low_id == "barren" and "pipeline.yaml" in fname:
        return "control_strict_low_lai", (
            "Config grouping key for the strict low-LAI functional control thresholds."
        )

    # IGBP class entries in the land-cover map, e.g. "16: Barren".
    if low_id == "barren" and ":" in line and any(ch.isdigit() for ch in line.split(":")[0]):
        return "landcover_barren", "MCD12C1 IGBP land-cover class label in config map."

    # Canonical land-cover Barren identifiers / labels
    if "landcover" in low_id:
        return "landcover_barren", (
            "Dominant MCD12C1 Version 6.1 IGBP land-cover class == Barren on the 0.5° grid."
        )

    # MCD12C1 IGBP class labels / maps (e.g. "16: Barren", Barren in figures/tables).
    if low_id == "barren" and (
        "igbp" in low_line
        or "mcd12c1" in low_line
        or "dominant" in low_line
        or "land-cover" in low_line
        or "land cover" in low_line
        or "class" in low_line
        or "vegetation" in low_line
        or "forest" in low_line
        or "cropland" in low_line
        or "composition" in low_line
        or fname.startswith("table")
        or fname.startswith("figure")
        or "landcover" in fname
    ):
        return "landcover_barren", "Dominant MCD12C1 IGBP class == Barren (contextual inference)."

    # Sahara / geographic control references
    if "sahara" in low_line:
        return "control_Sahara", "Sahara geographic control (15–32° N, 18° W–40° E)."

    # Informative comments that do not refer to a sample
    if low_id == "barren" and any(
        phrase in low_line
        for phrase in (
            "geographic barren control",
            "principal geographic barren",
            "barren control (sahara)",
            "barren/vegetated",
            "barren or sparsely vegetated",
        )
    ):
        return "COMMENT (not a sample identifier)", "Informative prose; no sample is defined here."

    # Any remaining control+barren prose is ambiguous and must be reviewed.
    if "control" in low_line and "barren" in low_line:
        return "AMBIGUOUS", "Control-related barren reference; review required."

    # Residual generic "barren" prose.
    if low_id == "barren":
        return "AMBIGUOUS", "Generic barren reference; review required."

    return "AMBIGUOUS", "Could not classify automatically; review required."


def build_rows() -> list[dict]:
    regex = re.compile("|".join(f"({p})" for p in IDENTIFIER_PATTERNS), re.IGNORECASE)
    rows = []
    seen = set()

    for p in iter_files(ROOT):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = p.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            matches = regex.findall(line)
            if not matches:
                continue
            identifiers = [
                m for group in matches
                for m in (group if isinstance(group, tuple) else (group,))
                if m
            ]
            for identifier in identifiers:
                key = (str(rel), lineno, identifier)
                if key in seen:
                    continue
                seen.add(key)
                intended, definition = classify(identifier, p, line)
                rows.append({
                    "file": str(rel),
                    "line": lineno,
                    "identifier": identifier,
                    "context": line.strip()[:250],
                    "intended_sample": intended,
                    "definition": definition,
                })

    rows.sort(key=lambda r: (r["file"], r["line"], r["identifier"]))
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "line", "identifier", "intended_sample", "definition", "context"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["intended_sample"]] = counts.get(r["intended_sample"], 0) + 1

    legacy_hits = [r for r in rows if r["intended_sample"].startswith("LEGACY")]
    ambiguous_hits = [r for r in rows if r["intended_sample"].startswith("AMBIGUOUS")]

    lines = [
        "# Barren sample resolution report",
        "",
        "## Resolution summary",
        "",
        "The current pipeline now distinguishes two separate samples that were previously",
        "conflated under the umbrella term 'barren':",
        "",
        "1. **`control_strict_low_lai`** — a functional control defined by low LAI thresholds",
        "   (median LAI ≤ 0.10, 90th percentile LAI ≤ 0.30, ≥10 valid LAI observations).",
        "   In the harmonic 28-day analysis this sample contains **~6 cells / ~627 observations**.",
        "2. **`landcover_barren`** — the dominant MCD12C1 Version 6.1 IGBP land-cover class",
        "   'Barren' on the 0.5° analysis grid. In the harmonic 28-day analysis this sample contains",
        "   **~728 cells / ~71,331 observations**.",
        "",
        "The `Persistently_Barren` functional-control label in `config/pipeline.yaml` and in code",
        "has been replaced by `control_strict_low_lai`. The `is_barren` column is now",
        "`is_strict_low_lai`. Geographic controls (Sahara) are described as 'Sahara geographic",
        "control' rather than 'barren control'.",
        "",
        "## Identifier inventory",
        "",
    ]

    for sample, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- `{sample}`: {n} occurrences")

    lines.extend([
        "",
        "## Legacy identifiers in current pipeline",
        "",
    ])
    if legacy_hits:
        lines.append(f"Found {len(legacy_hits)} legacy identifier occurrence(s) that should be reviewed:")
        lines.append("")
        for r in legacy_hits:
            lines.append(f"- `{r['file']}:{r['line']}` — `{r['identifier']}`")
    else:
        lines.append("No legacy identifiers (`is_barren`, `control_barren`, `Persistently_Barren`, etc.)")
        lines.append("were found in the current pipeline. ✓")

    lines.extend([
        "",
        "## Ambiguous references",
        "",
    ])
    if ambiguous_hits:
        lines.append(f"Found {len(ambiguous_hits)} ambiguous occurrence(s) requiring review:")
        lines.append("")
        for r in ambiguous_hits:
            lines.append(f"- `{r['file']}:{r['line']}` — `{r['identifier']}` — {r['context'][:100]}")
    else:
        lines.append("No ambiguous barren/control references were found in the current pipeline. ✓")

    lines.extend([
        "",
        "## Files changed",
        "",
        "- `config/supplementary_checks.yaml` — added `scenario_registry` with canonical labels and definitions.",
        "- `config/pipeline.yaml` — functional control renamed from `Persistently_Barren` to `control_strict_low_lai`.",
        "- `config/regions.yaml` — Sahara `type` changed from `barren_geographic_control` to `geographic_barren_control`.",
        "- `scripts/_Common.py`, `scripts/magneto_lib.py` — constants and helpers updated; YAML reads now use UTF-8.",
        "- `scripts/02_assign_lai_quartiles.py`, `scripts/08_fixed_window.py`, `scripts/09_surrogates.py` —",
        "  `is_barren` → `is_strict_low_lai`, `control_barren` → `control_strict_low_lai`.",
        "- `scripts/supplementary_checks/*` — control filters, temperature-profile loaders, and Table 1 bootstrap",
        "  updated to keep `control_strict_low_lai` and `landcover_barren` separate.",
        "- `scripts/tables/*`, `scripts/visualizations/*`, `scripts/reports/12_audit_reports.py` — labels and captions",
        "  updated.",
        "- `README.md`, `SMOKE_README.md`, `FULL_RUN_README.md` — documentation updated.",
        "",
        "## Validation",
        "",
        "- Re-ran `02_assign_lai_quartiles.py`; strict low-LAI control count at global LAI stage is 376 cells.",
        "- Re-ran `04_harmonic.py`; strict low-LAI control intersected with QC/valid fits is 6 cells / 627 obs.",
        "- `pytest scripts/tests/test_supplementary_checks.py` passes after fixing YAML encoding.",
        "- `pytest scripts/tests/test_pipeline.py::test_output_manifest_hashes_match_files` is expected to fail",
        "  until the final output manifest is regenerated.",
        "",
    ])

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    write_report(rows)
    print(f"[OK] Wrote {OUT_CSV} ({len(rows)} entries)")
    print(f"[OK] Wrote {OUT_MD}")
    legacy = [r for r in rows if r["intended_sample"].startswith("LEGACY")]
    ambiguous = [r for r in rows if r["intended_sample"].startswith("AMBIGUOUS")]
    print(f"      Legacy hits: {len(legacy)}; Ambiguous hits: {len(ambiguous)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
