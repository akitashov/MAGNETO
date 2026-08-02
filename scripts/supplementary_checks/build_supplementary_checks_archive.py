#!/usr/bin/env python3
"""
Build three clean, separate archives for the supplementary_checks layer:

  1. magneto_supplementary_checks_scripts.zip
     All scripts (core + supplementary_checks + tests + reports) and configs.

  2. magneto_supplementary_checks_results.zip
     Machine-readable results (CSV, parquet, manifests, consistency checks).

  3. magneto_supplementary_checks_reports.zip
     Human-readable reports, prepared analytical data, and execution logs.

Figures, __pycache__, stale revision-named files, and traceback files are
excluded.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXCLUDES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
}

EXCLUDE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".tmp",
    ".traceback",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
)

EXCLUDE_NAME_PATTERNS = (
    "revision_manifest",
    "revision_consistency_checks",
    "run_revision",
    "pytest_revision",
    "test_revision",
    "S09_build_revision_outputs",
    "_revision_common",
)


def should_include(path: Path, rel: Path) -> bool:
    """Return True if *path* should be included in an archive."""
    if path.is_dir():
        return False
    if any(part in DEFAULT_EXCLUDES for part in rel.parts):
        return False
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if any(pattern in rel.name for pattern in EXCLUDE_NAME_PATTERNS):
        return False
    # Never bundle rendered figures.
    if "figures/supplementary_checks" in str(rel):
        return False
    return True


def collect_files(include_paths: list[Path], root: Path) -> list[Path]:
    files: list[Path] = []
    for item in include_paths:
        if not item.exists():
            print(f"[WARN] Skipping missing path: {item.relative_to(root)}")
            continue
        if item.is_dir():
            for path in sorted(item.rglob("*")):
                rel = path.relative_to(root)
                if should_include(path, rel):
                    files.append(path)
        else:
            rel = item.relative_to(root)
            if should_include(item, rel):
                files.append(item)
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def write_archive(name: str, files: list[Path], root: Path) -> Path:
    out = root / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, str(path.relative_to(root)))
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[OK] {out.name}: {len(files)} files, {out.stat().st_size:,} bytes")
    print(f"[OK] SHA-256: {sha256}")
    return out


def build_archives(root: Path) -> dict[str, Path]:
    # Archive 1: code + configuration.
    code_paths = [
        root / "scripts",
        root / "pipeline.sh",
        root / "config" / "supplementary_checks.yaml",
        root / "config" / "pipeline.yaml",
        root / "config" / "regions.yaml",
        root / "environment.yml",
        root / "requirements.txt",
        root / "requirements-lock.txt",
        root / "pyproject.toml",
        root / "README.md",
        root / "GIT_COMMIT_INFO.txt",
    ]
    code_files = collect_files(code_paths, root)

    # Archive 2: numerical results.
    results_paths = [
        root / "results" / "supplementary_checks",
    ]
    results_files = collect_files(results_paths, root)

    # Archive 3: reports, prepared analytical data, and logs.
    report_paths = [
        root / "reports" / "supplementary_checks",
        root / "data" / "interim" / "supplementary_checks",
        root / "logs" / "supplementary_checks",
    ]
    report_files = collect_files(report_paths, root)

    archives = {
        "magneto_supplementary_checks_scripts.zip": code_files,
        "magneto_supplementary_checks_results.zip": results_files,
        "magneto_supplementary_checks_reports.zip": report_files,
    }

    built: dict[str, Path] = {}
    for name, files in archives.items():
        built[name] = write_archive(name, files, root)
    return built


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build supplementary_checks archives")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    build_archives(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
