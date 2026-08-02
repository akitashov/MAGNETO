#!/usr/bin/env python3
"""Clean and regenerate all MAGNETO figures.

This script is the single entry point for rebuilding the manuscript figures.
It:

1. Removes old PNG/PDF/source CSV/description Markdown outputs for the active
   visualization scripts (but never deletes manifest, validation, or inventory
   files).
2. Runs the reporting preparation step for Figure S3.
3. Runs every active plotting script in ``scripts/visualizations/``.
4. Verifies that each script produced a PNG and a Markdown description.

Run from the repository root:

    python scripts/run_all_figures.py

To clean without regenerating:

    python scripts/run_all_figures.py --clean-only

To skip the clean step:

    python scripts/run_all_figures.py --no-clean
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VIS_DIR = SCRIPTS_DIR / "visualizations"
REPORTING_DIR = SCRIPTS_DIR / "reporting"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LOGS_DIR = PROJECT_ROOT / "logs" / "figures"

PREPARE_SCRIPT = REPORTING_DIR / "prepare_figureS3_surrogate_diagnostics.py"

PROTECTED_FILES = {
    "supplementary_figures_manifest.csv",
    "supplementary_figures_validation.md",
    "supplementary_figure_input_inventory.csv",
}


def discover_active_scripts() -> list[Path]:
    """Return sorted list of active figure plotting scripts."""
    scripts = [
        p for p in VIS_DIR.glob("figure*.py")
        if p.is_file() and p.name != "_figure_text_export.py"
    ]
    scripts.sort(key=lambda p: p.name)
    return scripts


def expected_basenames(script: Path) -> list[str]:
    """Return the output basenames produced by a plotting script."""
    return [script.stem]


def basename_outputs(basename: str) -> list[Path]:
    """Return output files owned by a plotting script basename."""
    return [
        FIGURES_DIR / f"{basename}.png",
        FIGURES_DIR / f"{basename}.pdf",
        FIGURES_DIR / f"{basename}_source.csv",
        FIGURES_DIR / f"{basename}_description.md",
        LOGS_DIR / f"{basename}.log",
    ]


def clean_active_outputs(scripts: list[Path], dry_run: bool = False) -> list[Path]:
    """Remove outputs for the listed active scripts plus stray PDFs."""
    removed: list[Path] = []

    for script in scripts:
        for basename in expected_basenames(script):
            for path in basename_outputs(basename):
                if path.exists():
                    if not dry_run:
                        path.unlink()
                    removed.append(path)

    # Remove any remaining PDFs in the figures directory.
    for pdf in FIGURES_DIR.glob("*.pdf"):
        if pdf.name not in PROTECTED_FILES:
            if not dry_run:
                pdf.unlink()
            removed.append(pdf)

    return removed


def run_script(script: Path, env: dict[str, str]) -> tuple[bool, str]:
    """Run one script and return (ok, short_message)."""
    log_path = LOGS_DIR / f"{script.stem}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except Exception as exc:  # pragma: no cover
        return False, f"launcher error: {exc}"

    if result.returncode != 0:
        return False, f"exit code {result.returncode} (see {log_path})"
    return True, f"ok (log: {log_path})"


def run_all(scripts: list[Path], no_clean: bool = False) -> int:
    """Clean outputs and regenerate figures. Returns exit code."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not no_clean:
        removed = clean_active_outputs(scripts)
        print(f"Removed {len(removed)} old output files.")
    else:
        print("Skipped clean step (--no-clean).")

    env = {**dict(subprocess.os.environ), "PYTHONPATH": str(SCRIPTS_DIR)}

    results: list[tuple[str, bool, str]] = []

    # Figure S3 needs a prepared source table before plotting.
    if PREPARE_SCRIPT.exists():
        print(f"\nRunning preparation: {PREPARE_SCRIPT.name}")
        ok, msg = run_script(PREPARE_SCRIPT, env)
        results.append((PREPARE_SCRIPT.name, ok, msg))
        if not ok:
            print(f"  FAILED: {msg}")
        else:
            print(f"  OK: {msg}")
    else:
        print(f"\nWarning: preparation script not found: {PREPARE_SCRIPT}")

    print(f"\nRunning {len(scripts)} visualization scripts:")
    for script in scripts:
        print(f"  {script.name} ...", end=" ", flush=True)
        ok, msg = run_script(script, env)
        results.append((script.name, ok, msg))
        print("OK" if ok else f"FAILED: {msg}")

    # Verification: every active script must leave a PNG and description.
    print("\nVerification:")
    missing: list[str] = []
    for script in scripts:
        for basename in expected_basenames(script):
            png = FIGURES_DIR / f"{basename}.png"
            md = FIGURES_DIR / f"{basename}_description.md"
            if not png.exists():
                missing.append(f"{basename}.png")
            if not md.exists():
                missing.append(f"{basename}_description.md")

    if missing:
        print("  MISSING outputs:")
        for name in missing:
            print(f"    - {name}")
    else:
        print("  All expected PNG and Markdown outputs are present.")

    # Summary table.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    failures = [name for name, ok, _ in results if not ok]
    print(f"Scripts run:    {len(results)}")
    print(f"Failures:       {len(failures)}")
    if failures:
        print("Failed scripts:")
        for name in failures:
            print(f"  - {name}")

    return 1 if failures or missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean and regenerate all MAGNETO figures.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only remove old outputs; do not run plotting scripts.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the clean step and regenerate over existing files.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="",
        help="Comma-separated list of figure basenames to run (e.g. figureS2,figureS3).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of figure basenames to skip.",
    )
    args = parser.parse_args(argv)

    all_scripts = discover_active_scripts()

    if args.figures:
        wanted = {s.strip() for s in args.figures.split(",") if s.strip()}
        scripts = [s for s in all_scripts if s.stem in wanted]
        missing = wanted - {s.stem for s in scripts}
        if missing:
            print(f"Warning: requested scripts not found: {sorted(missing)}")
    else:
        scripts = all_scripts

    if args.exclude:
        skip = {s.strip() for s in args.exclude.split(",") if s.strip()}
        scripts = [s for s in scripts if s.stem not in skip]

    if args.clean_only:
        removed = clean_active_outputs(scripts)
        print(f"Cleaned {len(removed)} files for {len(scripts)} active figure scripts.")
        return 0

    return run_all(scripts, no_clean=args.no_clean)


if __name__ == "__main__":
    sys.exit(main())
