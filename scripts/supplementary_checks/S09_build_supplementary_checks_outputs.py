#!/usr/bin/env python3
"""
S09_build_supplementary_checks_outputs.py — Collect supplementary checks outputs, build manifests,
write README, and run consistency checks.

This script is intentionally lightweight: it does not recompute statistics. It
reads the products written by S01-S08, validates their presence, checks that the
21- and 28-day pooled SIF 771 nm estimates are identical to the validated core
tables, and produces a machine-readable manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from _supplementary_checks_common import (
    PROJECT_ROOT,
    SUPPLEMENTARY_CHECKS_RESULTS,
    SUPPLEMENTARY_CHECKS_FIGURES,
    SUPPLEMENTARY_CHECKS_REPORTS,
    SUPPLEMENTARY_CHECKS_DATA,
    make_supplementary_checks_dirs,
    atomic_write,
    full_sha256,
    load_main_fixed_window_results,
    load_supplementary_checks_config,
    supplementary_checks_config_hash,
    git_commit,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build supplementary checks outputs and manifest")
    p.add_argument("--rebuild-figures", action="store_true", default=False)
    p.add_argument("--smoke-test", action="store_true",
                   help="Expect S06b outputs in results/supplementary_checks/smoke/.")
    return p.parse_args(argv)


def _expected_supplementary_checks_outputs(cfg: dict, smoke: bool = False) -> dict[str, Path]:
    o = cfg["outputs"]
    s06b_dir = SUPPLEMENTARY_CHECKS_RESULTS / "smoke" if smoke else SUPPLEMENTARY_CHECKS_RESULTS
    return {
        "window_scan_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["window_scan_csv"],
        "kp_comparison_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["kp_comparison_csv"],
        "f107_comparison_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["f107_comparison_csv"],
        "saa_control_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["saa_control_csv"],
        "saa_control_surrogate_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["saa_control_surrogate_csv"],
        "control_temperature_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["control_temperature_csv"],
        "control_temperature_surrogate_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["control_temperature_surrogate_csv"],
        "control_temperature_null_distributions": SUPPLEMENTARY_CHECKS_RESULTS / "control_temperature_profiles_null_distributions.parquet",
        "landcover_grid_parquet": SUPPLEMENTARY_CHECKS_DATA / o["landcover_grid_parquet"],
        "landcover_diagnostics_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["landcover_diagnostics_csv"],
        "landcover_analysis_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["landcover_analysis_csv"],
        "landcover_analysis_surrogate_csv": SUPPLEMENTARY_CHECKS_RESULTS / "landcover_analysis_surrogates.csv",
        "landcover_analysis_null_distributions": SUPPLEMENTARY_CHECKS_RESULTS / "landcover_analysis_null_distributions.parquet",
        "landcover_temperature_inference_csv": s06b_dir / "landcover_temperature_inference.csv",
        "landcover_temperature_bootstrap_parquet": s06b_dir / "landcover_temperature_bootstrap.parquet",
        "landcover_temperature_null_distributions": s06b_dir / "landcover_temperature_null_distributions.parquet",
        "landcover_temperature_contrasts_csv": s06b_dir / "landcover_temperature_contrasts.csv",
        "landcover_temperature_composition_csv": s06b_dir / "landcover_temperature_composition.csv",
        "landcover_lai_composition_csv": s06b_dir / "landcover_lai_composition.csv",
        "landcover_temperature_audit_json": s06b_dir / "landcover_temperature_audit.json",
        "graphical_abstract_csv": SUPPLEMENTARY_CHECKS_RESULTS / "graphical_abstract_matrix.csv",
        "temperature_scenarios_harmonic_csv": SUPPLEMENTARY_CHECKS_RESULTS / "temperature_scenarios_harmonic.csv",
        "temperature_scenarios_spline_csv": SUPPLEMENTARY_CHECKS_RESULTS / "temperature_scenarios_spline.csv",
        "lai_quartile_bootstrap_draws_parquet": SUPPLEMENTARY_CHECKS_RESULTS / "lai_quartile_bootstrap_draws.parquet",
        "lai_quartile_bootstrap_summary_csv": SUPPLEMENTARY_CHECKS_RESULTS / "lai_quartile_bootstrap_summary.csv",
        "table1_bootstrap_summary_csv": SUPPLEMENTARY_CHECKS_RESULTS / "table1_bootstrap_summary.csv",
        "sii_environment_correlation_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["sii_environment_correlation_csv"],
        "driver_r2_shapley_long_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["driver_r2_shapley_long_csv"],
        "driver_r2_shapley_models_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["driver_r2_shapley_models_csv"],
        "driver_r2_shapley_by_window_csv": SUPPLEMENTARY_CHECKS_RESULTS / o["driver_r2_shapley_by_window_csv"],
        "supplementary_checks_consistency_checks_csv": SUPPLEMENTARY_CHECKS_RESULTS / "supplementary_checks_consistency_checks.csv",
        "window_scan_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["window_scan_fig"],
        "kp_comparison_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["kp_comparison_fig"],
        "f107_comparison_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["f107_comparison_fig"],
        "saa_control_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["saa_control_fig"],
        "landcover_analysis_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["landcover_analysis_fig"],
        "graphical_abstract_fig": SUPPLEMENTARY_CHECKS_FIGURES / o["graphical_abstract_fig"],
    }


def _build_manifest(cfg: dict, smoke: bool = False) -> dict[str, Any]:
    expected = _expected_supplementary_checks_outputs(cfg, smoke=smoke)
    entries = []
    for name, path in expected.items():
        # In the no-figure final run we do not list figure assets in the manifest.
        if "_fig" in name:
            continue
        if path.exists():
            entries.append({
                "name": name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": full_sha256(path),
                "size_bytes": path.stat().st_size,
                "present": True,
            })
        else:
            entries.append({
                "name": name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": "",
                "size_bytes": 0,
                "present": False,
            })
    return {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "supplementary_checks_config_hash": supplementary_checks_config_hash(),
        "git_commit": git_commit(),
        "entries": entries,
    }


def _core_consistency_check(cfg: dict) -> pd.DataFrame:
    """Compare 21/28-day pooled SIF 771 nm estimates with the core final table."""
    core = load_main_fixed_window_results()
    core = core[
        (core["sample_type"] == "pooled_full") &
        (core["method"].isin(["harmonic", "cyclic_spline"])) &
        (core["sii_window_days"].isin(cfg["windows"]["fixed"]))
    ][["method", "sii_window_days", "spearman_rho", "ols_slope_per_1nT", "n_obs", "n_cells"]]

    rows = []
    for _, r in core.iterrows():
        rows.append({
            "check": "core_pooled_present",
            "method": r["method"],
            "window": r["sii_window_days"],
            "expected_rho": float(r["spearman_rho"]),
            "status": "ok",
            "detail": f"n_obs={int(r['n_obs'])}, n_cells={int(r['n_cells'])}",
        })

    # Verify supplementary checks window-scan file contains the same 21/28-day values.
    scan_path = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["window_scan_csv"]
    if scan_path.exists():
        scan = pd.read_csv(scan_path)
        scan = scan[scan["sample_label"] == "pooled"]
        for _, r in core.iterrows():
            sub = scan[
                (scan["method"] == r["method"]) &
                (scan["sii_window_days"] == r["sii_window_days"])
            ]
            if sub.empty:
                status = "missing"
                diff = None
            else:
                diff = float(sub["spearman_rho"].iloc[0] - r["spearman_rho"])
                status = "ok" if abs(diff) < 1e-12 else "mismatch"
            rows.append({
                "check": "window_scan_matches_core",
                "method": r["method"],
                "window": r["sii_window_days"],
                "expected_rho": float(r["spearman_rho"]),
                "observed_rho": float(sub["spearman_rho"].iloc[0]) if not sub.empty else None,
                "rho_diff": diff,
                "status": status,
                "detail": "" if status == "ok" else f"diff={diff}",
            })
    else:
        rows.append({
            "check": "window_scan_matches_core",
            "method": "all",
            "window": None,
            "status": "skipped",
            "detail": f"{scan_path} not found",
        })

    return pd.DataFrame(rows)


def _write_readme(manifest: dict, consistency: pd.DataFrame, cfg: dict) -> None:
    lines = [
        "# MAGNETO supplementary checks / Supplementary Outputs",
        "",
        "This directory collects additional sensitivity and robustness analyses that are",
        "kept separate from the primary SIF 771 nm pipeline. They are intended for",
        "Supplementary Materials and internal validation, not as primary evidence.",
        "",
        "## Contents",
        "",
    ]
    for e in manifest["entries"]:
        status = "✓" if e["present"] else "✗ missing"
        lines.append(f"- {status} `{e['path']}` ({e['size_bytes']:,} bytes)")
    lines += [
        "",
        "## Consistency checks",
        "",
    ]
    if consistency.empty:
        lines.append("No checks performed.")
    else:
        cols = [c for c in consistency.columns if consistency[c].notna().any()]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in consistency.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Reproduction",
        "",
        "Run the stages in order using `scripts/supplementary_checks/run_supplementary_checks.sh`. Each stage",
        "is independent and can be commented out if not needed. The scripts do not",
        "modify any validated core result files.",
        "",
        f"- supplementary checks config hash: `{manifest['supplementary_checks_config_hash']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Created: {manifest['created_utc']}",
    ]
    out = SUPPLEMENTARY_CHECKS_REPORTS / cfg["outputs"]["supplementary_checks_readme"]
    out.write_text("\n".join(lines), encoding="utf-8")


def _rebuild_figures_if_needed(cfg: dict) -> None:
    """Optionally re-run S07 to ensure the graphical abstract figure exists.

    Temperature-scenario figures are now produced by the dedicated plotting
    script in scripts/visualizations/ and are not rebuilt here.
    """
    expected = _expected_supplementary_checks_outputs(cfg)
    missing = [expected["graphical_abstract_fig"]]
    if all(p.exists() for p in missing):
        return
    print("[INFO] Rebuilding missing graphical abstract figure via S07 ...")
    import importlib.util
    s07 = Path(__file__).with_name("S07_make_graphical_abstract_matrix.py")
    spec = importlib.util.spec_from_file_location(s07.stem, s07)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv=[])


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_supplementary_checks_config()
    make_supplementary_checks_dirs()

    print("=" * 64)
    print("MAGNETO supplementary checks — Build outputs and manifest")
    print("=" * 64)

    if args.rebuild_figures:
        _rebuild_figures_if_needed(cfg)

    print("[1/3] Running consistency checks ...")
    consistency = _core_consistency_check(cfg)
    out_consistency = SUPPLEMENTARY_CHECKS_RESULTS / "supplementary_checks_consistency_checks.csv"
    atomic_write(consistency, out_consistency)
    print(f"  Written: {out_consistency}")
    print(consistency[["check", "method", "window", "status"]].to_string(index=False))

    print("[2/3] Building manifest ...")
    manifest = _build_manifest(cfg, smoke=args.smoke_test)
    out_json = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["supplementary_checks_manifest_json"]
    atomic_write(manifest, out_json)
    print(f"  Written: {out_json}")

    manifest_csv = SUPPLEMENTARY_CHECKS_RESULTS / cfg["outputs"]["supplementary_checks_manifest_csv"]
    atomic_write(pd.DataFrame(manifest["entries"]), manifest_csv)
    print(f"  Written: {manifest_csv}")

    print("[3/3] Writing README ...")
    _write_readme(manifest, consistency, cfg)
    out_readme = SUPPLEMENTARY_CHECKS_REPORTS / cfg["outputs"]["supplementary_checks_readme"]
    print(f"  Written: {out_readme}")

    missing = [e["name"] for e in manifest["entries"] if not e["present"]]
    if missing:
        print(f"\n[WARN] Missing outputs: {missing}")
    else:
        print("\n[OK] All expected supplementary checks outputs present.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
