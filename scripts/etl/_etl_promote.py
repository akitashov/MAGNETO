#!/usr/bin/env python3
"""ETL candidate ↔ canonical comparison and promotion helper.

Each upstream ETL writes a candidate file next to the canonical intermediate.
This module compares the candidate against the existing canonical (if any) on:

  * file format and schema;
  * row count and primary-key uniqueness;
  * spatial grid step and coordinate uniqueness;
  * date range;
  * numeric summary statistics (mean, std, min, max, quantiles) within tolerance.

If the comparison passes, the candidate is promoted to canonical.  If it fails,
a detailed JSON report is written and the caller receives a non-PASS status so
that the pipeline can stop for operator review instead of silently replacing
validated data.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import PARQUET_ENGINE, PROJECT_ROOT, atomic_write  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "results" / "etl_promotion_audits"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Default tolerances for numeric summaries (relative).
DEFAULT_REL_TOL = 1e-5
DEFAULT_ABS_TOL = 1e-9
BATCH_SIZE = 2_000_000


def _now_str() -> str:
    return datetime.now().isoformat()


def _write_report(name: str, report: dict[str, Any]) -> Path:
    path = REPORT_DIR / f"{name}_promotion_audit.json"
    atomic_write(report, path, fmt="json")
    return path


def _load_minimal(path: Path, fmt: str | None, columns: list[str] | None) -> pd.DataFrame:
    fmt = fmt or path.suffix.lstrip(".")
    if fmt == "parquet":
        return pd.read_parquet(path, columns=columns, engine=PARQUET_ENGINE)
    if fmt == "feather":
        return pd.read_feather(path, columns=columns)
    raise ValueError(f"Unsupported format: {fmt}")


def _schema_names(path: Path, fmt: str | None) -> set[str]:
    fmt = fmt or path.suffix.lstrip(".")
    if fmt == "parquet":
        return set(pq.ParquetFile(path).schema.names)
    df = pd.read_feather(path)
    return set(df.columns)


def _grid_step(ids: pd.Series) -> float:
    """Return the smallest positive difference between unique coordinate IDs."""
    vals = pd.to_numeric(ids, errors="coerce").dropna().unique()
    vals = np.sort(vals)
    if len(vals) < 2:
        return np.nan
    diffs = np.diff(vals)
    return float(diffs[diffs > 0].min())


def _numeric_summary(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in cols:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            out[col] = {"present": True, "n_valid": 0}
            continue
        q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        out[col] = {
            "present": True,
            "n_valid": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
            "q01": float(q[0.01]),
            "q25": float(q[0.25]),
            "q50": float(q[0.5]),
            "q75": float(q[0.75]),
            "q99": float(q[0.99]),
        }
    return out


def _compare_numeric(
    cur: dict[str, Any],
    ref: dict[str, Any],
    rel_tol: float,
    abs_tol: float,
) -> list[str]:
    diffs: list[str] = []
    for col in set(cur) & set(ref):
        c = cur[col]
        r = ref[col]
        if not c.get("present") or not r.get("present"):
            continue
        if c.get("n_valid", 0) != r.get("n_valid", 0):
            diffs.append(f"{col}: n_valid {r.get('n_valid')} → {c.get('n_valid')}")
            continue
        for stat in ("mean", "std", "min", "max", "q01", "q25", "q50", "q75", "q99"):
            if stat not in c or stat not in r:
                continue
            a, b = float(c[stat]), float(r[stat])
            if not (np.isfinite(a) and np.isfinite(b)):
                if not (np.isnan(a) and np.isnan(b)):
                    diffs.append(f"{col}.{stat}: reference={r[stat]}, current={c[stat]}")
                continue
            if abs(a - b) > max(rel_tol * max(abs(a), abs(b)), abs_tol):
                diffs.append(f"{col}.{stat}: reference={b:.9g}, current={a:.9g}")
    return diffs


def _compare_sets(cur: set, ref: set, label: str) -> list[str]:
    diffs: list[str] = []
    extra = cur - ref
    missing = ref - cur
    if extra:
        diffs.append(f"{label}: {len(extra)} extra items (e.g. {list(extra)[:5]})")
    if missing:
        diffs.append(f"{label}: {len(missing)} missing items (e.g. {list(missing)[:5]})")
    return diffs


def _compare_feather(
    canonical: Path,
    candidate: Path,
    spec: dict[str, Any],
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    """Full in-memory comparison for moderately sized feather files."""
    required = spec.get("required", [])
    key_cols = spec.get("key_cols", [])
    metric_cols = spec.get("metric_cols", [])
    grid_cols = spec.get("grid_cols", ["lat_id", "lon_id"])

    report: dict[str, Any] = {"canonical": str(canonical), "candidate": str(candidate)}

    ref_names = _schema_names(canonical, "feather")
    cur_names = _schema_names(candidate, "feather")
    schema_diff = _compare_sets(cur_names, ref_names, "schema columns")
    if schema_diff:
        report["passed"] = False
        report["diffs"] = schema_diff
        return report

    ref = _load_minimal(canonical, "feather", required + metric_cols + grid_cols + key_cols)
    cur = _load_minimal(candidate, "feather", required + metric_cols + grid_cols + key_cols)

    diffs: list[str] = []

    if len(cur) != len(ref):
        diffs.append(f"row_count: reference={len(ref):,}, current={len(cur):,}")

    if "date" in cur.columns and "date" in ref.columns:
        cur["date"] = pd.to_datetime(cur["date"])
        ref["date"] = pd.to_datetime(ref["date"])
        if cur["date"].min() != ref["date"].min() or cur["date"].max() != ref["date"].max():
            diffs.append(
                f"date_range: reference={ref['date'].min().date()} → {ref['date'].max().date()}, "
                f"current={cur['date'].min().date()} → {cur['date'].max().date()}"
            )

    for col in grid_cols:
        if col in cur.columns and col in ref.columns:
            cur_step = _grid_step(cur[col])
            ref_step = _grid_step(ref[col])
            if not (np.isnan(cur_step) and np.isnan(ref_step)) and cur_step != ref_step:
                diffs.append(f"grid_step.{col}: reference={ref_step}, current={cur_step}")
            cur_ids = set(pd.to_numeric(cur[col], errors="coerce").dropna().unique())
            ref_ids = set(pd.to_numeric(ref[col], errors="coerce").dropna().unique())
            diffs.extend(_compare_sets(cur_ids, ref_ids, f"grid_ids.{col}"))

    if key_cols:
        ref_dups = int(ref.duplicated(subset=key_cols).sum())
        cur_dups = int(cur.duplicated(subset=key_cols).sum())
        if ref_dups or cur_dups:
            diffs.append(f"duplicate_keys: reference={ref_dups}, current={cur_dups}")
        if set(key_cols).issubset(ref.columns) and set(key_cols).issubset(cur.columns):
            ref_keys = set(zip(*[ref[c] for c in key_cols]))
            cur_keys = set(zip(*[cur[c] for c in key_cols]))
            diffs.extend(_compare_sets(cur_keys, ref_keys, "primary_keys"))

    cur_num = _numeric_summary(cur, metric_cols)
    ref_num = _numeric_summary(ref, metric_cols)
    diffs.extend(_compare_numeric(cur_num, ref_num, rel_tol, abs_tol))

    report["passed"] = len(diffs) == 0
    report["diffs"] = diffs
    report["row_count"] = {"reference": len(ref), "current": len(cur)}
    report["numeric_summary"] = {"reference": ref_num, "current": cur_num}
    return report


def _compare_parquet_metadata(
    canonical: Path,
    candidate: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Lightweight metadata comparison for very large parquet files."""
    required = spec.get("required", [])
    pf_ref = pq.ParquetFile(canonical)
    pf_cur = pq.ParquetFile(candidate)
    report: dict[str, Any] = {"canonical": str(canonical), "candidate": str(candidate)}
    diffs: list[str] = []

    ref_names = set(pf_ref.schema.names)
    cur_names = set(pf_cur.schema.names)
    diffs.extend(_compare_sets(cur_names, ref_names, "schema columns"))

    missing_ref = [c for c in required if c not in ref_names]
    missing_cur = [c for c in required if c not in cur_names]
    if missing_ref:
        diffs.append(f"canonical_missing_required: {missing_ref}")
    if missing_cur:
        diffs.append(f"candidate_missing_required: {missing_cur}")

    if pf_ref.metadata.num_rows != pf_cur.metadata.num_rows:
        diffs.append(
            f"row_count: reference={pf_ref.metadata.num_rows:,}, "
            f"current={pf_cur.metadata.num_rows:,}"
        )

    report["passed"] = len(diffs) == 0
    report["diffs"] = diffs
    report["row_count"] = {"reference": pf_ref.metadata.num_rows, "current": pf_cur.metadata.num_rows}
    return report


def compare_and_promote(
    canonical: Path,
    candidate: Path,
    spec: dict[str, Any],
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    backup: bool = True,
) -> dict[str, Any]:
    """Compare candidate to canonical and promote if equivalent.

    Returns a report dict with ``passed`` bool.  If canonical does not exist,
    the candidate is promoted unconditionally.
    """
    name = spec.get("name", canonical.stem)
    report: dict[str, Any] = {
        "name": name,
        "timestamp": _now_str(),
        "canonical": str(canonical.relative_to(PROJECT_ROOT)),
        "candidate": str(candidate.relative_to(PROJECT_ROOT)),
    }

    if not candidate.exists():
        report["passed"] = False
        report["diffs"] = ["candidate file does not exist"]
        _write_report(name, report)
        return report

    if not canonical.exists():
        print(f"[PROMOTE] {name}: canonical missing; promoting candidate")
        candidate.replace(canonical)
        report["passed"] = True
        report["action"] = "promoted (canonical missing)"
        _write_report(name, report)
        return report

    fmt = spec.get("fmt", canonical.suffix.lstrip("."))
    if fmt == "feather":
        cmp = _compare_feather(canonical, candidate, spec, rel_tol, abs_tol)
    elif fmt == "parquet":
        cmp = _compare_parquet_metadata(canonical, candidate, spec)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    report.update(cmp)

    if report["passed"]:
        print(f"[PROMOTE] {name}: candidate equivalent to canonical; promoting")
        if backup:
            backup_path = canonical.with_suffix(canonical.suffix + f".bak.{datetime.now():%Y%m%dT%H%M%S}")
            shutil.copy2(canonical, backup_path)
            report["canonical_backup"] = str(backup_path.relative_to(PROJECT_ROOT))
        candidate.replace(canonical)
        report["action"] = "promoted (equivalent)"
    else:
        print(f"[BLOCK] {name}: candidate differs from canonical; promotion blocked")
        for d in cmp["diffs"][:20]:
            print(f"  - {d}")
        report["action"] = "blocked (differences detected)"

    report_path = _write_report(name, report)
    print(f"  report: {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Promote an ETL candidate if equivalent to canonical")
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fmt", default="feather")
    parser.add_argument("--key-cols", nargs="+", default=[])
    parser.add_argument("--metric-cols", nargs="+", default=[])
    parser.add_argument("--grid-cols", nargs="+", default=["lat_id", "lon_id"])
    parser.add_argument("--required-cols", nargs="+", default=[])
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)

    spec = {
        "name": args.name,
        "fmt": args.fmt,
        "key_cols": args.key_cols,
        "metric_cols": args.metric_cols,
        "grid_cols": args.grid_cols,
        "required": args.required_cols,
    }
    report = compare_and_promote(
        args.canonical,
        args.candidate,
        spec,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        backup=not args.no_backup,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
