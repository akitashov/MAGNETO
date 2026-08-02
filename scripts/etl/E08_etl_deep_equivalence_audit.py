#!/usr/bin/env python3
"""E08_etl_deep_equivalence_audit.py — deep upstream ETL equivalence audit.

This is the heavy-weight variant of the ETL audit trio.  It is intended for a
machine with enough RAM (native Kubuntu or similar) and performs:

  * full verification of primary-key uniqueness for files that fit in memory;
  * numeric distribution summaries (mean, std, quantiles, missing counts);
  * date-range and year-level sanity checks;
  * optional comparison against a previously saved reference manifest.

For very large files (e.g. ERA5) the script falls back to a chunked strategy:
exact mean/min/max via Welford's online algorithm, approximate quantiles from a
deterministic per-batch random sample, and within-batch duplicate checks.  It
never materialises a multi-hundred-gigabyte file in RAM.

Use E06 for a cheap schema/metadata-only audit, and E07 for a memory-safe
chunked audit on modest hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Allow imports of shared utilities from scripts/ regardless of where this
# ETL script is executed.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _Common import (
    FILE_ERA5,
    FILE_MODIS,
    FILE_OMNI,
    FILE_SIF,
    PARQUET_ENGINE,
    PROJECT_ROOT,
    atomic_write,
    full_sha256,
    setup_dirs,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "ETL_DEEP_AUDIT.md"
REFERENCE_DIR = PROJECT_ROOT / "results" / "etl_deep_audit_references"
REFERENCE_PATH = REFERENCE_DIR / "etl_deep_audit_reference.json"
LATEST_PATH = REFERENCE_DIR / "etl_deep_audit_latest.json"

BATCH_SIZE = 2_000_000
SAMPLE_N = 2_000_000
PACKED_KEY_THRESHOLD = 200_000_000
DATE_EPOCH = pd.Timestamp("1970-01-01")

DATASETS: dict[str, dict[str, Any]] = {
    "MODIS": {
        "path": FILE_MODIS,
        "fmt": "parquet",
        "key_cols": ["date", "lat_id", "lon_id"],
        "required": ["date", "lat_id", "lon_id", "lai",
                     "cloud_fraction", "aerosol_fraction", "quality_flag"],
        "metric_cols": ["lai", "cloud_fraction", "aerosol_fraction", "quality_flag"],
    },
    "OMNI2": {
        "path": FILE_OMNI,
        "fmt": "feather",
        "key_cols": ["date"],
        "required": ["date", "sii_mean", "kp_mean", "f10_7_mean"],
        "metric_cols": ["sii_mean", "kp_mean", "f10_7_mean"],
    },
    "OCO-2 SIF": {
        "path": FILE_SIF,
        "fmt": "feather",
        "key_cols": ["date", "lat_id", "lon_id"],
        "required": ["date", "lat_id", "lon_id", "sif_771nm", "region_flags"],
        "metric_cols": ["sif_757nm", "sif_771nm", "sif_740nm", "sif_stress_index"],
    },
    "ERA5": {
        "path": FILE_ERA5,
        "fmt": "parquet",
        "key_cols": ["date", "lat_id", "lon_id"],
        "required": ["date", "lat_id", "lon_id", "temp_c", "temp_c_ma10", "vpd", "par"],
        "metric_cols": ["temp_c", "temp_c_ma10", "vpd", "par"],
    },
}


def _as_python(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types for JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _as_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(v) for v in value]
    return value


def _to_date_days(series: pd.Series) -> np.ndarray:
    """Return days since 1970-01-01 as int64."""
    dt = pd.to_datetime(series, errors="coerce").dt.floor("D")
    return ((dt - DATE_EPOCH) // pd.Timedelta("1D")).values.astype(np.int64)


def _pack_keys(days: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pack (date, lat_id, lon_id) into a uint64 array for uniqueness checks."""
    d = days.astype(np.int64)
    la = lat.astype(np.int64)
    lo = lon.astype(np.int64)
    packed = (
        ((d + (1 << 31)).astype(np.uint64) & 0xFFFFFFFF) << 32
        | ((la + 20000).astype(np.uint64) & 0xFFFF) << 16
        | ((lo + 40000).astype(np.uint64) & 0xFFFF)
    )
    return packed


class _Welford:
    """Online mean, variance, min and max."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        n = x.size
        if n == 0:
            return
        self.min = float(min(self.min, float(x.min())))
        self.max = float(max(self.max, float(x.max())))
        batch_mean = float(x.mean())
        batch_m2 = float(((x - batch_mean) ** 2).sum())
        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        new_count = self.count + n
        new_mean = (self.mean * self.count + batch_mean * n) / new_count
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * self.count * n / new_count
        )
        self.mean = new_mean
        self.count = new_count

    def std(self) -> float:
        if self.count < 2:
            return np.nan
        return float(np.sqrt(self.m2 / (self.count - 1)))

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "n_valid": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        return {
            "n_valid": int(self.count),
            "mean": float(self.mean),
            "std": self.std(),
            "min": float(self.min),
            "max": float(self.max),
        }


def _read_full(path: Path, fmt: str, columns: list[str]) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path, columns=columns, engine=PARQUET_ENGINE)
    if fmt == "feather":
        return pd.read_feather(path, columns=columns)
    raise ValueError(f"Unsupported format: {fmt}")


def _schema_check(path: Path, required: list[str]) -> tuple[set[str], list[str]]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    missing = [c for c in required if c not in names]
    return names, missing


def _numeric_summary_full(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for col in tqdm(cols, desc="  numeric summary", leave=False):
        if col not in df.columns:
            summary[col] = {"present": False}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        n_valid = int(len(valid))
        n_missing = int(len(s) - n_valid)
        if n_valid == 0:
            summary[col] = {
                "present": True,
                "n_valid": n_valid,
                "n_missing": n_missing,
            }
            continue
        q = valid.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        summary[col] = {
            "present": True,
            "n_valid": n_valid,
            "n_missing": n_missing,
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=1)),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "q01": float(q[0.01]),
            "q25": float(q[0.25]),
            "q50": float(q[0.5]),
            "q75": float(q[0.75]),
            "q99": float(q[0.99]),
        }
    return summary


def _date_summary(df: pd.DataFrame) -> dict[str, Any]:
    dt = pd.to_datetime(df["date"], errors="coerce")
    year_counts = dt.dt.year.value_counts().sort_index().astype(int).to_dict()
    month_counts = dt.dt.month.value_counts().sort_index().astype(int).to_dict()
    return {
        "date_min": str(dt.min().date()),
        "date_max": str(dt.max().date()),
        "year_counts": {int(k): int(v) for k, v in year_counts.items()},
        "month_counts": {int(k): int(v) for k, v in month_counts.items()},
    }


def _key_uniqueness_full(df: pd.DataFrame, key_cols: list[str]) -> dict[str, Any]:
    n_rows = len(df)
    dups = int(df.duplicated(subset=key_cols).sum())
    n_unique = n_rows - dups
    return {
        "n_rows": n_rows,
        "n_unique_keys": n_unique,
        "n_duplicate_keys": dups,
        "unique": dups == 0,
    }


def _global_key_uniqueness_packed(path: Path, key_cols: list[str]) -> dict[str, Any] | None:
    """Exact global uniqueness via packed uint64 keys. Returns None if too large."""
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    if n_rows > PACKED_KEY_THRESHOLD:
        return None

    chunks = []
    for batch in tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE, columns=key_cols),
        total=max(1, n_rows // BATCH_SIZE),
        desc="  uniqueness",
        leave=False,
    ):
        df = batch.to_pandas()
        days = _to_date_days(df["date"])
        lat = df["lat_id"].values.astype(np.int64)
        lon = df["lon_id"].values.astype(np.int64)
        chunks.append(_pack_keys(days, lat, lon))
        del df, batch

    if not chunks:
        return {"n_rows": 0, "n_unique_keys": 0, "n_duplicate_keys": 0, "unique": True}

    packed = np.concatenate(chunks)
    packed_sorted = np.sort(packed)
    dups = int((np.diff(packed_sorted) == 0).sum())
    return {
        "n_rows": int(n_rows),
        "n_unique_keys": int(n_rows - dups),
        "n_duplicate_keys": dups,
        "unique": dups == 0,
    }


def _chunked_audit(
    path: Path,
    label: str,
    required: list[str],
    metric_cols: list[str],
    sample_n: int,
    seed: int,
) -> dict[str, Any]:
    """Deep audit for very large parquet files without loading them fully."""
    print(f"\n[AUDIT] {label} (chunked deep)")
    pf = pq.ParquetFile(path)
    schema_names, missing = _schema_check(path, required)
    n_rows = pf.metadata.num_rows
    print(f"  rows={n_rows:,}, columns={len(schema_names)}")

    if missing:
        return {
            "exists": True,
            "n_rows": n_rows,
            "missing_columns": missing,
            "ok": False,
        }

    date_min = pd.Timestamp.max
    date_max = pd.Timestamp.min
    year_counts: dict[int, int] = {}
    month_counts: dict[int, int] = {}
    bad_batches: list[str] = []

    welfords = {col: _Welford() for col in metric_cols}
    samples: dict[str, list[np.ndarray]] = {col: [] for col in metric_cols}
    rng = np.random.default_rng(seed)

    n_batches = max(1, n_rows // BATCH_SIZE)
    sample_fraction = min(1.0, sample_n / max(1, n_rows))

    for i, batch in enumerate(
        tqdm(
            pf.iter_batches(batch_size=BATCH_SIZE, columns=required),
            total=n_batches,
            desc=f"  {label} batches",
            leave=False,
        )
    ):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        batch_min = df["date"].min()
        batch_max = df["date"].max()
        if pd.notna(batch_min):
            date_min = min(date_min, batch_min)
        if pd.notna(batch_max):
            date_max = max(date_max, batch_max)

        for yr, cnt in df["date"].dt.year.value_counts().items():
            year_counts[int(yr)] = year_counts.get(int(yr), 0) + int(cnt)
        for mo, cnt in df["date"].dt.month.value_counts().items():
            month_counts[int(mo)] = month_counts.get(int(mo), 0) + int(cnt)

        # Within-batch duplicate detection only; global uniqueness is reported
        # as approximate for files above PACKED_KEY_THRESHOLD.
        if any(key in df.columns for key in ("lat_id", "lon_id")):
            key_cols = [c for c in ["date", "lat_id", "lon_id"] if c in df.columns]
            dups = int(df.duplicated(subset=key_cols).sum())
            if dups:
                bad_batches.append(f"batch {i}: {dups} duplicate keys")

        for col in metric_cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(np.float64)
            welfords[col].update(vals)
            if vals.size:
                n_sample = max(1, int(vals.size * sample_fraction)) if sample_fraction < 1.0 else vals.size
                n_sample = min(n_sample, vals.size)
                if n_sample > 0:
                    idx = rng.choice(vals.size, size=n_sample, replace=False)
                    samples[col].append(vals[idx])

        del df, batch

    numeric_summary: dict[str, dict[str, Any]] = {}
    for col in metric_cols:
        w = welfords[col]
        base = w.summary()
        base["present"] = True
        if samples[col]:
            pooled = np.concatenate(samples[col])
            q = np.nanpercentile(pooled, [1, 25, 50, 75, 99])
            base.update({
                "q01": float(q[0]),
                "q25": float(q[1]),
                "q50": float(q[2]),
                "q75": float(q[3]),
                "q99": float(q[4]),
                "sample_size": int(len(pooled)),
            })
        else:
            base.update({"q01": np.nan, "q25": np.nan, "q50": np.nan, "q75": np.nan, "q99": np.nan})
        numeric_summary[col] = base

    ok = len(missing) == 0 and len(bad_batches) == 0

    return {
        "exists": True,
        "n_rows": n_rows,
        "missing_columns": missing,
        "date_min": str(date_min.date()) if pd.notna(date_min) else "?",
        "date_max": str(date_max.date()) if pd.notna(date_max) else "?",
        "year_counts": year_counts,
        "month_counts": month_counts,
        "numeric_summary": numeric_summary,
        "bad_batches": bad_batches,
        "note": "Global uniqueness was not checked because the file exceeds the packed-key threshold.",
        "ok": ok,
    }


def _full_audit(
    path: Path,
    label: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Deep audit for files that fit comfortably in memory."""
    print(f"\n[AUDIT] {label} (full deep)")
    required = spec["required"]
    key_cols = spec["key_cols"]
    metric_cols = spec["metric_cols"]

    if spec["fmt"] == "parquet":
        names, missing = _schema_check(path, required)
        n_rows = pq.ParquetFile(path).metadata.num_rows
    else:
        # Feather: read minimal columns to infer schema
        df_cols = pd.read_feather(path, columns=[required[0]])
        n_rows = len(df_cols)
        del df_cols
        cols_in_file = set(pd.read_feather(path).columns)
        names = cols_in_file
        missing = [c for c in required if c not in names]

    print(f"  rows={n_rows:,}, columns={len(names)}")
    if missing:
        return {
            "exists": True,
            "n_rows": n_rows,
            "missing_columns": missing,
            "ok": False,
        }

    df = _read_full(path, spec["fmt"], list(names))
    key_info = _key_uniqueness_full(df, key_cols)
    date_info = _date_summary(df)
    numeric_summary = _numeric_summary_full(df, metric_cols)

    return {
        "exists": True,
        "n_rows": key_info["n_rows"],
        "n_unique_keys": key_info["n_unique_keys"],
        "n_duplicate_keys": key_info["n_duplicate_keys"],
        "unique_keys": key_info["unique"],
        "missing_columns": missing,
        "date_min": date_info["date_min"],
        "date_max": date_info["date_max"],
        "year_counts": date_info["year_counts"],
        "month_counts": date_info["month_counts"],
        "numeric_summary": numeric_summary,
        "ok": key_info["unique"],
    }


def _audit_dataset(name: str, spec: dict[str, Any], sample_n: int, seed: int) -> dict[str, Any]:
    path: Path = spec["path"]
    if not path.exists():
        print(f"\n[AUDIT] {name}\n  file not found: {path}")
        return {"exists": False, "ok": False, "missing_columns": spec["required"]}

    fmt = spec["fmt"]
    n_rows: int | None = None
    if fmt == "parquet":
        try:
            n_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            print(f"  [WARN] could not read parquet metadata: {exc}")

    result: dict[str, Any]
    if fmt == "parquet" and n_rows is not None and n_rows > PACKED_KEY_THRESHOLD:
        result = _chunked_audit(path, name, spec["required"], spec["metric_cols"], sample_n, seed)
    else:
        result = _full_audit(path, name, spec)

    result["file"] = str(path.relative_to(PROJECT_ROOT))
    result["sha256"] = full_sha256(path)
    result["format"] = fmt
    result["columns"] = sorted(set(spec["required"]) | set(spec["metric_cols"]))
    return result


def _load_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_value(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, float) or isinstance(b, float):
        if not (np.isfinite(a) and np.isfinite(b)):
            return (bool(np.isnan(a)) and bool(np.isnan(b))) or (a == b)
        return abs(float(a) - float(b)) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    return a == b


def _compare_with_reference(
    current: dict[str, Any],
    reference: dict[str, Any],
    rel_tol: float,
    abs_tol: float,
) -> list[str]:
    diffs: list[str] = []
    current_ds = current.get("datasets", {})
    reference_ds = reference.get("datasets", {})

    for name in current_ds:
        cur = current_ds[name]
        ref = reference_ds.get(name)
        if ref is None:
            diffs.append(f"{name}: no reference entry")
            continue

        for field in ("n_rows", "n_unique_keys"):
            if field in cur and field in ref:
                if cur[field] != ref[field]:
                    diffs.append(f"{name}: {field} changed from {ref[field]} to {cur[field]}")

        if "date_min" in cur and "date_min" in ref:
            if cur["date_min"] != ref["date_min"] or cur.get("date_max") != ref.get("date_max"):
                diffs.append(f"{name}: date range changed")

        cur_num = cur.get("numeric_summary", {})
        ref_num = ref.get("numeric_summary", {})
        for col in set(cur_num) & set(ref_num):
            cstats = cur_num[col]
            rstats = ref_num[col]
            for stat in ("mean", "std", "min", "max", "q01", "q25", "q50", "q75", "q99"):
                if stat in cstats and stat in rstats:
                    if not _compare_value(cstats[stat], rstats[stat], rel_tol, abs_tol):
                        diffs.append(
                            f"{name}.{col}.{stat}: reference={rstats[stat]}, current={cstats[stat]}"
                        )

    return diffs


def _write_report(
    results: dict[str, dict[str, Any]],
    reference_diffs: list[str],
    save_reference: bool,
) -> None:
    lines = [
        "# MAGNETO upstream ETL deep audit",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "This report combines full in-memory checks for moderately sized files with",
        "chunked Welford + reservoir-sampling checks for files larger than the",
        f"packed-key threshold ({PACKED_KEY_THRESHOLD:,} rows).",
        "",
        "## Results",
        "",
        "| Dataset | Rows | Unique keys | Duplicates | Date range | OK |",
        "|---------|------|-------------|------------|------------|----|",
    ]
    for name, res in results.items():
        rows = f"{res.get('n_rows', 'N/A'):,}" if "n_rows" in res else "N/A"
        unique = f"{res.get('n_unique_keys', 'N/A'):,}" if "n_unique_keys" in res else "N/A"
        dups = f"{res.get('n_duplicate_keys', 'N/A'):,}" if "n_duplicate_keys" in res else "N/A"
        drange = f"{res.get('date_min', '?')} → {res.get('date_max', '?')}"
        ok = "✅" if res.get("ok") else "❌"
        lines.append(f"| {name} | {rows} | {unique} | {dups} | {drange} | {ok} |")

    lines.extend(["", "## File hashes", ""])
    for name, res in results.items():
        sha = res.get("sha256", "N/A")
        file = res.get("file", "N/A")
        lines.append(f"- **{name}**: `{file}` → `{sha}`")

    if reference_diffs:
        lines.extend(["", "## Reference comparison", ""])
        for diff in reference_diffs:
            lines.append(f"- {diff}")
    else:
        lines.extend(["", "## Reference comparison", "", "No differences against reference."])

    lines.extend(["", "## Notes", ""])
    for name, res in results.items():
        if res.get("note"):
            lines.append(f"- **{name}**: {res['note']}")
        if res.get("bad_batches"):
            lines.append(f"- **{name}** batch issues:")
            for msg in res["bad_batches"][:10]:
                lines.append(f"  - {msg}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")

    if save_reference:
        out = {
            "generated": datetime.now().isoformat(),
            "datasets": _as_python(results),
        }
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(out, REFERENCE_PATH, fmt="json")
        atomic_write(out, LATEST_PATH, fmt="json")
        print(f"Reference manifest written: {REFERENCE_PATH}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MAGNETO deep upstream ETL equivalence audit")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["ALL"],
        default=["ALL"],
        help="Datasets to audit (default: ALL)",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="Directory containing the reference manifest",
    )
    parser.add_argument(
        "--save-reference",
        action="store_true",
        help="Write the current audit results as the new reference manifest",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=SAMPLE_N,
        help="Reservoir target sample size for chunked metric summaries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for reference comparison",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reference comparison",
    )
    args = parser.parse_args(argv)

    print("MAGNETO — deep upstream ETL equivalence audit")
    print(f"Project root: {PROJECT_ROOT}")
    setup_dirs()

    selected = list(DATASETS.keys()) if "ALL" in args.datasets else args.datasets
    print(f"Selected datasets: {', '.join(selected)}")

    reference_path = args.reference_dir / "etl_deep_audit_reference.json"
    reference = _load_reference(reference_path)

    results: dict[str, dict[str, Any]] = {}
    for name in tqdm(selected, desc="Datasets"):
        results[name] = _audit_dataset(name, DATASETS[name], args.sample_n, args.seed)

    current_manifest = {"generated": datetime.now().isoformat(), "datasets": _as_python(results)}
    reference_diffs = []
    if reference:
        reference_diffs = _compare_with_reference(current_manifest, reference, args.rel_tol, args.abs_tol)
        if reference_diffs:
            print("\n[REF] Differences against reference:")
            for d in reference_diffs[:20]:
                print(f"  - {d}")
        else:
            print("\n[REF] No differences against reference.")
    else:
        print("\n[REF] No reference manifest found; comparison skipped.")

    _write_report(results, reference_diffs, args.save_reference)

    all_ok = all(r.get("ok") for r in results.values())
    print("\nAUDIT PASSED" if all_ok else "\nAUDIT FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
