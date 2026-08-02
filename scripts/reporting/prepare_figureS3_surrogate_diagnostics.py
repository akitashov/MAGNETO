"""Prepare source data for Supplementary Figure S3: temporal surrogate diagnostics.

Reads the surrogate summary table and the full null distributions, extracts the
relevant subset (SIF 771 nm, 28-day SII window, full and persistently vegetated
samples, harmonic and cyclic spline, three temporal null models), and writes a
clean source CSV used by the plotting script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    summary_path = project_root / "results" / "surrogate_summary.csv"
    nulls_path = project_root / "results" / "surrogate_null_distributions.parquet"
    output_path = project_root / "reports" / "figures" / "figureS3_surrogate_diagnostics_source.csv"

    print(f"Reading summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    print(f"  rows={len(summary):,}")

    print(f"Reading null distributions: {nulls_path}")
    nulls = pq.read_table(nulls_path).to_pandas()
    print(f"  rows={len(nulls):,}")

    sample_types = ["pooled_full", "control_vegetated"]
    methods = ["harmonic", "cyclic_spline"]
    surrogate_modes = ["year_perm", "circ_shift", "block_perm"]
    window_label = "sii_28d"

    sample_labels = {
        "pooled_full": "Full sample",
        "control_vegetated": "Persistently vegetated",
    }
    method_labels = {
        "harmonic": "Harmonic",
        "cyclic_spline": "Cyclic spline",
    }
    surrogate_labels = {
        "year_perm": "Year permutation",
        "circ_shift": "Circular shift",
        "block_perm": "Block permutation",
    }
    panel_map = {
        "year_perm": "A",
        "circ_shift": "B",
        "block_perm": "C",
    }

    # Filter summary to the figure-relevant subset.
    sub_summary = summary[
        (summary["sample_type"].isin(sample_types))
        & (summary["method"].isin(methods))
        & (summary["surrogate_mode"].isin(surrogate_modes))
        & (summary["sii_window"] == window_label)
    ].copy()

    if sub_summary.empty:
        raise ValueError("No matching rows found in surrogate_summary.csv.")

    # Compute empirical null quantiles from the parquet distributions.
    sub_nulls = nulls[
        (nulls["sample_type"].isin(sample_types))
        & (nulls["method"].isin(methods))
        & (nulls["surrogate_mode"].isin(surrogate_modes))
        & (nulls["window"] == window_label)
    ].copy()

    null_quantiles = (
        sub_nulls.groupby(["sample_type", "method", "surrogate_mode", "window"])["rho"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(columns={0.025: "null_q025", 0.5: "null_median", 0.975: "null_q975"})
        .reset_index()
    )

    # Merge observed values and metadata from the summary with computed nulls.
    keep_cols = [
        "sample_type",
        "method",
        "surrogate_mode",
        "sii_window",
        "rho_obs",
        "p_value",
        "n_obs_used",
        "n_completed",
        "n_requested",
    ]
    merged = pd.merge(
        sub_summary[keep_cols],
        null_quantiles,
        on=["sample_type", "method", "surrogate_mode"],
        how="left",
        suffixes=("", "_from_nulls"),
    )

    # Determine display order within each panel (full harmonic, full spline,
    # vegetated harmonic, vegetated spline).
    order_map = {
        ("pooled_full", "harmonic"): 1,
        ("pooled_full", "cyclic_spline"): 2,
        ("control_vegetated", "harmonic"): 3,
        ("control_vegetated", "cyclic_spline"): 4,
    }
    merged["display_order"] = merged.apply(
        lambda r: order_map[(r["sample_type"], r["method"])], axis=1
    )
    merged["panel"] = merged["surrogate_mode"].map(panel_map)
    merged["figure_id"] = "figureS3"
    merged["sample_label"] = merged["sample_type"].map(sample_labels)
    merged["surrogate_label"] = merged["surrogate_mode"].map(surrogate_labels)
    merged["method_label"] = merged["method"].map(method_labels)

    # Consistency check: p_value cannot be smaller than 1/(n_completed+1).
    # Allow a tiny tolerance for floating-point representation of exact minima.
    min_reportable = 1.0 / (merged["n_completed"] + 1)
    inconsistent = merged["p_value"] < (min_reportable - 1e-12)
    if inconsistent.any():
        n_bad = inconsistent.sum()
        print(
            f"WARNING: {n_bad} row(s) have p_value < 1/(n_completed+1) "
            f"(minimum reportable).",
            file=sys.stderr,
        )
        for _, row in merged[inconsistent].iterrows():
            min_p = 1.0 / (row["n_completed"] + 1)
            print(
                f"  {row['sample_type']}-{row['method']}-{row['surrogate_mode']}: "
                f"p={row['p_value']:.6f} < {min_p:.6f}",
                file=sys.stderr,
            )

    # Final column order matching the requested output schema.
    out_cols = [
        "figure_id",
        "panel",
        "display_order",
        "sample_type",
        "sample_label",
        "method",
        "surrogate_mode",
        "surrogate_label",
        "sii_window",
        "rho_obs",
        "p_value",
        "null_median",
        "null_q025",
        "null_q975",
        "n_completed",
        "n_requested",
        "n_obs_used",
    ]
    out = merged[out_cols].sort_values(["panel", "display_order"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {len(out)} rows to {output_path}")
    print(f"p_value range: {out['p_value'].min():.6f} to {out['p_value'].max():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
