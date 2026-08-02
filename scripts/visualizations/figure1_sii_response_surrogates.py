#!/usr/bin/env python3
"""
Figure 1: observed SII–SIF response with temporal-surrogate inset.

Main panel
----------
Observed exposure–response profile for the primary analysis:
    outcome     = SIF 771-nm harmonic residual
    exposure    = 28-day trailing-mean SII
    sample      = pooled primary sample

The exposure is divided into equal-count bins. The plotted value is the mean
standardized SIF residual in each bin. The shaded interval is a 95% cell-cluster
bootstrap interval obtained by resampling spatial cells with replacement.

Inset
-----
The three temporal-surrogate null distributions are overlaid:
    year permutation
    circular shift
    30-day block permutation

The vertical line is the observed pooled Spearman rho. Empirical p-values are
calculated using the plus-one two-sided rule unless a compatible summary table
is available.

This script performs visualization plus descriptive bin/bootstrap calculations.
It does not rerun ETL, detrending, exposure construction, or surrogate generation.

Recommended location:
    <project_root>/scripts/visualizations/figure1_sii_response_surrogates.py

Expected inputs:
    data/processed/harmonic_analysis.parquet
    results/surrogate_null_distributions.parquet

Optional:
    results/surrogate_summary.csv

Outputs:
    reports/figures/figure1_sii_response_surrogates.png
    reports/figures/figure1_sii_response_surrogates.pdf
    reports/figures/figure1_sii_response_surrogates_source.csv
    reports/figures/figure1_sii_response_surrogates_null_summary.csv
"""

from __future__ import annotations
from _figure_text_export import export_figure_text

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Allow imports of project helpers (e.g., magneto_lib) from scripts/visualizations.
_SCRIPTS_DIR = str(PROJECT_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import magneto_lib as ml  # noqa: E402

DATA_PARQUET = PROJECT_ROOT / "data" / "processed" / "harmonic_analysis.parquet"
EXPOSURE_PARQUET = PROJECT_ROOT / "data" / "interim" / "global_qc.parquet"
NULL_PARQUET = PROJECT_ROOT / "results" / "surrogate_null_distributions.parquet"
SURROGATE_SUMMARY_CSV = PROJECT_ROOT / "results" / "surrogate_summary.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figure1_sii_response_surrogates.png"
OUTPUT_PDF = OUTPUT_DIR / "figure1_sii_response_surrogates.pdf"
OUTPUT_SOURCE_CSV = OUTPUT_DIR / "figure1_sii_response_surrogates_source.csv"
OUTPUT_NULL_CSV = OUTPUT_DIR / "figure1_sii_response_surrogates_null_summary.csv"


# =============================================================================
# ANALYTICAL SPECIFICATION
# =============================================================================

WINDOW_DAYS = 28
ADDITIONAL_WINDOWS = [21]
ALL_WINDOWS = sorted(set([WINDOW_DAYS] + ADDITIONAL_WINDOWS))
N_EXPOSURE_BINS = 10
N_BOOTSTRAP = 1000
SEED = 20260726

METHOD = "harmonic"
OUTCOME = "sif_771nm"
SAMPLE = "pooled"

STANDARDIZE_RESIDUAL = True

# If automatic detection selects the wrong field, set an exact column name here.
COLUMN_OVERRIDES = {
    "residual": None,
    "exposure": None,
    "cell_id": None,
}

# Optional exact filters for the null-distribution table.
# Example:
# NULL_FILTER_OVERRIDES = {
#     "method": "harmonic",
#     "sii_window": "sii_28d",
#     "sample_type": "pooled",
#     "outcome": "sif_771nm",
# }
NULL_FILTER_OVERRIDES: dict[str, object] = {}


# =============================================================================
# DISPLAY
# =============================================================================

FIGSIZE = (10.5, 6.4)

OBSERVED_COLOR = "#173f73"
OBSERVED_FILL = "#4f81bd"

WINDOW_LINE_STYLES = {
    28: {
        "color": OBSERVED_COLOR,
        "fill": OBSERVED_FILL,
        "linewidth": 2.6,
        "markersize": 6,
        "alpha": 1.0,
        "zorder": 3,
    },
    10: {
        "color": "#54278f",
        "fill": "#bcbddc",
        "linewidth": 1.2,
        "markersize": 3,
        "alpha": 0.75,
        "zorder": 2,
    },
    21: {
        "color": "#8c510a",
        "fill": "#d8b365",
        "linewidth": 1.4,
        "markersize": 3.5,
        "alpha": 0.85,
        "zorder": 2,
    },

    60: {
        "color": "#a50f15",
        "fill": "#fc9272",
        "linewidth": 1.2,
        "markersize": 3,
        "alpha": 0.75,
        "zorder": 2,
    },
}

NULL_STYLES = {
    "year_perm": {
        "label": "Year permutation",
        "color": "#7f7f7f",
    },
    "circ_shift": {
        "label": "Circular shift",
        "color": "#d95f02",
    },
    "block_perm": {
        "label": "30-day block permutation",
        "color": "#1b9e77",
    },
}

MODE_ORDER = ["year_perm", "circ_shift", "block_perm"]


# =============================================================================
# COLUMN DETECTION
# =============================================================================

DATA_COLUMN_CANDIDATES = {
    "residual": [
        "sif_771_residual",
        "sif771_residual",
        "sif_771nm_residual",
        "sif_residual",
        "residual",
        "residual_anomaly",
        "sif_anomaly",
    ],
    "exposure": [
        f"sii_{WINDOW_DAYS}d",
        f"sii_{WINDOW_DAYS}",
        f"sii{WINDOW_DAYS}",
        f"sii_roll_{WINDOW_DAYS}",
        f"sii_rolling_{WINDOW_DAYS}",
        f"sii_window_{WINDOW_DAYS}",
        f"sii_{WINDOW_DAYS}day",
        f"sii_{WINDOW_DAYS}_day",
        f"sii_ma{WINDOW_DAYS}",
        f"sii_cumulative_{WINDOW_DAYS}",
        f"sii_cum_{WINDOW_DAYS}d",
    ],
    "cell_id": [
        "cell_id",
        "grid_id",
        "grid_cell_id",
        "cell_key",
        "cell",
    ],
}

NULL_COLUMN_CANDIDATES = {
    "mode": [
        "surrogate_mode",
        "null_model",
        "mode",
    ],
    "rho": [
        "surrogate_rho",
        "spearman_rho",
        "rho",
        "null_rho",
    ],
    "method": [
        "method",
        "detrending_method",
    ],
    "window": [
        "sii_window",
        "sii_window_days",
        "window_days",
        "window",
    ],
    "sample": [
        "sample_type",
        "sample",
        "scenario",
        "analysis_sample",
    ],
    "outcome": [
        "outcome",
        "target",
        "sif_outcome",
    ],
}

SUMMARY_COLUMN_CANDIDATES = {
    "mode": ["surrogate_mode", "null_model", "mode"],
    "p": ["p_value", "empirical_p", "p_empirical"],
    "method": ["method", "detrending_method"],
    "window": ["sii_window", "sii_window_days", "window_days", "window"],
    "sample": ["sample_type", "sample", "scenario", "analysis_sample"],
    "outcome": ["outcome", "target", "sif_outcome"],
}


def resolve_column(
    df: pd.DataFrame,
    role: str,
    candidates: dict[str, list[str]],
    override: str | None = None,
    required: bool = True,
) -> str | None:
    """Resolve one column name with an optional exact override."""
    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"Configured {role!r} column {override!r} is absent.\n"
                f"Available columns:\n{list(df.columns)}"
            )
        return override

    for candidate in candidates[role]:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Could not identify the {role!r} column.\n"
            f"Tried: {candidates[role]}\n"
            f"Available columns:\n{list(df.columns)}\n"
            f"Set COLUMN_OVERRIDES[{role!r}] to the exact name."
        )
    return None



def resolve_exposure_column(df: pd.DataFrame) -> str:
    """
    Resolve the trailing-mean SII exposure column.

    First uses the explicit override and canonical names. If they are absent,
    searches for columns containing both "sii" and the selected window length.
    """
    override = COLUMN_OVERRIDES["exposure"]
    if override is not None:
        return resolve_column(
            df,
            "exposure",
            DATA_COLUMN_CANDIDATES,
            override=override,
        )

    direct = resolve_column(
        df,
        "exposure",
        DATA_COLUMN_CANDIDATES,
        required=False,
    )
    if direct is not None:
        return direct

    pattern_matches = [
        column
        for column in df.columns
        if "sii" in str(column).lower()
        and str(WINDOW_DAYS) in str(column).lower()
    ]

    if len(pattern_matches) == 1:
        return pattern_matches[0]

    raise ValueError(
        "Could not identify the trailing-mean SII exposure column in "
        f"{EXPOSURE_PARQUET}.\n"
        f"Canonical names tried: {DATA_COLUMN_CANDIDATES['exposure']}\n"
        f"Pattern matches containing 'sii' and '{WINDOW_DAYS}': "
        f"{pattern_matches}\n"
        f"Available columns:\n{list(df.columns)}\n"
        "Set COLUMN_OVERRIDES['exposure'] to the exact name."
    )


def build_cell_id(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return an existing cell ID or derive one from lat_id and lon_id."""
    override = COLUMN_OVERRIDES["cell_id"]
    existing = resolve_column(
        df,
        "cell_id",
        DATA_COLUMN_CANDIDATES,
        override=override,
        required=False,
    )
    if existing is not None:
        return df[existing].astype(str), existing

    required = {"lat_id", "lon_id"}
    if required.issubset(df.columns):
        cell_id = (
            df["lat_id"].astype(str)
            + "__"
            + df["lon_id"].astype(str)
        )
        return cell_id, "derived_from_lat_id_lon_id"

    raise ValueError(
        "Could not identify or derive a spatial cell identifier.\n"
        "Expected an existing cell ID column or both lat_id and lon_id.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def choose_merge_keys(
    residual_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
) -> list[str]:
    """
    Choose the strongest common observation keys.

    The residual table is normally indexed by date, lat_id, and lon_id. If the
    exposure source only contains a global daily series, merging by date is
    accepted.
    """
    preferred = ["date", "lat_id", "lon_id"]
    keys = [
        column
        for column in preferred
        if column in residual_df.columns and column in exposure_df.columns
    ]

    if "date" not in keys:
        raise ValueError(
            "Cannot merge harmonic residuals with trailing-mean SII: no common "
            "'date' column was found."
        )

    return keys


def prepare_exposure_table(
    exposure_raw: pd.DataFrame,
    exposure_col: str,
    merge_keys: list[str],
) -> pd.DataFrame:
    """
    Reduce the exposure source to one value per merge key.

    Duplicate rows with identical exposure are collapsed. Conflicting exposure
    values for the same merge key raise an error to avoid row multiplication.
    """
    exposure = exposure_raw[merge_keys + [exposure_col]].copy()
    exposure[exposure_col] = pd.to_numeric(
        exposure[exposure_col], errors="coerce"
    )
    exposure = exposure.dropna(subset=merge_keys + [exposure_col])

    conflicts = (
        exposure.groupby(merge_keys, dropna=False)[exposure_col]
        .nunique(dropna=True)
    )
    conflicting_keys = conflicts[conflicts > 1]

    if not conflicting_keys.empty:
        examples = conflicting_keys.head(10).reset_index().to_dict("records")
        raise ValueError(
            "The exposure source contains conflicting trailing-mean SII values "
            f"for the same merge key(s) {merge_keys}.\n"
            f"Examples: {examples}"
        )

    return exposure.drop_duplicates(subset=merge_keys)


# =============================================================================
# PRIMARY DATA
# =============================================================================

def build_exposure_from_omni(
    windows: list[int] = ALL_WINDOWS,
) -> pd.DataFrame:
    """Build the trailing-mean SII exposure table directly from OMNI raw data."""
    daily_sii = ml.build_daily_sii()
    exposure = ml.compute_sii_windows(
        daily_sii,
        sii_col=ml.SII_RAW_COL,
        windows=windows,
    )
    return exposure[["date"] + [f"sii_{w}d" for w in windows]].copy()


def load_primary_data() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load harmonic SIF residuals and merge trailing-mean SII exposure windows.

    harmonic_analysis.parquet contains the fitted SIF residuals. Cumulative
    geomagnetic exposure is not stored in global_qc.parquet, so it is rebuilt
    here from the OMNI daily SII series using the same logic as the core
    pipeline (shift-then-rolling-mean). If an exposure file is supplied via
    COLUMN_OVERRIDES or EXPOSURE_PARQUET contains a matching column, that
    source is used instead for backward compatibility.
    """
    if not DATA_PARQUET.exists():
        raise FileNotFoundError(
            f"Primary harmonic dataset not found:\n{DATA_PARQUET}"
        )

    residual_raw = pd.read_parquet(DATA_PARQUET)
    residual_col = resolve_column(
        residual_raw,
        "residual",
        DATA_COLUMN_CANDIDATES,
        COLUMN_OVERRIDES["residual"],
    )

    # Prefer an explicit exposure override or a compatible column in the legacy
    # exposure file; otherwise build exposure from OMNI for all windows.
    exposure_override = COLUMN_OVERRIDES.get("exposure")
    if exposure_override is not None:
        if not EXPOSURE_PARQUET.exists():
            raise FileNotFoundError(
                f"Trailing-mean SII source not found:\n{EXPOSURE_PARQUET}"
            )
        exposure_raw = pd.read_parquet(EXPOSURE_PARQUET)
        exposure_col = resolve_exposure_column(exposure_raw)
        exposure_source = str(EXPOSURE_PARQUET)
        exposure_cols = [exposure_col]
    elif (
        EXPOSURE_PARQUET.exists()
        and any(
            candidate in pd.read_parquet(EXPOSURE_PARQUET, columns=[]).columns
            for candidate in DATA_COLUMN_CANDIDATES["exposure"]
        )
    ):
        exposure_raw = pd.read_parquet(EXPOSURE_PARQUET)
        exposure_col = resolve_exposure_column(exposure_raw)
        exposure_source = str(EXPOSURE_PARQUET)
        exposure_cols = [exposure_col]
    else:
        exposure_raw = build_exposure_from_omni(ALL_WINDOWS)
        exposure_cols = [f"sii_{w}d" for w in ALL_WINDOWS]
        exposure_source = "built_from_omni_via_magneto_lib"

    merge_keys = choose_merge_keys(residual_raw, exposure_raw)
    exposure = exposure_raw.drop_duplicates(subset=merge_keys)

    residual_columns = list(dict.fromkeys(
        merge_keys + [residual_col, "lat_id", "lon_id"]
    ))
    residual_columns = [
        column for column in residual_columns
        if column in residual_raw.columns
    ]
    residual = residual_raw[residual_columns].copy()

    cell_id, cell_source = build_cell_id(residual)
    residual["cell_id"] = cell_id

    data = residual.merge(
        exposure,
        on=merge_keys,
        how="inner",
        validate="many_to_one",
    )
    data = data.rename(columns={residual_col: "sif_residual"})

    data["sif_residual"] = pd.to_numeric(
        data["sif_residual"], errors="coerce"
    )
    for col in exposure_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(
        subset=["sif_residual", "cell_id"] + exposure_cols
    )

    if data.empty:
        raise ValueError(
            "No finite observations remain after merging harmonic residuals "
            "with trailing-mean SII."
        )

    if STANDARDIZE_RESIDUAL:
        residual_sd = float(data["sif_residual"].std(ddof=1))
        if not np.isfinite(residual_sd) or residual_sd <= 0:
            raise ValueError("Residual standard deviation is not positive.")
        data["sif_plot"] = (
            data["sif_residual"] - data["sif_residual"].mean()
        ) / residual_sd
    else:
        data["sif_plot"] = data["sif_residual"]

    print(
        "Merged primary data using keys "
        f"{merge_keys}: {len(data):,} rows, "
        f"{data['cell_id'].nunique():,} cells."
    )

    return data, {
        "residual": residual_col,
        "exposure": ",".join(exposure_cols),
        "cell_id": cell_source,
        "merge_keys": ",".join(merge_keys),
        "exposure_source": exposure_source,
    }


def assign_equal_count_bins(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign equal-count exposure bins.

    rank(method='first') avoids qcut failure when the exposure contains many
    ties. Bin order still follows the original SII exposure.
    """
    out = data.copy()
    ranked = out["sii_exposure"].rank(method="first")
    out["exposure_bin"] = pd.qcut(
        ranked,
        q=N_EXPOSURE_BINS,
        labels=False,
    ).astype(int)

    observed_bins = sorted(out["exposure_bin"].unique().tolist())
    expected_bins = list(range(N_EXPOSURE_BINS))
    if observed_bins != expected_bins:
        raise ValueError(
            f"Expected exposure bins {expected_bins}, got {observed_bins}."
        )

    return out


def cluster_bootstrap_bin_means(
    data: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute observed bin means and cell-cluster bootstrap intervals.

    Cells are sampled with replacement. Repeatedly selected cells retain their
    multiplicity through bootstrap weights.
    """
    grouped = (
        data.groupby(["cell_id", "exposure_bin"], observed=True)["sif_plot"]
        .agg(["sum", "count"])
        .reset_index()
    )

    cell_values = pd.Index(grouped["cell_id"].drop_duplicates())
    cell_to_index = {cell: i for i, cell in enumerate(cell_values)}
    n_cells = len(cell_values)

    sum_matrix = np.zeros((n_cells, N_EXPOSURE_BINS), dtype=float)
    count_matrix = np.zeros((n_cells, N_EXPOSURE_BINS), dtype=float)

    for row in grouped.itertuples(index=False):
        i = cell_to_index[row.cell_id]
        j = int(row.exposure_bin)
        sum_matrix[i, j] = float(row.sum)
        count_matrix[i, j] = float(row.count)

    observed_sum = sum_matrix.sum(axis=0)
    observed_count = count_matrix.sum(axis=0)
    observed_mean = np.divide(
        observed_sum,
        observed_count,
        out=np.full(N_EXPOSURE_BINS, np.nan),
        where=observed_count > 0,
    )

    rng = np.random.default_rng(seed)
    bootstrap_means = np.full(
        (N_BOOTSTRAP, N_EXPOSURE_BINS),
        np.nan,
        dtype=float,
    )

    for replicate in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n_cells, size=n_cells)
        weights = np.bincount(sampled, minlength=n_cells).astype(float)

        boot_sum = weights @ sum_matrix
        boot_count = weights @ count_matrix
        bootstrap_means[replicate] = np.divide(
            boot_sum,
            boot_count,
            out=np.full(N_EXPOSURE_BINS, np.nan),
            where=boot_count > 0,
        )

    ci_low = np.nanquantile(bootstrap_means, 0.025, axis=0)
    ci_high = np.nanquantile(bootstrap_means, 0.975, axis=0)

    summary = (
        data.groupby("exposure_bin", observed=True)
        .agg(
            mean_sif=("sif_plot", "mean"),
            median_sii=("sii_exposure", "median"),
            sii_q25=("sii_exposure", lambda x: x.quantile(0.25)),
            sii_q75=("sii_exposure", lambda x: x.quantile(0.75)),
            n_obs=("sif_plot", "size"),
            n_cells=("cell_id", "nunique"),
        )
        .reset_index()
        .sort_values("exposure_bin")
    )

    summary["boot_ci_lo"] = ci_low
    summary["boot_ci_hi"] = ci_high
    summary["bin_number"] = summary["exposure_bin"] + 1
    summary["bin_label"] = [
        "Q1\nlow" if i == 1
        else f"Q{i}"
        if i < N_EXPOSURE_BINS
        else f"Q{i}\nhigh"
        for i in summary["bin_number"]
    ]

    if not np.allclose(
        summary["mean_sif"].to_numpy(dtype=float),
        observed_mean,
        equal_nan=True,
    ):
        raise RuntimeError("Observed bin means do not match matrix aggregation.")

    return summary, bootstrap_means


def compute_window_summaries(
    data: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], dict[int, float]]:
    """
    Compute observed rho, bins, and bootstrap intervals for every SII window.

    Each window is binned independently on its own exposure quantiles so that
    the x-axis remains comparable (equal-count deciles) while the SII scale
    varies across windows.
    """
    summaries: dict[int, pd.DataFrame] = {}
    rhos: dict[int, float] = {}

    for window in ALL_WINDOWS:
        exposure_col = f"sii_{window}d"
        if exposure_col not in data.columns:
            raise ValueError(
                f"Exposure column {exposure_col!r} not found in merged data."
            )

        sub = data.rename(
            columns={exposure_col: "sii_exposure"}
        )[["sif_residual", "sif_plot", "sii_exposure", "cell_id"]].copy()
        sub = assign_equal_count_bins(sub)

        rho = float(
            spearmanr(
                sub["sii_exposure"].to_numpy(dtype=float),
                sub["sif_residual"].to_numpy(dtype=float),
                nan_policy="omit",
            ).statistic
        )

        summary, _ = cluster_bootstrap_bin_means(
            sub, seed=SEED + window
        )
        summary["window_days"] = window
        summary["observed_rho"] = rho

        summaries[window] = summary
        rhos[window] = rho

    return summaries, rhos


# =============================================================================
# NULL DISTRIBUTIONS
# =============================================================================

def normalize_mode(value: object) -> str | None:
    text = str(value).strip().lower()
    if "year" in text:
        return "year_perm"
    if "circ" in text or "shift" in text:
        return "circ_shift"
    if "block" in text:
        return "block_perm"
    return None


def matches_window(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_match = numeric.eq(WINDOW_DAYS)

    text = series.astype(str).str.lower()
    text_match = (
        text.eq(f"sii_{WINDOW_DAYS}d")
        | text.str.contains(fr"\b{WINDOW_DAYS}\b", regex=True)
    )
    return numeric_match | text_match


def filter_analysis_identity(
    df: pd.DataFrame,
    candidates: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Restrict a result table to the pooled harmonic 28-day SIF-771 analysis.

    Exact overrides are applied first. Remaining common identity columns are
    filtered conservatively when they are present.
    """
    out = df.copy()

    for column, expected in NULL_FILTER_OVERRIDES.items():
        if column not in out.columns:
            raise ValueError(
                f"NULL_FILTER_OVERRIDES contains absent column {column!r}."
            )
        out = out[out[column].astype(str) == str(expected)].copy()

    method_col = resolve_column(
        out, "method", candidates, required=False
    )
    if method_col is not None:
        out = out[
            out[method_col].astype(str).str.lower().eq(METHOD)
        ].copy()

    window_col = resolve_column(
        out, "window", candidates, required=False
    )
    if window_col is not None:
        out = out[matches_window(out[window_col])].copy()

    sample_col = resolve_column(
        out, "sample", candidates, required=False
    )
    if sample_col is not None:
        sample_text = out[sample_col].astype(str).str.lower()
        pooled_mask = (
            sample_text.eq(SAMPLE)
            | sample_text.str.contains("pooled")
            | sample_text.str.contains("global")
        )
        if pooled_mask.any():
            out = out[pooled_mask].copy()

    outcome_col = resolve_column(
        out, "outcome", candidates, required=False
    )
    if outcome_col is not None:
        outcome_text = out[outcome_col].astype(str).str.lower()
        sif_mask = outcome_text.str.contains("771")
        if sif_mask.any():
            out = out[sif_mask].copy()

    return out


def load_null_distributions() -> dict[str, np.ndarray]:
    if not NULL_PARQUET.exists():
        raise FileNotFoundError(
            f"Surrogate null-distribution file not found:\n{NULL_PARQUET}"
        )

    raw = pd.read_parquet(NULL_PARQUET)
    raw = filter_analysis_identity(raw, NULL_COLUMN_CANDIDATES)

    mode_col = resolve_column(raw, "mode", NULL_COLUMN_CANDIDATES)
    rho_col = resolve_column(raw, "rho", NULL_COLUMN_CANDIDATES)

    raw["mode_normalized"] = raw[mode_col].map(normalize_mode)
    raw["null_rho"] = pd.to_numeric(raw[rho_col], errors="coerce")
    raw = raw.dropna(subset=["mode_normalized", "null_rho"])

    distributions: dict[str, np.ndarray] = {}
    for mode in MODE_ORDER:
        values = raw.loc[
            raw["mode_normalized"] == mode, "null_rho"
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError(
                f"No null values found for {mode!r} after filtering.\n"
                f"Available mode values: {sorted(raw[mode_col].astype(str).unique())}"
            )

        if len(values) not in {20, 100, 500, 1000}:
            warnings.warn(
                f"{mode}: found {len(values)} null values. "
                "Confirm that only one analysis stratum was selected."
            )

        distributions[mode] = values

    return distributions


def empirical_p_two_sided(observed_rho: float, null: np.ndarray) -> float:
    return float(
        (1 + np.sum(np.abs(null) >= abs(observed_rho)))
        / (len(null) + 1)
    )


def load_reported_p_values() -> dict[str, float]:
    """
    Load reported p-values when a compatible summary table exists.

    Returns an empty dictionary when the summary file is absent or cannot be
    reduced unambiguously to one row per surrogate mode.
    """
    if not SURROGATE_SUMMARY_CSV.exists():
        return {}

    try:
        raw = pd.read_csv(SURROGATE_SUMMARY_CSV)
        raw = filter_analysis_identity(raw, SUMMARY_COLUMN_CANDIDATES)

        mode_col = resolve_column(
            raw, "mode", SUMMARY_COLUMN_CANDIDATES
        )
        p_col = resolve_column(
            raw, "p", SUMMARY_COLUMN_CANDIDATES
        )

        raw["mode_normalized"] = raw[mode_col].map(normalize_mode)
        raw["reported_p"] = pd.to_numeric(raw[p_col], errors="coerce")
        raw = raw.dropna(subset=["mode_normalized", "reported_p"])

        result: dict[str, float] = {}
        for mode in MODE_ORDER:
            values = raw.loc[
                raw["mode_normalized"] == mode, "reported_p"
            ].dropna().unique()
            if len(values) == 1:
                result[mode] = float(values[0])

        return result
    except Exception as exc:
        warnings.warn(
            f"Could not use {SURROGATE_SUMMARY_CSV.name}: {exc}. "
            "P-values will be calculated from null distributions."
        )
        return {}


# =============================================================================
# PLOTTING
# =============================================================================

def density_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.nanstd(values) == 0:
        raise ValueError("A null distribution is too small for KDE.")

    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = max(upper - lower, 0.02)
    grid = np.linspace(lower - 0.12 * span, upper + 0.12 * span, 400)

    kde = gaussian_kde(values)
    density = kde(grid)
    return grid, density


def draw_main_panel(
    ax: plt.Axes,
    summaries: dict[int, pd.DataFrame],
    observed_rhos: dict[int, float],
) -> None:
    primary_summary = summaries[WINDOW_DAYS]
    x = primary_summary["bin_number"].to_numpy(dtype=float)

    for window in sorted(summaries.keys()):
        summary = summaries[window]
        style = WINDOW_LINE_STYLES[window]
        y = summary["mean_sif"].to_numpy(dtype=float)
        lo = summary["boot_ci_lo"].to_numpy(dtype=float)
        hi = summary["boot_ci_hi"].to_numpy(dtype=float)

        ax.fill_between(
            x,
            lo,
            hi,
            color=style["fill"],
            alpha=0.13,
            linewidth=0,
            zorder=style["zorder"] - 1,
        )
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=style["linewidth"],
            marker="o",
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=f"{window} d  ({observed_rhos[window]:+.3f})",
        )

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.72, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(primary_summary["bin_label"], rotation=0)
    ax.set_xlabel("Trailing-mean SII quantile")
    ax.set_ylabel(
        "Mean standardized SIF 771-nm anomaly"
        if STANDARDIZE_RESIDUAL
        else "Mean SIF 771-nm residual"
    )

    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower left",
        fontsize=9,
        frameon=True,
        framealpha=0.93,
        borderpad=0.5,
        handlelength=1.8,
    )


def draw_surrogate_inset(
    parent_ax: plt.Axes,
    distributions: dict[str, np.ndarray],
    observed_rho: float,
    p_values: dict[str, float],
) -> None:
    inset = parent_ax.inset_axes([0.565, 0.515, 0.405, 0.425])

    all_values = [observed_rho]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for mode in MODE_ORDER:
        grid, density = density_curve(distributions[mode])
        curves[mode] = (grid, density)
        all_values.extend(grid.tolist())

    x_min = min(all_values)
    x_max = max(all_values)
    span = max(x_max - x_min, 0.05)

    for mode in MODE_ORDER:
        style = NULL_STYLES[mode]
        grid, density = curves[mode]
        label = f"{style['label']}  p={p_values[mode]:.3f}"

        inset.fill_between(
            grid,
            0,
            density,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
        inset.plot(
            grid,
            density,
            color=style["color"],
            linewidth=1.5,
            alpha=0.90,
            label=label,
        )

    inset.axvline(
        observed_rho,
        color=OBSERVED_COLOR,
        linewidth=2.1,
        label=rf"Observed $\rho$={observed_rho:.3f}",
        zorder=6,
    )
    inset.axvline(0, color="black", linewidth=0.7, alpha=0.50)

    inset.set_xlim(x_min - 0.03 * span, x_max + 0.03 * span)
    inset.set_yticks([])
    inset.set_xlabel(r"Spearman $\rho$", fontsize=8.5)
    inset.set_title("Temporal surrogate distributions", fontsize=9.5, pad=5)
    inset.tick_params(axis="x", labelsize=8)

    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_visible(False)

    inset.legend(
        loc="upper right",
        fontsize=7.2,
        frameon=True,
        framealpha=0.93,
        borderpad=0.45,
        handlelength=1.6,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, resolved_columns = load_primary_data()
    summaries, observed_rhos = compute_window_summaries(data)

    observed_rho = observed_rhos[WINDOW_DAYS]
    primary_summary = summaries[WINDOW_DAYS]
    distributions = load_null_distributions()

    p_from_nulls = {
        mode: empirical_p_two_sided(observed_rho, distributions[mode])
        for mode in MODE_ORDER
    }
    p_reported = load_reported_p_values()

    p_values = {
        mode: p_reported.get(mode, p_from_nulls[mode])
        for mode in MODE_ORDER
    }

    for mode in MODE_ORDER:
        if mode in p_reported and not np.isclose(
            p_reported[mode],
            p_from_nulls[mode],
            atol=1 / (len(distributions[mode]) + 1) + 1e-12,
        ):
            warnings.warn(
                f"{mode}: reported p={p_reported[mode]:.6f}, "
                f"recalculated p={p_from_nulls[mode]:.6f}. "
                "The figure uses the reported summary value."
            )

    combined_summary = pd.concat(
        summaries.values(), ignore_index=True
    )
    combined_summary["method"] = METHOD
    combined_summary["outcome"] = OUTCOME
    combined_summary["residual_column"] = resolved_columns["residual"]
    combined_summary["exposure_column"] = resolved_columns["exposure"]
    combined_summary["cell_id_column"] = resolved_columns["cell_id"]
    combined_summary["merge_keys"] = resolved_columns["merge_keys"]
    combined_summary["exposure_source"] = resolved_columns["exposure_source"]
    combined_summary.to_csv(OUTPUT_SOURCE_CSV, index=False)

    null_summary = pd.DataFrame(
        {
            "surrogate_mode": MODE_ORDER,
            "n_surrogates": [
                len(distributions[mode]) for mode in MODE_ORDER
            ],
            "observed_rho": observed_rho,
            "empirical_p_used": [p_values[mode] for mode in MODE_ORDER],
            "empirical_p_recalculated": [
                p_from_nulls[mode] for mode in MODE_ORDER
            ],
        }
    )
    null_summary.to_csv(OUTPUT_NULL_CSV, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_main_panel(ax, summaries, observed_rhos)
    draw_surrogate_inset(ax, distributions, observed_rho, p_values)

    # No overall title: the manuscript caption supplies the analytical context.
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    export_figure_text(fig, OUTPUT_SOURCE_CSV, __doc__, OUTPUT_PNG)
    plt.close(fig)

    print(f"Primary rows: {len(data):,}")
    print(f"Primary cells: {data['cell_id'].nunique():,}")
    for window in sorted(summaries.keys()):
        print(
            f"Observed rho ({window} d): {observed_rhos[window]:.6f}"
        )
    for mode in MODE_ORDER:
        print(
            f"{mode}: n={len(distributions[mode]):,}, "
            f"p={p_values[mode]:.6f}"
        )
    print(f"Source table: {OUTPUT_SOURCE_CSV}")
    print(f"Null summary: {OUTPUT_NULL_CSV}")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())