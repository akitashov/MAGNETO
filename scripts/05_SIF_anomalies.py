#!/usr/bin/env python3
"""
OCO-2 SIF Seasonal Trend Analysis and Anomaly Detection.

Logic:
1.  Loads aggregated SIF data (including physiological indices).
2.  Fits a linear model with seasonal components for each grid cell and variable.
    Model: y = alpha0 + alpha_t * t + beta_cos * cos(2π * t_frac) + beta_sin * sin(2π * t_frac)
3.  Calculates Residuals: Observed value minus the predicted seasonal/trend value.
4.  Standardizes Residuals (Z-score) for inter-variable comparison.
5.  Saves intermediate results (checkpoints) to disk to optimize RAM usage.
6.  Merges checkpoints into final wavelength/index-specific files.

CONFIGURATION:
--------------
Paths and Constants are imported from `_Common.py`.

OUTPUT FILE DESCRIPTION:
------------------------
A. model_parameters_{variable}.feather:
   - Contains model coefficients and R-squared stats for each grid cell.
   - Columns: variable, latitude, longitude, lat_id, lon_id, alpha0, alpha_day, 
     beta_cos, beta_sin, sigma, r_squared, n_obs, amplitude, phase.

B. sif_residuals_{variable}.parquet:
   - Full time series of Observed, Predicted, and Residual values.
   - Columns: variable, date, latitude, longitude, lat_id, lon_id, 
     residual, std_residual.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
import gc
import shutil
from _Common import Config

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SIF_COLUMNS = ['sif_740nm', 'sif_757nm', 'sif_771nm', 'sif_stress_index']

# Pull from _Common.Config (single source of truth)
MIN_OBSERVATIONS = int(getattr(Config, "SIF_MODEL_MIN_OBSERVATIONS", 40))
BATCH_SIZE = int(getattr(Config, "SIF_MODEL_BATCH_SIZE", 120))

CHECKPOINT_DIR = Config.DIR_SIF_MODEL / "checkpoints"

# Workers: allow override from Config, otherwise use (cpu_count - 1)
_cfg_workers = getattr(Config, "SIF_MODEL_N_WORKERS", None)
N_WORKERS = int(_cfg_workers) if _cfg_workers else max(1, mp.cpu_count() - 1)


# ==============================================================================
# LOGIC
# ==============================================================================

class SIFSeasonalAnomalyProcessor:
    def _clean_checkpoints_start(self):
        if CHECKPOINT_DIR.exists(): shutil.rmtree(CHECKPOINT_DIR)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def load_source_data(self):
        print(f"\n[INFO] Loading: {Config.FILE_SIF_FINAL}")
        df = pd.read_feather(Config.FILE_SIF_FINAL)
        df['date'] = pd.to_datetime(df['date'])
        df["day_of_year"] = df["date"].dt.dayofyear

        # days in each year (365 / 366)
        df["days_in_year"] = df["date"].dt.is_leap_year.map({True: 366, False: 365})

        # fractional year
        df["t_frac"] = (
            df["date"].dt.year
            + (df["day_of_year"] - 1) / df["days_in_year"]
        ).astype("float64")

        df["time_numeric"] = (df["date"] - df["date"].min()).dt.days
        
        # Ensure floats
        float_cols = ['latitude', 'longitude', 'day_of_year', 'time_numeric']
        for c in float_cols: df[c] = df[c].astype('float32')
        return df
    
    @staticmethod
    def fit_linear_model(y, t_frac, time_normalized):
        """
        Fit seasonal-trend linear model to time series.

        Model:
            y = alpha0 + alpha_t*t + beta_cos*cos(2πt) + beta_sin*sin(2πt)

        Parameters:
            y (ndarray): Observed values.
            t_frac (ndarray): Fractional year.
            time_normalized (ndarray | None): Normalized time index.

        Returns:
            tuple:
                (dict, ndarray)

                dict contains:
                    alpha0, alpha_t, beta_cos, beta_sin,
                    amplitude, phase, sigma, r_squared, n_obs

                ndarray is predicted signal.
        """
        n = len(y)

        # seasonal phase
        t_frac = t_frac.astype(np.float64)  
        phase = 2.0 * np.pi * t_frac


        use_trend = bool(getattr(Config, "SIF_MODEL_USE_TREND", True)) and (time_normalized is not None)

        # Design matrix
        # [1, t, cos, sin] or [1, cos, sin]
        if use_trend:
            t = time_normalized.astype(np.float64)
            X = np.column_stack([
                np.ones(n, dtype=np.float64),
                t,
                np.cos(phase),
                np.sin(phase),
            ])
        else:
            X = np.column_stack([
                np.ones(n, dtype=np.float64),
                np.cos(phase),
                np.sin(phase),
            ])

        try:
            # Prefer lstsq for stability (singular/ill-conditioned X happens)
            if bool(getattr(Config, "SIF_MODEL_USE_LSTSQ", True)):
                beta, *_ = np.linalg.lstsq(X, y.astype(np.float64), rcond=None)
            else:
                beta = np.linalg.solve(X.T @ X, X.T @ y.astype(np.float64))

            y_pred = X @ beta
            residuals = y.astype(np.float64) - y_pred

            sst = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - (np.sum(residuals ** 2) / sst) if sst > 0 else 0.0

            # unpack coefficients
            if use_trend:
                alpha0, alpha_t, beta_cos, beta_sin = beta
            else:
                alpha0, beta_cos, beta_sin = beta
                alpha_t = np.nan

            amplitude = float(np.sqrt(beta_cos ** 2 + beta_sin ** 2))
            season_phase = float(np.arctan2(beta_sin, beta_cos))  # radians

            return {
                "alpha0": float(alpha0),
                "alpha_t": float(alpha_t) if np.isfinite(alpha_t) else np.nan,
                "beta_cos": float(beta_cos),
                "beta_sin": float(beta_sin),
                "amplitude": amplitude,
                "phase": season_phase,
                "sigma": float(np.std(residuals)),
                "r_squared": float(r_squared),
                "n_obs": int(n),
            }, y_pred.astype(np.float32)

        except Exception:
            return {}, np.zeros_like(y, dtype=np.float32)

    @staticmethod
    def process_cell_worker(cell_data):
        """
        Fit seasonal model and compute residuals for one grid cell.

        Parameters:
            cell_data (tuple):
                (lat_id, lon_id, lat, lon, data_dict, variable_name)

        Returns:
            dict | None:
                Model parameters and residuals, or None if insufficient data.
        """
        lat_id, lon_id, lat, lon, cell_df_dict, var_name = cell_data
        try:
            cell_df = pd.DataFrame(cell_df_dict)
            cell_df["date"] = pd.to_datetime(cell_df["date"])
            cell_df = cell_df.sort_values("date")

            # --- (A) calendar reindex: make continuous daily timeline ---
            full_dates = pd.date_range(cell_df["date"].min(), cell_df["date"].max(), freq="D")
            cell_df = (
                cell_df.set_index("date")
                    .reindex(full_dates)
                    .rename_axis("date")
                    .reset_index()
            )

            # recompute time features for the full calendar
            cell_df["day_of_year"] = cell_df["date"].dt.dayofyear
            cell_df["days_in_year"] = cell_df["date"].dt.is_leap_year.map({True: 366, False: 365})
            cell_df["t_frac"] = (
                cell_df["date"].dt.year + (cell_df["day_of_year"] - 1) / cell_df["days_in_year"]
            ).astype("float64")

            # local monotonic time index is enough for trend (normalized later)
            cell_df["time_numeric"] = (cell_df["date"] - cell_df["date"].min()).dt.days.astype("float32")

            # --- (B) fit only on observed points (non-NaN y) ---
            obs = cell_df[var_name].notna().values
            cell_obs = cell_df.loc[obs].copy()
            if len(cell_obs) < MIN_OBSERVATIONS:
                return None

            y = cell_obs[var_name].values.astype("float32")
            time = cell_obs["time_numeric"].values.astype("float32")
            den = (time.max() - time.min())
            t_norm = (time - time.min()) / den if den > 0 else np.zeros_like(time, dtype="float32")

            params, y_pred = SIFSeasonalAnomalyProcessor.fit_linear_model(
                y, cell_obs["t_frac"].values.astype("float64"), t_norm
            )
            if not params:
                return None

            residuals_obs = y - y_pred
            res_std = float(np.nanstd(residuals_obs))

            # full-calendar residuals: NaN for missing days
            res_full = np.full(len(cell_df), np.nan, dtype="float32")
            res_full[obs] = residuals_obs.astype("float32")
            std_full = res_full / (res_std if res_std > 0 else 1.0)

            residuals_data = {
                "variable": var_name,
                "date": cell_df["date"].values,  # full calendar
                "latitude": np.full(len(cell_df), float(lat), dtype="float32"),
                "longitude": np.full(len(cell_df), float(lon), dtype="float32"),
                "lat_id": np.full(len(cell_df), int(lat_id), dtype="int16"),
                "lon_id": np.full(len(cell_df), int(lon_id), dtype="int16"),
                "residual": res_full,
                "std_residual": std_full.astype("float32"),
            }

            # collecting full_params
            full_params = {
                "variable": var_name,
                "latitude": float(lat),
                "longitude": float(lon),
                "lat_id": int(lat_id),
                "lon_id": int(lon_id),
                **params,  # alpha0, alpha_t, beta_cos, beta_sin, amplitude, phase, sigma, r_squared, n_obs
            }

            return {"params": full_params, "residuals_data": residuals_data}
        except Exception as e:
            print(f"[ERROR] process_cell_worker failed for {lat_id=}, {lon_id=}, {var_name=}: {e}")
            raise

    def run(self):
        self._clean_checkpoints_start()
        df = self.load_source_data()
        
        all_tasks = []
        for (lat_id, lon_id), cell_df in tqdm(df.groupby(['lat_id', 'lon_id']), desc="Grouping"):
            lat, lon = cell_df['latitude'].iloc[0], cell_df['longitude'].iloc[0]
            for var in SIF_COLUMNS:
                if cell_df[var].notna().sum() >= MIN_OBSERVATIONS:
                    cols = ["date", var]
                    all_tasks.append((int(lat_id), int(lon_id), float(lat), float(lon), cell_df[cols].to_dict('list'), var))
        del df; gc.collect()

        batches = [all_tasks[i:i + BATCH_SIZE] for i in range(0, len(all_tasks), BATCH_SIZE)]
        with tqdm(total=len(all_tasks)) as pbar:
            for idx, batch in enumerate(batches):
                batch_res = []
                with ProcessPoolExecutor(max_workers=N_WORKERS) as exc:
                    futures = {exc.submit(self.process_cell_worker, t): t for t in batch}
                    for f in as_completed(futures):
                        res = f.result()
                        if res: batch_res.append(res)
                        pbar.update(1)
                
                # Save Batch
                by_var = {v: {'p': [], 'r': []} for v in SIF_COLUMNS}
                for res in batch_res:
                    v = res['params']['variable']
                    by_var[v]['p'].append(res['params'])
                    by_var[v]['r'].append(pd.DataFrame(res['residuals_data']))
                
                for v in SIF_COLUMNS:
                    if by_var[v]['p']:
                        pd.DataFrame(by_var[v]['p']).to_feather(CHECKPOINT_DIR / f"b{idx}_{v}_p.feather")
                        pd.concat(by_var[v]['r']).to_parquet(CHECKPOINT_DIR / f"b{idx}_{v}_r.parquet")
                del batch_res; gc.collect()

        # Merge
        Config.DIR_SIF_MODEL.mkdir(parents=True, exist_ok=True)
        for var in SIF_COLUMNS:
            p_files = sorted(CHECKPOINT_DIR.glob(f"*_{var}_p.feather"))
            if p_files: pd.concat((pd.read_feather(f) for f in p_files)).to_feather(Config.DIR_SIF_MODEL / f"model_parameters_{var}.feather")
            
            r_files = sorted(CHECKPOINT_DIR.glob(f"*_{var}_r.parquet"))
            if r_files: pd.concat((pd.read_parquet(f) for f in r_files)).to_parquet(Config.DIR_SIF_MODEL / f"sif_residuals_{var}.parquet")

        shutil.rmtree(CHECKPOINT_DIR)
        print("[SUCCESS] Done.")

if __name__ == "__main__":
    SIFSeasonalAnomalyProcessor().run()