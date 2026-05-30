"""
WastewaterTFT — NeuralForecast TFT wrapper for probabilistic wastewater forecasting.

Covariate mapping (processor output → NeuralForecast roles)
------------------------------------------------------------
unique_id  ← county_fips
ds         ← sample_collect_date
y          ← log1p_new_cases  (RobustScaled prediction target; WW is hist_exog)

Static (one value per series, time-invariant):
  log_population       ← log1p(population_served)        [derived]
  county_fips_encoded  ← integer-encoded FIPS             [derived]
  is_sludge            ← 1=copies/g dry sludge, 0=liquid  [processor Stage 3]

Historical / past-only (known up to forecast origin, not in future):
  log1p_concentration        ← WW signal at t              [processor Stage 11]
  log1p_concentration_lag1w  ← WW at t-1                   [processor Stage 14]
  log1p_concentration_lag2w  ← WW at t-2
  log1p_concentration_lag3w  ← WW at t-3
  log1p_new_cases_lag1w      ← cases at t-1 (VSN momentum) [processor Stage 14]
  log1p_new_cases_lag2w      ← cases at t-2
  log1p_new_cases_lag3w      ← cases at t-3
  growth_rate_1w             ← WW week-over-week growth rate
  relative_decay_rate        ← 7-day % change on smoothed signal [processor Stage 9]
  outlier_flag_int           ← bool→int                    [processor Stage 14]

Future-known (calendar features, computable for any future date):
  sin_annual_1, cos_annual_1
  sin_annual_2, cos_annual_2
  sin_annual_3, cos_annual_3
  day_of_week_sin, day_of_week_cos
  month_sin, month_cos
  week_of_year

Calibration design
------------------
scaler_type="identity": the processor's RobustScaler already standardises all
features.  A second NeuralForecast robust pass double-compresses variance and
collapses quantile spread.  Identity passes data to the TFT unchanged.

Non-negativity: NOT enforced inside the model.  Softplus was removed from
PINNWastewaterLoss.domain_map because the RobustScaled target is legitimately
negative for below-median weeks; Softplus was collapsing all quantile outputs
toward zero (→ 0% PI coverage).  Non-negativity is enforced post-hoc in
_build_decoded_forecast via np.expm1(...).clip(lower=0).

Cross-county attention
----------------------
NeuralForecast TFT is a **global model**: all county series are packed into the
same batch and trained with shared weights.  Each county's series is distinguished
by its static covariates (county_fips_encoded, is_sludge, log_population), which
are routed through the Variable Selection Network before the temporal attention
blocks.  This allows the model to learn county-specific signal shapes while
sharing the temporal attention parameters — a form of implicit cross-county
information sharing.  Explicit spatial graph attention (one county informing
another's attention directly) would require a graph-based architecture and is
a future extension.

Usage
-----
    proc   = WastewaterProcessor()
    train  = proc.run(raw_train)
    val    = proc.transform(raw_val)

    model  = WastewaterTFT()
    model.fit(train, val_df=val)
    preds  = model.predict(horizon_df)        # quantile forecast DataFrame
    vi     = model.variable_importance()      # dict of attention weights
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from pytorch_lightning.callbacks import Callback

from src.config import (
    BAY_AREA_FIPS,
    BAY_AREA_POPULATION,
    COUNTY_COL,
    LOGS_DIR,
    MODELS_DIR,
    NWSS_DATE_COL,
    QUANTILE_LEVELS,
    TARGET_COL,
    TFT_CONFIG,
)
from src.models.loss_functions import PINNWastewaterLoss

# ---------------------------------------------------------------------------
# Column role declarations
# ---------------------------------------------------------------------------

#: Calendar columns that can be recomputed for any future date
FUTURE_COVARIATES: list[str] = [
    "sin_annual_1", "cos_annual_1",
    "sin_annual_2", "cos_annual_2",
    "sin_annual_3", "cos_annual_3",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "week_of_year",
]

#: Columns only available up to the forecast origin (must *not* be in futr_exog_list)
HIST_COVARIATES: list[str] = [
    # Wastewater signal — primary leading indicator
    "log1p_concentration",        # WW at t
    "log1p_concentration_lag1w",  # WW at t-1
    "log1p_concentration_lag2w",  # WW at t-2
    "log1p_concentration_lag3w",  # WW at t-3
    # Case momentum — explicit lags surface WW→cases timing in VSN weights
    "log1p_new_cases_lag1w",      # cases at t-1
    "log1p_new_cases_lag2w",      # cases at t-2
    "log1p_new_cases_lag3w",      # cases at t-3
    # Slope / outbreak-phase features
    "growth_rate_1w",             # WW relative week-over-week change
    "relative_decay_rate",        # WW 7-day relative change; captures surge & decay
    "outlier_flag_int",
    # Derivative expansion — velocity, acceleration, and rolling momentum (Phase 2/3)
    "vel_concentration",               # 1st derivative: absolute weekly Δ (velocity)
    "accel_concentration",            # 2nd derivative: Δ velocity (acceleration / inflection)
    "vel_concentration_lag1w",        # velocity 1 week ago (momentum direction context)
    "log1p_concentration_2w_ma",      # 2-week rolling mean (short baseline)
    "log1p_concentration_4w_ma",      # 4-week rolling mean (medium baseline)
    "log1p_concentration_2w_std",     # 2-week rolling std (local volatility signal)
    "log1p_concentration_4w_std",     # 4-week rolling std (medium volatility signal)
    # Phase-shift gravity — velocity divergence, no level anchor (Phase 5)
    "ww_momentum_lead",               # vel_concentration[t] − (cases[t-1] − cases[t-2])
]

#: One row per unique_id — derived at fit time
STATIC_COVARIATES: list[str] = [
    "log_population",
    "county_fips_encoded",
    "is_sludge",             # 1=copies/g dry sludge (all 9 Bay Area counties), 0=liquid track
]

# FIPS → integer encoding (stable across runs)
_FIPS_INT: dict[str, int] = {
    fips: idx for idx, fips in enumerate(sorted(BAY_AREA_FIPS.values()))
}


# ---------------------------------------------------------------------------
# Helper: build calendar feature columns for an arbitrary date range
# ---------------------------------------------------------------------------

def build_future_df(
    unique_ids: list[str],
    last_date: pd.Timestamp,
    h: int,
    freq: str = "W-WED",
) -> pd.DataFrame:
    """Build a ``futr_df`` with calendar features for the forecast horizon.

    Parameters
    ----------
    unique_ids : County FIPS codes present in the training data.
    last_date  : Last observed date in the training set.
    h          : Forecast horizon (number of steps).
    freq       : Resampling frequency used by the processor (default "W").
    """
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=h,
        freq=freq,
    )
    rows = []
    for uid in unique_ids:
        for ds in future_dates:
            doy = ds.dayofyear
            dow = ds.dayofweek
            rows.append({
                "unique_id": uid,
                "ds": ds,
                "sin_annual_1": np.sin(2 * np.pi * 1 * doy / 365.25),
                "cos_annual_1": np.cos(2 * np.pi * 1 * doy / 365.25),
                "sin_annual_2": np.sin(2 * np.pi * 2 * doy / 365.25),
                "cos_annual_2": np.cos(2 * np.pi * 2 * doy / 365.25),
                "sin_annual_3": np.sin(2 * np.pi * 3 * doy / 365.25),
                "cos_annual_3": np.cos(2 * np.pi * 3 * doy / 365.25),
                "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
                "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
                "month_sin": np.sin(2 * np.pi * ds.month / 12),
                "month_cos": np.cos(2 * np.pi * ds.month / 12),
                "week_of_year": int(ds.isocalendar()[1]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training progress callback
# ---------------------------------------------------------------------------

class StepProgressCallback(Callback):
    """Logs step-based training progress every N steps.

    NeuralForecast trains with ``max_steps`` (gradient steps), but PyTorch
    Lightning's default progress bar shows ``Epoch N/-2`` with no percentage
    because it doesn't know the total when ``max_steps`` is set instead of
    ``max_epochs``.  This callback fires on every training batch and logs a
    clean progress line every ``log_every_n_steps`` gradient steps:

        Step  100/2000 ( 5.0%)  train_loss=0.2341  valid_loss=0.6830  ETA 28.5min

    Automatically disabled for CV / outbreak-validation models (those set
    ``enable_progress_bar=False`` in ``trainer_kwargs``, which prevents this
    callback from being added by ``WastewaterTFT``).
    """

    def __init__(self, max_steps: int, log_every_n_steps: int = 100) -> None:
        self.max_steps         = max_steps
        self.log_every_n_steps = log_every_n_steps
        self._t0: float | None = None
        self._last_valid: float = float("nan")

    def on_train_start(self, trainer, pl_module) -> None:
        self._t0 = time.time()
        logger.info("Training started — {} steps total.", self.max_steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        m          = trainer.callback_metrics
        train_loss = float(m.get("train_loss_step", m.get("train_loss", float("nan"))))
        valid_loss = float(m.get("valid_loss", self._last_valid))
        if not math.isnan(valid_loss):
            self._last_valid = valid_loss

        pct     = min(step / self.max_steps * 100, 100.0)
        elapsed = time.time() - (self._t0 or time.time())
        eta_m   = elapsed / step * (self.max_steps - step) / 60 if step > 0 else 0.0

        tl = f"{train_loss:.4f}" if not math.isnan(train_loss) else "—"
        vl = f"{valid_loss:.4f}" if not math.isnan(valid_loss) else "—"
        logger.info(
            "Step {:4d}/{} ({:5.1f}%)  train_loss={}  valid_loss={}  ETA {:.1f}min",
            step, self.max_steps, pct, tl, vl, eta_m,
        )

    def on_train_end(self, trainer, pl_module) -> None:
        if self._t0:
            logger.info(
                "Training complete — {} steps in {:.1f}min.",
                trainer.global_step,
                (time.time() - self._t0) / 60,
            )


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------

class WastewaterTFT:
    """Thin wrapper around ``neuralforecast.models.TFT`` with:

    - Automatic processor-column → NeuralForecast column renaming
    - Derived static covariates (log_population, encoded FIPS)
    - PINNWastewaterLoss (Softplus + pinball + growth-rate penalty)
    - ``fit`` / ``predict`` / ``variable_importance`` interface

    Parameters
    ----------
    h           : Forecast horizon (number of weekly steps ahead).
    input_size  : Encoder look-back window (number of weekly steps).
    max_steps   : Training gradient steps (overrides TFT_CONFIG default).
    random_seed : For reproducibility.
    trainer_kwargs : Extra kwargs forwarded to PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        h: int = TFT_CONFIG["h"],
        input_size: int = TFT_CONFIG["input_size"],
        max_steps: int = TFT_CONFIG["max_steps"],
        random_seed: int = 42,
        trainer_kwargs: Optional[dict] = None,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self._nf: Optional[NeuralForecast] = None
        self._unique_ids: list[str] = []
        self._last_train_date: Optional[pd.Timestamp] = None

        _hw = TFT_CONFIG.get("horizon_weight")
        if _hw is not None:
            _hw = np.array(_hw, dtype=np.float32)
        self._loss = PINNWastewaterLoss(quantiles=QUANTILE_LEVELS, horizon_weight=_hw)

        _trainer_kwargs = {"enable_progress_bar": True, "log_every_n_steps": 10}
        if trainer_kwargs:
            _trainer_kwargs.update(trainer_kwargs)

        # early_stop_patience_steps may be overridden via trainer_kwargs
        # (e.g. CV folds pass -1 to disable early stopping so val_size=0 is valid).
        # Pop it before **_trainer_kwargs to avoid the "multiple values" TypeError.
        _early_stop = _trainer_kwargs.pop(
            "early_stop_patience_steps",
            TFT_CONFIG["early_stop_patience_steps"],
        )

        # Verify hist/futr lists don't overlap (NeuralForecast raises if they do)
        assert not set(HIST_COVARIATES) & set(FUTURE_COVARIATES), (
            "A covariate appears in both hist and futr lists."
        )

        self._tft = TFT(
            h=h,
            input_size=input_size,
            # Architecture
            hidden_size=TFT_CONFIG["hidden_size"],
            n_head=TFT_CONFIG["n_head"],
            attn_dropout=TFT_CONFIG["attn_dropout"],
            dropout=TFT_CONFIG["dropout"],
            n_rnn_layers=TFT_CONFIG.get("encoder_layers", 2),
            # Covariates
            stat_exog_list=STATIC_COVARIATES,
            hist_exog_list=HIST_COVARIATES,
            futr_exog_list=FUTURE_COVARIATES,
            # Loss (Softplus + pinball + PINN penalty)
            loss=self._loss,
            # Training
            max_steps=max_steps,
            learning_rate=TFT_CONFIG["learning_rate"],
            batch_size=TFT_CONFIG["batch_size"],
            valid_batch_size=TFT_CONFIG["valid_batch_size"],
            scaler_type=TFT_CONFIG["scaler_type"],
            early_stop_patience_steps=_early_stop,
            random_seed=random_seed,
            # Pad series shorter than input_size (e.g. counties that joined NWSS late)
            # rather than raising a ValueError.
            start_padding_enabled=TFT_CONFIG.get("start_padding_enabled", True),
            # NeuralForecast 3.x: trainer kwargs are passed directly as **kwargs,
            # not nested under a 'trainer_kwargs' key.
            **_trainer_kwargs,
        )

        logger.info(
            "WastewaterTFT initialised — h={}, input_size={}, "
            "quantiles={}, max_steps={}",
            h, input_size, QUANTILE_LEVELS, max_steps,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        val_size: int = 0,
    ) -> "WastewaterTFT":
        """Fit the TFT on processor output.

        Parameters
        ----------
        train_df : Output of ``WastewaterProcessor.run()``.
        val_df   : Optional pre-split validation set; if provided, used for
                   early stopping instead of the rolling val_size window.
        val_size : Number of trailing weeks to use as validation when val_df
                   is not provided.
        """
        nf_train, static_df = self._to_nf_format(train_df)
        nf_val = self._to_nf_format(val_df)[0] if val_df is not None else None

        self._unique_ids = nf_train["unique_id"].unique().tolist()
        self._last_train_date = nf_train["ds"].max()

        self._nf = NeuralForecast(models=[self._tft], freq="W-WED")
        self._nf.fit(
            df=nf_train,
            static_df=static_df,
            val_size=val_size,
            val_df=nf_val,
        )
        logger.info(
            "Training complete — {} series, last date {}.",
            len(self._unique_ids), self._last_train_date.date(),
        )
        return self

    def predict(
        self,
        futr_df: Optional[pd.DataFrame] = None,
        horizon_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate quantile forecasts for the next ``h`` weeks.

        Parameters
        ----------
        futr_df     : Pre-built future DataFrame from ``build_future_df()``.
                      If None, built automatically from the training horizon.
        horizon_df  : Alias for futr_df (accepted for API symmetry).
        """
        if self._nf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        futr = futr_df or horizon_df
        if futr is None:
            futr = build_future_df(
                unique_ids=self._unique_ids,
                last_date=self._last_train_date,
                h=self.h,
            )

        preds = self._nf.predict(futr_df=futr)
        logger.info("Forecast produced for {} series × {} steps.", len(self._unique_ids), self.h)
        return preds

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted NeuralForecast checkpoint."""
        if self._nf is None:
            raise RuntimeError("Model not fitted.")
        save_path = path or MODELS_DIR / "wastewater_tft"
        self._nf.save(str(save_path), overwrite=True)
        logger.info("Model saved to {}.", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "WastewaterTFT":
        """Load a previously saved model."""
        load_path = path or MODELS_DIR / "wastewater_tft"
        instance = cls.__new__(cls)
        instance._nf = NeuralForecast.load(str(load_path))
        instance._tft = instance._nf.models[0]
        instance._loss = instance._tft.loss
        instance._unique_ids = []
        instance._last_train_date = None
        logger.info("Model loaded from {}.", load_path)
        return instance

    def variable_importance(self) -> dict[str, pd.DataFrame]:
        """Extract learned variable importance weights from TFT attention.

        Returns a dict with keys ``"static"``, ``"historical"``, ``"future"``
        mapping to DataFrames of (variable, importance_score).

        Note: NeuralForecast does not expose a built-in VI API; this extracts
        the encoder variable-selection GRN weights from the underlying module.
        """
        if self._nf is None:
            raise RuntimeError("Model not fitted.")

        model = self._tft
        result: dict[str, pd.DataFrame] = {}

        def _grn_weights(grn_module, names: list[str]) -> pd.DataFrame:
            """Average the absolute weights of the first linear layer in a GRN."""
            try:
                # GRN has an attribute `fc1` or similar — inspect available params
                for name, param in grn_module.named_parameters():
                    if "weight" in name and param.ndim == 2:
                        scores = param.detach().abs().mean(dim=0).cpu().numpy()
                        # Trim or pad to match number of variable names
                        scores = scores[: len(names)]
                        return pd.DataFrame(
                            {"variable": names[: len(scores)], "importance": scores}
                        ).sort_values("importance", ascending=False)
            except Exception:
                pass
            return pd.DataFrame({"variable": names, "importance": [float("nan")] * len(names)})

        try:
            result["static"] = _grn_weights(
                model.stat_exog_encoder, STATIC_COVARIATES
            )
        except AttributeError:
            logger.debug("stat_exog_encoder not found — static VI unavailable.")

        try:
            result["historical"] = _grn_weights(
                model.hist_exog_encoder, HIST_COVARIATES
            )
        except AttributeError:
            logger.debug("hist_exog_encoder not found — historical VI unavailable.")

        try:
            result["future"] = _grn_weights(
                model.futr_exog_encoder, FUTURE_COVARIATES
            )
        except AttributeError:
            logger.debug("futr_exog_encoder not found — future VI unavailable.")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_nf_format(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert processor output to the ``(timeseries_df, static_df)`` pair
        expected by NeuralForecast.

        Processor column → NeuralForecast column
        -----------------------------------------
        county_fips          → unique_id
        sample_collect_date  → ds
        log1p_concentration  → y
        outlier_flag (bool)  → outlier_flag_int (int)
        population_served    → used to derive log_population (static)
        """
        df = df.copy()

        # Rename to NeuralForecast standard names
        df = df.rename(columns={
            COUNTY_COL: "unique_id",
            NWSS_DATE_COL: "ds",
            TARGET_COL: "y",
        })

        # Derived historical covariate: bool flag → int
        if "outlier_flag" in df.columns:
            df["outlier_flag_int"] = df["outlier_flag"].astype(int)

        # Drop any columns not used by the model to keep the DataFrame clean
        keep = (
            ["unique_id", "ds", "y"]
            + [c for c in HIST_COVARIATES if c in df.columns]
            + [c for c in FUTURE_COVARIATES if c in df.columns]
        )
        ts_df = df[[c for c in keep if c in df.columns]].copy()

        # Drop warm-up rows where lag features are NaN (first 1–3 rows per county).
        # NeuralForecast raises ValueError on any NaN in hist_exog columns.
        hist_present = [c for c in HIST_COVARIATES if c in ts_df.columns]
        if hist_present:
            n_before = len(ts_df)
            ts_df = ts_df.dropna(subset=hist_present)
            dropped = n_before - len(ts_df)
            if dropped:
                logger.debug(
                    "Dropped {} warm-up rows with NaN lag/decay features.", dropped
                )

        # Build static_df (one row per unique_id)
        static_df = self._build_static_df(df)

        return ts_df, static_df

    def _build_static_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive static covariates from the time-series DataFrame.

        log_population      = log1p(median population_served per county)
        county_fips_encoded = integer from the stable FIPS→int mapping
        """
        if "population_served" in df.columns:
            pop_per_county = (
                df.groupby("unique_id")["population_served"]
                .median()
                .rename("population_served")
            )
        else:
            logger.debug(
                "population_served not in DataFrame — using 2020 Census county populations."
            )
            fips_list = df["unique_id"].unique()
            pop_per_county = pd.Series(
                {fips: float(BAY_AREA_POPULATION.get(str(fips), 1.0)) for fips in fips_list},
                name="population_served",
            )

        static = pop_per_county.reset_index()
        static.columns = ["unique_id", "population_served"]
        static["log_population"] = np.log1p(static["population_served"])
        # Z-score log_population so it sits in the same ~[-2, +2] range as
        # RobustScaled time-varying features.  Without this, log_population
        # (~13–16 for Bay Area counties) is 6–50× larger than every other
        # input, distorting VSN gradient signal toward the static channel.
        # Self-contained: no scaler state needed; std is ~0 for a single county
        # (1-county runs) → safely clamped to 0.
        _pop_std = float(static["log_population"].std(ddof=0))
        if _pop_std > 1e-6:
            _pop_mean = float(static["log_population"].mean())
            static["log_population"] = (static["log_population"] - _pop_mean) / _pop_std
        else:
            static["log_population"] = 0.0

        static["county_fips_encoded"] = static["unique_id"].map(
            lambda fips: float(_FIPS_INT.get(fips, -1))
        )
        # is_sludge: added by processor Stage 3; default 1.0 (copies/g is primary unit)
        if "is_sludge" in df.columns:
            is_sludge_map = df.groupby("unique_id")["is_sludge"].first()
            static["is_sludge"] = static["unique_id"].map(is_sludge_map)
        else:
            static["is_sludge"] = 1.0
        return static[["unique_id"] + STATIC_COVARIATES]
