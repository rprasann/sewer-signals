"""
Biosurveillance Evaluation Framework.

Public surface
--------------
QuantileColumns         — maps semantic quantile roles to actual DataFrame column names
wis()                   — Weighted Interval Score (Bracher et al. 2021)
coverage()              — 50 % and 95 % PI coverage rates
smape()                 — Symmetric Mean Absolute Percentage Error
OutbreakDetector        — onset detection: ≥25 % over 4-week rolling baseline
                          (absolute threshold guards against low-prevalence noise)
RecoveryEvent           — dataclass: peak date/value, recovery date, duration in weeks
OutbreakRecovery        — fall-phase detection: time from peak to ≥50 % sustained drop
LeadTimeEvaluator       — binary classifier: sensitivity, specificity, AUC, lead days
LagTimeResult           — dataclass: ww_trough_date, clinical_peak_date, lag_days
LagTimeAnalyzer         — lag between WW trough and clinical case peak (per county)
expanding_window_cv()   — expanding-window time-series cross-validation loop
evaluate()              — all-in-one convenience wrapper → EvalResult
EvalResult              — dataclass holding every metric for one evaluation window

WIS formula (Bracher et al. 2021 / COVID-19 Forecast Hub)
----------------------------------------------------------
WIS(F, y) = [1 / (K + 0.5)] × [0.5 × |y − m|  +  Σ_k (α_k/2) × IS_k]

IS_α(y, l, u) = (u − l)  +  (2/α) × max(l − y, 0)  +  (2/α) × max(y − u, 0)

For K=2 prediction intervals (95 % and 50 %):
  α_1 = 0.05  →  (l, u) = (q 0.025, q 0.975)
  α_2 = 0.50  →  (l, u) = (q 0.250, q 0.750)
  normalisation = 1 / (2 + 0.5) = 0.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from src.config import (
    COUNTY_COL,
    LEAD_TIME_WINDOW_MIN,
    LEAD_TIME_WINDOW_MAX,
    NWSS_DATE_COL,
    OUTBREAK_GROWTH_THRESHOLD,
    OUTBREAK_SUSTAINED_DAYS,
    QUANTILE_LEVELS,
    TARGET_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
    Z_OUTBREAK_THRESHOLD,
    Z_SCORE_BASELINE_WEEKS,
)

# ---------------------------------------------------------------------------
# Column-name mapping (NeuralForecast predict() output → semantic roles)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantileColumns:
    """Maps semantic quantile roles to the column names in a forecast DataFrame.

    Defaults match the output of ``PINNWastewaterLoss`` (MQLoss with
    quantiles=[0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]) wrapped in a
    ``TFT`` model.  The prefix is the model alias; NeuralForecast appends the
    output suffix.

    Example column names: ``"TFT-lo-95.0"``, ``"TFT-median"``

    ``q10`` and ``q90`` are optional: they are present when the TFT is
    configured with 7 quantiles but absent in test fixtures and legacy runs
    that used 5 quantiles.  When ``None``, the 80 % PI band is silently
    skipped in display/inversion steps.
    """

    q025: str       = "TFT-lo-95.0"
    q10:  str | None = None           # 80 % PI lower (q=0.10) — populated by auto_detect
    q25:  str       = "TFT-lo-50.0"
    q50:  str       = "TFT-median"
    q75:  str       = "TFT-hi-50.0"
    q90:  str | None = None           # 80 % PI upper (q=0.90) — populated by auto_detect
    q975: str       = "TFT-hi-95.0"

    @classmethod
    def auto_detect(cls, df: pd.DataFrame) -> "QuantileColumns":
        """Infer column names by scanning the DataFrame for known suffixes.

        Required columns (q025, q25, q50, q75, q975) raise ``ValueError`` if
        absent.  Optional columns (q10, q90) return ``None`` when absent so
        that 5-quantile legacy forecasts remain compatible.
        """
        cols = df.columns.tolist()

        def _find(suffix: str) -> str:
            matches = [c for c in cols if c.endswith(suffix)]
            if not matches:
                raise ValueError(f"No column ending with '{suffix}' found in {cols}")
            return matches[0]

        def _find_opt(suffix: str) -> str | None:
            matches = [c for c in cols if c.endswith(suffix)]
            return matches[0] if matches else None

        return cls(
            q025=_find("-lo-95.0"),
            q10=_find_opt("-lo-80.0"),
            q25=_find("-lo-50.0"),
            q50=_find("-median"),
            q75=_find("-hi-50.0"),
            q90=_find_opt("-hi-80.0"),
            q975=_find("-hi-95.0"),
        )


_DEFAULT_Q = QuantileColumns()


# ---------------------------------------------------------------------------
# Weighted Interval Score
# ---------------------------------------------------------------------------

def _interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-observation Interval Score for a single central prediction interval.

    IS_α = (u − l) + (2/α) × [max(l − y, 0) + max(y − u, 0)]
    """
    spread = upper - lower
    undershoot = np.maximum(lower - y_true, 0.0)
    overshoot = np.maximum(y_true - upper, 0.0)
    return spread + (2.0 / alpha) * (undershoot + overshoot)


def wis(
    y_true: np.ndarray | pd.Series,
    y_pred_df: pd.DataFrame,
    q_cols: QuantileColumns = _DEFAULT_Q,
) -> np.ndarray:
    """Weighted Interval Score per forecast horizon step (Bracher et al. 2021).

    Parameters
    ----------
    y_true   : Observed values aligned to ``y_pred_df`` row-by-row.
    y_pred_df: Forecast DataFrame with quantile columns (from NeuralForecast
               ``predict()``); rows must match ``y_true`` order.
    q_cols   : Column-name map; defaults match TFT + PINNWastewaterLoss output.

    Returns
    -------
    np.ndarray of shape ``(len(y_true),)`` — WIS per observation.
    """
    y = np.asarray(y_true, dtype=float)
    q025 = y_pred_df[q_cols.q025].to_numpy(dtype=float)
    q25  = y_pred_df[q_cols.q25 ].to_numpy(dtype=float)
    q50  = y_pred_df[q_cols.q50 ].to_numpy(dtype=float)
    q75  = y_pred_df[q_cols.q75 ].to_numpy(dtype=float)
    q975 = y_pred_df[q_cols.q975].to_numpy(dtype=float)

    K = 2  # number of central intervals
    norm = 1.0 / (K + 0.5)   # = 0.4

    ae = np.abs(y - q50)                                  # absolute error vs median

    is_95 = _interval_score(y, q025, q975, alpha=0.05)    # 95 % PI
    is_50 = _interval_score(y, q25,  q75,  alpha=0.50)    # 50 % PI

    return norm * (0.5 * ae + (0.05 / 2) * is_95 + (0.50 / 2) * is_50)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(
    y_true: np.ndarray | pd.Series,
    y_pred_df: pd.DataFrame,
    q_cols: QuantileColumns = _DEFAULT_Q,
) -> dict[str, float]:
    """Empirical coverage rates for the 50 % and 95 % prediction intervals.

    Returns
    -------
    dict with keys ``"coverage_50"`` and ``"coverage_95"`` (values in [0, 1]).
    """
    y    = np.asarray(y_true, dtype=float)
    q25  = y_pred_df[q_cols.q25 ].to_numpy(dtype=float)
    q75  = y_pred_df[q_cols.q75 ].to_numpy(dtype=float)
    q025 = y_pred_df[q_cols.q025].to_numpy(dtype=float)
    q975 = y_pred_df[q_cols.q975].to_numpy(dtype=float)

    cov_50 = float(np.mean((y >= q25)  & (y <= q75)))
    cov_95 = float(np.mean((y >= q025) & (y <= q975)))
    return {"coverage_50": cov_50, "coverage_95": cov_95}


# ---------------------------------------------------------------------------
# SMAPE
# ---------------------------------------------------------------------------

def smape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Symmetric Mean Absolute Percentage Error in [0, 2].

    SMAPE = mean(2 |y − ŷ| / (|y| + |ŷ| + ε))
    """
    y = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_pred, dtype=float)
    denom = np.abs(y) + np.abs(yh) + 1e-8
    return float(np.mean(2.0 * np.abs(y - yh) / denom))


# ---------------------------------------------------------------------------
# Outbreak onset detection
# ---------------------------------------------------------------------------

class OutbreakDetector:
    """Flags onset weeks using either a Z-score or a relative growth threshold.

    Onset conditions (all must hold simultaneously):
    1. Signal exceeds threshold — either:
       - Z-score mode (Phase 4, default): z ≥ ``z_threshold`` relative to the
         ``z_baseline_weeks``-week rolling mean/std of the *preceding* period.
         Adapts to local signal variance so the same threshold is meaningful
         across quiet inter-wave troughs and active surge periods.
       - Growth mode (Phase 3 legacy): WoW growth ≥ ``growth_threshold``
         relative to the ``baseline_weeks``-week rolling mean.
    2. Absolute signal ≥ ``min_absolute`` (prevents noise-driven false alarms
       during low-prevalence troughs where concentrations are near LOD).
    3. Condition sustained for ``sustained_steps`` consecutive weeks.

    Parameters
    ----------
    z_threshold      : Phase 4 Z-score threshold (default Z_OUTBREAK_THRESHOLD = 2.0).
                       Set to None to fall back to legacy growth_threshold mode.
    z_baseline_weeks : Rolling window for Z-score baseline (default Z_SCORE_BASELINE_WEEKS).
    growth_threshold : Phase 3 legacy fractional growth threshold (default 0.25).
                       Used only when z_threshold is None.
    baseline_weeks   : Rolling window for growth-mode baseline (default 4).
    min_absolute     : Absolute signal floor below which no onset is flagged.
    sustained_steps  : Consecutive above-threshold weeks to confirm onset.
    """

    def __init__(
        self,
        z_threshold: float | None = Z_OUTBREAK_THRESHOLD,
        z_baseline_weeks: int = Z_SCORE_BASELINE_WEEKS,
        growth_threshold: float = OUTBREAK_GROWTH_THRESHOLD,
        baseline_weeks: int = 4,
        min_absolute: float = 0.1,
        sustained_steps: int = OUTBREAK_SUSTAINED_DAYS,
    ) -> None:
        self.z_threshold = z_threshold
        self.z_baseline_weeks = z_baseline_weeks
        self.growth_threshold = growth_threshold
        self.baseline_weeks = baseline_weeks
        self.min_absolute = min_absolute
        self.sustained_steps = sustained_steps

    def detect(self, series: pd.Series) -> pd.Series:
        """Return a boolean Series aligned to ``series``.

        A True value marks the *first* week of a confirmed onset event.
        Multiple consecutive True values represent an ongoing wave.

        Parameters
        ----------
        series : Weekly signal (e.g. ``log1p_new_cases``), sorted chronologically.
        """
        series = series.copy().astype(float)

        if self.z_threshold is not None:
            # Phase 4: Z-score relative to rolling baseline (shift avoids leakage)
            shifted = series.shift(1)
            roll_mean = shifted.rolling(window=self.z_baseline_weeks, min_periods=4).mean()
            roll_std  = shifted.rolling(window=self.z_baseline_weeks, min_periods=4).std().fillna(0.0)
            z_score = (series - roll_mean) / (roll_std + 1e-8)
            per_step_flag = (z_score >= self.z_threshold) & (series >= self.min_absolute)
        else:
            # Phase 3 legacy: relative growth over short baseline
            baseline = (
                series.shift(1)
                .rolling(window=self.baseline_weeks, min_periods=2)
                .mean()
            )
            growth = (series - baseline) / (baseline.abs() + 1e-8)
            per_step_flag = (growth >= self.growth_threshold) & (series >= self.min_absolute)

        confirmed = (
            per_step_flag
            .rolling(window=self.sustained_steps, min_periods=self.sustained_steps)
            .sum()
            .ge(self.sustained_steps)
        )
        return confirmed.fillna(False)

    def detect_df(
        self,
        df: pd.DataFrame,
        signal_col: str = TARGET_COL,
        id_col: str = COUNTY_COL,
        date_col: str = NWSS_DATE_COL,
    ) -> pd.DataFrame:
        """Apply detection to every county in a processed DataFrame.

        Returns the input DataFrame with an added ``onset`` boolean column.
        """
        df = df.copy()
        flags = []
        for uid, grp in df.groupby(id_col):
            grp = grp.sort_values(date_col)
            flag = self.detect(grp[signal_col])
            flag_series = pd.Series(flag, index=grp.index, name="onset")
            flags.append(flag_series)
        df["onset"] = pd.concat(flags).reindex(df.index).fillna(False)
        return df


# ---------------------------------------------------------------------------
# Outbreak recovery ("The Fall")
# ---------------------------------------------------------------------------

@dataclass
class RecoveryEvent:
    """Describes a single outbreak wave's fall phase for one county series."""

    county: str
    peak_date: pd.Timestamp
    peak_value: float
    recovery_date: Optional[pd.Timestamp]   # None = wave not yet resolved
    duration_weeks: Optional[float]         # weeks peak→recovery; None if pending


class OutbreakRecovery:
    """Identifies the fall phase of an outbreak wave in a weekly signal.

    Recovery is declared when the signal drops to ≤ ``peak_fraction × peak_value``
    and **stays** there for at least ``sustained_steps`` consecutive weeks.

    The peak is the global maximum of the series (single-wave assumption).
    Only the post-peak portion is searched, preventing false recoveries before
    the true outbreak peak.

    Parameters
    ----------
    peak_fraction   : Fraction of peak value that defines the recovery threshold.
                      Default 0.5 = "signal fell to 50 % of its outbreak high".
    sustained_steps : Consecutive below-threshold weeks needed to confirm recovery.
    min_peak_value  : Peaks below this level are ignored (noise suppression).
    """

    def __init__(
        self,
        peak_fraction: float = 0.5,
        sustained_steps: int = 2,
        min_peak_value: float = 0.1,
    ) -> None:
        self.peak_fraction = peak_fraction
        self.sustained_steps = sustained_steps
        self.min_peak_value = min_peak_value

    def detect(
        self,
        series: pd.Series,
        county: str = "",
    ) -> list[RecoveryEvent]:
        """Detect the recovery event for a single county series.

        Parameters
        ----------
        series : Weekly WW signal with a sortable index.
        county : Optional county label stored in the returned RecoveryEvent.

        Returns
        -------
        List with at most one RecoveryEvent (empty if series is too short or
        peak is below ``min_peak_value``).
        """
        series = series.dropna().astype(float)
        if len(series) < 4 or series.max() < self.min_peak_value:
            return []

        pk_pos = int(series.values.argmax())
        peak_date = series.index[pk_pos]
        peak_value = float(series.iloc[pk_pos])
        threshold = self.peak_fraction * peak_value

        post_peak = series.iloc[pk_pos + 1:]
        if post_peak.empty:
            return [RecoveryEvent(county=county, peak_date=peak_date,
                                  peak_value=peak_value,
                                  recovery_date=None, duration_weeks=None)]

        below = (post_peak.values <= threshold).astype(int)

        # Rolling sum: position i has value k if below[i:i+k] are all 1
        if len(below) >= self.sustained_steps:
            kernel = np.ones(self.sustained_steps, dtype=int)
            rolling = np.convolve(below, kernel, mode="valid")
            run_starts = np.where(rolling == self.sustained_steps)[0]
        else:
            run_starts = np.array([])

        if run_starts.size > 0:
            rec_date = post_peak.index[int(run_starts[0])]
            duration_weeks = (rec_date - peak_date).days / 7.0
        else:
            rec_date = None
            duration_weeks = None

        return [RecoveryEvent(
            county=county,
            peak_date=peak_date,
            peak_value=peak_value,
            recovery_date=rec_date,
            duration_weeks=duration_weeks,
        )]

    def detect_df(
        self,
        df: pd.DataFrame,
        signal_col: str = TARGET_COL,
        id_col: str = COUNTY_COL,
        date_col: str = NWSS_DATE_COL,
    ) -> list[RecoveryEvent]:
        """Apply recovery detection to every county in a processed DataFrame."""
        all_events: list[RecoveryEvent] = []
        for uid, grp in df.groupby(id_col):
            grp = grp.sort_values(date_col).set_index(date_col)
            if signal_col not in grp.columns:
                continue
            events = self.detect(grp[signal_col], county=str(uid))
            all_events.extend(events)
        return all_events


# ---------------------------------------------------------------------------
# Lead-time & binary classifier evaluation
# ---------------------------------------------------------------------------

@dataclass
class LeadTimeResult:
    """Output of ``LeadTimeEvaluator.evaluate()``."""

    sensitivity: float      # TP / (TP + FN)
    specificity: float      # TN / (TN + FP)
    auc: float              # area under ROC using forecast slope as score
    mean_lead_days: float   # mean(actual_onset_date − predicted_alert_date)
    std_lead_days: float    # std of lead times; NaN when < 2 events
    lead_times: list[float] = field(default_factory=list)  # per-event lead days

    # Confusion-matrix counts for transparency
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def __str__(self) -> str:
        return (
            f"Sensitivity={self.sensitivity:.3f}  Specificity={self.specificity:.3f}  "
            f"AUC={self.auc:.3f}  "
            f"Lead={self.mean_lead_days:.1f}±{self.std_lead_days:.1f} days"
        )


class LeadTimeEvaluator:
    """Binary outbreak classifier and lead-time analyser.

    **Score**: Phase 4 default uses the maximum Z-score of the predicted median
    relative to an 8-week rolling baseline drawn from the actual history
    immediately before the forecast origin.  Adapts to local signal variance,
    producing a more calibrated continuous score for AUC computation than raw
    growth rates.

    **Decision threshold**: ``z_alert_threshold`` (default Z_OUTBREAK_THRESHOLD = 2.0
    when Z-score scoring is active, else ``growth_threshold`` = 0.25).

    **Label**: 1 if at least one confirmed onset (from ``OutbreakDetector``)
    falls within the forecast horizon window, else 0.

    **Lead time**: for each TP event,
    ``actual_onset_date − forecast_origin_date`` in calendar days.
    Positive = model alerted before the confirmed clinical onset.

    Parameters
    ----------
    detector              : Pre-configured OutbreakDetector for onset labelling.
    growth_threshold      : Phase 3 legacy binary alert threshold.
    horizon_weeks         : Forecast horizon length (weeks).
    z_score_baseline_weeks: Weeks of actual history before forecast origin used to
                            compute the Z-score baseline.  Set to 0 to fall back to
                            growth-rate scoring (Phase 3 behaviour).
    z_alert_threshold     : Decision threshold when Z-score scoring is active.
    """

    def __init__(
        self,
        detector: Optional[OutbreakDetector] = None,
        growth_threshold: float = OUTBREAK_GROWTH_THRESHOLD,
        horizon_weeks: int = 2,
        z_score_baseline_weeks: int = 0,
        z_alert_threshold: float = Z_OUTBREAK_THRESHOLD,
    ) -> None:
        self.detector = detector or OutbreakDetector()
        self.growth_threshold = growth_threshold
        self.horizon_weeks = horizon_weeks
        self.z_score_baseline_weeks = z_score_baseline_weeks
        self.z_alert_threshold = z_alert_threshold

    def evaluate(
        self,
        actual_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        q_cols: QuantileColumns = _DEFAULT_Q,
        id_col: str = "unique_id",
        date_col: str = "ds",
        actual_signal_col: str = TARGET_COL,
        actual_id_col: str = COUNTY_COL,
        actual_date_col: str = NWSS_DATE_COL,
    ) -> LeadTimeResult:
        """Compute sensitivity, specificity, AUC, and lead times.

        Parameters
        ----------
        actual_df   : Processed ground-truth DataFrame from the processor.
        forecast_df : NeuralForecast ``predict()`` output.
        q_cols      : Quantile column map.
        """
        # --- Label actual onsets in the evaluation window ---
        detector = self.detector
        labelled_actual = detector.detect_df(
            actual_df,
            signal_col=actual_signal_col,
            id_col=actual_id_col,
            date_col=actual_date_col,
        )

        # --- Compute per-window (county × forecast origin) binary labels & scores ---
        labels: list[int] = []
        scores: list[float] = []
        lead_times: list[float] = []
        tp = fp = tn = fn = 0

        # Pre-index actual history for fast per-county baseline lookup
        actual_indexed = actual_df.copy()

        for uid, fcast_grp in forecast_df.groupby(id_col):
            fcast_grp = fcast_grp.sort_values(date_col)

            # Actual onsets for this county
            actual_county = labelled_actual[
                labelled_actual[actual_id_col] == uid
            ].sort_values(actual_date_col)

            # Horizon window
            horizon_start = fcast_grp[date_col].min()
            horizon_end   = fcast_grp[date_col].max()

            # Label: was there an actual onset in this horizon window?
            onset_in_window = actual_county[
                (actual_county[actual_date_col] >= horizon_start)
                & (actual_county[actual_date_col] <= horizon_end)
                & actual_county["onset"]
            ]
            label = int(len(onset_in_window) > 0)

            # Score: Phase 4 growth-rate Z-score — z-score of the forecast's WoW
            # growth rates relative to the distribution of historical growth rates.
            # A flat forecast scores ~0 regardless of its level; only accelerating
            # trajectories score high.  Falls back to raw max growth rate when the
            # historical baseline is too short for a reliable std estimate.
            median = fcast_grp[q_cols.q50].to_numpy(dtype=float)
            use_z = self.z_score_baseline_weeks > 0 and len(median) > 1

            if use_z:
                actual_county_all = actual_indexed[
                    actual_indexed[actual_id_col] == uid
                ].sort_values(actual_date_col)
                history = actual_county_all[
                    actual_county_all[actual_date_col] < horizon_start
                ][actual_signal_col].to_numpy(dtype=float)
                if len(history) >= 4:
                    recent = history[-self.z_score_baseline_weeks:]
                    hist_growth = np.diff(recent) / (np.abs(recent[:-1]) + 1e-8)
                    bl_mean = float(np.mean(hist_growth))
                    bl_std  = float(np.std(hist_growth)) + 1e-8
                    fcast_growth = (median[1:] - median[:-1]) / (np.abs(median[:-1]) + 1e-8)
                    z_scores = (fcast_growth - bl_mean) / bl_std
                    score = float(z_scores.max())
                else:
                    use_z = False

            if not use_z:
                if len(median) > 1:
                    denom = np.abs(median[:-1]) + 1e-8
                    score = float(((median[1:] - median[:-1]) / denom).max())
                else:
                    score = 0.0

            labels.append(label)
            scores.append(score)

            # Binary decision uses Z-score threshold or growth threshold accordingly
            alert_threshold = (
                self.z_alert_threshold
                if self.z_score_baseline_weeks > 0
                else self.growth_threshold
            )
            alert = score >= alert_threshold

            if label == 1 and alert:
                tp += 1
                # Lead time: first actual onset date − forecast origin date
                first_onset = onset_in_window[actual_date_col].min()
                lead_days = (first_onset - horizon_start).days
                lead_times.append(float(lead_days))

            elif label == 1 and not alert:
                fn += 1
            elif label == 0 and alert:
                fp += 1
            else:
                tn += 1

        # --- Sensitivity / Specificity ---
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        # --- AUC (requires at least one positive and one negative example) ---
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(labels, scores))
        else:
            logger.warning(
                "AUC undefined: {} positive, {} negative windows. Returning 0.5.",
                n_pos, n_neg,
            )
            auc = 0.5

        # --- Lead time summary ---
        mean_lead = float(np.mean(lead_times)) if lead_times else float("nan")
        std_lead  = float(np.std(lead_times))  if len(lead_times) >= 2 else float("nan")

        return LeadTimeResult(
            sensitivity=sensitivity,
            specificity=specificity,
            auc=auc,
            mean_lead_days=mean_lead,
            std_lead_days=std_lead,
            lead_times=lead_times,
            tp=tp, fp=fp, tn=tn, fn=fn,
        )

    def roc_curve_data(
        self,
        actual_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        q_cols: QuantileColumns = _DEFAULT_Q,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (fpr, tpr, thresholds) for a full ROC curve.

        Uses the same Z-score scoring logic as ``evaluate()`` for consistency.
        """
        labels: list[int] = []
        scores: list[float] = []

        detector = self.detector
        labelled = detector.detect_df(actual_df, signal_col=TARGET_COL)

        for uid, fcast_grp in forecast_df.groupby("unique_id"):
            fcast_grp = fcast_grp.sort_values("ds")
            start, end = fcast_grp["ds"].min(), fcast_grp["ds"].max()

            actual_county = labelled[labelled[COUNTY_COL] == uid]
            onset_in_window = actual_county[
                (actual_county[NWSS_DATE_COL] >= start)
                & (actual_county[NWSS_DATE_COL] <= end)
                & actual_county["onset"]
            ]
            labels.append(int(len(onset_in_window) > 0))

            median = fcast_grp[q_cols.q50].to_numpy(dtype=float)
            use_z = self.z_score_baseline_weeks > 0 and len(median) > 1
            if use_z:
                hist_vals = actual_df[actual_df[COUNTY_COL] == uid].copy()
                hist_vals = hist_vals[hist_vals[NWSS_DATE_COL] < start][TARGET_COL].to_numpy(dtype=float)
                if len(hist_vals) >= 4:
                    recent = hist_vals[-self.z_score_baseline_weeks:]
                    hist_growth = np.diff(recent) / (np.abs(recent[:-1]) + 1e-8)
                    bl_mean = float(np.mean(hist_growth))
                    bl_std  = float(np.std(hist_growth)) + 1e-8
                    fcast_growth = (median[1:] - median[:-1]) / (np.abs(median[:-1]) + 1e-8)
                    scores.append(float(((fcast_growth - bl_mean) / bl_std).max()))
                    continue
            if len(median) > 1:
                denom = np.abs(median[:-1]) + 1e-8
                scores.append(float(((median[1:] - median[:-1]) / denom).max()))
            else:
                scores.append(0.0)

        labels_arr = np.array(labels)
        scores_arr = np.array(scores)
        if labels_arr.sum() == 0 or (1 - labels_arr).sum() == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        return roc_curve(labels_arr, scores_arr)


# ---------------------------------------------------------------------------
# Lag-time analysis
# ---------------------------------------------------------------------------

@dataclass
class LagTimeResult:
    """Lag between WW signal trough and clinical case peak for one county.

    ``lag_days = clinical_peak_date − ww_trough_date``

    * Positive lag → clinical cases peaked AFTER WW troughed
      (WW is a leading indicator of recovery: it began recovering before cases peaked).
    * Negative lag → clinical cases peaked BEFORE WW troughed
      (cases started declining while WW was still elevated).
    """

    county: str
    ww_trough_date: Optional[pd.Timestamp]
    clinical_peak_date: Optional[pd.Timestamp]
    lag_days: Optional[float]


class LagTimeAnalyzer:
    """Computes the lag between the WW signal trough and the clinical case peak.

    The WW trough is the minimum value in the ``trough_window_weeks`` period
    immediately following the WW peak.  The clinical peak is the global maximum
    of the provided case series.

    Parameters
    ----------
    trough_window_weeks : How far post-WW-peak to search for the trough (default 12).
    """

    def __init__(self, trough_window_weeks: int = 12) -> None:
        self.trough_window_weeks = trough_window_weeks

    def compute(
        self,
        ww_series: pd.Series,
        case_series: pd.Series,
        county: str = "",
    ) -> LagTimeResult:
        """Compute the lag for a single county.

        Parameters
        ----------
        ww_series   : Weekly WW concentration (log1p-scaled), DatetimeIndex.
        case_series : Weekly clinical case counts, DatetimeIndex.
        county      : Label stored in the result.

        Returns
        -------
        LagTimeResult — ``lag_days`` is None when either series is too short.
        """
        ww = ww_series.dropna().astype(float)
        cases = case_series.dropna().astype(float)

        _null = LagTimeResult(county=county, ww_trough_date=None,
                              clinical_peak_date=None, lag_days=None)

        if len(ww) < 4 or len(cases) < 2:
            return _null

        # WW trough: minimum within trough_window_weeks after the WW peak
        ww_pk_pos = int(ww.values.argmax())
        ww_peak_date = ww.index[ww_pk_pos]
        window_end = ww_peak_date + pd.Timedelta(weeks=self.trough_window_weeks)
        post_peak_ww = ww[(ww.index > ww_peak_date) & (ww.index <= window_end)]

        if post_peak_ww.empty:
            return _null

        ww_trough_date = post_peak_ww.index[int(post_peak_ww.values.argmin())]

        # Clinical case peak: global maximum of the case series
        clinical_peak_date = cases.index[int(cases.values.argmax())]

        lag_days = float((clinical_peak_date - ww_trough_date).days)

        return LagTimeResult(
            county=county,
            ww_trough_date=ww_trough_date,
            clinical_peak_date=clinical_peak_date,
            lag_days=lag_days,
        )

    def compute_df(
        self,
        ww_df: pd.DataFrame,
        case_df: pd.DataFrame,
        ww_signal_col: str = TARGET_COL,
        ww_id_col: str = COUNTY_COL,
        ww_date_col: str = NWSS_DATE_COL,
        case_signal_col: str = "new_cases",
        case_id_col: str = COUNTY_COL,
        case_date_col: str = NWSS_DATE_COL,
    ) -> list[LagTimeResult]:
        """Compute lag for every county shared between the two DataFrames."""
        results: list[LagTimeResult] = []
        for uid, ww_grp in ww_df.groupby(ww_id_col):
            ww_grp = ww_grp.sort_values(ww_date_col).set_index(ww_date_col)
            case_grp = (
                case_df[case_df[case_id_col] == uid]
                .sort_values(case_date_col)
                .set_index(case_date_col)
            )
            if case_grp.empty or case_signal_col not in case_grp.columns:
                results.append(LagTimeResult(county=str(uid), ww_trough_date=None,
                                             clinical_peak_date=None, lag_days=None))
                continue
            result = self.compute(
                ww_grp[ww_signal_col], case_grp[case_signal_col], county=str(uid)
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# All-in-one evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """All metrics for a single evaluation window (one CV fold or final eval)."""

    # Probabilistic
    mean_wis: float
    wis_per_county: dict[str, float]    # county FIPS → mean WIS
    coverage_50: float
    coverage_95: float
    smape: float

    # Outbreak binary classification
    n_actual_onsets: int
    n_predicted_alerts: int
    lead_time: LeadTimeResult

    # Recovery ("The Fall")
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    mean_recovery_weeks: float = float("nan")

    # Window metadata
    cutoff_date: Optional[pd.Timestamp] = None
    n_series: int = 0
    n_observations: int = 0

    def to_dict(self) -> dict:
        d = {
            "cutoff_date": self.cutoff_date,
            "mean_wis": self.mean_wis,
            "coverage_50": self.coverage_50,
            "coverage_95": self.coverage_95,
            "smape": self.smape,
            "n_actual_onsets": self.n_actual_onsets,
            "n_predicted_alerts": self.n_predicted_alerts,
            "sensitivity": self.lead_time.sensitivity,
            "specificity": self.lead_time.specificity,
            "auc": self.lead_time.auc,
            "mean_lead_days": self.lead_time.mean_lead_days,
            "mean_recovery_weeks": self.mean_recovery_weeks,
            "n_observations": self.n_observations,
        }
        d.update({f"wis_{k}": v for k, v in self.wis_per_county.items()})
        return d

    def __str__(self) -> str:
        rec_str = (
            f"  Recovery: {self.mean_recovery_weeks:.1f} wks avg ({len(self.recovery_events)} waves)"
            if not np.isnan(self.mean_recovery_weeks)
            else "  Recovery: pending/insufficient data"
        )
        return (
            f"WIS={self.mean_wis:.4f}  Cov50={self.coverage_50:.2%}  "
            f"Cov95={self.coverage_95:.2%}  SMAPE={self.smape:.4f}\n"
            f"Outbreaks actual={self.n_actual_onsets}  alerts={self.n_predicted_alerts}\n"
            f"{self.lead_time}\n{rec_str}"
        )


def evaluate(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    q_cols: Optional[QuantileColumns] = None,
    min_absolute: float = 0.1,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> EvalResult:
    """Compute all metrics for one evaluation window.

    Parameters
    ----------
    actual_df   : Processed ground-truth DataFrame (processor output); must
                  contain ``COUNTY_COL``, ``NWSS_DATE_COL``, ``TARGET_COL``.
    forecast_df : NeuralForecast ``predict()`` output; must contain
                  ``unique_id``, ``ds``, and quantile columns.
    q_cols      : Column map; auto-detected if None.
    min_absolute: Absolute signal floor for the outbreak detector.
    cutoff_date : Training cutoff for this fold (stored in EvalResult metadata).
    """
    if q_cols is None:
        q_cols = QuantileColumns.auto_detect(forecast_df)

    # --- Align actual to forecast rows ---
    # Merge on unique_id (= county_fips) and ds (= sample_collect_date)
    merged = forecast_df.copy()
    merged = merged.merge(
        actual_df.rename(columns={
            COUNTY_COL: "unique_id",
            NWSS_DATE_COL: "ds",
            TARGET_COL: "y_true",
        })[["unique_id", "ds", "y_true"]],
        on=["unique_id", "ds"],
        how="inner",
    )

    if merged.empty:
        logger.warning(
            "evaluate(): forecast and actuals share no overlapping (unique_id, ds) pairs "
            "— returning empty EvalResult (n_observations=0).  "
            "Check that the forecast window overlaps the evaluation period."
        )
        _null_lead = LeadTimeResult(
            sensitivity=float("nan"),
            specificity=float("nan"),
            auc=float("nan"),
            mean_lead_days=float("nan"),
            std_lead_days=float("nan"),
        )
        return EvalResult(
            mean_wis=float("nan"),
            wis_per_county={},
            coverage_50=float("nan"),
            coverage_95=float("nan"),
            smape=float("nan"),
            n_actual_onsets=0,
            n_predicted_alerts=0,
            lead_time=_null_lead,
            recovery_events=[],
            mean_recovery_weeks=float("nan"),
            cutoff_date=cutoff_date,
            n_series=0,
            n_observations=0,
        )

    y_true = merged["y_true"].to_numpy(dtype=float)

    # --- Probabilistic metrics ---
    wis_vals = wis(y_true, merged, q_cols)
    cov      = coverage(y_true, merged, q_cols)
    smape_val = smape(y_true, merged[q_cols.q50].to_numpy(dtype=float))

    wis_per_county = {
        uid: float(wis_vals[merged["unique_id"] == uid].mean())
        for uid in merged["unique_id"].unique()
    }

    # --- Outbreak / lead-time metrics (Phase 4: Z-score detector + Z-score AUC) ---
    # Explicitly opt into Z-score detection in the all-in-one wrapper.
    detector   = OutbreakDetector(min_absolute=min_absolute, z_threshold=Z_OUTBREAK_THRESHOLD)
    evaluator  = LeadTimeEvaluator(
        detector=detector,
        z_score_baseline_weeks=Z_SCORE_BASELINE_WEEKS,  # opt-in to Phase 4 Z-score scoring
    )
    lt_result  = evaluator.evaluate(actual_df, forecast_df, q_cols)

    labelled = detector.detect_df(actual_df)
    n_actual = int(labelled["onset"].sum())

    # Alert count: windows where the model predicted an onset (TP + FP from Z-score evaluator).
    # Previously computed from a stale OUTBREAK_GROWTH_THRESHOLD loop that was inconsistent
    # with the Z-score detector used everywhere else.  Now derived directly from lt_result
    # so the reported n_predicted_alerts always reflects the same scoring logic as AUC/Sensitivity.
    n_alerts = lt_result.tp + lt_result.fp

    # --- Recovery / fall-phase metrics ---
    recovery_analyzer = OutbreakRecovery(min_peak_value=min_absolute)
    recovery_events = recovery_analyzer.detect_df(actual_df)
    completed = [e for e in recovery_events if e.duration_weeks is not None]
    mean_recovery_weeks = (
        float(np.mean([e.duration_weeks for e in completed]))
        if completed else float("nan")
    )

    return EvalResult(
        mean_wis=float(wis_vals.mean()),
        wis_per_county=wis_per_county,
        coverage_50=cov["coverage_50"],
        coverage_95=cov["coverage_95"],
        smape=smape_val,
        n_actual_onsets=n_actual,
        n_predicted_alerts=n_alerts,
        lead_time=lt_result,
        recovery_events=recovery_events,
        mean_recovery_weeks=mean_recovery_weeks,
        cutoff_date=cutoff_date,
        n_series=merged["unique_id"].nunique(),
        n_observations=len(merged),
    )


# ---------------------------------------------------------------------------
# Targeted outbreak-window validation
# ---------------------------------------------------------------------------

@dataclass
class OutbreakWindowResult:
    """Evaluation result for one historical outbreak validation window."""
    name:        str
    eval_start:  str
    eval_end:    str
    n_counties:  int
    eval_result: EvalResult


def run_outbreak_validation(
    processed_df: pd.DataFrame,
    model_factory: Callable[[], Any],
    windows: list[dict],
) -> list[OutbreakWindowResult]:
    """Train and evaluate one model per historical outbreak window.

    For each window the model is trained on all data up to 1 week before
    ``eval_start``, then asked to forecast H steps starting from ``eval_start``.
    This simulates operational deployment: the model never sees data from
    the outbreak period it is being asked to detect.

    Parameters
    ----------
    processed_df  : Full scaled dataset (all splits concatenated).
    model_factory : Callable returning a fresh, unfitted model — should use
                    fast settings (e.g. max_steps=500, no progress bar).
    windows       : List of dicts with keys:
                      name        — display label
                      eval_start  — ISO date string (W-WED)
                      eval_end    — ISO date string (W-WED)
                      counties    — list of FIPS strings, or None for all
    """
    model_h: int = getattr(model_factory(), "h", 8)
    results: list[OutbreakWindowResult] = []

    for win in windows:
        name       = win["name"]
        eval_start = pd.Timestamp(win["eval_start"])
        eval_end   = pd.Timestamp(win["eval_end"])
        counties   = win.get("counties")

        df_win = (
            processed_df[processed_df[COUNTY_COL].isin(counties)].copy()
            if counties else processed_df.copy()
        )
        if df_win.empty:
            logger.warning("Outbreak window '{}': no data for counties {}. Skipping.", name, counties)
            continue

        # Train cutoff: 1 week before eval_start so the FIRST forecast step
        # lands exactly on eval_start, covering the full eval window.
        # (H weeks before eval_start would make the forecast *end* at eval_start,
        # leaving only a single overlapping data point.)
        train_cutoff = eval_start - pd.Timedelta(weeks=1)
        train_data   = df_win[pd.to_datetime(df_win[NWSS_DATE_COL]) <= train_cutoff].copy()

        if train_data.empty or train_data[COUNTY_COL].nunique() < 1:
            logger.warning(
                "Outbreak window '{}': no training data before {}. Skipping.",
                name, train_cutoff.date(),
            )
            continue

        logger.info(
            "Outbreak validation '{}' — train ≤ {}  eval {} → {}  ({} counties)",
            name, train_cutoff.date(), eval_start.date(), eval_end.date(),
            df_win[COUNTY_COL].nunique(),
        )

        try:
            model = model_factory()
            model.fit(train_data, val_size=model.h)
            forecast_df = model.predict()
        except Exception as exc:
            logger.warning("Outbreak window '{}': training failed — {}. Skipping.", name, exc)
            continue

        forecast_eval = forecast_df[
            (pd.to_datetime(forecast_df["ds"]) >= eval_start) &
            (pd.to_datetime(forecast_df["ds"]) <= eval_end)
        ].copy()
        actual_eval = df_win[
            (pd.to_datetime(df_win[NWSS_DATE_COL]) >= eval_start) &
            (pd.to_datetime(df_win[NWSS_DATE_COL]) <= eval_end)
        ].copy()

        if forecast_eval.empty or actual_eval.empty:
            logger.warning(
                "Outbreak window '{}': forecast does not overlap eval period "
                "(forecast covers {} → {}).  Skipping.",
                name,
                forecast_df["ds"].min().date() if not forecast_df.empty else "?",
                forecast_df["ds"].max().date() if not forecast_df.empty else "?",
            )
            continue

        q_cols     = QuantileColumns.auto_detect(forecast_eval)
        eval_result = evaluate(actual_df=actual_eval, forecast_df=forecast_eval, q_cols=q_cols)

        sens = eval_result.lead_time.sensitivity
        logger.info(
            "  → Cov95={:.1%}  WIS={:.3f}  Sensitivity={}  AUC={:.3f}",
            eval_result.coverage_95,
            eval_result.mean_wis,
            f"{sens:.2f}" if not (isinstance(sens, float) and np.isnan(sens)) else "N/A",
            eval_result.lead_time.auc,
        )

        results.append(OutbreakWindowResult(
            name=name,
            eval_start=win["eval_start"],
            eval_end=win["eval_end"],
            n_counties=df_win[COUNTY_COL].nunique(),
            eval_result=eval_result,
        ))

    return results


# ---------------------------------------------------------------------------
# Expanding-window time-series cross-validation
# ---------------------------------------------------------------------------

def expanding_window_cv(
    processed_df: pd.DataFrame,
    model_factory: Callable[[], Any],
    initial_train_end: str = TRAIN_END_DATE,
    eval_end: str = VAL_END_DATE,
    step_weeks: int = 4,
    h: Optional[int] = None,
    q_cols: Optional[QuantileColumns] = None,
    min_absolute: float = 0.1,
    date_col: str = NWSS_DATE_COL,
    forecast_collector: Optional[list] = None,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    Each fold trains on all data up to a cutoff date, then evaluates on
    the next ``h`` weeks.  The training window expands by ``step_weeks``
    each iteration.

    Parameters
    ----------
    processed_df    : Full processor-output DataFrame (all splits combined).
    model_factory   : Zero-argument callable returning a fresh, unfitted model.
                      Example: ``lambda: WastewaterTFT(max_steps=500)``
    initial_train_end : ISO date string; training starts here (first cutoff).
    eval_end        : ISO date string; last evaluation cutoff.
    step_weeks      : Weeks between successive cutoff dates.
    h               : Forecast horizon (weeks).  Inferred from model if None.
    q_cols          : Quantile column map (auto-detected if None).
    min_absolute    : Absolute floor for outbreak detector.
    date_col        : Date column name in processed_df.

    Returns
    -------
    pd.DataFrame with one row per fold and all metrics as columns.
    """
    # W-WED anchor keeps cutoffs on Wednesdays, aligned with the data spine.
    cutoffs = pd.date_range(
        start=pd.Timestamp(initial_train_end),
        end=pd.Timestamp(eval_end),
        freq=f"{step_weeks}W-WED",
    )

    if len(cutoffs) == 0:
        raise ValueError(
            f"No cutoffs between {initial_train_end} and {eval_end} "
            f"at {step_weeks}-week steps."
        )

    results: list[dict] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]CV fold[/bold cyan] {task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("", total=len(cutoffs))

        for fold_idx, cutoff in enumerate(cutoffs):
            progress.update(
                task,
                description=f"[white]{fold_idx + 1}/{len(cutoffs)}[/white] "
                            f"cutoff [yellow]{cutoff.date()}[/yellow]",
            )
            logger.info(
                "CV fold {}/{} — cutoff date {}.",
                fold_idx + 1, len(cutoffs), cutoff.date(),
            )

            # Expanding training window
            train = processed_df[processed_df[date_col] <= cutoff].copy()

            # Determine horizon from model or default to 2 weeks
            model = model_factory()
            horizon = h if h is not None else getattr(model, "h", 2)

            val_end = cutoff + pd.Timedelta(weeks=horizon)
            val = processed_df[
                (processed_df[date_col] > cutoff)
                & (processed_df[date_col] <= val_end)
            ].copy()

            if len(train) < horizon * 4:
                logger.warning(
                    "Fold {}: insufficient training rows ({}) — skipping.",
                    fold_idx + 1, len(train),
                )
                progress.advance(task)
                continue
            if val.empty:
                logger.warning(
                    "Fold {}: empty validation window — skipping.", fold_idx + 1
                )
                progress.advance(task)
                continue

            try:
                # val_size=0: CV fold models are expected to have early stopping
                # disabled (early_stop_patience_steps=-1 in their trainer_kwargs).
                # This is unconditionally safe because:
                #   - NeuralForecast requires val_size ∈ {0} ∪ [h, ∞); no mid-values.
                #   - val_size=0 requires early stopping to be off, otherwise NF raises.
                #   - Short-history counties (e.g. Napa ≤ h rows at early cutoffs)
                #     make val_size=h crash regardless of the above.
                # CV evaluation is done externally via evaluate(), so internal early
                # stopping is unnecessary.
                model.fit(train, val_size=0)
                forecast = model.predict()

                if forecast_collector is not None and not forecast.empty:
                    _fc = forecast.copy()
                    _fc["cutoff_date"] = cutoff
                    forecast_collector.append(_fc)

                # Auto-detect column names on first fold with data
                if q_cols is None and not forecast.empty:
                    try:
                        q_cols = QuantileColumns.auto_detect(forecast)
                        logger.info("Auto-detected quantile columns: {}", q_cols)
                    except ValueError as exc:
                        logger.error("Could not auto-detect quantile columns: {}", exc)
                        raise

                result = evaluate(
                    actual_df=val,
                    forecast_df=forecast,
                    q_cols=q_cols,
                    min_absolute=min_absolute,
                    cutoff_date=cutoff,
                )
                if result.n_observations > 0:
                    results.append(result.to_dict())
                else:
                    logger.warning(
                        "Fold {}: no overlapping observations — fold skipped from results.",
                        fold_idx + 1,
                    )

            except Exception as exc:
                logger.error("Fold {} failed: {}", fold_idx + 1, exc)
            finally:
                progress.advance(task)

    if not results:
        logger.warning("No CV folds completed successfully.")
        return pd.DataFrame()

    cv_df = pd.DataFrame(results)
    logger.info(
        "CV complete — {} folds, mean WIS={:.4f}, mean AUC={:.3f}.",
        len(cv_df),
        cv_df["mean_wis"].mean(),
        cv_df["auc"].mean(),
    )
    return cv_df
