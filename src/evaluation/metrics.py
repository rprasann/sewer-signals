"""
Pure metric functions for probabilistic forecasting and outbreak detection.

Category 1 — Probabilistic (continuous):
  wis()                   Weighted Interval Score (Bracher et al. 2021)
  pinball_loss()          Per-quantile pinball loss — validates quantile calibration
  coverage()              Empirical 50% and 95% PI coverage rates
  mae()                   Mean Absolute Error on the median forecast

Category 2 — Event-based (binary outbreak detection):
  match_alerts_to_onsets()  TP/FP/FN matching within a configurable time window
  detection_score()         Precision, Recall, F1, and TTD from TP/FP/FN counts

Result dataclasses:
  ProbabilisticResult  — aggregated Category 1 output for one evaluation window
  DetectionResult      — aggregated Category 2 output for one evaluation window
  EvalReport           — combined report (both categories) for one evaluation window

WIS formula (Bracher et al. 2021 / COVID-19 Forecast Hub)
----------------------------------------------------------
WIS(F, y) = [1 / (K + 0.5)] × [0.5 × |y − m|  +  Σ_k (α_k/2) × IS_k]

IS_α(y, l, u) = (u − l)  +  (2/α) × max(l − y, 0)  +  (2/α) × max(y − u, 0)

For K=2 prediction intervals (95% and 50%):
  α_1 = 0.05  →  (l, u) = (q0.025, q0.975)
  α_2 = 0.50  →  (l, u) = (q0.250, q0.750)
  normalisation = 1 / (2 + 0.5) = 0.4
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name mapping (NeuralForecast predict() output → semantic roles)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantileColumns:
    """Maps semantic quantile roles to column names in a forecast DataFrame.

    Defaults match TFT + PINNWastewaterLoss output with 7 quantiles.
    Optional q10/q90 are None when absent (5-quantile legacy forecasts).
    """

    q025: str       = "TFT-lo-95.0"
    q10:  str | None = None
    q25:  str       = "TFT-lo-50.0"
    q50:  str       = "TFT-median"
    q75:  str       = "TFT-hi-50.0"
    q90:  str | None = None
    q975: str       = "TFT-hi-95.0"

    @classmethod
    def auto_detect(cls, df: pd.DataFrame) -> "QuantileColumns":
        """Infer column names from DataFrame columns by matching known suffixes."""
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


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticResult:
    """Category 1 metrics for one evaluation window."""

    mean_wis:            float
    wis_by_county:       dict[str, float]        # FIPS → mean WIS
    pinball_by_quantile: dict[str, float]        # "q0.025" → mean pinball loss
    coverage_50:         float
    coverage_95:         float
    mae:                 float                   # MAE on median forecast
    n_observations:      int
    n_series:            int


@dataclass
class DetectionResult:
    """Category 2 metrics for one evaluation window.

    TTD (Time-to-Detection) = onset_date − alert_date in calendar days.
    Positive TTD = alert fired BEFORE the true onset (desired WW lead time).
    Negative TTD = alert fired AFTER the true onset (late detection).
    """

    tp:             int
    fp:             int
    fn:             int
    precision:      float    # TP / (TP + FP)
    recall:         float    # TP / (TP + FN)
    f1:             float    # harmonic mean of precision and recall
    mean_ttd_days:  float
    std_ttd_days:   float
    ttd_by_event:   list[float] = field(default_factory=list)
    n_actual_onsets: int = 0
    n_alerts:        int = 0


@dataclass
class EvalReport:
    """Combined probabilistic + detection report for one evaluation window."""

    prob:         ProbabilisticResult
    det:          Optional[DetectionResult]   # None when no OutbreakClassifier output
    cutoff_date:  Optional[pd.Timestamp] = None
    eval_start:   Optional[pd.Timestamp] = None
    eval_end:     Optional[pd.Timestamp] = None
    label:        str = ""

    def to_dict(self) -> dict:
        """Flat dict for CSV / DataFrame serialisation.

        Includes per-quantile pinball loss as ``pinball_q025`` … ``pinball_q975``
        when available, so cv_results.csv captures the full fold breakdown.
        """
        d: dict = {
            "cutoff_date":    self.cutoff_date,
            "mean_wis":       self.prob.mean_wis,
            "coverage_50":    self.prob.coverage_50,
            "coverage_95":    self.prob.coverage_95,
            "mae":            self.prob.mae,
            "n_observations": self.prob.n_observations,
        }
        # Flatten pinball dict: "q0.025" → column "pinball_q025"
        for q_label, pb_val in self.prob.pinball_by_quantile.items():
            col = "pinball_" + q_label.replace("q", "q").replace(".", "")
            d[col] = pb_val
        d.update({f"wis_{k}": v for k, v in self.prob.wis_by_county.items()})
        # Derived: pinball bias ratio — q0.10 / q0.90 (>1 = upward bias, <1 = downward)
        _pb = self.prob.pinball_by_quantile
        _q10 = _pb.get("q0.10", float("nan"))
        _q90 = _pb.get("q0.90", float("nan"))
        d["pinball_ratio"] = (_q10 / _q90) if (_q90 != 0 and not (np.isnan(_q10) or np.isnan(_q90))) else float("nan")
        if self.det is not None:
            d.update({
                "n_actual_onsets": self.det.n_actual_onsets,
                "n_alerts":        self.det.n_alerts,
                "tp":              self.det.tp,
                "fp":              self.det.fp,
                "fn":              self.det.fn,
                "precision":       self.det.precision,
                "recall":          self.det.recall,
                "f1":              self.det.f1,
                "mean_ttd_days":   self.det.mean_ttd_days,
            })
        else:
            d.update({
                "n_actual_onsets": 0,
                "n_alerts":        0,
                "tp": 0, "fp": 0, "fn": 0,
                "precision":     float("nan"),
                "recall":        float("nan"),
                "f1":            float("nan"),
                "mean_ttd_days": float("nan"),
            })
        return d

    def __str__(self) -> str:
        lines = [
            f"WIS={self.prob.mean_wis:.4f}  "
            f"Cov50={self.prob.coverage_50:.2%}  "
            f"Cov95={self.prob.coverage_95:.2%}  "
            f"MAE={self.prob.mae:.4f}"
        ]
        if self.det is not None:
            def _f(v: float) -> str:
                return f"{v:.3f}" if not np.isnan(v) else "N/A"
            lines.append(
                f"Precision={_f(self.det.precision)}  "
                f"Recall={_f(self.det.recall)}  "
                f"F1={_f(self.det.f1)}  "
                f"TTD={_f(self.det.mean_ttd_days)}d"
            )
        else:
            lines.append("Detection: N/A (no OutbreakClassifier output)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category 1 — Probabilistic metrics
# ---------------------------------------------------------------------------

def _interval_score(
    y_true: np.ndarray,
    lower:  np.ndarray,
    upper:  np.ndarray,
    alpha:  float,
) -> np.ndarray:
    spread    = upper - lower
    undershoot = np.maximum(lower - y_true, 0.0)
    overshoot  = np.maximum(y_true - upper, 0.0)
    return spread + (2.0 / alpha) * (undershoot + overshoot)


_DEFAULT_Q = QuantileColumns()


def wis(
    y_true:   np.ndarray | pd.Series,
    y_pred_df: pd.DataFrame,
    q_cols:   QuantileColumns = _DEFAULT_Q,
) -> np.ndarray:
    """Weighted Interval Score per observation (Bracher et al. 2021).

    Uses two central intervals (95% and 50%) plus the absolute error on the
    median.  Returns an array of shape (n_obs,); take .mean() for the scalar.
    """
    y    = np.asarray(y_true, dtype=float)
    q025 = y_pred_df[q_cols.q025].to_numpy(dtype=float)
    q25  = y_pred_df[q_cols.q25 ].to_numpy(dtype=float)
    q50  = y_pred_df[q_cols.q50 ].to_numpy(dtype=float)
    q75  = y_pred_df[q_cols.q75 ].to_numpy(dtype=float)
    q975 = y_pred_df[q_cols.q975].to_numpy(dtype=float)

    K    = 2
    norm = 1.0 / (K + 0.5)
    ae   = np.abs(y - q50)
    is95 = _interval_score(y, q025, q975, alpha=0.05)
    is50 = _interval_score(y, q25,  q75,  alpha=0.50)
    return norm * (0.5 * ae + (0.05 / 2) * is95 + (0.50 / 2) * is50)


def pinball_loss(
    y_true:    np.ndarray | pd.Series,
    y_pred_df: pd.DataFrame,
    q_cols:    QuantileColumns = _DEFAULT_Q,
) -> dict[str, float]:
    """Mean pinball (quantile) loss per quantile level.

    pinball_q(y, ŷ) = q × max(y − ŷ, 0) + (1−q) × max(ŷ − y, 0)

    Returns a dict mapping "q0.025" … "q0.975" to mean loss.
    Missing optional columns (q10, q90) are silently skipped.
    """
    y = np.asarray(y_true, dtype=float)

    quantile_map = [
        ("q0.025", q_cols.q025,  0.025),
        ("q0.10",  q_cols.q10,   0.10),
        ("q0.25",  q_cols.q25,   0.25),
        ("q0.50",  q_cols.q50,   0.50),
        ("q0.75",  q_cols.q75,   0.75),
        ("q0.90",  q_cols.q90,   0.90),
        ("q0.975", q_cols.q975,  0.975),
    ]

    result: dict[str, float] = {}
    for label, col, q in quantile_map:
        if col is None or col not in y_pred_df.columns:
            continue
        yhat   = y_pred_df[col].to_numpy(dtype=float)
        errors = y - yhat
        loss   = np.where(errors >= 0, q * errors, (q - 1) * errors)
        result[label] = float(loss.mean())
    return result


def coverage(
    y_true:    np.ndarray | pd.Series,
    y_pred_df: pd.DataFrame,
    q_cols:    QuantileColumns = _DEFAULT_Q,
) -> dict[str, float]:
    """Empirical coverage rates for the 50% and 95% prediction intervals.

    Returns dict with keys "coverage_50" and "coverage_95" in [0, 1].
    """
    y    = np.asarray(y_true, dtype=float)
    q25  = y_pred_df[q_cols.q25 ].to_numpy(dtype=float)
    q75  = y_pred_df[q_cols.q75 ].to_numpy(dtype=float)
    q025 = y_pred_df[q_cols.q025].to_numpy(dtype=float)
    q975 = y_pred_df[q_cols.q975].to_numpy(dtype=float)
    return {
        "coverage_50": float(np.mean((y >= q25)  & (y <= q75))),
        "coverage_95": float(np.mean((y >= q025) & (y <= q975))),
    }


def mae(
    y_true:        np.ndarray | pd.Series,
    y_pred_median: np.ndarray | pd.Series,
) -> float:
    """Mean Absolute Error of the median forecast against ground truth."""
    y   = np.asarray(y_true,        dtype=float)
    yh  = np.asarray(y_pred_median, dtype=float)
    return float(np.mean(np.abs(y - yh)))


# ---------------------------------------------------------------------------
# Category 2 — Event-based metrics
# ---------------------------------------------------------------------------

def match_alerts_to_onsets(
    alerts:           list[tuple[str, pd.Timestamp]],
    onsets:           list,      # list[OnsetEvent] — typed as list to avoid circular dep
    pre_window_weeks: int = 4,   # alert may fire up to this many weeks BEFORE onset
    post_window_weeks: int = 2,  # alert may fire up to this many weeks AFTER onset
) -> tuple[int, int, int, list[float]]:
    """Greedily match binary alerts to true onset events.

    Match window:
      [onset_date − pre_window_weeks×7,  onset_date + post_window_weeks×7]

    For each onset (sorted chronologically per county):
      - Find the earliest unmatched alert within the window → TP.
      - No matching alert → FN.
    Unmatched alerts → FP.

    Parameters
    ----------
    alerts : List of (county_id, alert_date) from OutbreakClassifier.
    onsets : List of OnsetEvent objects from OnsetLabeler.
    pre_window_weeks  : WW early-warning lead; default 4 weeks.
    post_window_weeks : Late-but-credited detection; default 2 weeks.

    Returns
    -------
    (tp, fp, fn, ttd_days_list)

    TTD = onset_date − alert_date in calendar days.
    Positive TTD = alert fired BEFORE onset (good early warning).
    """
    pre_days  = pre_window_weeks  * 7
    post_days = post_window_weeks * 7

    alerts_by_county: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for county, date in alerts:
        alerts_by_county[str(county)].append(pd.Timestamp(date))
    for county in alerts_by_county:
        alerts_by_county[county].sort()

    matched_keys: set[tuple[str, pd.Timestamp]] = set()
    tp = fn = 0
    ttd_days: list[float] = []

    sorted_onsets = sorted(onsets, key=lambda e: (e.county, e.onset_date))

    for event in sorted_onsets:
        county     = str(event.county)
        onset_date = pd.Timestamp(event.onset_date)
        best_alert: pd.Timestamp | None = None

        for alert_date in alerts_by_county.get(county, []):
            if (county, alert_date) in matched_keys:
                continue
            delta = (onset_date - alert_date).days
            # delta > 0 → alert is early (before onset); delta < 0 → alert is late
            if -post_days <= delta <= pre_days:
                if best_alert is None or alert_date < best_alert:
                    best_alert = alert_date

        if best_alert is not None:
            tp += 1
            matched_keys.add((county, best_alert))
            ttd_days.append(float((onset_date - best_alert).days))
        else:
            fn += 1

    total_alerts = sum(len(v) for v in alerts_by_county.values())
    fp = total_alerts - len(matched_keys)

    return tp, fp, fn, ttd_days


def detection_score(
    tp:              int,
    fp:              int,
    fn:              int,
    ttd_days:        list[float],
    n_actual_onsets: int = 0,
    n_alerts:        int = 0,
) -> DetectionResult:
    """Compute Precision, Recall, F1, and mean TTD from event counts.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 × Precision × Recall / (Precision + Recall)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    mean_ttd = float(np.mean(ttd_days)) if ttd_days       else float("nan")
    std_ttd  = float(np.std(ttd_days))  if len(ttd_days) >= 2 else float("nan")

    return DetectionResult(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        mean_ttd_days=mean_ttd, std_ttd_days=std_ttd,
        ttd_by_event=ttd_days,
        n_actual_onsets=n_actual_onsets,
        n_alerts=n_alerts,
    )
