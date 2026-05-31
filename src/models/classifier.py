"""
OutbreakClassifier — Stage 1 of the two-stage inference system.

Design: Non-Elastic Baseline + Momentum Confirmation
------------------------------------------------------
The classifier uses two independent signals that must both exceed their
thresholds before issuing a surge trigger.

Signal 1 — Z-score with non-elastic baseline
  The baseline is anchored to the QUIET-PERIOD distribution of the training
  data — specifically, observations below the training median.  Once fitted,
  this baseline is frozen: it never ingests surge data.

  Why non-elastic matters:
    Rolling Z-score (8-week window):
      Week 0: signal=1.0, baseline_mean=1.1,  Z=-0.1  (quiet ✓)
      Week 1: signal=3.0, baseline_mean=1.2,  Z=1.5   (alert ✓)
      Week 6: signal=4.0, baseline_mean=2.8,  Z=0.43  (MISS — baseline
              absorbed the surge, normalising it away)

    Non-elastic (training-anchored) baseline:
      Week 0: signal=1.0, baseline_mean=0.9,  Z=0.14  (quiet ✓)
      Week 1: signal=3.0, baseline_mean=0.9,  Z=2.1   (alert ✓)
      Week 6: signal=4.0, baseline_mean=0.9,  Z=3.4   (sustained alert ✓)

Signal 2 — Momentum divergence (ww_momentum_lead)
  ww_momentum_lead[t] = vel_concentration[t]
                        - (log1p_cases[t-1] - log1p_cases[t-2])

  A large positive value means WW is accelerating while cases are still
  flat or declining — the exact signature of WW leading a case surge.
  This rejects false alarms caused by single-week WW spikes that are not
  accompanied by an accelerating WW trend.

Gate logic:
  triggered = (z_score >= effective_threshold) AND (momentum >= MOMENTUM_THRESHOLD)

Volatility-Adjusted Threshold
------------------------------
In high-noise environments (e.g. noisy WW signal in winter), a fixed Z-score
threshold produces false positives because normal seasonal fluctuations exceed
the baseline by 1.5σ.  To prevent this, the effective threshold scales with
local WW volatility (log1p_concentration_4w_std):

  vol_ratio           = local_vol / training_mean_vol
  effective_threshold = z_threshold × (1 + volatility_scale × max(vol_ratio − 1, 0))

When the 4-week WW std is 2× the training mean std, the threshold rises from
1.5 → 2.25.  When it is at or below average, the threshold stays at 1.5.

Cold-Start Handling
-------------------
Counties with < min_observations training rows cannot produce a reliable
baseline.  The classifier returns all-False (Quiet) for these counties —
OutbreakForecaster then issues the flat quiet prior.

Public API
----------
  OutbreakClassifier.fit(train_df)
  OutbreakClassifier.classify_df(eval_df)  → pd.DataFrame with triggered column
  OutbreakClassifier.classify_series(signal_series, momentum_series, county_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    CLASSIFIER_BASELINE_WEEKS,
    CLASSIFIER_MIN_SIGNAL,
    CLASSIFIER_MOMENTUM_THRESHOLD,
    CLASSIFIER_VOLATILITY_COL,
    CLASSIFIER_VOLATILITY_SCALE,
    CLASSIFIER_Z_THRESHOLD,
    COUNTY_COL,
    MIN_BASELINE_OBSERVATIONS,
    NWSS_DATE_COL,
    TARGET_COL,
    WW_FEATURE_COL,
)


# ---------------------------------------------------------------------------
# Per-observation output
# ---------------------------------------------------------------------------

@dataclass
class ClassifierRecord:
    """Classification result for one (county, date) observation."""

    unique_id:  str
    date:       pd.Timestamp
    triggered:  bool
    z_score:    float
    momentum:   float
    signal:     float    # raw (unscaled) signal value


# ---------------------------------------------------------------------------
# Baseline state (fitted per county)
# ---------------------------------------------------------------------------

@dataclass
class _CountyBaseline:
    """Non-elastic baseline parameters for one county.

    Derived exclusively from training-period quiet observations (below
    the training median) to anchor the reference distribution.
    """

    mean:            float
    std:             float
    quiet_threshold: float    # training median; separates quiet vs. active
    mean_vol:        float = 1.0  # mean training volatility (4w std) for threshold scaling


# ---------------------------------------------------------------------------
# OutbreakClassifier
# ---------------------------------------------------------------------------

class OutbreakClassifier:
    """Lightweight two-signal gatekeeper for Stage 1 of the inference system.

    Replaces the legacy Z-score rolling-baseline detector with a fixed,
    non-elastic baseline that does not drift upward during sustained surges.

    Parameters
    ----------
    z_threshold         : Base Z-score threshold.  Scaled up in high-volatility
                          environments (see volatility_scale).
    momentum_threshold  : ``ww_momentum_lead`` must be ≥ this value to confirm.
                          0.0 = any positive divergence; raise for fewer FPs.
    baseline_weeks      : Maximum quiet-period training weeks for baseline.
    min_signal          : Absolute signal floor; values below this are never
                          triggered (prevents LOD-noise alerts in quiet troughs).
    volatility_col      : Column for local WW volatility (log1p_concentration_4w_std).
                          When present, enables the dynamic threshold.
    volatility_scale    : Sensitivity of the threshold adjustment.
                          effective_z = z × (1 + scale × max(local_vol/mean_vol − 1, 0))
    min_observations    : Counties with fewer training rows are cold-started
                          (all-False, quiet prior returned).
    signal_col          : Column name for the WW surveillance signal.
    momentum_col        : Column name for the momentum divergence feature.
    id_col              : Series identifier column.
    date_col            : Date column.
    """

    def __init__(
        self,
        z_threshold:        float      = CLASSIFIER_Z_THRESHOLD,
        momentum_threshold: float      = CLASSIFIER_MOMENTUM_THRESHOLD,
        baseline_weeks:     int        = CLASSIFIER_BASELINE_WEEKS,
        min_signal:         float      = CLASSIFIER_MIN_SIGNAL,
        volatility_col:     str | None = CLASSIFIER_VOLATILITY_COL,
        volatility_scale:   float      = CLASSIFIER_VOLATILITY_SCALE,
        min_observations:   int        = MIN_BASELINE_OBSERVATIONS,
        signal_col:         str        = WW_FEATURE_COL,
        momentum_col:       str        = "ww_momentum_lead",
        id_col:             str        = COUNTY_COL,
        date_col:           str        = NWSS_DATE_COL,
    ) -> None:
        self.z_threshold         = z_threshold
        self.momentum_threshold  = momentum_threshold
        self.baseline_weeks      = baseline_weeks
        self.min_signal          = min_signal
        self.volatility_col      = volatility_col
        self.volatility_scale    = volatility_scale
        self.min_observations    = min_observations
        self.signal_col          = signal_col
        self.momentum_col        = momentum_col
        self.id_col              = id_col
        self.date_col            = date_col
        self._baselines: dict[str, _CountyBaseline] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "OutbreakClassifier":
        """Compute per-county non-elastic baseline from training data.

        The baseline is anchored to the quiet-period distribution of each
        county's training signal.  "Quiet" = weeks where the signal is
        below the per-county training median.

        Parameters
        ----------
        train_df : Processed training DataFrame; must contain signal_col,
                   id_col, and date_col.

        Returns self for method chaining.
        """
        self._baselines = {}
        for county, grp in train_df.groupby(self.id_col):
            grp_sorted = grp.sort_values(self.date_col)
            signal = grp_sorted[self.signal_col].dropna().to_numpy(dtype=float)

            # Cold-start: too few observations to establish a reliable baseline
            if len(signal) < self.min_observations:
                logger.warning(
                    "OutbreakClassifier.fit: county {} has only {} signal rows "
                    "(< min_observations={}) — cold-start, will return all-False.",
                    county, len(signal), self.min_observations,
                )
                continue

            quiet_threshold = float(np.median(signal))
            quiet_signal    = signal[signal < quiet_threshold]
            if len(quiet_signal) < 2:
                quiet_signal = signal
            if len(quiet_signal) > self.baseline_weeks:
                quiet_signal = quiet_signal[-self.baseline_weeks:]

            # Track mean training volatility for the dynamic threshold
            mean_vol = 1.0
            if self.volatility_col and self.volatility_col in grp_sorted.columns:
                vol_vals = grp_sorted[self.volatility_col].dropna().to_numpy(dtype=float)
                if len(vol_vals) > 0:
                    mean_vol = float(vol_vals.mean()) + 1e-8

            self._baselines[str(county)] = _CountyBaseline(
                mean=float(quiet_signal.mean()),
                std=float(quiet_signal.std(ddof=1) if len(quiet_signal) > 1 else 0.0),
                quiet_threshold=quiet_threshold,
                mean_vol=mean_vol,
            )

        logger.info(
            "OutbreakClassifier fitted on {} counties "
            "(Z≥{:.1f}, momentum≥{:.2f}, vol_scale={:.2f}).",
            len(self._baselines), self.z_threshold, self.momentum_threshold,
            self.volatility_scale,
        )
        return self

    # ------------------------------------------------------------------
    # Inference — per-series
    # ------------------------------------------------------------------

    def classify_series(
        self,
        signal_series:   pd.Series,
        momentum_series: Optional[pd.Series],
        county_id:       str,
        vol_series:      Optional[pd.Series] = None,
    ) -> list[ClassifierRecord]:
        """Classify every observation in a single county's time series.

        Parameters
        ----------
        signal_series   : Weekly WW signal, sorted chronologically.
        momentum_series : ww_momentum_lead values, same index as signal_series.
                          Pass None to run in Z-score-only mode.
        county_id       : County FIPS string (or other id used during fit).
        vol_series      : Local WW volatility (log1p_concentration_4w_std).
                          When provided, scales the Z-score threshold up in
                          high-noise windows (cold-start-safe).

        Returns
        -------
        List of ClassifierRecord, one per observation.
        """
        baseline = self._baselines.get(str(county_id))
        if baseline is None:
            logger.debug(
                "OutbreakClassifier: no baseline for county {} (cold-start) — "
                "returning all-False records.",
                county_id,
            )
            return [
                ClassifierRecord(
                    unique_id=str(county_id),
                    date=idx,
                    triggered=False,
                    z_score=float("nan"),
                    momentum=float("nan"),
                    signal=float(v),
                )
                for idx, v in signal_series.items()
            ]

        std_safe = baseline.std + 1e-8
        records  = []

        for idx, raw_signal in signal_series.items():
            sig     = float(raw_signal)
            z_score = (sig - baseline.mean) / std_safe

            # ── Volatility-adjusted effective threshold ───────────────────
            # When local WW volatility exceeds the training-period mean, raise
            # the required Z-score to prevent false positives in noisy windows.
            if (
                vol_series is not None
                and idx in vol_series.index
                and self.volatility_scale > 0
            ):
                local_vol = float(vol_series[idx])
                vol_ratio = local_vol / baseline.mean_vol
                effective_z = self.z_threshold * (
                    1.0 + self.volatility_scale * max(vol_ratio - 1.0, 0.0)
                )
            else:
                effective_z = self.z_threshold

            if momentum_series is not None and idx in momentum_series.index:
                momentum = float(momentum_series[idx])
            else:
                momentum = float("nan")

            z_ok      = z_score >= effective_z
            m_ok      = np.isnan(momentum) or momentum >= self.momentum_threshold
            above_lod = sig >= self.min_signal
            triggered = bool(z_ok and m_ok and above_lod)

            records.append(ClassifierRecord(
                unique_id=str(county_id),
                date=pd.Timestamp(idx),
                triggered=triggered,
                z_score=z_score,
                momentum=momentum,
                signal=sig,
            ))
        return records

    # ------------------------------------------------------------------
    # Inference — full DataFrame
    # ------------------------------------------------------------------

    def classify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify every (county, date) row in a processed DataFrame.

        Parameters
        ----------
        df : Processed DataFrame with signal_col, momentum_col (optional),
             id_col, and date_col columns.

        Returns
        -------
        DataFrame with columns:
          unique_id, date, triggered (bool), z_score, momentum, signal
        One row per (county, date) observation.
        """
        has_momentum   = self.momentum_col in df.columns
        has_volatility = (
            self.volatility_col is not None and self.volatility_col in df.columns
        )
        all_records: list[ClassifierRecord] = []

        for county, grp in df.groupby(self.id_col):
            grp = grp.sort_values(self.date_col).set_index(self.date_col)
            signal_s   = grp[self.signal_col].astype(float)
            momentum_s = (
                grp[self.momentum_col].astype(float) if has_momentum else None
            )
            vol_s = (
                grp[self.volatility_col].astype(float)
                if has_volatility else None
            )
            records = self.classify_series(
                signal_s, momentum_s, county_id=county, vol_series=vol_s
            )
            all_records.extend(records)

        if not all_records:
            return pd.DataFrame(
                columns=["unique_id", "date", "triggered", "z_score", "momentum", "signal"]
            )

        return pd.DataFrame([
            {
                "unique_id": r.unique_id,
                "date":      r.date,
                "triggered": r.triggered,
                "z_score":   r.z_score,
                "momentum":  r.momentum,
                "signal":    r.signal,
            }
            for r in all_records
        ])

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def triggered_counties(self, clf_df: pd.DataFrame) -> list[str]:
        """Return sorted list of counties with at least one triggered observation."""
        return sorted(clf_df.loc[clf_df["triggered"], "unique_id"].unique().tolist())

    def suppressed_counties(self, clf_df: pd.DataFrame) -> list[str]:
        """Return sorted list of counties where no observation triggered."""
        return sorted(
            set(clf_df["unique_id"].unique()) - set(self.triggered_counties(clf_df))
        )

    def summary(self, clf_df: pd.DataFrame) -> str:
        """One-line human-readable classification summary."""
        n_total     = clf_df["unique_id"].nunique()
        n_triggered = len(self.triggered_counties(clf_df))
        n_rows      = len(clf_df)
        n_alerts    = int(clf_df["triggered"].sum())
        return (
            f"OutbreakClassifier: {n_alerts}/{n_rows} weeks triggered "
            f"across {n_triggered}/{n_total} counties"
        )
