"""
Tests for src/evaluation/evaluator.py  (Phase 6 evaluation pipeline)

What this suite validates
--------------------------
OnsetLabeler
  - Threshold is exactly the Nth percentile of the training signal
  - get_onset_events() back-dates the onset to the actual first crossing week
  - Quiet signal (all below threshold) produces zero onset events
  - Unfitted labeler raises RuntimeError on use
  - County not seen at fit time is silently skipped (not a crash)

Evaluator.score()
  - Returns populated ProbabilisticResult (WIS, coverage, MAE, pinball)
  - Returns det=None when alert_df is omitted
  - Returns DetectionResult with correct counts when alert_df is provided
  - Empty overlap between forecast and actuals returns null EvalReport

Data shape / leakage
  - OnsetLabeler.fit() only sees training data; threshold changes if training
    distribution changes (i.e., the threshold is truly computed from training)

Run with:
    pytest tests/test_evaluator.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import COUNTY_COL, NWSS_DATE_COL, TARGET_COL
from src.evaluation.evaluator import Evaluator, OnsetEvent, OnsetLabeler
from src.evaluation.metrics import QuantileColumns


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _weekly_df(
    county: str,
    start: str,
    values: list[float],
    signal_col: str = TARGET_COL,
    date_col:   str = NWSS_DATE_COL,
    id_col:     str = COUNTY_COL,
) -> pd.DataFrame:
    """Build a weekly W-WED signal DataFrame for one county."""
    dates = pd.date_range(start=start, periods=len(values), freq="W-WED")
    return pd.DataFrame({
        id_col:     [county] * len(values),
        date_col:   dates,
        signal_col: values,
    })


def _forecast_df_from_actual(
    actual_df: pd.DataFrame,
    offset: float = 0.0,
    id_col:   str = COUNTY_COL,
    date_col: str = NWSS_DATE_COL,
) -> pd.DataFrame:
    """Build a synthetic 7-quantile forecast DataFrame aligned to actual_df dates.

    All quantile columns are y_true + offset, making coverage deterministic.
    """
    df = actual_df.copy().rename(columns={id_col: "unique_id", date_col: "ds"})
    y = df[TARGET_COL].to_numpy(dtype=float)
    return pd.DataFrame({
        "unique_id":    df["unique_id"],
        "ds":           df["ds"],
        "TFT-lo-95.0":  y + offset - 2.0,
        "TFT-lo-80.0":  y + offset - 1.5,
        "TFT-lo-50.0":  y + offset - 0.5,
        "TFT-median":   y + offset,
        "TFT-hi-50.0":  y + offset + 0.5,
        "TFT-hi-80.0":  y + offset + 1.5,
        "TFT-hi-95.0":  y + offset + 2.0,
    })


# ===========================================================================
# OnsetLabeler — threshold computation
# ===========================================================================

class TestOnsetLabelerThreshold:

    def test_threshold_equals_percentile_of_training(self):
        """Threshold must be exactly np.percentile(train_signal, p)."""
        values = list(range(1, 21))    # 1 … 20
        train_df = _weekly_df("06075", "2021-01-06", values)
        labeler = OnsetLabeler(percentile=75).fit(train_df)
        expected = float(np.percentile(values, 75))
        assert labeler.thresholds["06075"] == pytest.approx(expected, rel=1e-9)

    def test_threshold_changes_with_training_distribution(self):
        """Threshold computed from a high-value training set is higher than from low."""
        low_train  = _weekly_df("06075", "2021-01-06", [1.0] * 20)
        high_train = _weekly_df("06075", "2021-01-06", [100.0] * 20)
        low_thresh  = OnsetLabeler(percentile=75).fit(low_train).thresholds["06075"]
        high_thresh = OnsetLabeler(percentile=75).fit(high_train).thresholds["06075"]
        assert high_thresh > low_thresh

    def test_unfitted_labeler_raises_on_predict(self):
        """Calling get_onset_events() before fit() raises RuntimeError."""
        labeler = OnsetLabeler()
        df = _weekly_df("06075", "2021-01-06", [5.0, 6.0, 7.0])
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.get_onset_events(df)

    def test_county_not_in_training_is_skipped(self):
        """County absent from fit returns zero events (not a crash)."""
        train_df = _weekly_df("06075", "2021-01-06", list(range(1, 21)))
        labeler  = OnsetLabeler(percentile=50).fit(train_df)
        # Evaluate on a county that wasn't in training
        test_df  = _weekly_df("99999", "2022-01-05", [100.0] * 10)
        events   = labeler.get_onset_events(test_df)
        assert events == []


# ===========================================================================
# OnsetLabeler — onset detection
# ===========================================================================

class TestOnsetLabelerDetection:

    def test_quiet_signal_has_no_onsets(self):
        """Signal always below threshold produces zero onset events."""
        train_df = _weekly_df("06075", "2021-01-06", [5.0] * 52)
        labeler  = OnsetLabeler(percentile=75).fit(train_df)
        # Threshold ≈ 5.0; test signal is well below
        test_df  = _weekly_df("06075", "2022-01-05", [1.0] * 20)
        assert labeler.get_onset_events(test_df) == []

    def test_sustained_high_signal_produces_onset(self):
        """Signal crossing and sustaining above threshold for ≥ sustained_weeks creates an event."""
        values = [1.0] * 10 + [10.0] * 10   # crosses high after 10 weeks
        train_df = _weekly_df("06075", "2021-01-06", values[:10])   # low training → low threshold
        labeler  = OnsetLabeler(percentile=75, sustained_weeks=2).fit(train_df)
        test_df  = _weekly_df("06075", "2022-01-05", values)
        events   = labeler.get_onset_events(test_df)
        assert len(events) >= 1

    def test_onset_is_backdated_to_first_crossing(self):
        """onset_date must be the first above-threshold week, not the confirmation week.

        With sustained_weeks=2:
          - Signal crosses threshold at week W.
          - Confirmation fires at week W+1 (after 2 consecutive above-threshold weeks).
          - onset_date should be W, not W+1.
        """
        # Train on quiet signal so threshold is low (~1.0)
        train_df = _weekly_df("06075", "2021-01-06", [1.0] * 20)
        labeler  = OnsetLabeler(percentile=50, sustained_weeks=2).fit(train_df)
        # threshold ≈ 1.0; test: first crossing at week 5 (index 4)
        vals  = [0.5] * 4 + [5.0, 5.0, 5.0]   # crosses at index 4
        test_df = _weekly_df("06075", "2022-01-05", vals)
        events  = labeler.get_onset_events(test_df)
        assert len(events) == 1
        crossing_date = pd.date_range("2022-01-05", periods=7, freq="W-WED")[4]
        assert events[0].onset_date == crossing_date, (
            f"Expected onset on {crossing_date.date()}, got {events[0].onset_date.date()}"
        )

    def test_onset_value_equals_signal_at_onset_date(self):
        """onset_value must be the signal at onset_date, not the confirmation date."""
        train_df  = _weekly_df("06075", "2021-01-06", [1.0] * 20)
        labeler   = OnsetLabeler(percentile=50, sustained_weeks=2).fit(train_df)
        # Crossing value = 8.0; confirmation value = 9.0
        vals      = [0.5] * 4 + [8.0, 9.0, 9.0]
        test_df   = _weekly_df("06075", "2022-01-05", vals)
        events    = labeler.get_onset_events(test_df)
        assert len(events) == 1
        assert events[0].onset_value == pytest.approx(8.0, abs=0.01)


# ===========================================================================
# Evaluator.score() — probabilistic path
# ===========================================================================

class TestEvaluatorProbabilistic:

    @pytest.fixture()
    def actual_df(self):
        return _weekly_df("06075", "2023-01-04", [3.0, 3.5, 4.0, 4.5, 5.0])

    @pytest.fixture()
    def perfect_forecast_df(self, actual_df):
        """Forecast where median = y_true and PI perfectly brackets it."""
        return _forecast_df_from_actual(actual_df, offset=0.0)

    @pytest.fixture()
    def labeler(self, actual_df):
        """Labeler fitted on the actual data (to simplify fixture setup)."""
        return OnsetLabeler(percentile=75).fit(actual_df)

    def test_score_returns_eval_report(self, actual_df, perfect_forecast_df, labeler):
        ev = Evaluator(labeler=labeler).score(actual_df, perfect_forecast_df)
        from src.evaluation.metrics import EvalReport
        assert isinstance(ev, EvalReport)

    def test_score_populates_prob(self, actual_df, perfect_forecast_df, labeler):
        ev = Evaluator(labeler=labeler).score(actual_df, perfect_forecast_df)
        assert ev.prob.n_observations == len(actual_df)
        assert ev.prob.n_series == 1
        assert ev.prob.coverage_95 == pytest.approx(1.0)   # all inside ±2.0
        assert ev.prob.coverage_50 == pytest.approx(1.0)   # all inside ±0.5
        assert ev.prob.mae == pytest.approx(0.0, abs=1e-6)  # perfect median

    def test_score_populates_pinball_keys(self, actual_df, perfect_forecast_df, labeler):
        ev = Evaluator(labeler=labeler).score(actual_df, perfect_forecast_df)
        expected_keys = {"q0.025", "q0.10", "q0.25", "q0.50", "q0.75", "q0.90", "q0.975"}
        assert set(ev.prob.pinball_by_quantile.keys()) == expected_keys

    def test_score_det_is_none_without_alert_df(self, actual_df, perfect_forecast_df, labeler):
        """Detection metrics are None when no OutbreakClassifier output is supplied."""
        ev = Evaluator(labeler=labeler).score(actual_df, perfect_forecast_df)
        assert ev.det is None

    def test_null_report_on_empty_overlap(self, actual_df, labeler):
        """Forecast covering dates entirely outside actual_df → null EvalReport."""
        forecast_in_future = _forecast_df_from_actual(
            _weekly_df("06075", "2030-01-01", [5.0] * 5)
        )
        ev = Evaluator(labeler=labeler).score(actual_df, forecast_in_future)
        assert ev.prob.n_observations == 0
        assert np.isnan(ev.prob.mean_wis)


# ===========================================================================
# Evaluator.score() — detection path
# ===========================================================================

class TestEvaluatorDetection:

    def _make_alert_df(self, county: str, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "unique_id": [county] * len(dates),
            "ds":        pd.to_datetime(dates),
            "alert":     [True] * len(dates),
        })

    def test_no_alerts_no_onsets_produces_nan_det(self):
        """When no alerts and no onsets: TP=FP=FN=0, precision/recall/f1=NaN."""
        actual   = _weekly_df("06075", "2023-01-04", [1.0] * 10)
        forecast = _forecast_df_from_actual(actual)
        train    = _weekly_df("06075", "2021-01-05", [5.0] * 52)  # high training → high threshold
        labeler  = OnsetLabeler(percentile=90).fit(train)          # actual below threshold → 0 onsets
        alert_df = self._make_alert_df("06075", [])                # no alerts

        ev = Evaluator(labeler=labeler).score(actual, forecast, alert_df=alert_df)
        assert ev.det is not None
        assert ev.det.tp == 0
        assert ev.det.fp == 0
        assert ev.det.fn == 0
        assert np.isnan(ev.det.precision)

    def test_correct_tp_fp_fn_counts(self):
        """One onset + one matching alert = 1 TP, 0 FP, 0 FN."""
        # Training: low signal → low threshold
        train    = _weekly_df("06075", "2021-01-05", [0.5] * 52)
        labeler  = OnsetLabeler(percentile=50, sustained_weeks=2).fit(train)

        # Test signal: crosses threshold at week 6 (index 5)
        vals     = [0.3] * 5 + [5.0, 5.0, 5.0] + [0.3] * 5
        actual   = _weekly_df("06075", "2022-01-05", vals)
        forecast = _forecast_df_from_actual(actual)

        # Place alert 7 days before the first above-threshold week
        dates = pd.date_range("2022-01-05", periods=len(vals), freq="W-WED")
        onset_date = dates[5]   # first crossing
        alert_date = str((onset_date - pd.Timedelta(days=7)).date())

        alert_df = self._make_alert_df("06075", [alert_date])
        ev = Evaluator(labeler=labeler).score(actual, forecast, alert_df=alert_df)

        assert ev.det is not None
        assert ev.det.tp == 1
        assert ev.det.fp == 0
        assert ev.det.fn == 0
        assert ev.det.precision == pytest.approx(1.0)
        assert ev.det.recall    == pytest.approx(1.0)

    def test_to_dict_includes_pinball_columns(self):
        """EvalReport.to_dict() serialises per-quantile pinball as flat columns."""
        actual   = _weekly_df("06075", "2023-01-04", [3.0, 4.0, 5.0, 6.0])
        forecast = _forecast_df_from_actual(actual)
        labeler  = OnsetLabeler(percentile=75).fit(actual)
        ev       = Evaluator(labeler=labeler).score(actual, forecast)
        d        = ev.to_dict()

        expected_pinball_keys = {
            "pinball_q0025", "pinball_q010", "pinball_q025",
            "pinball_q050",  "pinball_q075", "pinball_q090", "pinball_q0975",
        }
        assert expected_pinball_keys.issubset(d.keys()), (
            f"Missing pinball columns in to_dict(): "
            f"{expected_pinball_keys - set(d.keys())}"
        )
