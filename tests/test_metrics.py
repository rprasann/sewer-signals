"""
Tests for src/evaluation/metrics.py  (Phase 6 evaluation engine)

What this suite validates
--------------------------
Category 1 — Probabilistic:
  wis()           Analytic correctness; perfect-forecast floor; ordering
  pinball_loss()  Known analytic values; asymmetry direction; all-7-quantile shape
  coverage()      100% and 0% edge cases; partial calibration boundary
  mae()           Known values; relation to pinball at q=0.5

Category 2 — Event-based:
  match_alerts_to_onsets()  TP/FP/FN counting; TTD sign; pre/post window
  detection_score()         Precision/Recall/F1 arithmetic; NaN edge cases

QuantileColumns  auto_detect happy path and missing-column error

Run with:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    EvalReport,
    QuantileColumns,
    coverage,
    detection_score,
    mae,
    match_alerts_to_onsets,
    pinball_loss,
    wis,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def q_cols():
    """QuantileColumns using short synthetic column names."""
    return QuantileColumns(
        q025="lo95", q10=None, q25="lo50", q50="med", q75="hi50", q90=None, q975="hi95"
    )


def _forecast_df(y_true, *, lo95, lo50, med, hi50, hi95):
    """Build a minimal forecast DataFrame from scalar offsets applied to y_true."""
    y = np.asarray(y_true, dtype=float)
    return pd.DataFrame({
        "lo95": y + lo95,
        "lo50": y + lo50,
        "med":  y + med,
        "hi50": y + hi50,
        "hi95": y + hi95,
    })


# ===========================================================================
# wis()
# ===========================================================================

class TestWIS:

    def test_perfect_forecast_has_minimal_wis(self, q_cols):
        """When median = y_true and PI perfectly contains y_true, WIS is very small."""
        y_true = np.array([1.0, 2.0, 3.0])
        # PI exactly brackets y_true; median is exact
        df = _forecast_df(y_true, lo95=-1.0, lo50=-0.1, med=0.0, hi50=0.1, hi95=1.0)
        score = wis(y_true, df, q_cols)
        assert score.shape == (3,)
        assert float(score.mean()) < 0.5, "Perfect forecast should score near 0"

    def test_wide_miss_has_large_wis(self, q_cols):
        """Predictions far from y_true should score much higher than a good forecast."""
        y_true = np.array([5.0, 5.0, 5.0])
        good = _forecast_df(y_true, lo95=-1.0, lo50=-0.1, med=0.0, hi50=0.1, hi95=1.0)
        bad  = _forecast_df(y_true, lo95=-1.0, lo50=-0.1, med=5.0, hi50=5.1, hi95=6.0)
        good_score = float(wis(y_true, good, q_cols).mean())
        bad_score  = float(wis(y_true, bad, q_cols).mean())
        assert bad_score > good_score * 3, "Bad forecast should score significantly higher"

    def test_wis_is_non_negative(self, q_cols):
        """WIS must be ≥ 0 for any input."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 20)
        df = _forecast_df(y_true, lo95=-2.0, lo50=-0.5, med=0.1, hi50=0.5, hi95=2.0)
        scores = wis(y_true, df, q_cols)
        assert (scores >= 0).all(), "WIS must be non-negative"

    def test_wis_returns_per_observation_array(self, q_cols):
        """wis() returns one score per row, not a scalar."""
        n = 10
        y_true = np.ones(n)
        df = _forecast_df(y_true, lo95=-1.0, lo50=-0.1, med=0.0, hi50=0.1, hi95=1.0)
        scores = wis(y_true, df, q_cols)
        assert scores.shape == (n,)

    def test_analytic_wis_undershoot(self, q_cols):
        """When y_true is below all quantiles, overshoot penalty dominates.

        For K=2 intervals, WIS = norm × (0.5×AE + (α1/2)×IS95 + (α2/2)×IS50).
        y_true = 0, all quantile predictions > 0 → undershoot on every interval.
        """
        y_true = np.array([0.0])
        df = pd.DataFrame({
            "lo95": [2.0], "lo50": [3.0], "med": [4.0], "hi50": [5.0], "hi95": [6.0]
        })
        score = float(wis(y_true, df, q_cols)[0])
        # Manual: AE=4, IS95=(6-2)+(2/0.05)*(2-0)=4+80=84; IS50=(5-3)+(2/0.5)*(3-0)=2+12=14
        # WIS = (1/2.5) * (0.5*4 + (0.025)*84 + (0.25)*14) = 0.4 * (2 + 2.1 + 3.5) = 0.4*7.6 = 3.04
        assert pytest.approx(score, abs=0.01) == 3.04


# ===========================================================================
# pinball_loss()
# ===========================================================================

class TestPinballLoss:

    def test_returns_dict_with_expected_keys(self, q_cols):
        """Result should have one key per non-None quantile column."""
        y_true = np.array([1.0, 2.0, 3.0])
        df = _forecast_df(y_true, lo95=-1.0, lo50=-0.1, med=0.0, hi50=0.1, hi95=1.0)
        result = pinball_loss(y_true, df, q_cols)
        # q_cols has q10=None, q90=None → 5 keys
        assert set(result.keys()) == {"q0.025", "q0.25", "q0.50", "q0.75", "q0.975"}

    def test_median_pinball_equals_half_mae(self, q_cols):
        """Pinball at q=0.5 is exactly 0.5 × MAE (symmetric penalty)."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        df = _forecast_df(y_true, lo95=-2.0, lo50=-0.5, med=0.5, hi50=1.0, hi95=2.0)
        result = pinball_loss(y_true, df, q_cols)
        mae_val = float(np.mean(np.abs(y_true - df["med"].to_numpy())))
        assert pytest.approx(result["q0.50"], rel=1e-6) == mae_val / 2

    def test_low_quantile_penalises_overprediction_more(self, q_cols):
        """For q=0.025, overprediction penalty is (1-0.025) = 0.975× the error;
        underprediction is only 0.025× the error.
        """
        # y_true = 0; lo95 (q=0.025) prediction = +1 → overprediction of 1 unit
        df_over  = pd.DataFrame({"lo95": [1.0], "lo50": [0.0], "med": [0.0], "hi50": [0.0], "hi95": [0.0]})
        # y_true = 1; lo95 prediction = 0 → underprediction of 1 unit
        df_under = pd.DataFrame({"lo95": [0.0], "lo50": [0.0], "med": [0.0], "hi50": [0.0], "hi95": [0.0]})

        y_over  = np.array([0.0])
        y_under = np.array([1.0])

        pb_over  = pinball_loss(y_over,  df_over,  q_cols)["q0.025"]
        pb_under = pinball_loss(y_under, df_under, q_cols)["q0.025"]

        # Overprediction at q=0.025: (1-0.025)*1 = 0.975
        # Underprediction at q=0.025: 0.025*1 = 0.025
        assert pytest.approx(pb_over,  abs=1e-6) == 0.975
        assert pytest.approx(pb_under, abs=1e-6) == 0.025

    def test_high_quantile_penalises_underprediction_more(self, q_cols):
        """For q=0.975, underprediction penalty is (1-0.975) mapped through formula."""
        # hi95 prediction = 0, y_true = 1 → underprediction (y > yhat)
        df_under = pd.DataFrame({"lo95": [0.0], "lo50": [0.0], "med": [0.0], "hi50": [0.0], "hi95": [0.0]})
        y = np.array([1.0])
        pb = pinball_loss(y, df_under, q_cols)["q0.975"]
        # q=0.975, y>yhat: 0.975 * 1 = 0.975
        assert pytest.approx(pb, abs=1e-6) == 0.975

    def test_zero_loss_when_quantile_exactly_equals_y_true(self, q_cols):
        """When predicted quantile == y_true, pinball loss for that quantile is 0."""
        y_true = np.array([3.0, 3.0, 3.0])
        df = _forecast_df(y_true, lo95=-1.0, lo50=-0.5, med=0.0, hi50=0.5, hi95=1.0)
        result = pinball_loss(y_true, df, q_cols)
        # med = y_true + 0 = y_true → q0.50 pinball = 0
        assert pytest.approx(result["q0.50"], abs=1e-9) == 0.0


# ===========================================================================
# coverage()
# ===========================================================================

class TestCoverage:

    def test_full_coverage_when_pi_always_contains_y(self, q_cols):
        """100% coverage when y_true is always inside [lo95, hi95] and [lo50, hi50]."""
        y_true = np.array([0.0, 1.0, 2.0])
        df = _forecast_df(y_true, lo95=-10.0, lo50=-10.0, med=0.0, hi50=10.0, hi95=10.0)
        result = coverage(y_true, df, q_cols)
        assert result["coverage_50"] == 1.0
        assert result["coverage_95"] == 1.0

    def test_zero_coverage_when_y_always_outside_pi(self, q_cols):
        """0% coverage when y_true is always above all predicted intervals.

        Use absolute column values (not y_true + offset) so the PI is fixed well
        below y_true regardless of what y_true is.
        """
        y_true = np.array([100.0, 200.0, 300.0])
        # Fixed low PI: intervals span [-2, 1] — far below y_true
        df = pd.DataFrame({
            "lo95": [-2.0, -2.0, -2.0],
            "lo50": [-1.0, -1.0, -1.0],
            "med":  [ 0.0,  0.0,  0.0],
            "hi50": [ 0.5,  0.5,  0.5],
            "hi95": [ 1.0,  1.0,  1.0],
        })
        result = coverage(y_true, df, q_cols)
        assert result["coverage_50"] == 0.0
        assert result["coverage_95"] == 0.0

    def test_partial_coverage_is_fractional(self, q_cols):
        """4 out of 5 observations inside → 80% coverage."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        df = _forecast_df(y_true[:-1].tolist() + [0.0],
                         lo95=-5.0, lo50=-2.0, med=0.0, hi50=2.0, hi95=5.0)
        # Rebuild properly: 4 inside + 1 outside the outer PI
        df = pd.DataFrame({
            "lo95": [0.0] * 4 + [-5.0],
            "lo50": [0.5] * 4 + [-2.0],
            "med":  [1.0] * 4 + [0.0],
            "hi50": [1.5] * 4 + [2.0],
            "hi95": [2.0] * 4 + [5.0],
        })
        result = coverage(y_true, df, q_cols)
        assert result["coverage_95"] == pytest.approx(4 / 5, abs=1e-9)

    def test_coverage_50_leq_coverage_95(self, q_cols):
        """Coverage of the 50% PI is always ≤ coverage of the wider 95% PI."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 50)
        df = _forecast_df(y_true, lo95=-3.0, lo50=-0.7, med=0.0, hi50=0.7, hi95=3.0)
        result = coverage(y_true, df, q_cols)
        assert result["coverage_50"] <= result["coverage_95"]


# ===========================================================================
# mae()
# ===========================================================================

class TestMAE:

    def test_known_value(self):
        """MAE([0,1,2,3], [1,1,1,1]) = mean(|−1,0,1,2|) = 1.0."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        assert pytest.approx(mae(y_true, y_pred), abs=1e-9) == 1.0

    def test_perfect_prediction_is_zero(self):
        y_true = np.array([1.5, 2.5, 3.5])
        assert pytest.approx(mae(y_true, y_true), abs=1e-9) == 0.0

    def test_mae_is_symmetric(self):
        """MAE(y, ŷ) == MAE(ŷ, y) — absolute error is symmetric."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 1.0])
        assert pytest.approx(mae(y_true, y_pred)) == mae(y_pred, y_true)


# ===========================================================================
# match_alerts_to_onsets()
# ===========================================================================

class TestMatchAlertsToOnsets:

    def _make_onset(self, county, onset_date):
        """Create a minimal OnsetEvent-like object."""
        from src.evaluation.evaluator import OnsetEvent
        return OnsetEvent(
            county=county,
            onset_date=pd.Timestamp(onset_date),
            onset_value=5.0,
            confirmation_date=pd.Timestamp(onset_date) + pd.Timedelta(weeks=1),
        )

    def test_perfect_match_all_tp(self):
        """One alert per onset, fired 7 days before each — all TP, TTD = 7."""
        onsets = [
            self._make_onset("06075", "2023-01-11"),
            self._make_onset("06085", "2023-02-01"),
        ]
        # Alerts fire 7 days before each onset
        alerts = [
            ("06075", pd.Timestamp("2023-01-04")),
            ("06085", pd.Timestamp("2023-01-25")),
        ]
        tp, fp, fn, ttd = match_alerts_to_onsets(alerts, onsets, pre_window_weeks=4)
        assert tp == 2
        assert fp == 0
        assert fn == 0
        assert all(t == 7.0 for t in ttd), f"Expected TTD=7 days each, got {ttd}"

    def test_no_onsets_all_alerts_are_fp(self):
        """With no onsets, every alert is a false positive."""
        alerts = [
            ("06075", pd.Timestamp("2023-01-04")),
            ("06075", pd.Timestamp("2023-03-01")),
        ]
        tp, fp, fn, ttd = match_alerts_to_onsets(alerts, [], pre_window_weeks=4)
        assert tp == 0
        assert fp == 2
        assert fn == 0
        assert ttd == []

    def test_no_alerts_all_onsets_are_fn(self):
        """With no alerts, every onset is a false negative."""
        onsets = [self._make_onset("06075", "2023-01-11")]
        tp, fp, fn, ttd = match_alerts_to_onsets([], onsets, pre_window_weeks=4)
        assert tp == 0
        assert fp == 0
        assert fn == 1

    def test_alert_outside_window_is_fp(self):
        """Alert fired 60 days before onset (beyond pre_window_weeks=4) → FP."""
        onsets = [self._make_onset("06075", "2023-03-01")]
        alerts = [("06075", pd.Timestamp("2023-01-01"))]  # 59 days early
        tp, fp, fn, ttd = match_alerts_to_onsets(alerts, onsets, pre_window_weeks=4)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_positive_ttd_means_early_alert(self):
        """Positive TTD = alert fired before onset (WW early warning)."""
        onsets = [self._make_onset("06075", "2023-01-18")]
        alerts = [("06075", pd.Timestamp("2023-01-04"))]   # 14 days early
        tp, fp, fn, ttd = match_alerts_to_onsets(alerts, onsets, pre_window_weeks=4)
        assert tp == 1
        assert ttd[0] == pytest.approx(14.0, abs=0.01)

    def test_negative_ttd_means_late_alert(self):
        """Negative TTD = alert fired after onset (still credited within post_window)."""
        onsets = [self._make_onset("06075", "2023-01-04")]
        alerts = [("06075", pd.Timestamp("2023-01-11"))]   # 7 days late
        tp, fp, fn, ttd = match_alerts_to_onsets(
            alerts, onsets, pre_window_weeks=4, post_window_weeks=2
        )
        assert tp == 1
        assert ttd[0] == pytest.approx(-7.0, abs=0.01)

    def test_each_alert_matches_at_most_one_onset(self):
        """One alert cannot be credited to two onsets."""
        onsets = [
            self._make_onset("06075", "2023-01-11"),
            self._make_onset("06075", "2023-01-18"),
        ]
        # One alert fired between both onsets — it should match the earlier one
        alerts = [("06075", pd.Timestamp("2023-01-07"))]
        tp, fp, fn, ttd = match_alerts_to_onsets(alerts, onsets, pre_window_weeks=4)
        assert tp == 1
        assert fn == 1   # second onset unmatched


# ===========================================================================
# detection_score()
# ===========================================================================

class TestDetectionScore:

    def test_perfect_detection(self):
        """All onsets caught, no FP → precision=recall=F1=1.0."""
        det = detection_score(tp=3, fp=0, fn=0, ttd_days=[7.0, 14.0, 3.0])
        assert det.precision == pytest.approx(1.0)
        assert det.recall    == pytest.approx(1.0)
        assert det.f1        == pytest.approx(1.0)

    def test_no_detections_f1_is_nan(self):
        """TP=0, FP=0, FN=3 → precision=NaN, recall=0, F1=NaN."""
        det = detection_score(tp=0, fp=0, fn=3, ttd_days=[])
        assert np.isnan(det.precision)
        assert det.recall == pytest.approx(0.0)
        assert np.isnan(det.f1)

    def test_f1_arithmetic(self):
        """F1 = 2PR/(P+R) where P=2/3, R=2/3 → F1=2/3."""
        det = detection_score(tp=2, fp=1, fn=1, ttd_days=[5.0, 10.0])
        p = 2 / 3
        r = 2 / 3
        expected_f1 = 2 * p * r / (p + r)
        assert det.precision == pytest.approx(p, rel=1e-6)
        assert det.recall    == pytest.approx(r, rel=1e-6)
        assert det.f1        == pytest.approx(expected_f1, rel=1e-6)

    def test_mean_ttd_is_average_of_list(self):
        det = detection_score(tp=3, fp=0, fn=0, ttd_days=[4.0, 8.0, 12.0])
        assert det.mean_ttd_days == pytest.approx(8.0, abs=1e-6)

    def test_zero_tp_zero_fp_precision_is_nan(self):
        """When no alerts are issued, precision is undefined (NaN), not 0."""
        det = detection_score(tp=0, fp=0, fn=2, ttd_days=[])
        assert np.isnan(det.precision)


# ===========================================================================
# QuantileColumns — auto_detect
# ===========================================================================

class TestQuantileColumns:

    def test_auto_detect_standard_tft_columns(self):
        """auto_detect recognises the standard TFT output column suffixes."""
        df = pd.DataFrame({
            "TFT-lo-95.0": [1.0],
            "TFT-lo-80.0": [2.0],
            "TFT-lo-50.0": [3.0],
            "TFT-median":  [4.0],
            "TFT-hi-50.0": [5.0],
            "TFT-hi-80.0": [6.0],
            "TFT-hi-95.0": [7.0],
        })
        qc = QuantileColumns.auto_detect(df)
        assert qc.q025  == "TFT-lo-95.0"
        assert qc.q50   == "TFT-median"
        assert qc.q975  == "TFT-hi-95.0"
        assert qc.q10   == "TFT-lo-80.0"
        assert qc.q90   == "TFT-hi-80.0"

    def test_auto_detect_raises_on_missing_required_column(self):
        """Missing a required column (e.g. -median) raises ValueError."""
        df = pd.DataFrame({"TFT-lo-95.0": [1.0], "TFT-lo-50.0": [2.0]})
        with pytest.raises(ValueError, match="-median"):
            QuantileColumns.auto_detect(df)

    def test_auto_detect_optional_columns_return_none(self):
        """q10/q90 are None when 80% PI columns are absent."""
        df = pd.DataFrame({
            "TFT-lo-95.0": [1.0],
            "TFT-lo-50.0": [2.0],
            "TFT-median":  [3.0],
            "TFT-hi-50.0": [4.0],
            "TFT-hi-95.0": [5.0],
        })
        qc = QuantileColumns.auto_detect(df)
        assert qc.q10 is None
        assert qc.q90 is None
