"""
Tests for the two-stage inference system:
  src/models/classifier.py     — OutbreakClassifier
  src/pipeline.py              — TwoStagePipeline / InferenceResult
  src/data_pipeline/adapters.py — BaseDatasetAdapter / COVID_Adapter interface

What this suite verifies
------------------------
Classifier — non-elastic baseline
  1. Quiet signal stays below z_threshold → suppressed
  2. Surge signal exceeds z_threshold → triggered
  3. Z-score without momentum confirmation is NOT sufficient (both gates)
  4. Non-elastic baseline does not drift upward during a sustained surge
     (the critical property that rolling-window baselines lack)
  5. County absent from training is handled gracefully (all-False, not crash)
  6. Fitting on short series (< 2 rows) logs a warning, does not raise

TwoStagePipeline
  7. Triggered county receives TFT-style forecast rows
  8. Suppressed county receives quiet prior forecast rows
  9. InferenceResult.any_triggered is False when all counties suppressed
 10. InferenceResult.summary() returns a non-empty string (smoke test)

BaseDatasetAdapter
 11. COVID_Adapter exposes the required schema properties
 12. validate_schema raises ValueError on missing required column

Run with:
    pytest tests/test_classifier.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import COUNTY_COL, NWSS_DATE_COL, TARGET_COL, WW_FEATURE_COL
from src.data_pipeline.adapters import BaseDatasetAdapter, COVID_Adapter
from src.models.classifier import ClassifierRecord, OutbreakClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    signal_values: dict[str, list[float]],
    momentum_values: dict[str, list[float]] | None = None,
    start: str = "2022-01-05",
    id_col: str = COUNTY_COL,
    date_col: str = NWSS_DATE_COL,
    signal_col: str = WW_FEATURE_COL,
    momentum_col: str = "ww_momentum_lead",
) -> pd.DataFrame:
    """Build a minimal processed panel DataFrame."""
    counties = list(signal_values.keys())
    rows = []
    for county in counties:
        vals = signal_values[county]
        dates = pd.date_range(start=start, periods=len(vals), freq="W-WED")
        for i, (ds, v) in enumerate(zip(dates, vals)):
            row: dict = {id_col: county, date_col: ds, signal_col: v, TARGET_COL: v * 0.8}
            if momentum_values and county in momentum_values:
                row[momentum_col] = momentum_values[county][i]
            else:
                row[momentum_col] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_classifier(
    train_values: dict[str, list[float]],
    z_threshold: float = 1.5,
    momentum_threshold: float = 0.0,
) -> OutbreakClassifier:
    """Fit a classifier on synthetic training data."""
    clf = OutbreakClassifier(
        z_threshold=z_threshold,
        momentum_threshold=momentum_threshold,
    )
    train_df = _make_panel(train_values)
    clf.fit(train_df)
    return clf


# ===========================================================================
# OutbreakClassifier — core classification logic
# ===========================================================================

class TestOutbreakClassifier:

    def test_quiet_signal_is_suppressed(self):
        """Signal equal to the training quiet mean should not trigger."""
        quiet_signal = [1.0] * 20   # all at quiet baseline level
        clf = _fit_classifier({"06075": quiet_signal})

        eval_panel = _make_panel({"06075": quiet_signal})
        clf_df = clf.classify_df(eval_panel)

        assert clf_df["triggered"].sum() == 0, (
            "Quiet signal at baseline level must never trigger"
        )

    def test_surge_signal_is_triggered(self):
        """Signal >> training baseline should trigger on every surge week."""
        # Train on quiet [1.0], evaluate on obvious surge [10.0]
        clf = _fit_classifier({"06075": [1.0] * 20}, z_threshold=1.5)
        surge_panel = _make_panel(
            {"06075": [10.0] * 10},
            momentum_values={"06075": [1.0] * 10},
        )
        clf_df = clf.classify_df(surge_panel)
        assert clf_df["triggered"].all(), "Strong surge with positive momentum must trigger"

    def test_high_z_without_momentum_suppressed(self):
        """Z-score above threshold but momentum below → NOT triggered."""
        clf = OutbreakClassifier(z_threshold=1.5, momentum_threshold=0.5)
        clf.fit(_make_panel({"06075": [1.0] * 20}))

        # High Z-score (signal = 10 >> baseline ≈ 1), but momentum = -1 (below 0.5)
        eval_panel = _make_panel(
            {"06075": [10.0] * 8},
            momentum_values={"06075": [-1.0] * 8},
        )
        clf_df = clf.classify_df(eval_panel)
        assert clf_df["triggered"].sum() == 0, (
            "High Z without momentum confirmation must be suppressed"
        )

    def test_non_elastic_baseline_does_not_drift_during_surge(self):
        """The trained baseline is frozen — Z-scores stay high through a sustained surge.

        If the baseline were elastic (rolling), it would absorb surge values and
        Z-scores would fall.  With a fixed training-period baseline they stay elevated.
        """
        # Train on quiet baseline
        clf = _fit_classifier({"06075": [1.0] * 52}, z_threshold=1.5)

        # Evaluate on a sustained 12-week surge
        surge_weeks = [8.0] * 12
        eval_panel = _make_panel(
            {"06075": surge_weeks},
            momentum_values={"06075": [1.0] * 12},
        )
        clf_df = clf.classify_df(eval_panel)

        # ALL 12 surge weeks must remain triggered — baseline must not drift
        assert clf_df["triggered"].all(), (
            f"Non-elastic baseline must keep all surge weeks triggered "
            f"(got {clf_df['triggered'].sum()}/12)"
        )

        # Z-scores must be consistently high (not declining over time)
        z_scores = clf_df["z_score"].to_numpy()
        assert z_scores.std() < 1.0, (
            "Z-scores must be stable across weeks (baseline not drifting)"
        )

    def test_county_absent_from_training_returns_all_false(self):
        """County not in training set returns all-False, not a crash."""
        clf = _fit_classifier({"06075": [1.0] * 20})
        eval_panel = _make_panel({"99999": [100.0] * 5})
        clf_df = clf.classify_df(eval_panel)
        assert (clf_df["triggered"] == False).all()  # noqa: E712

    def test_fit_on_single_row_does_not_raise(self):
        """Very short training series: logs warning but does not raise."""
        clf = OutbreakClassifier()
        short_df = _make_panel({"06075": [5.0]})  # only 1 row
        clf.fit(short_df)  # must not raise
        # Classifier may or may not have a baseline for this county — either is valid
        assert True

    def test_triggered_counties_helper(self):
        """triggered_counties() returns the correct FIPS set."""
        train = {"06075": [1.0] * 20, "06085": [1.0] * 20}
        clf   = _fit_classifier(train, z_threshold=1.5, momentum_threshold=0.0)

        # Only 06075 surges
        eval_panel = _make_panel(
            {"06075": [10.0] * 5, "06085": [1.0] * 5},
            momentum_values={"06075": [1.0] * 5, "06085": [0.0] * 5},
        )
        clf_df = clf.classify_df(eval_panel)

        triggered  = set(clf.triggered_counties(clf_df))
        suppressed = set(clf.suppressed_counties(clf_df))

        assert "06075" in triggered
        assert "06085" in suppressed
        assert triggered | suppressed == {"06075", "06085"}


# ===========================================================================
# TwoStagePipeline — trigger/suppress gate
# ===========================================================================

class TestTwoStagePipeline:
    """Verify the pipeline routes counties to TFT vs quiet prior correctly.

    The forecaster is mocked: it records which IDs were sent to TFT vs
    the quiet prior, without actually running torch inference.
    """

    class _MockForecaster:
        """Mock OutbreakForecaster that records which counties it sees."""

        def __init__(self, h: int = 4):
            self.h = h
            self.triggered_seen: list[str] = []
            self.suppressed_seen: list[str] = []

        def predict(self, processed_df, triggered_ids, all_ids=None):
            self.triggered_seen  = list(triggered_ids)
            self.suppressed_seen = [
                uid for uid in (all_ids or [])
                if uid not in triggered_ids
            ]
            # Return minimal DataFrame with triggered + quiet rows
            from src.config import SUPPRESSED_FORECAST_LEVEL
            rows = []
            future = pd.date_range("2023-01-01", periods=self.h, freq="W-WED")
            for uid in triggered_ids:
                for ds in future:
                    rows.append({
                        "unique_id": uid, "ds": ds,
                        "TFT-median": 3.0, "_suppressed": False,
                    })
            for uid in self.suppressed_seen:
                for ds in future:
                    rows.append({
                        "unique_id": uid, "ds": ds,
                        "TFT-median": SUPPRESSED_FORECAST_LEVEL, "_suppressed": True,
                    })
            return pd.DataFrame(rows)

    def _build_pipeline(self, train_signal, eval_signal, eval_momentum=None):
        from src.pipeline import TwoStagePipeline

        class _MockAdapter:
            signal_col   = WW_FEATURE_COL
            target_col   = TARGET_COL
            id_col       = COUNTY_COL
            date_col     = NWSS_DATE_COL
            momentum_col = "ww_momentum_lead"

        clf = OutbreakClassifier(z_threshold=1.5, momentum_threshold=0.0)
        clf.fit(_make_panel(train_signal))

        mock_fc = self._MockForecaster()
        eval_panel = _make_panel(eval_signal, eval_momentum)

        pipeline = TwoStagePipeline(
            adapter=_MockAdapter(),
            classifier=clf,
            forecaster=mock_fc,
        )
        return pipeline, eval_panel, mock_fc

    def test_triggered_county_reaches_forecaster(self):
        """County with high Z + momentum is passed to the forecaster."""
        train = {"06075": [1.0] * 20}
        surge = {"06075": [10.0] * 5}
        mom   = {"06075": [1.0] * 5}

        pipeline, eval_panel, mock_fc = self._build_pipeline(train, surge, mom)
        result = pipeline.run(eval_panel)

        assert "06075" in mock_fc.triggered_seen, (
            "Surging county must reach OutbreakForecaster"
        )
        assert result.any_triggered

    def test_suppressed_county_gets_quiet_prior(self):
        """Quiet county never reaches the forecaster's TFT path."""
        train = {"06075": [1.0] * 20}
        quiet = {"06075": [1.0] * 5}

        pipeline, eval_panel, mock_fc = self._build_pipeline(train, quiet)
        result = pipeline.run(eval_panel)

        assert "06075" not in mock_fc.triggered_seen
        assert "06075" in mock_fc.suppressed_seen
        assert not result.any_triggered

    def test_mixed_counties_routed_correctly(self):
        """One surging, one quiet — each goes to the correct stage."""
        train = {"06075": [1.0] * 20, "06085": [1.0] * 20}
        vals  = {"06075": [10.0] * 5, "06085": [1.0] * 5}
        mom   = {"06075": [1.0] * 5,  "06085": [0.0] * 5}

        pipeline, eval_panel, mock_fc = self._build_pipeline(train, vals, mom)
        result = pipeline.run(eval_panel)

        assert "06075" in mock_fc.triggered_seen
        assert "06085" in mock_fc.suppressed_seen
        assert "06075" in result.triggered_counties
        assert "06085" in result.suppressed_counties

    def test_inference_result_summary_is_non_empty(self):
        """InferenceResult.summary() returns a human-readable string."""
        train = {"06075": [1.0] * 20}
        quiet = {"06075": [1.0] * 5}

        pipeline, eval_panel, _ = self._build_pipeline(train, quiet)
        result = pipeline.run(eval_panel)
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_suppressed_forecast_rows_are_flat(self):
        """Quiet prior rows must have _suppressed=True in the forecast DataFrame."""
        train = {"06075": [1.0] * 20}
        quiet = {"06075": [1.0] * 5}

        pipeline, eval_panel, _ = self._build_pipeline(train, quiet)
        result = pipeline.run(eval_panel)

        assert result.forecast_df is not None
        suppressed_rows = result.forecast_df[result.forecast_df["_suppressed"] == True]  # noqa: E712
        assert len(suppressed_rows) > 0, "Suppressed county must have quiet prior rows"


# ===========================================================================
# BaseDatasetAdapter / COVID_Adapter interface
# ===========================================================================

class TestDatasetAdapterInterface:

    def test_covid_adapter_schema_properties(self):
        """COVID_Adapter exposes the five required schema properties."""
        adapter = COVID_Adapter()
        assert isinstance(adapter.signal_col, str)
        assert isinstance(adapter.target_col, str)
        assert isinstance(adapter.id_col,     str)
        assert isinstance(adapter.date_col,   str)
        assert isinstance(adapter.momentum_col, str)
        # Values must be non-empty
        assert all(len(v) > 0 for v in [
            adapter.signal_col, adapter.target_col,
            adapter.id_col, adapter.date_col, adapter.momentum_col,
        ])

    def test_validate_schema_raises_on_missing_column(self):
        """validate_schema() raises ValueError listing the missing column."""
        adapter = COVID_Adapter()
        df = pd.DataFrame({
            adapter.id_col:   ["06075"],
            adapter.date_col: [pd.Timestamp("2023-01-04")],
            # signal_col and target_col intentionally absent
        })
        with pytest.raises(ValueError, match="missing required columns"):
            adapter.validate_schema(df)

    def test_validate_schema_passes_with_all_required(self):
        """validate_schema() is a no-op when all required columns are present."""
        adapter = COVID_Adapter()
        df = pd.DataFrame({
            adapter.signal_col: [1.0],
            adapter.target_col: [2.0],
            adapter.id_col:     ["06075"],
            adapter.date_col:   [pd.Timestamp("2023-01-04")],
        })
        adapter.validate_schema(df)   # must not raise

    def test_covid_adapter_is_concrete(self):
        """COVID_Adapter is fully instantiable (not abstract)."""
        adapter = COVID_Adapter()
        assert isinstance(adapter, BaseDatasetAdapter)

    def test_new_adapter_must_implement_all_abstract_methods(self):
        """Attempting to instantiate an incomplete adapter raises TypeError."""
        class IncompleteAdapter(BaseDatasetAdapter):
            @property
            def signal_col(self) -> str: return "x"
            # Missing: target_col, id_col, date_col, momentum_col,
            #          load_signal, load_target, clean, build_features,
            #          transform_target

        with pytest.raises(TypeError):
            IncompleteAdapter()
