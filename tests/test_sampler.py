"""
Tests for src/data_pipeline/sampler.py

What this suite validates
--------------------------
PhaseLabeler
  1. fit() computes per-county thresholds from training data
  2. label() assigns 'baseline' to quiet, flat-velocity weeks
  3. label() assigns 'onset' to weeks with strongly rising WW velocity
  4. label() assigns 'peak' to elevated, low-velocity weeks
  5. label() assigns 'decay' to weeks with strongly falling velocity
  6. Unfitted labeler raises RuntimeError on label()

StratifiedWindowSampler
  7. sample() returns at least as many rows as the input DataFrame
  8. sample() creates synthetic unique_ids for onset/peak/decay windows
  9. Synthetic unique_ids have the source FIPS as the first token (for tft_model compatibility)
 10. sample() raises ValueError when phase_col is absent

Run with:
    pytest tests/test_sampler.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import COUNTY_COL, NWSS_DATE_COL, WW_FEATURE_COL
from src.data_pipeline.sampler import (
    PHASE_COL,
    PhaseLabeler,
    StratifiedWindowSampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weekly_series(
    county: str,
    start: str,
    values: list[float],
    vel_values: list[float] | None = None,
    signal_col: str = WW_FEATURE_COL,
    vel_col: str = "vel_concentration",
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(values), freq="W-WED")
    df = pd.DataFrame({
        COUNTY_COL:   [county] * len(values),
        NWSS_DATE_COL: dates,
        signal_col:   values,
    })
    if vel_values is not None:
        df[vel_col] = vel_values
    else:
        df[vel_col] = pd.Series(values).diff().fillna(0.0).values
    return df


# ===========================================================================
# PhaseLabeler — thresholds
# ===========================================================================

class TestPhaseLabelerFit:

    def test_fit_computes_per_county_thresholds(self):
        """fit() should store thresholds for each county present in training."""
        low_train  = _weekly_series("06075", "2021-01-06", [1.0] * 20)
        high_train = _weekly_series("06085", "2021-01-06", [5.0] * 20)
        train_df   = pd.concat([low_train, high_train], ignore_index=True)

        labeler = PhaseLabeler()
        labeler.fit(train_df)

        assert "06075" in labeler._params
        assert "06085" in labeler._params
        # p40 of 06075 (all-1.0) ≈ 1.0; p40 of 06085 (all-5.0) ≈ 5.0
        assert labeler._params["06075"].p40_signal == pytest.approx(1.0, abs=0.01)
        assert labeler._params["06085"].p40_signal == pytest.approx(5.0, abs=0.01)

    def test_unfitted_labeler_raises(self):
        """label() before fit() must raise RuntimeError."""
        labeler = PhaseLabeler()
        df = _weekly_series("06075", "2021-01-06", [1.0] * 5)
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.label(df)


# ===========================================================================
# PhaseLabeler — phase assignment correctness
# ===========================================================================

class TestPhaseLabelerAssignment:

    def _make_labeler(self) -> PhaseLabeler:
        """Labeler trained on quiet [0.5] signal; p40≈0.5, vel_threshold≈0."""
        train = _weekly_series("06075", "2021-01-06", [0.5] * 30)
        return PhaseLabeler().fit(train)

    def test_quiet_flat_signal_is_baseline(self):
        """Signal below p40 with near-zero velocity → baseline."""
        labeler = self._make_labeler()
        # All quiet — signal never rises
        df     = _weekly_series("06075", "2022-01-05", [0.3] * 8,
                                vel_values=[0.0] * 8)
        result = labeler.label(df)
        assert (result[PHASE_COL] == "baseline").all(), (
            f"Expected all baseline, got: {result[PHASE_COL].value_counts().to_dict()}"
        )

    def test_rising_velocity_is_onset(self):
        """Positive velocity above threshold AND signal above p20 → onset."""
        labeler = self._make_labeler()
        # vel_threshold ≈ 0 (all-zero training), so any positive velocity qualifies
        vals = [0.5] * 4 + [1.0, 2.0, 3.0, 4.0]
        vels = [0.0] * 4 + [0.5, 1.0, 1.0, 1.0]   # strongly positive
        df   = _weekly_series("06075", "2022-01-05", vals, vel_values=vels)

        result  = labeler.label(df)
        # The last 4 weeks should be onset
        onsets  = result[result[PHASE_COL] == "onset"]
        assert len(onsets) >= 3, (
            f"Expected ≥3 onset weeks, got {len(onsets)}"
        )

    def test_elevated_flat_signal_is_peak(self):
        """Signal above p70 AND |velocity| near zero → peak."""
        labeler = self._make_labeler()
        # p70 ≈ 0.5 (all training = 0.5); test with high, flat signal
        vals = [5.0] * 8   # well above p70
        vels = [0.0] * 8   # flat (near peak)
        df   = _weekly_series("06075", "2022-01-05", vals, vel_values=vels)

        result = labeler.label(df)
        assert (result[PHASE_COL] == "peak").all(), (
            f"Expected all peak, got: {result[PHASE_COL].value_counts().to_dict()}"
        )

    def test_falling_velocity_is_decay(self):
        """Strongly negative velocity AND signal above p20 → decay."""
        labeler = self._make_labeler()
        vals = [4.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5]
        vels = [-1.0, -1.0, -0.5, -0.5, -0.4, -0.2, -0.1, -0.05]
        df   = _weekly_series("06075", "2022-01-05", vals, vel_values=vels)

        result = labeler.label(df)
        decays = result[result[PHASE_COL] == "decay"]
        assert len(decays) >= 3, (
            f"Expected ≥3 decay weeks at high signal with negative vel, got {len(decays)}"
        )


# ===========================================================================
# StratifiedWindowSampler
# ===========================================================================

class TestStratifiedWindowSampler:

    def _labeled_panel(self) -> pd.DataFrame:
        """Panel with a clear onset episode in county 06075."""
        # Train on quiet; label includes an onset window
        train = _weekly_series("06075", "2021-01-06", [0.5] * 40)
        labeler = PhaseLabeler().fit(train)

        vals = [0.5] * 10 + [1.0, 2.0, 3.0, 4.0, 4.0] + [0.5] * 10
        vels = [0.0] * 10 + [0.5, 1.0, 1.0, 0.5, 0.0] + [0.0] * 10
        df   = _weekly_series("06075", "2022-01-05", vals, vel_values=vels)
        return labeler.label(df)

    def test_sample_returns_at_least_original_rows(self):
        """sample() must return ≥ as many rows as the input."""
        labeled = self._labeled_panel()
        sampler = StratifiedWindowSampler(input_size=8, h=2)
        result  = sampler.sample(labeled)
        assert len(result) >= len(labeled)

    def test_sample_creates_synthetic_unique_ids(self):
        """Onset windows must produce synthetic unique_ids."""
        labeled = self._labeled_panel()
        sampler = StratifiedWindowSampler(input_size=8, h=2,
                                          oversample_factors={"baseline": 0, "onset": 2, "peak": 0, "decay": 0})
        result  = sampler.sample(labeled)
        synth   = result[result[COUNTY_COL] != "06075"][COUNTY_COL].unique()
        assert len(synth) > 0, "Expected synthetic unique_ids for onset windows"

    def test_synthetic_ids_have_source_fips_as_prefix(self):
        """All synthetic unique_ids must start with the source county FIPS."""
        labeled = self._labeled_panel()
        sampler = StratifiedWindowSampler(input_size=8, h=2)
        result  = sampler.sample(labeled)
        synth   = result[result[COUNTY_COL] != "06075"][COUNTY_COL].unique()
        for uid in synth:
            prefix = uid.split("_")[0]
            assert prefix == "06075", (
                f"Synthetic id '{uid}' must start with source FIPS '06075'"
            )

    def test_sample_raises_without_phase_col(self):
        """sample() must raise ValueError when phase_col is absent."""
        df = _weekly_series("06075", "2022-01-05", [1.0] * 20)
        sampler = StratifiedWindowSampler(input_size=8, h=2)
        with pytest.raises(ValueError, match="not found"):
            sampler.sample(df)

    def test_zero_oversample_keeps_only_original(self):
        """oversample_factors all-zero → result equals input (no augmentation)."""
        labeled = self._labeled_panel()
        sampler = StratifiedWindowSampler(
            input_size=8, h=2,
            oversample_factors={"baseline": 0, "onset": 0, "peak": 0, "decay": 0},
        )
        result = sampler.sample(labeled)
        assert len(result) == len(labeled)
        assert set(result[COUNTY_COL].unique()) == {"06075"}
