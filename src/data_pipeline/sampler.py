"""
Phase-Aware Training Data Sampler.

Purpose
-------
Solves the "Dumbbell Problem": when the training set is dominated by quiet
baseline weeks (~80% of the Bay Area timeline), the TFT learns to always
predict near-baseline levels.  Onset/peak/decay windows are so rare that
the model never adequately learns the dynamics it is deployed to detect.

Solution: Stratified Window Sampling
-------------------------------------
1. PhaseLabeler labels each (county, date) observation as one of four
   epidemic phases using the WW signal's level and velocity:

   Baseline    Stable, low-amplitude.  Signal below training p40 AND
               |velocity| below the training median absolute velocity.

   Onset       The rising edge.  Velocity strongly positive AND signal
               above the local quiet floor — WW is accelerating.

   Peak        Near the outbreak apex.  Signal above training p70 AND
               velocity near zero (wave has crested).

   Decay       The falling edge.  Velocity strongly negative AND signal
               still elevated above baseline.

2. StratifiedWindowSampler oversamples onset/peak/decay windows by
   duplicating them as synthetic sub-series with modified unique_ids.
   NeuralForecast trains on these as additional series with shared weights,
   so the model sees proportionally more transition-phase examples without
   any architectural change.

The 26-week lookback constraint is always satisfied: each synthetic
sub-series is extracted starting INPUT_SIZE weeks before the phase window.

Key design choices
------------------
- Labeling is done on the WW signal (log1p_concentration), not on cases,
  because WW leads cases by 1–3 weeks — it is the signal we want the model
  to learn from.
- Velocity threshold = median |vel_concentration| in training data (per
  county), so the threshold adapts to each county's noise level.
- Synthetic unique_ids have the format "{source_fips}_{phase_code}_{idx:03d}"
  so tft_model._build_static_df() can recover the source FIPS for the
  county_fips_encoded static covariate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    COUNTY_COL,
    NWSS_DATE_COL,
    WW_FEATURE_COL,
)

# Phase codes embedded in synthetic unique_ids
_PHASE_CODE: dict[str, str] = {
    "baseline": "bsl",
    "onset":    "ons",
    "peak":     "pek",
    "decay":    "dcy",
}
PHASE_COL = "epidemic_phase"


# ---------------------------------------------------------------------------
# Per-county labeling parameters (fitted on training data)
# ---------------------------------------------------------------------------

@dataclass
class _CountyPhaseParams:
    """Thresholds for one county, derived from its training distribution."""

    p40_signal:     float   # below this = baseline level
    p70_signal:     float   # above this = peak/elevated level
    p20_signal:     float   # above this = onset eligible (above floor)
    vel_threshold:  float   # |vel| > this = meaningful directional move


# ---------------------------------------------------------------------------
# PhaseLabeler
# ---------------------------------------------------------------------------

class PhaseLabeler:
    """Labels each (county, date) row with an epidemic phase.

    Phase assignment is deterministic and purely rule-based — no ML.
    The rules use the WW signal level (relative to the training distribution)
    and velocity direction (relative to the per-county noise level).

    Parameters
    ----------
    signal_col    : WW concentration column (default WW_FEATURE_COL).
    velocity_col  : 1st derivative of WW signal (default "vel_concentration").
    id_col        : County identifier column.
    date_col      : Date column.
    phase_col     : Output column name for the phase label.
    """

    def __init__(
        self,
        signal_col:   str = WW_FEATURE_COL,
        velocity_col: str = "vel_concentration",
        id_col:       str = COUNTY_COL,
        date_col:     str = NWSS_DATE_COL,
        phase_col:    str = PHASE_COL,
    ) -> None:
        self.signal_col   = signal_col
        self.velocity_col = velocity_col
        self.id_col       = id_col
        self.date_col     = date_col
        self.phase_col    = phase_col
        self._params: dict[str, _CountyPhaseParams] = {}

    def fit(self, train_df: pd.DataFrame) -> "PhaseLabeler":
        """Compute per-county phase thresholds from training data only."""
        self._params = {}
        for county, grp in train_df.groupby(self.id_col):
            sig  = grp[self.signal_col].dropna()
            if len(sig) < 4:
                logger.warning(
                    "PhaseLabeler.fit: county {} has < 4 signal rows — using global defaults.",
                    county,
                )
                self._params[str(county)] = _CountyPhaseParams(
                    p40_signal=1.0, p70_signal=2.0, p20_signal=0.5, vel_threshold=0.1
                )
                continue

            vel_vals = (
                grp[self.velocity_col].dropna().abs()
                if self.velocity_col in grp.columns
                else pd.Series([0.1])
            )

            self._params[str(county)] = _CountyPhaseParams(
                p40_signal=float(np.percentile(sig, 40)),
                p70_signal=float(np.percentile(sig, 70)),
                p20_signal=float(np.percentile(sig, 20)),
                vel_threshold=float(
                    np.percentile(vel_vals, 50) if len(vel_vals) > 1 else 0.1
                ),
            )
        logger.info(
            "PhaseLabeler fitted on {} counties.", len(self._params)
        )
        return self

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``phase_col`` column with Baseline / Onset / Peak / Decay labels.

        Phase assignment logic (applied to WW signal, not cases):

        Baseline: signal < p40  AND  |velocity| <= vel_threshold
        Onset:    velocity > vel_threshold  AND  signal > p20
                  (WW accelerating above the quiet floor)
        Peak:     signal > p70  AND  |velocity| <= vel_threshold
                  (WW elevated but no longer rising fast)
        Decay:    velocity < −vel_threshold  AND  signal > p20
                  (WW falling from an elevated level)
        Default:  baseline (handles ambiguous weeks)
        """
        if not self._params:
            raise RuntimeError("PhaseLabeler not fitted. Call fit(train_df) first.")

        df = df.copy()
        phase_series: list[pd.Series] = []

        for county, grp in df.groupby(self.id_col):
            p = self._params.get(str(county))
            if p is None:
                # Unknown county → default everything to baseline
                phase_series.append(
                    pd.Series("baseline", index=grp.index, name=self.phase_col)
                )
                continue

            sig  = grp[self.signal_col].astype(float)
            vel  = (
                grp[self.velocity_col].astype(float)
                if self.velocity_col in grp.columns
                else pd.Series(0.0, index=grp.index)
            )

            phase = pd.Series("baseline", index=grp.index, dtype=str, name=self.phase_col)

            # Apply in priority order (onset/peak/decay override baseline)
            onset_mask  = (vel > p.vel_threshold)  & (sig > p.p20_signal)
            peak_mask   = (sig > p.p70_signal)     & (vel.abs() <= p.vel_threshold)
            decay_mask  = (vel < -p.vel_threshold) & (sig > p.p20_signal)

            phase[onset_mask] = "onset"
            phase[peak_mask]  = "peak"
            phase[decay_mask] = "decay"
            # baseline is the default; no mask needed

            phase_series.append(phase)

        df[self.phase_col] = pd.concat(phase_series).reindex(df.index)
        return df

    def phase_distribution(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame of phase counts per county."""
        if self.phase_col not in labeled_df.columns:
            raise ValueError(f"Column '{self.phase_col}' not found. Call label() first.")
        return (
            labeled_df.groupby([self.id_col, self.phase_col])
            .size()
            .rename("n_weeks")
            .reset_index()
        )


# ---------------------------------------------------------------------------
# StratifiedWindowSampler
# ---------------------------------------------------------------------------

class StratifiedWindowSampler:
    """Balances training data across epidemic phases by oversampling minority windows.

    For each minority-phase run (onset / peak / decay), a synthetic sub-series
    is extracted that includes:
      - ``input_size`` weeks of context BEFORE the phase window
      - the full phase run
      - ``h`` weeks AFTER the phase window (prediction targets)

    This sub-series is duplicated ``oversample_factor[phase]`` times under a
    modified unique_id so NeuralForecast's internal dataloader draws
    proportionally more gradient-steps from it.

    The synthetic unique_id format is ``{source_fips}_{phase_code}_{idx:03d}``
    (e.g. ``06075_ons_001``) so ``tft_model._build_static_df()`` can recover
    the source FIPS for county_fips_encoded via a prefix split.

    Parameters
    ----------
    input_size          : TFT lookback window (default 26 = INPUT_SIZE).
    h                   : Forecast horizon (default 8 = H).
    oversample_factors  : How many synthetic copies per phase run.
                          Default emphasises onset/decay (transitions) 3× and
                          peak 2× — the peak is easier to model (high signal
                          is distinctive) but transitions are the hardest.
    phase_col           : Column containing PhaseLabeler output.
    id_col              : County identifier column.
    date_col            : Date column.
    """

    DEFAULT_OVERSAMPLE: dict[str, int] = {
        "baseline": 0,   # not oversampled; full series already covers baseline
        "onset":    3,
        "peak":     2,
        "decay":    3,
    }

    def __init__(
        self,
        input_size:         int               = 26,
        h:                  int               = 8,
        oversample_factors: dict[str, int]    | None = None,
        phase_col:          str               = PHASE_COL,
        id_col:             str               = COUNTY_COL,
        date_col:           str               = NWSS_DATE_COL,
    ) -> None:
        self.input_size         = input_size
        self.h                  = h
        self.oversample_factors = oversample_factors or dict(self.DEFAULT_OVERSAMPLE)
        self.phase_col          = phase_col
        self.id_col             = id_col
        self.date_col           = date_col

    def sample(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Return a balanced DataFrame for TFT training.

        The original full series are always included (providing baseline
        coverage).  Minority-phase windows are appended as additional
        sub-series.

        Parameters
        ----------
        labeled_df : Output of ``PhaseLabeler.label()``.  Must contain
                     ``phase_col`` and all TFT training columns.

        Returns
        -------
        pd.DataFrame — the original rows PLUS synthetic oversampled rows.
        Synthetic unique_ids are prefixed with the source FIPS for static
        covariate recovery.
        """
        if self.phase_col not in labeled_df.columns:
            raise ValueError(
                f"Column '{self.phase_col}' not found. "
                "Pass PhaseLabeler.label() output to StratifiedWindowSampler.sample()."
            )

        frames: list[pd.DataFrame] = [labeled_df]

        for county, grp in labeled_df.groupby(self.id_col):
            grp     = grp.sort_values(self.date_col).copy()
            dates   = grp[self.date_col].tolist()
            phases  = grp[self.phase_col].tolist()

            for phase, n_copies in self.oversample_factors.items():
                if n_copies == 0:
                    continue

                # Find all runs of this phase
                runs = _find_phase_runs(dates, phases, phase)
                if not runs:
                    continue

                phase_code = _PHASE_CODE[phase]
                copy_idx   = 0

                for run_start, run_end in runs:
                    context_start = run_start - pd.Timedelta(weeks=self.input_size)
                    window_end    = run_end    + pd.Timedelta(weeks=self.h)

                    sub = grp[
                        (grp[self.date_col] >= context_start) &
                        (grp[self.date_col] <= window_end)
                    ].copy()

                    if len(sub) < self.input_size + 1:
                        # Not enough context (e.g. phase window very early in series)
                        continue

                    for k in range(n_copies):
                        copy_idx += 1
                        dup = sub.copy()
                        dup[self.id_col] = (
                            f"{county}_{phase_code}_{copy_idx:03d}"
                        )
                        frames.append(dup)

        result = pd.concat(frames, ignore_index=True)

        # Log the augmentation summary
        orig_len  = len(labeled_df)
        added_len = len(result) - orig_len
        n_synth   = result[self.id_col].nunique() - labeled_df[self.id_col].nunique()
        logger.info(
            "StratifiedWindowSampler: {} original rows + {} augmented rows "
            "({} synthetic sub-series added).",
            orig_len, added_len, n_synth,
        )

        # Log phase distribution in the augmented set
        _log_phase_distribution(result, self.id_col, self.phase_col)

        return result

    def phase_summary(self, labeled_df: pd.DataFrame) -> dict[str, int]:
        """Return total weeks per phase before sampling (for inspection)."""
        return dict(
            labeled_df[self.phase_col]
            .value_counts()
            .reindex(["baseline", "onset", "peak", "decay"], fill_value=0)
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_phase_runs(
    dates: list[pd.Timestamp],
    phases: list[str],
    target_phase: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start_date, end_date) for each contiguous run of target_phase."""
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_run    = False
    run_start = None

    for date, phase in zip(dates, phases):
        if phase == target_phase and not in_run:
            run_start = date
            in_run    = True
        elif phase != target_phase and in_run:
            runs.append((run_start, prev_date))  # type: ignore[arg-type]
            in_run = False
        prev_date = date

    if in_run and run_start is not None:
        runs.append((run_start, dates[-1]))

    return runs


def _log_phase_distribution(df: pd.DataFrame, id_col: str, phase_col: str) -> None:
    """Debug-log phase counts for the augmented DataFrame."""
    if phase_col not in df.columns:
        return
    counts = df[phase_col].value_counts()
    total  = len(df)
    parts  = [
        f"{phase}={counts.get(phase, 0)} ({counts.get(phase, 0)/total:.1%})"
        for phase in ("baseline", "onset", "peak", "decay")
    ]
    logger.debug("Phase distribution after sampling: {}", "  ".join(parts))
