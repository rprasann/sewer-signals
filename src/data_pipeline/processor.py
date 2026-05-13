"""
Wastewater-to-case alignment pipeline.

Primary unit: copies/g dry sludge (covers all 9 Bay Area counties).
Pass ``target_unit="copies/l wastewater"`` to WastewaterProcessor for the
Section 4.1 liquid-track comparison only.

Stages (in order):
  1.  Column cleaning    — strip commas, coerce numerics
  2.  FIPS zero-padding  — ensure 5-digit county codes
  3.  Unit filtering     — keep TARGET_UNIT rows (default: copies/g dry sludge)
  4.  Non-detect censoring — replace non-detects with LOD/2
  5.  QC filters         — drop rec_eff_percent < 10%; flag inhibition
  6.  County filter      — keep Bay Area FIPS
  7.  Multi-county explosion — one row per county for shared sewersheds
  8.  County-day aggregation — population-weighted mean: Σ(Conc×Pop)/ΣPop
  9.  Centered 7-day rolling mean + relative_decay_rate (daily grain)
  10. Resample daily → weekly (W-WED: week-ending Wednesday, cases-spine aligned)
  11. WW log-transform  — WW_FEATURE_COL = log1p(concentration)
  12. Cases merge       — inner join with cases_df on (county_fips, W-WED date);
                          TARGET_COL = log1p(new_cases)
  13. Calendar features — cyclical DOY sin/cos
  14. Lag features      — 1-, 2-, 3-week lags of WW_FEATURE_COL + WW growth rate
  15. Leakage-free scaling — RobustScaler fit on train, applied to val/test
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.config import (
    BAY_AREA_FIPS,
    CA_WW_SIGNAL_COL,
    COUNTY_COL,
    FOURIER_ORDER,
    INTERPOLATION_MAX_GAP,
    MIN_RECOVERY_EFFICIENCY,
    NWSS_DATE_COL,
    OUTLIER_Z_THRESHOLD,
    POPULATION_COL,
    SECONDARY_UNIT,
    SEWERSHED_COL,
    TARGET_COL,
    TARGET_UNIT,
    TRAIN_END_DATE,
    VAL_END_DATE,
    WW_FEATURE_COL,
)

# CDC NWSS column names as they appear in the raw API response
_CONC_COL = "pcr_target_avg_conc"
_UNIT_COL = "pcr_target_units"
_DETECT_COL = "pcr_target_below_lod"
_LOD_COL = "lod_sewage"
_REC_EFF_COL = "rec_eff_percent"
_INHIBITION_COL = "inhibition_detect"

_NUMERIC_COLS = [
    _CONC_COL, POPULATION_COL, _LOD_COL, _REC_EFF_COL, "flow_rate",
    "hum_frac_cov_conc", "hum_frac_mic_conc",
]


class WastewaterProcessor:
    """End-to-end transform from raw NWSS records to model-ready county panel.

    Usage
    -----
    proc = WastewaterProcessor()
    train_df = proc.run(raw_train)          # fits scaler on training data
    val_df   = proc.transform(raw_val)      # applies fitted scaler (no leakage)
    test_df  = proc.transform(raw_test)
    train, val, test = proc.split(full_processed_df)
    """

    def __init__(
        self,
        fips_filter: list[str] | None = None,
        target_unit: str | None = None,
    ) -> None:
        self.fips_filter: list[str] = fips_filter or list(BAY_AREA_FIPS.values())
        # Defaults to copies/g dry sludge; pass SECONDARY_UNIT for liquid-track comparison
        self.target_unit: str = target_unit if target_unit is not None else TARGET_UNIT
        self._scaler: RobustScaler | None = None   # alias to first county's scaler
        self._scalers: dict[str, RobustScaler] = {}  # per-county scalers (Problem 1 fix)
        self._scale_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        raw: pd.DataFrame,
        cases_df: Optional[pd.DataFrame] = None,
        *,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """Full pipeline. ``fit_scaler=True`` on train; ``False`` for val/test.

        Parameters
        ----------
        raw      : Raw NWSS wastewater CSV rows (pre-filtered to the correct
                   time window by the caller).
        cases_df : Weekly CDC case data for the same time window, with columns
                   ``COUNTY_COL`` (zero-padded FIPS), ``NWSS_DATE_COL``
                   (Wednesday dates), and ``new_cases`` (int, ≥ 0).
                   When None, TARGET_COL is set to NaN (liquid-track mode).
        """
        df = (
            raw
            .pipe(self._clean_columns)
            .pipe(self._zero_pad_fips)
            .pipe(self._filter_units)
            .pipe(self._apply_nondetect_censoring)
            .pipe(self._apply_qc_filters)
            .pipe(self._filter_counties)
            .pipe(self._explode_multi_county)
            .pipe(self._aggregate_to_county_daily)
            .pipe(self._rolling_smooth)
            .pipe(self._resample_to_weekly)
            .pipe(self._log_transform)                          # → WW_FEATURE_COL
            .pipe(lambda d: self._merge_cases(d, cases_df))    # → TARGET_COL
            .pipe(self._add_calendar_features)
            .pipe(self._add_lag_features)                      # lags of WW_FEATURE_COL
        )
        df = self._apply_scaling(df, fit=fit_scaler)
        logger.info(
            "Pipeline complete: {} county-week rows, {} counties.",
            len(df), df[COUNTY_COL].nunique(),
        )
        return df

    def transform(
        self,
        raw: pd.DataFrame,
        cases_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Apply a previously fitted scaler — call after ``run()`` on train."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call run() on training data first.")
        return self.run(raw, cases_df, fit_scaler=False)

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, val, test) slices using config date boundaries."""
        train_end = pd.Timestamp(TRAIN_END_DATE)
        val_end = pd.Timestamp(VAL_END_DATE)
        date = df[NWSS_DATE_COL]
        train = df[date <= train_end].copy()
        val = df[(date > train_end) & (date <= val_end)].copy()
        test = df[date > val_end].copy()
        logger.info(
            "Split → train={}, val={}, test={} rows.", len(train), len(val), len(test)
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Stage 1: Column cleaning
    # ------------------------------------------------------------------

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip commas from numeric strings and coerce to float."""
        df = df.copy()
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )
        df[NWSS_DATE_COL] = pd.to_datetime(df[NWSS_DATE_COL], errors="coerce")
        df = df.dropna(subset=[NWSS_DATE_COL])
        return df

    # ------------------------------------------------------------------
    # Stage 2: FIPS zero-padding
    # ------------------------------------------------------------------

    def _zero_pad_fips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure county_fips values are zero-padded to 5 characters."""
        df = df.copy()
        if COUNTY_COL not in df.columns:
            return df
        # Values may be comma-separated (multi-county sites) — pad each token
        df[COUNTY_COL] = (
            df[COUNTY_COL]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.split(",")
            .apply(lambda tokens: ",".join(t.zfill(5) for t in tokens if t))
        )
        return df

    # ------------------------------------------------------------------
    # Stage 3: Unit filtering
    # ------------------------------------------------------------------

    def _filter_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows whose unit contains self.target_unit (case-insensitive substring).

        Default target_unit is 'copies/g dry sludge' (all 9 Bay Area counties).
        Rows with a non-matching unit are flagged and dropped. Pass
        target_unit=SECONDARY_UNIT to WastewaterProcessor to switch to the
        liquid track for the Section 4.1 sludge-vs-liquid comparison.
        """
        df = df.copy()
        if _UNIT_COL not in df.columns:
            logger.warning("Column '{}' not found — skipping unit filter.", _UNIT_COL)
            return df

        normalised = df[_UNIT_COL].astype(str).str.lower().str.strip()
        # Substring match: handles "copies/g dry sludge", "Copies/G Dry Sludge", etc.
        is_target = normalised.str.contains(self.target_unit.lower(), regex=False)
        is_other = ~is_target

        if is_other.any():
            excluded_units = df.loc[is_other, _UNIT_COL].value_counts().to_dict()
            excluded_sites = (
                df.loc[is_other, SEWERSHED_COL].unique().tolist()
                if SEWERSHED_COL in df.columns
                else []
            )
            logger.warning(
                "Excluding {} rows with non-'{}' units: {}. Affected sites (up to 20): {}",
                is_other.sum(), self.target_unit, excluded_units, excluded_sites[:20],
            )

        df["unit_excluded_flag"] = is_other
        df = df[is_target].copy()
        # Binary indicator: 1=copies/g dry sludge (sludge track), 0=liquid track.
        # Constant per processor run; used by TFT variable selection to distinguish tracks.
        df["is_sludge"] = 1.0 if "copies/g" in self.target_unit.lower() else 0.0
        return df

    # ------------------------------------------------------------------
    # Stage 4: Left-censoring for non-detects
    # ------------------------------------------------------------------

    def _apply_nondetect_censoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace non-detect samples with LOD/2 (Kaplan-Meier convention)."""
        df = df.copy()
        if _CONC_COL not in df.columns:
            return df

        df["concentration"] = df[_CONC_COL].clip(lower=0)

        if _DETECT_COL in df.columns and _LOD_COL in df.columns:
            is_nondetect = (
                df[_DETECT_COL].astype(str).str.lower().isin({"yes", "1", "true"})
            )
            lod_half = df[_LOD_COL].clip(lower=0) / 2.0
            n_nd = is_nondetect.sum()
            if n_nd:
                df.loc[is_nondetect, "concentration"] = lod_half[is_nondetect]
                logger.debug("Left-censored {} non-detect rows with LOD/2.", n_nd)
        elif _LOD_COL in df.columns:
            # Fallback: treat zero/NaN concentration as non-detect when LOD is available
            mask = df["concentration"].isna() | (df["concentration"] == 0)
            df.loc[mask, "concentration"] = df.loc[mask, _LOD_COL].clip(lower=0) / 2.0

        df["concentration"] = df["concentration"].clip(lower=0)
        return df

    # ------------------------------------------------------------------
    # Stage 5: Quality control
    # ------------------------------------------------------------------

    def _apply_qc_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop low-recovery samples; flag PCR inhibition."""
        df = df.copy()

        # Recovery efficiency filter
        if _REC_EFF_COL in df.columns:
            before = len(df)
            # Rows with NaN recovery are kept (no data ≠ fail)
            qc_fail = df[_REC_EFF_COL].notna() & (
                df[_REC_EFF_COL] < MIN_RECOVERY_EFFICIENCY
            )
            df = df[~qc_fail].copy()
            dropped = before - len(df)
            if dropped:
                logger.debug(
                    "QC: dropped {} rows with rec_eff_percent < {:.0f}%.",
                    dropped, MIN_RECOVERY_EFFICIENCY,
                )

        # Inhibition flag (retain rows but mark them)
        if _INHIBITION_COL in df.columns:
            df["inhibition_flag"] = (
                df[_INHIBITION_COL].astype(str).str.lower().isin({"yes", "1", "true"})
            )
            n_inh = df["inhibition_flag"].sum()
            if n_inh:
                logger.warning(
                    "QC: {} samples flagged for PCR inhibition (retained).", n_inh
                )
        else:
            df["inhibition_flag"] = False

        return df

    # ------------------------------------------------------------------
    # Stage 6: County filter
    # ------------------------------------------------------------------

    def _filter_counties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows that contain at least one Bay Area FIPS code."""
        fips_set = set(self.fips_filter)

        def _has_bay_fips(fips_str: str) -> bool:
            return bool(fips_set.intersection(fips_str.split(",")))

        mask = df[COUNTY_COL].apply(_has_bay_fips)
        return df[mask].copy()

    # ------------------------------------------------------------------
    # Stage 7: Explode multi-county sites
    # ------------------------------------------------------------------

    def _explode_multi_county(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split comma-delimited county_fips into one row per county."""
        df = df.copy()
        df[COUNTY_COL] = df[COUNTY_COL].str.split(",")
        df = df.explode(COUNTY_COL).copy()
        df[COUNTY_COL] = df[COUNTY_COL].str.strip()
        # After explosion keep only known Bay Area FIPS
        return df[df[COUNTY_COL].isin(self.fips_filter)].copy()

    # ------------------------------------------------------------------
    # Stage 8: Population-weighted aggregation to county-day
    # ------------------------------------------------------------------

    def _aggregate_to_county_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weighted mean: Σ(Conc × Pop) / ΣPop per county-day."""
        df = df.copy()
        pop = df[POPULATION_COL].fillna(0) if POPULATION_COL in df.columns else pd.Series(1.0, index=df.index)
        df["_w"] = pop
        df["_wc"] = df["concentration"] * pop

        agg = (
            df.groupby([COUNTY_COL, NWSS_DATE_COL], sort=True)
            .agg(
                _wc_sum=("_wc", "sum"),
                _w_sum=("_w", "sum"),
                sewershed_count=(SEWERSHED_COL, "nunique") if SEWERSHED_COL in df.columns else ("_w", "count"),
                total_population=("_w", "sum"),
                inhibition_flag=("inhibition_flag", "any"),
            )
            .reset_index()
        )
        agg["concentration"] = agg["_wc_sum"] / agg["_w_sum"].replace(0.0, np.nan)
        agg = agg.drop(columns=["_wc_sum", "_w_sum"])
        agg = agg.rename(columns={"total_population": POPULATION_COL})

        n_missing = agg["concentration"].isna().sum()
        if n_missing:
            logger.debug("Aggregation: {} county-days have no valid concentration.", n_missing)

        return agg

    # ------------------------------------------------------------------
    # Stage 9: Centered 7-day rolling mean + relative_decay_rate (daily grain)
    # ------------------------------------------------------------------

    def _rolling_smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to daily frequency, apply centered 7-day rolling mean, and
        compute relative_decay_rate = (conc_t - conc_{t-7}) / (conc_{t-7} + ε).

        The relative_decay_rate captures both growth (positive) and recovery
        (negative) on a 7-day timescale — a key feature for predicting outbreak
        decay that liquid-matrix signals fail to provide reliably.
        """
        frames: list[pd.DataFrame] = []
        for fips, grp in df.groupby(COUNTY_COL):
            grp = grp.set_index(NWSS_DATE_COL).sort_index()
            full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
            grp = grp.reindex(full_idx)
            grp[COUNTY_COL] = fips

            # Linear interpolation for short gaps only
            grp["concentration"] = grp["concentration"].interpolate(
                method="linear", limit=INTERPOLATION_MAX_GAP
            )
            # Centered 7-day mean (min_periods=3 preserves edge weeks)
            grp["concentration"] = (
                grp["concentration"].rolling(7, center=True, min_periods=3).mean()
            )
            # Relative decay rate: 7-day % change on smoothed signal
            lagged_7d = grp["concentration"].shift(7)
            grp["relative_decay_rate"] = (
                (grp["concentration"] - lagged_7d) / (lagged_7d.abs() + 1e-9)
            ).clip(-5.0, 5.0)  # winsorise extreme values from early warm-up

            frames.append(
                grp.reset_index().rename(columns={"index": NWSS_DATE_COL})
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Stage 10: Resample daily → weekly (W-WED: week-ending Wednesday)
    # ------------------------------------------------------------------

    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate smoothed daily signal to weekly W-WED grain.

        W-WED aligns with the CDC cases dataset, which is strictly
        Wednesday-anchored.  Using the same anchor ensures a clean
        inner join in _merge_cases() with no date-offset mismatches.
        """
        frames: list[pd.DataFrame] = []
        for fips, grp in df.groupby(COUNTY_COL):
            grp = grp.set_index(NWSS_DATE_COL).sort_index()
            weekly_conc = grp["concentration"].resample("W-WED").mean()
            weekly_pop = (
                grp[POPULATION_COL].resample("W-WED").last()
                if POPULATION_COL in grp.columns else None
            )
            weekly_rdr = (
                grp["relative_decay_rate"].resample("W-WED").mean()
                if "relative_decay_rate" in grp.columns else None
            )

            week_df = weekly_conc.rename("concentration").to_frame()
            if weekly_pop is not None:
                week_df[POPULATION_COL] = weekly_pop
            if weekly_rdr is not None:
                week_df["relative_decay_rate"] = weekly_rdr
            week_df[COUNTY_COL] = fips
            frames.append(
                week_df.reset_index().rename(columns={"index": NWSS_DATE_COL})
            )

        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["concentration"]).copy()
        return out

    # ------------------------------------------------------------------
    # Stage 11: WW log-feature transform
    # ------------------------------------------------------------------

    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log1p(concentration) → WW_FEATURE_COL (hist_exog, not target).

        The prediction target (TARGET_COL = log1p_new_cases) is added later
        in _merge_cases() once the cases data is joined in.
        """
        df = df.copy()
        df[WW_FEATURE_COL] = np.log1p(df["concentration"].clip(lower=0))
        return df

    # ------------------------------------------------------------------
    # Stage 12: Cases merge — adds TARGET_COL = log1p(new_cases)
    # ------------------------------------------------------------------

    def _merge_cases(
        self,
        df: pd.DataFrame,
        cases_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Inner-join WW weekly data with cases data on (county_fips, W-WED date).

        Computes TARGET_COL = log1p(new_cases) as the prediction target.
        The inner join naturally restricts to the overlap window where both
        datasets have observations.

        When cases_df is None (liquid-track mode), TARGET_COL is set to NaN
        so the rest of the pipeline still runs.
        """
        if cases_df is None or cases_df.empty:
            logger.warning(
                "_merge_cases: no cases_df supplied — {} set to NaN "
                "(liquid-track / dashboard-only mode).",
                TARGET_COL,
            )
            df = df.copy()
            df[TARGET_COL] = np.nan
            df["new_cases"] = np.nan
            return df

        cases = cases_df.copy()

        # Normalise column names to (COUNTY_COL, NWSS_DATE_COL, new_cases)
        for src, dst in [
            ("fips_code", COUNTY_COL),
            ("date",      NWSS_DATE_COL),
            ("end_date",  NWSS_DATE_COL),
        ]:
            if src in cases.columns and dst not in cases.columns:
                cases = cases.rename(columns={src: dst})

        cases[NWSS_DATE_COL] = pd.to_datetime(cases[NWSS_DATE_COL], errors="coerce")
        cases[COUNTY_COL]    = cases[COUNTY_COL].astype(str).str.strip().str.zfill(5)

        if "new_cases" not in cases.columns:
            logger.warning("_merge_cases: 'new_cases' column missing — TARGET_COL set to NaN.")
            df = df.copy()
            df[TARGET_COL] = np.nan
            df["new_cases"] = np.nan
            return df

        cases["new_cases"] = pd.to_numeric(
            cases["new_cases"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).clip(lower=0).fillna(0)

        df = df.copy()
        df[NWSS_DATE_COL] = pd.to_datetime(df[NWSS_DATE_COL], errors="coerce")

        n_before = len(df)
        df = df.merge(
            cases[[COUNTY_COL, NWSS_DATE_COL, "new_cases"]],
            on=[COUNTY_COL, NWSS_DATE_COL],
            how="inner",
        )
        dropped = n_before - len(df)
        if dropped:
            logger.debug(
                "_merge_cases: dropped {} WW rows with no matching case week.", dropped
            )

        df["new_cases"] = df["new_cases"].clip(lower=0)
        df[TARGET_COL]  = np.log1p(df["new_cases"])
        logger.info(
            "Cases merged: {} county-week rows, {} counties.",
            len(df), df[COUNTY_COL].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Stage 13: Calendar features (cyclical DOY)
    # ------------------------------------------------------------------

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        doy = df[NWSS_DATE_COL].dt.dayofyear
        dow = df[NWSS_DATE_COL].dt.dayofweek
        month = df[NWSS_DATE_COL].dt.month

        for k in range(1, FOURIER_ORDER + 1):
            df[f"sin_annual_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            df[f"cos_annual_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

        df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["week_of_year"] = df[NWSS_DATE_COL].dt.isocalendar().week.astype(int)
        return df

    # ------------------------------------------------------------------
    # Stage 14: WW lag features (1–3 weeks of WW_FEATURE_COL)
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lagged WW features, derivative expansion, and lagged target features.

        Problem 2 fix — Calendar-safe lags:
        The inner join in _merge_cases() silently drops weeks where WW and cases
        data don't overlap (holiday reporting gaps, etc.).  pandas shift(n) moves
        by row count, not calendar time — so lag1w at week t may actually contain
        the value from week t-2 when t-1 was dropped.  Fix: reindex each county
        to a complete W-WED spine before shifting; missing weeks become NaN rows
        which produce NaN lags (correct), and are dropped after lag computation.

        Problem 3 fix — True relative growth rate in original concentration space:
        The old formula (Δlog1p / |log1p|) was a log-space hybrid that made the
        same 100% concentration doubling produce 0.15 at low baseline but only
        0.06 at Omicron peak — the opposite of the desired surge signal.  The
        new formula uses raw concentration: a doubling always → growth_rate_1w ≈ 1.0.
        """
        df = df.sort_values([COUNTY_COL, NWSS_DATE_COL]).copy()

        # Reindex each county to a complete W-WED spine (Problem 2 fix).
        # After reindexing, shift(n) is calendar-aligned: a gap week becomes a NaN
        # row whose downstream lags are also NaN, causing those rows to be dropped
        # by _to_nf_format() rather than silently using a stale value.
        frames = []
        for fips, grp in df.groupby(COUNTY_COL):
            grp = grp.set_index(NWSS_DATE_COL).sort_index()
            full_spine = pd.date_range(grp.index.min(), grp.index.max(), freq="W-WED")
            grp = grp.reindex(full_spine)
            grp[COUNTY_COL] = fips          # fill NaN county_fips on synthetic rows
            frames.append(grp.reset_index().rename(columns={"index": NWSS_DATE_COL}))
        df = pd.concat(frames, ignore_index=True)

        # WW position lags — computed on complete spine so shifts are calendar-aligned
        ww_grp = df.groupby(COUNTY_COL)[WW_FEATURE_COL]
        for lag_weeks in [1, 2, 3]:
            df[f"{WW_FEATURE_COL}_lag{lag_weeks}w"] = ww_grp.shift(lag_weeks)

        # growth_rate_1w — true relative rate in original concentration space (Problem 3 fix).
        # A 100% doubling now consistently maps to ≈ 1.0 regardless of baseline magnitude.
        conc_grp = df.groupby(COUNTY_COL)["concentration"]
        conc_lag1 = conc_grp.shift(1)
        df["growth_rate_1w"] = (
            (df["concentration"] - conc_lag1) / (conc_lag1.abs() + 1e-6)
        ).clip(-5.0, 5.0)

        # Derivative expansion — velocity, acceleration, and rolling momentum statistics
        df["diff_concentration"] = ww_grp.diff(1)                 # 1st derivative (velocity)

        # Acceleration = second difference; captures inflection points of surges.
        # Positive accel + positive velocity → acceleration phase (high alert).
        # Negative accel + positive velocity → deceleration / approaching peak.
        df["ww_accel"] = df.groupby(COUNTY_COL)["diff_concentration"].transform(
            lambda s: s.diff(1)
        )

        # Lagged velocity: where was momentum 1 week ago?  Gives the model
        # explicit "momentum direction" without recomputing from raw lags.
        df["diff_concentration_lag1w"] = df.groupby(COUNTY_COL)["diff_concentration"].shift(1)

        df["log1p_concentration_2w_ma"]  = ww_grp.transform(
            lambda s: s.rolling(2, min_periods=1).mean()
        )
        df["log1p_concentration_4w_ma"]  = ww_grp.transform(
            lambda s: s.rolling(4, min_periods=1).mean()
        )
        df["log1p_concentration_2w_std"] = ww_grp.transform(
            lambda s: s.rolling(2, min_periods=2).std()
        )
        df["log1p_concentration_4w_std"] = ww_grp.transform(
            lambda s: s.rolling(4, min_periods=2).std()
        )

        # Target lags (sludge track only; skipped when TARGET_COL is all-NaN)
        if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
            case_grp = df.groupby(COUNTY_COL)[TARGET_COL]
            for lag_weeks in [1, 2, 3]:
                df[f"{TARGET_COL}_lag{lag_weeks}w"] = case_grp.shift(lag_weeks)

        # Z-score outlier flag on raw WW concentration (per-county)
        z = df.groupby(COUNTY_COL)["concentration"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
        df["outlier_flag"] = z.abs() > OUTLIER_Z_THRESHOLD

        # Drop synthetic rows inserted by reindexing (WW_FEATURE_COL is NaN there)
        df = df.dropna(subset=[WW_FEATURE_COL]).copy()
        return df

    # ------------------------------------------------------------------
    # Stage 15: Leakage-free RobustScaler
    # ------------------------------------------------------------------

    def _apply_scaling(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """Per-county RobustScaler — normalise within each county's training rows.

        Problem 1 fix: fitting a single global scaler on all counties combined
        compresses Santa Clara's large surge values against the IQR of smaller
        counties (Napa, Solano), systematically underscaling SC's dynamic range
        and causing over-narrow prediction intervals.

        Approach: for each county, fit a RobustScaler on that county's training
        rows only, then transform that county's rows.  All counties are then on
        a consistent [IQR-normalised] scale before pooling into the TFT batch.

        Stored state:
            self._scalers     : dict FIPS → RobustScaler (one per county)
            self._scale_cols  : list of columns that were scaled
            self._scaler      : alias to the first county's scaler (backward compat
                                with _invert_scaling_to_log1p fallback path)
        """
        candidates = [
            TARGET_COL,                       # log1p_new_cases  (prediction target)
            WW_FEATURE_COL,                   # log1p_concentration  (WW feature at t)
            f"{WW_FEATURE_COL}_lag1w",        # WW at t-1
            f"{WW_FEATURE_COL}_lag2w",        # WW at t-2
            f"{WW_FEATURE_COL}_lag3w",        # WW at t-3
            f"{TARGET_COL}_lag1w",            # cases at t-1  (temporal momentum)
            f"{TARGET_COL}_lag2w",            # cases at t-2
            f"{TARGET_COL}_lag3w",            # cases at t-3
            "growth_rate_1w",                 # WW relative week-over-week growth (raw-conc basis)
            "relative_decay_rate",            # WW 7-day relative change
            "diff_concentration",             # absolute weekly Δ log1p_conc (velocity)
            "ww_accel",                       # second Δ log1p_conc (acceleration)
            "diff_concentration_lag1w",       # velocity 1 week ago (momentum direction)
            "log1p_concentration_2w_ma",      # 2-week rolling mean
            "log1p_concentration_4w_ma",      # 4-week rolling mean
            "log1p_concentration_2w_std",     # 2-week rolling std (local volatility)
            "log1p_concentration_4w_std",     # 4-week rolling std (medium volatility)
        ]
        self._scale_cols = [c for c in candidates if c in df.columns]
        if not self._scale_cols:
            return df

        df = df.sort_values([COUNTY_COL, NWSS_DATE_COL]).copy()

        if fit:
            self._scalers = {}
            frames = []
            for fips, grp in df.groupby(COUNTY_COL):
                scaler = RobustScaler()
                grp = grp.copy()
                grp[self._scale_cols] = scaler.fit_transform(grp[self._scale_cols])
                self._scalers[fips] = scaler
                frames.append(grp)
            # Alias for backward compat with single-scaler consumers
            self._scaler = next(iter(self._scalers.values())) if self._scalers else None
            df = pd.concat(frames, ignore_index=True)
            logger.info(
                "Per-county scalers fitted for {} counties; scale_cols: {}",
                len(self._scalers), self._scale_cols,
            )
        else:
            if not self._scalers:
                raise RuntimeError(
                    "Per-county scalers not fitted. Call run() on training data first."
                )
            frames = []
            for fips, grp in df.groupby(COUNTY_COL):
                grp = grp.copy()
                if fips not in self._scalers:
                    logger.warning(
                        "_apply_scaling: no scaler for FIPS {} — rows left unscaled.", fips
                    )
                    frames.append(grp)
                    continue
                grp[self._scale_cols] = self._scalers[fips].transform(grp[self._scale_cols])
                frames.append(grp)
            df = pd.concat(frames, ignore_index=True)
        return df


class CAWastewaterProcessor(WastewaterProcessor):
    """Processes California state WW + Cases datasets for the extended spine.

    Replaces CDC-specific stages (FIPS explosion, unit filter, QC, nondetect
    censoring) with CA-appropriate logic.  All downstream stages — rolling
    smooth, weekly resample, log transform, cases merge, calendar features,
    lag features, scaling — are inherited unchanged from WastewaterProcessor.

    Input contract
    --------------
    raw_ww   : CA WW CSV pre-filtered to Bay Area + SARS-CoV-2 + solid track
               by _load_ca_ww_csv() in main.py.  Must have columns:
                 NWSS_DATE_COL (already renamed from 'Sample Date'),
                 'county_name' (already renamed from 'County'),
                 ww_signal_col ('Raw Concentration' or 'Norm Pmmov').
    cases_df : Weekly W-WED cases produced by _load_ca_cases_csv() in main.py.
               Must have columns: COUNTY_COL (FIPS), NWSS_DATE_COL, 'new_cases'.
               The existing _merge_cases() handles this format unchanged.

    Stage mapping
    -------------
    _clean_ca_ww()                  ← replaces stages 1–7 (clean/FIPS/unit/QC/filter/explode)
    _aggregate_ca_to_county_daily() ← replaces stage 8 (population-weighted agg)
    _rolling_smooth()               inherited (stage 9)
    _resample_to_weekly()           inherited (stage 10)
    _log_transform()                inherited (stage 11)
    _merge_cases()                  inherited (stage 12; cases_df pre-formatted by loader)
    _add_calendar_features()        inherited (stage 13)
    _add_lag_features()             inherited (stage 14)
    _apply_scaling()                inherited (stage 15)
    """

    def __init__(self, ww_signal_col: str = CA_WW_SIGNAL_COL) -> None:
        super().__init__()  # initialises _scaler, _scale_cols; fips_filter/target_unit unused
        self._ww_signal_col = ww_signal_col

    # ------------------------------------------------------------------
    # Public API — override run() to use CA-specific early stages
    # ------------------------------------------------------------------

    def run(
        self,
        raw_ww: pd.DataFrame,
        cases_df: Optional[pd.DataFrame] = None,
        *,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        df = (
            raw_ww
            .pipe(self._clean_ca_ww)
            .pipe(self._aggregate_ca_to_county_daily)
            .pipe(self._rolling_smooth)
            .pipe(self._resample_to_weekly)
            .pipe(self._log_transform)
            .pipe(lambda d: self._merge_cases(d, cases_df))
            .pipe(self._add_calendar_features)
            .pipe(self._add_lag_features)
        )
        df = self._apply_scaling(df, fit=fit_scaler)
        logger.info(
            "CA pipeline complete: {} county-week rows, {} counties.",
            len(df), df[COUNTY_COL].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Stage CA-1: Column normalisation + county-name → FIPS mapping
    # ------------------------------------------------------------------

    def _clean_ca_ww(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise CA WW columns to the internal schema.

        Input expects columns already renamed by _load_ca_ww_csv():
          NWSS_DATE_COL  (was 'Sample Date')
          'county_name'  (was 'County')
          self._ww_signal_col ('Raw Concentration' or 'Norm Pmmov')

        Output: NWSS_DATE_COL, COUNTY_COL (FIPS), 'concentration',
                'county_site' (for sewershed count).
        """
        df = df.copy()

        # Signal → concentration (numeric, non-negative)
        if self._ww_signal_col not in df.columns:
            raise ValueError(
                f"Signal column '{self._ww_signal_col}' not found. "
                f"Available: {list(df.columns)}"
            )
        df["concentration"] = pd.to_numeric(
            df[self._ww_signal_col], errors="coerce"
        ).clip(lower=0)

        # Site identifier for sewershed count in aggregation
        site_col = next(
            (c for c in ("County (City/Utility)", "Abbreviated Name") if c in df.columns),
            None,
        )
        df["county_site"] = df[site_col] if site_col else "unknown"

        # County name → FIPS (internal join key used by all downstream stages)
        if "county_name" not in df.columns:
            raise ValueError("Column 'county_name' missing — ensure _load_ca_ww_csv renamed 'County'.")
        df[COUNTY_COL] = df["county_name"].map(BAY_AREA_FIPS)

        # Drop rows with missing critical values
        before = len(df)
        df = df.dropna(subset=[NWSS_DATE_COL, "concentration", COUNTY_COL]).copy()
        dropped = before - len(df)
        if dropped:
            logger.debug("_clean_ca_ww: dropped {} rows (null date/signal/county).", dropped)

        logger.info(
            "CA WW cleaned: {} rows, {} counties.",
            len(df), df[COUNTY_COL].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Stage CA-2: Aggregate multiple sites → county-day (median, no pop weights)
    # ------------------------------------------------------------------

    def _aggregate_ca_to_county_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median concentration per county-day across all reporting sites.

        CA WW data has no population_served column, so we use a simple
        median across sites rather than a population-weighted mean.
        inhibition_flag is set to False (no such QC column in CA data).
        """
        agg = (
            df.groupby([COUNTY_COL, NWSS_DATE_COL], sort=True)
            .agg(
                concentration=("concentration", "median"),
                sewershed_count=("county_site", "nunique"),
            )
            .reset_index()
        )
        agg["inhibition_flag"] = False

        n_missing = agg["concentration"].isna().sum()
        if n_missing:
            logger.debug(
                "CA aggregation: {} county-days with no valid concentration.", n_missing
            )
        return agg
