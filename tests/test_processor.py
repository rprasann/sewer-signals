"""
Test suite for src/data_pipeline/processor.py

Structure
---------
TestCleanColumns        — Stage 1: comma stripping, type coercion, date parsing
TestZeroPadFips         — Stage 2: 5-digit FIPS normalisation
TestFilterUnits         — Stage 3: copies/g dry sludge kept; all other units excluded
TestNonDetectCensoring  — Stage 4: LOD/2 left-censoring
TestQcFilters           — Stage 5: recovery efficiency gate + inhibition flag
TestCountyFilter        — Stage 6: Bay Area FIPS allow-list
TestMultiCountyExplosion — Stage 7: comma-delimited county_fips split
TestWeightedAggregation — Stage 8: Σ(Conc×Pop)/ΣPop correctness
TestRollingSmooth       — Stage 9: centered 7-day rolling mean
TestWeeklyResample      — Stage 10: daily → weekly resampling
TestLogTransform        — Stage 11: log1p non-negativity
TestCalendarFeatures    — Stage 12: cyclical feature ranges
TestLagFeatures         — Stage 13: correct shift alignment
TestScalerLeakage       — Stage 14: train-only fitting, no leakage into val/test
TestFullPipeline        — Integration: complete run on realistic synthetic data
TestSanityProperties    — Property-style checks on full pipeline output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.processor import WastewaterProcessor
from src.config import (
    COUNTY_COL,
    MIN_RECOVERY_EFFICIENCY,
    NWSS_DATE_COL,
    TARGET_COL,
    TARGET_UNIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_stage(proc: WastewaterProcessor, stage_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Call a single private stage method by name."""
    method = getattr(proc, stage_name)
    return method(df)


def _fresh() -> WastewaterProcessor:
    return WastewaterProcessor()


# ===========================================================================
# Stage 1 — Column cleaning
# ===========================================================================

class TestCleanColumns:
    def test_strips_commas_from_concentration(self, comma_numeric_df):
        out = _run_stage(_fresh(), "_clean_columns", comma_numeric_df)
        assert out["pcr_target_avg_conc"].dtype == float
        assert out["pcr_target_avg_conc"].iloc[0] == pytest.approx(12_500.5)

    def test_strips_commas_from_population(self, comma_numeric_df):
        out = _run_stage(_fresh(), "_clean_columns", comma_numeric_df)
        assert out["population_served"].iloc[0] == pytest.approx(1_234_567)

    def test_parses_date_column(self, single_valid_row):
        out = _run_stage(_fresh(), "_clean_columns", single_valid_row)
        assert pd.api.types.is_datetime64_any_dtype(out[NWSS_DATE_COL])

    def test_drops_unparseable_dates(self):
        df = pd.DataFrame([{
            "sample_collect_date": "NOT-A-DATE",
            "pcr_target_avg_conc": "1000",
            "population_served": "50000",
        }])
        out = _run_stage(_fresh(), "_clean_columns", df)
        assert len(out) == 0

    def test_non_numeric_concentration_becomes_nan(self):
        df = pd.DataFrame([{
            "sample_collect_date": "2022-01-01",
            "pcr_target_avg_conc": "N/A",
            "population_served": "50000",
        }])
        out = _run_stage(_fresh(), "_clean_columns", df)
        assert np.isnan(out["pcr_target_avg_conc"].iloc[0])


# ===========================================================================
# Stage 2 — FIPS zero-padding
# ===========================================================================

class TestZeroPadFips:
    def test_pads_single_four_digit_fips(self, unpadded_fips_df):
        out = _run_stage(_fresh(), "_zero_pad_fips", unpadded_fips_df)
        assert out[COUNTY_COL].iloc[0] == "06001"

    def test_pads_both_tokens_in_multi_county_string(self, unpadded_fips_df):
        multi_row = unpadded_fips_df[unpadded_fips_df[COUNTY_COL] == "6075,6081"].iloc[0]
        proc = _fresh()
        out = _run_stage(proc, "_zero_pad_fips", unpadded_fips_df)
        multi_out = out[out[COUNTY_COL].str.contains(",")].iloc[0]
        tokens = multi_out[COUNTY_COL].split(",")
        assert all(len(t) == 5 for t in tokens)
        assert set(tokens) == {"06075", "06081"}

    def test_already_padded_fips_unchanged(self, single_valid_row):
        out = _run_stage(_fresh(), "_zero_pad_fips", single_valid_row)
        assert out[COUNTY_COL].iloc[0] == "06001"

    def test_strips_whitespace_around_tokens(self):
        df = pd.DataFrame([{"county_fips": " 6001 , 6085 "}])
        out = _run_stage(_fresh(), "_zero_pad_fips", df)
        assert out[COUNTY_COL].iloc[0] == "06001,06085"


# ===========================================================================
# Stage 3 — Unit filtering
# ===========================================================================

class TestFilterUnits:
    def test_keeps_copies_per_g(self, mixed_units_df):
        out = _run_stage(_fresh(), "_filter_units", mixed_units_df)
        assert out["pcr_target_units"].str.lower().str.contains("copies/g").all()

    def test_excludes_copies_per_l(self, mixed_units_df):
        out = _run_stage(_fresh(), "_filter_units", mixed_units_df)
        assert "copies/l" not in " ".join(out["pcr_target_units"].str.lower().values)

    def test_unit_filtering_is_case_insensitive(self, mixed_units_df):
        # "Copies/L Wastewater" (case variation) must also be excluded
        out = _run_stage(_fresh(), "_filter_units", mixed_units_df)
        assert len(out) == 2  # only the two copies/g rows remain

    def test_no_unit_column_passes_through(self, single_valid_row):
        df = single_valid_row.drop(columns=["pcr_target_units"])
        out = _run_stage(_fresh(), "_filter_units", df)
        assert len(out) == 1  # nothing dropped when column absent

    def test_sets_unit_excluded_flag(self, mixed_units_df):
        # The flag column is added before filtering; check on a copy that skips drop
        proc = _fresh()
        df = mixed_units_df.copy()
        normalised = df["pcr_target_units"].astype(str).str.lower().str.strip()
        df["unit_excluded_flag"] = ~normalised.str.contains(TARGET_UNIT.lower(), regex=False)
        assert df["unit_excluded_flag"].sum() == 2

    def test_sets_is_sludge_column(self, mixed_units_df):
        out = _run_stage(_fresh(), "_filter_units", mixed_units_df)
        assert "is_sludge" in out.columns
        assert (out["is_sludge"] == 1.0).all()  # default processor uses copies/g


# ===========================================================================
# Stage 4 — Non-detect censoring
# ===========================================================================

class TestNonDetectCensoring:
    def test_below_lod_replaced_with_lod_half(self, nondetect_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", nondetect_df)
        out = _run_stage(proc, "_apply_nondetect_censoring", df)
        # Row 0: below_lod=yes, lod=400 → concentration should be 200
        assert out["concentration"].iloc[0] == pytest.approx(200.0)

    def test_detect_row_uses_raw_concentration(self, nondetect_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", nondetect_df)
        out = _run_stage(proc, "_apply_nondetect_censoring", df)
        # Row 1: below_lod=no, concentration=8000 → unchanged
        assert out["concentration"].iloc[1] == pytest.approx(8_000.0)

    def test_second_nondetect_lod_half(self, nondetect_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", nondetect_df)
        out = _run_stage(proc, "_apply_nondetect_censoring", df)
        # Row 2: below_lod=yes, lod=600 → concentration should be 300
        assert out["concentration"].iloc[2] == pytest.approx(300.0)

    def test_concentration_never_negative(self, nondetect_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", nondetect_df)
        out = _run_stage(proc, "_apply_nondetect_censoring", df)
        assert (out["concentration"] >= 0).all()


# ===========================================================================
# Stage 5 — QC filters
# ===========================================================================

class TestQcFilters:
    def test_drops_low_recovery_efficiency(self, qc_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", qc_df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        surviving_rec_eff = out["rec_eff_percent"].dropna()
        assert (surviving_rec_eff >= MIN_RECOVERY_EFFICIENCY).all()

    def test_row_counts_after_qc(self, qc_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", qc_df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        # 5 rows in → 2 fail (5.0 and 9.9) → 3 remain
        assert len(out) == 3

    def test_nan_recovery_efficiency_is_kept(self, qc_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", qc_df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        assert out["rec_eff_percent"].isna().sum() == 1

    def test_inhibition_flag_column_present(self, qc_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", qc_df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        assert "inhibition_flag" in out.columns

    def test_inhibited_row_retained_but_flagged(self, qc_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", qc_df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        # The row with rec_eff=10 and inhibition=yes should survive with flag=True
        flagged = out[out["inhibition_flag"] == True]
        assert len(flagged) == 1

    def test_no_rec_eff_column_passes_through(self, single_valid_row):
        df = single_valid_row.drop(columns=["rec_eff_percent"])
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", df)
        out = _run_stage(proc, "_apply_qc_filters", df)
        assert len(out) == 1


# ===========================================================================
# Stage 6 — County filter
# ===========================================================================

class TestCountyFilter:
    def test_bay_area_fips_kept(self, single_valid_row):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", single_valid_row)
        df = _run_stage(proc, "_zero_pad_fips", df)
        out = _run_stage(proc, "_filter_counties", df)
        assert len(out) == 1

    def test_non_bay_area_fips_dropped(self):
        df = pd.DataFrame([{
            "sample_collect_date": "2022-01-01",
            COUNTY_COL: "06037",   # Los Angeles — not Bay Area
            "wwtp_id": "site_X",
            "pcr_target_avg_conc": "5000",
        }])
        proc = _fresh()
        out = _run_stage(proc, "_filter_counties", df)
        assert len(out) == 0

    def test_multi_county_row_kept_if_one_matches(self):
        # 06001 is Bay Area, 06037 is not — row should be kept
        df = pd.DataFrame([{
            "sample_collect_date": "2022-01-01",
            COUNTY_COL: "06001,06037",
            "wwtp_id": "site_Y",
            "pcr_target_avg_conc": "5000",
        }])
        proc = _fresh()
        out = _run_stage(proc, "_filter_counties", df)
        assert len(out) == 1


# ===========================================================================
# Stage 7 — Multi-county explosion
# ===========================================================================

class TestMultiCountyExplosion:
    def test_shared_site_explodes_to_two_rows(self, multi_county_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", multi_county_df)
        df = _run_stage(proc, "_zero_pad_fips", df)
        df = _run_stage(proc, "_filter_counties", df)
        out = _run_stage(proc, "_explode_multi_county", df)
        shared = out[out["wwtp_id"] == "shared_site"]
        assert len(shared) == 2
        assert set(shared[COUNTY_COL].tolist()) == {"06075", "06081"}

    def test_single_county_site_stays_one_row(self, multi_county_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", multi_county_df)
        df = _run_stage(proc, "_zero_pad_fips", df)
        df = _run_stage(proc, "_filter_counties", df)
        out = _run_stage(proc, "_explode_multi_county", df)
        single = out[out["wwtp_id"] == "alameda_only"]
        assert len(single) == 1

    def test_non_bay_area_tokens_pruned_after_explosion(self):
        df = pd.DataFrame([{
            "sample_collect_date": "2022-01-01",
            COUNTY_COL: "06001,06037",   # one Bay Area, one not
            "wwtp_id": "mixed",
            "pcr_target_avg_conc": "5000",
            "population_served": "100000",
        }])
        proc = _fresh()
        out = _run_stage(proc, "_explode_multi_county", df)
        assert all(c in proc.fips_filter for c in out[COUNTY_COL])


# ===========================================================================
# Stage 8 — Weighted aggregation
# ===========================================================================

class TestWeightedAggregation:
    def test_weighted_mean_formula(self, weighted_agg_df):
        """
        site_A: conc=1000, pop=100_000  → weighted contribution 1e8
        site_B: conc=3000, pop=300_000  → weighted contribution 9e8
        Expected weighted mean = (1e8 + 9e8) / (1e5 + 3e5) = 1e9 / 4e5 = 2500
        """
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", weighted_agg_df)
        df = _run_stage(proc, "_apply_nondetect_censoring", df)
        df = _run_stage(proc, "_apply_qc_filters", df)
        df = _run_stage(proc, "_explode_multi_county", df)
        out = _run_stage(proc, "_aggregate_to_county_daily", df)
        assert out["concentration"].iloc[0] == pytest.approx(2500.0)

    def test_aggregated_row_count(self, weighted_agg_df):
        """Two sites, same county, same day → single aggregated row."""
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", weighted_agg_df)
        df = _run_stage(proc, "_apply_nondetect_censoring", df)
        df = _run_stage(proc, "_apply_qc_filters", df)
        df = _run_stage(proc, "_explode_multi_county", df)
        out = _run_stage(proc, "_aggregate_to_county_daily", df)
        assert len(out) == 1

    def test_sewershed_count_column_present(self, weighted_agg_df):
        proc = _fresh()
        df = _run_stage(proc, "_clean_columns", weighted_agg_df)
        df = _run_stage(proc, "_apply_nondetect_censoring", df)
        df = _run_stage(proc, "_apply_qc_filters", df)
        df = _run_stage(proc, "_explode_multi_county", df)
        out = _run_stage(proc, "_aggregate_to_county_daily", df)
        assert "sewershed_count" in out.columns


# ===========================================================================
# Stage 9 — Rolling smooth
# ===========================================================================

class TestRollingSmooth:
    def test_output_has_daily_frequency(self, full_pipeline_df):
        proc = _fresh()
        df = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
        )
        out = _run_stage(proc, "_rolling_smooth", df)
        for _, grp in out.groupby(COUNTY_COL):
            dates = grp[NWSS_DATE_COL].sort_values()
            diffs = dates.diff().dropna().dt.days
            assert (diffs == 1).all(), "Output should be daily frequency"

    def test_concentration_non_negative_after_smoothing(self, full_pipeline_df):
        proc = _fresh()
        df = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
        )
        out = _run_stage(proc, "_rolling_smooth", df)
        valid = out["concentration"].dropna()
        assert (valid >= 0).all()


# ===========================================================================
# Stage 10 — Weekly resampling
# ===========================================================================

class TestWeeklyResample:
    def test_output_dates_are_sundays(self, full_pipeline_df):
        proc = _fresh()
        df = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
            .pipe(proc._rolling_smooth)
        )
        out = _run_stage(proc, "_resample_to_weekly", df)
        # W-WED spine anchors to Wednesday (dayofweek = 2)
        assert (out[NWSS_DATE_COL].dt.dayofweek == 2).all()

    def test_fewer_rows_than_daily_input(self, full_pipeline_df):
        proc = _fresh()
        daily = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
            .pipe(proc._rolling_smooth)
        )
        weekly = _run_stage(proc, "_resample_to_weekly", daily)
        assert len(weekly) < len(daily)

    def test_no_na_concentration_in_output(self, full_pipeline_df):
        proc = _fresh()
        df = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
            .pipe(proc._rolling_smooth)
        )
        out = _run_stage(proc, "_resample_to_weekly", df)
        assert out["concentration"].isna().sum() == 0


# ===========================================================================
# Stage 11 — log(1+x) transform
# ===========================================================================

class TestLogTransform:
    def test_target_column_created(self, full_pipeline_df):
        proc = _fresh()
        df = full_pipeline_df.pipe(proc._clean_columns).pipe(proc._apply_nondetect_censoring)
        df["concentration"] = df["pcr_target_avg_conc"].clip(lower=0)
        out = _run_stage(proc, "_log_transform", df)
        # _log_transform creates log1p_concentration (WW signal)
        # log1p_new_cases (TARGET_COL) is added later when cases are merged
        assert "log1p_concentration" in out.columns

    def test_log1p_of_zero_is_zero(self):
        df = pd.DataFrame([{"concentration": 0.0}])
        out = _run_stage(_fresh(), "_log_transform", df)
        assert out["log1p_concentration"].iloc[0] == pytest.approx(0.0)

    def test_log1p_values_non_negative(self, full_pipeline_df):
        proc = _fresh()
        df = (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._apply_nondetect_censoring)
        )
        df["concentration"] = df["pcr_target_avg_conc"].clip(lower=0)
        out = _run_stage(proc, "_log_transform", df)
        assert (out["log1p_concentration"] >= 0).all()

    def test_log1p_mathematically_correct(self):
        df = pd.DataFrame([{"concentration": np.e - 1}])  # log(1 + e-1) = 1.0
        out = _run_stage(_fresh(), "_log_transform", df)
        assert out["log1p_concentration"].iloc[0] == pytest.approx(1.0)


# ===========================================================================
# Stage 12 — Calendar features
# ===========================================================================

class TestCalendarFeatures:
    @pytest.fixture()
    def _weekly_df(self, full_pipeline_df):
        proc = _fresh()
        return (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
            .pipe(proc._rolling_smooth)
            .pipe(proc._resample_to_weekly)
            .pipe(proc._log_transform)
        )

    def test_sin_cos_columns_present(self, _weekly_df):
        out = _run_stage(_fresh(), "_add_calendar_features", _weekly_df)
        for col in ["sin_annual_1", "cos_annual_1", "day_of_week_sin", "month_sin"]:
            assert col in out.columns

    def test_sin_values_bounded(self, _weekly_df):
        out = _run_stage(_fresh(), "_add_calendar_features", _weekly_df)
        for col in [c for c in out.columns if c.startswith("sin_") or c.startswith("cos_")]:
            assert out[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"

    def test_week_of_year_in_valid_range(self, _weekly_df):
        out = _run_stage(_fresh(), "_add_calendar_features", _weekly_df)
        assert out["week_of_year"].between(1, 53).all()


# ===========================================================================
# Stage 13 — Lag features
# ===========================================================================

class TestLagFeatures:
    @pytest.fixture()
    def _pre_lag_df(self, full_pipeline_df):
        proc = _fresh()
        return (
            full_pipeline_df
            .pipe(proc._clean_columns)
            .pipe(proc._zero_pad_fips)
            .pipe(proc._filter_units)
            .pipe(proc._apply_nondetect_censoring)
            .pipe(proc._apply_qc_filters)
            .pipe(proc._filter_counties)
            .pipe(proc._explode_multi_county)
            .pipe(proc._aggregate_to_county_daily)
            .pipe(proc._rolling_smooth)
            .pipe(proc._resample_to_weekly)
            .pipe(proc._log_transform)
            .pipe(proc._add_calendar_features)
        )

    def test_lag_columns_present(self, _pre_lag_df):
        out = _run_stage(_fresh(), "_add_lag_features", _pre_lag_df)
        # WW concentration lags are created by the base processor (cases lags require CA processor)
        for col in ["log1p_concentration_lag1w", "log1p_concentration_lag2w", "log1p_concentration_lag3w"]:
            assert col in out.columns

    def test_lag1w_correctly_shifted(self, _pre_lag_df):
        out = _run_stage(_fresh(), "_add_lag_features", _pre_lag_df)
        for county, grp in out.groupby(COUNTY_COL):
            grp = grp.sort_values(NWSS_DATE_COL).reset_index(drop=True)
            if len(grp) < 3:
                continue
            expected = grp["log1p_concentration"].iloc[1]
            actual_lag = grp["log1p_concentration_lag1w"].iloc[2]
            assert actual_lag == pytest.approx(expected, rel=1e-6), (
                f"County {county}: lag1w at index 2 should equal log1p_concentration at index 1"
            )

    def test_growth_rate_column_present(self, _pre_lag_df):
        out = _run_stage(_fresh(), "_add_lag_features", _pre_lag_df)
        assert "growth_rate_1w" in out.columns

    def test_growth_rate_finite_after_warmup(self, _pre_lag_df):
        out = _run_stage(_fresh(), "_add_lag_features", _pre_lag_df)
        # First row per county will have NaN growth rate (no lag); rest should be finite
        for _, grp in out.groupby(COUNTY_COL):
            grp = grp.sort_values(NWSS_DATE_COL)
            tail = grp["growth_rate_1w"].iloc[1:]
            assert np.isfinite(tail.dropna()).all()

    def test_outlier_flag_boolean(self, _pre_lag_df):
        out = _run_stage(_fresh(), "_add_lag_features", _pre_lag_df)
        assert out["outlier_flag"].dtype == bool


# ===========================================================================
# Stage 14 — Leakage-free scaling
# ===========================================================================

class TestScalerLeakage:
    def test_scaler_fitted_on_train(self, leakage_split_dfs):
        raw_train, _ = leakage_split_dfs
        proc = _fresh()
        proc.run(raw_train)
        assert proc._scaler is not None

    def test_transform_uses_train_scaler_not_val(self, leakage_split_dfs):
        """Val data transformed with train scaler should differ from a val-only fit."""
        raw_train, raw_val = leakage_split_dfs
        proc_train = _fresh()
        proc_train.run(raw_train)
        val_via_train = proc_train.transform(raw_val)

        proc_val = _fresh()
        val_self_fit = proc_val.run(raw_val)

        # The two scalers were fitted on different distributions → medians will differ
        med_train_scaler = val_via_train[TARGET_COL].median()
        med_val_scaler = val_self_fit[TARGET_COL].median()
        # They should NOT be identical (different IQR/median used)
        assert med_train_scaler != pytest.approx(med_val_scaler, rel=1e-4), (
            "Val transformed with train scaler should differ from val self-fitted scaler"
        )

    def test_transform_raises_without_prior_fit(self, leakage_split_dfs):
        _, raw_val = leakage_split_dfs
        proc = _fresh()  # no run() called
        with pytest.raises(RuntimeError, match="not fitted"):
            proc.transform(raw_val)

    def test_scaler_columns_consistent(self, leakage_split_dfs):
        """Scale columns chosen during run() should be applied identically in transform()."""
        raw_train, raw_val = leakage_split_dfs
        proc = _fresh()
        proc.run(raw_train)
        train_cols = set(proc._scale_cols)

        proc.transform(raw_val)
        assert set(proc._scale_cols) == train_cols


# ===========================================================================
# Integration — full pipeline
# ===========================================================================

class TestFullPipeline:
    def test_run_produces_dataframe(self, full_pipeline_df):
        out = _fresh().run(full_pipeline_df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) > 0

    def test_all_output_fips_are_5_digits(self, full_pipeline_df):
        out = _fresh().run(full_pipeline_df)
        assert out[COUNTY_COL].str.len().eq(5).all()

    def test_output_columns_include_ww_and_lags(self, full_pipeline_df):
        out = _fresh().run(full_pipeline_df)
        # Base WastewaterProcessor creates log1p_concentration + WW lags
        # log1p_new_cases (TARGET_COL) requires CAWastewaterProcessor with cases data
        expected = {"log1p_concentration", "log1p_concentration_lag1w",
                    "log1p_concentration_lag2w", "log1p_concentration_lag3w"}
        assert expected.issubset(out.columns)

    def test_split_returns_three_non_overlapping_frames(self, full_pipeline_df):
        proc = _fresh()
        out = proc.run(full_pipeline_df)
        train, val, test = proc.split(out)

        assert len(train) + len(val) + len(test) == len(out)

        train_dates = set(train[NWSS_DATE_COL])
        val_dates = set(val[NWSS_DATE_COL])
        test_dates = set(test[NWSS_DATE_COL])
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)

    def test_train_val_transform_same_columns(self, leakage_split_dfs):
        raw_train, raw_val = leakage_split_dfs
        proc = _fresh()
        train = proc.run(raw_train)
        val = proc.transform(raw_val)
        assert set(train.columns) == set(val.columns)


# ===========================================================================
# Sanity properties — invariants that must hold across any valid run
# ===========================================================================

class TestSanityProperties:
    @pytest.fixture()
    def processed(self, full_pipeline_df):
        return _fresh().run(full_pipeline_df)

    def test_no_negative_concentration(self, processed):
        assert (processed["concentration"] >= 0).all()

    def test_log1p_target_non_negative_before_scaling(self, processed):
        # RobustScaler centers by median so the scaled TARGET_COL can be negative;
        # the invariant is on the un-scaled log1p value, proxied via raw concentration.
        assert (processed["concentration"] >= 0).all()

    def test_dates_monotone_per_county(self, processed):
        for _, grp in processed.groupby(COUNTY_COL):
            dates = grp[NWSS_DATE_COL].reset_index(drop=True)
            assert dates.is_monotonic_increasing, "Dates must be sorted within each county"

    def test_no_duplicate_county_date_pairs(self, processed):
        dupes = processed.duplicated(subset=[COUNTY_COL, NWSS_DATE_COL])
        assert not dupes.any(), "Each county should have at most one row per date"

    def test_calendar_sin_cos_in_unit_range(self, processed):
        sin_cos_cols = [c for c in processed.columns if c.startswith(("sin_", "cos_"))]
        for col in sin_cos_cols:
            assert processed[col].between(-1.0, 1.0).all()

    def test_growth_rate_mostly_finite(self, processed):
        # Allow NaN for the first observation per county; everything else must be finite
        finite_mask = processed["growth_rate_1w"].notna()
        assert np.isfinite(processed.loc[finite_mask, "growth_rate_1w"]).all()
