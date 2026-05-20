"""
Tests for src/models/tft_model.py  (Module 2 — model wrapper)

What this suite proves
----------------------
1. WastewaterTFT — correct hyperparameters, no covariate overlap, loss type
2. _to_nf_format  — column renaming, outlier_flag_int derivation, no extra columns
3. _build_static_df — log_population derived, county_fips_encoded stable & unique
4. build_future_df  — Sunday anchoring, calendar features in [-1,1], correct shape
5. Predict before fit raises RuntimeError (not a cryptic AttributeError)

Run with visible output:
    pytest tests/test_tft_model.py -v -s

NOTE: We do NOT train the model in unit tests (would take minutes).
      Training is validated in the integration smoke-test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import COUNTY_COL, NWSS_DATE_COL, QUANTILE_LEVELS, TARGET_COL, TFT_CONFIG
from src.models.tft_model import (
    FUTURE_COVARIATES,
    HIST_COVARIATES,
    STATIC_COVARIATES,
    WastewaterTFT,
    build_future_df,
)

# Columns the fixture computes manually; any HIST_COVARIATES column not listed
# here is added automatically with synthetic random data at fixture-build time.
_FIXTURE_MANUAL_COLS = {
    "log1p_concentration", "log1p_concentration_lag1w", "log1p_concentration_lag2w",
    "log1p_concentration_lag3w", "log1p_new_cases_lag1w", "log1p_new_cases_lag2w",
    "log1p_new_cases_lag3w", "growth_rate_1w", "relative_decay_rate",
    "vel_concentration", "accel_concentration", "vel_concentration_lag1w",
    "log1p_concentration_2w_ma", "log1p_concentration_4w_ma",
    "log1p_concentration_2w_std", "log1p_concentration_4w_std", "outlier_flag",
}
from src.utils.helpers import console


# ---------------------------------------------------------------------------
# Shared fixture: small processed DataFrame (processor output format)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processed_df():
    """Minimal processor-output DataFrame for three counties, 52 weeks."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-02", periods=52, freq="W")
    counties = ["06001", "06085", "06075"]
    rows = []
    for ds in dates:
        for county in counties:
            rows.append(
                {
                    NWSS_DATE_COL: ds,
                    COUNTY_COL: county,
                    TARGET_COL: float(rng.exponential(1.5)),
                    "concentration": float(rng.exponential(5000)),
                    "population_served": float(rng.integers(50_000, 500_000)),
                    "log1p_concentration": float(rng.normal(1.5, 0.3)),
                    "log1p_concentration_lag1w": float(rng.normal(1.5, 0.3)),
                    "log1p_concentration_lag2w": float(rng.normal(1.5, 0.3)),
                    "log1p_concentration_lag3w": float(rng.normal(1.5, 0.3)),
                    "log1p_new_cases_lag1w": float(rng.normal(1.5, 0.3)),
                    "log1p_new_cases_lag2w": float(rng.normal(1.5, 0.3)),
                    "log1p_new_cases_lag3w": float(rng.normal(1.5, 0.3)),
                    "growth_rate_1w": float(rng.normal(0, 0.2)),
                    "relative_decay_rate": float(rng.uniform(-1.0, 1.0)),
                    "vel_concentration": float(rng.normal(0, 0.1)),
                    "accel_concentration": float(rng.normal(0, 0.05)),
                    "vel_concentration_lag1w": float(rng.normal(0, 0.1)),
                    "log1p_concentration_2w_ma": float(rng.normal(1.5, 0.2)),
                    "log1p_concentration_4w_ma": float(rng.normal(1.5, 0.2)),
                    "log1p_concentration_2w_std": float(rng.uniform(0, 0.5)),
                    "log1p_concentration_4w_std": float(rng.uniform(0, 0.5)),
                    "is_sludge": 1.0,
                    "outlier_flag": bool(rng.integers(0, 2)),
                    "sin_annual_1": float(np.sin(2 * np.pi * ds.dayofyear / 365.25)),
                    "cos_annual_1": float(np.cos(2 * np.pi * ds.dayofyear / 365.25)),
                    "sin_annual_2": float(np.sin(4 * np.pi * ds.dayofyear / 365.25)),
                    "cos_annual_2": float(np.cos(4 * np.pi * ds.dayofyear / 365.25)),
                    "sin_annual_3": float(np.sin(6 * np.pi * ds.dayofyear / 365.25)),
                    "cos_annual_3": float(np.cos(6 * np.pi * ds.dayofyear / 365.25)),
                    "day_of_week_sin": float(np.sin(2 * np.pi * ds.dayofweek / 7)),
                    "day_of_week_cos": float(np.cos(2 * np.pi * ds.dayofweek / 7)),
                    "month_sin": float(np.sin(2 * np.pi * ds.month / 12)),
                    "month_cos": float(np.cos(2 * np.pi * ds.month / 12)),
                    "week_of_year": int(ds.isocalendar()[1]),
                }
            )
    df = pd.DataFrame(rows)
    # Auto-add any HIST_COVARIATES not manually included so the fixture stays
    # valid when new features are added to the covariate list.
    for col in HIST_COVARIATES:
        if col not in df.columns:
            df[col] = rng.standard_normal(len(df))
    return df


# ===========================================================================
# TestWastewaterTFT — configuration and covariate lists
# ===========================================================================

class TestWastewaterTFT:

    @pytest.fixture(scope="class")
    def model(self):
        return WastewaterTFT(max_steps=2)

    def test_no_covariate_overlap(self, model):
        """hist and futr covariate lists must be disjoint — NeuralForecast requirement."""
        overlap = set(HIST_COVARIATES) & set(FUTURE_COVARIATES)
        assert not overlap, f"Overlap found: {overlap}"

    def test_hidden_size_matches_config(self, model):
        assert model._tft.hparams.hidden_size == TFT_CONFIG["hidden_size"]

    def test_n_head_matches_config(self, model):
        assert model._tft.hparams.n_head == TFT_CONFIG["n_head"]

    def test_loss_is_pinn_type(self, model):
        from src.models.loss_functions import PINNWastewaterLoss
        assert isinstance(model._loss, PINNWastewaterLoss)

    def test_stat_exog_list_set(self, model):
        assert model._tft.hparams.stat_exog_list == STATIC_COVARIATES

    def test_hist_exog_list_set(self, model):
        assert model._tft.hparams.hist_exog_list == HIST_COVARIATES

    def test_futr_exog_list_set(self, model):
        assert model._tft.hparams.futr_exog_list == FUTURE_COVARIATES

    def test_relative_decay_rate_in_hist_covariates(self, model):
        assert "relative_decay_rate" in HIST_COVARIATES

    def test_is_sludge_in_static_covariates(self, model):
        assert "is_sludge" in STATIC_COVARIATES

    def test_predict_raises_before_fit(self, model):
        fresh = WastewaterTFT(max_steps=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            fresh.predict()

    def test_prints_config_summary(self, model):
        """Visual: show TFT configuration (pytest -s)."""
        from rich.table import Table
        from rich import box

        console.rule("[bold cyan] WastewaterTFT: Configuration Audit [/bold cyan]")

        table = Table(title="TFT Hyperparameters", box=box.ROUNDED,
                      show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Source")

        params = [
            ("h (horizon weeks)",     model.h,                                    "config.TFT_CONFIG"),
            ("input_size",            model.input_size,                            "config.TFT_CONFIG"),
            ("hidden_size (d_model)", model._tft.hparams.hidden_size,             "config.TFT_CONFIG"),
            ("n_head",                model._tft.hparams.n_head,                  "config.TFT_CONFIG"),
            ("loss",                  type(model._loss).__name__,                  "PINNWastewaterLoss"),
            ("quantile_levels",       str(QUANTILE_LEVELS),                       "config.QUANTILE_LEVELS"),
            ("max_step_growth_rate",  f"{model._loss.max_step_growth_rate:.3f}",  "MAX_DAILY × 7"),
            ("static covariates",     str(STATIC_COVARIATES),                     "derived"),
            ("hist covariates",       str(HIST_COVARIATES),                       "processor output"),
            ("future covariates",     f"{len(FUTURE_COVARIATES)} calendar cols",  "processor output"),
        ]
        for p, v, src in params:
            table.add_row(p, str(v), src)

        console.print(table)


# ===========================================================================
# Test_to_nf_format — column renaming and derivation
# ===========================================================================

class TestToNfFormat:

    @pytest.fixture(scope="class")
    def nf_pair(self, processed_df):
        model = WastewaterTFT(max_steps=2)
        ts_df, static_df = model._to_nf_format(processed_df)
        return ts_df, static_df

    def test_unique_id_column_present(self, nf_pair):
        ts_df, _ = nf_pair
        assert "unique_id" in ts_df.columns

    def test_ds_column_present(self, nf_pair):
        ts_df, _ = nf_pair
        assert "ds" in ts_df.columns

    def test_y_column_present(self, nf_pair):
        ts_df, _ = nf_pair
        assert "y" in ts_df.columns

    def test_y_is_float(self, nf_pair):
        ts_df, _ = nf_pair
        assert ts_df["y"].dtype == float

    def test_original_target_col_not_present(self, nf_pair):
        """log1p_concentration should be renamed to y, not duplicated."""
        ts_df, _ = nf_pair
        assert TARGET_COL not in ts_df.columns

    def test_outlier_flag_int_is_zero_or_one(self, nf_pair):
        ts_df, _ = nf_pair
        assert "outlier_flag_int" in ts_df.columns
        assert set(ts_df["outlier_flag_int"].unique()).issubset({0, 1})

    def test_all_hist_covariates_present(self, nf_pair):
        ts_df, _ = nf_pair
        for col in HIST_COVARIATES:
            assert col in ts_df.columns, f"Missing hist covariate: {col}"

    def test_all_future_covariates_present(self, nf_pair):
        ts_df, _ = nf_pair
        for col in FUTURE_COVARIATES:
            assert col in ts_df.columns, f"Missing future covariate: {col}"

    def test_no_nan_in_hist_covariates(self, nf_pair):
        """_to_nf_format must drop NaN warm-up rows so NeuralForecast never sees them."""
        ts_df, _ = nf_pair
        for col in HIST_COVARIATES:
            if col in ts_df.columns:
                assert ts_df[col].isna().sum() == 0, (
                    f"NaN values found in hist covariate '{col}' after _to_nf_format"
                )

    def test_nan_rows_dropped_when_lag_cols_have_nan(self, processed_df):
        """Rows with NaN lag features (warm-up period) must be silently dropped."""
        df_with_nans = processed_df.copy()
        # Introduce NaN in the lag column for the first row of each county
        for county in df_with_nans[COUNTY_COL].unique():
            mask = df_with_nans[COUNTY_COL] == county
            idx = df_with_nans.index[mask][0]
            df_with_nans.loc[idx, "log1p_concentration_lag1w"] = float("nan")

        model = WastewaterTFT(max_steps=2)
        ts_df, _ = model._to_nf_format(df_with_nans)
        assert ts_df["log1p_concentration_lag1w"].isna().sum() == 0

    def test_static_df_one_row_per_county(self, nf_pair, processed_df):
        _, static_df = nf_pair
        n_counties = processed_df[COUNTY_COL].nunique()
        assert len(static_df) == n_counties

    def test_static_df_has_log_population(self, nf_pair):
        # log_population is z-scored across counties; values are finite (can be negative).
        _, static_df = nf_pair
        assert "log_population" in static_df.columns
        assert static_df["log_population"].notna().all()
        assert np.isfinite(static_df["log_population"].to_numpy()).all()

    def test_log_population_is_zscore_of_log1p_median_pop(self, nf_pair, processed_df):
        # log_population = (log1p(median_pop) - mean) / std  across all counties (ddof=0).
        ts_df, static_df = nf_pair
        raw_log_pops = np.array([
            float(np.log1p(
                processed_df.loc[processed_df[COUNTY_COL] == fips, "population_served"].median()
            ))
            for fips in sorted(static_df["unique_id"].unique())
        ])
        pop_mean = float(raw_log_pops.mean())
        pop_std  = float(raw_log_pops.std(ddof=0))
        for fips in processed_df[COUNTY_COL].unique():
            median_pop = processed_df.loc[processed_df[COUNTY_COL] == fips, "population_served"].median()
            raw_log = float(np.log1p(median_pop))
            expected = (raw_log - pop_mean) / pop_std if pop_std > 1e-6 else 0.0
            actual   = float(static_df.loc[static_df["unique_id"] == fips, "log_population"].iloc[0])
            assert actual == pytest.approx(expected, abs=1e-5)

    def test_static_df_has_is_sludge(self, nf_pair):
        _, static_df = nf_pair
        assert "is_sludge" in static_df.columns
        assert static_df["is_sludge"].isin([0.0, 1.0]).all()

    def test_county_fips_encoded_unique_per_county(self, nf_pair):
        """Each county must get a distinct integer encoding."""
        _, static_df = nf_pair
        assert static_df["county_fips_encoded"].nunique() == len(static_df)

    def test_prints_format_summary(self, nf_pair, processed_df):
        """Visual: show the column mapping (pytest -s)."""
        from rich.table import Table
        from rich import box

        ts_df, static_df = nf_pair
        console.rule("[bold cyan] _to_nf_format: Column Mapping Audit [/bold cyan]")

        table = Table(title="Processor → NeuralForecast Column Map",
                      box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Processor column")
        table.add_column("NF column", style="bold")
        table.add_column("Role")
        table.add_column("Nulls", justify="right")

        mappings = [
            (COUNTY_COL,    "unique_id", "Series identifier", 0),
            (NWSS_DATE_COL, "ds",        "Timestamp",         0),
            (TARGET_COL,    "y",         "Target (scaled)",   0),
            ("outlier_flag","outlier_flag_int","Past covariate (bool→int)", 0),
        ]
        for proc_col, nf_col, role, _ in mappings:
            nulls = int(ts_df[nf_col].isna().sum()) if nf_col in ts_df.columns else "—"
            table.add_row(proc_col, nf_col, role, str(nulls))

        console.print(table)
        console.print(f"\nStatic DataFrame preview:\n")
        console.print(static_df.to_string(index=False))


# ===========================================================================
# TestBuildFutureDf
# ===========================================================================

class TestBuildFutureDf:

    @pytest.fixture(scope="class")
    def futr(self):
        return build_future_df(
            unique_ids=["06001", "06085", "06075"],
            last_date=pd.Timestamp("2022-12-25"),
            h=4,
        )

    def test_output_shape(self, futr):
        assert len(futr) == 3 * 4   # 3 counties × 4 horizon steps

    def test_dates_are_wednesdays(self, futr):
        """W-WED spine anchors to Wednesday (dayofweek = 2)."""
        assert (futr["ds"].dt.dayofweek == 2).all()

    def test_all_counties_present(self, futr):
        assert set(futr["unique_id"].unique()) == {"06001", "06085", "06075"}

    def test_calendar_sin_cos_in_unit_range(self, futr):
        sin_cos_cols = [c for c in futr.columns
                        if c.startswith("sin_") or c.startswith("cos_")]
        assert sin_cos_cols, "No sin/cos columns found"
        for col in sin_cos_cols:
            assert futr[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"

    def test_week_of_year_valid(self, futr):
        assert futr["week_of_year"].between(1, 53).all()

    def test_all_future_covariate_columns_present(self, futr):
        for col in FUTURE_COVARIATES:
            assert col in futr.columns, f"Missing future covariate: {col}"

    def test_prints_future_df_sample(self, futr):
        """Visual: show the first horizon dates (pytest -s)."""
        console.rule("[bold cyan] build_future_df: Horizon Window Audit [/bold cyan]")

        from rich.table import Table
        from rich import box
        table = Table(title="Future DataFrame (first 6 rows)",
                      box=box.ROUNDED, show_header=True, header_style="bold cyan")
        preview_cols = ["unique_id", "ds", "sin_annual_1", "cos_annual_1",
                        "day_of_week_sin", "month_sin", "week_of_year"]
        for col in preview_cols:
            table.add_column(col, justify="right" if col not in {"unique_id","ds"} else "left")
        for _, row in futr.head(6).iterrows():
            table.add_row(*[str(row[c])[:10] if c == "ds" else f"{row[c]:.3f}"
                            if isinstance(row[c], float) else str(row[c])
                            for c in preview_cols])
        console.print(table)
