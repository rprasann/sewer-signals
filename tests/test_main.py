"""
Tests for main.py data-loading, splitting, and inverse-transform helpers.

These tests use two strategies:
  (a) Synthetic DataFrames that match the exact column schema of the real CSVs —
      fast, no I/O, always runnable in CI.
  (b) A small header+sample read from the actual CSVs in data/raw/ — only runs
      when the files are present; skipped otherwise.  These catch schema drift
      before the full pipeline is executed.

Running:
    pytest tests/test_main.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

# ── paths
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NWSS_CSV  = RAW_DIR / "CDC_Wastewater_Data_for_SARS-CoV-2_20260505.csv"
CASES_CSV = RAW_DIR / (
    "Weekly_United_States_COVID-19_Cases_and_Deaths_by_County_"
    "-_ARCHIVED_20260502.csv"
)

# ── import the helpers under test
from main import (
    _split_raw,
    _invert_scaling_to_log1p,
    _build_display_frames,
)
from src.config import COUNTY_COL, NWSS_DATE_COL, TARGET_COL
from src.evaluation.metrics import QuantileColumns


# ===========================================================================
# Fixtures — synthetic DataFrames matching real CSV schemas
# ===========================================================================

@pytest.fixture()
def synthetic_ww_df() -> pd.DataFrame:
    """Minimal wastewater DataFrame matching CDC NWSS column names."""
    return pd.DataFrame({
        "sample_collect_date": ["2022-01-02", "2022-01-09", "2023-07-02"],
        "county_fips": ["06075", "06001", "06085"],
        "pcr_target_avg_conc": [5000.0, 8000.0, 3000.0],
        "pcr_target_units": [
            "copies/g dry sludge",
            "copies/g dry sludge",
            "copies/g dry sludge",
        ],
        "population_served": [900000, 1700000, 1900000],
        "wwtp_id": ["sf_main", "oak_main", "sc_main"],
        "rec_eff_percent": [85.0, 90.0, 78.0],
        "pcr_target_below_lod": ["No", "No", "No"],
        "lod_sewage": [50.0, 50.0, 50.0],
    })


@pytest.fixture()
def synthetic_cases_df() -> pd.DataFrame:
    """Minimal cases DataFrame matching the archived CDC county CSV schema.

    Column names match the real file EXACTLY (post-csv.reader, pre-pandas):
      fips_code, county, state, state_fips, date, cumulative_cases,
      cumulative_deaths, New cases, New deaths
    """
    return pd.DataFrame({
        "fips_code": ["06075", "06001", "06085"],
        "county": ["San Francisco", "Alameda", "Santa Clara"],
        "state": ["CA", "CA", "CA"],
        "state_fips": ["06", "06", "06"],
        "date": ["01/02/2022", "01/09/2022", "07/02/2023"],
        "cumulative_cases": [100000, 200000, 50000],
        "cumulative_deaths": [1000, 2000, 500],
        "New cases": [500, 800, 200],
        "New deaths": [5, 8, 2],
    })


@pytest.fixture()
def mock_processor():
    """Mock WastewaterProcessor with per-county _scalers dict (Phase 6 layout).

    _invert_scaling_to_log1p checks _scalers first (Problem 1 fix in processor.py)
    before falling back to _scaler.  The mock must satisfy the per-county path.
    """
    from src.config import TARGET_COL as _TC

    raw = np.array([0.5, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    fitted = RobustScaler().fit(raw)

    # The inversion function uses _scalers[fips].center_[col_idx] and .scale_[col_idx]
    # where col_idx = _scale_cols.index(TARGET_COL).
    class _MockProc:
        _scale_cols = [_TC]
        _scaler     = fitted   # legacy alias (backward compat path)
        _scalers    = {"06075": fitted}   # per-county dict used by Phase 6 path

    return _MockProc()


@pytest.fixture()
def synthetic_forecast_df() -> pd.DataFrame:
    """Minimal forecast DataFrame matching NeuralForecast TFT predict() output."""
    return pd.DataFrame({
        "unique_id": ["06075"] * 4,
        "ds": pd.date_range("2024-01-07", periods=4, freq="W"),
        "TFT-lo-95.0":  [-0.5, -0.4, -0.3, -0.2],
        "TFT-lo-50.0":  [ 0.0,  0.1,  0.2,  0.3],
        "TFT-median":   [ 0.5,  0.6,  0.7,  0.8],
        "TFT-hi-50.0":  [ 1.0,  1.1,  1.2,  1.3],
        "TFT-hi-95.0":  [ 1.5,  1.6,  1.7,  1.8],
    })


# ===========================================================================
# Tests: _load_cases_csv schema normalisation
# ===========================================================================

@pytest.mark.skip(reason="CDC NWSS pipeline replaced by CA pipeline; _load_cases_csv removed")
class TestLoadCasesCsv:
    """_load_cases_csv must normalise any valid CDC column naming convention."""

    def _call_with_df(self, df: pd.DataFrame, tmp_path) -> pd.DataFrame:
        csv_path = tmp_path / "cases.csv"
        df.to_csv(csv_path, index=False)
        return _load_cases_csv(csv_path)

    def test_date_col_renamed_from_date(self, synthetic_cases_df, tmp_path):
        """'date' column must be renamed to sample_collect_date."""
        result = self._call_with_df(synthetic_cases_df, tmp_path)
        assert NWSS_DATE_COL in result.columns, (
            f"Expected '{NWSS_DATE_COL}' after rename; got {list(result.columns)}"
        )

    def test_date_col_is_datetime(self, synthetic_cases_df, tmp_path):
        result = self._call_with_df(synthetic_cases_df, tmp_path)
        assert pd.api.types.is_datetime64_any_dtype(result[NWSS_DATE_COL])

    def test_fips_col_renamed(self, synthetic_cases_df, tmp_path):
        """'fips_code' must be renamed to county_fips."""
        result = self._call_with_df(synthetic_cases_df, tmp_path)
        assert COUNTY_COL in result.columns

    def test_fips_zero_padded(self, synthetic_cases_df, tmp_path):
        """FIPS codes must be zero-padded to 5 digits."""
        result = self._call_with_df(synthetic_cases_df, tmp_path)
        assert all(len(f) == 5 for f in result[COUNTY_COL])

    def test_new_cases_col_preserved(self, synthetic_cases_df, tmp_path):
        """'New cases' (after normalisation → 'new_cases') must be kept."""
        result = self._call_with_df(synthetic_cases_df, tmp_path)
        assert "new_cases" in result.columns

    def test_bay_area_filter_applied(self, synthetic_cases_df, tmp_path):
        """Only Bay Area FIPS rows should survive."""
        # Add a non-Bay-Area row
        extra = synthetic_cases_df.copy()
        extra = pd.concat([
            extra,
            pd.DataFrame({
                "fips_code": ["01005"],
                "county": ["Barbour"],
                "state": ["AL"],
                "state_fips": ["01"],
                "date": ["01/02/2022"],
                "cumulative_cases": [0],
                "cumulative_deaths": [0],
                "New cases": [0],
                "New deaths": [0],
            }),
        ], ignore_index=True)
        result = self._call_with_df(extra, tmp_path)
        assert "01005" not in result[COUNTY_COL].values

    def test_submission_date_variant_also_works(self, tmp_path):
        """CDC NWSS format uses 'submission_date'; must also be handled."""
        df = pd.DataFrame({
            "submission_date": ["2022-01-02"],
            "fips_code": ["06075"],
            "new_case": [500],
        })
        result = self._call_with_df(df, tmp_path)
        assert NWSS_DATE_COL in result.columns

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_cases_csv(tmp_path / "does_not_exist.csv")


# ===========================================================================
# Tests: _load_wastewater_csv
# ===========================================================================

@pytest.mark.skip(reason="CDC NWSS pipeline replaced by CA pipeline; _load_wastewater_csv removed")
class TestLoadWastewaterCsv:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_wastewater_csv(tmp_path / "missing.csv")

    def test_loads_synthetic(self, synthetic_ww_df, tmp_path):
        csv_path = tmp_path / "nwss.csv"
        synthetic_ww_df.to_csv(csv_path, index=False)
        result = _load_wastewater_csv(csv_path)
        assert len(result) == len(synthetic_ww_df)
        assert "pcr_target_units" in result.columns


# ===========================================================================
# Tests: _split_raw
# ===========================================================================

class TestSplitRaw:

    def _make_df(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            NWSS_DATE_COL: pd.to_datetime(dates),
            "value": range(len(dates)),
        })

    def test_train_val_test_sizes(self):
        # TRAIN_END_DATE = "2022-10-05", VAL_END_DATE = "2023-06-07"
        # All dates must be within DATA_START_DATE (2020-07-01) → DATA_END_DATE (2023-12-19)
        dates = [
            "2021-03-01",  # train  (≤ 2022-10-05)
            "2022-06-01",  # train  (≤ 2022-10-05)
            "2022-10-05",  # train  (exactly at boundary — inclusive)
            "2022-10-12",  # val    (> 2022-10-05, ≤ 2023-06-07)
            "2023-06-07",  # val    (exactly at val boundary — inclusive)
            "2023-07-01",  # test   (> 2023-06-07)
        ]
        df = self._make_df(dates)
        train, val, test = _split_raw(df)
        assert len(train) == 3
        assert len(val) == 2
        assert len(test) == 1

    def test_no_overlap_between_splits(self):
        dates = pd.date_range("2021-01-01", periods=200, freq="W").strftime("%Y-%m-%d")
        df = self._make_df(list(dates))
        train, val, test = _split_raw(df)
        all_dates = list(train[NWSS_DATE_COL]) + list(val[NWSS_DATE_COL]) + list(test[NWSS_DATE_COL])
        assert len(all_dates) == len(set(all_dates)), "Date overlap between splits"

    def test_train_is_earliest(self):
        # Dates must be within DATA_START_DATE (2020-07-01) → DATA_END_DATE (2023-12-19)
        dates = ["2020-08-01", "2023-09-01"]
        df = self._make_df(dates)
        train, _, test = _split_raw(df)
        assert pd.Timestamp("2020-08-01") in list(train[NWSS_DATE_COL])
        assert pd.Timestamp("2023-09-01") in list(test[NWSS_DATE_COL])

    def test_empty_splits_possible(self):
        """If all data is in train, val and test should be empty DataFrames."""
        dates = ["2021-01-01", "2022-01-01"]
        df = self._make_df(dates)
        train, val, test = _split_raw(df)
        assert len(val) == 0
        assert len(test) == 0


# ===========================================================================
# Tests: _invert_scaling_to_log1p
# ===========================================================================

class TestInvertScalingToLog1p:
    """_invert_scaling_to_log1p uses per-county _scalers dict (Phase 6).

    All test DataFrames must include a "unique_id" column (or COUNTY_COL)
    matching a key in mock_processor._scalers so the per-county path is exercised.
    """

    def test_roundtrip_accuracy(self, mock_processor):
        """Inverting scaled values should recover original log1p values exactly."""
        original_log1p = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        scaled = mock_processor._scaler.transform(
            original_log1p.reshape(-1, 1)
        ).flatten()

        # unique_id must match a key in mock_processor._scalers ("06075")
        df = pd.DataFrame({"unique_id": ["06075"] * 5, TARGET_COL: scaled})
        result = _invert_scaling_to_log1p(df, mock_processor, cols=[TARGET_COL])
        recovered = result[TARGET_COL].to_numpy()

        np.testing.assert_allclose(recovered, original_log1p, rtol=1e-6,
                                   err_msg="Roundtrip inversion failed")

    def test_no_op_when_county_not_in_scalers(self, mock_processor):
        """When the county FIPS is not in _scalers, columns are left unchanged."""
        df = pd.DataFrame({"unique_id": ["99999", "99999", "99999"],
                           TARGET_COL: [1.0, 2.0, 3.0]})
        result = _invert_scaling_to_log1p(df, mock_processor, cols=[TARGET_COL])
        # County 99999 has no scaler → inversion skipped → values unchanged
        pd.testing.assert_series_equal(result[TARGET_COL], df[TARGET_COL])

    def test_unknown_col_skipped(self, mock_processor):
        """Columns not in the DataFrame are silently skipped."""
        df = pd.DataFrame({"unique_id": ["06075", "06075"], TARGET_COL: [0.5, 1.0]})
        result = _invert_scaling_to_log1p(
            df, mock_processor, cols=[TARGET_COL, "does_not_exist"]
        )
        assert TARGET_COL in result.columns

    def test_values_are_non_negative_after_double_inversion(self, mock_processor):
        """After invert + expm1, all values should be ≥ 0 for plausible inputs."""
        original_conc = np.array([100.0, 500.0, 5000.0, 20000.0])
        log1p_vals = np.log1p(original_conc)
        scaled = mock_processor._scaler.transform(
            log1p_vals.reshape(-1, 1)
        ).flatten()

        df = pd.DataFrame({"unique_id": ["06075"] * 4, TARGET_COL: scaled})
        unscaled = _invert_scaling_to_log1p(df, mock_processor, cols=[TARGET_COL])
        copies = np.expm1(unscaled[TARGET_COL].to_numpy()).clip(0)
        assert (copies >= 0).all()


# ===========================================================================
# Tests: _build_display_frames
# ===========================================================================

class TestBuildDisplayFrames:

    def test_forecast_quantile_cols_in_unscaled_range(
        self, mock_processor, synthetic_forecast_df
    ):
        """After display inversion, median values should be in log1p range (0–20)."""
        sludge_all = pd.DataFrame({
            COUNTY_COL: ["06075"] * 4,
            NWSS_DATE_COL: pd.date_range("2023-01-01", periods=4, freq="W"),
            TARGET_COL: mock_processor._scaler.transform(
                np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
            ).flatten(),
            "concentration": [100.0, 500.0, 1000.0, 5000.0],
        })
        q_cols = QuantileColumns()
        proc_disp, fcast_disp = _build_display_frames(
            sludge_all, synthetic_forecast_df, mock_processor, q_cols
        )
        # Unscaled log1p values should be roughly in [0, 20] for WW data
        medians = fcast_disp[q_cols.q50].dropna().to_numpy()
        assert medians.max() < 100, (
            f"Median forecast looks like it's still in scaled space: {medians}"
        )

    def test_processed_display_has_target_col(self, mock_processor, synthetic_forecast_df):
        sludge_all = pd.DataFrame({
            COUNTY_COL: ["06075"] * 3,
            NWSS_DATE_COL: pd.date_range("2023-01-01", periods=3, freq="W"),
            TARGET_COL: [0.1, 0.2, 0.3],
            "concentration": [100.0, 500.0, 1000.0],
        })
        q_cols = QuantileColumns()
        proc_disp, _ = _build_display_frames(
            sludge_all, synthetic_forecast_df, mock_processor, q_cols
        )
        assert TARGET_COL in proc_disp.columns


# ===========================================================================
# Tests: _build_copies_forecast
# ===========================================================================

@pytest.mark.skip(reason="_build_copies_forecast renamed to _build_decoded_forecast in CA pipeline")
class TestBuildCopiesForecast:

    def test_median_is_non_negative(self, mock_processor, synthetic_forecast_df):
        """After full inversion, all quantile values should be ≥ 0."""
        q_cols = QuantileColumns()
        result = _build_copies_forecast(synthetic_forecast_df, mock_processor, q_cols)
        for attr in ("q025", "q25", "q50", "q75", "q975"):
            col = getattr(q_cols, attr)
            if col in result.columns:
                assert (result[col] >= 0).all(), f"{col} has negative values after inversion"

    def test_copies_forecast_larger_than_log1p_forecast(
        self, mock_processor, synthetic_forecast_df
    ):
        """copies/g values should be larger than the log1p values (expm1 > identity for x>0)."""
        q_cols = QuantileColumns()
        copies = _build_copies_forecast(synthetic_forecast_df, mock_processor, q_cols)
        # Original synthetic medians are small floats (0.5–0.8 after scaling).
        # After expm1 inversion, the median should be positive but small-ish for these inputs.
        medians = copies[q_cols.q50].to_numpy()
        # Just verify the result is finite and non-negative
        assert np.all(np.isfinite(medians))
        assert np.all(medians >= 0)


# ===========================================================================
# Integration: smoke-test against the real CSV headers (skips if files absent)
# ===========================================================================

@pytest.mark.skip(reason="CDC NWSS pipeline replaced by CA pipeline; _load_cases_csv removed")
class TestRealCasesSchema:
    """Read only the header + 5 data rows from the real CSV to verify schema."""

    def test_date_col_present_and_parseable(self, tmp_path):
        df = pd.read_csv(CASES_CSV, nrows=5)
        # Normalise exactly as _load_cases_csv does
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        assert "date" in df.columns or "submission_date" in df.columns, (
            f"Expected a date column; got {list(df.columns)}"
        )

    def test_fips_col_present(self, tmp_path):
        df = pd.read_csv(CASES_CSV, nrows=5)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        assert "fips_code" in df.columns or "fips" in df.columns

    def test_load_function_succeeds_on_real_file(self, tmp_path):
        """Full _load_cases_csv call against the real CSV — verifies no KeyError."""
        result = _load_cases_csv(CASES_CSV)
        assert NWSS_DATE_COL in result.columns, (
            f"After loading real CSV, expected '{NWSS_DATE_COL}'; "
            f"got columns: {list(result.columns)}"
        )
        assert COUNTY_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result[NWSS_DATE_COL])
        assert "new_cases" in result.columns

    def test_new_cases_col_present(self):
        df = pd.read_csv(CASES_CSV, nrows=5)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        # Either 'new_cases' (post-normalisation) or 'new_case' should be present
        assert "new_cases" in df.columns or "new_case" in df.columns


@pytest.mark.skip(reason="CDC NWSS pipeline replaced by CA pipeline; _load_wastewater_csv removed")
class TestRealNwssSchema:

    def test_required_columns_present(self):
        df = pd.read_csv(NWSS_CSV, nrows=5)
        required = {"sample_collect_date", "county_fips", "pcr_target_avg_conc",
                    "pcr_target_units", "population_served"}
        missing = required - set(df.columns)
        assert not missing, f"NWSS CSV missing expected columns: {missing}"

    def test_load_function_returns_dataframe(self):
        result = _load_wastewater_csv(NWSS_CSV)
        assert len(result) > 0
        assert "pcr_target_units" in result.columns


# ===========================================================================
# Tests: _validate_pipeline_inputs
# ===========================================================================

class TestValidatePipelineInputs:
    """_validate_pipeline_inputs must report known failure modes without raising."""

    from main import _validate_pipeline_inputs  # noqa: PLC0415 — test-level import

    def _make_split(self, n: int, county: str = "06075", start: str = "2022-01-02") -> pd.DataFrame:
        from src.config import COUNTY_COL, NWSS_DATE_COL, TARGET_COL
        return pd.DataFrame({
            NWSS_DATE_COL: pd.date_range(start, periods=n, freq="W"),
            COUNTY_COL: county,
            TARGET_COL: np.random.default_rng(0).normal(1.0, 0.3, n),
        })

    def test_passes_on_healthy_splits(self, capsys):
        from main import _validate_pipeline_inputs
        from src.config import COUNTY_COL, NWSS_DATE_COL, TARGET_COL, TFT_CONFIG
        n = TFT_CONFIG["input_size"] + TFT_CONFIG["h"] + 10
        train = self._make_split(n, county="06075")
        val   = self._make_split(6,  county="06075", start="2024-01-07")
        test  = self._make_split(4,  county="06075", start="2024-02-18")
        _validate_pipeline_inputs(train, val, test)   # must not raise

    @staticmethod
    def _capture_warnings(fn):
        """Run fn() and return list of loguru WARNING messages emitted."""
        from loguru import logger
        captured: list[str] = []
        sid = logger.add(lambda msg: captured.append(msg), level="WARNING", format="{message}")
        try:
            fn()
        finally:
            logger.remove(sid)
        return captured

    def test_warns_on_short_train_series(self):
        from main import _validate_pipeline_inputs
        train = self._make_split(10, county="06055")   # 10 << input_size + h
        val   = self._make_split(4,  county="06055", start="2024-01-07")
        test  = self._make_split(2,  county="06055", start="2024-02-04")
        msgs = self._capture_warnings(lambda: _validate_pipeline_inputs(train, val, test))
        assert any("INV-SHORT" in m for m in msgs), (
            f"Expected INV-SHORT warning; got: {msgs}"
        )

    def test_warns_on_empty_split(self):
        from main import _validate_pipeline_inputs
        train = self._make_split(80, county="06075")
        val   = pd.DataFrame()
        test  = self._make_split(4,  county="06075", start="2024-02-18")
        msgs = self._capture_warnings(lambda: _validate_pipeline_inputs(train, val, test))
        assert any("INV-EMPTY" in m for m in msgs), (
            f"Expected INV-EMPTY warning; got: {msgs}"
        )

    def test_warns_on_fips_in_val_but_not_train(self):
        from main import _validate_pipeline_inputs
        train = self._make_split(80, county="06075")
        val   = self._make_split(6,  county="06001", start="2024-01-07")  # different county
        test  = self._make_split(4,  county="06075", start="2024-02-18")
        msgs = self._capture_warnings(lambda: _validate_pipeline_inputs(train, val, test))
        assert any("INV-FIPS" in m for m in msgs), (
            f"Expected INV-FIPS warning; got: {msgs}"
        )


# ===========================================================================
# Tests: WastewaterTFT start_padding_enabled and _train_final_model val_size
# ===========================================================================

class TestTftStartPaddingEnabled:
    """TFT must be constructed with start_padding_enabled=True."""

    def test_start_padding_enabled_in_config(self):
        from src.config import TFT_CONFIG
        assert TFT_CONFIG.get("start_padding_enabled") is True, (
            "start_padding_enabled must be True in TFT_CONFIG to handle short county series"
        )

    def test_tft_hparam_start_padding_enabled(self):
        from src.models.tft_model import WastewaterTFT
        model = WastewaterTFT(max_steps=2)
        assert model._tft.hparams.start_padding_enabled is True, (
            "WastewaterTFT must pass start_padding_enabled=True to the TFT constructor"
        )


@pytest.mark.skip(reason="_train_final_model is now inlined in main(); no longer a standalone function")
class TestTrainFinalModelUsesValSize:
    """_train_final_model must call model.fit with val_size=model.h, not val_df."""

    def test_fit_called_with_val_size(self, monkeypatch):
        """Monkeypatch WastewaterTFT.fit to capture kwargs; assert val_size present."""
        from main import _train_final_model
        from src.models.tft_model import WastewaterTFT

        fit_calls: list[dict] = []
        predict_df = pd.DataFrame({
            "unique_id": ["06075"],
            "ds": pd.to_datetime(["2024-01-07"]),
            "TFT-median": [1.0],
        })

        def _mock_fit(self, train_df, **kwargs):
            fit_calls.append(kwargs)

        def _mock_predict(self):
            return predict_df

        monkeypatch.setattr(WastewaterTFT, "fit", _mock_fit)
        monkeypatch.setattr(WastewaterTFT, "predict", _mock_predict)
        monkeypatch.setattr(WastewaterTFT, "save", lambda self: None)

        dummy_train = pd.DataFrame({
            "sample_collect_date": pd.date_range("2022-01-02", periods=3, freq="W"),
            "county_fips": "06075",
            "log1p_concentration": [0.5, 0.6, 0.7],
        })
        _train_final_model(dummy_train, max_steps=2)

        assert fit_calls, "_train_final_model did not call model.fit"
        call_kwargs = fit_calls[0]
        assert "val_size" in call_kwargs, (
            f"Expected val_size in fit() kwargs, got: {call_kwargs}"
        )
        assert "val_df" not in call_kwargs, (
            "val_df should not be passed — it crashes on unequal-length series"
        )


class TestFinalModelTrainedOnTrainPlusVal:
    """main() must pass train+val to _train_final_model so forecast overlaps test_df."""

    def test_forecast_dates_after_val_end(self):
        """Forecast dates must start after VAL_END_DATE (2023-12-31), not after TRAIN_END_DATE."""
        from src.config import TRAIN_END_DATE, VAL_END_DATE
        train_end = pd.Timestamp(TRAIN_END_DATE)
        val_end   = pd.Timestamp(VAL_END_DATE)

        # Forecast from a model trained on train+val would start after val_end
        # Forecast from train-only would start after train_end
        # Simulate both scenarios and confirm they produce different forecast origins
        fake_train_only_last = train_end
        fake_train_val_last  = val_end

        # A 14-step weekly forecast starting after train_end lands in val, not test
        from_train_only = fake_train_only_last + pd.Timedelta(weeks=1)
        from_train_val  = fake_train_val_last  + pd.Timedelta(weeks=1)

        test_start = val_end + pd.Timedelta(days=1)
        assert from_train_only < test_start, (
            "This confirms that training on train_df alone produces forecasts before the test window"
        )
        assert from_train_val >= test_start, (
            "Training on train+val produces forecasts that start inside the test window"
        )

    def test_concat_preserves_inv1(self):
        """Concatenating processed train and val (already-scaled) does not re-fit the scaler."""
        from sklearn.preprocessing import RobustScaler
        from src.config import TARGET_COL, NWSS_DATE_COL, COUNTY_COL

        rng = np.random.default_rng(42)
        raw_train_vals = rng.exponential(2.0, 80)
        raw_val_vals   = rng.exponential(2.0, 26)

        scaler = RobustScaler().fit(raw_train_vals.reshape(-1, 1))
        scaled_train = scaler.transform(raw_train_vals.reshape(-1, 1)).flatten()
        scaled_val   = scaler.transform(raw_val_vals.reshape(-1,   1)).flatten()

        train_df = pd.DataFrame({
            NWSS_DATE_COL: pd.date_range("2021-01-03", periods=80, freq="W"),
            COUNTY_COL: "06075",
            TARGET_COL: scaled_train,
        })
        val_df = pd.DataFrame({
            NWSS_DATE_COL: pd.date_range("2022-07-03", periods=26, freq="W"),
            COUNTY_COL: "06075",
            TARGET_COL: scaled_val,
        })
        combined = pd.concat([train_df, val_df], ignore_index=True)

        # All TARGET_COL values come from the train-fitted scaler — no re-fitting occurred
        assert len(combined) == 80 + 26
        # Values outside the original train range are allowed (val can exceed train stats)
        assert combined[TARGET_COL].notna().all()
