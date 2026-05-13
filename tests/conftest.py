"""
Shared pytest fixtures for the wastewater pipeline test suite.

All fixtures produce DataFrames that mirror the raw CDC NWSS schema so that
individual pipeline stages can be tested in isolation without hitting the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    """Return a minimal valid NWSS row, with any field overrideable."""
    defaults = {
        "sample_collect_date": "2022-06-01",
        "county_fips": "06001",
        "wwtp_id": "site_A",
        "population_served": "100000",
        "pcr_target_avg_conc": "5000.0",
        "pcr_target_units": "copies/g dry sludge",  # primary unit per project default
        "pcr_target_below_lod": "no",
        "lod_sewage": "200.0",
        "rec_eff_percent": "80.0",
        "inhibition_detect": "no",
    }
    return {**defaults, **overrides}


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def single_valid_row() -> pd.DataFrame:
    """One perfectly clean row — every stage should pass through unchanged."""
    return _make_df([_base_row()])


@pytest.fixture()
def comma_numeric_df() -> pd.DataFrame:
    """Numeric columns that contain comma-formatted strings."""
    return _make_df([
        _base_row(population_served="1,234,567", pcr_target_avg_conc="12,500.5"),
        _base_row(population_served="500,000",   pcr_target_avg_conc="3,000"),
    ])


@pytest.fixture()
def unpadded_fips_df() -> pd.DataFrame:
    """FIPS codes that need zero-padding (4-digit state+county)."""
    return _make_df([
        _base_row(county_fips="6001"),   # Alameda — missing leading zero
        _base_row(county_fips="6085"),   # Santa Clara
        _base_row(county_fips="6075,6081"),  # multi-county, both need padding
    ])


@pytest.fixture()
def mixed_units_df() -> pd.DataFrame:
    """Mix of copies/g dry sludge (keep) and copies/l wastewater (exclude) rows.

    Primary unit is copies/g dry sludge — the two copies/l rows should be
    excluded by _filter_units; the two copies/g rows (including case variation)
    should be retained.
    """
    return _make_df([
        _base_row(county_fips="06001", wwtp_id="site_A", pcr_target_units="copies/g dry sludge"),
        _base_row(county_fips="06085", wwtp_id="site_B", pcr_target_units="copies/g dry sludge"),
        _base_row(county_fips="06085", wwtp_id="site_C", pcr_target_units="copies/l wastewater"),
        _base_row(county_fips="06085", wwtp_id="site_D", pcr_target_units="Copies/L Wastewater"),  # case variation
    ])


@pytest.fixture()
def nondetect_df() -> pd.DataFrame:
    """Some rows flagged as below LOD — should be replaced with lod_sewage/2."""
    return _make_df([
        _base_row(pcr_target_avg_conc="0",     pcr_target_below_lod="yes", lod_sewage="400"),
        _base_row(pcr_target_avg_conc="8000",  pcr_target_below_lod="no",  lod_sewage="400"),
        _base_row(pcr_target_avg_conc="500",   pcr_target_below_lod="yes", lod_sewage="600"),
    ])


@pytest.fixture()
def qc_df() -> pd.DataFrame:
    """Rows spanning pass, fail, and missing recovery-efficiency values."""
    return _make_df([
        _base_row(rec_eff_percent="80.0",  inhibition_detect="no"),   # pass
        _base_row(rec_eff_percent="5.0",   inhibition_detect="no"),   # fail — drop
        _base_row(rec_eff_percent="9.9",   inhibition_detect="no"),   # fail — drop
        _base_row(rec_eff_percent="10.0",  inhibition_detect="yes"),  # pass rec_eff, flagged inhibition
        _base_row(rec_eff_percent="",      inhibition_detect="no"),   # NaN rec_eff — keep
    ])


@pytest.fixture()
def multi_county_df() -> pd.DataFrame:
    """Sites serving multiple counties — each should explode into separate rows."""
    return _make_df([
        _base_row(county_fips="06075,06081", wwtp_id="shared_site", population_served="200000"),
        _base_row(county_fips="06001",       wwtp_id="alameda_only", population_served="100000"),
    ])


@pytest.fixture()
def weighted_agg_df() -> pd.DataFrame:
    """Two sites in the same county on the same day — test weighted mean math."""
    return _make_df([
        _base_row(
            sample_collect_date="2022-06-01",
            county_fips="06001",
            wwtp_id="site_A",
            population_served="100000",
            pcr_target_avg_conc="1000",
        ),
        _base_row(
            sample_collect_date="2022-06-01",
            county_fips="06001",
            wwtp_id="site_B",
            population_served="300000",
            pcr_target_avg_conc="3000",
        ),
    ])


@pytest.fixture()
def full_pipeline_df() -> pd.DataFrame:
    """
    ~18 months of synthetic daily data across three Bay Area counties.
    Enough rows for rolling/resampling stages not to drop everything.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", "2022-06-30", freq="D")
    counties = ["06001", "06085", "06075"]
    rows = []
    for date in dates:
        for county in counties:
            rows.append(
                _base_row(
                    sample_collect_date=str(date.date()),
                    county_fips=county,
                    wwtp_id=f"site_{county[-3:]}",
                    population_served=str(rng.integers(50_000, 500_000)),
                    pcr_target_avg_conc=str(float(rng.exponential(5_000))),
                    rec_eff_percent=str(float(rng.uniform(15, 120))),
                    inhibition_detect="no",
                    pcr_target_below_lod="no",
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def leakage_split_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and val raw DataFrames with a clean date boundary for leakage tests."""
    rng = np.random.default_rng(1)

    def _make_split(start: str, end: str) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="D")
        rows = []
        for date in dates:
            rows.append(
                _base_row(
                    sample_collect_date=str(date.date()),
                    county_fips="06001",
                    wwtp_id="site_A",
                    population_served="250000",
                    pcr_target_avg_conc=str(float(rng.exponential(8_000))),
                    rec_eff_percent=str(float(rng.uniform(20, 100))),
                )
            )
        return pd.DataFrame(rows)

    return _make_split("2021-01-01", "2022-06-30"), _make_split("2022-07-01", "2022-12-31")
