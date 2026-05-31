"""
Dataset Adapter Framework for Sewer Signals.

Purpose
-------
Decouple the model architecture from any specific pathogen, geography, or
data source.  A developer building an Influenza or RSV surveillance system
implements the five abstract methods of ``BaseDatasetAdapter`` and plugs the
adapter into ``TwoStagePipeline`` — zero changes to the model files.

Classes
-------
BaseDatasetAdapter  Abstract interface contract.
COVID_Adapter       Concrete implementation for Bay Area COVID-19 WW data.

Forking guide
-------------
To add a new pathogen/dataset:

    from src.data_pipeline.adapters import BaseDatasetAdapter

    class Influenza_Adapter(BaseDatasetAdapter):
        @property
        def signal_col(self)  -> str: return "log1p_influenza_conc"
        @property
        def target_col(self)  -> str: return "log1p_ili_visits"
        @property
        def id_col(self)      -> str: return "site_id"
        @property
        def date_col(self)    -> str: return "collection_date"
        @property
        def momentum_col(self)-> str: return "ww_momentum_lead"

        def load_signal(self, path): ...
        def load_target(self, path): ...
        def clean(self, df):         ...
        def build_features(self, df):...
        def transform_target(self, df): ...

All geography (FIPS codes, county names), pathogen parameters (target unit,
signal column names), and data-source metadata live inside the adapter.
The TFT model, loss functions, and evaluator are unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    BAY_AREA_COUNTIES,
    BAY_AREA_FIPS,
    CA_WW_SIGNAL_COL,
    COUNTY_COL,
    EXCLUDE_FIPS,
    FIPS_TO_COUNTY,
    NWSS_DATE_COL,
    TARGET_COL,
    WW_FEATURE_COL,
)
from src.data_pipeline.processor import CAWastewaterProcessor


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseDatasetAdapter(ABC):
    """Abstract base class for pathogen-agnostic data ingestion.

    All geography, pathogen-specific parameters, and data-source metadata
    must be encapsulated here — not in the model files.

    Implementors must define:
      - Five schema properties (signal_col, target_col, id_col, date_col,
        momentum_col) so downstream components can work with any adapter
        without knowing the concrete type.
      - Five processing methods (load_signal, load_target, clean,
        build_features, transform_target).

    The ``run()`` orchestration method is provided and calls the abstract
    methods in the correct order.  Override it for non-standard pipelines.
    """

    # ------------------------------------------------------------------
    # Schema contract — must return stable column names
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def signal_col(self) -> str:
        """Column name for the log-transformed WW surveillance signal."""

    @property
    @abstractmethod
    def target_col(self) -> str:
        """Column name for the prediction target (e.g. log1p_new_cases)."""

    @property
    @abstractmethod
    def id_col(self) -> str:
        """Column name for the series identifier (e.g. FIPS code, site_id)."""

    @property
    @abstractmethod
    def date_col(self) -> str:
        """Column name for the weekly date index."""

    @property
    @abstractmethod
    def momentum_col(self) -> str:
        """Column name for the WW momentum divergence feature.

        This feature drives the OutbreakClassifier's momentum gate.  It must
        be a lead-time signal: positive when WW velocity is outpacing the
        lagged case velocity (surge leading edge), negative in recovery.

        For COVID, this is ``ww_momentum_lead``.  For a new pathogen,
        implement the analogous velocity-divergence quantity.
        """

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @abstractmethod
    def load_signal(self, path: Path) -> pd.DataFrame:
        """Load and minimally parse the raw wastewater surveillance file.

        Returns a DataFrame that ``clean()`` can accept.
        Must NOT perform feature engineering — that belongs in build_features().
        """

    @abstractmethod
    def load_target(self, path: Path) -> pd.DataFrame:
        """Load and minimally parse the target variable file.

        Returns a DataFrame that ``_merge()`` can join to the signal data on
        (id_col, date_col).  Column name of the raw target count should be
        ``"new_cases"`` (or override ``_merge()`` to rename it).
        """

    # ------------------------------------------------------------------
    # Processing stages
    # ------------------------------------------------------------------

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pathogen/geography-specific QC and normalisation.

        Input:  raw signal DataFrame from ``load_signal()``.
        Output: cleaned DataFrame ready for ``build_features()``.
        """

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity, acceleration, momentum divergence, lags.

        Input:  merged signal+target DataFrame.
        Output: feature-complete DataFrame ready for scaling and TFT ingestion.

        The momentum_col feature MUST be present in the output — the
        OutbreakClassifier depends on it.
        """

    @abstractmethod
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pathogen-specific target transformation.

        For COVID: log1p(new_cases) → TARGET_COL column.
        For %ILI: logit(fraction) or direct use.
        For mortality: log1p(deaths) → target_col.

        If transform_target is a no-op (target already computed in
        build_features), return df unchanged.
        """

    # ------------------------------------------------------------------
    # Schema validation (provided — no override needed)
    # ------------------------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Assert the adapter's required columns are present.

        Raises ValueError listing every missing column so errors surface
        before training starts rather than mid-training.
        """
        required = {self.signal_col, self.target_col, self.id_col, self.date_col}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{type(self).__name__}: missing required columns after pipeline: "
                f"{sorted(missing)}.  Check clean() / build_features() outputs."
            )
        if self.momentum_col not in df.columns:
            logger.warning(
                "{}: momentum column '{}' absent — OutbreakClassifier will "
                "fall back to Z-score-only mode.",
                type(self).__name__, self.momentum_col,
            )

    # ------------------------------------------------------------------
    # Orchestration (override for non-standard pipelines)
    # ------------------------------------------------------------------

    def run(
        self,
        signal_path: Path,
        target_path: Path,
        *,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """Execute the full ingestion → clean → merge → features → target pipeline.

        Parameters
        ----------
        signal_path : Path to the raw wastewater surveillance file.
        target_path : Path to the target variable file.
        fit_scaler  : Whether to fit the scaler on this data (True for train).

        Returns
        -------
        Feature-complete, schema-validated DataFrame ready for the TFT model.
        """
        signal_df = self.load_signal(signal_path)
        target_df = self.load_target(target_path)
        df        = self.clean(signal_df)
        df        = self._merge(df, target_df)
        df        = self.build_features(df)
        df        = self.transform_target(df)
        self.validate_schema(df)
        return df

    def _merge(
        self,
        signal_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inner join signal and target on (id_col, date_col).

        Override for non-standard joins (e.g. lagged merge, fuzzy date match).
        """
        on = [self.id_col, self.date_col]
        merged = signal_df.merge(target_df, on=on, how="inner")
        dropped = len(signal_df) - len(merged)
        if dropped:
            logger.debug(
                "{}: dropped {} signal rows with no matching target record.",
                type(self).__name__, dropped,
            )
        return merged


# ---------------------------------------------------------------------------
# COVID-19 Bay Area concrete implementation
# ---------------------------------------------------------------------------

class COVID_Adapter(BaseDatasetAdapter):
    """Dataset adapter for Bay Area COVID-19 wastewater surveillance.

    Wraps ``CAWastewaterProcessor`` with the ``BaseDatasetAdapter`` interface.
    Provides a complete template for adapting the pipeline to new pathogens.

    Geography   : 9-county SF Bay Area (CA FIPS 06001–06097)
    WW signal   : copies/g dry sludge (solid track, CA WW Surveillance CSV)
    Target      : log1p(weekly new COVID-19 cases) per county
    Source docs : data/raw/California_Wastewater_Surveillance_Data.csv
                  data/raw/Statewide_COVID-19_Cases_Deaths_Tests.csv

    Forking: to adapt for Influenza or RSV
    ----------------------------------------
    1. Subclass or copy this class.
    2. Override ``load_signal`` and ``load_target`` with the new data loaders.
    3. Update ``signal_col`` / ``target_col`` / ``momentum_col`` properties.
    4. Adapt ``clean`` for the new QC rules (units, LOD, inhibition flags).
    5. No changes to TFT model, loss functions, or evaluator.
    """

    def __init__(
        self,
        ww_signal_col: str = CA_WW_SIGNAL_COL,
        exclude_fips: Optional[list[str]] = None,
    ) -> None:
        self._ww_signal_col = ww_signal_col
        self._exclude_fips  = exclude_fips or EXCLUDE_FIPS
        self._proc          = CAWastewaterProcessor(ww_signal_col=ww_signal_col)

    # ------------------------------------------------------------------
    # Schema contract
    # ------------------------------------------------------------------

    @property
    def signal_col(self) -> str:
        return WW_FEATURE_COL          # "log1p_concentration"

    @property
    def target_col(self) -> str:
        return TARGET_COL              # "log1p_new_cases"

    @property
    def id_col(self) -> str:
        return COUNTY_COL              # "county_fips"

    @property
    def date_col(self) -> str:
        return NWSS_DATE_COL           # "sample_collect_date"

    @property
    def momentum_col(self) -> str:
        return "ww_momentum_lead"      # vel_concentration[t] − Δ(log1p_cases[t-1])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_signal(self, path: Path) -> pd.DataFrame:
        """Load CA Wastewater Surveillance CSV; filter to Bay Area solid track.

        Geography filter: ``BAY_AREA_COUNTIES`` minus ``_exclude_fips``.
        Only solid-track rows (Sample Type == 'solid') are retained.
        Numeric signal columns are coerced to float.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CA WW CSV not found at {path}.")

        logger.info("COVID_Adapter.load_signal: {}", path.name)
        df = pd.read_csv(path, dtype=str, low_memory=False)

        # Exclude counties with insufficient WW history
        active_counties = [
            name for fips, name in FIPS_TO_COUNTY.items()
            if fips not in self._exclude_fips
        ]
        df = df[
            df["County"].isin(active_counties) &
            (df["PCR Target"] == "SARS-CoV-2") &
            (df["Sample Type"].str.lower() == "solid")
        ].copy()

        df = df.rename(columns={"Sample Date": NWSS_DATE_COL, "County": "county_name"})
        df[NWSS_DATE_COL] = pd.to_datetime(df[NWSS_DATE_COL], errors="coerce")
        df = df.dropna(subset=[NWSS_DATE_COL])

        for col in ("Raw Concentration", "Norm Pmmov"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "  {} solid-track rows ({} counties).",
            len(df), df["county_name"].nunique(),
        )
        return df

    def load_target(self, path: Path) -> pd.DataFrame:
        """Load CA Statewide Cases CSV; resample daily → weekly W-WED per county.

        Output schema: (COUNTY_COL, NWSS_DATE_COL, 'new_cases').
        Pivot for a new target variable: override this method and return
        a DataFrame with (id_col, date_col, '<raw_count_column>').
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CA Cases CSV not found at {path}.")

        logger.info("COVID_Adapter.load_target: {}", path.name)
        df = pd.read_csv(path, dtype=str, low_memory=False)

        df["date"]  = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
        df["cases"] = pd.to_numeric(df["cases"], errors="coerce")

        active_counties = [
            name for fips, name in FIPS_TO_COUNTY.items()
            if fips not in self._exclude_fips
        ]
        df = df[
            df["area"].isin(active_counties) & (df["area_type"] == "County")
        ].copy()
        df = df.dropna(subset=["date", "cases"])
        df[COUNTY_COL] = df["area"].map(BAY_AREA_FIPS)
        df = df.dropna(subset=[COUNTY_COL])

        weekly = (
            df.set_index("date")
            .groupby(COUNTY_COL)["cases"]
            .resample("W-WED")
            .sum()
            .clip(lower=0)
            .reset_index()
            .rename(columns={"date": NWSS_DATE_COL, "cases": "new_cases"})
        )
        logger.info("  {} county-week rows.", len(weekly))
        return weekly

    # ------------------------------------------------------------------
    # Processing stages
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """CA-specific WW cleaning: county-name → FIPS, numeric coercion, dropna."""
        return self._proc._clean_ca_ww(df)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline: smooth → resample → log → lag → momentum.

        Delegates to the internal CAWastewaterProcessor pipeline stages.
        Override individual stages for a new pathogen (e.g. different
        smoothing window or lag structure).
        """
        df = self._proc._aggregate_ca_to_county_daily(df)
        df = self._proc._rolling_smooth(df)
        df = self._proc._resample_to_weekly(df)
        df = self._proc._log_transform(df)
        df = self._proc._add_calendar_features(df)
        df = self._proc._add_lag_features(df)
        return df

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Target already created as log1p(new_cases) in _merge_cases; pass through."""
        return df

    # ------------------------------------------------------------------
    # Override _merge to use processor's cases merge (handles anti-leakage)
    # ------------------------------------------------------------------

    def _merge(
        self,
        signal_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge WW signal with target using processor's anti-leakage inner join."""
        return self._proc._merge_cases(signal_df, target_df)

    # ------------------------------------------------------------------
    # Fit scaler (exposes the processor's scaling step)
    # ------------------------------------------------------------------

    def fit_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit per-county RobustScaler on df and return scaled DataFrame.

        Call on training data only.  Use ``transform()`` for val/test.
        """
        return self._proc._apply_scaling(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the previously fitted scaler without refitting."""
        return self._proc._apply_scaling(df, fit=False)

    def save_scalers(self, path: Path) -> None:
        self._proc.save_scalers(path)

    def load_scalers(self, path: Path) -> None:
        self._proc.load_scalers(path)
