"""
TwoStagePipeline — end-to-end inference orchestration.

Execution flow
--------------
  raw data
      │
      ▼  BaseDatasetAdapter.run()
  processed panel
      │
      ▼  OutbreakClassifier.classify_df()
  classification output  (triggered / suppressed per county)
      │
      ├── triggered counties ──→  OutbreakForecaster (full TFT)
      └── suppressed counties ──→ OutbreakForecaster (flat quiet prior)
      │
      ▼
  InferenceResult  (combined forecast + metadata)

Usage
-----
    from src.pipeline import TwoStagePipeline
    from src.data_pipeline.adapters import COVID_Adapter
    from src.models.classifier import OutbreakClassifier
    from src.models.forecaster import OutbreakForecaster
    from src.models.tft_model import WastewaterTFT

    adapter    = COVID_Adapter()
    classifier = OutbreakClassifier().fit(train_df)
    forecaster = OutbreakForecaster(model=WastewaterTFT.load())

    pipeline   = TwoStagePipeline(adapter, classifier, forecaster)
    result     = pipeline.run(processed_df)

    print(result.summary())
    result.forecast_df.to_parquet("forecast.parquet")

Target-variable pivot
---------------------
To switch from COVID cases → mortality or %positive:
1. Re-fit the adapter with the new target column.
2. Re-train WastewaterTFT with the new target.
3. Pass the new adapter + forecaster to TwoStagePipeline.
No changes to the pipeline, classifier, or evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import COUNTY_COL, NWSS_DATE_COL
from src.data_pipeline.adapters import BaseDatasetAdapter
from src.models.classifier import OutbreakClassifier
from src.models.forecaster import OutbreakForecaster


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Complete output of one TwoStagePipeline.run() call.

    Attributes
    ----------
    run_date            : Timestamp when inference was executed.
    clf_df              : Full classification output (one row per county × week).
    forecast_df         : Combined forecast (TFT for triggered, quiet prior for
                          suppressed).  None only if an unrecoverable error occurred.
    triggered_counties  : FIPS codes where OutbreakClassifier fired Stage 2.
    suppressed_counties : FIPS codes where Stage 2 was gated out.
    adapter_name        : Name of the adapter used (for logging/audit).
    """

    run_date:            pd.Timestamp
    clf_df:              pd.DataFrame
    forecast_df:         Optional[pd.DataFrame]
    triggered_counties:  list[str] = field(default_factory=list)
    suppressed_counties: list[str] = field(default_factory=list)
    adapter_name:        str = ""

    def summary(self) -> str:
        n_triggered  = len(self.triggered_counties)
        n_suppressed = len(self.suppressed_counties)
        n_total      = n_triggered + n_suppressed
        fc_rows      = len(self.forecast_df) if self.forecast_df is not None else 0
        return (
            f"InferenceResult [{self.run_date.date()}]  "
            f"adapter={self.adapter_name}  "
            f"triggered={n_triggered}/{n_total}  "
            f"forecast_rows={fc_rows}"
        )

    @property
    def any_triggered(self) -> bool:
        return len(self.triggered_counties) > 0


# ---------------------------------------------------------------------------
# TwoStagePipeline
# ---------------------------------------------------------------------------

class TwoStagePipeline:
    """Orchestrates the full two-stage inference system.

    Parameters
    ----------
    adapter     : Fitted BaseDatasetAdapter for data loading + feature engineering.
    classifier  : Fitted OutbreakClassifier (call .fit(train_df) before passing).
    forecaster  : OutbreakForecaster wrapping a fitted WastewaterTFT.
    id_col      : Series identifier column in processed_df.
    date_col    : Date column in processed_df.
    """

    def __init__(
        self,
        adapter:    BaseDatasetAdapter,
        classifier: OutbreakClassifier,
        forecaster: OutbreakForecaster,
        id_col:     str = COUNTY_COL,
        date_col:   str = NWSS_DATE_COL,
    ) -> None:
        self.adapter    = adapter
        self.classifier = classifier
        self.forecaster = forecaster
        self.id_col     = id_col
        self.date_col   = date_col

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        processed_df: pd.DataFrame,
        run_date:     Optional[pd.Timestamp] = None,
    ) -> InferenceResult:
        """Execute the two-stage pipeline on an already-processed panel.

        Parameters
        ----------
        processed_df : Adapter-processed DataFrame (after scaling).  Typically
                       the output of ``adapter.run()`` → ``adapter.fit_and_scale()``.
        run_date     : Inference timestamp; defaults to the latest date in the panel.

        Returns
        -------
        InferenceResult with classification output, combined forecast, and metadata.
        """
        if run_date is None:
            run_date = pd.Timestamp(processed_df[self.date_col].max())

        all_ids = sorted(processed_df[self.id_col].unique().tolist())
        logger.info(
            "TwoStagePipeline.run() [{date}]: {n} counties.",
            date=run_date.date(), n=len(all_ids),
        )

        # ── Stage 1: classify ────────────────────────────────────────────
        clf_df = self.classifier.classify_df(processed_df)

        triggered  = self.classifier.triggered_counties(clf_df)
        suppressed = self.classifier.suppressed_counties(clf_df)

        logger.info(
            "Stage 1 complete: triggered={}, suppressed={}",
            triggered, suppressed,
        )

        # ── Stage 2: conditional forecasting ─────────────────────────────
        forecast_df = self.forecaster.predict(
            processed_df=processed_df,
            triggered_ids=triggered,
            all_ids=all_ids,
        )

        return InferenceResult(
            run_date=run_date,
            clf_df=clf_df,
            forecast_df=forecast_df if not forecast_df.empty else None,
            triggered_counties=triggered,
            suppressed_counties=suppressed,
            adapter_name=type(self.adapter).__name__,
        )

    # ------------------------------------------------------------------
    # Convenience: run from raw files
    # ------------------------------------------------------------------

    def run_from_files(
        self,
        signal_path,
        target_path,
        *,
        train_df: Optional[pd.DataFrame] = None,
    ) -> InferenceResult:
        """Load, process, and run inference from raw data files.

        If train_df is provided it is used to fit the adapter's scaler
        (leakage-free).  Otherwise the full dataset is used for scaler
        fitting (suitable for one-shot inference on new data).
        """
        full_df = self.adapter.run(
            signal_path=signal_path,
            target_path=target_path,
            fit_scaler=(train_df is None),
        )
        if train_df is not None:
            self.adapter.fit_and_scale(train_df)
            full_df = self.adapter.transform(full_df)

        return self.run(full_df)
