"""
Unified evaluation pipeline for Sewer Signals.

Classes
-------
OnsetEvent      Dataclass for a single detected true outbreak onset.
OnsetLabeler    Labels true onset weeks using a fixed percentile threshold
                computed from training data (no rolling-window contamination).
Evaluator       Scoring loop: probabilistic + detection metrics; report generator.

Functions
---------
expanding_window_cv()      Expanding-window time-series cross-validation.
run_outbreak_validation()  Targeted evaluation against historical outbreak windows.

OnsetLabeler design
-------------------
The labeler uses a percentile threshold fitted ONCE on the training distribution
per county (e.g. 75th percentile of log1p_new_cases).  An onset is confirmed when
the signal exceeds this threshold for ``sustained_weeks`` consecutive weeks.
The true onset date is back-dated to the FIRST week of that above-threshold run,
not the confirmation week — this gives an unbiased TTD measurement.

This avoids the rolling-window contamination that afflicts Z-score onset labeling:
  - Z-score baseline computed over trailing N weeks conflates surge periods with
    quiet baselines, producing onset dates that vary depending on window placement.
  - Fixed-percentile threshold from training produces a constant reference level
    that is stable across the entire evaluation horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config import (
    COUNTY_COL,
    NWSS_DATE_COL,
    TARGET_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from src.evaluation.metrics import (
    EvalReport,
    ProbabilisticResult,
    QuantileColumns,
    coverage,
    detection_score,
    mae,
    match_alerts_to_onsets,
    pinball_loss,
    wis,
)

# ---------------------------------------------------------------------------
# True-onset representation
# ---------------------------------------------------------------------------

@dataclass
class OnsetEvent:
    """A single confirmed outbreak onset for one county.

    Attributes
    ----------
    county           : County FIPS string.
    onset_date       : First week the signal crossed the threshold (true onset).
    onset_value      : Signal value at onset_date.
    confirmation_date: Date the run was confirmed (onset_date + (sustained−1) weeks).
    """

    county:            str
    onset_date:        pd.Timestamp
    onset_value:       float
    confirmation_date: pd.Timestamp


# ---------------------------------------------------------------------------
# OnsetLabeler — fixed-percentile onset detection
# ---------------------------------------------------------------------------

class OnsetLabeler:
    """Labels true outbreak onset events using a training-data percentile threshold.

    Parameters
    ----------
    percentile      : Nth percentile of the training signal used as the onset
                      threshold (default 75).  Choose based on the clinical
                      significance threshold for your target population.
    sustained_weeks : Number of consecutive above-threshold weeks required to
                      confirm an onset (default 2).
    signal_col      : Column containing the signal to threshold (default TARGET_COL).
    id_col          : County identifier column (default COUNTY_COL).
    date_col        : Date column (default NWSS_DATE_COL).
    """

    def __init__(
        self,
        percentile:      float = 75.0,
        sustained_weeks: int   = 2,
        signal_col:      str   = TARGET_COL,
        id_col:          str   = COUNTY_COL,
        date_col:        str   = NWSS_DATE_COL,
    ) -> None:
        self.percentile      = percentile
        self.sustained_weeks = sustained_weeks
        self.signal_col      = signal_col
        self.id_col          = id_col
        self.date_col        = date_col
        self._thresholds: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "OnsetLabeler":
        """Compute per-county onset threshold from training data.

        Must be called on training data ONLY to avoid leakage.
        """
        self._thresholds = {}
        for county, grp in train_df.groupby(self.id_col):
            vals = grp[self.signal_col].dropna()
            if len(vals) >= 4:
                self._thresholds[str(county)] = float(
                    np.percentile(vals, self.percentile)
                )
            else:
                logger.warning(
                    "OnsetLabeler: county {} has {} training values — "
                    "fewer than 4, threshold not set.",
                    county, len(vals),
                )
        logger.info(
            "OnsetLabeler fitted on {} counties  "
            "(p{:.0f} threshold range: {:.3f}–{:.3f}).",
            len(self._thresholds),
            self.percentile,
            min(self._thresholds.values(), default=float("nan")),
            max(self._thresholds.values(), default=float("nan")),
        )
        return self

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def get_onset_events(self, df: pd.DataFrame) -> list[OnsetEvent]:
        """Return all confirmed onset events in df.

        Onset back-dating:
          Confirmation fires at the end of a sustained run.
          True onset = confirmation_date − (sustained_weeks − 1) × 7 days.
          This ensures TTD measures lead time to the ACTUAL crossing point.
        """
        if not self._thresholds:
            raise RuntimeError("OnsetLabeler not fitted. Call fit(train_df) first.")

        events: list[OnsetEvent] = []

        for county, grp in df.groupby(self.id_col):
            threshold = self._thresholds.get(str(county))
            if threshold is None:
                logger.debug(
                    "OnsetLabeler.get_onset_events: no threshold for county {} — skipping.",
                    county,
                )
                continue

            grp = grp.sort_values(self.date_col).copy().set_index(self.date_col)
            signal = grp[self.signal_col].astype(float)
            above  = signal >= threshold

            confirmed = (
                above
                .rolling(self.sustained_weeks, min_periods=self.sustained_weeks)
                .sum()
                .ge(self.sustained_weeks)
                .fillna(False)
            )

            # Transition False→True = start of each new confirmed above-threshold run
            new_run = confirmed & ~confirmed.shift(1).fillna(False)

            for conf_date in new_run[new_run].index:
                # Back-date to the actual first above-threshold week
                onset_date = conf_date - pd.Timedelta(weeks=self.sustained_weeks - 1)
                onset_val  = float(signal.get(onset_date, signal[conf_date]))
                events.append(OnsetEvent(
                    county=str(county),
                    onset_date=onset_date,
                    onset_value=onset_val,
                    confirmation_date=conf_date,
                ))

        return events

    def label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with an added boolean 'onset' column."""
        events    = self.get_onset_events(df)
        onset_keys = {(e.county, e.onset_date) for e in events}
        df = df.copy()
        df["onset"] = df.apply(
            lambda r: (str(r[self.id_col]), r[self.date_col]) in onset_keys,
            axis=1,
        )
        return df


# ---------------------------------------------------------------------------
# Evaluator — scoring loop + report generator
# ---------------------------------------------------------------------------

class Evaluator:
    """Unified scoring pipeline for probabilistic forecasting and outbreak detection.

    Parameters
    ----------
    labeler          : Fitted OnsetLabeler for identifying true onset events.
                       Required for Category 2 detection metrics.
                       Pass None to skip detection metrics (probabilistic only).
    q_cols           : Quantile column map; auto-detected from forecast_df if None.
    match_pre_weeks  : Pre-window for alert→onset matching (default 4 weeks).
    match_post_weeks : Post-window for alert→onset matching (default 2 weeks).
    """

    def __init__(
        self,
        labeler:          Optional[OnsetLabeler]   = None,
        q_cols:           Optional[QuantileColumns] = None,
        match_pre_weeks:  int = 4,
        match_post_weeks: int = 2,
    ) -> None:
        self.labeler          = labeler
        self.q_cols           = q_cols
        self.match_pre_weeks  = match_pre_weeks
        self.match_post_weeks = match_post_weeks

    # ------------------------------------------------------------------
    # score() — main entry point
    # ------------------------------------------------------------------

    def score(
        self,
        actual_df:   pd.DataFrame,
        forecast_df: pd.DataFrame,
        alert_df:    Optional[pd.DataFrame] = None,
        cutoff_date: Optional[pd.Timestamp] = None,
    ) -> EvalReport:
        """Compute all available metrics for one evaluation window.

        Parameters
        ----------
        actual_df   : Ground-truth from processor (COUNTY_COL, NWSS_DATE_COL, TARGET_COL).
        forecast_df : NeuralForecast predict() output (unique_id, ds, quantile cols).
        alert_df    : OutbreakClassifier output with columns unique_id, ds, alert (bool).
                      Pass None to skip Category 2 detection metrics.
        cutoff_date : Training cutoff for this fold (stored in EvalReport metadata).

        Returns
        -------
        EvalReport with prob always populated; det populated only when
        alert_df is provided and a fitted labeler is available.
        """
        q_cols = self.q_cols or QuantileColumns.auto_detect(forecast_df)

        # ── Align actual to forecast rows ────────────────────────────────────
        merged = forecast_df.merge(
            actual_df.rename(columns={
                COUNTY_COL:   "unique_id",
                NWSS_DATE_COL: "ds",
                TARGET_COL:   "y_true",
            })[["unique_id", "ds", "y_true"]],
            on=["unique_id", "ds"],
            how="inner",
        )

        if merged.empty:
            logger.warning(
                "Evaluator.score(): forecast and actuals share no overlapping "
                "(unique_id, ds) pairs — returning null report."
            )
            return self._null_report(cutoff_date)

        y_true = merged["y_true"].to_numpy(dtype=float)

        # ── Category 1: Probabilistic metrics ───────────────────────────────
        wis_vals  = wis(y_true, merged, q_cols)
        cov       = coverage(y_true, merged, q_cols)
        pb        = pinball_loss(y_true, merged, q_cols)
        mae_val   = mae(y_true, merged[q_cols.q50].to_numpy(dtype=float))

        wis_by_county = {
            uid: float(wis_vals[merged["unique_id"] == uid].mean())
            for uid in merged["unique_id"].unique()
        }

        prob = ProbabilisticResult(
            mean_wis=float(wis_vals.mean()),
            wis_by_county=wis_by_county,
            pinball_by_quantile=pb,
            coverage_50=cov["coverage_50"],
            coverage_95=cov["coverage_95"],
            mae=mae_val,
            n_observations=len(merged),
            n_series=merged["unique_id"].nunique(),
        )

        # ── Category 2: Detection metrics ────────────────────────────────────
        det: Optional[object] = None

        if alert_df is not None and self.labeler is not None:
            onsets = self.labeler.get_onset_events(actual_df)

            alert_rows  = alert_df[alert_df["alert"].astype(bool)]
            alerts_list = list(zip(
                alert_rows["unique_id"].astype(str),
                pd.to_datetime(alert_rows["ds"]),
            ))

            tp, fp, fn, ttd = match_alerts_to_onsets(
                alerts=alerts_list,
                onsets=onsets,
                pre_window_weeks=self.match_pre_weeks,
                post_window_weeks=self.match_post_weeks,
            )
            det = detection_score(
                tp=tp, fp=fp, fn=fn,
                ttd_days=ttd,
                n_actual_onsets=len(onsets),
                n_alerts=len(alerts_list),
            )

        return EvalReport(prob=prob, det=det, cutoff_date=cutoff_date)

    # ------------------------------------------------------------------
    # report() — human-readable plain-text summary
    # ------------------------------------------------------------------

    def report(self, er: EvalReport, title: str = "Evaluation Report") -> str:
        """Format EvalReport as a plain-text table."""
        sep  = "─" * 62
        lines = [sep, f"  {title}", sep]

        lines += [
            "  Probabilistic Metrics",
            f"    WIS         {er.prob.mean_wis:.4f}  (lower is better)",
            f"    Coverage 50 {er.prob.coverage_50:.1%}  (target 50%)",
            f"    Coverage 95 {er.prob.coverage_95:.1%}  (target 95%)",
            f"    MAE         {er.prob.mae:.4f}  (median forecast absolute error)",
        ]
        if er.prob.pinball_by_quantile:
            pb_str = "  ".join(
                f"{k}={v:.4f}" for k, v in er.prob.pinball_by_quantile.items()
            )
            lines.append(f"    Pinball     {pb_str}")
        lines.append(
            f"  n_obs={er.prob.n_observations}  n_series={er.prob.n_series}"
        )

        if er.prob.wis_by_county:
            lines.append("  WIS by county:")
            for fips, val in sorted(er.prob.wis_by_county.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {fips}  {val:.4f}")

        if er.det is not None:
            def _f(v: float) -> str:
                return f"{v:.3f}" if not np.isnan(v) else "N/A"
            lines += [
                "  Detection Metrics",
                f"    True Onsets  {er.det.n_actual_onsets}",
                f"    Alerts       {er.det.n_alerts}",
                f"    TP={er.det.tp}  FP={er.det.fp}  FN={er.det.fn}",
                f"    Precision    {_f(er.det.precision)}",
                f"    Recall       {_f(er.det.recall)}",
                f"    F1           {_f(er.det.f1)}",
                f"    Mean TTD     {_f(er.det.mean_ttd_days)} days"
                f"  (positive = early alert)",
            ]
        else:
            lines.append(
                "  Detection: N/A (provide alert_df + fitted labeler)"
            )

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _null_report(self, cutoff_date: Optional[pd.Timestamp]) -> EvalReport:
        return EvalReport(
            prob=ProbabilisticResult(
                mean_wis=float("nan"),
                wis_by_county={},
                pinball_by_quantile={},
                coverage_50=float("nan"),
                coverage_95=float("nan"),
                mae=float("nan"),
                n_observations=0,
                n_series=0,
            ),
            det=None,
            cutoff_date=cutoff_date,
        )


# ---------------------------------------------------------------------------
# OutbreakWindowResult — used by run_outbreak_validation()
# ---------------------------------------------------------------------------

@dataclass
class OutbreakWindowResult:
    """Evaluation result for one historical outbreak validation window."""

    name:        str
    eval_start:  str
    eval_end:    str
    n_counties:  int
    eval_report: EvalReport


# ---------------------------------------------------------------------------
# expanding_window_cv — time-series cross-validation loop
# ---------------------------------------------------------------------------

def expanding_window_cv(
    processed_df:      pd.DataFrame,
    model_factory:     Callable[[], Any],
    initial_train_end: str               = TRAIN_END_DATE,
    eval_end:          str               = VAL_END_DATE,
    step_weeks:        int               = 4,
    h:                 Optional[int]     = None,
    q_cols:            Optional[QuantileColumns] = None,
    date_col:          str               = NWSS_DATE_COL,
    forecast_collector: Optional[list]   = None,
    fold_store_dir:    Optional[Path]    = None,
    two_stage:         bool              = False,
    phase_aware:       bool              = False,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    Each fold trains on all data up to a cutoff date, then evaluates on the
    next ``h`` weeks.  The training window expands by ``step_weeks`` each fold.

    Parameters
    ----------
    processed_df     : Full processor-output DataFrame (all splits combined).
    model_factory    : Zero-argument callable returning a fresh, unfitted model.
    initial_train_end: ISO date string; first CV cutoff (Wednesday).
    eval_end         : ISO date string; last CV cutoff (Wednesday).
    step_weeks       : Weeks between successive cutoffs.
    h                : Forecast horizon; inferred from model if None.
    q_cols           : Quantile column map; auto-detected if None.
    date_col         : Date column name in processed_df.
    forecast_collector: If provided, each fold's forecast (with cutoff_date added)
                        is appended here — used for rolling-holdout stitching.
    fold_store_dir   : If provided, each fold's forecast and actuals are written
                       to ``fold_store_dir/fold_NN_YYYY-MM-DD_{forecast,actuals}.parquet``.
                       Enables post-hoc per-fold metric recomputation with all
                       7 metrics (including full pinball breakdown).
    two_stage        : When True, each fold fits an OutbreakClassifier on that fold's
                       training window and uses OutbreakForecaster for conditional
                       prediction — triggered counties get the full TFT forecast,
                       suppressed counties get the data-driven quiet prior.  This
                       ensures the two-stage gate is evaluated under the same
                       information constraints as the static inference path.
    phase_aware      : When True, each fold applies PhaseLabeler + StratifiedWindowSampler
                       to its training window before fitting — the same oversampling of
                       onset/peak/decay phases that ``--phase-aware-train`` applies to
                       the final model.  Without this, rolling folds train on raw
                       unsampled data even when the flag is set on the main pipeline.

    Returns
    -------
    pd.DataFrame with one row per completed fold and metric columns
    (including per-quantile pinball columns ``pinball_q025`` … ``pinball_q975``).
    """
    if fold_store_dir is not None:
        Path(fold_store_dir).mkdir(parents=True, exist_ok=True)
    cutoffs = pd.date_range(
        start=pd.Timestamp(initial_train_end),
        end=pd.Timestamp(eval_end),
        freq=f"{step_weeks}W-WED",
    )
    if len(cutoffs) == 0:
        raise ValueError(
            f"No cutoffs between {initial_train_end} and {eval_end} "
            f"at {step_weeks}-week steps."
        )

    evaluator = Evaluator(labeler=None, q_cols=q_cols)
    results: list[dict] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]CV fold[/bold cyan] {task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("", total=len(cutoffs))

        for fold_idx, cutoff in enumerate(cutoffs):
            progress.update(
                task,
                description=(
                    f"[white]{fold_idx + 1}/{len(cutoffs)}[/white] "
                    f"cutoff [yellow]{cutoff.date()}[/yellow]"
                ),
            )
            logger.info(
                "CV fold {}/{} — cutoff date {}.",
                fold_idx + 1, len(cutoffs), cutoff.date(),
            )

            train = processed_df[processed_df[date_col] <= cutoff].copy()

            model   = model_factory()
            horizon = h if h is not None else getattr(model, "h", 2)

            val_end_dt = cutoff + pd.Timedelta(weeks=horizon)
            val = processed_df[
                (processed_df[date_col] > cutoff)
                & (processed_df[date_col] <= val_end_dt)
            ].copy()

            if len(train) < horizon * 4:
                logger.warning(
                    "Fold {}: insufficient training rows ({}) — skipping.",
                    fold_idx + 1, len(train),
                )
                progress.advance(task)
                continue
            if val.empty:
                logger.warning(
                    "Fold {}: empty validation window — skipping.", fold_idx + 1
                )
                progress.advance(task)
                continue

            try:
                # ── Phase-aware oversampling (per fold) ───────────────────────
                # Each fold's PhaseLabeler is fit on that fold's training window
                # only — thresholds never see future data.
                train_fit = train
                if phase_aware:
                    from src.data_pipeline.sampler import PhaseLabeler, StratifiedWindowSampler
                    _pl = PhaseLabeler()
                    _pl.fit(train)
                    _labeled = _pl.label(train)
                    _sampler = StratifiedWindowSampler(
                        input_size=cfg.TFT_CONFIG["input_size"],
                        h=cfg.TFT_CONFIG["h"],
                    )
                    train_fit = _sampler.sample(_labeled)
                    logger.info(
                        "  Phase-aware fold {}: {} → {} rows after stratified sampling.",
                        fold_idx + 1, len(train), len(train_fit),
                    )

                model.fit(train_fit, val_size=0)

                if two_stage:
                    # ── Two-stage: fit a per-fold classifier, then use
                    # OutbreakForecaster for conditional prediction.
                    # The classifier is fit on this fold's training window only
                    # (no leakage from future folds).  The current outbreak state
                    # is determined from the last H weeks of that training window
                    # so the decision reflects what was knowable at the cutoff.
                    from src.models.classifier import OutbreakClassifier
                    from src.models.forecaster import OutbreakForecaster

                    fold_clf = OutbreakClassifier()
                    fold_clf.fit(train)

                    # Classify the training window to establish current state
                    clf_df = fold_clf.classify_df(train)

                    # Only look at the last H weeks so the trigger decision
                    # reflects the outbreak state AT the cutoff, not historical
                    # surges from months ago.
                    lookback_start = cutoff - pd.Timedelta(weeks=horizon)
                    recent_clf     = clf_df[clf_df[date_col] > lookback_start]
                    triggered  = fold_clf.triggered_counties(recent_clf)
                    suppressed = fold_clf.suppressed_counties(recent_clf)
                    all_ids    = sorted(train[COUNTY_COL].unique().tolist())

                    fold_q_cols = evaluator.q_cols  # may be None on first fold
                    forecaster  = OutbreakForecaster(model=model, q_cols=fold_q_cols)
                    forecast    = forecaster.predict(
                        processed_df=train,
                        triggered_ids=triggered,
                        all_ids=all_ids,
                    )
                    logger.info(
                        "  Two-stage fold {}: {}/{} counties triggered.",
                        fold_idx + 1, len(triggered), len(all_ids),
                    )
                else:
                    forecast = model.predict()

                if forecast_collector is not None and not forecast.empty:
                    _fc = forecast.copy()
                    _fc["cutoff_date"] = cutoff
                    forecast_collector.append(_fc)

                # Persist raw fold data for post-hoc full-metric recomputation
                if fold_store_dir is not None and not forecast.empty:
                    _slug = f"fold_{fold_idx + 1:02d}_{cutoff.date()}"
                    forecast.to_parquet(
                        Path(fold_store_dir) / f"{_slug}_forecast.parquet",
                        index=False,
                    )
                    val.to_parquet(
                        Path(fold_store_dir) / f"{_slug}_actuals.parquet",
                        index=False,
                    )

                if evaluator.q_cols is None and not forecast.empty:
                    try:
                        evaluator.q_cols = QuantileColumns.auto_detect(forecast)
                        logger.info("Auto-detected quantile columns: {}", evaluator.q_cols)
                    except ValueError as exc:
                        logger.error("Could not auto-detect quantile columns: {}", exc)
                        raise

                report = evaluator.score(
                    actual_df=val,
                    forecast_df=forecast,
                    cutoff_date=cutoff,
                )
                if report.prob.n_observations > 0:
                    results.append(report.to_dict())
                else:
                    logger.warning(
                        "Fold {}: no overlapping observations — fold skipped.", fold_idx + 1
                    )

            except Exception as exc:
                logger.error("Fold {} failed: {}", fold_idx + 1, exc)
            finally:
                progress.advance(task)

    if not results:
        logger.warning("No CV folds completed successfully.")
        return pd.DataFrame()

    cv_df = pd.DataFrame(results)
    logger.info(
        "CV complete — {} folds, mean WIS={:.4f}.",
        len(cv_df), cv_df["mean_wis"].mean(),
    )
    return cv_df


# ---------------------------------------------------------------------------
# run_outbreak_validation — targeted historical window evaluation
# ---------------------------------------------------------------------------

def run_outbreak_validation(
    processed_df:  pd.DataFrame,
    model_factory: Callable[[], Any],
    windows:       list[dict],
) -> list[OutbreakWindowResult]:
    """Train and evaluate one model per historical outbreak validation window.

    For each window the model is trained on all data strictly before
    ``eval_start``, then asked to forecast H steps covering the eval period.

    Parameters
    ----------
    processed_df  : Full scaled dataset (all splits concatenated).
    model_factory : Callable returning a fresh, unfitted model.
    windows       : List of dicts with keys:
                      name       — display label
                      eval_start — ISO date string (W-WED)
                      eval_end   — ISO date string (W-WED)
                      counties   — list of FIPS strings, or None for all
    """
    evaluator  = Evaluator(labeler=None)
    results: list[OutbreakWindowResult] = []

    for win in windows:
        name       = win["name"]
        eval_start = pd.Timestamp(win["eval_start"])
        eval_end   = pd.Timestamp(win["eval_end"])
        counties   = win.get("counties")

        df_win = (
            processed_df[processed_df[COUNTY_COL].isin(counties)].copy()
            if counties else processed_df.copy()
        )
        if df_win.empty:
            logger.warning(
                "Outbreak window '{}': no data for counties {}. Skipping.",
                name, counties,
            )
            continue

        train_cutoff = eval_start - pd.Timedelta(weeks=1)
        train_data   = df_win[
            pd.to_datetime(df_win[NWSS_DATE_COL]) <= train_cutoff
        ].copy()

        if train_data.empty or train_data[COUNTY_COL].nunique() < 1:
            logger.warning(
                "Outbreak window '{}': no training data before {}. Skipping.",
                name, train_cutoff.date(),
            )
            continue

        logger.info(
            "Outbreak validation '{}' — train ≤ {}  eval {} → {}  ({} counties)",
            name, train_cutoff.date(), eval_start.date(), eval_end.date(),
            df_win[COUNTY_COL].nunique(),
        )

        try:
            model = model_factory()
            model.fit(train_data, val_size=model.h)
            forecast_df = model.predict()
        except Exception as exc:
            logger.warning(
                "Outbreak window '{}': training failed — {}. Skipping.", name, exc
            )
            continue

        forecast_eval = forecast_df[
            (pd.to_datetime(forecast_df["ds"]) >= eval_start) &
            (pd.to_datetime(forecast_df["ds"]) <= eval_end)
        ].copy()
        actual_eval = df_win[
            (pd.to_datetime(df_win[NWSS_DATE_COL]) >= eval_start) &
            (pd.to_datetime(df_win[NWSS_DATE_COL]) <= eval_end)
        ].copy()

        if forecast_eval.empty or actual_eval.empty:
            logger.warning(
                "Outbreak window '{}': forecast does not overlap eval period. Skipping.",
                name,
            )
            continue

        try:
            evaluator.q_cols = QuantileColumns.auto_detect(forecast_eval)
        except ValueError:
            pass

        eval_report = evaluator.score(
            actual_df=actual_eval,
            forecast_df=forecast_eval,
        )

        cov95 = eval_report.prob.coverage_95
        logger.info(
            "  → Cov95={:.1%}  WIS={:.3f}  MAE={:.4f}",
            cov95,
            eval_report.prob.mean_wis,
            eval_report.prob.mae,
        )

        results.append(OutbreakWindowResult(
            name=name,
            eval_start=win["eval_start"],
            eval_end=win["eval_end"],
            n_counties=df_win[COUNTY_COL].nunique(),
            eval_report=eval_report,
        ))

    return results
