"""
run_santa_clara.py — Single-County Validation Pipeline (Santa Clara County).

Filters the 9-county CA pipeline to Santa Clara County only (FIPS 06085),
then runs the full pipeline: expanding-window CV + final model + evaluation.

This serves as a "unit test" for calibration: one well-observed county
(longest WW history in the Bay Area) in isolation, so that multi-county
averaging cannot mask underdispersion in any single series.

Key Phase 2 changes active in this run:
  - PINNWastewaterLoss with underdispersion penalty (lambda=0.1, min_pi_width=1.5)
  - OUTBREAK_GROWTH_THRESHOLD raised to 0.40 (from 0.25) to reduce false positives
  - GROWTH_RATE_LAMBDA = 0.0 (smoothing prior disabled)
  - 7 quantiles + near-term horizon upweighting

Artefacts are written to data/processed/santa_clara/ and
models_saved/santa_clara/ to avoid overwriting the 9-county run.

Usage:
    python run_santa_clara.py
    python run_santa_clara.py --skip-cv --no-dash
    python run_santa_clara.py --fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

import src.config as cfg
from src.config import (
    CA_CASES_FILENAME,
    CA_WW_FILENAME,
    CA_WW_SIGNAL_COL,
    COUNTY_COL,
    MODELS_DIR,
    NWSS_DATE_COL,
    PROCESSED_DIR,
    RAW_DIR,
    SANTA_CLARA_FIPS,
    TARGET_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from src.data_pipeline.processor import CAWastewaterProcessor
from src.evaluation.metrics import (
    EvalResult,
    QuantileColumns,
    evaluate,
    expanding_window_cv,
)
from src.models.tft_model import WastewaterTFT
from src.utils.helpers import (
    console,
    print_cv_summary,
    print_eval_report,
    setup_logger,
)

# Import shared helpers from the 9-county pipeline
from main import (
    _load_ca_ww_csv,
    _load_ca_cases_csv,
    _split_raw,
    _invert_scaling_to_log1p,
    _build_display_frames,
    _validate_pipeline_inputs,
)

CA_WW_CSV    = RAW_DIR / CA_WW_FILENAME
CA_CASES_CSV = RAW_DIR / CA_CASES_FILENAME

# Santa Clara-specific output directories (separate from 9-county artefacts)
SC_PROCESSED_DIR = PROCESSED_DIR / "santa_clara"
SC_MODELS_DIR    = MODELS_DIR / "santa_clara"
SC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SC_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Santa Clara single-county validation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fast", action="store_true", help="200 steps, skip CV")
    p.add_argument("--skip-cv", action="store_true", help="Skip expanding-window CV")
    p.add_argument("--no-dash", action="store_true", help="Skip dashboard launch")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Filter to Santa Clara
# ---------------------------------------------------------------------------

def _filter_santa_clara(
    raw_ww: pd.DataFrame,
    raw_cases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict raw WW and cases DataFrames to Santa Clara County only."""
    ww_sc = raw_ww[raw_ww["county_name"] == "Santa Clara"].copy()
    cases_sc = raw_cases[raw_cases[COUNTY_COL] == SANTA_CLARA_FIPS].copy()
    logger.info(
        "Santa Clara filter: WW={} rows, cases={} rows.",
        len(ww_sc), len(cases_sc),
    )
    if ww_sc.empty:
        raise ValueError(
            "No WW rows found for 'Santa Clara' — verify the CA WW CSV contains "
            "county_name == 'Santa Clara' and Sample Type == 'solid'."
        )
    if cases_sc.empty:
        raise ValueError(
            f"No case rows found for FIPS {SANTA_CLARA_FIPS} — verify the CA "
            "Cases CSV was loaded and the county filter ran correctly."
        )
    return ww_sc, cases_sc


# ---------------------------------------------------------------------------
# Step 2 — Leakage-free processing (single-county)
# ---------------------------------------------------------------------------

def _process_sc(
    raw_ww: pd.DataFrame,
    raw_cases: pd.DataFrame,
) -> tuple[CAWastewaterProcessor, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process Santa Clara sludge track — same invariants as the 9-county run.

    [INV-1] RobustScaler fitted ONLY on Santa Clara training rows.
    Because only one county is present, the scaler's center/scale parameters
    reflect Santa Clara's local distribution rather than the pooled 9-county
    distribution — the key calibration difference versus the global model.
    """
    raw_ww_train,    raw_ww_val,    raw_ww_test    = _split_raw(raw_ww)
    raw_cases_train, raw_cases_val, raw_cases_test = _split_raw(raw_cases)

    proc = CAWastewaterProcessor(ww_signal_col=CA_WW_SIGNAL_COL)

    logger.info("SC: Fitting per-county RobustScaler on training split …")
    train_df = proc.run(raw_ww_train, cases_df=raw_cases_train)

    logger.info("SC: Applying stored scaler to val split …")
    val_df = proc.transform(raw_ww_val, cases_df=raw_cases_val)

    logger.info("SC: Applying stored scaler to test split …")
    test_df = proc.transform(raw_ww_test, cases_df=raw_cases_test)

    logger.info(
        "SC processing complete — train={}, val={}, test={} rows.",
        len(train_df), len(val_df), len(test_df),
    )
    return proc, train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Step 3 — Expanding-window CV (single-county)
# ---------------------------------------------------------------------------

def _run_sc_cv(cv_data: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """Expanding-window CV on Santa Clara data.

    Same fold structure as the 9-county run (TRAIN_END_DATE → VAL_END_DATE,
    step=4 weeks) but with only one series.  A single-county fold trains
    faster and makes calibration issues easier to diagnose.
    """
    cv_trainer_kwargs = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "early_stop_patience_steps": -1,
    }
    return expanding_window_cv(
        processed_df=cv_data,
        model_factory=lambda: WastewaterTFT(
            max_steps=max_steps,
            trainer_kwargs=cv_trainer_kwargs,
        ),
        initial_train_end=TRAIN_END_DATE,
        eval_end=VAL_END_DATE,
        step_weeks=4,
    )


# ---------------------------------------------------------------------------
# Step 4 — Export
# ---------------------------------------------------------------------------

def _export_sc_results(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    eval_result: EvalResult,
) -> None:
    """Write Santa Clara artefacts to data/processed/santa_clara/."""
    SC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(   SC_PROCESSED_DIR / "train.parquet",    index=False)
    val_df.to_parquet(     SC_PROCESSED_DIR / "val.parquet",      index=False)
    test_df.to_parquet(    SC_PROCESSED_DIR / "test.parquet",     index=False)
    forecast_df.to_parquet(SC_PROCESSED_DIR / "forecast.parquet", index=False)
    if not cv_df.empty:
        cv_df.to_csv(SC_PROCESSED_DIR / "cv_results.csv", index=False)
    (SC_PROCESSED_DIR / "eval_summary.json").write_text(
        json.dumps(eval_result.to_dict(), indent=2, default=str)
    )
    logger.info("Santa Clara artefacts exported to {}.", SC_PROCESSED_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = _parse_args()
    setup_logger()
    cfg.settings = cfg.EnvSettings()

    skip_cv   = args.skip_cv or args.fast
    max_steps = 200 if args.fast else cfg.TFT_CONFIG["max_steps"]
    cv_steps  = min(max_steps, 500)

    console.rule(
        "[bold white] Sewer Signals — Santa Clara Single-County Validation [/bold white]"
    )
    logger.info(
        "SC run: max_steps={}, skip_cv={}, no_dash={}",
        max_steps, skip_cv, args.no_dash,
    )

    # ── 1. Load 9-county raw data ─────────────────────────────────────────────
    raw_ww    = _load_ca_ww_csv(CA_WW_CSV)
    raw_cases = _load_ca_cases_csv(CA_CASES_CSV)

    # ── 2. Filter to Santa Clara only ─────────────────────────────────────────
    raw_ww_sc, raw_cases_sc = _filter_santa_clara(raw_ww, raw_cases)

    # ── 3. Leakage-free processing [INV-1: scaler fit on SC train rows only] ──
    proc, train_df, val_df, test_df = _process_sc(raw_ww_sc, raw_cases_sc)
    cv_data    = pd.concat([train_df, val_df], ignore_index=True)
    sludge_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── 4. Pre-flight validation ──────────────────────────────────────────────
    _validate_pipeline_inputs(train_df, val_df, test_df)

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    cv_df = pd.DataFrame()
    if not skip_cv:
        cv_df = _run_sc_cv(cv_data, max_steps=cv_steps)
        print_cv_summary(cv_df)

    # ── 6. Final model ────────────────────────────────────────────────────────
    logger.info("Training final SC model (max_steps={}) …", max_steps)
    model = WastewaterTFT(max_steps=max_steps)
    model.fit(cv_data, val_size=model.h)
    forecast_df = model.predict()
    model.save(path=SC_MODELS_DIR / "wastewater_tft")
    q_cols = QuantileColumns.auto_detect(forecast_df)

    # ── 7. Evaluation on holdout ──────────────────────────────────────────────
    eval_result = evaluate(actual_df=test_df, forecast_df=forecast_df, q_cols=q_cols)
    if eval_result.n_observations > 0:
        print_eval_report(eval_result, title="Santa Clara — Holdout Evaluation")
    else:
        logger.warning(
            "Holdout evaluation skipped: forecast window [{} → {}] does not overlap "
            "holdout. Check TRAIN_END_DATE / VAL_END_DATE.",
            forecast_df["ds"].min().date() if not forecast_df.empty else "?",
            forecast_df["ds"].max().date() if not forecast_df.empty else "?",
        )

    # ── 8. Inverse-transform for display ─────────────────────────────────────
    processed_display, forecast_display = _build_display_frames(
        sludge_all, forecast_df, proc, q_cols
    )

    # ── 9. Export ─────────────────────────────────────────────────────────────
    _export_sc_results(
        train_df, val_df, test_df,
        forecast_display,
        cv_df, eval_result,
    )

    console.rule("[bold green] Santa Clara Validation Complete [/bold green]")
    logger.info(
        "Results at {}.  Compare coverage_95 with 9-county run ({}).",
        SC_PROCESSED_DIR, PROCESSED_DIR / "eval_summary.json",
    )


if __name__ == "__main__":
    main()
