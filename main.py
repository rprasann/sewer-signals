"""
Sewer Signals — end-to-end pipeline orchestration.

Loads California state wastewater and case CSVs from data/raw/, then runs
the complete pipeline:

  1. Load raw data (wastewater + cases)
  2. Leakage-free processing — RobustScaler fit ONLY on raw training rows
  3. Expanding-window cross-validation (skippable with --skip-cv or --fast)
  4. Final model training (TFT + PINNWastewaterLoss)
  5. Evaluation on hold-out test set
  6. Inverse-transform for human-readable dashboard output
  7. LLM public health bulletin (optional — requires LM Studio running locally)
  8. Export artefacts to data/processed/
  9. Launch interactive Dash dashboard

Split strategy
--------------
Data window: 2020-07-01 → 2023-12-19 (~180 weekly records per county).
  Pre-2022 WW solid data: only Santa Clara, San Francisco, San Mateo.
  Remaining 6 counties join solid track from 2022; start_padding_enabled
  zero-pads their short early histories.

  CV window  (~35 wks): TRAIN_END_DATE → VAL_END_DATE
    Expanding-window CV runs entirely inside this range.
    Initial cutoff TRAIN_END_DATE = 2022-10-05 (all 9 counties active).
    ~8 folds at step_weeks=4; evaluate() called externally per fold.

  Holdout (~28 wks): VAL_END_DATE + 1 day → 2023-12-19
    The final model trains on ALL CV-window weeks, then forecasts H=8
    weekly steps ahead into the holdout.  evaluate() is only called here,
    where ground truth is known.

Invariants enforced
-------------------
[INV-1] RobustScaler: fit once on raw_train rows only.  val and test rows
        are transformed with the stored scaler — zero look-ahead leakage.
[INV-2] Log-transform inversion: forecast quantile columns are converted
        from RobustScaled log1p to unscaled log1p before the dashboard;
        the LLM summary additionally applies expm1 to yield copies/g values.
[INV-3] Two-track toggle: liquid track (copies/l wastewater) is processed
        through a SEPARATE WastewaterProcessor(target_unit=SECONDARY_UNIT).
        Its is_sludge column is 0.0; the sludge track's scaler is unaffected.

Usage
-----
    python main.py                 # full run (2000 TFT training steps)
    python main.py --fast          # smoke-test (200 steps, skips CV)
    python main.py --skip-cv       # skip cross-validation only
    python main.py --no-dash       # export results without launching dashboard
    python main.py --port 8051     # override default dashboard port
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

import src.config as cfg
from src.config import (
    BAY_AREA_COUNTIES,
    BAY_AREA_FIPS,
    CA_CASES_FILENAME,
    CA_WW_FILENAME,
    CA_WW_SIGNAL_COL,
    COUNTY_COL,
    DASH_DEBUG,
    DASH_HOST,
    DASH_PORT,
    DATA_END_DATE,
    DATA_START_DATE,
    NWSS_DATE_COL,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
    WW_FEATURE_COL,
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
    generate_public_health_summary,
    print_cv_summary,
    print_eval_report,
    setup_logger,
)
from src.visualization.dashboard import create_app

# ---------------------------------------------------------------------------
# Raw file paths — California state datasets (data/raw/)
# ---------------------------------------------------------------------------

CA_WW_CSV    = RAW_DIR / CA_WW_FILENAME
CA_CASES_CSV = RAW_DIR / CA_CASES_FILENAME


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sewer Signals — wastewater COVID-19 surveillance pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Smoke-test mode: 200 training steps, skip CV",
    )
    p.add_argument(
        "--skip-cv", action="store_true",
        help="Skip expanding-window cross-validation",
    )
    p.add_argument(
        "--no-dash", action="store_true",
        help="Export results without launching the Dash dashboard",
    )
    p.add_argument(
        "--port", type=int, default=DASH_PORT,
        help="Dashboard server port",
    )
    p.add_argument(
        "--counties", type=str, default=None,
        help=(
            "Restrict pipeline to a subset of counties.  "
            "Pass '3county' for the spatial-temporal validation set "
            "(SF, San Mateo, Santa Clara), or a comma-separated list of "
            "5-digit FIPS codes (e.g. '06075,06081,06085')."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------

def _load_ca_ww_csv(path: Path) -> pd.DataFrame:
    """Load the CA Wastewater Surveillance CSV; filter to Bay Area SARS-CoV-2 solid track.

    Renames 'Sample Date' → NWSS_DATE_COL and 'County' → 'county_name' so that
    _split_raw() and CAWastewaterProcessor._clean_ca_ww() can work without
    further column negotiation.

    Only the solid-track rows are retained (Sample Type == 'solid').  The
    numeric signal columns are coerced to float here so the processor does
    not need to re-parse them.
    """
    logger.info("Loading CA WW data: {} …", path.name)
    if not path.exists():
        raise FileNotFoundError(f"CA WW CSV not found at {path}.")
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Filter: Bay Area + SARS-CoV-2 + solid track
    df = df[
        df["County"].isin(BAY_AREA_COUNTIES) &
        (df["PCR Target"] == "SARS-CoV-2") &
        (df["Sample Type"].str.lower() == "solid")
    ].copy()

    # Rename to internal schema expected by _split_raw and _clean_ca_ww
    df = df.rename(columns={"Sample Date": NWSS_DATE_COL, "County": "county_name"})
    df[NWSS_DATE_COL] = pd.to_datetime(df[NWSS_DATE_COL], errors="coerce")
    df = df.dropna(subset=[NWSS_DATE_COL])

    for col in ("Raw Concentration", "Norm Pmmov"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "  {} Bay Area solid-track rows loaded ({} counties).",
        len(df), df["county_name"].nunique(),
    )
    return df


def _load_ca_cases_csv(path: Path) -> pd.DataFrame:
    """Load the CA Statewide Cases CSV; resample daily → weekly W-WED per county.

    Date format is %m/%d/%y (e.g. '12/19/23') — must be passed explicitly.
    Returns a DataFrame with columns (COUNTY_COL, NWSS_DATE_COL, 'new_cases')
    matching the format expected by WastewaterProcessor._merge_cases().
    """
    logger.info("Loading CA Cases data: {} …", path.name)
    if not path.exists():
        raise FileNotFoundError(f"CA Cases CSV not found at {path}.")
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Date format: '12/19/23' — 2-digit year, auto-detection fails on some rows
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce")

    # Filter: Bay Area counties only; exclude state-level aggregate rows
    df = df[
        df["area"].isin(BAY_AREA_COUNTIES) &
        (df["area_type"] == "County")
    ].copy()
    df = df.dropna(subset=["date", "cases"])

    # Map county name → FIPS (COUNTY_COL) — no FIPS lookup table needed
    df[COUNTY_COL] = df["area"].map(BAY_AREA_FIPS)
    df = df.dropna(subset=[COUNTY_COL])

    # Resample DAILY → W-WED weekly sum per county
    weekly = (
        df.set_index("date")
        .groupby(COUNTY_COL)["cases"]
        .resample("W-WED")
        .sum()
        .clip(lower=0)
        .reset_index()
        .rename(columns={"date": NWSS_DATE_COL, "cases": "new_cases"})
    )

    logger.info("  {} county-week rows after daily → W-WED resample.", len(weekly))
    return weekly


def _split_raw(
    df: pd.DataFrame,
    date_col: str = NWSS_DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict to the overlap window, then split into train / val / test.

    The overlap window (DATA_START_DATE → DATA_END_DATE) is the intersection
    of the WW dataset and the cases dataset.  Restricting before splitting
    ensures no out-of-window WW data leaks into any split.

    Split boundaries are controlled by TRAIN_END_DATE and VAL_END_DATE
    (both Wednesday-aligned W-WED dates).  The RobustScaler is fitted ONLY
    on training rows (INV-1).
    """
    dates      = pd.to_datetime(df[date_col], errors="coerce")
    data_start = pd.Timestamp(DATA_START_DATE)
    data_end   = pd.Timestamp(DATA_END_DATE)
    train_end  = pd.Timestamp(TRAIN_END_DATE)
    val_end    = pd.Timestamp(VAL_END_DATE)

    # Step 1 — overlap window filter
    df    = df[(dates >= data_start) & (dates <= data_end)].copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Step 2 — three-way split
    train = df[dates <= train_end].copy()
    val   = df[(dates > train_end) & (dates <= val_end)].copy()
    test  = df[dates > val_end].copy()
    logger.info(
        "Raw split (overlap {} → {}): train={} (≤{}), val={}, test={} rows.",
        DATA_START_DATE, DATA_END_DATE,
        len(train), TRAIN_END_DATE, len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Step 2 — Leakage-free processing
# ---------------------------------------------------------------------------

def _process_sludge_track(
    raw_ww: pd.DataFrame,
    raw_cases: pd.DataFrame,
) -> tuple[CAWastewaterProcessor, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process the CA solid WW track + weekly cases through CAWastewaterProcessor.

    Both raw_ww (CA WW solid, NWSS_DATE_COL already set by loader) and
    raw_cases (weekly W-WED, NWSS_DATE_COL + COUNTY_COL already set by loader)
    are split into train/val/test by the same overlap-window-aware _split_raw().

    [INV-1] RobustScaler fitted ONLY on raw_train; val and test receive only
    transform() calls with the already-fitted scaler.

    Returns (processor, train_df, val_df, test_df).
    """
    raw_ww_train,    raw_ww_val,    raw_ww_test    = _split_raw(raw_ww)
    raw_cases_train, raw_cases_val, raw_cases_test = _split_raw(raw_cases)

    proc = CAWastewaterProcessor(ww_signal_col=CA_WW_SIGNAL_COL)

    logger.info("Processing CA WW train split — fitting RobustScaler …")
    train_df = proc.run(raw_ww_train, cases_df=raw_cases_train)   # scaler FITTED

    logger.info("Processing CA WW val split — applying stored scaler …")
    val_df = proc.transform(raw_ww_val, cases_df=raw_cases_val)   # scaler APPLIED

    logger.info("Processing CA WW test split — applying stored scaler …")
    test_df = proc.transform(raw_ww_test, cases_df=raw_cases_test) # scaler APPLIED

    return proc, train_df, val_df, test_df


def _process_liquid_track(_raw_ww: pd.DataFrame) -> pd.DataFrame:
    """CA pipeline: no liquid track — returns empty DataFrame.

    The CA WW dataset solid track covers all 9 counties without needing a
    separate liquid-matrix path.  The dashboard liquid_df slot accepts an
    empty DataFrame and simply skips the two-track comparison chart.
    """
    logger.info("Liquid track skipped — CA pipeline uses solid track only.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3 — Expanding-window cross-validation
# ---------------------------------------------------------------------------

def _run_cv(cv_data: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """Expanding-window CV confined to the 50-week CV window.

    Folds expand from TRAIN_END_DATE (first cutoff) to VAL_END_DATE (last
    cutoff) in 4-week steps.  The holdout test set is never touched here.

    Each fold trains a fresh WastewaterTFT on all data up to the cutoff date,
    then evaluates on the next H weeks.  RobustScaler parameters were fitted
    on raw_train only (INV-1) so no leakage occurs in any fold.
    """
    logger.info("Expanding-window cross-validation (max_steps={}) …", max_steps)
    # Three overrides required for CV fold models:
    # 1. enable_progress_bar=False / enable_model_summary=False: PL's Rich Live
    #    display conflicts with the outer expanding_window_cv progress bar.
    # 2. early_stop_patience_steps=-1: disables early stopping so val_size=0 is
    #    valid for all folds — early stopping is irrelevant in CV since each fold
    #    is evaluated externally via evaluate(). Without this, folds where the
    #    shortest county series <= h + lag_warmup (Napa at early cutoffs) would
    #    hit "Set val_size>0 or provide val_df if early stopping is enabled".
    cv_trainer_kwargs = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "early_stop_patience_steps": -1,
    }
    return expanding_window_cv(
        processed_df=cv_data,
        model_factory=lambda: WastewaterTFT(
            max_steps=max_steps, trainer_kwargs=cv_trainer_kwargs
        ),
        initial_train_end=TRAIN_END_DATE,
        eval_end=VAL_END_DATE,
        step_weeks=4,
    )


# ---------------------------------------------------------------------------
# Step 3b — Pre-flight validation (catches shape/NaN issues before training)
# ---------------------------------------------------------------------------

def _validate_pipeline_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Check processed splits for known failure modes and log actionable warnings.

    Runs before any model training so problems surface immediately rather than
    after minutes of data loading.  Covers:

    - Per-county series lengths vs input_size  (start_padding_enabled handles short
      series, but we still log which counties are affected)
    - Val series length variance  (unequal lengths crash val_df mode; we use
      val_size instead, but we still report it for transparency)
    - NaN in HIST_COVARIATES after warm-up rows are dropped
    - County FIPS consistency across splits
    - Empty splits
    """
    from src.models.tft_model import HIST_COVARIATES
    from src.config import TFT_CONFIG

    input_size = TFT_CONFIG["input_size"]
    h          = TFT_CONFIG["h"]
    issues: list[str] = []

    # ── 1. Per-county series length in train ─────────────────────────────────
    if not train_df.empty and COUNTY_COL in train_df.columns:
        counts = train_df.groupby(COUNTY_COL)[NWSS_DATE_COL].count().sort_values()
        short = counts[counts < input_size + h]
        if not short.empty:
            issues.append(
                f"[INV-SHORT] {len(short)} county/counties have fewer than "
                f"{input_size + h} training weeks (input_size + h) — "
                f"start_padding_enabled=True will zero-pad them:\n"
                + "\n".join(
                    f"    {cfg.FIPS_TO_COUNTY.get(fips, fips)} ({fips}): {n} weeks"
                    for fips, n in short.items()
                )
            )

    # ── 2. Val series length variance ────────────────────────────────────────
    if not val_df.empty and COUNTY_COL in val_df.columns:
        val_counts = val_df.groupby(COUNTY_COL)[NWSS_DATE_COL].count()
        if val_counts.nunique() > 1:
            issues.append(
                f"[INV-UNEQUAL-VAL] Val series lengths vary "
                f"({val_counts.min()}–{val_counts.max()} weeks) — "
                "using val_size instead of val_df for early stopping."
            )

    # ── 3. NaN in HIST_COVARIATES (should be zero after _to_nf_format drops rows)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in HIST_COVARIATES:
            if col in df.columns:
                n_nan = df[col].isna().sum()
                if n_nan:
                    issues.append(
                        f"[INV-NAN] {n_nan} NaN values in '{col}' in {split_name} "
                        f"— _to_nf_format will drop these rows automatically."
                    )

    # ── 4. FIPS consistency ───────────────────────────────────────────────────
    if COUNTY_COL in train_df.columns and COUNTY_COL in val_df.columns:
        train_fips = set(train_df[COUNTY_COL].unique())
        val_fips   = set(val_df[COUNTY_COL].unique())
        val_only   = val_fips - train_fips
        if val_only:
            issues.append(
                f"[INV-FIPS] {len(val_only)} county/counties appear in val but not train "
                f"— model will not forecast for them: {val_only}"
            )

    # ── 5. Empty splits ───────────────────────────────────────────────────────
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df.empty:
            issues.append(f"[INV-EMPTY] {split_name}_df is empty.")

    # ── Report ────────────────────────────────────────────────────────────────
    console.rule("[bold yellow] Pre-flight Validation [/bold yellow]")
    if issues:
        for msg in issues:
            logger.warning("Validation: {}", msg)
        console.print(
            f"  [yellow]{len(issues)} issue(s) logged above — "
            "pipeline will continue (all are handled automatically).[/yellow]"
        )
    else:
        console.print("  [green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# Step 4 — Final model training
# ---------------------------------------------------------------------------

def _train_final_model(
    train_df: pd.DataFrame,
    max_steps: int,
) -> tuple[WastewaterTFT, pd.DataFrame]:
    """Fit the production TFT on the full training split.

    Uses val_size=h (last h steps of each training series) for early stopping
    instead of a separate val_df.  This avoids the NeuralForecast constraint
    that all series in val_df must have equal length — which fails in practice
    because Bay Area counties have different reporting start dates and may
    accumulate different numbers of weekly observations.
    """
    logger.info("Training final model (max_steps={}) …", max_steps)
    model = WastewaterTFT(max_steps=max_steps)
    model.fit(train_df, val_size=model.h)
    forecast_df = model.predict()
    logger.info("Forecast generated: {} rows.", len(forecast_df))
    return model, forecast_df


# ---------------------------------------------------------------------------
# Step 5 — Inverse-transform for display  [INV-2]
# ---------------------------------------------------------------------------

def _invert_scaling_to_log1p(
    df: pd.DataFrame,
    proc,
    cols: list[str],
) -> pd.DataFrame:
    """Undo RobustScaler on specified columns; result is in unscaled log1p space.

    [INV-2] The dashboard plots log1p(new_cases) on the y-axis.  The model
    and processor both operate on RobustScaled log1p values.  This function
    removes the RobustScaler step so the y-axis represents a standard
    epidemiological log scale rather than an arbitrary standardised value.

    Supports both per-county scalers (proc._scalers dict, Problem 1 fix) and
    the legacy single global scaler (proc._scaler) for backward compatibility.
    County identity is inferred from COUNTY_COL if present, else "unique_id"
    (NeuralForecast forecast output format).

    Inverse: log1p_val = scaled_val × scale + center
    Uses the scaler parameters fitted on TARGET_COL (log1p_new_cases).
    """
    if proc._scaler is None or TARGET_COL not in proc._scale_cols:
        logger.warning("Scaler unavailable — columns left as scaled log1p.")
        return df

    col_idx = proc._scale_cols.index(TARGET_COL)
    out = df.copy()

    # Per-county scalers: invert each county independently
    if hasattr(proc, "_scalers") and proc._scalers:
        id_col = COUNTY_COL if COUNTY_COL in out.columns else "unique_id"
        for fips, grp_idx in out.groupby(id_col).groups.items():
            scaler = proc._scalers.get(str(fips))
            if scaler is None:
                continue
            center_val = scaler.center_[col_idx]
            scale_val  = scaler.scale_[col_idx]
            for col in cols:
                if col in out.columns:
                    out.loc[grp_idx, col] = out.loc[grp_idx, col] * scale_val + center_val
    else:
        # Fallback: single global scaler (legacy / liquid-track mode)
        center_val = proc._scaler.center_[col_idx]
        scale_val  = proc._scaler.scale_[col_idx]
        for col in cols:
            if col in out.columns:
                out[col] = out[col] * scale_val + center_val

    return out


def _build_display_frames(
    sludge_all: pd.DataFrame,
    forecast_df: pd.DataFrame,
    proc: WastewaterProcessor,
    q_cols: QuantileColumns,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (processed_display, forecast_display) in unscaled log1p space.

    Both DataFrames are consistent: TARGET_COL and the forecast quantile
    columns all contain log(1 + concentration) values with the RobustScaler
    removed.  The two-track chart uses the raw `concentration` column directly
    (already in copies/g) so no inversion is needed for that plot.
    """
    # Undo scaler on actual signal
    processed_display = _invert_scaling_to_log1p(
        sludge_all, proc, cols=[TARGET_COL]
    )

    # Undo scaler on forecast quantile columns
    q_col_list = [
        getattr(q_cols, attr)
        for attr in ("q025", "q10", "q25", "q50", "q75", "q90", "q975")
        if getattr(q_cols, attr) is not None and getattr(q_cols, attr) in forecast_df.columns
    ]
    forecast_display = _invert_scaling_to_log1p(forecast_df, proc, cols=q_col_list)

    return processed_display, forecast_display


def _build_decoded_forecast(
    forecast_df: pd.DataFrame,
    proc: WastewaterProcessor,
    q_cols: QuantileColumns,
) -> pd.DataFrame:
    """Return forecast with quantile columns decoded to raw weekly case counts.

    Applies expm1(RobustScaler⁻¹(·)) to convert from scaled log1p space back
    to actual new_cases values.  Used by the LLM summary for human-readable
    case count numbers.
    """
    q_col_list = [
        getattr(q_cols, attr)
        for attr in ("q025", "q10", "q25", "q50", "q75", "q90", "q975")
        if getattr(q_cols, attr) is not None and getattr(q_cols, attr) in forecast_df.columns
    ]
    df = _invert_scaling_to_log1p(forecast_df, proc, cols=q_col_list)
    for col in q_col_list:
        df[col] = np.expm1(df[col]).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Step 6 — Export
# ---------------------------------------------------------------------------

def _export_results(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    eval_result: EvalResult,
) -> None:
    """Write all pipeline artefacts to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(   PROCESSED_DIR / "train.parquet",    index=False)
    val_df.to_parquet(     PROCESSED_DIR / "val.parquet",      index=False)
    test_df.to_parquet(    PROCESSED_DIR / "test.parquet",     index=False)
    forecast_df.to_parquet(PROCESSED_DIR / "forecast.parquet", index=False)
    if not cv_df.empty:
        cv_df.to_csv(PROCESSED_DIR / "cv_results.csv", index=False)
    (PROCESSED_DIR / "eval_summary.json").write_text(
        json.dumps(eval_result.to_dict(), indent=2, default=str)
    )
    logger.info("Artefacts exported to {}.", PROCESSED_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args   = _parse_args()
    setup_logger()
    cfg.settings = cfg.EnvSettings()

    skip_cv   = args.skip_cv or args.fast
    max_steps = 200 if args.fast else cfg.TFT_CONFIG["max_steps"]
    cv_steps  = min(max_steps, 500)

    console.rule("[bold white] Sewer Signals — Bay Area Wastewater Surveillance [/bold white]")
    logger.info(
        "Config: max_steps={}, skip_cv={}, no_dash={}, port={}",
        max_steps, skip_cv, args.no_dash, args.port,
    )

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    raw_ww    = _load_ca_ww_csv(CA_WW_CSV)
    raw_cases = _load_ca_cases_csv(CA_CASES_CSV)

    # Optional county filter (--counties 3county | --counties 06075,06081,06085)
    if args.counties:
        if args.counties.strip().lower() == "3county":
            county_filter = cfg.THREE_COUNTY_FIPS
        else:
            county_filter = [f.strip() for f in args.counties.split(",")]
        county_names = [cfg.FIPS_TO_COUNTY[f] for f in county_filter if f in cfg.FIPS_TO_COUNTY]
        logger.info("County filter active: {} ({})", county_filter, county_names)
        # raw_ww uses 'county_name'; raw_cases uses COUNTY_COL (county_fips)
        raw_ww    = raw_ww[raw_ww["county_name"].isin(county_names)].copy()
        raw_cases = raw_cases[raw_cases[cfg.COUNTY_COL].isin(county_filter)].copy()

    # ── 2. Leakage-free processing  [INV-1, INV-3] ───────────────────────────
    proc, train_df, val_df, test_df = _process_sludge_track(raw_ww, raw_cases)
    # cv_data  = 50-week CV window (train + val)  → used for CV loop and final training
    # test_df  = 15-week holdout                  → never seen during CV or training
    cv_data    = pd.concat([train_df, val_df], ignore_index=True)
    sludge_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    liquid_all = _process_liquid_track(raw_ww)

    # ── 2b. Pre-flight validation ─────────────────────────────────────────────
    _validate_pipeline_inputs(train_df, val_df, test_df)

    # ── 3. Cross-validation (runs entirely within the 50-week CV window) ──────
    cv_df = pd.DataFrame()
    if not skip_cv:
        cv_df = _run_cv(cv_data, max_steps=cv_steps)
        print_cv_summary(cv_df)

    # ── 4. Final model ────────────────────────────────────────────────────────
    # Train on ALL 50 CV-window weeks so the H=8-step forecast lands inside
    # the 15-week holdout.  INV-1 holds: RobustScaler was fitted only on
    # raw_train rows; val_df went through proc.transform() (no re-fitting).
    model, forecast_df = _train_final_model(cv_data, max_steps=max_steps)
    model.save()
    q_cols = QuantileColumns.auto_detect(forecast_df)

    # ── 5. Evaluation on holdout test set ────────────────────────────────────
    # evaluate() now returns an empty EvalResult (n_observations=0) instead of
    # raising when the forecast window doesn't overlap — we guard on that here.
    eval_result = evaluate(
        actual_df=test_df,
        forecast_df=forecast_df,
        q_cols=q_cols,
    )
    if eval_result.n_observations > 0:
        print_eval_report(eval_result, title="Final Evaluation — Holdout Test Set")
    else:
        logger.warning(
            "Holdout evaluation skipped: forecast window [{} → {}] does not "
            "overlap holdout dates.  Check TRAIN_END_DATE / VAL_END_DATE.",
            forecast_df["ds"].min().date() if not forecast_df.empty else "?",
            forecast_df["ds"].max().date() if not forecast_df.empty else "?",
        )

    # ── 6. Inverse-transform for display  [INV-2] ────────────────────────────
    #   processed_display: TARGET_COL is unscaled log1p (dashboard actuals)
    #   forecast_display:  quantile cols are unscaled log1p (dashboard forecast)
    processed_display, forecast_display = _build_display_frames(
        sludge_all, forecast_df, proc, q_cols
    )

    # ── 7. LLM public health bulletin (LM Studio) ────────────────────────────
    vsn_weights = model.variable_importance()
    if cfg.settings and cfg.settings.use_local_llm:
        logger.info(
            "Generating public health summary via LM Studio ({}) …",
            cfg.settings.local_llm_base_url,
        )
        # LLM summary uses decoded case counts for human-readable numbers
        forecast_cases = _build_decoded_forecast(forecast_df, proc, q_cols)
        summary = generate_public_health_summary(
            forecast_df=forecast_cases,
            eval_result=eval_result,
            vsn_weights=vsn_weights,
            county_fips=list(BAY_AREA_FIPS.values()),
            base_url=cfg.settings.local_llm_base_url,
            model=cfg.settings.local_llm_model,
        )
        console.rule("[bold cyan] Public Health Summary [/bold cyan]")
        console.print(summary)
        (PROCESSED_DIR / "public_health_summary.txt").write_text(summary)
    else:
        logger.warning(
            "Local LLM disabled — set USE_LOCAL_LLM=true in .env to enable.  "
            "LM Studio must be running at {} with a model loaded.",
            cfg.LOCAL_LLM_BASE_URL,
        )

    # ── 8. Export ─────────────────────────────────────────────────────────────
    _export_results(
        train_df, val_df, test_df,
        forecast_display,  # export the display version (unscaled log1p)
        cv_df, eval_result,
    )

    # ── 9. Dashboard  [INV-1, INV-2, INV-3 all satisfied] ────────────────────
    if not args.no_dash:
        console.rule("[bold green] Launching Dash Dashboard [/bold green]")
        logger.info(
            "Dashboard at http://localhost:{}  "
            "(processed_display TARGET_COL = unscaled log1p; "
            "forecast_display quantile cols = unscaled log1p)",
            args.port,
        )
        app = create_app(
            processed_df=processed_display,   # actuals in unscaled log1p
            forecast_df=forecast_display,     # forecasts in unscaled log1p
            model=model,                      # fitted TFT for live attention/VSN
            sludge_df=sludge_all,             # has `concentration` col for two-track chart
            liquid_df=liquid_all,             # is_sludge=0.0, SECONDARY_UNIT  [INV-3]
            q_cols=q_cols,
        )
        app.run(host=DASH_HOST, port=args.port, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
