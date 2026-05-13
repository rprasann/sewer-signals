"""
serve_dashboard.py — Launch the Dash dashboard from saved artefacts.

Re-runs only the fast data-processing step (no model training) to reconstruct
the RobustScaler, then loads the saved model and forecast parquet and starts
the dashboard at http://localhost:8050.

Usage:
    python serve_dashboard.py
    python serve_dashboard.py --port 8051
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    NWSS_DATE_COL,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_COL,
)
from src.data_pipeline.processor import CAWastewaterProcessor
from src.evaluation.metrics import EvalResult, LeadTimeResult, QuantileColumns
from src.models.tft_model import WastewaterTFT
from src.utils.helpers import console, setup_logger
from src.visualization.dashboard import create_app

# Import the loader functions and helpers from main
from main import (
    _load_ca_ww_csv,
    _load_ca_cases_csv,
    _process_sludge_track,
    _invert_scaling_to_log1p,
)

CA_WW_CSV    = RAW_DIR / CA_WW_FILENAME
CA_CASES_CSV = RAW_DIR / CA_CASES_FILENAME


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Dash dashboard from saved artefacts.")
    p.add_argument("--port", type=int, default=cfg.DASH_PORT)
    return p.parse_args()


def _load_eval_result(path: Path) -> EvalResult | None:
    """Reconstruct an EvalResult from a saved eval_summary.json.

    The JSON may contain bare NaN tokens (written by Python's json module with
    allow_nan=True), so we use pandas to parse it safely.
    """
    if not path.exists():
        logger.warning("eval_summary.json not found at {} — bio-table will show static only.", path)
        return None
    try:
        # pandas read_json handles NaN tokens that vanilla json.load rejects
        series = pd.read_json(path, typ="series")
        d = series.to_dict()

        def _f(key: str) -> float:
            v = d.get(key, float("nan"))
            return float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

        wis_per_county = {
            k.replace("wis_", ""): float(v)
            for k, v in d.items()
            if k.startswith("wis_") and pd.notna(v)
        }

        lt = LeadTimeResult(
            sensitivity=_f("sensitivity"),
            specificity=_f("specificity"),
            auc=_f("auc"),
            mean_lead_days=_f("mean_lead_days"),
            std_lead_days=_f("mean_lead_days"),  # not stored separately; reuse mean as placeholder
        )

        result = EvalResult(
            mean_wis=_f("mean_wis"),
            wis_per_county=wis_per_county,
            coverage_50=_f("coverage_50"),
            coverage_95=_f("coverage_95"),
            smape=_f("smape"),
            n_actual_onsets=int(d.get("n_actual_onsets", 0) or 0),
            n_predicted_alerts=int(d.get("n_predicted_alerts", 0) or 0),
            lead_time=lt,
            n_observations=int(d.get("n_observations", 0) or 0),
        )
        logger.info(
            "Eval loaded — WIS={:.3f}  coverage_95={:.1f}%  SMAPE={:.1f}%",
            result.mean_wis,
            result.coverage_95 * 100,
            result.smape * 100,
        )
        return result

    except Exception as exc:
        logger.warning("Could not load eval_summary.json ({}), continuing without.", exc)
        return None


def _load_cv_results(path: Path) -> pd.DataFrame | None:
    """Load cv_results.csv if present."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["cutoff_date"])
        logger.info("CV results loaded — {} folds.", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not load cv_results.csv ({}), continuing without.", exc)
        return None


def main() -> None:
    load_dotenv()
    args = _parse_args()
    setup_logger()
    cfg.settings = cfg.EnvSettings()

    console.rule("[bold white] Sewer Signals — Dashboard Server [/bold white]")

    # ── 1. Load raw data & refit scaler (fast — no training) ─────────────────
    logger.info("Loading raw CSVs to reconstruct scaler …")
    raw_ww    = _load_ca_ww_csv(CA_WW_CSV)
    raw_cases = _load_ca_cases_csv(CA_CASES_CSV)
    proc, train_df, val_df, test_df = _process_sludge_track(raw_ww, raw_cases)
    logger.info("Scaler reconstructed from train split.")

    # ── 2. Load saved forecast (already in unscaled log1p — display-ready) ───
    forecast_path = PROCESSED_DIR / "forecast.parquet"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"No forecast found at {forecast_path}. Run main.py first."
        )
    forecast_display = pd.read_parquet(forecast_path)
    logger.info("Forecast loaded: {} rows.", len(forecast_display))

    # ── 3. Build processed_display (unscaled log1p actuals) ──────────────────
    sludge_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    processed_display = _invert_scaling_to_log1p(sludge_all, proc, cols=[TARGET_COL])

    # ── 4. Detect quantile columns ────────────────────────────────────────────
    q_cols = QuantileColumns.auto_detect(forecast_display)

    # ── 5. Load saved eval metrics & CV results ───────────────────────────────
    eval_result = _load_eval_result(PROCESSED_DIR / "eval_summary.json")
    cv_results  = _load_cv_results(PROCESSED_DIR / "cv_results.csv")

    # ── 6. Load saved model ───────────────────────────────────────────────────
    logger.info("Loading saved TFT model …")
    model = WastewaterTFT.load()

    # ── 7. Launch dashboard ───────────────────────────────────────────────────
    console.rule("[bold green] Launching Dash Dashboard [/bold green]")
    logger.info("Dashboard at http://localhost:{}", args.port)
    app = create_app(
        processed_df=processed_display,
        forecast_df=forecast_display,
        model=model,
        sludge_df=sludge_all,
        liquid_df=pd.DataFrame(),   # CA pipeline: solid track only
        q_cols=q_cols,
        eval_result=eval_result,
        cv_results=cv_results,
    )
    app.run(host=DASH_HOST, port=args.port, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
