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
from src.evaluation.metrics import QuantileColumns
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

    # ── 5. Load saved model ───────────────────────────────────────────────────
    logger.info("Loading saved TFT model …")
    model = WastewaterTFT.load()

    # ── 6. Launch dashboard ───────────────────────────────────────────────────
    console.rule("[bold green] Launching Dash Dashboard [/bold green]")
    logger.info("Dashboard at http://localhost:{}", args.port)
    app = create_app(
        processed_df=processed_display,
        forecast_df=forecast_display,
        model=model,
        sludge_df=sludge_all,
        liquid_df=pd.DataFrame(),   # CA pipeline: solid track only
        q_cols=q_cols,
    )
    app.run(host=DASH_HOST, port=args.port, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
