"""
serve_dashboard.py — Launch the Dash dashboard from saved artefacts.

Loads the last exported parquets (or a specific run directory) and starts
the dashboard at http://localhost:8050.  No model training is performed.

Usage:
    uv run serve_dashboard.py                    # load from data/processed/
    uv run serve_dashboard.py --port 8051
    uv run serve_dashboard.py --run run_006_20260530_0106  # specific run
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
    DASH_DEBUG,
    DASH_HOST,
    PROCESSED_DIR,
)
from src.evaluation.metrics import EvalReport, QuantileColumns
from src.utils.helpers import console, setup_logger
from src.utils.run_manager import RUNS_DIR
from src.visualization.dashboard import create_app


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Dash dashboard from saved artefacts.")
    p.add_argument("--port", type=int, default=cfg.DASH_PORT)
    p.add_argument(
        "--run", type=str, default=None,
        help="Run ID to load (e.g. run_006_20260530_0106). Defaults to data/processed/.",
    )
    return p.parse_args()


def _load_eval_summary(path: Path) -> object | None:
    """Load eval_summary.json and return a thin wrapper exposing .to_dict().

    The dashboard calls eval_result.to_dict() to populate its metrics panel.
    The Phase 6 eval_summary.json already stores the flat dict from
    EvalReport.to_dict(), so we just wrap it.
    """
    if not path.exists():
        logger.warning("eval_summary.json not found at {} — metrics panel will be empty.", path)
        return None
    try:
        # pandas read_json handles bare NaN tokens that json.load rejects
        series = pd.read_json(path, typ="series")
        d = series.to_dict()

        class _EvalWrapper:
            """Minimal wrapper so create_app can call .to_dict() unchanged."""
            def __init__(self, data: dict) -> None:
                self._data = data
            def to_dict(self) -> dict:
                return self._data

        result = _EvalWrapper(d)
        wis = d.get("mean_wis", float("nan"))
        cov = d.get("coverage_95", float("nan"))
        mae = d.get("mae", float("nan"))
        logger.info(
            "Eval loaded — WIS={:.3f}  Coverage95={:.1%}  MAE={:.4f}",
            float("nan") if wis is None else float(wis),
            float("nan") if cov is None else float(cov),
            float("nan") if mae is None else float(mae),
        )
        return result
    except Exception as exc:
        logger.warning("Could not load eval_summary.json ({}), continuing without.", exc)
        return None


def _load_cv_results(path: Path) -> pd.DataFrame | None:
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

    # ── Choose source directory ───────────────────────────────────────────────
    if args.run:
        src_dir = RUNS_DIR / args.run
        if not src_dir.exists():
            raise FileNotFoundError(
                f"Run directory not found: {src_dir}\n"
                f"Available runs: {[d.name for d in RUNS_DIR.iterdir() if d.is_dir()]}"
            )
        logger.info("Loading from run archive: {}", src_dir)
    else:
        src_dir = PROCESSED_DIR
        logger.info("Loading from data/processed/ (most recent run)")

    # ── Load required parquets ────────────────────────────────────────────────
    for fname in ("train.parquet", "val.parquet", "test.parquet", "forecast.parquet"):
        if not (src_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found in {src_dir}. "
                "Run 'uv run main.py --no-dash' first to generate artefacts."
            )

    processed_df = pd.concat([
        pd.read_parquet(src_dir / "train.parquet"),
        pd.read_parquet(src_dir / "val.parquet"),
        pd.read_parquet(src_dir / "test.parquet"),
    ], ignore_index=True)
    logger.info("Actuals loaded: {} county-week rows.", len(processed_df))

    forecast_df = pd.read_parquet(src_dir / "forecast.parquet")
    logger.info("Initial forecast loaded: {} rows.", len(forecast_df))

    # ── Load rolling 28-week forecast (for full holdout view) ─────────────────
    rolling_path  = src_dir / "rolling_forecast.parquet"
    rolling_df    = pd.read_parquet(rolling_path) if rolling_path.exists() else pd.DataFrame()
    if not rolling_df.empty:
        logger.info(
            "Rolling holdout forecast loaded: {} rows ({} → {}).",
            len(rolling_df),
            rolling_df["ds"].min().date(),
            rolling_df["ds"].max().date(),
        )
    else:
        logger.info(
            "No rolling_forecast.parquet found — dashboard will show 8-week view only. "
            "Re-run with --rolling-holdout to enable the full 28-week holdout panel."
        )

    # ── Quantile columns ──────────────────────────────────────────────────────
    q_cols = QuantileColumns.auto_detect(forecast_df)

    # ── Metrics + CV results ──────────────────────────────────────────────────
    eval_result = _load_eval_summary(src_dir / "eval_summary.json")
    cv_results  = _load_cv_results(src_dir / "cv_results.csv")

    # ── Launch ────────────────────────────────────────────────────────────────
    console.rule("[bold green] Launching Dash Dashboard [/bold green]")
    logger.info(
        "Dashboard at http://localhost:{} — {} counties, {} forecast rows, "
        "{} rolling rows",
        args.port,
        processed_df["county_fips"].nunique() if "county_fips" in processed_df.columns else "?",
        len(forecast_df),
        len(rolling_df),
    )
    app = create_app(
        processed_df=processed_df,
        forecast_df=forecast_df,
        q_cols=q_cols,
        eval_result=eval_result,
        cv_results=cv_results,
        runs_dir=RUNS_DIR,
        rolling_forecast_df=rolling_df,
    )
    app.run(host=DASH_HOST, port=args.port, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
