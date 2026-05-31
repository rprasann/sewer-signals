"""
Run versioning utilities for Sewer Signals pipeline.

Every pipeline run is snapshotted into data/runs/run_NNN_<label>/ with a
run_meta.json index file.  The dashboard uses list_runs() and load_run_data()
to populate the run selector dropdown.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR

RUNS_DIR: Path = PROCESSED_DIR.parent / "runs"   # data/runs/


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    run_id:     str              # "run_001"
    label:      str              # human-readable label
    run_date:   str              # "2025-05-13"
    counties:   list[str]        # county names present in run
    n_counties: int
    phase:      str              # "Phase 3", "Phase 4", …
    metrics:    dict             # mean_wis, coverage_95, smape, auc (may have NaN)
    notes:      str
    run_dir:    Path

    @property
    def dropdown_label(self) -> str:
        """One-line label shown in the run selector dropdown."""
        wis  = self.metrics.get("mean_wis")
        cov  = self.metrics.get("coverage_95")
        f1   = self.metrics.get("f1")
        wis_str = f"WIS {wis:.3f}" if isinstance(wis, float) and not _isnan(wis) else ""
        cov_str = f"Cov95 {cov*100:.1f}%" if isinstance(cov, float) and not _isnan(cov) else ""
        f1_str  = f"F1 {f1:.2f}" if isinstance(f1, float) and not _isnan(f1) else ""
        parts = [p for p in [wis_str, cov_str, f1_str] if p]
        suffix = f"  [{', '.join(parts)}]" if parts else ""
        cnt_word = "county" if self.n_counties == 1 else "counties"
        return f"{self.label}  ·  {self.run_date}  ·  {self.n_counties} {cnt_word}{suffix}"


def _isnan(v) -> bool:
    try:
        import math
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapshot_run(
    proc_dir:  Path | None     = None,
    run_label: str             = "",
    counties:  list[str]       = (),
    phase:     str             = "Phase 4",
    notes:     str             = "",
    runs_dir:  Path | None     = None,
) -> tuple[str, Path]:
    """Copy all artefacts from proc_dir into a new versioned run directory.

    Returns (run_id, run_dir).  The run_meta.json inside run_dir becomes the
    index entry that list_runs() and the dashboard dropdown use.
    """
    if proc_dir is None:
        proc_dir = PROCESSED_DIR
    proc_dir = Path(proc_dir)

    if runs_dir is None:
        runs_dir = RUNS_DIR
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Next sequential run number
    existing = sorted([d for d in runs_dir.iterdir() if d.is_dir() and (d / "run_meta.json").exists()])
    next_num = len(existing) + 1

    safe = run_label.lower().replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    run_id  = f"run_{next_num:03d}_{safe}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy artefacts
    _ARTEFACTS = [
        "train.parquet", "val.parquet", "test.parquet",
        "forecast.parquet", "rolling_forecast.parquet", "cv_results.csv",
        "eval_summary.json", "scalers.joblib",
        "public_health_summary.txt",
        "classification.parquet",      # OutbreakClassifier output (--two-stage)
        "two_stage_forecast.parquet",  # gated forecast (TFT + quiet prior)
    ]
    for fname in _ARTEFACTS:
        src = proc_dir / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)

    # Extract metrics from eval_summary.json
    eval_path = run_dir / "eval_summary.json"
    metrics: dict = {}
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        for k in ("mean_wis", "coverage_50", "coverage_95", "mae", "pinball_ratio", "precision", "recall", "f1"):
            metrics[k] = ev.get(k)

    meta = {
        "run_id":     run_id,
        "label":      run_label,
        "run_date":   datetime.now().strftime("%Y-%m-%d"),
        "counties":   list(counties),
        "n_counties": len(counties),
        "phase":      phase,
        "metrics":    metrics,
        "notes":      notes,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return run_id, run_dir


def list_runs(runs_dir: Path | None = None) -> list[RunMeta]:
    """Return all archived runs sorted oldest → newest."""
    if runs_dir is None:
        runs_dir = RUNS_DIR
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []

    out: list[RunMeta] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "run_meta.json"
        if not meta_path.exists():
            continue
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        out.append(RunMeta(
            run_id     = m.get("run_id", d.name),
            label      = m.get("label", d.name),
            run_date   = m.get("run_date", ""),
            counties   = m.get("counties", []),
            n_counties = m.get("n_counties", 0),
            phase      = m.get("phase", ""),
            metrics    = m.get("metrics", {}),
            notes      = m.get("notes", ""),
            run_dir    = d,
        ))
    return out


def load_run_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Load (processed_df, forecast_df, q_cols, cv_df, eval_dict, rolling_forecast_df, clf_df) from a run directory.

    processed_df     : train + val + test concatenated — full historical timeline.
    rolling_forecast_df : stitched 28-week rolling holdout forecast (empty when not available).
    clf_df           : OutbreakClassifier output (empty when --two-stage was not used).
    All DataFrames are already in unscaled log1p space (as exported by main.py).
    """
    from src.evaluation.metrics import QuantileColumns

    run_dir = Path(run_dir)

    # Full historical actuals
    frames = []
    for split in ("train", "val", "test"):
        p = run_dir / f"{split}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    processed_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Forecast
    fc_path = run_dir / "forecast.parquet"
    forecast_df = pd.read_parquet(fc_path) if fc_path.exists() else pd.DataFrame()

    # Quantile column mapping
    try:
        q_cols = QuantileColumns.auto_detect(forecast_df) if not forecast_df.empty else QuantileColumns()
    except Exception:
        q_cols = QuantileColumns()

    # CV results
    cv_path = run_dir / "cv_results.csv"
    cv_df   = pd.DataFrame()
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        if "cutoff_date" in cv_df.columns:
            cv_df["cutoff_date"] = pd.to_datetime(cv_df["cutoff_date"])

    # Eval summary
    eval_path = run_dir / "eval_summary.json"
    eval_dict = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    # Rolling holdout forecast (optional — only present when --rolling-holdout was used)
    rolling_path = run_dir / "rolling_forecast.parquet"
    rolling_forecast_df = pd.read_parquet(rolling_path) if rolling_path.exists() else pd.DataFrame()

    # Classification output (present only when --two-stage was used)
    clf_path = run_dir / "classification.parquet"
    clf_df   = pd.read_parquet(clf_path) if clf_path.exists() else pd.DataFrame()

    return processed_df, forecast_df, q_cols, cv_df, eval_dict, rolling_forecast_df, clf_df
