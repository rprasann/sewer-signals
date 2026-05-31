# Sewer Signals — Production Deployment Blueprint

> **Status:** Technical design document for the Phase 1 two-stage architecture.
> Describes how to move from research model to a weekly automated surveillance system.

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Weekly Automated Pipeline                            │
│                                                                          │
│  WastewaterSCAN / CA WW API                                              │
│          │                                                               │
│          ▼  COVID_Adapter.load_signal()                                  │
│  County Cases / CDC API                                                  │
│          │                                                               │
│          ▼  COVID_Adapter.load_target()                                  │
│                                                                          │
│  ┌───────────────────────────────┐                                       │
│  │  Adapter.clean()              │  QC, unit filter, FIPS normalise      │
│  │  Adapter.build_features()     │  velocity, momentum, lags             │
│  │  Adapter.transform()          │  apply stored RobustScaler            │
│  └───────────────────────────────┘                                       │
│          │                                                               │
│          ▼                                                               │
│  ┌───────────────────────────────┐                                       │
│  │  OutbreakClassifier           │  Stage 1: non-elastic Z + momentum    │
│  │  .classify_df(panel)          │  → triggered / suppressed per county  │
│  └───────────────────────────────┘                                       │
│          │                          │                                    │
│     triggered                  suppressed                                │
│          │                          │                                    │
│          ▼                          ▼                                    │
│  ┌────────────────┐    ┌────────────────────────────┐                   │
│  │  WastewaterTFT │    │  Quiet Prior               │                   │
│  │  (full TFT)    │    │  (flat constant forecast)  │                   │
│  └────────────────┘    └────────────────────────────┘                   │
│          │                          │                                    │
│          └──────────┬───────────────┘                                   │
│                     ▼                                                    │
│             InferenceResult                                              │
│              ├── forecast.parquet                                        │
│              ├── classification.parquet                                  │
│              └── eval_summary.json                                       │
│                     │                                                    │
│          ┌──────────┴─────────────┐                                      │
│          ▼                        ▼                                      │
│     Dashboard               Alerting                                     │
│    (Dash app)          (email / Slack / ESSENCE)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Automated Pipeline — Weekly Execution

### 2.1 Trigger

A cron job (or cloud scheduler) fires every Wednesday at 18:00 PT, after
WastewaterSCAN and the CA Open Data cases feed publish their weekly updates.

### 2.2 Data Pull

```python
# Example — WastewaterSCAN API (HTTPS JSON endpoint)
from src.data_pipeline.adapters import COVID_Adapter

adapter = COVID_Adapter()
# Adapter encapsulates all source-specific logic:
raw_ww    = adapter.load_signal(path=Path("data/raw/ww_latest.csv"))
raw_cases = adapter.load_target(path=Path("data/raw/cases_latest.csv"))
```

For fully automated pulls, replace file paths with streaming API responses.
The adapter's `load_signal` / `load_target` can be extended to call REST APIs
directly without changing downstream code.

### 2.3 Feature Engineering + Scaling

```python
# The scaler is NEVER re-fit in production — only the training-period
# scaler is applied. This prevents concept drift in the scaled feature space.
processed_df = adapter.run(signal_path, target_path, fit_scaler=False)
adapter.load_scalers(Path("models_saved/scalers.joblib"))
processed_df = adapter.transform(processed_df)
```

### 2.4 Two-Stage Inference

```python
from src.pipeline import TwoStagePipeline
from src.models.classifier import OutbreakClassifier
from src.models.forecaster import OutbreakForecaster
from src.models.tft_model import WastewaterTFT

# Load pre-fitted components (no retraining at inference time)
classifier = OutbreakClassifier().fit(train_df)  # fitted at training time
model      = WastewaterTFT.load(Path("models_saved/wastewater_tft"))
forecaster = OutbreakForecaster(model=model)

pipeline = TwoStagePipeline(adapter, classifier, forecaster)
result   = pipeline.run(processed_df)
```

### 2.5 Outputs

| File | Content | Consumer |
|---|---|---|
| `forecast.parquet` | 8-week quantile forecast per county | Dashboard, email alerts |
| `classification.parquet` | Z-score, momentum, triggered flag per week | Audit log, drift monitoring |
| `eval_summary.json` | WIS, coverage, MAE for this week's actuals vs prior forecast | Drift detector |
| `public_health_summary.txt` | LLM-generated bulletin | Epidemiologist email |

---

## 3. Drift Detection & Retraining

### 3.1 Drift Signal: WIS-Based Detection

After each week, the prior week's 1-step-ahead forecast is evaluated against
the now-observed actuals.  WIS is computed per county.

```python
# Pseudocode — runs in background after actuals are collected
from src.evaluation.metrics import wis, QuantileColumns

wis_now  = wis(new_actuals, prior_forecast, q_cols).mean()
wis_base = rolling_mean_wis[-8:]          # 8-week baseline from stable period
drift_z  = (wis_now - wis_base.mean()) / (wis_base.std() + 1e-8)

if drift_z > DRIFT_Z_THRESHOLD:          # e.g. 2.0
    trigger_retraining()
```

### 3.2 Retraining Triggers

| Event | Action |
|---|---|
| WIS drift Z-score > 2.0 for ≥ 2 consecutive weeks | Full retrain on all available data |
| New variant designation (CDC/CDC NWSS flag) | Regime-shift retrain (see §3.4) |
| > 10% of county series below `CLASSIFIER_MIN_SIGNAL` for 4+ weeks | Baseline recalibration |
| Annual cadence | Full retrain regardless of drift |

### 3.3 Retraining Procedure

```bash
# 1. Pull all historical data through current week
# 2. Re-run full pipeline with fit_scaler=True
uv run main.py --skip-cv --no-dash --max-steps 2000

# 3. Run rolling holdout validation to confirm WIS improvement
uv run main.py --skip-cv --no-dash --rolling-holdout

# 4. If WIS improves, promote new model weights to production
# 5. Re-fit OutbreakClassifier on new training period
```

### 3.4 Regime-Shift Protocol (New Variant / Novel Pathogen)

When a new variant displaces the dominant strain (the XBB.1.5 problem):

1. **Detect:** WIS spike + OutbreakClassifier Z-scores decoupled from actual case onset
   (model predicts surge but cases are quiet, or vice versa).

2. **Quarantine:** Hold back the new regime data from the training set until
   ≥ 4 weeks of confirmed variant data are available.

3. **Fine-tune:** Re-train with increased `horizon_weight` on the most recent 8 weeks
   to up-weight the new regime, without discarding outbreak-period priors.

4. **Validate:** Rolling holdout on the 4 weeks immediately following the variant
   declaration.  Only promote if WIS ≤ pre-variant baseline.

---

## 4. Target Variable Pivot

The system is designed for zero-friction target pivots via the adapter pattern.

### 4.1 COVID Cases → Mortality (if CDC stops case reporting)

```python
class COVID_Mortality_Adapter(BaseDatasetAdapter):
    @property
    def target_col(self) -> str:
        return "log1p_deaths"

    def load_target(self, path: Path) -> pd.DataFrame:
        # Load CDC weekly death data, return (county_fips, date, new_deaths)
        ...

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log1p_deaths"] = np.log1p(df["new_deaths"].clip(lower=0))
        return df
```

Everything else — TFT architecture, loss function, evaluator, pipeline — is unchanged.

### 4.2 COVID Cases → %-Positive (ILI sentinel surveillance)

```python
class COVID_PCT_Positive_Adapter(BaseDatasetAdapter):
    @property
    def target_col(self) -> str:
        return "logit_pct_positive"

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        p = df["pct_positive"].clip(1e-4, 1 - 1e-4)
        df["logit_pct_positive"] = np.log(p / (1 - p))
        return df
```

### 4.3 New Pathogen (Influenza, RSV, Mpox)

Implement `Influenza_Adapter(BaseDatasetAdapter)` with:
- `signal_col = "log1p_influenza_conc"` — from WW influenza assay data
- `target_col = "log1p_ili_visits"`    — from ILINet sentinel network
- Pathogen-specific LOD, recovery efficiency QC in `clean()`
- Velocity divergence redefined for influenza kinetics in `build_features()`

The TFT re-trains on the new adapter's output.  OutbreakClassifier re-fits
on the new training distribution.  No model architecture changes needed.

---

## 5. Infrastructure Requirements

### 5.1 Minimal Production Stack

| Component | Spec |
|---|---|
| Inference compute | 2× vCPU, 8 GB RAM, no GPU (TFT inference is CPU-sufficient at weekly cadence) |
| Training compute | 4× vCPU, 16 GB RAM or 1× GPU (NVIDIA T4) for retraining runs |
| Storage | 5 GB / year for parquet artefacts, model weights, scaler state |
| Scheduler | cron (Linux) or AWS EventBridge / GCP Cloud Scheduler |
| Data sources | WastewaterSCAN API (free, public) + CDC Open Data API (free, public) |

### 5.2 Model Versioning

Every run is snapshot-versioned via `run_manager.py`.  Production deployment
promotes a specific `run_id` (e.g. `run_007_20260101`) rather than "latest,"
so rollback is a one-line config change.

```python
PRODUCTION_RUN_ID = "run_007_20260101"
```

### 5.3 Alerting Integration

OutbreakClassifier's `triggered_counties` list flows directly into any alerting
endpoint.  No threshold tuning in the alerting layer — all logic is in the
classifier.

```python
if result.any_triggered:
    send_alert(
        counties=result.triggered_counties,
        forecast=result.forecast_df,
        z_scores=result.clf_df[result.clf_df["triggered"]]["z_score"],
    )
```

---

## 6. Failure Modes & Mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| WW data missing for a county | `adapter.validate_schema()` raises | Fill with NaN; classifier returns all-False for that county; dashboard shows "data gap" banner |
| API endpoint down | FileNotFoundError in `load_signal()` | Retry 3× with backoff; use last available parquet as fallback |
| TFT inference OOM | RuntimeError in `OutbreakForecaster.predict()` | Fall back to quiet prior for ALL counties; log CRITICAL |
| Scaler not fitted | `RuntimeError` from `_apply_scaling()` | Catch in pipeline; fall back to unscaled input; send alert to ops |
| Classifier drift (all counties triggered) | > 80% trigger rate for 3 consecutive weeks | Automatic baseline recalibration (re-fit `OutbreakClassifier` on recent quiet weeks) |
| New FIPS introduced in API | KeyError in `FIPS_TO_COUNTY` | Log warning; skip new FIPS; include in next scheduled retrain |
