# Sewer Signals — Project DNA (memory.md)

> **Permanent reference.** This file does not track in-progress work.
> It holds the scientific mission, architecture decisions, and design rationale
> that are stable across sessions. Update only when a fundamental decision changes.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | Sewer Signals: Attention-Based Forecasting of COVID-19 Outbreaks from Wastewater |
| **Repository** | `/Users/prasann/Dev/support_vectors/LLM/labs/wastewater` |
| **Geography** | Multi-geography capable; default = 9-county SF Bay Area (CA FIPS 06001–06097) |
| **Active geographies** | `bay_area` (default), `upper_socal` (Kern, LA, SLO, Santa Barbara, Ventura) |
| **Data source** | CA Wastewater Surveillance CSV + CA Statewide Cases/Deaths/Tests CSV (covers all CA counties) |
| **Primary unit** | copies/g dry sludge (solid track) |
| **Secondary unit** | copies/l wastewater (liquid track, Section 4.1 comparison only) |
| **Validation set** | Defined per geography in YAML; Bay Area default = SF/San Mateo/Santa Clara |

---

## 2. Scientific Hypothesis: The Momentum-Lead Relationship

Wastewater SARS-CoV-2 signal leads clinical case counts by approximately **1–3 weeks**.
The model must learn three temporal relationships:

1. **The Lead:** A sustained rise in WW concentration precedes a case surge.
   The rate of change (velocity) is a stronger predictor of surge onset than level alone.
2. **The Lag:** WW signal decays *after* the clinical peak — it confirms recovery.
3. **The Volatility:** Clinical case counts are "spiky" (high week-over-week variance).
   The model must produce wide, calibrated prediction intervals, not smooth trend lines.

The core goal is **Outbreak Detection**, not trend-following.

---

## 3. The Pivot (Critical Design Decision)

### Before the Pivot
- **Target (y):** `log1p_concentration` (WW signal)
- **Problem:** Predicting WW from WW is circular; model cannot alert on clinical outcomes

### After the Pivot (current architecture)
- **Target (y):** `log1p_new_cases` (log1p-transformed weekly new COVID-19 cases, RobustScaled)
- **Primary hist_exog:** `log1p_concentration` (WW signal is the leading indicator input)
- **Rationale:** The model now forecasts what clinicians care about, using WW as early warning

---

## 4. Data Pipeline Architecture

```
Raw CA WW CSV  +  Raw CA Cases CSV  (covers all CA counties — no new files for new geographies)
        │
        ▼  COVID_Adapter (src/data_pipeline/adapters.py)
        │     Filters to active geography counties  (from config/geographies/<name>.yaml)
        │  or CAWastewaterProcessor (src/data_pipeline/processor.py)
        │
WastewaterProcessor — 15 stages:
  1–8  : QC, unit filter, county filter, population-weighted aggregation
  9    : Centered 7-day rolling mean + relative_decay_rate
  10   : Resample daily → weekly (W-WED)
  11   : WW log-transform → log1p_concentration
  12   : Cases merge → log1p_new_cases (anti-leakage inner join)
  13   : Calendar features (cyclical sin/cos Fourier, DOY)
  14   : Lag features → lags + derivative expansion + ww_momentum_lead
  15   : RobustScaler (per-county, fit on train ONLY; SCALER_IQR_FLOOR=0.3 for sparse counties)
        │
        ▼
  Leakage-free panel DataFrame (county_fips × W-WED date)
```

**Anti-leakage rule:** RobustScaler always `fit` on training rows only.
Target lags (`log1p_new_cases_lag1w/2w/3w`) are strictly past observations.

**IQR floor:** `SCALER_IQR_FLOOR = 0.3` clamps each county's per-feature IQR to a minimum
before normalisation. Prevents Napa/Solano (2–3 training weeks) from inflating scaled values
10–20× vs active counties.

---

## 5. Model Architecture

### Temporal Fusion Transformer (TFT)
- **Framework:** NeuralForecast `TFT` (PyTorch Lightning)
- **Wrapper:** `WastewaterTFT` in `src/models/tft_model.py`
- **Training paradigm:** Global model — all county series in one batch with shared weights

### Covariate Roles

| Role | Columns | Count |
|---|---|---|
| **Target (y)** | `log1p_new_cases` | 1 |
| **Historical exog** | See table below | 18 |
| **Future-known** | Calendar Fourier + DOY | 11 |
| **Static** | `log_population`, `county_fips_encoded`, `is_sludge` | 3 |

### Historical Exogenous Features (HIST_COVARIATES — 18 total)

| Group | Feature | Phase | Description |
|---|---|---|---|
| WW signal | `log1p_concentration` | 1 | WW at t |
| WW lags | `log1p_concentration_lag1w/2w/3w` | 1 | WW at t-1, t-2, t-3 |
| Case momentum | `log1p_new_cases_lag1w/2w/3w` | 2 | Cases at t-1, t-2, t-3 |
| Slope/phase | `growth_rate_1w` | 1 | Relative WW week-over-week change |
| Slope/phase | `relative_decay_rate` | 1 | 7-day relative change on smoothed signal |
| Slope/phase | `outlier_flag_int` | 1 | Z-score spike flag (QC) |
| Velocity | `vel_concentration` | 2 | Absolute weekly Δ log1p_conc |
| Acceleration | `accel_concentration` | 3 | 2nd derivative: Δ velocity |
| Momentum context | `vel_concentration_lag1w` | 3 | Velocity 1 week ago |
| Rolling baseline | `log1p_concentration_2w_ma` | 2 | 2-week rolling mean |
| Rolling baseline | `log1p_concentration_4w_ma` | 2 | 4-week rolling mean |
| Local volatility | `log1p_concentration_2w_std` | 2 | 2-week local volatility |
| Local volatility | `log1p_concentration_4w_std` | 2 | 4-week medium volatility (used by classifier vol adjustment) |
| Biological gravity | `ww_case_ratio` | 5 | log1p_concentration − log1p_new_cases |

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| H (horizon) | 8 weeks | Clinically meaningful alert window |
| INPUT_SIZE | 26 weeks | One epidemiological half-year (≥ 3×H) |
| hidden_size | 128 | d_model for embedding + LSTM |
| n_head | 4 | Multi-head self-attention |
| dropout / attn_dropout | 0.3 / 0.3 | Phase 5 RC-3: overfitting reduction |
| quantile_levels | [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] | 7-quantile grid |
| scaler_type | "identity" | Processor RobustScaler sufficient |
| horizon_weight | [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8] | Upweight near-term steps |

### Loss Function: PINNWastewaterLoss (`src/models/loss_functions.py`)

- **domain_map:** Median-anchored cumulative softplus. Monotonicity guaranteed.
- **Underdispersion penalty:** `effective_lambda = clamp(K × pinball_loss, min=0.5)`.
- **Minimum PI width:** `max(case_vol_4w, forecast_vol_H) × 3.0` clamped to 1.5.
- **Growth-rate penalty:** Asymmetric sigmoid gate — upward-only, decays at outbreak scale.

---

## 6. Phase-Aware Training — Regime Stratified Sampling (`src/data_pipeline/sampler.py`)

### The Dumbbell Problem
When training on the full Bay Area timeline, ~80% of weeks are inter-wave baseline.
The TFT minimises loss across this distribution, producing a forecast that collapses
to the baseline prior even during genuine surges. The pinball diagnostic from run_006
showed q0.10 = 1.136, confirming the model piles mass ~1.3 log1p units above
quiet-period actuals.

### Solution: PhaseLabeler + StratifiedWindowSampler

**`PhaseLabeler`** assigns each (county, date) row one of four phases using
WW signal level + velocity, fitted on training data per county:

| Phase | Rule | Epidemiological meaning |
|---|---|---|
| baseline | signal < p40 AND \|vel\| ≤ vel_threshold | Inter-wave trough |
| onset | vel > vel_threshold AND signal > p20 | Rising edge — WW accelerating |
| peak | signal > p70 AND \|vel\| ≤ vel_threshold | Wave apex — WW elevated but flat |
| decay | vel < −vel_threshold AND signal > p20 | Falling edge after peak |

All thresholds are per-county, from training data only (no leakage).

**`StratifiedWindowSampler`** oversamples minority phases by duplicating each
phase window as a synthetic sub-series:
- **onset:** 3× copies
- **peak:** 2× copies
- **decay:** 3× copies

Synthetic unique_ids: `{source_fips}_{phase_code}_{idx:03d}` (e.g. `06075_ons_001`).
`_get_fips_int()` in tft_model recovers the source FIPS via `uid.split("_")[0]`.

The original full series is always included — baseline coverage is preserved.
NeuralForecast trains on all series with shared weights.

---

## 7. Two-Stage Inference System (`src/models/`, `src/pipeline.py`)

### Architecture Overview

```
processed panel
    │
    ▼  OutbreakClassifier.classify_df()  (Stage 1 — lightweight gatekeeper)
classification: triggered / suppressed per county-week
    │
    ├── triggered ──→ WastewaterTFT (full inference)
    └── suppressed ──→ data-driven quiet prior  (see _quiet_prior() below)
    │
    ▼
InferenceResult (forecast_df + clf_df + metadata)
```

### OutbreakClassifier — Non-Elastic Baseline with Volatility Adjustment

**Non-elastic baseline:** Training-anchored quiet-period mean/std per county.
Frozen after `fit(train_df)`. Never ingests surge data, preventing Z-score erosion.

**Volatility-adjusted threshold:**
```
effective_z = z_threshold × (1 + volatility_scale × max(local_vol/mean_vol − 1, 0))
```
When the 4-week WW std is 2× the training mean (noisy winter signal), the effective
threshold rises 1.5 → 2.25, suppressing false positives in high-noise windows.

**Cold-start handling:** Counties with < `min_observations = 4` training rows cannot
produce a reliable baseline. They are suppressed (all-False) — OutbreakForecaster
returns the data-driven quiet prior. Napa and Solano are handled here automatically.

**Gate logic:**
```
triggered = (z_score >= effective_z) AND (ww_momentum_lead >= 0.0)
```

Both signals must fire:
- Z-score: absolute elevation above training quiet period
- Momentum: WW is accelerating faster than cases are changing (surge leading edge)

### OutbreakForecaster

Runs full TFT only on triggered counties.  For suppressed counties, returns a
**data-driven quiet prior** computed from the last 8 weeks of observed `TARGET_COL`
per county (not a hardcoded constant).

**Why not 0.0?**  The previous design used `SUPPRESSED_FORECAST_LEVEL=0.0` (zero cases).
That gave Coverage95=0% for endemic holdout data (180+ cases/week).  The correct
quiet prior is the **current endemic baseline**:
```
center   = mean(TARGET_COL, last 8W, per county)   ≈ log1p(5.2) for Bay Area summer 2023
half_95  = 1.96 × std → PI ≈ [3.26, 6.91] log1p
half_50  = 0.674 × std
```
Empirical result on 2023 holdout: Coverage95 ≈ 95.1%, Coverage50 ≈ 41.6%.

**Conceptual split:**
- OutbreakClassifier → solves **detection** (prevents false alerts; improves F1/TTD)
- Data-driven quiet prior → solves **calibration** (Coverage50/95 during suppressed periods)
These are independent problems requiring independent solutions.

`SUPPRESSED_FORECAST_LEVEL` in `config.py` is DEPRECATED — forecaster no longer reads it.

---

## 8. Geography Configuration System (`src/config_geographies.py`)

### Design Principle
Every region-specific constant lives in a YAML file. Adding a new region requires
zero code changes — only a new YAML in `config/geographies/`.

### GeographyConfig Fields
- `county_fips`: county_name → FIPS (must match CA WW CSV "County" column)
- `populations`: FIPS → 2020 Census population (fallback when population_served absent)
- `validation_counties`: 3-county validation set (used by `--counties validation`)
- `exclude_fips`: FIPS to skip in training (insufficient WW history)
- `data_start_date`, `data_end_date`, `train_end_date`, `val_end_date`
- `map_center_lat/lon`, `map_zoom`, `centroids`: dashboard map configuration
- `outbreak_validation_windows`: named historical surge episodes

### Active Geographies
| Shortname | Counties | Notes |
|---|---|---|
| `bay_area` | 9-county SF Bay Area | Default; backward-compatible |
| `upper_socal` | Kern, LA, SLO, Santa Barbara, Ventura | ⚠ verify `train_end_date` before running |

### How apply_geography() Works
`apply_geography(geo)` overwrites module-level variables in `src.config` in-place:
`BAY_AREA_FIPS`, `FIPS_TO_COUNTY`, `BAY_AREA_POPULATION`, `BAY_AREA_COUNTIES`,
`THREE_COUNTY_FIPS`, `EXCLUDE_FIPS`, `DATA_START_DATE/END_DATE`, `TRAIN_END_DATE`,
`VAL_END_DATE`, `OUTBREAK_VALIDATION_WINDOWS`, `ACTIVE_GEOGRAPHY`.

`_get_fips_int()` in `tft_model.py` reads the current geography's `fips_int_map`
at fit time — never stale. Dashboard `_get_centroids()` and `_get_map_center()` also
read the active geography dynamically.

---

## 9. Dataset Adapter Framework (`src/data_pipeline/adapters.py`)

### BaseDatasetAdapter Interface
Abstract base: 5 schema properties + 5 processing methods. All geography and
pathogen metadata lives in the adapter — zero leakage into model files.

**Required properties:** `signal_col`, `target_col`, `id_col`, `date_col`, `momentum_col`

**Required methods:** `load_signal()`, `load_target()`, `clean()`, `build_features()`, `transform_target()`

### Forking a New Pathogen
Create a subclass, implement 5 methods. TFT, loss functions, evaluator, and pipeline unchanged.
Example skeleton in `adapters.py` docstring.

---

## 10. Evaluation Engine (`src/evaluation/`)

### metrics.py — 7 pure metric functions

| Function | Category |
|---|---|
| `wis()` | Probabilistic — Weighted Interval Score (Bracher 2021) |
| `pinball_loss()` | Probabilistic — per-quantile dict (all 7 levels) |
| `coverage()` | Probabilistic — empirical 50% and 95% PI rates |
| `mae()` | Probabilistic — Mean Absolute Error on median |
| `match_alerts_to_onsets()` | Detection — greedy TP/FP/FN; TTD in days |
| `detection_score()` | Detection — Precision, Recall, F1 from counts |

**TTD convention:** `onset_date − alert_date` days. Positive = WW alerted BEFORE onset.

**`EvalReport.to_dict()`** serialises all metrics including per-quantile pinball
as flat columns (`pinball_q025` … `pinball_q0975`) for CSV/parquet storage.

### evaluator.py

**`OnsetLabeler`:** p75 percentile of training signal per county. No rolling window.
Onset back-dated to first above-threshold week for unbiased TTD.

**`Evaluator.score(actual_df, forecast_df, alert_df=None)`:** probabilistic metrics
always computed; detection metrics populated when `alert_df` provided.

---

## 11. Temporal Map (Bay Area Default)

### Data spine
- **Frequency:** W-WED (Wednesday-anchored weekly)
- **Full window:** 2020-07-01 → 2023-12-19 (CA dataset cutoff)

### Cross-Validation
| Stage | Window | Notes |
|---|---|---|
| First CV cutoff | 2022-10-05 | After all 9 counties active |
| CV end | 2023-06-07 | ~35 W-WED periods |
| **Holdout** | **2023-06-08 → 2023-12-19** | **28 W; post-XBB quiet period** |

### Why the Holdout is Hard
The holdout is a quiet period (actuals log1p ≈ 5.2–6.1) vs big training waves
(Omicron BA.1/BA.2 log1p ≈ 8–9). The TFT applies outbreak priors to a quiet period
→ systematic upward bias (~1.3 log1p units). Phase-aware training + two-stage gating
address this at training time and inference time respectively.

---

## 12. Source Map

```
src/
  config.py                        — ALL hyperparameters, paths, FIPS, thresholds, ACTIVE_GEOGRAPHY
  config_geographies.py            — GeographyConfig, load_geography(), apply_geography()
  data_pipeline/
    processor.py                   — 15-stage pipeline; WastewaterProcessor + CAWastewaterProcessor
    adapters.py                    — BaseDatasetAdapter (abstract) + COVID_Adapter (concrete)
    sampler.py                     — PhaseLabeler (Baseline/Onset/Peak/Decay) + StratifiedWindowSampler
  models/
    tft_model.py                   — WastewaterTFT wrapper; HIST/FUTR/STATIC lists; _get_fips_int()
    loss_functions.py              — PINNWastewaterLoss
    classifier.py                  — OutbreakClassifier; non-elastic baseline; volatility-adjusted Z
    forecaster.py                  — OutbreakForecaster; conditional TFT + quiet prior
  evaluation/
    metrics.py                     — 7 pure metric functions + dataclasses
    evaluator.py                   — OnsetLabeler, Evaluator, expanding_window_cv
  visualization/
    dashboard.py                   — Dash app; Classifier Timeline; rolling holdout view; run selector
    attention_plots.py             — All Plotly figure builders
  utils/
    helpers.py                     — print_classification_summary; Rich reporting; httpx LM Studio call
    run_manager.py                 — Run versioning; snapshot_run (includes classification.parquet)
pipeline.py                        — TwoStagePipeline + InferenceResult
main.py                            — CLI: --geography, --phase-aware-train, --two-stage, --rolling-holdout
serve_dashboard.py                 — Stand-alone dashboard server; --run flag for archived runs
config/
  geographies/
    bay_area.yaml                  — Bay Area 9-county geography config
    upper_socal.yaml               — Kern/LA/SLO/Santa Barbara/Ventura; sticking-point notes
tests/
  conftest.py                      — Shared fixtures
  test_processor.py                — 15 pipeline stages + scaler leakage
  test_metrics.py                  — 7 metrics
  test_evaluator.py                — OnsetLabeler + Evaluator
  test_classifier.py               — Two-stage + adapter interface
  test_sampler.py                  — PhaseLabeler + StratifiedWindowSampler
  test_loss_functions.py           — domain_map monotonicity + PINN penalty
  test_main.py                     — main.py utility functions
  test_tft_model.py                — WastewaterTFT structure
documents/
  production_blueprint.md          — Deployment, drift detection, retraining, target-variable pivot
```

---

## 13. Sludge vs. Liquid

- **Sludge track** (`copies/g dry sludge`): primary, all 9 counties
- **Liquid track** (`copies/l wastewater`): Section 4.1 only
- `is_sludge` static covariate (1/0) distinguishes tracks in the global TFT.

---

## 14. Notebook Plotting Style Guide

**Library:** matplotlib + seaborn only (no Plotly in notebooks)

**Theme:** `sns.set_theme(style="whitegrid", font_scale=1.1)` + `figure.dpi=120`

**Palette:** `C_WW="steelblue"`, `C_CASES="crimson"`, `C_ACCENT="darkorange"`

**Title:** `fig.suptitle("Title\nSubtitle: units/details", fontsize=13, fontweight="bold")`

**Close every figure:** `plt.tight_layout(); plt.show()`

---

## 15. Phase History

| Phase | Name | Key Outcome |
|---|---|---|
| 1 | Initial Baseline | Single-county TFT; target = log1p_concentration |
| 2 | Momentum Pivot | Target → log1p_new_cases; 7 quantiles; 15 HIST_COVARIATES |
| 3 | PoC Breakthrough | Underdispersion penalty; accel + vel_lag1w; 3-county validation; data → 2023-12-19 |
| 4 | Heuristics to Dynamics | Median-anchored softplus; dynamic underdispersion/min-width; Z-score onset |
| 5 | Reliability & Detection | RC-1/RC-2/RC-3 fixes; ww_case_ratio; GROWTH_RATE_LAMBDA gate; 28W holdout |
| **6** | **Production Readiness** | **7-metric eval engine; OutbreakClassifier (non-elastic + volatility Z); OutbreakForecaster (data-driven quiet prior — NOT 0.0); BaseDatasetAdapter; TwoStagePipeline; PhaseLabeler + StratifiedWindowSampler; Geography config system; dashboard three-tier rebuild + covariate timeline + CV stability chart; tft_model predict() StratifiedWindowSampler fix; 217 tests** |
