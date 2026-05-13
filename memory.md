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
| **Geography** | 9-county San Francisco Bay Area (CA FIPS 06001–06097) |
| **Data source** | CDC NWSS (wastewater) × CDC archived county cases (Wednesday-anchored) |
| **Primary unit** | copies/g dry sludge (sludge track, all 9 counties) |
| **Secondary unit** | copies/l wastewater (liquid track, Section 4.1 comparison only) |

---

## 2. Scientific Hypothesis: The Momentum-Lead Relationship

Wastewater SARS-CoV-2 signal leads clinical case counts by approximately **1–3 weeks**.
The model must learn three temporal relationships:

1. **The Lead:** A sustained rise in wastewater concentration precedes a case surge.
   The rate of change (velocity) is a stronger predictor of surge onset than level alone.
2. **The Lag:** Wastewater signal decays *after* the clinical peak — it confirms recovery.
3. **The Volatility:** Clinical case counts are "spiky" (high week-over-week variance).
   The model must produce wide, calibrated prediction intervals, not smooth trend lines.

The core goal is **Outbreak Detection**, not trend-following.

---

## 3. The Pivot (Critical Design Decision)

### Before the Pivot
- **Target (y):** `log1p_concentration` (wastewater signal)
- **Problem:** Predicting WW from WW is circular; model cannot alert on clinical outcomes

### After the Pivot (current architecture)
- **Target (y):** `log1p_new_cases` (log1p-transformed weekly new COVID-19 cases, RobustScaled)
- **Primary hist_exog:** `log1p_concentration` (WW signal is the leading indicator input)
- **Rationale:** The model now forecasts what clinicians care about, using WW as early warning

### Why This Matters for Interpretability
The VSN (Variable Selection Network) now produces attention weights over WW features,
case lags, and momentum features. These weights are directly interpretable as:
*"how much does the model rely on wastewater vs. recent case counts vs. volatility signals?"*

---

## 4. Data Pipeline Architecture

```
Raw NWSS CSV  +  Raw CDC Cases CSV
        │
        ▼
WastewaterProcessor (src/data_pipeline/processor.py)
  Stage 1–8  : QC, unit filter, county filter, population-weighted aggregation
  Stage 9    : Centered 7-day rolling mean + relative_decay_rate (daily grain)
  Stage 10   : Resample daily → weekly (W-WED: week-ending Wednesday)
  Stage 11   : WW log-transform  →  log1p_concentration
  Stage 12   : Cases merge       →  log1p_new_cases  (anti-leakage inner join)
  Stage 13   : Calendar features (cyclical sin/cos Fourier, DOY)
  Stage 14   : Lag features      →  lags 1/2/3w + derivative expansion
  Stage 15   : RobustScaler      →  fit on train ONLY; transform val/test
        │
        ▼
  Leakage-free panel DataFrame (county_fips × W-WED date)
```

**Anti-leakage rule:** The RobustScaler is always `fit` on training rows only.
Val and test splits call `transform()` with the stored scaler. Target lags
(`log1p_new_cases_lag1w/2w/3w`) are strictly past observations — never future.

---

## 5. Model Architecture

### Temporal Fusion Transformer (TFT)
- **Framework:** NeuralForecast `TFT` (PyTorch Lightning)
- **Wrapper:** `WastewaterTFT` in `src/models/tft_model.py`
- **Training paradigm:** Global model — all 9 county series in one batch with shared weights

### Covariate Roles

| Role | Columns | Count |
|---|---|---|
| **Target (y)** | `log1p_new_cases` | 1 |
| **Historical exog** | See table below | 15 |
| **Future-known** | Calendar Fourier + DOY | 11 |
| **Static** | `log_population`, `county_fips_encoded`, `is_sludge` | 3 |

### Historical Exogenous Features (HIST_COVARIATES — 15 total)

| Group | Feature | Description |
|---|---|---|
| WW signal | `log1p_concentration` | WW at t |
| WW lags | `log1p_concentration_lag1w/2w/3w` | WW at t-1, t-2, t-3 |
| Case momentum | `log1p_new_cases_lag1w/2w/3w` | Cases at t-1, t-2, t-3 (VSN interpretability) |
| Slope/phase | `growth_rate_1w` | Relative WW week-over-week change |
| Slope/phase | `relative_decay_rate` | 7-day relative change on smoothed signal |
| Slope/phase | `outlier_flag_int` | Z-score spike flag (QC) |
| Derivative | `diff_concentration` | Absolute weekly velocity (Δ log1p_conc) |
| Derivative | `log1p_concentration_2w_ma` | 2-week rolling mean (short baseline) |
| Derivative | `log1p_concentration_4w_ma` | 4-week rolling mean (medium baseline) |
| Derivative | `log1p_concentration_2w_std` | 2-week local volatility |
| Derivative | `log1p_concentration_4w_std` | 4-week medium volatility |

### Key Hyperparameters (current)

| Parameter | Value | Rationale |
|---|---|---|
| H (horizon) | 8 weeks | Clinically meaningful alert window |
| INPUT_SIZE | 26 weeks | One epidemiological half-year (≥ 3×H) |
| hidden_size | 128 | d_model for embedding + LSTM |
| n_head | 4 | Multi-head self-attention |
| quantile_levels | [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] | 7-quantile grid for finer tail calibration |
| scaler_type | "identity" | Processor RobustScaler is sufficient; double-scaling collapses PI |
| horizon_weight | [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8] | Upweight near-term steps (WW→cases lead window) |

### Loss Function: PINNWastewaterLoss (`src/models/loss_functions.py`)

- **Base:** `MQLoss` (NeuralForecast pinball loss)
- **Addition:** PINN growth-rate penalty (quadratic violation above `MAX_DAILY_GROWTH_RATE × 7`)
- **Current lambda:** `GROWTH_RATE_LAMBDA = 0.0` (disabled for Phase 2 calibration)
- **domain_map:** Identity reshape only — Softplus was **removed** because RobustScaled
  target is legitimately negative; Softplus was collapsing all quantiles to zero

---

## 6. Temporal Map

### Data spine
- **Frequency:** W-WED (Wednesday-anchored weekly, aligned to CDC cases dataset)
- **Overlap window:** 2022-02-07 → 2023-05-10 (intersection of NWSS + cases datasets)
- **First W-WED observation:** 2022-02-09

### Cross-Validation — 5 Expanding-Window Folds (step = 4 weeks)

| Fold | Train window | Eval window | COVID wave in eval |
|---|---|---|---|
| 1 | 2022-02-09 → 2022-10-05 | 2022-10-12 → 2022-11-30 | BQ.1/BQ.1.1 fall surge **onset** |
| 2 | 2022-02-09 → 2022-11-02 | 2022-11-09 → 2022-12-28 | BQ.1 **peak & decline** |
| 3 | 2022-02-09 → 2022-11-30 | 2022-12-07 → 2023-01-25 | BQ.1 tail + XBB emergence |
| 4 | 2022-02-09 → 2022-12-28 | 2023-01-04 → 2023-02-22 | XBB.1.5 **rise** |
| 5 | 2022-02-09 → 2023-01-25 | 2023-02-01 → 2023-03-22 | XBB.1.5 **peak & initial decline** |

### Final Model & Holdout

| Stage | Window | Length | COVID context |
|---|---|---|---|
| Final train | 2022-02-09 → 2023-01-25 | 51 W | BA.2, BA.4/5, BQ.1 all in training |
| **Holdout** | **2023-02-01 → 2023-05-10** | **15 W** | **XBB.1.5 peak → decline → end** |

---

## 7. Source Map

```
src/
  config.py                        — all hyperparameters, paths, FIPS codes
  data_pipeline/
    processor.py                   — 15-stage pipeline; WastewaterProcessor class
  models/
    tft_model.py                   — WastewaterTFT wrapper; HIST/FUTR/STATIC lists
    loss_functions.py              — PINNWastewaterLoss (MQLoss + growth penalty)
  evaluation/
    metrics.py                     — WIS, coverage, SMAPE, OutbreakDetector,
                                     LeadTimeEvaluator, expanding_window_cv, evaluate()
  visualization/
    dashboard.py                   — Dash app builder (create_app, run_demo)
    attention_plots.py             — All Plotly figure functions
  utils/
    helpers.py                     — LLM public health bulletin generator
main.py                            — CLI entry point (--fast, --skip-cv, --no-dash)
```

---

## 8. Sludge vs. Liquid (Section 4.1)

The pipeline supports two signal tracks:
- **Sludge track** (`copies/g dry sludge`): primary track, all 9 Bay Area counties,
  sludge-matrix concentration gives sharper decay-rate resolution
- **Liquid track** (`copies/l wastewater`): secondary track, 6–7 counties,
  used only for the Section 4.1 comparison and dashboard two-track chart

The `is_sludge` static covariate (1/0) allows the global TFT to learn
track-specific signal shapes within shared attention weights.
