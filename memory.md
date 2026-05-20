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
| **Data source** | CDC NWSS (wastewater) × CA county cases dataset (Wednesday-anchored) |
| **Primary unit** | copies/g dry sludge (sludge track, all 9 counties) |
| **Secondary unit** | copies/l wastewater (liquid track, Section 4.1 comparison only) |
| **Validation set** | 3-county (San Francisco 06075, San Mateo 06081, Santa Clara 06085) |

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
Raw NWSS CSV  +  Raw CA Cases CSV
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
| **Historical exog** | See table below | 17 |
| **Future-known** | Calendar Fourier + DOY | 11 |
| **Static** | `log_population`, `county_fips_encoded`, `is_sludge` | 3 |

### Historical Exogenous Features (HIST_COVARIATES — 17 total)

| Group | Feature | Phase Added | Description |
|---|---|---|---|
| WW signal | `log1p_concentration` | 1 | WW at t |
| WW lags | `log1p_concentration_lag1w/2w/3w` | 1 | WW at t-1, t-2, t-3 |
| Case momentum | `log1p_new_cases_lag1w/2w/3w` | 2 | Cases at t-1, t-2, t-3 (VSN interpretability) |
| Slope/phase | `growth_rate_1w` | 1 | Relative WW week-over-week change |
| Slope/phase | `relative_decay_rate` | 1 | 7-day relative change on smoothed signal |
| Slope/phase | `outlier_flag_int` | 1 | Z-score spike flag (QC) |
| Velocity | `vel_concentration` | 2 | Absolute weekly Δ log1p_conc |
| Acceleration | `accel_concentration` | 3 | 2nd derivative: Δ velocity (inflection detector) |
| Momentum context | `vel_concentration_lag1w` | 3 | Velocity 1 week ago (direction context) |
| Rolling baseline | `log1p_concentration_2w_ma` | 2 | 2-week rolling mean (short baseline) |
| Rolling baseline | `log1p_concentration_4w_ma` | 2 | 4-week rolling mean (medium baseline) |
| Local volatility | `log1p_concentration_2w_std` | 2 | 2-week local volatility |
| Local volatility | `log1p_concentration_4w_std` | 2 | 4-week medium volatility |

### Key Hyperparameters (current Phase 4 entry state)

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
- **PINN growth-rate penalty:** `GROWTH_RATE_LAMBDA = 0.0` (disabled since Phase 2)
- **Underdispersion penalty:** Penalises 95% PI narrower than `MIN_PI_WIDTH` scaled units.
  `UNDERDISPERSION_LAMBDA = 0.5` (Phase 3 static value; Phase 4 target: make dynamic)
- **Minimum PI width:** `MIN_PI_WIDTH = 2.5` scaled units
  (Phase 3 static value; Phase 4 target: replace with `multiplier × σ_t`)
- **domain_map:** Identity reshape only — Softplus **removed** (Phase 2) because
  RobustScaled target is legitimately negative; Softplus collapsed all quantiles to zero

### Outbreak Detection: `OutbreakDetector` (`src/evaluation/metrics.py`)

- **Current trigger:** `OUTBREAK_GROWTH_THRESHOLD = 0.25` — static 25% WoW growth
  (Phase 3: lowered from 0.40 to improve AUC sensitivity)
- **Phase 4 target:** Rolling z-score: `z = (x_t - μ_8w) / σ_8w > Z_THRESHOLD`

---

## 6. Temporal Map

### Data spine
- **Frequency:** W-WED (Wednesday-anchored weekly, aligned to CA cases dataset)
- **Full window:** 2020-07-01 → 2023-12-19 (CA dataset cutoff)
- **Effective start:** 2020-07-16 (earliest Santa Clara solid-track data)

### Cross-Validation — Expanding-Window (~8 folds, step = 4 weeks)

| Stage | Window | Notes |
|---|---|---|
| First CV cutoff | 2022-10-05 | After all 9 counties active (last: Napa 2022-09-26) |
| CV end / last cutoff | 2023-06-07 | ~35 W-WED periods beyond first cutoff |
| **Holdout** | **2023-06-08 → 2023-12-19** | **28 W; post-XBB.1.5 + summer/fall 2023** |

### Why These Dates
- Pre-2022-10-05 counties are zero-padded for early folds (`start_padding_enabled=True`)
- 117 W-WED weeks of training before first CV fold >> INPUT_SIZE + H = 34 minimum
- Holdout spans 3.5× H, giving reliable evaluation of the full 8-week forecast window

---

## 7. Source Map

```
src/
  config.py                        — all hyperparameters, paths, FIPS codes
  data_pipeline/
    processor.py                   — 15-stage pipeline; WastewaterProcessor class
  models/
    tft_model.py                   — WastewaterTFT wrapper; HIST/FUTR/STATIC lists
    loss_functions.py              — PINNWastewaterLoss (MQLoss + underdispersion + growth penalty)
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
- **Sludge track** (`copies/g dry sludge`): primary track, all 9 Bay Area counties
- **Liquid track** (`copies/l wastewater`): secondary track, 6–7 counties,
  used only for Section 4.1 comparison and dashboard two-track chart

The `is_sludge` static covariate (1/0) allows the global TFT to learn
track-specific signal shapes within shared attention weights.

---

---

## 9. Notebook Plotting Style Guide

**Library:** matplotlib + seaborn (NOT Plotly — static images preferred over interactive)

**Theme setup (every notebook):**
```python
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})
```

**Color palette — Bay Area (blue / cherry-red / orange):**
- `C_WW = "steelblue"` — wastewater signal
- `C_CASES = "crimson"` — clinical cases
- `C_ACCENT = "darkorange"` — forecast / accent

**Title format:**
```python
fig.suptitle("Specific Descriptive Title — What Is Featured\nSubtitle: key details (county filter, units, transform)", fontsize=13, fontweight="bold")
```
Examples: `"Merged WW Concentrations & Weekly Cases — 4 Illustrative Counties\nWW (left axis, steelblue log scale) leads Cases (right axis, crimson) by ~1–3 weeks"`

**Axes:** always labeled with units — `ax.set_xlabel("Date (W-WED)", fontsize=10)` / `ax.set_ylabel("Copies/g (log)", fontsize=10)`

**Legend:** `ax.legend(fontsize=9, loc="upper right")` — inside plot, top-right corner

**Wave context bands:** `ax.axvspan(pd.Timestamp(ws), pd.Timestamp(we), alpha=0.07, color=wc, zorder=0)` using `ALL_WAVE_SPANS`

**Dual-axis plots:** `ax.twinx()` — WW (`C_WW`) on left axis, cases (`C_CASES`) on right axis

**Close every figure:** `plt.tight_layout()` then `plt.show()`

---

## 10. Phase History

| Phase | Name | Key Outcome |
|---|---|---|
| 1 | Initial Baseline | Single-county TFT on Santa Clara; target = log1p_concentration |
| 2 | Momentum Pivot | Target → log1p_new_cases; 7 quantiles; horizon_weight; 15 HIST_COVARIATES; identity scaler |
| 3 | PoC Breakthrough | Underdispersion penalty; MIN_PI_WIDTH; accel_concentration; vel_concentration_lag1w; 3-county validation; data extended to 2023-12-19; 17 HIST_COVARIATES |
| **4** | **Heuristics to Dynamics** | **Replace static λ, MIN_PI_WIDTH, OUTBREAK_GROWTH_THRESHOLD with dynamic computations** |
