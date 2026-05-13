# Sewer Surveillance

**Attention-based outbreak forecasting with COVID-19 wastewater data**

---

## Abstract

Wastewater-based epidemiology (WBE) detects SARS-CoV-2 RNA shed by infected individuals days before they seek clinical care, creating a leading public-health signal that is immune to testing-access inequities and asymptomatic underreporting. This project operationalises that signal into actionable 8-week case-count forecasts for the 9-county San Francisco Bay Area using a Temporal Fusion Transformer (TFT) trained jointly across all counties on 3.5 years of matched wastewater concentration and COVID-19 case data (July 2020 – December 2023). The model outputs calibrated 7-quantile prediction intervals, flags outbreak onset via a growth-rate detector, and serves results through an interactive Plotly/Dash dashboard with attention-weight interpretability. An expanding-window backtesting protocol spanning 4 outbreak waves provides out-of-sample validation of interval coverage, Weighted Interval Score (WIS), and outbreak-detection AUC.

---

## The Scientific Hypothesis

### Lead-Time Hypothesis

> *Sewage RNA concentration is a statistically reliable leading indicator of clinical COVID-19 case counts, with a measurable advance signal of 7–21 days.*

When an individual becomes infected, SARS-CoV-2 is shed in faeces within 24–48 hours of exposure — typically 4–7 days before symptom onset and 7–14 days before a positive clinical test is reported. Aggregated across a sewershed serving tens of thousands of residents, the wastewater signal smooths individual variation and surfaces community-level transmission dynamics that would otherwise be invisible until cases accumulate. This project quantifies the lead-time empirically: the model's variable-selection networks (VSN) learn to weight 1-week and 2-week wastewater lags most heavily, consistent with a 7–14-day advance window.

### Lag-Time Hypothesis

> *The temporal structure between wastewater peak and clinical case peak is stable enough across distinct outbreak waves to be learned by a sequence model and exploited for multi-week probabilistic forecasting.*

Not all WW→cases lead-times are equal. The lag narrows during rapid exponential growth (Omicron BA.1) because high community incidence saturates testing systems and compresses the reporting delay. It widens during late-wave decline because residual shedding from recovered individuals artificially extends the WW signal past the true clinical peak. A model that learns this nonlinear, wave-dependent lag structure can produce forecasts whose uncertainty bands widen appropriately during volatile onset phases — which is the defining challenge this architecture is designed to solve.

---

## Methodology

### Data

| Source | Coverage | Granularity |
|---|---|---|
| CA Wastewater Surveillance (`California_Wastewater_Surveillance_Data.csv`) | 9 Bay Area counties, Jul 2020 – Dec 2023 | 1–7 samples/week per sewershed |
| CA Statewide COVID-19 Cases/Deaths/Tests (`Statewide_COVID-19_Cases_Deaths_Tests.csv`) | All 58 CA counties | Daily reported cases |

Both datasets are resampled to a **Wednesday-anchored weekly spine** (`W-WED`) using median aggregation for wastewater and sum aggregation for cases. The joined time series spans **≈180 weekly observations** per county and covers four distinct epidemiological waves: Summer 2020 (ancestral strain), Winter 2020–21 (Alpha/Beta), Delta 2021, and the twin Omicron waves of Winter 2021–22 and Summer 2022.

### The Spine

The pipeline's central data structure is a **complete W-WED calendar spine** — a dense sequence of every Wednesday in `[DATA_START_DATE, DATA_END_DATE]` — reindexed independently for each county before any feature engineering. This design choice is non-obvious but critical: raw wastewater data contains irregular sampling gaps (lab closures, QC failures, sewershed outages). If lag features are computed on the raw ragged frame using row-index arithmetic (`shift(n)`), a 2-week gap causes a 1-week-ago lag to silently read a value that is actually 3 weeks old. Reindexing to the complete spine converts gaps into explicit `NaN` rows, making the resulting lag correctly `NaN` rather than stale.

### Feature Engineering

**Per-county normalisation** is applied via independent `RobustScaler` instances — one fitted on each county's training rows before pooling into the global model. This prevents large-population counties (Santa Clara: ~1.9M) from dominating the IQR used to scale small counties (Napa: ~140K), which would otherwise cause Napa's concentration signal to be compressed to near-zero relative magnitude.

**15 historical covariates** are computed for each county-week:

| Feature | Type | Captures |
|---|---|---|
| `log1p_concentration` | WW level | Baseline shedding signal |
| `log1p_concentration_lag1w/2w/3w` | WW lags | WW→cases lead-time structure |
| `growth_rate_1w` | WW velocity | Relative rate of change in concentration space: `(c_t - c_{t-1}) / (\|c_{t-1}\| + ε)` |
| `diff_concentration` | WW acceleration | Absolute Δlog1p: distinguishes from relative growth |
| `log1p_concentration_2w_ma` | Short baseline | 2-week rolling mean for deviation detection |
| `log1p_concentration_4w_ma` | Medium baseline | 4-week rolling mean |
| `log1p_concentration_2w_std` | Local volatility | Spikes during erratic onset phase |
| `log1p_concentration_4w_std` | Medium volatility | Multi-week dispersion |
| `log1p_new_cases_lag1w/2w/3w` | Case lags | Auto-regressive context for the target |
| `outlier_flag` | Data quality | Binary; marks QC-flagged samples |

**6 future-known covariates** (calendar): `day_of_week_sin/cos`, `month_sin/cos`, `is_holiday`, `days_since_last_holiday`.

**3 static covariates** (county-level): `log_population`, `sewershed_count`, `urban_density_index`.

### Model: Temporal Fusion Transformer

The TFT (Lim et al., 2021) is a multi-horizon attention architecture that processes time-varying past inputs through an LSTM encoder, conditions on future-known covariates via a separate encoder, and produces interpretable attention weights over both axes. Its Variable Selection Networks (VSN) learn soft feature importance masks for each time step, enabling post-hoc inspection of which covariates the model relies on for a given forecast.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Forecast horizon H | 8 weeks | Clinically actionable alert window |
| Lookback INPUT\_SIZE | 26 weeks | One epidemiological half-year (≥ 3×H) |
| Hidden size | 128 | d\_model for embedding, LSTM, and attention |
| Attention heads | 4 | Multi-scale temporal patterns |
| Encoder / decoder layers | 2 each | Depth sufficient for weekly seasonality |
| Quantile levels | 7: [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] | Full PI + inner 50% band |
| Horizon weights | [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8] | Up-weights near-term steps (weeks 1–4) |
| Total parameters | ≈4.4M | — |
| Scaler type | identity | Per-county RobustScaler applied upstream |

### Loss Function: PINNWastewaterLoss

The training objective combines three terms:

```
L = L_pinball + λ_growth · L_growth + λ_ud · L_underdispersion
```

**Pinball loss** (`L_pinball`): Standard quantile regression loss across all 7 levels, summed over the 8-step horizon with horizon weights applied.

**Growth-rate penalty** (`L_growth`, λ=0.0): A physics-informed penalty that penalises forecasted doubling times faster than the biological limit (~2 days). Disabled in the current phase: the model must first learn to produce wide enough intervals before the biological prior is re-introduced.

**Underdispersion penalty** (`L_underdispersion`, λ=0.1): Penalises the model when its predicted 95% PI width falls below 1.5 scaled units — a conservative floor that prevents complete interval collapse without dominating the pinball objective:

```
shortfall = clamp(1.5 − (q₀.₉₇₅ − q₀.₀₂₅), min=0)
L_underdispersion = 0.1 · mean(shortfall²)
```

After per-county RobustScaling, IQR ≈ 1.0 per feature, so a normal-equivalent 95% PI spans ≈ 2.9 scaled units. The 1.5 floor is ≈ 52% of that — conservative enough to allow genuine narrow intervals during flat baselines while preventing the model from predicting a single point estimate dressed as an interval.

### Outbreak Detection

Post-forecast, an `OutbreakDetector` applies a heuristic trigger on the median forecast trajectory:

- **Threshold**: week-over-week growth ≥ **40%** in predicted cases (raised from 25% to reduce false-positive alerts during endemic plateau periods)
- **Baseline**: 90th percentile of the trailing 90-day rolling distribution
- **Sustained signal**: alert fires when the signal remains elevated for ≥ 3 consecutive days after the weekly threshold is crossed
- **Lead-time window**: 7–21 days before clinical confirmation is the target detection range

---

## Results & Evaluation

### Expanding-Window Cross-Validation

Standard train/test splits are inappropriate for epidemiological time series because outbreak waves are non-stationary: a model trained only on Delta cannot be expected to generalise to Omicron without any exposure to the structural shift. The project uses **expanding-window cross-validation** — a backtesting protocol where the training set grows by 4 weeks at each fold while the test set is always the subsequent 8 weeks (H):

```
Fold 1:  train → [2020-07-01, 2022-10-05]   test → [2022-10-05, 2022-12-07]
Fold 2:  train → [2020-07-01, 2022-11-02]   test → [2022-11-02, 2023-01-04]
  ⋮
Fold 9:  train → [2020-07-01, 2023-06-07]   test → [2023-06-07, 2023-08-02]
```

Each fold is an independent model fit from scratch (`val_size=0`, early stopping disabled). This design spans the BQ.1 onset (late 2022) and the XBB.1.5 emergence (early 2023) — the two most structurally distinct periods in the holdout window — and provides ≈9 independent 8-week evaluation windows for metric aggregation.

### Metrics

**Weighted Interval Score (WIS)** is the primary proper scoring rule for probabilistic forecasts. It rewards both accuracy (sharpness) and calibration (coverage), and is equivalent to the mean absolute error when the forecast is a point estimate. Lower is better.

```
WIS = w₀ · |y − q₀.₅| + Σ_{(α_l, α_u)} w_α · [( q_u − q_l) + (2/α) · 1(y ∉ [q_l, q_u]) · dist(y, [q_l, q_u])]
```

Interval pairs used: `(0.025, 0.975)` and `(0.25, 0.75)`, each with weight 0.5.

**PI Coverage** measures the empirical fraction of actuals that fall inside the predicted interval:

- `coverage_50`: fraction of actuals inside the 25%–75% PI (target: ≥ 50%)
- `coverage_95`: fraction of actuals inside the 2.5%–97.5% PI (target: ≥ 95%)

Zero coverage — the pre-Phase 2 baseline — indicates an overconfident model whose upper quantile is consistently below actuals. The Phase 2 changes (underdispersion penalty, 7 quantiles, horizon upweighting, scaler fix) are designed to bring `coverage_95` above 50% as an initial milestone.

**Outbreak-Detection AUC** measures the area under the ROC curve for binary outbreak-onset classification (flagged vs. not flagged) over the CV folds. An AUC ≥ 0.70 with a lead time of ≥ 7 days would indicate actionable early-warning value over a baseline that uses only clinical cases.

**sMAPE** (symmetric mean absolute percentage error) is reported as a supplementary point-forecast accuracy metric on the median trajectory.

---

## Repository Layout

```
sewer-surveillance/
├── src/
│   ├── config.py                  # All hyperparameters and constants
│   ├── data_pipeline/
│   │   └── processor.py           # W-WED spine, per-county scaling, lag features
│   ├── models/
│   │   ├── tft_model.py           # WastewaterTFT wrapper around NeuralForecast TFT
│   │   └── loss_functions.py      # PINNWastewaterLoss (pinball + underdispersion)
│   ├── evaluation/
│   │   └── metrics.py             # WIS, coverage, AUC, OutbreakDetector
│   ├── visualization/
│   │   ├── dashboard.py           # Plotly/Dash interactive dashboard
│   │   └── attention_plots.py     # VSN importance + attention heatmaps
│   └── utils/
│       └── helpers.py             # LLM bulletin generation (optional)
├── main.py                        # Full 9-county pipeline entry point
├── run_santa_clara.py             # Single-county validation pipeline
├── data/
│   ├── raw/                       # Source CSVs (gitignored)
│   └── processed/                 # Intermediate Parquet artefacts (gitignored)
├── notebooks/                     # EDA and exploratory work
├── documents/                     # Analysis reports and forensic notes
├── tests/                         # Unit tests (pytest)
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Installation & Usage

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Apple Silicon / CUDA GPU strongly recommended for the 2000-step TFT fit

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/sewer-surveillance.git
cd sewer-surveillance

# Create virtual environment and install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env: set USE_LOCAL_LLM=true and LOCAL_LLM_MODEL if you want AI bulletins
```

### Data

Place the two California state datasets in `data/raw/`:

```
data/raw/California_Wastewater_Surveillance_Data.csv
data/raw/Statewide_COVID-19_Cases_Deaths_Tests.csv
```

Both are publicly available from the California Open Data Portal.

### Running the Pipeline

```bash
# Full run: CV + final model + interactive dashboard
python main.py

# Skip cross-validation (fastest end-to-end check, ~15–20 min on MPS)
python main.py --skip-cv --no-dash

# Fast smoke test (reduced max_steps)
python main.py --fast --no-dash

# Single-county validation (Santa Clara only)
python run_santa_clara.py --skip-cv --no-dash

# Custom training budget
python main.py --max-steps 500 --no-dash
```

The dashboard launches at `http://localhost:8050` and displays:
- Median forecast + 50%/95% prediction intervals
- WW concentration track (dual-axis)
- VSN feature importance heatmap
- Multi-head attention weights over the lookback window
- Outbreak-onset detection flags

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Sewer Surveillance Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
