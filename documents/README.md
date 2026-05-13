# Sewer Signals
### Early-Warning COVID-19 Outbreak Detection via Wastewater Surveillance

> **Forecasting clinical case surges 1–2 weeks in advance using probabilistic deep learning on community wastewater RNA signals.**

---

## Overview

When SARS-CoV-2 infects someone, viral RNA appears in their stool within days — often before symptoms, and well before a clinical test. That RNA travels through the sewer system and reaches wastewater treatment plants where it can be measured. At the population level, this creates a systematic **1–2 week lead time**: the wastewater signal rises before clinical cases do.

**Sewer Signals** operationalises this lead time. It ingests publicly available wastewater surveillance data from the CDC National Wastewater Surveillance System (NWSS) and produces calibrated, probabilistic forecasts of **weekly new COVID-19 case counts** across the nine-county San Francisco Bay Area. When the model detects an acceleration pattern in wastewater — the signature of an emerging surge — it issues an automated outbreak alert before the clinical surge arrives.

The project covers the BA.2, BQ.1/BQ.1.1, and XBB.1.5 variant epochs (February 2022 – May 2023) and was designed from the outset for **interpretability**: every forecast comes with feature attribution showing exactly which wastewater signals drove the model's prediction.

---

## Key Features

- **Probabilistic 8-Week Forecasts** — Outputs 7 calibrated quantile levels (2.5th through 97.5th percentile) for each county across an 8-week horizon, enabling decision-makers to reason about best-case, expected, and worst-case outcomes simultaneously.

- **Automated Outbreak Alerts** — A rule-based alert layer flags counties where the forecast trajectory meets criteria for imminent surge onset: ≥25% predicted growth sustained over 3 consecutive forecast steps above a minimum signal floor.

- **Interpretable Feature Attribution** — A Variable Selection Network (VSN) assigns importance weights to each input feature at every timestep, rendering a time × feature attention heatmap. Clinicians can see whether the model is relying on wastewater velocity, recent case trends, or local volatility to drive a given forecast.

- **Lead-Time Quantification** — An evaluator measures how many weeks before clinical onset the model first issues an outbreak alert, providing an empirical estimate of the actionable warning window.

- **Dual-Track Wastewater Processing** — Handles both sludge-normalised (copies/g dry sludge) and liquid-normalised (copies/litre) wastewater signals with separate processing pathways, recognising that they have different noise profiles and sensitivity to environmental confounders (e.g., precipitation dilution).

- **Leakage-Free Cross-Validation** — Evaluates model performance using a 5-fold expanding-window protocol that mirrors real deployment: the model is retrained on all data up to each cutoff and evaluated strictly on future observations it has never seen.

- **LLM-Generated Public Health Bulletins** — For each county forecast, an LLM generates a plain-language narrative summary suitable for county epidemiologists who need actionable insight, not probability intervals.

- **Interactive Dashboard** — A Plotly Dash application renders forecast traces, prediction intervals, outbreak alerts, wastewater tracks, and VSN attribution heatmaps in a single browser-based interface.

---

## How It Works

### The Forecasting Engine

At the core of Sewer Signals is a **Temporal Fusion Transformer (TFT)** — a state-of-the-art architecture for probabilistic multi-horizon time series forecasting. Unlike traditional methods (ARIMA, Prophet), the TFT is designed to handle heterogeneous inputs: it simultaneously processes the raw wastewater signal, derived velocity and volatility features, lagged clinical case counts, calendar seasonality, and static county attributes — weighting each dynamically based on what is most predictive at each point in time.

The model is trained as a **global model** across all nine Bay Area counties simultaneously. This is critical: with only ~15 months of usable training history per county, individual county models would overfit. The global model learns generalizable outbreak dynamics from the pooled signal, while static county covariates (population, FIPS code, signal track type) allow it to represent county-level variation.

A custom **Physics-Informed loss function** encodes the biological prior that community-level viral growth cannot be instantaneously discontinuous — a soft constraint that improves forecast smoothness during stable periods without suppressing the rapid growth trajectories that define surge onset.

### Sludge vs. Liquid: Why It Matters

Wastewater facilities report viral RNA concentrations in two ways: normalised by the dry weight of the collected sludge, or normalised by the volume of wastewater flow. These are not interchangeable.

**Sludge-normalised** concentrations are relatively stable across environmental conditions. Sludge is collected independently of flow volume, so a heavy rainstorm that dilutes the liquid fraction does not affect the sludge reading. This makes the sludge signal a cleaner representation of actual community viral load.

**Liquid-normalised** concentrations are sensitive to flow volume. Rain events dilute the wastewater, causing apparent concentration drops that have nothing to do with a real decrease in infections — a potential source of false-positive "recovery" signals.

Sewer Signals uses **sludge track as primary** for all nine counties. The liquid track is retained as a secondary signal and rendered in the dashboard's two-track comparison chart, allowing analysts to see when the two signals diverge (a useful indicator of environmental confounding). The model learns track-specific behaviour through a binary static covariate, handling both tracks within a single architecture.

---

## Installation & Usage

> **Requirements:** Python 3.10+, PyTorch (MPS/CUDA/CPU), NeuralForecast, Plotly Dash

```bash
# Clone the repository
git clone <repo-url>
cd wastewater

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (cross-validation + final model + dashboard)
python main.py

# Skip cross-validation for a faster end-to-end run
python main.py --skip-cv

# Skip the dashboard (headless environments)
python main.py --skip-cv --no-dash

# Smoke-test with reduced training steps
python main.py --fast --no-dash
```

The pipeline will:
1. Download and process wastewater + cases data into a leakage-free feature panel
2. Run 5-fold expanding-window cross-validation and report per-fold WIS and coverage
3. Train the final model on the full training window
4. Generate holdout evaluation metrics and county-level outbreak alerts
5. Launch the interactive dashboard at `http://localhost:8050`

> **Data:** Raw CDC NWSS wastewater CSV and CDC archived county cases CSV should be placed in the `data/` directory. See `src/config.py` for expected file paths.

---

## Scientific Impact

Traditional disease surveillance relies on clinical test positivity rates — a signal that is already 5–10 days behind the moment of infection, and further delayed by test access, reporting lags, and healthcare-seeking behaviour. During the COVID-19 pandemic, this lag consistently meant that surge warnings arrived after hospital systems were already under pressure.

Wastewater epidemiology inverts this dependency. By measuring viral RNA in the community's shared sewer infrastructure, surveillance becomes **population-level, passive, and near-real-time** — independent of whether individuals seek clinical care. In California alone, the NWSS now covers over 50% of the population with active wastewater monitoring sites.

Sewer Signals demonstrates that this passive surveillance signal can be operationalised into a calibrated probabilistic forecast with a measurable early-warning window. The system's design prioritises three properties that clinical translation demands:

- **Calibration over point accuracy** — A forecast whose 95% prediction interval actually covers the true outcome 95% of the time is more useful for planning than a sharp forecast that is wrong when it matters most.
- **Interpretability over capacity** — Feature attribution that shows *why* the model is forecasting a surge allows epidemiologists to cross-check the model's reasoning against domain knowledge and trust the alert.
- **Honest uncertainty at the surge onset** — The hardest problem in outbreak forecasting is producing appropriately wide intervals during rapid emergence events, when uncertainty is highest and the cost of overconfidence is greatest.

The project is an open contribution to the intersection of environmental surveillance, probabilistic deep learning, and public health decision support.

---

## Project Structure

```
wastewater/
├── main.py                     # Entry point — CLI flags: --fast, --skip-cv, --no-dash
├── src/
│   ├── config.py               # Hyperparameters, data paths, FIPS codes
│   ├── data_pipeline/
│   │   └── processor.py        # 15-stage feature engineering pipeline
│   ├── models/
│   │   ├── tft_model.py        # WastewaterTFT wrapper; covariate definitions
│   │   └── loss_functions.py   # PINNWastewaterLoss (MQLoss + growth penalty)
│   ├── evaluation/
│   │   └── metrics.py          # WIS, coverage, SMAPE, OutbreakDetector, CV
│   ├── visualization/
│   │   ├── dashboard.py        # Plotly Dash app
│   │   └── attention_plots.py  # All chart functions
│   └── utils/
│       └── helpers.py          # LLM bulletin generation
├── data/                       # Raw CDC NWSS + cases CSVs (not committed)
├── project_overview.md         # Full lifecycle summary
├── technical_design_document.md # Deep-dive: math, architecture, design rationale
└── post_mortem.md              # Engineering post-mortem: four critical failure analyses
```

---

## Acknowledgements

- **CDC NWSS** — National Wastewater Surveillance System for publicly archived wastewater data
- **NeuralForecast** (Nixtla) — TFT implementation and probabilistic forecasting framework
- **Temporal Fusion Transformer** — Lim et al., 2020 (*International Journal of Forecasting*)

---

*This project is a research prototype. It is not approved for clinical or operational public health use.*
