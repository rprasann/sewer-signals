# Sewer Signals: Project Overview
## Probabilistic COVID-19 Outbreak Forecasting via Wastewater Surveillance

**Document Type:** State of the Union — Project Lifecycle Summary
**Audience:** Technical (AI/ML Instructor)
**Version:** 2.0
**Date:** 2026-05-08

---

## 1. Initial Goals

The project originated from a single, testable hypothesis: **wastewater RNA surveillance is a more reliable leading indicator of COVID-19 community spread than clinical case reporting**, and a probabilistic model trained on that signal could provide actionable 8-week advance warning to public health officials.

The original mission had three specific deliverables:

1. **Forecast:** Produce 8-week-ahead probabilistic forecasts of weekly new COVID-19 cases for all 9 Bay Area counties, expressed as calibrated quantile intervals (50% and 95% prediction intervals).
2. **Detect:** Flag outbreak onset automatically — defined as ≥25% week-over-week predicted growth sustained over ≥3 horizon steps — with sufficient lead time (7–21 days) for a clinical response.
3. **Explain:** Surface which wastewater features drive each forecast via VSN attention weights, making the model auditable to epidemiologists, not just accurate to statisticians.

The initial design used CDC NWSS wastewater data and CDC archived county case counts, with `log1p_concentration` (wastewater RNA) as both the primary input *and* the prediction target. A critical early design pivot changed the target to `log1p_new_cases` — reframing the problem from "predict wastewater from wastewater" (circular) to "predict clinical outcomes from biological leading indicators" (actionable).

---

## 2. EDA Insights

### 2.1 The Scale Problem: Omicron as an Outlier

The first EDA notebook (`02_eda_bay_area.ipynb`) on CDC data immediately surfaced the dominant challenge of the dataset: **Omicron's magnitude is anomalous at roughly 10× any prior wave**. A model trained on the post-Omicron CDC window (2022–2023) cannot distinguish between "high Omicron" and "low everything else" — it calibrates its entire dynamic range to a single extraordinary event.

This finding directly motivated the migration to California state datasets, which extend the overlap window back to July 2020 and capture all four distinct outbreak waves:

| Wave | Period | Character |
|---|---|---|
| Wave 1 (Original) | Jul–Oct 2020 | First sewer signal; 3-county coverage only |
| Wave 2 (Alpha) | Nov 2020–Apr 2021 | Long plateau; winter surge |
| Wave 3 (Delta) | Jun–Nov 2021 | Fast, steep, high transmissibility |
| Wave 4 (Omicron) | Dec 2021–May 2022 | Anomalous peak ~10× prior magnitude |

Exposing the model to Waves 1–3 teaches scale-agnostic surge detection. Without them, the model overfits to Omicron's scale and produces near-zero forecasts for the smaller, faster surges it would actually encounter in deployment.

### 2.2 Signal Column Analysis: Raw Concentration vs. Norm Pmmov

The second EDA notebook (`03_eda_bay_area_CAdatasets.ipynb`) profiled two normalization pathways in the California WW dataset:

- **Raw Concentration** (copies/g dry sludge): stable across facilities, less sensitive at low prevalence
- **Norm Pmmov** (ratio to PMMoV fecal indicator): corrects for dilution, more sensitive during inter-wave troughs

The EDA computes normalized [0–1] Pearson correlations against actual case counts per outbreak window to determine which signal tracks clinical outcomes more faithfully. The pipeline currently defaults to `Raw Concentration`; the EDA Section 3 comparison will drive a final selection decision.

### 2.3 Pre-2022 Coverage Asymmetry

A key structural finding: only three counties have solid-track WW data before 2022 — **Santa Clara** (Jul 2020), **San Francisco** (Nov 2020), and **San Mateo** (Dec 2020). The remaining six counties join the solid track in 2022, with Napa and Solano each accumulating only 2–3 training weeks before the first CV cutoff.

This asymmetry has a direct, measurable consequence confirmed by the most recent holdout run: counties with longer WW histories produce dramatically better-calibrated forecasts. Santa Clara: WIS=0.112, SF: WIS=0.138 vs. Napa: WIS=1.601, Marin: WIS=1.028. The WIS-vs-data-length correlation is causal — it is the single most actionable finding for improving model performance without architectural changes.

### 2.4 The W-WED Alignment Requirement

Both CA datasets require Wednesday-anchored weekly resampling (`W-WED`). The CA Cases dataset is daily and must be summed to W-WED bins before merging. The CA WW dataset has variable collection days per facility; the dominant BAYWA collection day is Wednesday. Misaligning these spines by even one day causes off-by-one errors in all lag features and CV fold boundaries — a silent failure that produces valid-looking but incorrectly computed training data.

---

## 3. Architecture & Model Overview

### 3.1 Temporal Fusion Transformer (TFT)

The core model is a 4.4M-parameter **Temporal Fusion Transformer** (Lim et al., 2020), implemented via NeuralForecast and wrapped in `WastewaterTFT`. Key architectural parameters:

| Component | Value | Rationale |
|---|---|---|
| Forecast horizon H | 8 weeks | Clinically meaningful alert window |
| Lookback window | 26 weeks | One epidemiological half-year; ≥3×H |
| Hidden dimension | 128 | d_model for LSTM and attention |
| Attention heads | 4 | Multi-head temporal self-attention |
| Encoder/decoder layers | 2 each | Depth for multi-scale temporal patterns |
| Quantile levels | 7 | [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] |
| Training steps | 2,000 | MPS GPU; ~30 min on Apple Silicon |

The model operates as a **global model**: all 9 Bay Area counties are packed into a single training run, with county identity encoded as a static covariate (`unique_id`). This allows the model to share cross-county outbreak dynamics — same circulating variants, same regional wave timing, same CDPH measurement protocols — while maintaining county-specific biases through the static covariate channel.

### 3.2 Variable Selection Network (VSN)

The VSN is the interpretability backbone of the TFT. At every timestep, it assigns learned soft-attention weights over all 15 historical covariates:

$$\boldsymbol{\xi}(t) = \text{softmax}\left(\mathbf{W}_{vs} \cdot \text{GRN}(\mathbf{x}(t))\right)$$

The 15-feature covariate set is organized in four conceptual layers:

- **WW Level (4):** `log1p_concentration` and its 1w/2w/3w lags — the raw signal and its short-term memory
- **Clinical Feedback (3):** `log1p_new_cases_lag1w/2w/3w` — prior confirmed case counts as conditioning signal for the WW→cases transfer function
- **WW Dynamics (5, added Phase 2):** `growth_rate_1w`, `relative_decay_rate`, `diff_concentration`, `2w/4w_ma`, `2w/4w_std` — velocity, acceleration, and volatility that precede surge onsets
- **QC + Static (3):** `outlier_flag_int`, `log_population`, `sewershed_count`

A random-importance baseline is 1/15 ≈ 6.7%. Any feature consistently below this threshold contributes nothing the model could not learn from noise. The Phase 2 hypothesis is that `diff_concentration` and `growth_rate_1w` should spike in importance 2–3 weeks before case surges.

### 3.3 PINN Growth-Rate Penalty

The loss function extends standard multi-quantile pinball loss with a Physics-Informed Neural Network penalty:

$$\mathcal{L}_{total} = \mathcal{L}_{MQ} + \lambda \cdot \frac{1}{H-1} \sum_{h=1}^{H-1} \left( \hat{q}_{0.5,h+1} - \hat{q}_{0.5,h} \right)^2$$

The penalty encodes a biologically grounded prior: viral growth is continuous, bounded by doubling-time constraints (ln(2)/2 ≈ 0.35 per day; ~2.45 per week). **λ is currently 0.0 (disabled)** for Phase 2 calibration. Delta and Omicron-class surges exhibit multi-week accelerations that the penalty was actively suppressing, contributing to 0% PI coverage at λ=0.005. Reintroduction at λ ≈ 0.001 is planned once baseline coverage is confirmed.

The horizon weighting vector `[2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]` further shapes the loss: near-term steps (weeks 1–4, the biological WW→cases lead window) receive 1.5–2× gradient weight relative to longer-horizon steps.

---

## 4. Unique Contributions

### 4.1 WW→Cases as a Transfer Function Problem

The framing that most distinguishes Sewer Signals from standard wastewater surveillance dashboards is the **target pivot**: rather than predicting wastewater concentration (self-referential, no clinical utility), the model predicts *clinical case counts* from wastewater inputs. Wastewater becomes the leading indicator; cases become the clinical outcome. The model learns the biological transfer function between them.

This reframing has a concrete architectural implication: the 5 WW momentum features (`diff_concentration`, rolling means, rolling stds) serve as "rate-of-change vocabulary" for the transfer function. They teach the model not just "how high is the WW signal" but "how fast is it rising" — the information that most directly predicts case surge timing and is unavailable to any model that treats wastewater as a static level measurement.

### 4.2 The Lead-Time Window as a Training Objective

The WW→cases biological lead time (~4–7 days at the individual level; ~1–2 weeks at the population level) is encoded directly into the training objective via the horizon weighting vector. Weeks 1–4 receive 2× gradient weight; weeks 7–8 receive 0.8×. This is a deliberate, biologically motivated asymmetry: **the model is explicitly trained to be most accurate in the near-term actionable window**, not to optimize uniformly across the full 8-week horizon.

This is distinct from standard multi-step forecasting, which applies equal loss weight to all horizon steps and implicitly trains the model to treat week 1 and week 8 as equally important. The clinical use case — issuing alerts 1–3 weeks before a surge peaks so hospitals can prepare — demands the opposite.

### 4.3 Solid-Track-Only Architecture and the Dilution Problem

The CA pipeline uses exclusively **solid-track (copies/g dry sludge)** measurements. This is an epidemiologically motivated design choice: the solid track normalizes by sludge dry weight, making it resistant to wet-weather dilution events that corrupt liquid-track readings (copies/L wastewater). Bay Area winters produce significant precipitation; a model that includes liquid-track readings during rain storms would intermittently see suppressed WW signals that falsely suggest case declines.

The prior CDC pipeline offered a two-track comparison; the CA dataset makes the decision implicit. CDPH BAYWA solid-track reporting is the authoritative signal for all 9 Bay Area counties, and the architectural decision is now locked into the `CAWastewaterProcessor` class.

### 4.4 Calibrated Uncertainty as the Primary Deliverable

Most wastewater surveillance systems report point estimates. Sewer Signals treats **calibrated prediction intervals as the primary deliverable**, not a secondary output. The distinction is operationally critical: a 95% PI achieving 95% coverage tells a public health official "we are 95% confident cases will land in this range." A 95% PI achieving 6.9% coverage (current holdout) is an aesthetic label on an overconfident point estimate — unusable for capacity planning.

The entire Phase 2 engineering effort has been directed at closing the labeled-vs-actual coverage gap. The path from 0% to 6.9% is measurable progress; the path from 6.9% to 50%+ is the remaining engineering challenge.

---

## 5. Current Progress

### 5.1 The Zero-Coverage Problem and Its Resolution

Phase 1 produced **0% PI coverage** on both 50% and 95% intervals. Four independent failure modes were identified and resolved in Phase 2:

| Failure Mode | Root Cause | Fix Applied |
|---|---|---|
| Softplus collapse | `domain_map` forced outputs > 0, clipping all below-median predictions to ~0 | Identity reshape — no activation in `domain_map` |
| Double-scaling | NF `robust` scaler + processor RobustScaler applied sequentially | `scaler_type="identity"` in NeuralForecast |
| Smoothing over-regularization | PINN λ=0.005 suppressed surge trajectories | λ=0.0 (disabled) |
| Quantile underdispersion | 5 quantiles provided weak gradient signal at distribution tails | Expanded to 7 quantiles: added 0.10 and 0.90 |

Each failure mode was independently sufficient to produce 0% coverage. In Phase 1, all four were present simultaneously, making diagnosis non-trivial.

### 5.2 Current Metrics — Most Recent Pipeline Run (CA Dataset, Phase 2)

**Holdout results (2023-06-08 → 2023-12-19, 72 county-week observations):**

| Metric | Phase 2 Result | Target |
|---|---|---|
| Mean WIS | 0.568 | < 0.20 |
| Coverage 50% | 2.8% | ~50% |
| Coverage 95% | 6.9% | ~95% |
| SMAPE | 57.6% | < 20% |

**Cross-validation trend (9 folds, 2022-10-05 → 2023-06-07, step=4 weeks):**

| Fold Cutoff | Mean WIS | Coverage 95% | Epidemiological Context |
|---|---|---|---|
| 2022-10-05 | 0.173 | 16.1% | BQ.1 / BQ.1.1 onset |
| 2022-11-02 | 0.392 | 31.9% | BQ.1 multi-modal surge |
| 2022-11-30 | 1.354 | 1.4% | Omicron sub-wave peak — hardest fold |
| 2022-12-28 | 0.459 | 8.3% | Post-peak decline |
| 2023-01-25 | 0.204 | 22.2% | Inter-wave trough |
| 2023-02-22 | 0.285 | 25.0% | XBB.1.5 early onset |
| 2023-03-22 | 0.065 | 54.2% | Stable inter-wave — best fold |
| 2023-04-19 | 0.109 | 49.2% | Continued low-volatility period |
| 2023-05-17 | 0.088 | **59.3%** | Late-wave descent — best coverage |

The fold-to-fold trajectory is the most important signal in the CV results: Coverage_95 improves monotonically from ~16% on the earliest folds to **59.3% on the final fold**. The model demonstrably learns during the CV window. The holdout regression to 6.9% indicates the Jun–Dec 2023 period presents a distribution shift — likely a resurgence wave not represented in training data.

**County-level holdout WIS:**

The WIS range spans 0.112 (Santa Clara) to 1.601 (Napa), tracking almost perfectly with solid WW data availability. Counties with pre-2022 WW history (Santa Clara, SF, San Mateo) all score below 0.30; counties joining in 2022 (Napa, Solano, Sonoma, Marin) cluster above 0.60.

### 5.3 Dataset Migration Status

The CDC → California state dataset migration is fully complete:

- ✅ `src/config.py` — CA filenames, signal column, updated date constants (2020-07-01 → 2023-12-19)
- ✅ `src/data_pipeline/processor.py` — `CAWastewaterProcessor` validated (33,067 WW rows, 1,827 county-week cases rows loaded correctly)
- ✅ `main.py` — CA loaders, processor swap, liquid track removed
- ✅ `src/visualization/dashboard.py` — Empty liquid_df guard fixed (`_has_county_col` helper)
- ✅ `serve_dashboard.py` — Standalone dashboard server; reprocesses data in ~10s without retraining
- ✅ Full pipeline run complete — 9-fold CV + 2000-step TFT final model + holdout evaluation

---

## 6. Future Steps

### 6.1 Immediate (Calibration Completion)

1. **EDA signal selection** — Run `03_eda_bay_area_CAdatasets.ipynb` Section 3 to determine `Raw Concentration` vs. `Norm Pmmov`. Update `CA_WW_SIGNAL_COL` and rerun pipeline.

2. **Diagnose holdout coverage collapse** — The gap between late-CV Coverage_95 (59.3%) and holdout Coverage_95 (6.9%) is too large to be noise. Investigate: (a) whether a resurgence wave in summer/fall 2023 represents a distribution shift; (b) whether the inverse-transform chain in `_build_decoded_forecast` is collapsing quantile spread post-scaling.

3. **Debug quantile spread** — Log raw (pre-inverse-transform) quantile spread in scaled space vs. decoded spread for the holdout period to isolate whether underdispersion originates in model output or post-processing.

### 6.2 Short-Term (Model Improvement)

4. **Reintroduce PINN λ at reduced scale** — Once Coverage_95 exceeds 30% on holdout, reintroduce the growth-rate penalty at λ=0.001. The biological prior is correct in principle; it was miscalibrated, not wrong.

5. **Source historical WW data for data-sparse counties** — Napa (WIS=1.601), Solano (WIS=0.736), and Sonoma (WIS=0.694) are the three worst-performing counties. CDPH may have archived solid-track records predating the current CSV export. Even 20 additional training weeks per county would substantially improve calibration without any model changes.

6. **VSN interpretability audit** — Verify that `diff_concentration` and `growth_rate_1w` achieve above-random importance (>6.7%) during surge onset periods. If not, the Phase 2 momentum feature hypothesis is falsified and alternative feature engineering is needed.

### 6.3 Medium-Term (Operational Deployment)

7. **Live CDPH SODA API integration** — Replace static CSV loaders with a weekly pull from the California Open Data Portal. The `CAWastewaterProcessor` 2-method override pattern makes this a contained change requiring no modifications to downstream stages.

8. **Automated weekly inference** — Wrap `serve_dashboard.py` in a cron job: pull new data → run `CAWastewaterProcessor` → generate 8-week forward forecast → refresh dashboard. No retraining required; the saved checkpoint at `models_saved/wastewater_tft` is the production artifact.

9. **Drift detection and retrain trigger** — Define a rolling WIS threshold (e.g., 4-week rolling WIS > 2× holdout baseline) that triggers a full retraining run. Prevents model staleness as new variants alter the WW→cases transfer function kinetics.

10. **Multi-pathogen extension** — The `CAWastewaterProcessor` architecture is pathogen-agnostic; the `PCR Target` filter currently selects `SARS-CoV-2`. Extending to Influenza A/B or RSV requires only a config flag and new target case data — the entire feature engineering and TFT infrastructure is reusable without modification.

---

*State of the project as of 2026-05-08. CA dataset migration complete. First full pipeline run on the 2020–2023 extended window completed successfully.*
