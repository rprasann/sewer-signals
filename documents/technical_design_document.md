# Technical Design Document: Sewer Signals
## Probabilistic COVID-19 Outbreak Forecasting via Wastewater Surveillance

**Version:** 3.0 (Phase 2 — CA Dataset Migration)
**Date:** 2026-05-08
**Status:** Active Development

---

## Table of Contents

1. [COVID-19 Overview & Temporal Context](#1-covid-19-overview--temporal-context)
2. [Epidemiological & Biological Constraints](#2-epidemiological--biological-constraints)
3. [Mathematical Formulations](#3-mathematical-formulations)
4. [Model Specifics](#4-model-specifics)
5. [Architectural Justification](#5-architectural-justification)
6. [Technical Significance & Results](#6-technical-significance--results)
7. [Appendix: Implementation Quick Reference](#appendix-implementation-quick-reference)

---

## 1. COVID-19 Overview & Temporal Context

### Version A — Technical

The model's training and evaluation window (2020-07-01 → 2023-12-19) spans four distinct SARS-CoV-2 outbreak epochs in the Bay Area, all captured via California CDPH/BAYWA solid-track wastewater surveillance:

| Wave | Period | Dominant Variant | Epidemiological Character |
|---|---|---|---|
| Wave 1 | Jul–Oct 2020 | Original (D614G) | Summer surge; first wastewater signal detected at Santa Clara/SF/San Mateo |
| Wave 2 | Nov 2020–Apr 2021 | Alpha / Wild-type | Winter surge; steep rise and sustained plateau |
| Wave 3 | Jun–Nov 2021 | Delta (B.1.617.2) | High transmissibility; sharp multi-week acceleration |
| Wave 4 | Dec 2021–May 2022 | Omicron BA.1/BA.2 | Anomalous magnitude (~10× prior peaks); near-vertical ascent |

> **Critical asymmetry:** Only three counties — Santa Clara, San Francisco, and San Mateo — have solid-track WW data from Wave 1 (2020-07-16, 2020-11-09, 2020-12-08 respectively). The remaining six counties join the solid track in 2022. This is handled by `start_padding_enabled=True` in NeuralForecast.

**Dataset boundaries:**
- **Overlap window start:** 2020-07-01 (earliest CA WW solid data — Santa Clara)
- **Overlap window end:** 2023-12-19 (last date in CA Statewide Cases dataset — hard right boundary)
- **CV window:** 2022-10-05 → 2023-06-07 (~35 W-WED weeks, 9 expanding-window folds)
- **Holdout:** 2023-06-08 → 2023-12-19 (~28 W-WED weeks, 3.5× H)

The holdout window captures the post-XBB.1.5 inter-wave descent and any late-2023 resurgence. The 180-week full window — versus the prior 66-week CDC-constrained window — provides 2.7× more training signal and captures 4 complete outbreak waves instead of 3 partial variant epochs.

### Version B — Intuitive

The Bay Area from July 2020 through December 2023 played out in four acts.

The **first summer** showed the original strain making its debut in wastewater. Only the three largest sewersheds — Santa Clara, San Francisco, San Mateo — were measuring at that point. **Winter 2020-21** brought the Alpha wave: a long, grinding plateau. **Summer 2021** brought Delta — faster and steeper than anything before it. Then **Omicron** arrived in December 2021 and simply broke the scale, peaking at case levels roughly 10 times higher than prior waves.

Why does this timeline matter? A model that only trains on post-Omicron data (as our previous CDC-dataset version did) has never seen a "normal-sized" outbreak. It calibrates its sense of scale to Omicron's extraordinary magnitudes. By extending back to July 2020, the model now learns from all four waves — including the smaller, faster Delta surge that more closely resembles the kind of outbreaks a deployed surveillance system would actually need to detect.

---

## 2. Epidemiological & Biological Constraints

### Version A — Technical

#### 2.1 Viral Shedding Kinetics and the WW→Cases Lead Time

SARS-CoV-2 RNA appears in stool 2–3 days post-infection, peaks around symptom onset (day 4–6), and persists for 1–3 weeks. Community-aggregated wastewater signal therefore leads clinical case counts by approximately **4–7 days at the individual level**, which aggregates to approximately **1–2 weeks at the population level** when accounting for:

- Variable incubation period (mean 5.1 days, SD ~2.8 days; log-normal distribution)
- Reporting lag: clinical tests are performed days after symptom onset; results reported days after testing
- Dilution dynamics: wet weather events dilute fecal load; dry weather concentrates it

The model's horizon weighting vector `[2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]` encodes this biological prior: **weeks 1–4 (the WW→cases lead window) are weighted 1.5–2× relative to weeks 5–8**.

#### 2.2 The PINN Growth-Rate Biological Prior

The Physics-Informed Neural Network penalty encodes a soft constraint that viral growth cannot be arbitrarily discontinuous. Specifically, the penalty term is:

$$\mathcal{L}_{PINN} = \mathcal{L}_{MQ} + \lambda \cdot \frac{1}{H-1} \sum_{h=1}^{H-1} \left( \hat{q}_{0.5,h+1} - \hat{q}_{0.5,h} \right)^2$$

Where:
- $\mathcal{L}_{MQ}$ is the multi-quantile (pinball) loss
- $\lambda$ is `GROWTH_RATE_LAMBDA` (currently 0.0 — disabled for Phase 2)
- $\hat{q}_{0.5,h}$ is the predicted median at horizon step $h$

**Why λ=0.0 in Phase 2:** Outbreak waves — particularly Delta and Omicron — exhibit multi-week acceleration trajectories that the growth-rate penalty actively suppresses. At λ=0.005, the penalty adds sufficient smoothing pressure to flatten predicted surges, contributing directly to 0% PI coverage. The penalty should be reintroduced at a smaller value (e.g., λ=0.001) only after calibration is verified against coverage metrics.

#### 2.3 Signal Columns and Measurement Units

The California WW dataset (`California_Wastewater_Surveillance_Data.csv`) reports two normalization pathways per sample:

| Signal Column | Normalization | Notes |
|---|---|---|
| `Raw Concentration` (copies/g dry sludge) | Sludge dry weight | Primary signal; stable across facilities and wet-weather events |
| `Norm Pmmov` | PMMoV fecal indicator virus ratio | Corrects for sewage dilution; more sensitive to low-prevalence periods |

The pipeline currently uses `Raw Concentration` as `CA_WW_SIGNAL_COL`. The EDA notebook (`03_eda_bay_area_CAdatasets.ipynb`) Section 3 computes normalized [0–1] Pearson correlations against case counts per outbreak window to determine whether `Norm Pmmov` should replace it for future runs.

**Only the solid track is used.** The CA dataset is filtered to `Sample Type == "solid"` before any processing. There is no liquid track in the CA pipeline; the `_process_liquid_track()` function returns an empty DataFrame.

All WW signals are log1p-transformed before use:

$$x_{ww} = \log(1 + \text{copies/g})$$

This transformation compresses the dynamic range (~4 orders of magnitude across Waves 1–4) to a learnable scale while mapping zero to zero (no viral RNA detected = 0, not undefined).

### Version B — Intuitive

#### Wastewater as a Time Traveler

When someone gets infected with COVID, the virus starts shedding in their gut almost immediately — often before they feel sick, and certainly before they get tested. That viral RNA flows through the toilet, into the sewer, and reaches the wastewater treatment plant where sensors detect it.

By the time a person tests positive and gets counted in the case statistics, the wastewater was already telling us 1–2 weeks ago. This is the core value proposition: **wastewater is a leading indicator that travels ahead of clinical data**.

The model is designed to exploit this lead time. The near-term forecast horizon (weeks 1–4) gets extra emphasis in training because that's the window where the wastewater signal is most actionable — where a clinician can still prepare hospital capacity, pre-position antivirals, or issue public guidance.

#### Two Signal Columns, One Decision

The California dataset offers two ways to measure the same virus: raw copies per gram of sludge, or a ratio normalized to a "fecal marker" virus called PMMoV. The raw signal is more stable day to day; the normalized signal is more sensitive when virus levels are low. The EDA notebook compares both against actual case counts at each of the four outbreak peaks to let the data decide which signal correlates better. Until that analysis runs, we default to raw concentration.

#### Why the Physics Penalty Was Disabled

We originally included a "smoothness prior" — a mathematical incentive for the model to predict gradual changes rather than sudden jumps. In theory, viruses spread continuously, not discontinuously. In practice, at the population level (aggregated across a county), case surges can appear very sudden in weekly data.

The smoothness prior was suppressing exactly the surges we needed to detect. So for this development phase, we turned it off entirely and let the data speak for itself.

---

## 3. Mathematical Formulations

### 3.1 Target Transformation

**Raw target:** Weekly new COVID-19 cases (count, integer ≥ 0), resampled from the daily California Statewide Cases/Deaths/Tests dataset to Wednesday-anchored weekly bins.

**Step 1 — Resample (cases only):**

The CA Cases dataset is daily. Before entering the processor, cases are summed to W-WED bins per county:

$$\text{new\_cases}_w = \sum_{d \in \text{week}_w} \text{cases}_d$$

**Step 2 — Log compression:**
$$y_{raw} = \log(1 + \text{new\_cases})$$

Properties: Zero-preserving, monotone, compresses right tail, maps domain [0, ∞) to [0, ∞).

**Step 3 — RobustScaler (leakage-free):**
$$y = \frac{y_{raw} - Q_{50}(\text{train})}{Q_{75}(\text{train}) - Q_{25}(\text{train})}$$

The scaler is fit **on training rows only** (rows ≤ 2022-10-05). Val/test rows are transformed using the training-fit scaler. This ensures scaling parameters contain no future information.

**Critical implication:** The scaled target $y$ can be **legitimately negative** (below-median weeks). The Softplus activation $\log(1 + e^x)$ applied in `domain_map` clips all negatives toward zero, collapsing quantile spread. Softplus was removed; `domain_map` is now an identity reshape.

### 3.2 Pinball Loss (Quantile Loss)

For a single quantile level $\tau \in (0,1)$, prediction $\hat{y}$, and actual $y$:

$$\rho_\tau(y, \hat{y}) = \begin{cases} \tau \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1 - \tau) \cdot (\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

The multi-quantile loss over $K$ quantile levels and $N$ samples:

$$\mathcal{L}_{MQ} = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} w_h \cdot \rho_{\tau_k}(y_i, \hat{y}_{i,\tau_k})$$

Where $w_h$ is the horizon weight for step $h$. The model outputs 7 quantile levels:

$$\boldsymbol{\tau} = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]$$

### 3.3 Weighted Interval Score (WIS)

WIS is the proper scoring rule used for evaluation. It decomposes into sharpness and calibration:

$$\text{WIS}(y, \hat{F}) = \frac{1}{K} \sum_{k=1}^{K} \text{IS}_{\alpha_k}(y, l_k, u_k)$$

Where for a prediction interval $[l, u]$ at coverage level $1-\alpha$:

$$\text{IS}_\alpha(y, l, u) = (u - l) + \frac{2}{\alpha}(l - y)\mathbf{1}[y < l] + \frac{2}{\alpha}(y - u)\mathbf{1}[y > u]$$

Lower WIS is better. A sharp interval that misses pays a heavy penalty; a wide interval that always covers pays only an interval-width penalty.

### 3.4 Coverage Metrics

$$\text{Coverage}_{50} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[y_i \in [\hat{q}_{0.25,i}, \hat{q}_{0.75,i}]]$$

$$\text{Coverage}_{95} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[y_i \in [\hat{q}_{0.025,i}, \hat{q}_{0.975,i}]]$$

A well-calibrated model should achieve Coverage_50 ≈ 0.50 and Coverage_95 ≈ 0.95. The Phase 1 baseline achieved 0% on both. Phase 2 with CA datasets achieved 2.8% / 6.9% on the holdout — non-zero, but calibration remains the primary open problem.

### 3.5 Symmetric Mean Absolute Percentage Error (SMAPE)

$$\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

Used for point forecast accuracy on the median quantile. Symmetric formulation handles near-zero actuals better than standard MAPE.

### 3.6 OutbreakDetector Algorithm

Three conditions must hold simultaneously for an outbreak alert at time $t$:

1. **Growth rate:** $\frac{\hat{q}_{0.5,t+1} - \hat{q}_{0.5,t}}{|\hat{q}_{0.5,t}| + \epsilon} \geq 0.25$ (≥25% predicted weekly growth)
2. **Absolute floor:** $\hat{q}_{0.5,t+1} > \theta_{floor}$ (forecast above noise floor; prevents alerts on zero-baseline noise)
3. **Sustained signal:** Condition 1 holds for ≥3 consecutive horizon steps

The alert state is encoded as a boolean flag per county per week and rendered in the dashboard with a red/green indicator.

---

## 4. Model Specifics

### 4.1 Variable Selection Network (VSN) — Soft Attention Mechanism

The TFT's VSN assigns learned importance weights to each input feature at every time step:

$$\boldsymbol{\xi}(t) = \text{softmax}\left(\mathbf{W}_{vs} \cdot \text{GRN}(\mathbf{x}(t))\right)$$

$$\tilde{\mathbf{x}}(t) = \sum_{j=1}^{J} \xi_j(t) \cdot \tilde{x}_j(t)$$

Where:
- $J = 15$ (number of hist_exog covariates)
- $\xi_j(t)$ is the learned importance weight for feature $j$ at time $t$ (sums to 1)
- $\text{GRN}(\cdot)$ is a Gated Residual Network applied per feature
- $\tilde{x}_j(t)$ is the projected representation of feature $j$

The VSN weights are interpretable: higher $\xi_j$ means the model is leaning on feature $j$ more at that time step. This produces scientifically meaningful interpretability — we can see which WW dynamics the model prioritizes during surge vs. decline phases.

A random-importance baseline is $1/15 \approx 6.7\%$ per feature. Any feature consistently below this threshold is contributing nothing the model could not learn from noise.

### 4.2 The 15-Feature Covariate Set

| Feature | Type | Rationale |
|---|---|---|
| `log1p_concentration` | WW Level | Primary wastewater signal |
| `log1p_concentration_lag1w` | WW Level | 1-week memory |
| `log1p_concentration_lag2w` | WW Level | 2-week memory |
| `log1p_concentration_lag3w` | WW Level | 3-week memory |
| `log1p_new_cases_lag1w` | Cases Level | Clinical feedback, 1 week |
| `log1p_new_cases_lag2w` | Cases Level | Clinical feedback, 2 weeks |
| `log1p_new_cases_lag3w` | Cases Level | Clinical feedback, 3 weeks |
| `growth_rate_1w` | WW Dynamics | Relative week-over-week velocity |
| `relative_decay_rate` | WW Dynamics | PINN-derived exponential decay parameter |
| `outlier_flag_int` | QC | Binary flag for anomalous WW readings |
| `diff_concentration` | WW Momentum | Absolute weekly velocity (Δ log1p_conc) |
| `log1p_concentration_2w_ma` | WW Momentum | 2-week rolling mean (short baseline) |
| `log1p_concentration_4w_ma` | WW Momentum | 4-week rolling mean (medium baseline) |
| `log1p_concentration_2w_std` | WW Momentum | Local volatility — spikes during onset |
| `log1p_concentration_4w_std` | WW Momentum | Medium-window volatility |

The 5 momentum features (rows 11–15) were added in Phase 2 to give the model vocabulary for **velocity and acceleration patterns** that precede surge onsets — precisely the information needed to detect rapid rises like the Delta and Omicron waves.

### 4.3 Why Softplus Was Removed from domain_map

**The failure:**
```python
# Phase 1 (wrong):
def domain_map(self, y_hat):
    return F.softplus(y_hat)  # Forces outputs > 0
```

The model predicts in **RobustScaled space**, where the median is 0 by construction. Below-median weeks (roughly half of all observations) are negative in this space. Applying Softplus maps all negatives toward zero, collapsing the lower quantiles to a near-zero floor. The model learns to output a tight, near-zero-floored bundle — resulting in 0% PI coverage when actuals include negative (below-median) values.

**The fix:**
```python
# Phase 2 (correct):
def domain_map(self, y_hat):
    return y_hat.reshape(y_hat.shape[0], -1, self.n_outputs)  # Identity
```

No activation applied. The model outputs raw quantile predictions in scaled space. The post-processing pipeline (`_build_decoded_forecast`) handles inverse_transform → expm1 → clip to recover human-interpretable case counts.

### 4.4 Double-Scaling Failure Mode

**The failure:** Setting `scaler_type="robust"` in NeuralForecast while also applying RobustScaler in the data processor results in double-scaling:

1. Processor scales: $y \rightarrow \frac{y - Q_{50}}{IQR}$
2. NeuralForecast scales again: $y' \rightarrow \frac{y' - Q'_{50}}{IQR'}$

If IQR ≈ 1.0 after the first scaling, the second scaling compresses variance by an additional factor of ~1/IQR of the already-scaled data. This collapses the effective dynamic range to near-zero, making all quantiles predict essentially the same value.

**The fix:** `scaler_type="identity"` in NeuralForecast. The processor's RobustScaler is the single source of truth for scaling.

### 4.5 Horizon Weighting

The horizon weight vector $\mathbf{w} = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]$ is applied per-step during loss computation. It must be passed as `np.array(..., dtype=np.float32)` — NeuralForecast's `BasePointLoss.__init__` calls `.flatten()` on it, and a plain Python list raises `AttributeError`.

The weighting reflects the biological lead-time structure: the WW→cases window is approximately weeks 1–4, so these steps receive 1.5–2× gradient weight. Weeks 5–8 are de-emphasized (weight 0.8) — the model should not optimize heavily for long-horizon extrapolation at the expense of near-term precision.

---

## 5. Architectural Justification

### 5.1 Model Selection: TFT vs. Alternatives

| Model | Why Rejected / Inferior |
|---|---|
| **ARIMA / SARIMA** | Univariate, no covariate support; requires stationarity; cannot model non-linear dynamics |
| **Prophet** | Additive seasonality assumption; no multi-variate hist_exog; weak at irregular outbreak patterns |
| **LSTM** | No interpretability (black box); no built-in attention over input features; harder to train |
| **Vanilla Transformer** | No inductive bias for time series; point forecast only without separate quantile head wiring |
| **Gaussian Process** | Cubic complexity in sequence length; does not scale to multi-county global model |
| **TFT (chosen)** | Native multi-quantile, native hist_exog, interpretable VSN, proven on medical time series |

**Why TFT specifically:** The Temporal Fusion Transformer (Lim et al., 2020) combines:
- Variable Selection Networks for covariate importance
- Gated Residual Networks for skip-connection learning
- Multi-head attention over past time steps (temporal attention)
- Separate static, historical, and future covariate channels
- Native multi-quantile output head

This architecture was designed for exactly this problem class: heterogeneous multi-variate time series with multi-horizon probabilistic forecasting.

### 5.2 Global Model: One Model Across All 9 Counties

**The alternative (local models):** Train a separate TFT per county.

**Why global is correct:**
- Training data per county with the CA dataset: ~117 W-WED weeks up to the first CV cutoff for the 3 early-start counties; as few as 2 weeks for Napa
- A global model trained on 9 counties sees ~408 total training county-week rows, enabling the model to learn generalizable outbreak dynamics from cross-county signal
- Counties share the same viral variants (same regional wave timing), the same California wastewater measurement protocols (CDPH BAYWA), and overlapping demographic structures
- The model uses county ID as a static covariate (`unique_id`), allowing it to learn county-specific biases while sharing temporal dynamics across all counties

### 5.3 Attention Mechanism as Lead-Time Mechanism

The temporal self-attention component of TFT learns which past timesteps are most predictive of the future. During outbreak onset, we expect the model to assign high attention weight to the most recent 1–3 wastewater readings (the leading edge of the signal) rather than the longer-horizon baseline.

The attention heatmap visualization in the dashboard makes this directly auditable: a scientifically coherent model should show high attention weight on recent WW readings during surge phases, and lower weight during stable inter-wave periods.

### 5.4 Wednesday-Anchored Resampling (W-WED)

Both CA datasets require Wednesday-anchored weekly alignment:

- **CA WW data** has facility-level collection on variable weekdays. Median to county-day, then resample to `W-WED` bins aligns most collection events (the dominant BAYWA collection day is Wednesday).
- **CA Cases data** is daily. Summed to `W-WED` bins per county before entering the processor: `resample("W-WED").sum()`.

This alignment ensures:
1. The WW and cases time spines are identical — no off-by-one-day misalignment in the merge
2. The `pd.date_range(..., freq="4W-WED")` CV cutoffs land exactly on actual observation dates, preventing fold boundary misalignment

Using `"4W"` (Sunday-anchored) shifts all cutoffs by 4 days, causing the fold boundaries to land on dates with no observations, silently breaking the CV expanding-window logic.

### 5.5 Short-History Counties and start_padding

Five counties have fewer than `INPUT_SIZE + H = 26 + 8 = 34` training weeks, as confirmed by the pre-flight validator in the most recent run:

| County | Training Weeks | First WW Data |
|---|---|---|
| Napa | 2 | ~Sep 2022 |
| Solano | 3 | ~Sep 2022 |
| Sonoma | 15 | ~Jul 2022 |
| Marin | 16 | ~Jun 2022 |
| Contra Costa | 29 | ~Mar 2022 |

NeuralForecast's `start_padding_enabled=True` zero-pads the history to meet the minimum input size requirement. These counties produce valid predictions but with measurably lower reliability — the holdout WIS values confirm this directly: Napa (1.601), Marin (1.028), Solano (0.736), Sonoma (0.694) versus Santa Clara (0.112) and SF (0.138), which have had solid WW data since mid-2020.

### 5.6 CA Data Architecture: CAWastewaterProcessor

The transition from CDC NWSS datasets to California state datasets required a new processor subclass. The CDC pipeline had 7 dataset-specific preprocessing stages (FIPS explosion, unit filtering, QC flags, non-detect censoring) that do not apply to the CA WW format. The solution: `CAWastewaterProcessor` inherits from `WastewaterProcessor` and overrides only `run()`, replacing those 7 stages with 2 CA-specific ones, then calling all 7 downstream inherited stages unchanged.

```
CDC pipeline (WastewaterProcessor.run):
  Stages 1–7: FIPS join, unit filter, QC, non-detect censoring, outlier filter, ...
  Stages 8–15: rolling smooth, weekly resample, log transform, merge cases, ...

CA pipeline (CAWastewaterProcessor.run):
  _clean_ca_ww()                  # renames cols, maps county→FIPS, coerces numeric
  _aggregate_ca_to_county_daily() # median per county-day; counts sewersheds
  [Stages 9–15 inherited unchanged]
```

**County → FIPS mapping:** The CA WW dataset uses county names ("Alameda", "San Francisco") rather than FIPS codes. `_clean_ca_ww()` maps names to 5-digit FIPS via `BAY_AREA_FIPS`, keeping `COUNTY_COL` as FIPS throughout all downstream stages (groupby, merge, validation, display) with zero changes to those stages.

**Cases loader:** `_load_ca_cases_csv()` in `main.py` resamples daily cases to W-WED before returning, producing `(COUNTY_COL, NWSS_DATE_COL, new_cases)` — exactly the format `WastewaterProcessor._merge_cases()` expects, so that method inherits without modification.

---

## 6. Technical Significance & Results

### 6.1 Calibration Failure Taxonomy

The Phase 1 → Phase 2 evolution uncovered a specific, reproducible taxonomy of probabilistic calibration failures in NeuralForecast-based pipelines:

| Failure Mode | Mechanism | Symptom | Fix |
|---|---|---|---|
| **Softplus collapse** | Activation clips negative scaled values to ~0 floor | All quantiles near-zero; 0% coverage | Remove Softplus from domain_map |
| **Double-scaling** | Processor + NF both apply RobustScaler | Variance compressed to noise floor | Set scaler_type="identity" in NF |
| **Smoothing prior over-regularization** | Growth-rate penalty λ suppresses surge trajectories | Upper quantile below actuals during surges | Set λ=0.0 (or reduce substantially) |
| **Quantile underdispersion** | 5 quantiles insufficient for tail gradient signal | Pinball loss has weak gradient at extremes | Expand to 7 quantiles including 0.10/0.90 |

Each failure mode is independently sufficient to produce 0% PI coverage. In Phase 1, all four were present simultaneously.

### 6.2 Cross-Validation Results (9 Expanding-Window Folds)

CV window: 2022-10-05 → 2023-06-07, step = 4 weeks, H = 8 weeks.

| Fold Cutoff | Mean WIS | Coverage 50% | Coverage 95% | SMAPE | Epidemiological Context |
|---|---|---|---|---|---|
| 2022-10-05 | 0.173 | 16.1% | 16.1% | 79.3% | Omicron BA.2 descent / BQ.1 onset |
| 2022-11-02 | 0.392 | 5.6% | 31.9% | 88.0% | BQ.1 / BQ.1.1 multi-modal surge |
| 2022-11-30 | **1.354** | 1.4% | 1.4% | 163.8% | Omicron sub-wave peak — hardest fold |
| 2022-12-28 | 0.459 | 4.2% | 8.3% | 111.7% | Post-peak decline |
| 2023-01-25 | 0.204 | 1.4% | 22.2% | 53.3% | Inter-wave trough |
| 2023-02-22 | 0.285 | 13.9% | 25.0% | 59.4% | XBB.1.5 early onset |
| 2023-03-22 | **0.065** | 16.7% | **54.2%** | 9.8% | Stable inter-wave — best fold |
| 2023-04-19 | 0.109 | 23.8% | 49.2% | 13.1% | Continued low-volatility period |
| 2023-05-17 | 0.088 | 29.6% | **59.3%** | 9.3% | Late-wave descent — best 95% coverage |

**Key observation:** The model dramatically improves on later folds. Folds 7–9 (Mar–May 2023) achieve 49–59% Coverage_95 and sub-0.11 WIS — approaching calibration targets during stable inter-wave periods. Folds 2–4 coincide with the BQ.1/Omicron sub-wave — the most difficult epidemiological regime — where coverage collapses. This fold-to-fold variance is not random: it tracks outbreak phase with high fidelity.

### 6.3 Holdout Evaluation Results (2023-06-08 → 2023-12-19)

72 county-week observations across 9 counties, H=8 forecast steps.

| Metric | Value | Interpretation |
|---|---|---|
| Mean WIS | 0.568 | Higher than late-CV folds; holdout period includes resurgence activity |
| Coverage 50% | **2.8%** | Target: 50% — significant underdispersion remains |
| Coverage 95% | **6.9%** | Target: 95% — upper quantile still below actuals during surges |
| SMAPE | 57.6% | Point forecast accuracy; median prediction consistently off during surge onset |
| Predicted alerts | 6 | Model detects some elevated periods |
| Actual onsets detected | 0 | No ground-truth outbreak onset labels in this holdout window |

**County-level WIS breakdown:**

| County | WIS | Solid WW Data Since | Note |
|---|---|---|---|
| Santa Clara | **0.112** | Jul 2020 | Most historical data; best calibrated |
| San Francisco | **0.138** | Nov 2020 | Second-most data; dense urban sewershed |
| Alameda | 0.301 | 2022 | Adequate history after CV window opens |
| San Mateo | 0.268 | Dec 2020 | Good early history |
| Contra Costa | 0.236 | 2022 | 29 training weeks; adequate |
| Solano | 0.736 | ~Sep 2022 | Only 3 training weeks; zero-padded |
| Sonoma | 0.694 | ~Jul 2022 | 15 training weeks; marginal |
| Marin | **1.028** | ~Jun 2022 | 16 training weeks; high error |
| Napa | **1.601** | ~Sep 2022 | 2 training weeks — worst; near-random |

The correlation between WIS and historical data length is strong and causal: counties with more pre-2022 WW history produce far better-calibrated forecasts. This directly motivates backfilling solid-track data for Napa, Solano, and Sonoma if available from the CDPH archive.

### 6.4 VSN as a Scientific Deliverable

Beyond prediction accuracy, the VSN attention weights constitute a **scientific hypothesis test**: given a county's wastewater time series, which features does the model weight most when a surge is about to occur?

If the model is learning the right inductive structure, we expect:
- `diff_concentration` and `growth_rate_1w` to spike in importance 2–3 weeks before a case surge
- `log1p_concentration` (level) to be primary during stable inter-wave periods
- `log1p_concentration_2w_std` and `_4w_std` (volatility) to rise during uncertain onset phases

This is falsifiable. If the VSN weights are approximately uniform (1/15 ≈ 6.7% each, no concentration of attention), the model is not exploiting feature structure — it is fitting noise.

### 6.5 Expanding-Window CV as a Realistic Evaluation Protocol

Standard train/test splits are inappropriate for this domain because they assume i.i.d. data. The outbreak detection problem has:
- Strong temporal autocorrelation (this week's cases predict next week's)
- Concept drift (each variant epoch has different growth kinetics)
- Non-stationarity (the wastewater→cases relationship changes with immunity level)

Expanding-window CV with 4-week steps simulates a realistic deployment scenario: the model is trained on all data up to a cutoff and evaluated on the next 8 weeks. This:
- Preserves temporal ordering (no future leakage)
- Evaluates the model across multiple out-of-sample epidemiological regimes
- Provides 9 independent WIS scores that can be aggregated for a more robust estimate than a single train/test split

The final holdout is entirely excluded from CV to preserve a clean, unseen evaluation of the 2023 post-wave period.

### 6.6 The Target Pivot and Dataset Migration

**The original design** had `log1p_concentration` (wastewater) as both input and output. This is circular: it predicts wastewater from wastewater, provides no clinical decision support value, and the clinically relevant question is: "How many people will test positive next week?"

**The pivot to `log1p_new_cases`** reframes the problem correctly:
- WW is the **leading indicator** (hist_exog input)
- Cases are the **clinical outcome** (prediction target)
- The model learns the WW→cases transfer function, which is the scientifically meaningful relationship

**The dataset migration** from CDC NWSS to California state datasets extended this further:

| Dimension | CDC Pipeline | CA Pipeline |
|---|---|---|
| WW dataset | CDC NWSS (`2ew6-ywp6.json`) | CA CDPH BAYWA solid track |
| Cases dataset | CDC county-level archived (`pwn4-m3yp.json`) | CA Statewide COVID-19 Cases/Deaths/Tests |
| Overlap window | 2022-02-07 → 2023-05-10 (~66 weeks) | 2020-07-01 → 2023-12-19 (~180 weeks) |
| Outbreak waves covered | 3 partial (BA.2, BQ.1, XBB.1.5) | 4 complete (Wave 1 through Omicron) |
| County resolution | FIPS-native | County-name → FIPS mapped |
| Cases temporal resolution | Weekly (pre-aggregated) | Daily → W-WED resampled |
| Liquid track | Available (copies/L) | Unavailable (solid only) |
| Processor class | `WastewaterProcessor` | `CAWastewaterProcessor` (subclass) |

The extended window is the most significant improvement: 2.7× more training data and exposure to small, fast outbreaks (Delta) that teach the model scale-agnostic surge dynamics — critical for avoiding over-calibration to Omicron's anomalous magnitude.

---

## Appendix: Implementation Quick Reference

### File Structure

```
wastewater/
├── main.py                          # Entry point; CV + final model + dashboard
├── serve_dashboard.py               # Standalone dashboard server (no retraining)
├── src/
│   ├── config.py                    # TFT_CONFIG, PINN lambda, quantiles, horizon_weight
│   ├── data_pipeline/
│   │   └── processor.py             # WastewaterProcessor + CAWastewaterProcessor
│   ├── models/
│   │   ├── tft_model.py             # WastewaterTFT class; HIST_COVARIATES; loss wiring
│   │   └── loss_functions.py        # PINNWastewaterLoss; domain_map (identity)
│   ├── evaluation/
│   │   └── metrics.py               # evaluate(); expanding_window_cv(); WIS/coverage
│   ├── visualization/
│   │   ├── attention_plots.py       # All Plotly figures; color palette; VSN heatmap
│   │   └── dashboard.py             # Dash app; layout; callbacks
│   └── utils/
│       └── helpers.py               # LLM narrative generation; misc utilities
├── documents/
│   ├── 03_eda_bay_area_CAdatasets.ipynb  # EDA for CA datasets (signal comparison)
│   └── technical_design_document.md      # This document
├── data/
│   ├── raw/
│   │   ├── California_Wastewater_Surveillance_Data.csv
│   │   └── Statewide_COVID-19_Cases_Deaths_Tests.csv
│   └── processed/                   # forecast.parquet, train/val/test.parquet, etc.
└── models_saved/
    └── wastewater_tft/              # NeuralForecast checkpoint
```

### Key Configuration Values (Phase 2 — CA Dataset)

```python
# src/config.py

# CA dataset filenames
CA_WW_FILENAME    = "California_Wastewater_Surveillance_Data.csv"
CA_CASES_FILENAME = "Statewide_COVID-19_Cases_Deaths_Tests.csv"
CA_WW_SIGNAL_COL  = "Raw Concentration"  # update to "Norm Pmmov" if EDA Section 3 favours it

# Overlap window (CA datasets)
DATA_START_DATE = "2020-07-01"   # earliest CA WW solid data (Santa Clara)
DATA_END_DATE   = "2023-12-19"   # last date in CA Cases dataset
TRAIN_END_DATE  = "2022-10-05"   # first CV cutoff (all 9 counties active)
VAL_END_DATE    = "2023-06-07"   # end of CV window (~35 weeks, 9 folds)

# Model
TARGET_COL = "log1p_new_cases"
WW_FEATURE_COL = "log1p_concentration"
GROWTH_RATE_LAMBDA = 0.0  # disabled — re-enable at 0.001 after coverage verified

TFT_CONFIG = {
    "n_quantiles": 7,
    "quantile_levels": [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975],
    "horizon_weight": [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8],  # np.float32 array
    "scaler_type": "identity",
    "start_padding_enabled": True,
    ...
}
```

### CLI Reference

```bash
# Full run (CV + final model + dashboard)
python main.py

# Skip CV, skip dashboard (fastest end-to-end check)
python main.py --skip-cv --no-dash

# Fast mode (reduced steps for smoke test)
python main.py --fast --no-dash

# Custom steps
python main.py --max-steps 500 --no-dash

# Launch dashboard from saved artefacts (no retraining)
python serve_dashboard.py
python serve_dashboard.py --port 8051
```

### Known Bugs and Watchpoints

1. **`horizon_weight` must be `np.array`** — `BasePointLoss` calls `.flatten()`. A Python list raises `AttributeError`.
2. **Rich progress bar nesting** — pass `enable_progress_bar=False, enable_model_summary=False` in `cv_trainer_kwargs` to avoid `IndexError: pop from empty list`.
3. **`val_size=0` requires `early_stop_patience_steps=-1`** — NeuralForecast rejects `val_size=0` with early stopping enabled.
4. **CV cutoff frequency must be `"4W-WED"`** — `"4W"` anchors to Sunday (wrong); `"4W-WED"` anchors to Wednesday (correct).
5. **loguru format strings use `{}`** — not `%s`/`%d`. `logger.warning("foo: {}", value)`.
6. **Empty `liquid_df` in dashboard** — `_process_liquid_track()` returns `pd.DataFrame()` (no columns), not `None`. Dashboard's `_has_county_col()` guard handles this; the old `is not None` check would raise `KeyError: 'county_fips'`.
7. **CA Cases date format is `%m/%d/%y`** — not ISO. Must pass `format="%m/%d/%y"` explicitly to `pd.to_datetime()`; the default parser misreads two-digit years.
8. **Napa/Solano essentially zero-padded** — 2 and 3 training weeks respectively. Their forecasts are near-random (WIS ~1.0–1.6); treat county-level alerts for these two counties as low-confidence until additional historical WW data is sourced.
