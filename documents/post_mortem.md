# Post-Mortem: Sewer Signals
## A Pain Points Walkthrough — From Circular Predictions to Outbreak Detection

**Role:** Senior ML Engineer retrospective
**Audience:** AI/ML Instructor (highly technical)
**Project:** Probabilistic COVID-19 outbreak forecasting via wastewater surveillance
**Format:** Problem → Investigation → Solution → Insight

---

## Preface

This document is not a tutorial. It is a forensic account of four distinct technical crises that emerged during the development of a probabilistic TFT-based forecasting system. Each section follows the same structure: what went wrong, how we investigated it, what we changed, and what the failure revealed about the problem class.

The crises were not sequential. Several were simultaneously active. The most disorienting moment in this project was realising that 0% PI coverage — the central metric failure — had four independent root causes, each of which was individually sufficient to produce the symptom. We were debugging a system where every fix was necessary but none was individually sufficient.

---

## 1. The Target-Feature Mismatch

### The Problem

The original model was built with `log1p_concentration` (log-transformed wastewater RNA copies/gram dry sludge) as both the primary input feature and the prediction target `y`. The architecture was syntactically valid, trained without errors, and produced smooth, visually plausible forecasts.

It was also scientifically incoherent.

Predicting wastewater from wastewater is a self-referential task with no clinical utility. The system answered the question: "Given today's WW signal, what will the WW signal be next week?" A clinician cannot act on that answer. Hospital capacity planning, antiviral pre-positioning, and public health guidance all require a case count forecast, not a sewer measurement forecast.

The model had been optimised for the wrong objective from day one.

### The Investigation

The mismatch became undeniable when we examined the evaluation metrics in context. Even when WIS was non-trivial, the forecast traces were smooth exponential tails — the model was extrapolating the wastewater signal's own dynamics rather than learning to translate wastewater patterns into clinical outcomes.

The deeper problem was architectural: if `log1p_concentration` is both input and target, the model has a trivially easy path to low training loss — copy the input forward with a learned decay constant. This is not forecasting; it is curve-fitting to the wrong curve. The Variable Selection Network (VSN) would never learn meaningful feature attribution because the dominant signal (copy the WW forward) would overwhelm all covariate relationships.

Examining the CDC archived cases dataset revealed something critical: there was a well-defined overlap window between wastewater observations and clinical case counts — 2022-02-07 through 2023-05-10 (the CDC archived cases dataset ends at this hard boundary). Within this window, the wastewater signal systematically leads case counts by 1–2 weeks at the population level.

That lead-time relationship was the signal we were failing to learn.

### The Solution

**Target pivot:** `y = log1p_new_cases` (log1p of weekly new COVID-19 cases, then RobustScaled)
**Input reclassification:** `log1p_concentration` demoted to `hist_exog` — a leading indicator input, never the output

This required coordinated changes across the entire pipeline:

1. **Data pipeline:** A Stage 12 cases merge was added to `processor.py`, joining the CDC archived cases CSV on `(county, week_ending_date)` and computing `log1p_new_cases` as the target column.

2. **Temporal spine alignment:** The CDC data uses epidemiological weeks ending on specific dates. Misaligning the wastewater spine by even one day would destroy the join. This forced the adoption of `W-WED` (Wednesday-anchored) resampling throughout — discussed further in Section 4.

3. **RobustScaler leakage discipline:** The target is now a real clinical outcome with a meaningful inter-quartile range. The scaler must be fit exclusively on training rows and applied (transform only) to val/test rows. Any leakage of val/test statistics into the scaling would contaminate the evaluation.

4. **Lag feature cascade:** Once `log1p_new_cases` became the target, its own lag values (`log1p_new_cases_lag1w/2w/3w`) became valid hist_exog inputs — the model can now learn from the relationship between past cases and past WW to project future cases.

5. **LLM narrative prompt update:** `helpers.py` contained a natural language prompt template for generating county-level narrative summaries. It referred to "wastewater concentration" as the forecast output. This was updated to reflect that the forecast output is weekly new cases.

### The Insight

**Circular targets are a silent failure mode.** The model trains, the loss decreases, and everything looks normal. There is no error message for "your target is also your input." The failure only becomes visible when you ask whether the model is solving a clinically meaningful problem.

This is a broader pattern in applied ML: **the definition of the prediction target is not a technical choice, it is a scientific hypothesis.** The original target encoded the implicit hypothesis that wastewater dynamics are self-predictive. The pivoted target encodes the hypothesis that wastewater is a leading indicator of clinical outcomes. Only the second hypothesis has actionable implications.

The downstream consequence of the pivot was significant: the training window shrank from the full wastewater record to the CDC overlap window (roughly 15 months), and the model's evaluation became harder — predicting case counts from WW features is a more difficult transfer function to learn than predicting WW from WW. But the difficulty is appropriate. An easy problem with no clinical value is not worth solving.

---

## 2. The Zero-Coverage Crisis

### The Problem

After the target pivot, the model trained cleanly and produced a set of 7 quantile forecasts over the holdout period. Evaluation returned:

```
coverage_50:  0.00
coverage_95:  0.00
mean_wis:     [non-trivial value]
smape:        [non-trivial value]
```

Zero percent PI coverage at both the 50% and 95% intervals. The upper quantile (q0.975) was **consistently below the actuals**. The entire probability mass was below reality.

A calibrated model should have Coverage_95 ≈ 0.95 (95% of actuals fall within the 95% PI). A Coverage_95 of 0.00 means the model's most optimistic quantile was less than every single actual value. This is not miscalibration — it is the model being maximally wrong about the direction of its own uncertainty.

Visual inspection of the forecast confirmed it: the model predicted smooth, linear-decaying trajectories. Reality during the holdout (XBB.1.5 wave) was a rapid multi-week surge.

### The Investigation

A 0% coverage outcome is diagnostically unusual. It implies that the model's outputs are not merely noisy or wide in the wrong direction — they are **systematically wrong**. This points to a data transformation failure, not a model capacity failure. We systematically audited the full prediction pipeline from raw input to decoded output.

**Hypothesis 1: The domain_map activation is collapsing the distribution.**

`PINNWastewaterLoss.domain_map` contained:
```python
def domain_map(self, y_hat):
    return F.softplus(y_hat)
```

The Softplus function is $\log(1 + e^x)$. For negative inputs, it maps values toward zero asymptotically. For large negative inputs, the output approaches zero from above.

The prediction target is **RobustScaled**. By construction, the RobustScaler maps the median to 0 and normalises by IQR. Roughly 50% of all target values are below the median — that is, roughly 50% of all training targets are **negative** in scaled space.

If the model learns to output negative values (as it should, to cover below-median weeks), `domain_map` maps those negatives toward zero. All 7 quantiles are compressed toward the same near-zero floor. The model "learns" to output a tight near-zero bundle because that minimises the Softplus-clipped loss — but the resulting predictions have no spread and are systematically wrong during any above-median event (i.e., any surge).

Confirmed: **Softplus is incompatible with a RobustScaled target.**

**Hypothesis 2: Double-scaling is compressing variance to the noise floor.**

The data processor applies `RobustScaler` to all continuous features and the target. NeuralForecast also exposes a `scaler_type` parameter that, when set to `"robust"`, applies its own internal RobustScaler on top.

The composition is:
1. Processor: $y \rightarrow y' = (y - Q_{50}) / \text{IQR}$ → scaled to IQR ≈ 1.0
2. NeuralForecast: $y' \rightarrow y'' = (y' - Q'_{50}) / \text{IQR}'$

After step 1, the target has IQR ≈ 1.0. NeuralForecast's step 2 then computes IQR of an already-unit-IQR distribution. If the distribution is approximately standard, IQR' ≈ 1.35 (for a normal), and the second scaling further compresses variance by ~0.74×. Repeated transformations like this compound: with slight non-normality in the distribution, this double-scaling can collapse the dynamic range to a fraction of its intended value.

The model then learns quantile predictions in this doubly-compressed space. When predictions are decoded back to the original space, the quantile spread is expanded by both inverse transforms — but if the model's internal spread has collapsed to near-zero (because variance was near-zero during training), no inverse transform can recover meaningful uncertainty.

Confirmed: **`scaler_type="robust"` was double-scaling the already-scaled target.**

**Hypothesis 3: The PINN growth-rate penalty is suppressing surge trajectories.**

`PINNWastewaterLoss` added a regularisation term to the multi-quantile loss:
$$\mathcal{L}_{PINN} = \mathcal{L}_{MQ} + \lambda \cdot \frac{1}{H-1}\sum_{h=1}^{H-1}(\hat{q}_{0.5,h+1} - \hat{q}_{0.5,h})^2$$

With `GROWTH_RATE_LAMBDA = 0.05` (original) or `0.005` (reduced), this term penalises step-changes in the median forecast. Its biological motivation was sound: viral growth at the community level cannot be discontinuous. However, at weekly resolution aggregated over a county population, the observed case data exhibits large week-over-week jumps during variant emergence events.

The penalty was trained on the full dataset including stable inter-wave periods (where it is appropriate) but applied uniformly to all prediction windows including the XBB.1.5 surge (where it is actively harmful). The model learns to predict smooth trajectories because the penalty directly penalises the multi-week acceleration that defines a surge onset.

Confirmed: **The smoothing prior was penalising the signal we most needed to detect.**

**Hypothesis 4: 5 quantiles provide insufficient gradient signal at the tails.**

The original loss used 5 quantile levels: `[0.025, 0.25, 0.50, 0.75, 0.975]`. The gap between `0.025` and `0.25` is 0.225 in quantile space. The pinball loss gradient for the outer quantiles (0.025 and 0.975) is weighted by the level itself — a gradient weight of 0.025 or 0.975 on misses in the tails.

With 5 levels and equal spacing, the tails receive weak gradient signal per batch. The model can reduce the total multi-quantile loss by optimising heavily on the `q0.50` (median, gradient weight 0.5) at the expense of tail coverage. Expanding to 7 levels — adding `0.10` and `0.90` — provides denser gradient coverage of the outer intervals, giving the model more feedback about tail accuracy per training step.

Confirmed: **5-quantile loss had insufficient gradient signal at the tails to learn accurate outer intervals.**

### The Solution

All four failure modes were addressed simultaneously:

| Failure Mode | Fix |
|---|---|
| Softplus collapse | Removed Softplus from `domain_map`; replaced with identity reshape |
| Double-scaling | Changed `scaler_type="identity"` in NeuralForecast |
| Smoothing prior | `GROWTH_RATE_LAMBDA = 0.0` (disabled for Phase 2) |
| Quantile underdispersion | Expanded to 7 quantiles: `[0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]` |

Additionally, a horizon weight vector was introduced:
```python
horizon_weight = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]
```
This gives the near-term WW→cases window (weeks 1–4) 1.5–2× gradient weight, further focusing the loss on the clinically actionable forecast range.

One implementation hazard surfaced immediately: NeuralForecast's `BasePointLoss.__init__` calls `.flatten()` on the `horizon_weight` argument. Python lists have no `.flatten()` method — only numpy arrays do. Passing a Python list raised:
```
AttributeError: 'list' object has no attribute 'flatten'
```
Fix: always convert before passing: `np.array(horizon_weight, dtype=np.float32)`.

### The Insight

**Four independent sufficient conditions for 0% coverage were simultaneously active.** This is the key diagnostic lesson: when a critical metric is maximally wrong (not slightly off — zero), the failure is almost never a single subtle bug. It is the composition of multiple independently broken assumptions that all happen to push in the same direction.

The order of discovery mattered. Softplus was found first because it was the most mechanistically obvious (a non-linear activation applied to data that requires linear passthrough). Double-scaling was found second because it required auditing the full pipeline configuration. The PINN lambda was third — we suspected it but needed to confirm that disabling it did not simply shift the problem. The quantile count was last — it would not have been identifiable as a root cause with the other three still active.

The broader lesson: **probability calibration requires end-to-end invariant maintenance.** Every transformation in the pipeline (scaling, activation, loss weighting) must be consistent with the statistical properties of the target distribution. A single broken link collapses coverage regardless of how correct everything else is.

---

## 3. The WIS vs. Coverage Paradox

### The Problem

During the investigation of the zero-coverage crisis, we encountered a mathematically non-obvious tension in the evaluation metrics: a model can achieve relatively low WIS while simultaneously achieving near-zero coverage, and a model can achieve near-perfect coverage while achieving catastrophically high WIS.

These are not equivalent formulations of "forecast quality." They measure different properties of a probabilistic forecast, and optimising one can actively harm the other. In our context, the initial model had non-trivial WIS but 0% coverage. Understanding why required unpacking the decomposition of WIS.

### The Investigation

**The WIS decomposition:** The Weighted Interval Score decomposes into two components:

$$\text{WIS}(y, \hat{F}) = \underbrace{\frac{1}{K}\sum_{k=1}^{K}(u_k - l_k)}_{\text{Sharpness penalty}} + \underbrace{\frac{2}{\alpha K}\sum_{k=1}^{K}\left[(l_k - y)\mathbf{1}_{y < l_k} + (y - u_k)\mathbf{1}_{y > u_k}\right]}_{\text{Calibration penalty}}$$

- **Sharpness penalty:** Grows with interval width. A model that outputs wide intervals to "be safe" is penalised for being imprecise.
- **Calibration penalty:** Grows with the distance between actuals and the nearest interval bound when actuals fall outside the interval.

Consider two models:

**Model A (overconfident, phase 1 baseline):** Narrow intervals, all below actuals.
- Sharpness penalty: low (narrow intervals)
- Calibration penalty: high (actuals are far above the upper bound)
- WIS: moderate-to-high, driven entirely by calibration misses

**Model B (overcautious):** Extremely wide intervals, covering everything.
- Sharpness penalty: very high (intervals span the entire plausible range)
- Calibration penalty: zero (actuals always inside)
- WIS: high, driven entirely by sharpness

The Phase 1 model was a variant of Model A: its narrow intervals sat below actuals, so the calibration penalty was large, but the sharpness term was small (the model was confident, just confidently wrong). The WIS was non-trivial but not catastrophically high because the calibration penalty was bounded by the actual value distance, which during inter-wave periods was modest.

**Why coverage and WIS diverge:** Coverage is a binary per-observation check: is the actual inside the interval? It does not care how far outside the interval the actual falls — a miss is a miss regardless of magnitude. WIS cares about magnitude: a near-miss contributes less calibration penalty than a far-miss.

During stable inter-wave periods (most of the training window), the model's intervals were near the actuals but still below them. This produced:
- Coverage: 0% (every observation was above the upper bound)
- Calibration WIS penalty: small (the upper bound was close to actuals)
- Net: WIS looked acceptable; coverage was catastrophic

This masking effect is especially dangerous during development: if you monitor WIS as the primary metric, the model appears to be learning something reasonable. Only when you check coverage do you discover that the probabilistic forecast is useless — the intervals exclude all actuals.

**The surge asymmetry:** The XBB.1.5 holdout wave made this worse. During rapid surge periods, actuals were far above the upper bound — the calibration WIS penalty was high, and coverage was 0%. But during the inter-wave period that preceded it (most of the training data), actuals were near the upper bound — WIS was acceptable, coverage was 0%, and the error was invisible.

A model trained to minimise WIS on a dataset dominated by inter-wave periods will learn to be "nearly right but consistently below" — which appears acceptable on WIS but fails completely on coverage.

### The Solution

No single architectural change resolves the WIS/coverage tension; it requires a portfolio of calibration interventions:

**1. Expanding quantile density at the tails (7 vs. 5 quantiles):**
Adding `q0.10` and `q0.90` to the quantile set provides the loss function with two additional gradient signals from the outer intervals. At each training step, 2 more "coverage signals" are back-propagated compared to the 5-quantile setup. This directly increases the penalty for underdispersion at the outer intervals — pushing the model toward wider intervals that actually cover the true distribution.

**2. Removing the Softplus floor:**
The Softplus activation created a hard floor near zero for all quantile outputs. The 97.5th percentile cannot exceed reality if it cannot output values above a certain scale. Removing Softplus removed the artificial upper ceiling on the quantile spread, allowing the model to output genuinely wide intervals when uncertainty is high.

**3. Disabling the growth-rate penalty:**
The PINN smoothing prior penalises multi-step acceleration. Wide intervals require the outer quantiles to diverge rapidly from the median over the forecast horizon — this is precisely the pattern the growth-rate penalty was suppressing. Setting `λ=0.0` removed this interference, allowing the upper quantiles to accelerate away from the median during surge-onset signatures.

**4. Near-term horizon upweighting:**
The `horizon_weight = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]` vector concentrates gradient signal on the 1–4 week horizon. This is where the WIS/coverage tension is most acute — the lead-time window where WW→cases signal is strongest and where clinical decisions are made. Upweighting these steps forces the model to get the near-term interval spread right rather than averaging across the full 8-week horizon.

### The Insight

**WIS and coverage are complementary, not redundant, and can be gamed against each other.** Reporting only WIS creates a false sense of calibration quality — a model can look reasonable on WIS while being completely useless as a probabilistic forecaster. This is not a theoretical concern; it happened in this project.

The design principle that resolves the tension is: **proper scoring rules must be evaluated in the regime where clinical decisions are made.** For an outbreak detection model, the clinically relevant regime is the surge onset — rapid multi-week acceleration from a low baseline. A model should be evaluated primarily on this regime, not on the full distribution including stable inter-wave periods.

This argues for conditional coverage metrics: Coverage_95 conditional on "weeks within N weeks of a surge onset" versus "stable inter-wave weeks." A model that achieves 90% coverage during stable periods but 0% coverage during surge onset is more dangerous than a model with 50% coverage uniformly — because clinicians rely on the probabilistic forecast exactly when the situation is changing rapidly.

We did not implement conditional coverage in Phase 2 (it requires defining "surge onset" in terms of the forecast, which introduces circularity). But it remains the correct evaluation target for Phase 3.

---

## 4. The Temporal and Lead-Time Challenge

### The Problem

Wastewater-based epidemiology is fundamentally a temporal alignment problem. Multiple time series — viral RNA concentrations, clinical case counts, epidemiological week boundaries, laboratory reporting schedules, population-level incubation distributions — must be aligned on a common weekly spine before any model can learn the relationship between them.

Every misalignment is silent. There is no error message for "your WW signal is 4 days off from your case count signal." The model trains, the loss decreases, and the learned transfer function is wrong in a way that is extremely difficult to diagnose from metrics alone.

We encountered three distinct temporal alignment failures in this project.

### The Investigation

**Failure 1: Wednesday-anchored resampling.**

The BAYWA wastewater dataset contains facility-level readings with variable collection days (the most common being Wednesday). The CDC epidemiological cases data uses week-ending dates that anchor to specific weekdays depending on the reporting source.

When building the weekly feature spine, the naive approach is `pd.resample("W")` or `pd.date_range(freq="4W")`. Both of these anchor to **Sunday**. The result: the resampled WW spine has week boundaries on Sundays, while the case data has week boundaries tied to a different anchor day.

When a weekly WW observation (aggregated over Sunday–Saturday) is joined to a weekly cases observation (aggregated over a different seven-day window), the join is off by some number of days. The features do not represent the same seven-day period as the target. The model cannot learn the correct WW→cases transfer function because the features and target are consistently misaligned.

Diagnosis: examining the resulting `ds` column after resampling showed dates landing on Sundays — but the cases spine had no matching Sunday dates. The inner join was silently discarding observations.

Fix: `pd.resample("W-WED")` and `pd.date_range(freq="4W-WED")` — Wednesday-anchored resampling throughout. All week boundaries now align with the majority of WW collection events and the cases dataset spine.

**Failure 2: CV fold boundaries landing on non-observation dates.**

The expanding-window CV uses `pd.date_range` to generate fold cutoffs. The initial cutoff was set to `2022-10-05` (a Wednesday), and the step frequency was `4W` (4-week steps).

`pd.date_range("2022-10-05", periods=5, freq="4W")` generates:
```
[2022-10-09, 2022-11-06, 2022-12-04, 2023-01-01, 2023-01-29]
```

These dates are Sundays — 4 days away from any actual observation. Each fold's evaluation set was being constructed from a cutoff date that had no matching rows in the data spine. The fold's `evaluate()` call received no overlapping observation-forecast pairs and returned all-NaN metrics.

Fix: `pd.date_range("2022-10-05", periods=5, freq="4W-WED")` generates Wednesday-anchored cutoffs that match actual data spine dates. The observation-forecast overlap was restored.

This was the root cause of the "No Overlapping Pairs" `ValueError` encountered in the evaluation code. The deeper fix — having `evaluate()` return a null `EvalResult` rather than raising an exception — was also implemented as a defensive guard, but the correct root cause resolution was the frequency alignment.

**Failure 3: NeuralForecast val_size constraints with short-history counties.**

NeuralForecast enforces the constraint: `val_size ∈ {0} ∪ [h, ∞)`. That is, val_size must be either 0 (no validation) or at least `h` (the forecast horizon, 8 weeks). Any value from 1 to 7 is rejected.

During CV, we compute evaluation externally — we do not use NeuralForecast's internal val/test split, because the CV fold boundaries define our evaluation windows. So we set `val_size=0` to pass all data to the model for training and handle evaluation ourselves.

However, `val_size=0` triggers a secondary constraint: NeuralForecast will not accept `val_size=0` combined with `early_stopping_patience_steps > 0`. Early stopping requires a validation set to monitor. Setting `val_size=0` with early stopping enabled raises:
```
ValueError: val_size=0 requires early_stop_patience_steps=-1
```

For short-history counties (Napa, Solano, Sonoma, Marin, Contra Costa), this constraint becomes binding at early CV cutoffs: if the county has fewer than `INPUT_SIZE + H = 34` rows at a given fold, `val_size=h=8` would require at least 42 rows (34 + 8), which these counties do not have until later cutoffs.

Fix: In `WastewaterTFT.__init__`, `early_stop_patience_steps` is popped from `trainer_kwargs` before model construction. The CV caller then injects `cv_trainer_kwargs` with `early_stop_patience_steps=-1`, which is passed through to the NeuralForecast fit call. This decouples the final-model training configuration (which can use early stopping with `val_size>0`) from the CV configuration (which requires `val_size=0` and early stopping disabled).

**The Rich progress bar nesting failure (infrastructure-level temporal conflict):**

During CV development, the outer loop used a Rich `Progress` context manager to display fold-level progress. PyTorch Lightning's `RichProgressBar` callback also creates an internal Rich `Live` context during model training.

Rich's `Live` context manager uses an internal `_live_stack` to manage nested live displays. When PL's `RichProgressBar` initialises inside an already-active `Progress` context, the `_live_stack` is in an unexpected state. When the inner context exits, it drains the stack — leaving the outer `Progress` context with no corresponding stack entry. The next pop operation on the outer context raises:
```
IndexError: pop from empty list
```

This is a low-level conflict between two independently correct uses of a shared global state object. Neither Rich nor PyTorch Lightning is "wrong" — they simply have incompatible assumptions about stack ownership.

Fix: pass `enable_progress_bar=False, enable_model_summary=False` in `cv_trainer_kwargs`. This prevents PL from creating any Rich context during CV, leaving the outer Rich `Progress` context as the sole owner of the live stack.

### The Solution Summary

| Failure | Root Cause | Fix |
|---|---|---|
| WW/cases misalignment | `"W"` resampling anchors to Sunday | `"W-WED"` throughout |
| CV cutoffs on non-observation dates | `"4W"` anchors to Sunday | `"4W-WED"` in `pd.date_range` |
| Short-history county CV crashes | `val_size=0` incompatible with early stopping | Pop `early_stop_patience_steps` from base kwargs; inject `=-1` via CV kwargs |
| CV progress bar `IndexError` | Rich `_live_stack` conflicts with PL `RichProgressBar` | `enable_progress_bar=False` in `cv_trainer_kwargs` |

### The Insight

**Temporal alignment is a silent, compounding failure mode.** A 4-day misalignment at the weekly aggregation level looks like noise in the data. The model will still train, the loss will still decrease, and nothing will indicate that the features and target represent different seven-day windows.

The diagnostic heuristic we developed: **always verify alignment by looking at the `ds` column of the joined DataFrame and confirming that it contains dates from the expected data spine.** If the join produces dates that do not appear in the cases dataset or the WW dataset individually, the anchor frequency is wrong.

The broader principle: in time series ML, **frequency strings are not interchangeable syntactic sugar.** `"W"`, `"W-WED"`, `"W-MON"` all generate weekly series, but they generate *different* series anchored to different weekdays. Any dataset that has a non-arbitrary anchor day (because of how the data was collected, not how it was formatted) requires an explicit anchor specification. Defaulting to `"W"` (Sunday) when the data is Wednesday-anchored introduces a systematic 4-day lag into every join, which the model will attempt to absorb as noise and fail.

The lead-time structure of wastewater epidemiology — WW leads cases by 1–2 weeks — is a signal that operates at the multi-day resolution. A 4-day anchor misalignment is not small relative to the signal we are trying to learn. It is large enough to destructively interfere with the model's ability to identify the WW→cases lead-time relationship, which is the entire scientific premise of the system.

---

## Conclusion: What the Sum of These Failures Reveals

These four crises were not independent. They formed a specific syndrome characteristic of applying probabilistic deep learning to a new domain under real data constraints:

1. **The target definition crisis** is a domain framing problem — it requires scientific judgement, not engineering skill.
2. **The zero-coverage crisis** is a pipeline invariant problem — it requires auditing every transformation in the prediction chain against the statistical properties of the data.
3. **The WIS/coverage paradox** is an evaluation design problem — it requires choosing metrics that reflect the decision context, not just mathematical convenience.
4. **The temporal alignment crisis** is a data contract problem — it requires treating every frequency string, join key, and resample anchor as a domain-specific choice with measurable consequences.

The common thread: **probabilistic forecasting systems fail silently along dimensions that summary metrics do not observe.** WIS looked acceptable during the zero-coverage crisis. Training loss was decreasing during the temporal misalignment. Syntax was valid during the target mismatch.

The lesson for building systems in this class: construct a validation protocol that goes beyond aggregate metrics. Check the alignment of the data spine before training. Check the distribution of the scaled target before defining the activation. Check the actual vs. forecast traces visually during the surge periods, not just over the full evaluation window. And treat 0% coverage as a red flag requiring forensic investigation of the full pipeline, not a tuning problem to be resolved by adjusting learning rate.
