# Sewer Signals — Active Development Context (claude.md)

> **Operational context.** This file tracks the current development phase,
> active problems, and what was last changed. Update at the start of each
> session and whenever phase changes. For stable architecture see `memory.md`.

---

## Current Phase: Phase 4 — "Heuristics to Dynamics" (Starting)

**Objective:** Replace the three hard-coded "magic numbers" that were validated
in the Phase 3 PoC with dynamic, statistically-grounded computations. The
goal is an auto-calibrating model that does not require manual threshold tuning
when the data distribution shifts (new waves, new counties, new variants).

**Phase 3 Status:** ✅ Complete — PoC breakthrough confirmed on 3-county validation
(San Francisco, San Mateo, Santa Clara). Hard-coded heuristics broke the
"Smoothing Engine" and achieved non-zero PI coverage. Values are now the
baseline to replace.

---

## The PoC Baseline (Phase 3 Hard-Coded Heuristics)

These are the values that worked in Phase 3. Phase 4 must replace each one
with a dynamic equivalent that achieves the same or better performance.

| Constant | Value | Purpose | Phase 4 Replacement Target |
|---|---|---|---|
| `UNDERDISPERSION_LAMBDA` | `0.5` | Makes underdispersion penalty competitive with Pinball loss | **Scale-Aware Lambda**: `λ = k × mean_pinball_loss` |
| `MIN_PI_WIDTH` | `2.5` | Minimum 95% PI width in scaled units (~86% of theoretical Gaussian width) | **Volatility-Adjusted Width**: `w = m × rolling_std(signal, 4w)` |
| `OUTBREAK_GROWTH_THRESHOLD` | `0.25` | 25% WoW increase flags onset (Phase 3: lowered from 0.40 for AUC) | **Z-score / Percentile Trigger**: `z > Z_THRESHOLD` or `pct > P_THRESHOLD` |

---

## Phase 4 Refactoring Targets

### Target 1: Scale-Aware Lambda
**Current:** `UNDERDISPERSION_LAMBDA = 0.5` (static)
**Goal:** `λ_effective = k × mean_pinball_loss` where `k` is a dimensionless ratio
(target: penalty ≈ 50% of base loss magnitude at initialisation).
**Why static fails:** As Pinball loss drops during training, the underdispersion
penalty increasingly dominates, eventually fighting the calibration it was meant
to help.
**Files to change:** `src/models/loss_functions.py`, `src/config.py`

### Target 2: Volatility-Adjusted Minimum Width
**Current:** `MIN_PI_WIDTH = 2.5` (static, in scaled units)
**Goal:** `min_width_t = multiplier × σ_t` where σ_t is the 4-week rolling std
of `log1p_concentration` for that county-week.
**Why static fails:** A flat floor is too wide for low-variance periods (over-
penalises) and too narrow for high-variance surge onsets (under-penalises).
**Files to change:** `src/models/loss_functions.py`, `src/data_pipeline/processor.py`
(add `rolling_std` as a passed-in covariate or compute inline in loss)

### Target 3: Statistical Significance Outbreak Trigger
**Current:** `OUTBREAK_GROWTH_THRESHOLD = 0.25` (static 25% WoW)
**Goal:** Z-score relative to rolling baseline: `z = (x_t - μ_baseline) / σ_baseline`
where baseline = 8-week rolling mean/std. Alert when `z > Z_THRESHOLD`.
**Why static fails:** 25% growth is noise in a high-variance wave but significant
in a quiet inter-wave period — same threshold, opposite meaning.
**Files to change:** `src/evaluation/metrics.py`, `src/config.py`

---

## Phase 3 Summary (Completed)

### What Phase 3 Did
Phase 3 broke the "Smoothing Engine" by introducing an underdispersion penalty
and two new derivative features. Validated on 3-county set.

| Change | Before | After | File |
|---|---|---|---|
| Underdispersion penalty | None | `UNDERDISPERSION_LAMBDA=0.5` | `config.py` + `loss_functions.py` |
| Minimum PI width | None | `MIN_PI_WIDTH=2.5` | `config.py` + `loss_functions.py` |
| Outbreak threshold | 0.40 | **0.25** (higher AUC sensitivity) | `config.py` |
| `accel_concentration` feature | Not present | 2nd derivative of log1p_concentration | `processor.py` + `tft_model.py` |
| `vel_concentration_lag1w` | Not present | velocity 1 week ago | `processor.py` + `tft_model.py` |
| HIST_COVARIATES count | 15 | **17** | `tft_model.py` |
| Data end date | 2023-05-10 | **2023-12-19** (full CA dataset) | `config.py` |
| Holdout window | 15 W (XBB peak only) | **28 W** (2023-06-08 → 2023-12-19) | `config.py` |
| Validation set | 1-county (Santa Clara) | **3-county** (SF, San Mateo, SCC) | `config.py` |

### Phase 3 Confirmed Working
- Non-zero PI coverage on 3-county holdout ✅
- `accel_concentration` and `vel_concentration_lag1w` in VSN with non-trivial weights ✅
- Loss function compiles and trains without NaN ✅

---

## Current Architecture Snapshot

### HIST_COVARIATES (17 features)

| Group | Feature | Phase Added |
|---|---|---|
| WW signal | `log1p_concentration` | 1 |
| WW lags | `log1p_concentration_lag1w/2w/3w` | 1 |
| Case momentum | `log1p_new_cases_lag1w/2w/3w` | 2 |
| Slope/phase | `growth_rate_1w`, `relative_decay_rate`, `outlier_flag_int` | 1/2 |
| Velocity | `vel_concentration` | 2 |
| Acceleration | `accel_concentration` | 3 |
| Momentum context | `vel_concentration_lag1w` | 3 |
| Rolling baseline | `log1p_concentration_2w_ma`, `log1p_concentration_4w_ma` | 2 |
| Local volatility | `log1p_concentration_2w_std`, `log1p_concentration_4w_std` | 2 |

### Key Config Values (current state entering Phase 4)

| Parameter | Value |
|---|---|
| `UNDERDISPERSION_LAMBDA` | 0.5 (static — Phase 4 target) |
| `MIN_PI_WIDTH` | 2.5 (static — Phase 4 target) |
| `OUTBREAK_GROWTH_THRESHOLD` | 0.25 (static — Phase 4 target) |
| `GROWTH_RATE_LAMBDA` | 0.0 (PINN disabled) |
| `n_quantiles` | 7 |
| `quantile_levels` | [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] |
| `horizon_weight` | [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8] |
| `scaler_type` | "identity" |
| Holdout window | 2023-06-08 → 2023-12-19 (28 W) |

---

## Known Bugs / Watchpoints

### 1. Liquid-track NaN scaler warning
When liquid-track is processed, `TARGET_COL` (`log1p_new_cases`) is all-NaN.
`RobustScaler` raises `RuntimeWarning: All-NaN slice encountered`. Expected and
harmless — the scaler skips those columns.

### 2. Warmup NaN warnings at pre-flight validation
`_to_nf_format` drops rows with NaN hist_exog columns. The pre-flight checker
in `main.py` logs ~25 `[INV-NAN]` warnings for warmup NaN rows (first 1–3 rows
per county per lag/diff feature). Expected and handled automatically.

### 3. Short-history counties (Napa, Solano, Sonoma, Marin, Contra Costa)
5 counties have fewer than `INPUT_SIZE + H = 34` training weeks in early folds.
`start_padding_enabled=True` zero-pads them.

### 4. CV models use `val_size=0` + `early_stop_patience_steps=-1`
`expanding_window_cv` fits with `val_size=0` to satisfy NeuralForecast's
`val_size ∈ {0} ∪ [h, ∞)` constraint. Early stopping disabled via
`cv_trainer_kwargs`. CV evaluation done externally by `evaluate()`.

### 5. Rich progress bar nesting conflict
If `enable_progress_bar=True` in CV trainer kwargs, PyTorch Lightning's
`RichProgressBar` nests inside the outer Rich `Progress` context, causing
`IndexError: pop from empty list`. Always pass
`enable_progress_bar=False, enable_model_summary=False` in `cv_trainer_kwargs`.

### 6. `horizon_weight` must be np.array dtype float32
NeuralForecast calls `.flatten()` on `horizon_weight` internally.
A plain Python list raises `AttributeError`. Always pass
`np.array([...], dtype=np.float32)`.

---

## Immediate Next Steps (Phase 4)

1. **Analysis — understand heuristic coupling:** Before replacing any constant,
   run `python main.py --skip-cv --no-dash` to get a clean baseline metric
   snapshot (coverage_50, coverage_95, mean_wis, smape, AUC).

2. **Target 1 — Scale-Aware Lambda:** Implement `λ_effective` as a function of
   running mean Pinball loss in `PINNWastewaterLoss.forward()`. Add
   `UNDERDISPERSION_K` ratio constant to `config.py`.

3. **Target 2 — Volatility-Adjusted Width:** Compute `σ_t` from
   `log1p_concentration_4w_std` (already in HIST_COVARIATES). Pass it into
   the loss via the batch tensor; replace scalar `MIN_PI_WIDTH` with
   per-sample `min_width_t = multiplier × σ_t`.

4. **Target 3 — Z-score Trigger:** Replace `OutbreakDetector` growth threshold
   with a rolling z-score in `metrics.py`. Add `Z_OUTBREAK_THRESHOLD` to
   `config.py`; deprecate `OUTBREAK_GROWTH_THRESHOLD`.

5. **Regression check:** After each target, re-run evaluation and confirm
   coverage_95 ≥ Phase 3 baseline.

---

## CLI Reference

```bash
# Full run (CV + final model + dashboard)
python main.py

# Skip CV, skip dashboard (fastest end-to-end check)
python main.py --skip-cv --no-dash

# Fast mode (reduced steps for smoke test)
python main.py --fast --no-dash

# Custom steps
python main.py --max-steps 500 --no-dash
```

---

## Notebook Plotting Style Guide

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

**Axes:** always labeled with units — `ax.set_xlabel("Date (W-WED)", fontsize=10)` / `ax.set_ylabel("Copies/g (log)", fontsize=10)`

**Legend:** `ax.legend(fontsize=9, loc="upper right")` — inside plot, top-right corner

**Wave context bands:** `ax.axvspan(pd.Timestamp(ws), pd.Timestamp(we), alpha=0.07, color=wc, zorder=0)` using `ALL_WAVE_SPANS`

**Dual-axis plots:** `ax.twinx()` — WW (`C_WW`) on left axis, cases (`C_CASES`) on right axis

**Close every figure:** `plt.tight_layout()` then `plt.show()`

---

## File Change Log

| File | Last significant change |
|---|---|
| `src/config.py` | Phase 3: UNDERDISPERSION_LAMBDA=0.5, MIN_PI_WIDTH=2.5, OUTBREAK_GROWTH_THRESHOLD=0.25, data dates extended to 2023-12-19 |
| `src/data_pipeline/processor.py` | Phase 3: accel_concentration + vel_concentration_lag1w added |
| `src/models/tft_model.py` | Phase 3: HIST_COVARIATES 15→17 (accel_concentration, vel_concentration_lag1w) |
| `src/models/loss_functions.py` | **Phase 4**: cumulative-softplus domain_map (monotonicity); fallback floor restored to MIN_PI_WIDTH=2.5 |
| `src/data_pipeline/processor.py` | **Phase 4**: save_scalers() / load_scalers() — scaler persistence to disk |
| `main.py` | **Phase 4**: proc.save_scalers() after processing; _invert_scaling_to_log1p auto-loads; all exports in unscaled log1p |
| `src/evaluation/metrics.py` | Phase 3: OUTBREAK_GROWTH_THRESHOLD=0.25; no-overlap guard |
| `src/visualization/attention_plots.py` | Phase 2: Full colour palette refactor |
| `src/visualization/dashboard.py` | Phase 2: Accent colour updated |
| `src/utils/helpers.py` | Phase 2: LLM prompt updated for case target |
