# Sewer Signals — Active Development Context (claude.md)

> **Operational context.** This file tracks the current development phase,
> active problems, and what was last changed. Update at the start of each
> session and whenever phase changes. For stable architecture see `memory.md`.

---

## Current Phase: Phase 5 — "Reliability & Detection" (Ongoing)

**Objective:** Verify that all Phase 4 dynamic mechanisms are actually live at
training time, then re-enable the `GROWTH_RATE_LAMBDA` growth-rate penalty to
restore biological plausibility without flattening PI spread.

**Phase 4 Status:** ✅ Complete — All three dynamic replacements implemented in
code. Median-anchored cumulative softplus domain_map provides structural
monotonicity. Key open question: whether NeuralForecast actually passes
`y_insample` to the loss (which activates the dynamic min-width and
step-change cap paths). The first training run will log INFO or WARNING to
resolve this (see Phase 5 Step 1 below).

---

## Phase 4 Summary (Completed)

### What Phase 4 Built

| Change | Before (Phase 3) | After (Phase 4) | Where |
|---|---|---|---|
| `domain_map` | Identity reshape (Softplus removed in P2) | **Median-anchored cumulative softplus** — Q[0.50] is unconstrained anchor; lower/upper quantiles built via softplus increments; monotonicity guaranteed | `loss_functions.py` |
| Underdispersion lambda | `UNDERDISPERSION_LAMBDA = 0.5` (static) | `effective_lambda = UNDERDISPERSION_K × pinball_loss` — scales with loss magnitude so penalty stays proportional throughout training | `config.py` + `loss_functions.py` |
| Minimum PI width | `MIN_PI_WIDTH = 2.5` (static, scaled units) | `min_width_t = MIN_PI_WIDTH_MULTIPLIER × σ(y_insample[-4:])` clamped to `MIN_PI_WIDTH_FLOOR`; falls back to 2.5 if `y_insample` absent | `loss_functions.py` |
| Outbreak trigger | `OUTBREAK_GROWTH_THRESHOLD = 0.25` (static WoW) | `OutbreakDetector` defaults to Z-score mode: `z = (x_t − μ_8w) / σ_8w ≥ Z_OUTBREAK_THRESHOLD` | `metrics.py` + `config.py` |
| Step-change cap | Relative-rate formula (pathological near zero) | `dyn_cap = STEP_CHANGE_MULTIPLIER × σ(y_insample[-4:])` clamped to `MAX_WEEKLY_STEP_CHANGE`; **wired but inactive** (`GROWTH_RATE_LAMBDA = 0.0`) | `loss_functions.py` |

### Phase 4 Key Open Question
Whether `y_insample` is passed by NeuralForecast to `PINNWastewaterLoss.__call__`.
On the first training forward pass the loss logs:
- `INFO "y_insample RECEIVED"` → dynamic min-width and step-change cap are live
- `WARNING "y_insample ABSENT"` → static Phase 3 fallbacks (`MIN_PI_WIDTH = 2.5`) are in use

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

### HIST_COVARIATES (18 features)

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
| Biological gravity | `ww_case_ratio` | 5 |

### Key Config Values (Phase 5 — post RC-1/RC-2 fixes)

| Parameter | Value | Status |
|---|---|---|
| `UNDERDISPERSION_K` | 0.5 | **Phase 4 active** — effective_lambda = clamp(K × pinball_loss, min=LAMBDA) |
| `UNDERDISPERSION_LAMBDA` | 0.5 | **Phase 5 floor** — guarantees penalty ≥ 0.5 at convergence (was Phase 3 legacy/unused) |
| `MIN_PI_WIDTH_MULTIPLIER` | **3.0** | **Phase 5** — raised from 2.0; activates above Phase 3 floor at σ>0.83 (was σ>1.25) |
| `MIN_PI_WIDTH_FLOOR` | **1.5** | **Phase 5** — raised from 0.5; last-resort safety net (primary floor = MIN_PI_WIDTH) |
| `MIN_PI_WIDTH` | 2.5 | Static fallback when y_insample absent — not used as dynamic floor |
| `GROWTH_RATE_LAMBDA` | **0.005** | **Phase 5 active** — asymmetric sigmoid gate (upward-only, decays at outbreak scale) |
| `STEP_CHANGE_MULTIPLIER` | 3.0 | Wired, inactive while λ=0 |
| `MAX_WEEKLY_STEP_CHANGE` | 1.5 | Static floor / fallback for step-change cap |
| `Z_OUTBREAK_THRESHOLD` | 2.0 | **Phase 4 active** — Z-score onset trigger |
| `Z_SCORE_BASELINE_WEEKS` | 8 | **Phase 4 active** — rolling baseline window |
| `OUTBREAK_GROWTH_THRESHOLD` | 0.25 | Phase 3 legacy — used only when z_threshold=None |
| `n_quantiles` | 7 | unchanged |
| `quantile_levels` | [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975] | unchanged |
| `horizon_weight` | [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8] | unchanged |
| `scaler_type` | "identity" | unchanged |
| Holdout window | 2023-06-08 → 2023-12-19 (28 W) | unchanged |

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

### 3. Short-history counties (Sonoma, Marin, Contra Costa)
Napa and Solano are now **explicitly excluded** via `EXCLUDE_FIPS` in `config.py` —
they had 2 and 3 training weeks respectively, near-zero IQR scalers (0.089–0.146),
and were already being dropped silently by the NaN filter. They are not on the dashboard.
The remaining 3 short-history counties (Sonoma=15w, Marin=16w, Contra Costa=29w)
have fewer than `INPUT_SIZE + H = 34` training weeks in early folds.
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

## Immediate Next Steps (Phase 5)

### Completed
- ✅ **y_insample liveness confirmed:** `INFO y_insample RECEIVED` shape=(7,26,1).
  Dynamic min-width and step-change cap are live.
- ✅ **Phase 5 baseline captured** (run_003): coverage_95=3.6%, coverage_50=0%,
  WIS=0.723, SMAPE=0.946. Root causes identified: K-ratio penalty fading at
  convergence (RC-1) and dynamic min_width floor below Phase 3 in calm periods (RC-2).
- ✅ **RC-1 fixed:** `effective_lambda = clamp(K × pinball, min=UNDERDISPERSION_LAMBDA)`
  — penalty floor restored to Phase 3 strength (0.5) at convergence.
- ✅ **RC-2 fixed:** `_dynamic_min_width` now uses a two-tier vol approach — `local_vol`
  (last 4 steps, reactive) and `global_vol` (full insample, stable baseline per series).
  Floor = `multiplier × global_vol` clamped to `MIN_PI_WIDTH_FLOOR` (absolute last resort).
  From run_003 data: well-behaved counties have global_std≈0.62–0.72 in scaled space →
  floor≈1.85–2.15 per series, data-driven. `global_vol` capped at 2.0 to guard sparse counties.
  `MIN_PI_WIDTH_MULTIPLIER` raised 2.0→3.0. `MIN_PI_WIDTH_FLOOR` raised 0.5→1.5.

### Phase 5 Latest (post-run_004)
- **coverage_95 = 46.4%** (up from 3.6%) — RC-1/RC-2 confirmed working
- **coverage_50 = 0.0%** — median displacement; requires RC-3 (dropout fix)
- **5 false positive alerts** — GROWTH_RATE_LAMBDA still 0.0 at that point

### Changes Implemented
- `GROWTH_RATE_LAMBDA = 0.005` re-enabled with **asymmetric sigmoid gate** (upward-only, decays at outbreak scale)
- `ww_case_ratio = log1p_concentration − log1p_new_cases` added as HIST_COVARIATE 18
- Dropout raised 0.1 → 0.3, attn_dropout 0.1 → 0.3 (RC-3: overfitting)
- `run_outbreak_validation()` + `--outbreak-validation` CLI flag added

### Next
1. **Run `--skip-cv --no-dash`** — confirm coverage_95 holds ≥ 40% and false positives drop.
2. **Run `--outbreak-validation`** — verify sensitivity > 0 on Windows 2–4.
3. If coverage_50 still 0%: increase `UNDERDISPERSION_K` or tune dropout further.

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
| `src/config.py` | **Phase 5**: EXCLUDE_FIPS=[06055,06095] (Napa+Solano); MIN_PI_WIDTH_MULTIPLIER 2.0→3.0; MIN_PI_WIDTH_FLOOR 0.5→1.5 |
| `main.py` | **Phase 5**: EXCLUDE_FIPS filter applied after raw load; snapshot phase label Phase 4→Phase 5 |
| `src/models/loss_functions.py` | **Phase 5**: effective_lambda floor (RC-1); two-tier global/local vol floor (RC-2); asymmetric sigmoid growth gate (GROWTH_RATE_LAMBDA=0.005) |
| `src/data_pipeline/processor.py` | **Phase 5**: ww_case_ratio added to _add_lag_features() + _apply_scaling candidates |
| `src/models/tft_model.py` | **Phase 5**: ww_case_ratio added to HIST_COVARIATES (17→18) |
| `src/evaluation/metrics.py` | **Phase 5**: OutbreakWindowResult dataclass + run_outbreak_validation() |
| `src/evaluation/metrics.py` | **Phase 4**: OutbreakDetector defaults to Z-score mode; LeadTimeEvaluator Z-score scoring; evaluate() opts into Z_SCORE_BASELINE_WEEKS |
| `src/data_pipeline/processor.py` | **Phase 4**: save_scalers() / load_scalers() — scaler persistence to disk |
| `main.py` | **Phase 4**: proc.save_scalers() after processing; _invert_scaling_to_log1p auto-loads; all exports in unscaled log1p |
| `src/models/tft_model.py` | Phase 3: HIST_COVARIATES 15→17 (accel_concentration, vel_concentration_lag1w) |
| `src/visualization/attention_plots.py` | Phase 2: Full colour palette refactor |
| `src/visualization/dashboard.py` | Phase 2: Accent colour updated |
| `src/utils/helpers.py` | Phase 2: LLM prompt updated for case target |
