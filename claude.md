# Sewer Signals — Active Development Context (claude.md)

> **Operational context.** This file tracks the current development phase,
> active problems, and what was last changed. Update at the start of each
> session and whenever phase changes. For stable architecture see `memory.md`.

---

## Current Phase: Phase 2 — "Momentum Pivot" (In Progress)

**Objective:** Move from a "Smoothing Engine" to an "Outbreak Detection" model.
The model must capture the suddenness and volatility of clinical case surges,
not just the smooth inter-wave baseline.

**Last verified run:** `python main.py --skip-cv --no-dash` completed exit 0
(2000 steps, MPS GPU, 4.4M parameter TFT). Holdout evaluation metrics not yet
captured (run output was truncated). Full evaluation run is the immediate next step.

---

## The Core Problem: "Zero Coverage" (The Overconfident Smoother)

### Symptom
- Holdout PI coverage: **0% at both 50% and 95% intervals**
- The model's upper quantile (0.975) is consistently *below* actuals
- Predictions are smooth, linear trajectories; reality is "spiky" week-over-week surges

### Root Causes Identified and Addressed

| Root Cause | Fix Applied | Status |
|---|---|---|
| Softplus in `domain_map` collapsed RobustScaled negatives to zero floor | Removed Softplus; `domain_map` is now identity reshape | ✅ Done |
| `scaler_type="robust"` double-compressed variance | Changed to `scaler_type="identity"` | ✅ Done |
| `GROWTH_RATE_LAMBDA=0.05` penalised surge trajectories | Reduced to 0.005, then **disabled (0.0)** for Phase 2 | ✅ Done |
| Model lacked vocabulary for rate-of-change | Added 5 derivative/momentum features | ✅ Done (Phase 2) |
| Quantile loss underdispersed (5 quantiles, equal horizon weighting) | Expanded to 7 quantiles + near-term horizon upweighting | ✅ Done (Phase 2) |

### What Has NOT Been Verified Yet
- Whether the Phase 2 changes actually improve holdout coverage (awaiting full run)
- CV fold metrics with the new 15-feature set

---

## Phase 2 Changes — What Was Done This Session

### Task A: Derivative Expansion (5 new HIST_COVARIATES)
Added to `processor._add_lag_features()` and `_apply_scaling()` candidates:

| Feature | What it captures |
|---|---|
| `diff_concentration` | Absolute weekly velocity (Δ log1p_conc); distinct from relative `growth_rate_1w` |
| `log1p_concentration_2w_ma` | 2-week rolling mean (short baseline for deviation detection) |
| `log1p_concentration_4w_ma` | 4-week rolling mean (medium baseline) |
| `log1p_concentration_2w_std` | Local volatility — spikes during erratic onset phase |
| `log1p_concentration_4w_std` | Medium volatility window |

Total HIST_COVARIATES: **15** (was 10).

### Task B: Uncertainty Calibration
Changes to `src/config.py` and `src/models/tft_model.py`:

| Change | Before | After | File |
|---|---|---|---|
| PINN lambda | 0.005 | **0.0** (disabled) | `config.py` |
| n_quantiles | 5 | **7** | `config.py` |
| quantile_levels | [.025,.25,.5,.75,.975] | **[.025,.10,.25,.5,.75,.90,.975]** | `config.py` |
| horizon_weight | None | **[2,2,1.5,1.5,1,1,.8,.8]** | `config.py` + `tft_model.py` |

`horizon_weight` must be passed as `np.array(..., dtype=np.float32)` — NeuralForecast
calls `.flatten()` on it internally; a plain Python list raises `AttributeError`.

### Task C: "No Overlapping Pairs" Fix
Already done in a prior session. `evaluate()` in `src/evaluation/metrics.py`
returns a null `EvalResult` (all metrics NaN, `n_observations=0`) when forecast
and actuals share no overlapping date pairs. No `ValueError` is raised.

### Dashboard: Operational Clarity Refactor
`src/visualization/attention_plots.py` and `dashboard.py` — all visual changes:

| Element | Before | After |
|---|---|---|
| Actuals colour | `#212121` (near-black, invisible on dark bg) | `#F5F5F5` (near-white) |
| Forecast colour | `#9C27B0` (purple) | `#F97316` (orange) |
| PI bands | purple rgba | orange rgba (30%/12%) |
| Sludge track | `#2196F3` blue | `#38BDF8` sky blue |
| Liquid track | `#FF9800` orange | `#C084FC` violet (avoids clash with forecast) |
| Decay-rate axis | unlabelled | amber `#FDE68A` title + ticks |
| Attention heatmap | Viridis | Plasma (stronger cold→hot contrast) |
| Title font | 16px | 18px |
| Axis label font | 13px | 14px |
| Case momentum (VSN) | "Lag" category | "Case Lag" (orange = matches forecast) |

Dashboard changes are **not visible** in a currently running server — restart required.

---

## Known Bugs / Watchpoints

### 1. Liquid-track NaN scaler warning
When liquid-track is processed, `TARGET_COL` (`log1p_new_cases`) is all-NaN.
`RobustScaler` raises `RuntimeWarning: All-NaN slice encountered`. This is expected
and harmless — the scaler skips those columns. The target lag features
(`log1p_new_cases_lag1w/2w/3w`) are not computed on the liquid track (guarded by
`if TARGET_COL in df.columns and df[TARGET_COL].notna().any()`).

### 2. Warmup NaN warnings at pre-flight validation
`_to_nf_format` drops rows with NaN hist_exog columns. The pre-flight checker
in `main.py` logs ~25 `[INV-NAN]` warnings for warmup NaN rows (first 1–3 rows
per county per lag/diff feature). This is expected and handled automatically.

### 3. Short-history counties (Napa, Solano, Sonoma, Marin, Contra Costa)
5 counties have fewer than `INPUT_SIZE + H = 34` training weeks.
`start_padding_enabled=True` zero-pads them. These counties produce valid
predictions but with lower reliability — VSN will assign low weight to their
sparse early-history context window.

### 4. CV models use `val_size=0` + `early_stop_patience_steps=-1`
In `expanding_window_cv`, the model is fitted with `val_size=0` to avoid
NeuralForecast's constraint `val_size ∈ {0} ∪ [h, ∞)`. Early stopping is
disabled (via `cv_trainer_kwargs` passed through `WastewaterTFT`) so that
`val_size=0` is accepted. CV evaluation is done externally by `evaluate()`.

### 5. Rich progress bar nesting conflict
If `enable_progress_bar=True` is set in CV trainer kwargs, PyTorch Lightning's
`RichProgressBar` nests inside the outer Rich `Progress` context, draining the
`_live_stack` and causing `IndexError: pop from empty list`. Always pass
`enable_progress_bar=False, enable_model_summary=False` in `cv_trainer_kwargs`.

---

## Immediate Next Steps

1. **Run full evaluation:** `python main.py --skip-cv --no-dash`
   Capture holdout `coverage_50`, `coverage_95`, `mean_wis`, `smape`.
   Target: coverage_95 > 50% (any non-zero improvement from 0%).

2. **If coverage still 0%:** Inspect whether the issue is in `_build_decoded_forecast`
   — the inverse_transform + expm1 + clip chain may be collapsing the quantile spread.
   Debug by logging raw (scaled) quantile spread vs. decoded spread.

3. **Run full CV:** `python main.py --no-dash`
   Verify all 5 folds complete without crashes and produce non-NaN WIS.

4. **VSN interpretability check:** After a successful training run, verify that
   `diff_concentration` and the `log1p_new_cases_lag*` features appear in the
   VSN attention weights with non-trivial importance (> random 1/15 ≈ 6.7%).

5. **Dashboard end-to-end test:** Run `python main.py` (with dash) and verify
   the new orange forecast / near-white actuals palette is correct in browser.

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

## File Change Log (this session and prior sessions)

| File | What changed |
|---|---|
| `src/config.py` | TARGET_COL/WW_FEATURE_COL pivot; overlap dates; W-WED splits; 7 quantiles; λ=0.0; horizon_weight; scaler_type=identity |
| `src/data_pipeline/processor.py` | Cases merge (Stage 12); W-WED resample; lag features for WW+target; derivative expansion (5 features); scaler candidates updated |
| `src/models/tft_model.py` | HIST_COVARIATES 10→15; horizon_weight wired to loss; early_stop popped from trainer_kwargs for CV |
| `src/models/loss_functions.py` | Softplus removed from domain_map; GROWTH_RATE_LAMBDA inline docs updated |
| `src/evaluation/metrics.py` | evaluate() no-overlap guard (null EvalResult); CV freq="4W-WED"; val_size=0 |
| `src/visualization/attention_plots.py` | Full colour palette refactor; Task C fonts/legend; Plasma heatmap; two-track axis colour-coding; VSN category map updated |
| `src/visualization/dashboard.py` | Accent colour updated to match forecast orange |
| `src/utils/helpers.py` | LLM prompt updated for new target (cases, not WW concentration) |
| `main.py` | cv_trainer_kwargs (progress bar off, early stop -1); _split_raw overlap window; cases CSV loading; _run_cv wiring |
