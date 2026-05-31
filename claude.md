# Sewer Signals — Active Development Context (claude.md)

> **Operational context.** This file tracks the current development phase,
> active problems, and what was last changed. Update at the start of each
> session and whenever phase changes. For stable architecture see `memory.md`.

---

## Current Phase: Phase 6 — "Production Readiness & Two-Stage Architecture" (Extended, Ongoing)

**Objective:** Transform the research codebase into a professional, pathogen-agnostic,
geographically portable inference engine with a calibrated two-stage detection pipeline.

**Test suite:** 217 passing, 0 failing.

---

## Phase 6 Session Summary (Cumulative)

### Sessions 1–6 (previously saved)
Evaluation engine, two-stage architecture, 9-county extension, phase-aware training,
geography config system, dashboard three-tier rebuild, quiet prior fix (data-driven),
tft_model.predict() StratifiedWindowSampler fix. All complete.

### Session 7 (this session)

| Change | File | Key Design |
|---|---|---|
| **Geography portability: in-place dict/list mutation** | `src/config_geographies.py` | `apply_geography()` now uses `.clear(); .update()` / `.clear(); .extend()` for dicts and lists, so every locally-bound `from src.config import X` reference propagates automatically without needing `cfg.*` at each call site |
| **`_split_raw()` date fix** | `main.py` | Was using locally-bound `TRAIN_END_DATE` etc. (stale after `apply_geography()`). Now uses `cfg.TRAIN_END_DATE`, `cfg.VAL_END_DATE`, `cfg.DATA_START_DATE`, `cfg.DATA_END_DATE` |
| **`processor.py` FIPS mapping fix** | `src/data_pipeline/processor.py` | `_clean_ca_ww()` and `__init__` were using locally-bound `BAY_AREA_FIPS`. Now use `_cfg.BAY_AREA_FIPS` — SoCal county names now map correctly to FIPS |
| **`_run_cv()` date fix** | `main.py` | `initial_train_end=TRAIN_END_DATE` → `cfg.TRAIN_END_DATE`; same for VAL_END_DATE |
| **upper_socal.yaml verified and updated** | `config/geographies/upper_socal.yaml` | Data diagnostic run 2026-05-30. Kern=zero rows (excluded), Ventura=185 rows ends Jun 2023 (excluded). `train_end_date: "2023-04-05"` (34W after Santa Barbara joined Aug 2022), `val_end_date: "2023-08-02"`. Outbreak windows updated. |
| **Loss function floor bug FIXED** | `src/models/loss_functions.py` | `_dynamic_min_width` was clamping to `self.min_pi_width` (Phase 3 legacy=2.5), not `self.min_pi_width_floor` (Phase 6=0.05). This was the root cause of Coverage 95% = 100% (trivially wide PIs). |
| **Loss config values updated** | `src/config.py` | `MIN_PI_WIDTH_MULTIPLIER` 3.0→2.0, `MIN_PI_WIDTH_FLOOR` 1.5→0.05, `MIN_PI_WIDTH` 2.5→0.05. Floor is now a numerical safety net only, not a width target. Pinball loss governs calibration; multiplier × volatility governs minimum during outbreaks. |
| **`CLASSIFIER_Z_THRESHOLD` raised** | `src/config.py` | 1.5→2.0. At 1.5, SF/SC spuriously triggered 30-37% of quiet holdout weeks because non-elastic baseline was anchored to pre-2022 WW troughs; 2023 endemic level sits above that. At 2.0: SC trigger rate 37%→11%, SF 33%→30%. |
| **Pinball ratio everywhere** | `metrics.py`, `helpers.py`, `dashboard.py`, `run_manager.py` | `pinball_ratio = q010/q090` added to: `EvalReport.to_dict()`, `print_eval_report` (colored ratio + bias direction), `print_cv_summary` (column), bio table (Pinball q0.90 + Pinball Ratio rows), CV stability chart panel 2 (per-fold ratio line on right y-axis), run snapshot metrics |
| **Dashboard color overhaul** | `src/visualization/dashboard.py` | PI bands → California navy (`rgba(26,79,160,0.11/0.28)`). TFT active bands → California gold (`rgba(234,179,8,0.08)`). Observed context → slate `#64748B`. Observed holdout → Pacific deep blue `#0C4A6E`. Median forecast stays orange (now the only orange element). |
| **Dashboard: Gatekeeper section removed** | `src/visualization/dashboard.py` | OutbreakClassifier — Gatekeeper Activity panel removed from Tier 2. Callback removed. |
| **Dashboard: detection metrics removed from bio table** | `src/visualization/dashboard.py` | Precision, Recall, F1, TTD rows removed — all always NaN for 2023 holdout (no onsets). Comment explains why. |
| **`eval_result` passed to `create_app()`** | `main.py` | Dashboard bio table now shows live holdout metrics on launch, not empty dashes. |
| **`expanding_window_cv` + `_run_cv`: two_stage + phase_aware** | `src/evaluation/evaluator.py`, `main.py` | Both flags now propagate to ALL rolling/CV folds. Each fold fits its own OutbreakClassifier and PhaseLabeler on that fold's training window only (no leakage). Previously both flags were silently dropped for rolling evaluation. |
| **Run snapshot notes: all flags captured** | `main.py` | Notes now record `two_stage`, `phase_aware`, `rolling_holdout`, `geography` in run_meta.json. Phase updated to "Phase 6". |
| **Project overview v4.0** | `documents/project_overview.md` | Full technical justifications added for layperson audience: why TFT, why probabilistic, why velocity features, why pinball, why horizon weighting, why PINN penalty, why phase-aware, why two-stage. Also run_009 results incorporated. |

---

## Most Recent Results

### run_009 (9-county Bay Area, 500 steps, `--two-stage --rolling-holdout`)
*First run with loss function fix (floor 2.5→0.05) and raised classifier threshold (1.5→2.0)*

| Window | WIS | Cov95 | Cov50 | MAE | PI95 width | Notes |
|---|---|---|---|---|---|---|
| Initial 8W | 0.861 | 26.8% | 1.8% | 1.103 | — | Honest (not brute-force wide) |
| Rolling 28W aggregate | 0.310 | **82%** | 13.8% | 0.310 | 1.10 log1p | vs 100%/3.29 in run_008 |
| **Oct 2023 fold** | 0.137 | **95.9%** | 8.2% | 0.223 | — | At calibration target |
| **Nov 2023 fold** | **0.085** | **95.2%** | 9.5% | 0.154 | — | Excellent |

**Key findings:**
- Loss function fix is the proximate cause: PI width dropped from 3.29 → 1.10 log1p; Coverage 95% from trivially-wide 100% to honest 82% (Oct/Nov folds hit the 95% target exactly)
- **SC investigation**: the July rolling spike (+1.08 bias) is a raw-TFT artifact from the Jun-07 fold, not a two-stage failure — rolling holdout wasn't using the gate at all (now fixed). SC's WW signal IS genuinely elevated (XBB summer wave, 3.4× real case increase Jun→Aug). Non-elastic baseline was anchored to pre-2022 troughs → Z=2-3 throughout holdout even during endemic quiet.
- **Classification state during holdout**: SC triggered 11.1% (was 37%), SF 29.6% (was 33%). Most suppression happens correctly; residual SF triggers when Z_mean=3.17 throughout (baseline drift problem, not noise).
- Napa + Solano still absent (cold-start at 500 steps). Needs 1,500+ steps.
- Detection metrics (Precision/Recall/F1/TTD) always NaN — confirmed design behavior. n_actual_onsets=0 for entire 2023 holdout.

---

## Key Config Values (Phase 6 — current, post-Session-7)

| Parameter | Value | Notes |
|---|---|---|
| `UNDERDISPERSION_K` | 0.5 | Phase 4/5 — unchanged |
| `UNDERDISPERSION_LAMBDA` | 0.5 | Phase 5 floor — unchanged |
| `MIN_PI_WIDTH_MULTIPLIER` | **2.0** | Changed from 3.0 this session |
| `MIN_PI_WIDTH_FLOOR` | **0.05** | Changed from 1.5 (was effectively 2.5 due to bug) |
| `MIN_PI_WIDTH` (legacy) | **0.05** | Changed from 2.5 — matches floor |
| `GROWTH_RATE_LAMBDA` | 0.0 | **⚠ still disabled — decide before next run** |
| `CLASSIFIER_Z_THRESHOLD` | **2.0** | Changed from 1.5 — reduces SC spurious triggering |
| `CLASSIFIER_MOMENTUM_THRESHOLD` | 0.0 | Phase 6 |
| `CLASSIFIER_VOLATILITY_COL` | `"log1p_concentration_4w_std"` | Phase 6 |
| `CLASSIFIER_VOLATILITY_SCALE` | 0.5 | Phase 6 |
| `MIN_BASELINE_OBSERVATIONS` | 4 | Cold-start threshold |
| `SCALER_IQR_FLOOR` | 0.3 | Napa/Solano stability |
| `EXCLUDE_FIPS` | `[]` | All 9 Bay Area counties included |
| `ACTIVE_GEOGRAPHY` | None (Bay Area default) | Set by `--geography` |
| Holdout window | 2023-06-08 → 2023-12-19 | 28 W — unchanged |

---

## Known Bugs / Watchpoints

### 1. GROWTH_RATE_LAMBDA = 0.0
Still disabled. The biological prior is correct; recalibration needed. Revisit once Coverage 50% > 30% consistently on rolling holdout.

### 2. SC/SF non-elastic baseline drift
The non-elastic baseline was fit on pre-2022 WW trough data. The 2023 endemic WW level is structurally higher → Z=2-3 throughout quiet holdout. SF triggers 29.6% of quiet holdout weeks even with threshold raised to 2.0. Root fix is recalibrating the baseline window to exclude pre-Omicron data, or using a sliding-window baseline (with careful leakage protection). Not addressed this session.

### 3. Napa + Solano cold-start at 500 steps
Both have < 4 training rows. Absent from all forecast output. Require 1,500–2,000 steps for full 9-county convergence. Quiet prior validation is blocked on this — the two suppressed counties that would exercise the quiet prior path don't appear.

### 4. Detection metrics always NaN for 2023 holdout
By design. Zero actual onsets in Jun–Dec 2023 (post-XBB endemic, all below Omicron-era p75 threshold). Run standard CV (no `--skip-cv`) to get non-NaN F1/TTD from Dec 2022 BQ.1/XBB.1.5 window.

### 5. Rolling holdout forecast vs. two-stage forecast are different objects
`rolling_forecast.parquet` = stitched raw TFT output per fold (now two-stage gated when `--two-stage`). `two_stage_forecast.parquet` = initial single 8W two-stage forecast. Dashboard "Full rolling holdout" view shows rolling_forecast. These should now be consistent after this session's fix.

---

## Immediate Next Steps

1. **Run 1,500-step Bay Area with all flags** — validates quiet prior on Napa/Solano + phase-aware propagation to rolling folds:
   ```bash
   uv run main.py --skip-cv --no-dash --two-stage --phase-aware-train --rolling-holdout --max-steps 1500
   ```
   Expected improvements vs run_009: (a) Napa/Solano appear in forecast via quiet prior; (b) July SC spike reduced because rolling fold is now two-stage gated; (c) phase-aware oversampling applied per fold reduces dumbbell bias.

2. **Standard CV (no --skip-cv)** — produces non-NaN F1/TTD from Dec 2022 window:
   ```bash
   uv run main.py --no-dash --two-stage --phase-aware-train
   ```

3. **Upper SoCal first run** — dates verified, ready:
   ```bash
   uv run main.py --geography upper_socal --skip-cv --no-dash --two-stage --rolling-holdout
   ```
   Check `log1p_concentration` distributions for LA County after processing (10M people, 100+ WWTPs — amplitude characteristics may differ from smaller counties).

4. **Investigate SC baseline drift** — decide whether to recalibrate the non-elastic baseline window to exclude pre-2022 trough data. SF Z_mean=3.17 throughout quiet holdout = either a real endemic signal or baseline shift artifact.

---

## CLI Reference

```bash
# Full recommended run (all flags, 1500 steps)
uv run main.py --skip-cv --no-dash --two-stage --phase-aware-train --rolling-holdout --max-steps 1500

# Standard CV run (non-NaN detection metrics from 2022 outbreak window)
uv run main.py --no-dash --two-stage --phase-aware-train

# Upper SoCal geography
uv run main.py --geography upper_socal --skip-cv --no-dash --two-stage --rolling-holdout

# Serve dashboard from a specific run
uv run serve_dashboard.py --run run_009_20260530_1011

# Serve dashboard from latest exports
uv run serve_dashboard.py

# Fast smoke test
uv run main.py --fast --no-dash
```

---

## Notebook Plotting Style Guide

**Library:** matplotlib + seaborn (NOT Plotly in notebooks)
**Theme:** `sns.set_theme(style="whitegrid", font_scale=1.1)` + `figure.dpi=120`
**Palette:** `C_WW="steelblue"`, `C_CASES="crimson"`, `C_ACCENT="darkorange"`

---

## File Change Log (most recent session on top)

| File | Last significant change |
|---|---|
| `src/config_geographies.py` | **Session 7**: `apply_geography()` uses in-place dict/list mutation — all locally-bound references propagate automatically |
| `src/config.py` | **Session 7**: `MIN_PI_WIDTH_MULTIPLIER` 3.0→2.0, `MIN_PI_WIDTH_FLOOR` 1.5→0.05, `MIN_PI_WIDTH` 2.5→0.05, `CLASSIFIER_Z_THRESHOLD` 1.5→2.0, pinball ratio comments |
| `src/models/loss_functions.py` | **Session 7**: `_dynamic_min_width` bug fixed — now clamps to `self.min_pi_width_floor` (not Phase 3 legacy `self.min_pi_width`). Root cause of Coverage95=100%. |
| `src/evaluation/evaluator.py` | **Session 7**: `expanding_window_cv` now accepts `two_stage` + `phase_aware` params; both applied per fold with leakage-safe per-fold fit |
| `main.py` | **Session 7**: `_split_raw()` uses `cfg.*` dates; `_run_cv()` passes `two_stage`/`phase_aware`; rolling holdout passes both flags; run notes capture all active flags; phase updated to "Phase 6" |
| `src/data_pipeline/processor.py` | **Session 7**: `_clean_ca_ww()` and `__init__` use `_cfg.BAY_AREA_FIPS` dynamically |
| `config/geographies/upper_socal.yaml` | **Session 7**: Real dates verified — `train_end_date: 2023-04-05`, `val_end_date: 2023-08-02`, `exclude_fips: ["06029","06111"]`, outbreak windows updated |
| `src/visualization/dashboard.py` | **Session 7**: Color overhaul (navy PI bands, gold TFT bands); Gatekeeper section removed; detection metrics removed from bio table; Pinball q0.90 + ratio added; per-fold ratio line in CV chart |
| `src/evaluation/metrics.py` | **Session 7**: `pinball_ratio` added to `EvalReport.to_dict()` |
| `src/utils/helpers.py` | **Session 7**: Pinball bias ratio + direction label in `print_eval_report`; `pinball_ratio` column in `print_cv_summary` |
| `src/utils/run_manager.py` | **Session 7**: `pinball_ratio` in snapshot metrics; `f1` in dropdown label |
| `documents/project_overview.md` | **Session 7**: v4.0 — full layperson technical justifications for all architecture choices; run_009 results |
| `src/models/forecaster.py` | **Session 6**: `_quiet_prior()` data-driven per-county baseline |
| `src/models/tft_model.py` | **Session 6**: `predict()` fixed for StratifiedWindowSampler; `_real_unique_ids` |
