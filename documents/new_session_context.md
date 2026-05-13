# Sewer Signals — Session Context Injection
> Copy-paste this entire block as your first message in a new Claude Code window.

---

## Project Identity

- **Name:** Sewer Signals — Probabilistic COVID-19 Outbreak Forecasting via Wastewater Surveillance
- **Root:** `/Users/prasann/Dev/support_vectors/LLM/labs/wastewater/`
- **Phase:** 2 — "Momentum Pivot" (active development)
- **Goal:** 8-week probabilistic case forecasts for 9 Bay Area counties using wastewater as a leading indicator. Target = `log1p_new_cases` (RobustScaled). WW signal = `hist_exog` input, not the prediction target.

**Read these two files before doing anything else:**
- `memory.md` — permanent project DNA (architecture, covariate table, temporal map)
- `claude.md` — active dev context (Phase 2 changes, known bugs, next steps, CLI ref)

---

## Architecture in One Paragraph

Global **Temporal Fusion Transformer** (NeuralForecast, PyTorch Lightning) trained across all 9 counties simultaneously. 15 `hist_exog` features (WW level + 3 lags, case lags ×3, growth rate, decay rate, outlier flag, + 5 Phase-2 derivative/momentum features). 7-quantile output `[0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]` with horizon weights `[2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]`. Loss = `PINNWastewaterLoss` (MQLoss + λ·growth-rate penalty; λ=0.0 currently disabled). `scaler_type="identity"` in NeuralForecast; processor handles RobustScaling. `domain_map` is identity reshape (Softplus was removed — it collapsed negative RobustScaled targets to zero). W-WED (Wednesday-anchored) weekly spine throughout.

---

## Current Status

**Last verified:** 2000-step training run completed exit 0 on MPS GPU, 4.4M params.

**Phase 2 fixes applied but NOT yet empirically verified:**

| Fix | Root Cause Addressed |
|---|---|
| Removed Softplus from `domain_map` | Collapsed RobustScaled negatives to zero floor |
| `scaler_type="identity"` | Double-scaling compressed PI spread to noise floor |
| `GROWTH_RATE_LAMBDA = 0.0` | PINN penalty suppressed surge trajectories |
| 7 quantiles (was 5) | Insufficient tail gradient signal |
| 5 new derivative features | Model lacked velocity/acceleration vocabulary |

**Holdout Coverage_95 was 0% in Phase 1. Whether Phase 2 fixes it is still unknown.**

**Immediate next action:** `python main.py --skip-cv --no-dash` — capture `coverage_50`, `coverage_95`, `mean_wis`, `smape`. Do not pipe through `head`.

---

## Known Bugs (memorise these)

1. **`horizon_weight` must be `np.array`** — `BasePointLoss.flatten()` fails on Python list. Always: `np.array(TFT_CONFIG["horizon_weight"], dtype=np.float32)`
2. **Rich progress bar nesting** — PL `RichProgressBar` drains `_live_stack` inside outer `rich.Progress`. Fix: `enable_progress_bar=False, enable_model_summary=False` in `cv_trainer_kwargs`
3. **`val_size=0` requires `early_stop_patience_steps=-1`** — enforced in CV loop; `early_stop_patience_steps` is popped from base `trainer_kwargs` in `WastewaterTFT.__init__`
4. **CV freq must be `"4W-WED"`** not `"4W"` — `"4W"` anchors to Sunday, off by 4 days, fold cutoffs land on non-observation dates
5. **loguru uses `{}` not `%s`** — `logger.warning("msg: {}", val)`

---

## Documents Created This Session

All in `documents/`:

| File | Purpose |
|---|---|
| `technical_design_document.md` | Full TDD: math formulations, model specifics, architecture justification |
| `post_mortem.md` | 4-section engineering post-mortem (target mismatch, zero-coverage, WIS/coverage paradox, temporal alignment) |
| `project_overview.md` | State-of-the-Union with 6 embedded EDA figures |
| `README.md` | GitHub-facing project showcase |

EDA figures in `assets/` (generated from `02_eda_bay_area.ipynb`):
`fig_a_county_ww_timeseries.png`, `fig_5_cases_grid.png`, `fig_d_merged_ww_cases_dualaxis.png`, `fig_10_rate_of_change.png`, `fig_b_aggregate_zscore_leadtime.png`, `fig_13_wave_synchrony.png`

Figure generation scripts: `assets/generate_overview_figures.py`, `assets/generate_extra_figures.py`

---

## Pending Task: Notebook 03

**Task:** Create `documents/03_eda_bay_area_CAdatasets.ipynb` — EDA of two new California state-level datasets, still filtered to 9 Bay Area counties.

**Input files:**
- `data/raw/California_Wastewater_Surveillance_Data.csv`
- `data/raw/Statewide_COVID-19_Cases_Deaths_Tests.csv`

**Use `02_eda_bay_area.ipynb` as the structural and visual template.**

### Pre-profiled findings (do not re-profile, use these):

**CA WW dataset (535,473 rows):**
- After Bay Area + `PCR Target == 'SARS-CoV-2'` filter: **55,700 rows**, all 9 counties
- Columns: `Region`, `County`, `County (City/Utility)`, `Abbreviated Name`, `Sample Date`, `Sample Type`, `PCR Gene Target`, `PCR Target`, `Below Lod`, `Raw Concentration`, `Raw Conc Roll Average`, `Norm Pmmov`, `Norm Pmmov Roll Average`, `Data Source`
- `Sample Type`: solid (33,067) / liquid (22,633)
- `Data Source`: WastewaterSCAN (32,877), CDPH Drinking Water and Radiation Lab (16,432), CDC NWSS Commercial Contract Verily (6,201), CDPH NWSS Verily (190)
- Date range: 2020-07-16 → 2026-05-05
- `Raw Concentration`: ~50% null; valid range 250–155M; median ~2,975
- `Norm Pmmov`: ~50% null (same rows); valid range 0–509,135; median ~5.7
- `Below Lod`: False=27,547, True=512, NaN=27,641 (NaN = no measurement taken)
- Solid track null % for Raw Conc/Norm Pmmov: **37.6%**
- Sites per county: Alameda=5, CC=3, Marin=5, Napa=2, SF=8, SM=4, SC=5, Solano=2, Sonoma=3

**CA Cases dataset (86,559 rows):**
- Columns: `_id`, `date`, `area`, `area_type`, `population`, `cases`, `cumulative_cases`, `deaths`, `cumulative_deaths`, `total_tests`, `cumulative_total_tests`, `positive_tests`, `cumulative_positive_tests`
- **Date format: `%m/%d/%y`** (e.g., `'12/19/23'`) — parse with `pd.to_datetime(format='%m/%d/%y')`
- **Data is DAILY** — all 7 weekdays present uniformly; 1,418 unique dates per county
- Date range: 2020-02-01 → **2023-12-19** (7 months beyond the CDC 2023-05-10 cutoff)
- Bay Area rows: 12,771; all 9 counties present
- Area column maps **directly** to county names — no FIPS needed
- Null %: ~0.07% for `date`, `deaths`, `total_tests`, `positive_tests` (~9 rows); 0% otherwise
- Use `cases` as primary signal (`total_tests`/`positive_tests` have more nulls)

### Key structural differences from CDC datasets:
- `PCR Target` column (not `pcr_target`) — value is `'SARS-CoV-2'` (mixed case, not all-caps)
- `Sample Date` (not `sample_collect_date`)
- Two signal columns: `Raw Concentration` + `Norm Pmmov` (vs single `pcr_target_avg_conc`)
- Cases are **daily** → must resample to W-WED before any join
- `area` column (not `county` + FIPS) — direct name match
- New overlap window: ~2022-01-01 → 2023-12-19 (vs CDC window 2022-02-07 → 2023-05-10)

### Required notebook sections (mirror `02_eda_bay_area.ipynb` rigor and style):
1. **Setup** — constants, file checks, `sns.set_theme(style='whitegrid')`, light theme
2. **Section 1: CA WW Data** — load/filter, sample-type × data-source audit, Raw Conc vs Norm Pmmov scatter, statistical summary, missing data heatmap, 3×3 time-series grid, Norm Pmmov vs Raw Conc overlay (4 counties dual-axis)
3. **Section 2: CA Cases Data** — load/filter, date format audit (confirm daily), statistical summary, missing data analysis, 3×3 cases grid, extended temporal window vs CDC comparison
4. **Section 3: Cross-Dataset Alignment** — overlap windows, daily→W-WED resampling demo, county coverage matrix (joint modelable weeks CDC vs CA), merged dual-axis plots (4 counties), lead-time cross-correlation profiles
5. **Section 4: Signal Quality** — Raw Conc vs Norm Pmmov predictive correlation comparison (which tracks cases better?), PMMoV seasonal stability, data-source consistency (multi-source counties), cross-county correlation heatmap
6. **Section 5: Spine Readiness Verdict** — coverage table, differences-from-CDC table, YES/NO verdict per dataset with conditions

**Output goal:** Answer "Is this data clean enough to build our new unified spine?"

---

## CLI Reference

```bash
python main.py                        # Full run: CV + final model + dashboard
python main.py --skip-cv --no-dash    # Fastest: training + holdout eval only
python main.py --fast --no-dash       # Smoke test (reduced steps)
python main.py --max-steps 500 --no-dash
```
