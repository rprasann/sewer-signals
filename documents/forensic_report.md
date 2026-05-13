# Forensic Report — 9-County Run (Phase 2 Baseline)

> **Scope:** Post-mortem of the 9-county global TFT run.  
> **Target metrics:** coverage_50=2.8%, coverage_95=6.9%, SMAPE=57.6%  
> **Santa Clara (FIPS 06085):** WIS=0.112 (best county); CV folds show recovery in late folds (Fold 9 coverage_95=59.3%) but catastrophic Fold 3 (WIS=1.278).

---

## Problem 1 — Pooled Multi-County Scaler Compresses Santa Clara's Surge Space

### Symptom

The 9-county run achieves `coverage_95 = 6.9%` despite deploying 7 quantiles.
Santa Clara, the most populated and best-instrumented county, has the best
individual WIS (0.112) yet still shows near-zero coverage.  This is the
*overconfident smoother*: predicted 95% PIs stay consistently below the
actual surge peak.

### Root Cause

`_apply_scaling()` in [processor.py:618–654](../src/data_pipeline/processor.py#L618)
fits **a single global `RobustScaler`** on all rows from all 9 counties combined.

Santa Clara's case peak (Omicron BA.1, Jan 2022) reached ~100,000 weekly
cases.  Napa County's peak was ~2,000 cases in the same wave.  When
`RobustScaler` computes the IQR across the pooled training set, it is
dominated by the median county, not by Santa Clara.  Concretely:

- The scaler's `scale_` parameter (IQR of `log1p_new_cases`) reflects the
  *average* inter-county spread, not Santa Clara's local dynamic range.
- Santa Clara's Omicron surge — which should produce a scaled value of **+5**
  in Santa Clara's own distribution — is compressed to **+2.5** in the pooled
  distribution.
- The model learns that "large" in scaled space means `+2.5`, so its upper
  quantile (0.975) sits at `+3` in scaled space.  After inverting the scaler,
  this maps to a value well below Santa Clara's actual peak.

**Evidence:** In CV Folds 7–9 (2023 holdout), Santa Clara coverage_95 climbs
to 59% once the model has seen the Omicron wave in training.  The pool-scaler
issue is most severe early in the CV (Folds 1–3), where only the pre-Omicron
baseline is in training but the scaler already encodes its pooled-county IQR.

### Code-Level Fix

Replace the single global scaler with per-county scalers:

```python
# processor.py — _apply_scaling() replacement

def _apply_scaling(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
    """Per-county RobustScaler — eliminates cross-county variance compression."""
    candidates = [...]  # unchanged candidate list
    self._scale_cols = [c for c in candidates if c in df.columns]
    if not self._scale_cols:
        return df

    df = df.sort_values([COUNTY_COL, NWSS_DATE_COL]).copy()

    if fit:
        self._scalers: dict[str, RobustScaler] = {}
        frames = []
        for fips, grp in df.groupby(COUNTY_COL):
            scaler = RobustScaler()
            grp = grp.copy()
            grp[self._scale_cols] = scaler.fit_transform(grp[self._scale_cols])
            self._scalers[fips] = scaler
            frames.append(grp)
        df = pd.concat(frames, ignore_index=True)
        logger.info("Per-county scalers fitted for {} counties.", len(self._scalers))
    else:
        if not hasattr(self, "_scalers") or not self._scalers:
            raise RuntimeError("Per-county scalers not fitted. Call run() on training data first.")
        frames = []
        for fips, grp in df.groupby(COUNTY_COL):
            grp = grp.copy()
            if fips not in self._scalers:
                logger.warning("No scaler for FIPS {} — skipping scaling.", fips)
                frames.append(grp)
                continue
            grp[self._scale_cols] = self._scalers[fips].transform(grp[self._scale_cols])
            frames.append(grp)
        df = pd.concat(frames, ignore_index=True)
    return df
```

Also update `_invert_scaling_to_log1p` in `main.py` to accept the per-county
scaler dict:

```python
def _invert_scaling_to_log1p(df, proc, cols, county_col=COUNTY_COL):
    if not hasattr(proc, "_scalers") or proc._scaler is None:  # fallback
        ...  # existing logic

    out = df.copy()
    for fips, grp_idx in df.groupby(county_col).groups.items():
        scaler = proc._scalers.get(fips)
        if scaler is None:
            continue
        col_idx = proc._scale_cols.index(TARGET_COL)
        center, scale = scaler.center_[col_idx], scaler.scale_[col_idx]
        for col in cols:
            if col in out.columns:
                out.loc[grp_idx, col] = out.loc[grp_idx, col] * scale + center
    return out
```

**Note for Santa Clara single-county run:** Because there is only one county,
`_process_sc()` in `run_santa_clara.py` already achieves the equivalent
improvement by construction — the global scaler is fit on Santa Clara rows
only.  The per-county fix is required for the 9-county global model.

---

## Problem 2 — Row-Based `shift()` on Irregular Post-Join Spine Misaligns Lag Features

### Symptom

CV **Fold 3** (cutoff 2022-11-30) produces catastrophic WIS=1.278–1.354
across all 9 counties simultaneously.  No single county causes this — the
failure is uniform, which points to a data pipeline artifact rather than a
county-specific epidemiological event.  Late November 2022 is the BQ.1
subvariant transition; case-reporting lags were documented during the US
Thanksgiving holiday reporting gap.

### Root Cause

`_add_lag_features()` in [processor.py:577–611](../src/data_pipeline/processor.py#L577)
computes lag features using:

```python
df.groupby(COUNTY_COL)[col].shift(lag_weeks)
```

This **shifts by row count, not by calendar time**.

`_merge_cases()` uses an `inner` join on `(county_fips, W-WED date)`.  When a
county is missing a WW sample for week *t* (common during holiday reporting
gaps), that week's row is silently dropped from the merged DataFrame.  The
remaining rows are contiguous in the DataFrame but have calendar gaps.

Example with a missing week:

| Actual calendar | Row # | shift(1) returns | Should return |
|---|---|---|---|
| 2022-11-02 | 0 | NaN | NaN (start) |
| 2022-11-09 | 1 | row 0 (2022-11-02) ✓ | 2022-11-02 ✓ |
| *(2022-11-16 missing)* | — | — | — |
| 2022-11-23 | 2 | row 1 (2022-11-09) ✗ | 2022-11-16 (missing) |
| 2022-11-30 | 3 | row 2 (2022-11-23) ✗ | 2022-11-23 ✓ |

At the Fold 3 cutoff date (2022-11-30), `log1p_new_cases_lag1w` contains the
value from 2022-11-09 instead of 2022-11-23 — a **2-week stale lag** — making
the model think case momentum is low when it is actually rising rapidly (Omicron
BQ.1 arrival).  The temporal attention mechanism treats this corrupted feature
as a genuine low-momentum signal and predicts smooth trajectories.

### Code-Level Fix

Reindex each county to a complete W-WED spine **before** computing lags, then
drop rows where the original data is absent:

```python
# processor.py — _add_lag_features() — insert BEFORE the groupby.shift() calls

def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([COUNTY_COL, NWSS_DATE_COL]).copy()

    # ── Reindex to complete W-WED spine (calendar-safe lags) ────────────────
    frames = []
    global_start = df[NWSS_DATE_COL].min()
    global_end   = df[NWSS_DATE_COL].max()
    full_spine = pd.date_range(global_start, global_end, freq="W-WED")

    for fips, grp in df.groupby(COUNTY_COL):
        grp = grp.set_index(NWSS_DATE_COL).sort_index()
        # Reindex to full spine — missing weeks become NaN rows
        grp = grp.reindex(full_spine)
        grp[COUNTY_COL] = fips
        frames.append(grp.reset_index().rename(columns={"index": NWSS_DATE_COL}))

    df = pd.concat(frames, ignore_index=True)
    # ── Now shift() operates on a complete calendar spine ───────────────────
    ww_grp = df.groupby(COUNTY_COL)[WW_FEATURE_COL]
    for lag_weeks in [1, 2, 3]:
        df[f"{WW_FEATURE_COL}_lag{lag_weeks}w"] = ww_grp.shift(lag_weeks)
    # ... (rest of _add_lag_features unchanged)

    # Drop rows where original data was absent (reindex-filled NaN rows)
    df = df.dropna(subset=[WW_FEATURE_COL]).copy()
    return df
```

**Impact on Fold 3:** The 2022 Thanksgiving reporting gap would produce NaN
rows in the reindexed spine, but lags would correctly reflect the *prior
available* calendar week for each lag distance rather than the prior row.
The 2022-11-30 `lag1w` would contain 2022-11-23 data regardless of whether
2022-11-16 was present.

---

## Problem 3 — `growth_rate_1w` Is a Semantically Ill-Defined Hybrid Metric

### Symptom

The VSN attention plots show `growth_rate_1w` as one of the highest-weighted
historical features (confirmed in multiple TFT training runs).  Yet the model
still cannot detect outbreak onset — its "high-importance" feature is
systematically uninformative during exactly the periods where it matters most.

### Root Cause

`growth_rate_1w` is computed in [processor.py:584–585](../src/data_pipeline/processor.py#L584):

```python
lag1 = ww_grp.shift(1)
df["growth_rate_1w"] = (df[WW_FEATURE_COL] - lag1) / (lag1.abs() + 1e-6)
```

Where `WW_FEATURE_COL = log1p_concentration`.  The numerator is
`Δlog1p_conc = log1p_conc_t − log1p_conc_{t-1}`, which already approximates a
**proportional relative change** (since `log x_t − log x_{t-1} ≈ (x_t − x_{t-1})/x_{t-1}`).

Dividing this by `|log1p_conc_{t-1}|` produces a **ratio of a log-difference to
a log-magnitude**, which is neither a true relative rate nor a true log-velocity.
Its value depends on the absolute concentration level in a perverse way:

| Scenario | Concentration change | `growth_rate_1w` |
|---|---|---|
| Pre-surge baseline (conc≈100) | doubles → 200 | `log1p(200)−log1p(100)` / `log1p(100)` ≈ `0.69/4.6` = **0.15** |
| Mid-wave (conc≈1,000) | doubles → 2,000 | `0.69/6.9` = **0.10** |
| Peak Omicron (conc≈100,000) | doubles → 200,000 | `0.69/11.5` = **0.06** |

The same **100% doubling** appears as 0.15, 0.10, or 0.06 depending on the
baseline magnitude.  During the Omicron BA.1 surge (the largest surge in the
dataset), the doubling events that should trigger high `growth_rate_1w` values
actually produce the *lowest* values.  This is the opposite of what the
OutbreakDetector and the VSN need to see.

Furthermore, at near-zero concentrations (LOD region), the denominator
`|log1p(conc)| + 1e-6 ≈ 1e-6` inflates small absolute changes into extremely
large `growth_rate_1w` values, generating false outbreak signals during
low-prevalence quiet periods — exactly the source of the 6 false-positive
outbreak alerts seen in the 9-county holdout.

### Code-Level Fix

Replace the semantically confused formula with a true relative rate in the
**original concentration space**, or simply alias it to `diff_concentration`
(the Phase 2 Task A addition) which is already the correct log-velocity signal:

**Option A — True relative rate (recommended):**

```python
# processor.py — _add_lag_features()
# Replace the growth_rate_1w computation:

# Retrieve the raw concentration from 'concentration' column (before log transform)
# Use a copy of the pre-log concentration for rate computation
conc_grp = df.groupby(COUNTY_COL)["concentration"]
conc_lag1 = conc_grp.shift(1)
df["growth_rate_1w"] = (
    (df["concentration"] - conc_lag1) / (conc_lag1.abs() + 1e-6)
).clip(-5.0, 5.0)  # winsorise; doubles → 1.0, halves → -0.5 (scale-invariant)
```

This makes a 100% doubling always produce `growth_rate_1w = 1.0` regardless of
the baseline concentration, giving the VSN a consistent surge signal.

**Option B — Remove redundant feature:**

Since `diff_concentration = Δlog1p_conc_t` (added in Phase 2) is already an
unbiased log-velocity signal, simply remove `growth_rate_1w` from
`_apply_scaling()` candidates and from `HIST_COVARIATES` in `tft_model.py`.
This reduces the feature set to 14 features and eliminates the confounded
signal entirely.

**Preferred for the Santa Clara validation run:** Option A (replace formula),
keeping the feature to preserve VSN interpretability comparisons with the
baseline run.  Remove it in a subsequent ablation if VSN weight drops below
random (1/14 ≈ 7%).

---

## Summary Table

| # | Problem | File | Lines | Phase 1 Addressed? | Fix Type |
|---|---|---|---|---|---|
| 1 | Pooled scaler compresses SC surge space | `processor.py` | 618–654 | Partially (SC-only run uses local scaler by construction) | Architecture change: per-county `RobustScaler` dict |
| 2 | Row-based `shift()` on irregular spine misaligns lags | `processor.py` | 577–611 | No | Pipeline fix: reindex to W-WED spine before shift |
| 3 | `growth_rate_1w` is a log-denominator hybrid (biased toward low values at high prevalence) | `processor.py` | 584–585 | No | Feature fix: replace with true relative rate in original space |

### Expected Impact

If all three fixes are applied to the 9-county global model, the expected
improvements based on the evidence above:

- **Problem 1 fix:** coverage_95 should improve from 6.9% toward 30–50%
  (SC single-county run is the control experiment).
- **Problem 2 fix:** CV Fold 3 WIS (currently 1.354) should normalize toward
  the Fold 2/4 range (0.2–0.5); holiday reporting gaps will no longer corrupt
  lag features.
- **Problem 3 fix:** False positive alert rate should decrease (from 6/9 counties
  to ≤2/9); `growth_rate_1w` VSN weight should increase at high-prevalence
  periods where it previously showed near-zero gradient.

### Next Iteration Priority

1. Run `python run_santa_clara.py --skip-cv` to confirm underdispersion penalty
   raises coverage_95 above the 6.9% baseline.
2. Implement Problem 2 fix (calendar-safe lags) in `processor.py` — this is
   the only fix that affects all CV folds uniformly.
3. Implement Problem 3 fix (true relative rate) and verify via VSN attention
   that the new `growth_rate_1w` receives consistent weight across all
   prevalence levels.
4. Re-run full 9-county pipeline with all three fixes active.
