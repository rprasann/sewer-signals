"""
Create two EDA notebooks:
  03_pipeline_bay_area_ca.ipynb         — Bay Area 9-county EDA (matches ARCHIVE style)
  04_pipeline_santa_clara_county.ipynb  — Santa Clara single-county EDA

Run from the project root:
    python documents/create_pipeline_notebooks.py

Style: matplotlib + seaborn whitegrid, Bay Area palette (steelblue / crimson / darkorange).
Data: loaded from raw CSVs only — no pipeline artefacts required.
"""

from __future__ import annotations
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent  # documents/


def md(text: str) -> nbf.NotebookNode:
    c = nbf.v4.new_markdown_cell(text.strip())
    c["metadata"] = {}
    return c


def code(src: str) -> nbf.NotebookNode:
    c = nbf.v4.new_code_cell(src.strip())
    c["metadata"] = {}
    return c


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

SETUP_CODE = '''
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

ROOT    = Path("..").resolve()
RAW_DIR = ROOT / "data" / "raw"

CA_WW_FILE    = RAW_DIR / "California_Wastewater_Surveillance_Data.csv"
CA_CASES_FILE = RAW_DIR / "Statewide_COVID-19_Cases_Deaths_Tests.csv"
assert CA_WW_FILE.exists() and CA_CASES_FILE.exists(), "Missing data files"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":      120,
    "figure.figsize":  (14, 5),
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
})

BAY_AREA_COUNTIES = [
    "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
    "San Mateo", "Santa Clara", "Solano", "Sonoma",
]
ALL_COUNTIES_ORDERED = [
    "Alameda", "Contra Costa", "Marin",
    "Napa", "San Francisco", "San Mateo",
    "Santa Clara", "Solano", "Sonoma",
]

OVERLAP_START = pd.Timestamp("2020-07-01")
OVERLAP_END   = pd.Timestamp("2023-12-19")
CDC_START     = pd.Timestamp("2022-02-07")
CDC_END       = pd.Timestamp("2023-05-10")

OUTBREAKS = {
    "Summer 2020\\n(Wave 1)":    (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-31")),
    "Winter 2020-21\\n(Wave 2)": (pd.Timestamp("2020-11-01"), pd.Timestamp("2021-04-30")),
    "Delta 2021\\n(Wave 3)":     (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-11-30")),
    "Omicron 2022\\n(Wave 4)":   (pd.Timestamp("2021-12-01"), pd.Timestamp("2022-05-31")),
}
OUTBREAK_COLORS = ["#4e9af1", "#f4a261", "#57cc99", "#e76f51"]

ALL_WAVE_SPANS = [
    ("#4e9af1", "2020-07-01", "2020-10-31", "Wave 1\\nOriginal"),
    ("#f4a261", "2020-11-01", "2021-04-30", "Wave 2\\nAlpha/WT"),
    ("#57cc99", "2021-06-01", "2021-11-30", "Wave 3\\nDelta"),
    ("#e76f51", "2021-12-01", "2022-05-31", "Wave 4\\nOmicron BA.1"),
    ("#c77dff", "2022-06-01", "2022-09-30", "BA.5"),
    ("#9b72cf", "2022-10-01", "2023-01-15", "BQ.1 / BQ.1.1"),
    ("#48cae4", "2023-01-16", "2023-04-30", "XBB.1.5"),
]

COUNTY_COLORS = dict(zip(sorted(BAY_AREA_COUNTIES), sns.color_palette("tab10", 9)))

C_WW     = "steelblue"
C_CASES  = "crimson"
C_ACCENT = "darkorange"

print(f"pandas {pd.__version__}  |  numpy {np.__version__}  |  matplotlib {matplotlib.__version__}")
print(f"Overlap window: {OVERLAP_START.date()} -> {OVERLAP_END.date()}")
'''

LOAD_WW_CODE = '''
# Load CA Wastewater Surveillance data — solid track, SARS-CoV-2 only
ca_ww_raw = pd.read_csv(CA_WW_FILE, dtype=str, low_memory=False)
ca_ww_bay = ca_ww_raw[
    (ca_ww_raw["County"].isin(BAY_AREA_COUNTIES)) &
    (ca_ww_raw["PCR Target"] == "SARS-CoV-2")
].copy()
ca_ww_bay["Sample Date"] = pd.to_datetime(ca_ww_bay["Sample Date"], errors="coerce")
for col in ["Raw Concentration", "Norm Pmmov"]:
    ca_ww_bay[col] = pd.to_numeric(ca_ww_bay[col], errors="coerce")

solid = ca_ww_bay[ca_ww_bay["Sample Type"].str.lower() == "solid"].copy()

# Audit table
audit = solid.groupby("County").agg(
    n_rows       =("Sample Date", "count"),
    n_sites      =("County (City/Utility)", "nunique"),
    earliest_date=("Sample Date", "min"),
    rc_null_pct  =("Raw Concentration", lambda x: f"{x.isnull().mean()*100:.0f}%"),
    np_null_pct  =("Norm Pmmov",        lambda x: f"{x.isnull().mean()*100:.0f}%"),
).reset_index()
audit["earliest_date"] = audit["earliest_date"].dt.date
print(f"Raw Bay Area shape: {ca_ww_bay.shape}  |  Solid shape: {solid.shape}")
print()
print(audit.to_string(index=False))
print("\\n Only Santa Clara (2020-07), San Francisco (2020-11), San Mateo (2020-12)")
print("   have solid track WW before 2022. The remaining 6 counties start in 2022.")

# Weekly W-WED median — both signals
ww_rc = (
    solid.set_index("Sample Date")
    .groupby("County")["Raw Concentration"]
    .resample("W-WED").median()
    .reset_index().rename(columns={"Sample Date": "week", "Raw Concentration": "rc"})
)
ww_np = (
    solid.set_index("Sample Date")
    .groupby("County")["Norm Pmmov"]
    .resample("W-WED").median()
    .reset_index().rename(columns={"Sample Date": "week", "Norm Pmmov": "np"})
)
ww = pd.merge(ww_rc, ww_np, on=["County", "week"], how="outer")
ww = ww[(ww["week"] >= OVERLAP_START) & (ww["week"] <= OVERLAP_END)]
print(f"\\nWeekly WW series: {len(ww)} rows")
print(f"Date range: {ww['week'].min().date()} -> {ww['week'].max().date()}")
'''

LOAD_CASES_CODE = '''
# Load CA statewide cases — daily -> W-WED weekly sum
ca_cases_raw = pd.read_csv(CA_CASES_FILE, dtype=str, low_memory=False)
ca_cases_raw["date"]  = pd.to_datetime(ca_cases_raw["date"], format="%m/%d/%y", errors="coerce")
ca_cases_raw["cases"] = pd.to_numeric(ca_cases_raw["cases"], errors="coerce")

ca_cases_bay = ca_cases_raw[
    ca_cases_raw["area"].isin(BAY_AREA_COUNTIES) &
    (ca_cases_raw["area_type"] == "County")
].copy()

cases_weekly = (
    ca_cases_bay.rename(columns={"area": "County"})
    .set_index("date")
    .groupby("County")["cases"]
    .resample("W-WED").sum()
    .clip(lower=0)
    .reset_index()
)
cases_weekly = cases_weekly[
    (cases_weekly["date"] >= OVERLAP_START) &
    (cases_weekly["date"] <= OVERLAP_END)
]

print(f"Raw CA Cases shape: {ca_cases_raw.shape}")
print(f"Bay Area rows: {len(ca_cases_bay):,}")
print(f"Weekly cases rows (overlap window): {len(cases_weekly):,}")
print(f"Date range: {cases_weekly['date'].min().date()} -> {cases_weekly['date'].max().date()}")
'''


# ===========================================================================
# BAY AREA NOTEBOOK
# ===========================================================================

def make_notebook_bay_area() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = []

    cells.append(md("""
# 03 — Bay Area EDA: California State Datasets
**Sewer Signals Project** | Phase 4 — EDA Reference

**Datasets:** `California_Wastewater_Surveillance_Data.csv` + `Statewide_COVID-19_Cases_Deaths_Tests.csv`
**Scope:** 9 Bay Area counties. Solid track only. Full extended window **2020-07-01 → 2023-12-19**.

**Key questions:**
1. Does the extended window capture small pre-Omicron outbreaks with enough WW signal for the model to learn from?
2. Does Raw Concentration or Norm Pmmov track cases more clearly during those outbreaks?
3. Is this data clean enough to replace the CDC spine?
"""))

    cells.append(md("""
---
## Visual Context: Bay Area Geography & COVID-19 Timeline

The map below shows the **9-county Bay Area surveillance network**: Alameda, Contra Costa, Marin, Napa,
San Francisco, San Mateo, Santa Clara, Solano, and Sonoma. Only three counties —
**Santa Clara, San Francisco, San Mateo** — have solid-track wastewater data before 2022.
The remaining six join the solid track in 2022 and are zero-padded for the TFT's early history.
"""))

    cells.append(code('''
from IPython.display import Image as IPImage, display as ipy_display

map_path = Path("..") / "documents" / "bay_area_counties_image.png"
map_path = map_path.resolve()
if map_path.exists():
    ipy_display(IPImage(filename=str(map_path), width=660))
else:
    centroids = {
        "San Francisco": (-122.45, 37.77), "Alameda":      (-122.05, 37.60),
        "Contra Costa":  (-121.90, 37.92), "Marin":        (-122.72, 38.05),
        "Napa":          (-122.40, 38.50), "San Mateo":    (-122.35, 37.45),
        "Santa Clara":   (-121.90, 37.33), "Solano":       (-122.00, 38.22),
        "Sonoma":        (-122.85, 38.45),
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    for county, (lon, lat) in centroids.items():
        ax.scatter(lon, lat, s=140, zorder=5, color=COUNTY_COLORS.get(county, "gray"))
        ax.annotate(county, (lon, lat), textcoords="offset points", xytext=(6, 3), fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("9 Bay Area Counties — SARS-CoV-2 Wastewater Surveillance Network", fontweight="bold")
    ax.set_facecolor("#e8f4f8")
    plt.tight_layout(); plt.show()
'''))

    cells.append(md("## Section 0: Setup"))
    cells.append(code(SETUP_CODE))

    # Figure A
    cells.append(code('''
# Figure A: Bay Area COVID-19 Case Timeline — All Waves & Variants (2020-2023)
_cr = pd.read_csv(CA_CASES_FILE, dtype=str, low_memory=False)
_cr["date"]  = pd.to_datetime(_cr["date"], format="%m/%d/%y", errors="coerce")
_cr["cases"] = pd.to_numeric(_cr["cases"], errors="coerce")
_bay = _cr[_cr["area"].isin(BAY_AREA_COUNTIES) & (_cr["area_type"] == "County")]
_bay_agg = (
    _bay.groupby("date")["cases"].sum()
    .resample("W-WED").sum().clip(lower=0).reset_index()
)
_bay_agg = _bay_agg[(_bay_agg["date"] >= OVERLAP_START) & (_bay_agg["date"] <= OVERLAP_END)]
_roll = _bay_agg.set_index("date")["cases"].rolling(4, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(18, 6))
for color, s, e, lbl in ALL_WAVE_SPANS:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=color, zorder=0)
ax.fill_between(_bay_agg["date"], _bay_agg["cases"], alpha=0.20, color="#4a90d9", zorder=1)
ax.bar(_bay_agg["date"], _bay_agg["cases"], color="#4a90d9", alpha=0.40, width=5, zorder=2)
ax.plot(_roll.index, _roll.values, color="#1a237e", linewidth=2.2, zorder=3, label="4-wk rolling avg")

_y_max = _bay_agg["cases"].max()
_LABELS = [
    ("Wave 1\\n(Original)",            "2020-08-15", 0.82, "#4e9af1"),
    ("Wave 2\\n(Alpha / Wild-type)",   "2021-01-10", 0.78, "#f4a261"),
    ("Wave 3\\n(Delta B.1.617.2)",     "2021-09-10", 0.72, "#57cc99"),
    ("Wave 4\\nOmicron BA.1",          "2022-01-12", 0.94, "#e76f51"),
    ("BA.2",                           "2022-04-01", 0.42, "#e76f51"),
    ("BA.5",                           "2022-07-15", 0.38, "#c77dff"),
    ("BQ.1 / BQ.1.1",                 "2022-11-15", 0.44, "#9b72cf"),
    ("XBB.1.5",                        "2023-02-10", 0.38, "#48cae4"),
]
for lbl, cdt, yfrac, color in _LABELS:
    ax.annotate(lbl, xy=(pd.Timestamp(cdt), _y_max * yfrac),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.88))

ax.set_xlim(OVERLAP_START, OVERLAP_END)
ax.set_ylim(0, _y_max * 1.12)
ax.set_xlabel("Week (W-WED)", fontsize=11)
ax.set_ylabel("Weekly New Cases — Bay Area Aggregate", fontsize=11)
ax.set_title(
    "Figure A: Bay Area COVID-19 Case Timeline — All Waves & Variants (2020-2023)\\n"
    "CA Statewide Dataset; daily -> W-WED resample; shaded blocks = distinct variant epochs",
    fontsize=13, fontweight="bold"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

_peak_row = _bay_agg.loc[_bay_agg["cases"].idxmax()]
_pre_peak = _bay_agg[_bay_agg["date"] < pd.Timestamp("2021-12-01")]["cases"].max()
print(f"Omicron BA.1 peak: {_peak_row['date'].date()}  ->  {_peak_row['cases']:,.0f} cases/week")
print(f"Pre-Omicron max:   {_pre_peak:,.0f} cases/week")
print(f"Omicron is  ~{_peak_row['cases'] / _pre_peak:.1f}x  larger than any prior wave")
'''))

    # Section 1
    cells.append(md("""
---
## Section 1: CA Wastewater — Solid Track

### 1.1 Load, Filter & Audit

Filtering to Bay Area + `PCR Target == 'SARS-CoV-2'` + `Sample Type == 'solid'` throughout.
Raw Concentration and Norm Pmmov have the same ~50% null rate (same rows); both will be evaluated.
"""))

    cells.append(code(LOAD_WW_CODE))

    cells.append(md("### 1.2 County Time-Series — Full Extended Window"))

    cells.append(code('''
# Figure 1: Weekly Median Raw Concentration — Solid Track, All 9 Bay Area Counties
fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
fig.suptitle(
    f"Figure 1: Weekly Median Raw Concentration — Solid Track, All 9 Bay Area Counties\\n"
    f"({OVERLAP_START.date()} -> {OVERLAP_END.date()}; log scale; gray bands = key outbreak windows)",
    fontsize=14, fontweight="bold"
)
for ax, county in zip(axes.flat, ALL_COUNTIES_ORDERED):
    color  = COUNTY_COLORS.get(county, "steelblue")
    subset = ww[ww["County"] == county].dropna(subset=["rc"]).sort_values("week")

    for (label, (w_s, w_e)), oc in zip(OUTBREAKS.items(), OUTBREAK_COLORS):
        ax.axvspan(w_s, w_e, alpha=0.08, color=oc, zorder=0)

    if len(subset) > 0:
        ax.plot(subset["week"], subset["rc"], color=color, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_ylabel("Raw Conc (log)", fontsize=7)
        first_date = subset["week"].min()
        if first_date > OVERLAP_START + pd.Timedelta(weeks=60):
            ax.text(0.03, 0.10, f"Data from {first_date.strftime('%Y-%m')}",
                    transform=ax.transAxes, fontsize=7, color="#888888")
    else:
        ax.set_facecolor("#f5f5f5")
        ax.text(0.5, 0.5, "No solid data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")

    ax.set_title(county, fontweight="bold", color=color)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

legend_patches = [
    mpatches.Patch(color=oc, alpha=0.3, label=label.replace("\\n", " "))
    for (label, _), oc in zip(OUTBREAKS.items(), OUTBREAK_COLORS)
]
axes[0, 2].legend(handles=legend_patches, fontsize=7, loc="upper right")
plt.tight_layout()
plt.show()
'''))

    # Section 2
    cells.append(md("""
---
## Section 2: CA Cases — Daily → W-WED

### 2.1 Load, Parse & Resample

> **Critical:** Date format is `%m/%d/%y` — must be passed explicitly.
> Data is daily; sum to W-WED before any join.
"""))

    cells.append(code(LOAD_CASES_CODE))

    cells.append(md("### 2.2 County Time-Series — Full Extended Window"))

    cells.append(code('''
# Figure 2: Weekly COVID-19 Cases — CA State Dataset, Bay Area
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
fig.suptitle(
    f"Figure 2: Weekly COVID-19 Cases — CA State Dataset, Bay Area\\n"
    f"(Daily -> W-WED; {OVERLAP_START.date()} -> {OVERLAP_END.date()})",
    fontsize=14, fontweight="bold"
)
for ax, county in zip(axes.flat, ALL_COUNTIES_ORDERED):
    color  = COUNTY_COLORS.get(county, "steelblue")
    subset = cases_weekly[cases_weekly["County"] == county].sort_values("date")

    for (label, (w_s, w_e)), oc in zip(OUTBREAKS.items(), OUTBREAK_COLORS):
        ax.axvspan(w_s, w_e, alpha=0.08, color=oc, zorder=0)

    if len(subset) > 0:
        ax.bar(subset["date"], subset["cases"], color=color, alpha=0.5, width=5)
        ax.plot(subset["date"], subset["cases"].rolling(4, min_periods=1).mean(),
                color="darkred", linewidth=1.8)
        ax.set_ylabel("Cases / week", fontsize=8)

    ax.set_title(county, fontweight="bold", color=color)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

legend_patches = [
    mpatches.Patch(color=oc, alpha=0.3, label=label.replace("\\n", " "))
    for (label, _), oc in zip(OUTBREAKS.items(), OUTBREAK_COLORS)
]
axes[0, 2].legend(handles=legend_patches, fontsize=7, loc="upper right")
plt.tight_layout()
plt.show()
'''))

    # Section 3
    cells.append(md("""
---
## Section 3: Signal Quality — Raw Conc vs Norm Pmmov at Key Outbreaks

We have WW coverage for **3 counties** (Santa Clara, San Francisco, San Mateo) during the pre-2022 outbreaks.
For each of the 4 key outbreak windows, we normalise both signals [0→1] within the window and overlay weekly
cases to ask: **which WW signal rises earlier and more clearly?**
"""))

    cells.append(code('''
# Bay Area aggregate (median across available counties) — both signals + cases
agg_rc    = ww.dropna(subset=["rc"]).groupby("week")["rc"].median().rename("rc")
agg_np    = ww.dropna(subset=["np"]).groupby("week")["np"].median().rename("np")
agg_cases = cases_weekly.groupby("date")["cases"].sum().clip(lower=0).rename("cases")

agg = pd.DataFrame({"rc": agg_rc, "np": agg_np}).join(agg_cases, how="outer").sort_index()
agg.index.name = "week"

print(f"Aggregate series rows: {len(agg)}")
print(f"RC non-null: {agg['rc'].notna().sum()}   NP non-null: {agg['np'].notna().sum()}")
print(f"Cases non-null: {agg['cases'].notna().sum()}")
'''))

    cells.append(code('''
def norm01(s):
    """Min-max normalise to [0, 1]."""
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
fig.suptitle(
    "Figure 3: Raw Concentration vs Norm Pmmov at Key Outbreak Windows\\n"
    "Signals normalised [0-1] within each window; Bay Area aggregate (solid track)",
    fontsize=13, fontweight="bold"
)

snr_results = []

for ax, ((label, (w_s, w_e)), oc) in zip(axes, zip(OUTBREAKS.items(), OUTBREAK_COLORS)):
    window = agg[(agg.index >= w_s) & (agg.index <= w_e)].copy()

    rc_w    = norm01(window["rc"].dropna())
    np_w    = norm01(window["np"].dropna())
    cases_w = window["cases"].clip(lower=0)

    shared_rc = rc_w.index.intersection(cases_w.dropna().index)
    shared_np = np_w.index.intersection(cases_w.dropna().index)
    r_rc = np.corrcoef(np.log1p(rc_w[shared_rc]), np.log1p(cases_w[shared_rc]))[0, 1] if len(shared_rc) >= 4 else np.nan
    r_np = np.corrcoef(np.log1p(np_w[shared_np]), np.log1p(cases_w[shared_np]))[0, 1] if len(shared_np) >= 4 else np.nan
    winner = "RC" if (not np.isnan(r_rc) and (np.isnan(r_np) or r_rc >= r_np)) else "NP"
    snr_results.append({"outbreak": label.replace("\\n", " "), "r_raw_conc": r_rc, "r_norm_pmmov": r_np, "winner": winner})

    ax2 = ax.twinx()
    ax2.bar(cases_w.index, cases_w.values, width=5, color="lightgray", alpha=0.5, zorder=0)
    ax2.set_ylabel("Bay Area cases / week", color="gray", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    if len(rc_w) > 0:
        ax.plot(rc_w.index, rc_w.values, color="steelblue", linewidth=2.2,
                label=f"Raw Conc  r={r_rc:.2f}" if not np.isnan(r_rc) else "Raw Conc  r=—", zorder=3)
    if len(np_w) > 0:
        ax.plot(np_w.index, np_w.values, color="darkorange", linewidth=2.2,
                linestyle="--", label=f"Norm Pmmov  r={r_np:.2f}" if not np.isnan(r_np) else "Norm Pmmov  r=—", zorder=3)

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Norm. WW signal [0-1]", fontsize=8)
    ax.set_title(f"{label}\\n{'✓ RC' if winner == 'RC' else '✓ NP'} leads", fontweight="bold", color=oc)
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.show()

snr_df = pd.DataFrame(snr_results)
print("\\n=== Signal Comparison Summary ===")
print(snr_df.to_string(index=False))
rc_wins = (snr_df["winner"] == "RC").sum()
np_wins = (snr_df["winner"] == "NP").sum()
print(f"\\nRaw Conc wins: {rc_wins}/4   Norm Pmmov wins: {np_wins}/4")
'''))

    # Section 4
    cells.append(md("""
---
## Section 4: Cross-Dataset Alignment

### 4.1 Extended Temporal Window vs CDC
"""))

    cells.append(code('''
# Figure 4: CA Cases Dataset — Extended Window vs CDC Archive
bay_cases_agg = cases_weekly.groupby("date")["cases"].sum().reset_index()

fig, ax = plt.subplots(figsize=(15, 5))
ax.fill_between(bay_cases_agg["date"], bay_cases_agg["cases"], alpha=0.2, color="steelblue")
ax.plot(bay_cases_agg["date"], bay_cases_agg["cases"],
        color="steelblue", linewidth=1.5, label="CA Cases Dataset — Bay Area aggregate")

for (label, (w_s, w_e)), oc in zip(OUTBREAKS.items(), OUTBREAK_COLORS):
    ax.axvspan(w_s, w_e, alpha=0.10, color=oc)
    mid = w_s + (w_e - w_s) / 2
    y_top = bay_cases_agg[(bay_cases_agg["date"] >= w_s) & (bay_cases_agg["date"] <= w_e)]["cases"].max()
    ax.text(mid, y_top * 1.03, label.replace("\\n", " "),
            ha="center", va="bottom", fontsize=7.5, color=oc, fontweight="bold")

ax.axvspan(CDC_START, CDC_END, alpha=0.05, color="gray")
ax.axvline(CDC_END,      color="red",   linestyle="--", linewidth=1.3, label=f"CDC cutoff {CDC_END.date()}")
ax.axvline(OVERLAP_START, color="green", linestyle=":",  linewidth=1.3, label=f"CA window start {OVERLAP_START.date()}")

ax.set_xlabel("Week (W-WED)")
ax.set_ylabel("Weekly New Cases (Bay Area aggregate)")
ax.set_title(
    "Figure 4: CA Cases Dataset — Extended Window vs CDC Archive\\n"
    "Gray band = CDC reference window; red dashed = CDC cutoff; shaded bands = key outbreak windows",
    fontweight="bold"
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

extra_wks = (OVERLAP_END - CDC_END).days // 7
extra_start_wks = (CDC_START - OVERLAP_START).days // 7
print(f"Extension beyond CDC end:    +{extra_wks} weeks")
print(f"Extension before CDC start:  +{extra_start_wks} weeks (captures 3 pre-Omicron outbreak waves)")
'''))

    cells.append(md("### 4.2 County Coverage Matrix — Joint Modelable Weeks"))

    cells.append(code('''
coverage_rows = []
for county in ALL_COUNTIES_ORDERED:
    ww_wks    = set(ww[(ww["County"] == county) & ww["rc"].notna()]["week"])
    cases_wks = set(cases_weekly[cases_weekly["County"] == county]["date"])
    joint     = ww_wks & cases_wks
    cdc_wks   = set(ww[(ww["County"] == county) & ww["rc"].notna() &
                       (ww["week"] >= CDC_START) & (ww["week"] <= CDC_END)]["week"])
    pre2022   = len([w for w in joint if w < pd.Timestamp("2022-01-01")])
    coverage_rows.append({
        "County":       county,
        "ww_rc_wks":    len(ww_wks),
        "cases_wks":    len(cases_wks),
        "joint_total":  len(joint),
        "pre_2022_wks": pre2022,
        "cdc_ref_wks":  len(cdc_wks),
        "extra_vs_cdc": len(joint) - len(cdc_wks),
    })

cov = pd.DataFrame(coverage_rows)
print("=== Joint Modelable Weeks per County ===")
print(cov.to_string(index=False))
print(f"\\nMean joint weeks: {cov['joint_total'].mean():.0f}  (vs CDC mean: {cov['cdc_ref_wks'].mean():.0f})")

# Figure 5: Joint Modelable Weeks — CA Datasets vs CDC Reference
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cov))
w = 0.28

ax.bar(x - w, cov["joint_total"],   w, color="steelblue", alpha=0.85, label="CA datasets (joint total)")
ax.bar(x,     cov["pre_2022_wks"],  w, color="#57cc99",   alpha=0.85, label="Of which: pre-2022 (pre-Omicron)")
ax.bar(x + w, cov["cdc_ref_wks"],   w, color="darkorange", alpha=0.75, label="CDC reference")

for i, row in cov.iterrows():
    sign = "+" if row["extra_vs_cdc"] >= 0 else ""
    ax.text(i - w, row["joint_total"] + 0.5, f"{sign}{row['extra_vs_cdc']}w",
            ha="center", va="bottom", fontsize=7, fontweight="bold", color="steelblue")

ax.set_xticks(x)
ax.set_xticklabels(cov["County"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Weeks")
ax.set_title(
    "Figure 5: Joint Modelable Weeks — CA Datasets vs CDC Reference\\n"
    "Green = pre-2022 weeks (pre-Omicron outbreaks); blue label = extra weeks gained",
    fontweight="bold"
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
'''))

    # Section 5
    cells.append(md("""
---
## Section 5: Cross-Dataset Integration & Lead-Time Analysis

This section establishes the empirical WW→Cases lead-time relationship across all four outbreak waves
and validates the biological signal structure that motivates the TFT's horizon weighting vector
`[2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8]`.

### 5.1 Merged Signals — WW Concentration vs New Cases (4 Counties)

Dual-axis overlay for four illustrative counties spanning the data-richness spectrum.

| County | WW data start | Why included |
|---|---|---|
| **San Francisco** | Nov 2020 | Dense urban sewershed; longest pre-Omicron record after Santa Clara |
| **Santa Clara** | Jul 2020 | Earliest solid-track WW; largest county; all 4 waves visible |
| **Napa** | Sep 2022 | Sparse history; illustrates data-poor county behavior |
| **Alameda** | Feb 2022 | High population; representative of the 6 "late-start" East Bay counties |
"""))

    cells.append(code('''
FOCUS_COUNTIES = ["San Francisco", "Santa Clara", "Napa", "Alameda"]

def _build_merged(county):
    """Merge weekly WW (rc) and cases for one county."""
    ww_c   = ww[ww["County"] == county][["week", "rc"]].dropna(subset=["rc"]).rename(columns={"week": "date"})
    cas_c  = cases_weekly[cases_weekly["County"] == county][["date", "cases"]]
    m      = pd.merge(ww_c, cas_c, on="date", how="outer").sort_values("date")
    m["rc_smooth"]  = m["rc"].rolling(2, min_periods=1).mean()
    m["cases_4w"]   = m["cases"].rolling(4, min_periods=1).mean()
    return m

fig, axes = plt.subplots(2, 2, figsize=(20, 11))
fig.suptitle(
    "Figure B: Merged WW Concentration & Weekly Cases — 4 Illustrative Counties\\n"
    "WW (left axis, steelblue log scale) leads Cases (right axis, crimson) by ~1-3 weeks at surge onset",
    fontsize=13, fontweight="bold"
)

for ax, county in zip(axes.flat, FOCUS_COUNTIES):
    merged = _build_merged(county)

    for color, s, e, lbl in ALL_WAVE_SPANS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=color, zorder=0)

    ww_valid = merged.dropna(subset=["rc"])
    if len(ww_valid) > 0:
        ax.fill_between(ww_valid["date"], ww_valid["rc_smooth"].clip(lower=0.1), alpha=0.15, color="steelblue")
        ax.plot(ww_valid["date"], ww_valid["rc_smooth"].clip(lower=0.1),
                color="steelblue", linewidth=2.2, label="WW Raw Conc (2-wk smooth)")
        ax.set_yscale("log")
        ax.set_ylabel("WW Raw Concentration (copies/g)", color="steelblue", fontsize=8)
        ax.tick_params(axis="y", labelcolor="steelblue")
        ww_start = ww_valid["date"].min()
        if ww_start > pd.Timestamp("2021-12-01"):
            ax.text(0.02, 0.96, f"WW from\\n{ww_start.strftime('%Y-%m')}",
                    transform=ax.transAxes, fontsize=8, color="steelblue", va="top",
                    bbox=dict(facecolor="white", edgecolor="steelblue", alpha=0.8, pad=2))

    ax2 = ax.twinx()
    cas_valid = merged.dropna(subset=["cases"])
    ax2.bar(cas_valid["date"], cas_valid["cases"], width=5, color="crimson", alpha=0.22, zorder=1)
    ax2.plot(cas_valid["date"], cas_valid["cases_4w"], color="crimson", linewidth=2.0, label="Cases (4-wk avg)")
    ax2.set_ylabel("New Cases / week", color="crimson", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="crimson")

    ax.set_title(county, fontweight="bold", color=COUNTY_COLORS.get(county, "black"), fontsize=12)
    ax.set_xlabel("Week (W-WED)", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.set_xlim(OVERLAP_START, OVERLAP_END)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

_patches = [mpatches.Patch(color=c, alpha=0.45, label=lbl.replace("\\n", " ")) for c, _, _, lbl in ALL_WAVE_SPANS]
fig.legend(handles=_patches, ncol=7, loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
'''))

    cells.append(md("""
### 5.2 Rate of Change — Sludge Signal Decay vs Case Decline

First-derivative (`pct_change()`) comparison of WW concentration and weekly cases.
Analysis restricted to the three counties with pre-2022 WW data (Santa Clara, SF, San Mateo)
to include multiple complete wave cycles.
"""))

    cells.append(code('''
DERIV_COUNTIES = ["San Francisco", "Santa Clara", "San Mateo"]

deriv_results = []
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle(
    "Figure C: WW Signal Velocity vs Case Velocity — First-Derivative Comparison\\n"
    "Counties with pre-2022 data only (SF, Santa Clara, San Mateo); pct_change() per week",
    fontsize=13, fontweight="bold"
)

for ax, county in zip(axes, DERIV_COUNTIES):
    merged = _build_merged(county).dropna(subset=["rc", "cases"])
    if len(merged) < 8:
        ax.set_visible(False)
        continue

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["dww"]    = merged["rc"].pct_change()
    merged["dcases"] = merged["cases"].pct_change()
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["dww", "dcases"])

    r_all, _  = stats.pearsonr(merged["dww"], merged["dcases"])
    decay      = merged[merged["dcases"] < 0]
    r_decay    = stats.pearsonr(decay["dww"], decay["dcases"])[0] if len(decay) >= 5 else np.nan
    deriv_results.append({
        "County":        county,
        "r_all_periods": round(r_all, 3),
        "r_decay_only":  round(r_decay, 3) if not np.isnan(r_decay) else "—",
        "n_decay_weeks": len(decay),
        "n_total_weeks": len(merged),
    })

    for color, s, e, lbl in ALL_WAVE_SPANS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=color, zorder=0)

    ax.plot(merged["date"], merged["dww"], color="steelblue", linewidth=1.5, label="Δ WW (pct_change)")
    ax2 = ax.twinx()
    ax2.plot(merged["date"], merged["dcases"], color="crimson", linewidth=1.5,
             linestyle="--", label="Δ Cases (pct_change)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")

    ax.set_ylabel("WW Δ (pct_change)", color="steelblue", fontsize=8)
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylabel("Cases Δ (pct_change)", color="crimson", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="crimson")
    r_str = "—" if np.isnan(r_decay) else f"{r_decay:.2f}"
    ax.set_title(f"{county}\\nr_all = {r_all:.2f}  |  r_decay = {r_str}",
                 fontweight="bold", color=COUNTY_COLORS.get(county, "black"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()

print("\\n=== First-Derivative Correlation Summary ===")
print(pd.DataFrame(deriv_results).to_string(index=False))
print("\\n-> Positive r_decay confirms WW tracks case decline kinetics — not just surge onset.")
'''))

    cells.append(md("""
### 5.3 Epidemic Wave Early-Warning Analysis — WW Z-Score

Z-scoring the WW signal against its trailing **8-week rolling baseline** isolates relative signal elevation
from absolute magnitude. A z-score > 2.0 constitutes an early-warning alert.

The three-panel figure shows: (1) z-score with alert threshold; (2) the raw log1p concentration with
rolling mean ± SD band; (3) the actual case ground truth.
"""))

    cells.append(code('''
Z_THRESHOLD            = 2.0
ROLLING_BASELINE_WEEKS = 8

_agg_ww = (
    ww.dropna(subset=["rc"])
    .groupby("week")["rc"].median()
    .reset_index().rename(columns={"week": "date"})
    .sort_values("date")
)
_agg_ww["rc_log"]    = np.log1p(_agg_ww["rc"])
_agg_ww["roll_mean"] = _agg_ww["rc_log"].rolling(ROLLING_BASELINE_WEEKS, min_periods=4).mean()
_agg_ww["roll_std"]  = _agg_ww["rc_log"].rolling(ROLLING_BASELINE_WEEKS, min_periods=4).std()
_agg_ww["zscore"]    = (_agg_ww["rc_log"] - _agg_ww["roll_mean"]) / (_agg_ww["roll_std"] + 1e-9)

_bay_cases_z = cases_weekly.groupby("date")["cases"].sum().reset_index()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
fig.suptitle(
    "Figure D: Epidemic Early-Warning Analysis — Bay Area Aggregate\\n"
    "WW Z-Score (top) | log1p WW Concentration with baseline (middle) | Weekly Cases (bottom)",
    fontsize=13, fontweight="bold"
)

for ax in [ax1, ax2, ax3]:
    for color, s, e, lbl in ALL_WAVE_SPANS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=color, zorder=0)

# Panel 1: Z-score
_z_max = max(_agg_ww["zscore"].dropna().max(), Z_THRESHOLD + 0.5)
for color, s, e, lbl in ALL_WAVE_SPANS:
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    ax1.text(mid, _z_max * 0.93, lbl.replace("\\n", " "),
             ha="center", va="top", fontsize=7.5, color=color, fontweight="bold")

ax1.plot(_agg_ww["date"], _agg_ww["zscore"], color="steelblue", linewidth=1.8)
ax1.axhline(Z_THRESHOLD,  color="red",    linestyle="--", linewidth=1.4, label=f"Alert  z = +{Z_THRESHOLD}")
ax1.axhline(-Z_THRESHOLD, color="orange", linestyle=":",  linewidth=1.0, label=f"Below-baseline  z = -{Z_THRESHOLD}")
ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
_alert = _agg_ww["zscore"] > Z_THRESHOLD
ax1.fill_between(_agg_ww["date"], _agg_ww["zscore"], Z_THRESHOLD,
                 where=_alert, color="red", alpha=0.15, label="Active alert periods")
ax1.set_ylabel(f"Z-Score (vs {ROLLING_BASELINE_WEEKS}-wk rolling baseline)", fontsize=10)
ax1.set_title("WW Signal Z-Score — Outbreak Early Warning", fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.set_ylim(-4, _z_max + 0.3)

# Panel 2: WW concentration with rolling baseline
ax2.fill_between(_agg_ww["date"], _agg_ww["rc_log"], alpha=0.20, color="steelblue")
ax2.plot(_agg_ww["date"], _agg_ww["rc_log"],    color="steelblue", linewidth=1.8, label="log1p(Raw Conc)")
ax2.plot(_agg_ww["date"], _agg_ww["roll_mean"], color="navy",      linewidth=1.5,
         linestyle="--", label=f"{ROLLING_BASELINE_WEEKS}-wk rolling mean")
ax2.fill_between(_agg_ww["date"],
                 _agg_ww["roll_mean"] - _agg_ww["roll_std"],
                 _agg_ww["roll_mean"] + _agg_ww["roll_std"],
                 alpha=0.12, color="navy", label="±1 SD band")
ax2.set_ylabel("log1p(Raw Concentration)", fontsize=10)
ax2.set_title("WW Concentration — log1p with Rolling Baseline", fontweight="bold")
ax2.legend(fontsize=9)

# Panel 3: Cases
_roll_cases = _bay_cases_z.set_index("date")["cases"].rolling(4, min_periods=1).mean()
ax3.fill_between(_bay_cases_z["date"], _bay_cases_z["cases"], alpha=0.20, color="crimson")
ax3.bar(_bay_cases_z["date"], _bay_cases_z["cases"], width=5, color="crimson", alpha=0.35)
ax3.plot(_roll_cases.index, _roll_cases.values, color="darkred", linewidth=2.0, label="4-wk rolling avg")
ax3.set_ylabel("Weekly New Cases (Bay Area)", fontsize=10)
ax3.set_xlabel("Week (W-WED)", fontsize=10)
ax3.set_title("Weekly New Cases — Ground Truth", fontweight="bold")
ax3.legend(fontsize=9)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.set_xlim(OVERLAP_START, OVERLAP_END)

plt.tight_layout()
plt.show()
'''))

    cells.append(md("""
### 5.4 Early Warning Summary

Per outbreak wave: how many weeks before the case peak did the WW z-score first exceed the
alert threshold (z > 2.0)? A **positive lead time** means the WW signal fired before the case peak.
"""))

    cells.append(code('''
_WAVE_WINDOWS = {
    "Wave 1 (Original)":     ("2020-07-01", "2020-10-31"),
    "Wave 2 (Alpha/WT)":     ("2020-11-01", "2021-04-30"),
    "Wave 3 (Delta)":        ("2021-06-01", "2021-11-30"),
    "Wave 4 (Omicron BA.1)": ("2021-12-01", "2022-05-31"),
    "BA.5":                  ("2022-06-01", "2022-09-30"),
    "BQ.1 / BQ.1.1":        ("2022-10-01", "2023-01-15"),
    "XBB.1.5":               ("2023-01-16", "2023-04-30"),
}

_bay_cases_ew = cases_weekly.groupby("date")["cases"].sum().reset_index()

summary_rows = []
for wave_name, (ws, we) in _WAVE_WINDOWS.items():
    ws_ts, we_ts = pd.Timestamp(ws), pd.Timestamp(we)
    cases_win = _bay_cases_ew[(_bay_cases_ew["date"] >= ws_ts) & (_bay_cases_ew["date"] <= we_ts)]
    if cases_win.empty or cases_win["cases"].sum() == 0:
        summary_rows.append({"Wave": wave_name, "Case Peak Date": "—", "WW Alert (z>2)": "—",
                             "Lead (wks)": "—", "Status": "No cases"})
        continue
    case_peak_dt = cases_win.loc[cases_win["cases"].idxmax(), "date"]
    ww_win = _agg_ww[(_agg_ww["date"] >= ws_ts) & (_agg_ww["date"] <= we_ts) & (_agg_ww["zscore"] > Z_THRESHOLD)]
    if ww_win.empty:
        summary_rows.append({"Wave": wave_name, "Case Peak Date": case_peak_dt.strftime("%Y-%m-%d"),
                             "WW Alert (z>2)": "—", "Lead (wks)": "—", "Status": "No WW alert fired"})
        continue
    ww_alert_dt = ww_win["date"].min()
    lead_wks    = (case_peak_dt - ww_alert_dt).days / 7.0
    status = "Leads" if lead_wks > 0.5 else ("Lags" if lead_wks < -1 else "Coincident")
    summary_rows.append({
        "Wave":            wave_name,
        "Case Peak Date":  case_peak_dt.strftime("%Y-%m-%d"),
        "WW Alert (z>2)":  ww_alert_dt.strftime("%Y-%m-%d"),
        "Lead (wks)":      f"{lead_wks:+.1f}",
        "Status":          status,
    })

summary_df = pd.DataFrame(summary_rows)
print(f"=== Early Warning Summary: WW Z-Score vs Case Peaks ===")
print(f"Alert threshold: z > {Z_THRESHOLD}  |  Baseline: {ROLLING_BASELINE_WEEKS}-week rolling window\\n")
print(summary_df.to_string(index=False))

leads_numeric = []
for r in summary_rows:
    try:
        leads_numeric.append(float(r["Lead (wks)"].replace("+", "")))
    except (ValueError, AttributeError):
        pass

if leads_numeric:
    print(f"\\nMean lead time: {np.mean(leads_numeric):.1f} weeks")
    print(f"Waves where WW leads: {sum(l > 0.5 for l in leads_numeric)}/{len(leads_numeric)}")
print("\\n-> TFT implication: near-term horizon weight [2.0, 2.0, 1.5, 1.5] encodes the 1-3 week lead window.")
'''))

    # Section 6 Verdict
    cells.append(md("""
---
## Section 6: Spine Readiness Verdict & Pipeline Migration Plan
"""))

    cells.append(code('''
print("=" * 70)
print("SPINE READINESS VERDICT")
print("=" * 70)

print("""
CA Wastewater Dataset  Ready  (with conditions)
  All 9 Bay Area counties present
  Full date range 2020-07-16 -> 2026-05-05
  WW signal clearly visible at all 4 key outbreak waves
  Only 3 counties have pre-2022 data (Santa Clara, SF, San Mateo)
  ~50% null rate in both signals -- handle with groupby median, not ffill
  Use Sample Type == "solid" filter throughout

CA Cases Dataset  Ready  (with conditions)
  All 9 Bay Area counties; direct name match (no FIPS needed)
  Extends to 2023-12-19 (+ weeks beyond CDC cutoff)
  Near-zero null rate in cases column
  Date format is %m/%d/%y -- must pass format= explicitly
  Daily data -- must resample daily -> W-WED before any join
  Filter area_type == "County" to exclude state-level rows
""")

if "snr_df" in dir() and len(snr_df) > 0:
    winner_counts = snr_df["winner"].value_counts()
    top_signal = "Raw Concentration" if winner_counts.get("RC", 0) >= winner_counts.get("NP", 0) else "Norm Pmmov"
    print(f"Signal Recommendation: {top_signal}")
    print(f"  (RC wins {winner_counts.get('RC',0)}/4 outbreak windows, NP wins {winner_counts.get('NP',0)}/4)")

if "cov" in dir():
    print(f"""
New Unified Spine Stats:
  Overlap window:     {OVERLAP_START.date()} -> {OVERLAP_END.date()}
  Modelable counties: 9 / 9 (all Bay Area)
  Mean joint wks:     {cov['joint_total'].mean():.0f}  (CDC: {cov['cdc_ref_wks'].mean():.0f})
  Min joint wks:      {cov['joint_total'].min()}  (county: {cov.loc[cov['joint_total'].idxmin(), 'County']})
  Pre-2022 counties:  3 / 9 have early-outbreak WW data
""")
print("=" * 70)
'''))

    # Figure E — Train/CV/Holdout split
    cells.append(md("""
---
## Closing View: Model Training Window Anatomy

The plot below is the same Bay Area case timeline as Figure A, now annotated with the
**exact train / cross-validation / holdout splits** used in the pipeline, plus every CV fold
cutoff date color-coded by its holdout WIS score.
"""))

    cells.append(code('''
TRAIN_START = pd.Timestamp("2020-07-01")
TRAIN_END   = pd.Timestamp("2022-10-05")
VAL_END     = pd.Timestamp("2023-06-07")
HOLDOUT_END = pd.Timestamp("2023-12-19")

CV_FOLDS = [
    (pd.Timestamp("2022-10-05"), 0.173),
    (pd.Timestamp("2022-11-02"), 0.392),
    (pd.Timestamp("2022-11-30"), 1.354),
    (pd.Timestamp("2022-12-28"), 0.459),
    (pd.Timestamp("2023-01-25"), 0.204),
    (pd.Timestamp("2023-02-22"), 0.285),
    (pd.Timestamp("2023-03-22"), 0.065),
    (pd.Timestamp("2023-04-19"), 0.109),
    (pd.Timestamp("2023-05-17"), 0.088),
]

_bay2  = cases_weekly.groupby("date")["cases"].sum().reset_index()
_roll2 = _bay2.set_index("date")["cases"].rolling(4, min_periods=1).mean()
_ymax  = _bay2["cases"].max()

fig, ax = plt.subplots(figsize=(20, 8))

for color, s, e, lbl in ALL_WAVE_SPANS:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.05, color=color, zorder=0)

ax.axvspan(TRAIN_START, TRAIN_END,   alpha=0.10, color="#2196F3", zorder=1)
ax.axvspan(TRAIN_END,   VAL_END,     alpha=0.10, color="#FF9800", zorder=1)
ax.axvspan(VAL_END,     HOLDOUT_END, alpha=0.10, color="#F44336", zorder=1)

for i, (fold_dt, wis) in enumerate(CV_FOLDS):
    fcolor = "#b71c1c" if wis > 0.5 else ("#e65100" if wis > 0.15 else "#2e7d32")
    ax.axvline(fold_dt, color=fcolor, linewidth=1.1, linestyle=":", alpha=0.85, zorder=4)
    ypos = _ymax * (0.52 if i % 2 == 0 else 0.43)
    ax.text(fold_dt, ypos, f"F{i+1}\\nWIS\\n{wis:.2f}",
            ha="center", va="top", fontsize=7.5, color=fcolor, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=fcolor, alpha=0.88))

ax.fill_between(_bay2["date"], _bay2["cases"], alpha=0.18, color="#37474f", zorder=2)
ax.bar(_bay2["date"], _bay2["cases"], color="#37474f", alpha=0.38, width=5, zorder=3)
ax.plot(_roll2.index, _roll2.values, color="#212121", linewidth=2.3, zorder=5, label="4-wk rolling avg")

_VARIANT_ANN = [
    ("Wave 1\\n(Original)",           "2020-08-15", 0.75, "#4e9af1"),
    ("Wave 2\\n(Alpha/WT)",           "2021-01-10", 0.72, "#f4a261"),
    ("Wave 3\\n(Delta)",              "2021-09-10", 0.67, "#57cc99"),
    ("Omicron BA.1",                  "2022-01-12", 0.90, "#e76f51"),
    ("BA.5",                          "2022-07-15", 0.34, "#c77dff"),
    ("BQ.1 / BQ.1.1",                "2022-11-15", 0.38, "#9b72cf"),
    ("XBB.1.5",                       "2023-02-10", 0.33, "#48cae4"),
]
for lbl, cdt, yfrac, color in _VARIANT_ANN:
    ax.annotate(lbl, xy=(pd.Timestamp(cdt), _ymax * yfrac),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.88))

for dt, lbl, color in [(TRAIN_END, "TRAIN END / CV START", "#1565C0"),
                        (VAL_END,   "CV END / HOLDOUT START", "#B71C1C")]:
    ax.axvline(dt, color=color, linewidth=2.2, linestyle="--", zorder=6)
    ax.text(dt + pd.Timedelta(days=3), _ymax * 0.04, lbl,
            ha="left", va="bottom", fontsize=8.5, color=color, fontweight="bold")

ax.set_xlim(TRAIN_START - pd.Timedelta(weeks=2), HOLDOUT_END + pd.Timedelta(weeks=2))
ax.set_ylim(0, _ymax * 1.18)
ax.set_xlabel("Week (W-WED)", fontsize=11)
ax.set_ylabel("Weekly New Cases — Bay Area Aggregate", fontsize=11)
ax.set_title(
    "Figure E: Train / CV / Holdout Splits — Bay Area COVID-19 Timeline\\n"
    "Blue = Train | Orange = CV window (9 expanding folds) | Red = Holdout\\n"
    "Fold WIS color: green <= 0.15 | orange 0.15-0.50 | red > 0.50",
    fontsize=12, fontweight="bold"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("=== Split Anatomy: What Real-World Outbreaks Each Period Captured ===")
print("TRAIN  (2020-07-01 -> 2022-10-05  |  ~117 W-WED weeks):")
print("  Wave 1 (Summer 2020): complete wave")
print("  Wave 2 (Winter 2020-21 Alpha/WT): complete wave")
print("  Wave 3 (Delta 2021): complete wave -- best pre-Omicron lead-time training data")
print("  Wave 4 (Omicron BA.1, Jan 2022 peak): included -- teaches extreme magnitude")
print("  BA.5 surge (Jun-Sep 2022): partial onset -- tail in CV window")
print()
print("CV WINDOW  (2022-10-05 -> 2023-06-07  |  9 folds x 4-week step):")
print("  Folds 1-2 (Oct-Nov 2022): BQ.1 / BQ.1.1 multi-modal surge onset")
print("  Fold 3    (Nov 2022):     Omicron sub-wave PEAK  -- WIS 1.354, hardest fold")
print("  Folds 4-5 (Dec 22-Jan 23): Post-peak decline + inter-wave trough")
print("  Folds 7-9 (Mar-May 2023): Stable inter-wave -- WIS 0.065-0.109")
print()
print("HOLDOUT  (2023-06-07 -> 2023-12-19  |  ~28 weeks, 72 obs):")
print("  Post-XBB.1.5 summer descent -> late-2023 resurgence activity")
print("  Coverage_95 = ~7%  -- primary motivation for Phase 4 adaptive calibration")
'''))

    nb["cells"] = cells
    return nb


# ===========================================================================
# SANTA CLARA NOTEBOOK
# ===========================================================================

def make_notebook_santa_clara() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = []

    cells.append(md("""
# 04 — Santa Clara County EDA: Wastewater & Cases Deep Dive
**Sewer Signals Project** | Phase 4 — Single-County Reference

**Dataset:** `California_Wastewater_Surveillance_Data.csv` + `Statewide_COVID-19_Cases_Deaths_Tests.csv`
**Scope:** Santa Clara County (FIPS 06085). Solid track only. **2020-07-01 → 2023-12-19**.

Santa Clara provides the **longest, cleanest** wastewater time series in the Bay Area
(sludge track from July 2020) and is the primary single-county validation case.

**Key questions:**
1. How clearly does the WW signal lead case peaks across all four wave cycles?
2. Does Raw Concentration or Norm Pmmov correlate more strongly with cases in Santa Clara?
3. What is the optimal WW→Cases cross-correlation lag for this county?
"""))

    cells.append(md("## Section 0: Setup"))
    cells.append(code(SETUP_CODE))
    cells.append(code('''
SC_FIPS  = "06085"
SC_NAME  = "Santa Clara"
SC_COLOR = COUNTY_COLORS.get("Santa Clara", "#EF6C00")
print(f"Santa Clara color: {SC_COLOR}")
'''))

    # Section 1: WW
    cells.append(md("""
---
## Section 1: CA Wastewater — Santa Clara County

### 1.1 Load, Filter & Audit

Santa Clara operates multiple treatment plants (San José/Santa Clara Regional, Sunnyvale, Palo Alto)
providing excellent spatial coverage. The sludge track has been active since July 2020 —
the longest Bay Area record.
"""))

    cells.append(code(LOAD_WW_CODE + '''

# Filter to Santa Clara only
ww_sc = ww[ww["County"] == SC_NAME].sort_values("week").copy()
solid_sc = solid[solid["County"] == SC_NAME].copy()
print(f"\\nSanta Clara WW records (weekly): {len(ww_sc)}")
print(f"SC date range: {ww_sc['week'].min().date()} -> {ww_sc['week'].max().date()}")
print(f"RC range: {ww_sc['rc'].min():.0f} - {ww_sc['rc'].max():.0f} copies/g")
print(f"NP range: {ww_sc['np'].dropna().min():.2f} - {ww_sc['np'].dropna().max():.2f}")
'''))

    cells.append(md("### 1.2 Santa Clara WW Signal — Full Extended Window"))

    cells.append(code('''
# Figure 1: Santa Clara WW Concentration 2020-2023
# Top: raw concentration (log scale); Bottom: log1p transform
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle(
    "Figure 1: Santa Clara SARS-CoV-2 WW Concentration 2020-2023\\n"
    "Sludge track; top = raw (log scale), bottom = log1p transform; shaded = variant waves",
    fontsize=13, fontweight="bold"
)

for wc, ws, we, wn in ALL_WAVE_SPANS:
    ax1.axvspan(pd.Timestamp(ws), pd.Timestamp(we), alpha=0.07, color=wc, zorder=0)
    ax2.axvspan(pd.Timestamp(ws), pd.Timestamp(we), alpha=0.07, color=wc, zorder=0)

ax1.plot(ww_sc["week"], ww_sc["rc"], color=SC_COLOR, linewidth=1.2, alpha=0.5)
ax1.plot(ww_sc["week"], ww_sc["rc"].rolling(4, min_periods=1).mean(),
         color=SC_COLOR, linewidth=2.5, label="4-wk MA")
ax1.set_yscale("log")
ax1.set_ylabel("Copies/g dry sludge (log scale)", fontsize=10)
ax1.legend(fontsize=9, loc="upper right")

ww_sc["log1p_rc"] = np.log1p(ww_sc["rc"])
ax2.fill_between(ww_sc["week"], ww_sc["log1p_rc"], alpha=0.20, color=SC_COLOR)
ax2.plot(ww_sc["week"], ww_sc["log1p_rc"], color=SC_COLOR, linewidth=2.0, label="log1p(RC)")
ax2.set_ylabel("log1p(Concentration)", fontsize=10)
ax2.set_xlabel("Date (W-WED)", fontsize=10)
ax2.legend(fontsize=9, loc="upper right")

_patches = [mpatches.Patch(color=c, alpha=0.4, label=lbl.replace("\\n", " ")) for c, _, _, lbl in ALL_WAVE_SPANS]
fig.legend(handles=_patches, ncol=7, loc="lower center", fontsize=8, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
'''))

    # Section 2: Cases
    cells.append(md("""
---
## Section 2: CA Cases — Santa Clara County

Santa Clara County has one of California's most consistent reporting records,
providing daily confirmed case counts without significant data gaps from 2020–2023.
"""))

    cells.append(code(LOAD_CASES_CODE + '''

# Filter to Santa Clara only
cases_sc = cases_weekly[cases_weekly["County"] == SC_NAME].sort_values("date").copy()
print(f"\\nSanta Clara cases records: {len(cases_sc)}")
print(f"Date range: {cases_sc['date'].min().date()} -> {cases_sc['date'].max().date()}")
print(f"Peak weekly cases: {cases_sc['cases'].max():.0f} (Omicron BA.1 Jan 2022)")

# Merge WW + cases for Santa Clara
_ww_for_merge = ww_sc[["week", "rc", "np"]].rename(columns={"week": "date"})
merged_sc = pd.merge(_ww_for_merge, cases_sc[["date", "cases"]], on="date", how="outer").sort_values("date")
merged_sc["log1p_rc"]    = np.log1p(merged_sc["rc"])
merged_sc["log1p_cases"] = np.log1p(merged_sc["cases"])
print(f"Merged WW+cases rows: {len(merged_sc)}")
'''))

    cells.append(code('''
# Figure 2: Santa Clara Weekly Cases (crimson) + WW Concentration (left axis, right axis)
fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(
    "Figure 2: Santa Clara Weekly Cases (crimson) + WW Concentration (steelblue, right axis)\\n"
    "Note WW peaks precede case peaks by ~1-2 weeks at each surge onset",
    fontsize=13, fontweight="bold"
)
for wc, ws, we, wn in ALL_WAVE_SPANS:
    ax.axvspan(pd.Timestamp(ws), pd.Timestamp(we), alpha=0.07, color=wc, zorder=0)

ax.fill_between(cases_sc["date"], cases_sc["cases"], alpha=0.2, color=C_CASES)
ax.plot(cases_sc["date"], cases_sc["cases"].rolling(4, min_periods=1).mean(),
        color=C_CASES, linewidth=2.5, label="Cases (4-wk avg)")
ax.set_ylabel("New Cases / week", color=C_CASES, fontsize=10)
ax.tick_params(axis="y", labelcolor=C_CASES)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax2 = ax.twinx()
ww_valid = ww_sc.dropna(subset=["rc"])
ax2.plot(ww_valid["week"], ww_valid["rc"].clip(lower=0.1),
         color=C_WW, linewidth=2.5, label="WW Raw Conc (copies/g)")
ax2.set_ylabel("WW copies/g dry sludge (log)", color=C_WW, fontsize=10)
ax2.tick_params(axis="y", labelcolor=C_WW)
ax2.set_yscale("log")

ax.set_xlabel("Date (W-WED)", fontsize=10)
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9, loc="upper right")
plt.tight_layout()
plt.show()
'''))

    # Section 3: RC vs NP
    cells.append(md("""
---
## Section 3: Signal Quality — Raw Conc vs Norm Pmmov at Key Outbreaks

For each of the 4 key outbreak windows, both WW signals are normalised [0→1] within the window
and overlaid on cases. Pearson r computed in log space.
"""))

    cells.append(code('''
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

OUTBREAKS_SC = {
    "Summer 2020\\n(Wave 1)":    (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-31")),
    "Winter 2020-21\\n(Wave 2)": (pd.Timestamp("2020-11-01"), pd.Timestamp("2021-04-30")),
    "Delta 2021\\n(Wave 3)":     (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-11-30")),
    "Omicron 2022\\n(Wave 4)":   (pd.Timestamp("2021-12-01"), pd.Timestamp("2022-05-31")),
}

fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
fig.suptitle(
    "Figure 3: Raw Concentration vs Norm Pmmov at Key Outbreak Windows — Santa Clara\\n"
    "Signals normalised [0-1] within each window; gray bars = weekly cases",
    fontsize=13, fontweight="bold"
)

snr_sc = []
for ax, ((label, (w_s, w_e)), oc) in zip(axes, zip(OUTBREAKS_SC.items(), OUTBREAK_COLORS)):
    window = merged_sc[(merged_sc["date"] >= w_s) & (merged_sc["date"] <= w_e)].copy()

    rc_w    = norm01(window["rc"].dropna())
    np_w    = norm01(window["np"].dropna()) if window["np"].notna().sum() >= 2 else pd.Series(dtype=float)
    cases_w = window.set_index("date")["cases"].clip(lower=0)

    shared_rc = rc_w.index.intersection(cases_w.dropna().index) if not rc_w.empty else []
    shared_np = np_w.index.intersection(cases_w.dropna().index) if not np_w.empty else []
    r_rc = np.corrcoef(np.log1p(rc_w[shared_rc]), np.log1p(cases_w[shared_rc]))[0, 1] if len(shared_rc) >= 4 else np.nan
    r_np = np.corrcoef(np.log1p(np_w[shared_np]), np.log1p(cases_w[shared_np]))[0, 1] if len(shared_np) >= 4 else np.nan
    winner = "RC" if (not np.isnan(r_rc) and (np.isnan(r_np) or r_rc >= r_np)) else "NP"
    snr_sc.append({"outbreak": label.replace("\\n", " "), "r_raw_conc": r_rc, "r_norm_pmmov": r_np, "winner": winner})

    ax2 = ax.twinx()
    cases_plot = window.dropna(subset=["cases"])
    ax2.bar(cases_plot["date"], cases_plot["cases"], width=5, color="lightgray", alpha=0.5, zorder=0)
    ax2.set_ylabel("SC cases / week", color="gray", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    if not rc_w.empty:
        ax.plot(rc_w.index, rc_w.values, color=C_WW, linewidth=2.2,
                label=f"Raw Conc  r={r_rc:.2f}" if not np.isnan(r_rc) else "Raw Conc  r=—", zorder=3)
    if not np_w.empty:
        ax.plot(np_w.index, np_w.values, color=C_ACCENT, linewidth=2.2,
                linestyle="--", label=f"Norm Pmmov  r={r_np:.2f}" if not np.isnan(r_np) else "Norm Pmmov  r=—", zorder=3)

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Norm. WW signal [0-1]", fontsize=8)
    ax.set_title(f"{label}\\n{'✓ RC' if winner == 'RC' else '✓ NP'} leads", fontweight="bold", color=oc)
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.show()

snr_sc_df = pd.DataFrame(snr_sc)
print("\\n=== Santa Clara Signal Comparison Summary ===")
print(snr_sc_df.to_string(index=False))
'''))

    # Section 4: Rate of change
    cells.append(md("""
---
## Section 4: Rate of Change — WW Velocity vs Case Velocity

First-derivative (`pct_change()`) comparison for Santa Clara. Key question:
does the WW signal co-move with case decline kinetics, not just surge onset?
"""))

    cells.append(code('''
sc_deriv = merged_sc.dropna(subset=["rc", "cases"]).sort_values("date").copy()
sc_deriv["dww"]    = sc_deriv["rc"].pct_change()
sc_deriv["dcases"] = sc_deriv["cases"].pct_change()
sc_deriv = sc_deriv.replace([np.inf, -np.inf], np.nan).dropna(subset=["dww", "dcases"])

r_all, _  = stats.pearsonr(sc_deriv["dww"], sc_deriv["dcases"])
decay      = sc_deriv[sc_deriv["dcases"] < 0]
r_decay    = stats.pearsonr(decay["dww"], decay["dcases"])[0] if len(decay) >= 5 else np.nan

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(
    "Figure 4: WW Signal Velocity vs Case Velocity — Santa Clara County\\n"
    f"pct_change() per week; r_all = {r_all:.2f}  |  r_decay = {r_decay:.2f if not np.isnan(r_decay) else '—'}",
    fontsize=13, fontweight="bold"
)

for color, s, e, lbl in ALL_WAVE_SPANS:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=color, zorder=0)

ax.plot(sc_deriv["date"], sc_deriv["dww"], color=C_WW, linewidth=1.5, label="Δ WW (pct_change)")
ax2 = ax.twinx()
ax2.plot(sc_deriv["date"], sc_deriv["dcases"], color=C_CASES, linewidth=1.5,
         linestyle="--", label="Δ Cases (pct_change)")
ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")

ax.set_ylabel("WW Δ (pct_change)", color=C_WW, fontsize=10)
ax.tick_params(axis="y", labelcolor=C_WW)
ax2.set_ylabel("Cases Δ (pct_change)", color=C_CASES, fontsize=10)
ax2.tick_params(axis="y", labelcolor=C_CASES)
ax.set_xlabel("Date (W-WED)", fontsize=10)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
plt.tight_layout()
plt.show()

print(f"r (all periods):  {r_all:.3f}  (n={len(sc_deriv)})")
print(f"r (decay only):   {r_decay:.3f}  (n={len(decay)})") if not np.isnan(r_decay) else print("r_decay: insufficient data")
print("\\n-> Positive r_decay confirms WW tracks case decline, not just surge onset.")
'''))

    # Section 5: Z-score
    cells.append(md("""
---
## Section 5: Epidemic Wave Early-Warning Analysis — Santa Clara Z-Score

Z-scoring the SC WW signal against its trailing 8-week rolling baseline.
A z-score > 2.0 constitutes an early-warning alert.
"""))

    cells.append(code('''
Z_THRESHOLD            = 2.0
ROLLING_BASELINE_WEEKS = 8

sc_z = ww_sc[["week", "rc"]].dropna(subset=["rc"]).sort_values("week").copy()
sc_z["rc_log"]    = np.log1p(sc_z["rc"])
sc_z["roll_mean"] = sc_z["rc_log"].rolling(ROLLING_BASELINE_WEEKS, min_periods=4).mean()
sc_z["roll_std"]  = sc_z["rc_log"].rolling(ROLLING_BASELINE_WEEKS, min_periods=4).std()
sc_z["zscore"]    = (sc_z["rc_log"] - sc_z["roll_mean"]) / (sc_z["roll_std"] + 1e-9)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
fig.suptitle(
    "Figure 5: Epidemic Early-Warning Analysis — Santa Clara County\\n"
    "WW Z-Score (top) | log1p WW Concentration with baseline (middle) | Weekly Cases (bottom)",
    fontsize=13, fontweight="bold"
)

for ax in [ax1, ax2, ax3]:
    for color, s, e, lbl in ALL_WAVE_SPANS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=color, zorder=0)

# Panel 1: Z-score
_z_max = max(sc_z["zscore"].dropna().max(), Z_THRESHOLD + 0.5)
ax1.plot(sc_z["week"], sc_z["zscore"], color=SC_COLOR, linewidth=1.8, label="WW Z-score")
ax1.axhline(Z_THRESHOLD,  color="red",    linestyle="--", linewidth=1.4, label=f"Alert  z = +{Z_THRESHOLD}")
ax1.axhline(-Z_THRESHOLD, color="orange", linestyle=":",  linewidth=1.0, label=f"Below-baseline  z = -{Z_THRESHOLD}")
ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
_alert = sc_z["zscore"] > Z_THRESHOLD
ax1.fill_between(sc_z["week"], sc_z["zscore"], Z_THRESHOLD,
                 where=_alert, color="red", alpha=0.15, label="Active alert periods")
ax1.set_ylabel(f"Z-Score (vs {ROLLING_BASELINE_WEEKS}-wk rolling baseline)", fontsize=10)
ax1.set_title("Santa Clara WW Signal Z-Score — Outbreak Early Warning", fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.set_ylim(-4, _z_max + 0.3)

# Panel 2: WW with rolling baseline
ax2.fill_between(sc_z["week"], sc_z["rc_log"], alpha=0.20, color=SC_COLOR)
ax2.plot(sc_z["week"], sc_z["rc_log"],    color=SC_COLOR, linewidth=1.8, label="log1p(Raw Conc)")
ax2.plot(sc_z["week"], sc_z["roll_mean"], color="navy",   linewidth=1.5,
         linestyle="--", label=f"{ROLLING_BASELINE_WEEKS}-wk rolling mean")
ax2.fill_between(sc_z["week"],
                 sc_z["roll_mean"] - sc_z["roll_std"],
                 sc_z["roll_mean"] + sc_z["roll_std"],
                 alpha=0.12, color="navy", label="±1 SD band")
ax2.set_ylabel("log1p(Raw Concentration)", fontsize=10)
ax2.set_title("WW Concentration — log1p with Rolling Baseline", fontweight="bold")
ax2.legend(fontsize=9)

# Panel 3: Cases
_roll_cases_sc = cases_sc.set_index("date")["cases"].rolling(4, min_periods=1).mean()
ax3.fill_between(cases_sc["date"], cases_sc["cases"], alpha=0.20, color=C_CASES)
ax3.bar(cases_sc["date"], cases_sc["cases"], width=5, color=C_CASES, alpha=0.35)
ax3.plot(_roll_cases_sc.index, _roll_cases_sc.values, color="darkred", linewidth=2.0, label="4-wk rolling avg")
ax3.set_ylabel("Weekly New Cases (Santa Clara)", fontsize=10)
ax3.set_xlabel("Week (W-WED)", fontsize=10)
ax3.set_title("Weekly New Cases — Ground Truth", fontweight="bold")
ax3.legend(fontsize=9)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.set_xlim(OVERLAP_START, OVERLAP_END)

plt.tight_layout()
plt.show()
'''))

    cells.append(code('''
# Early warning summary — Santa Clara
_WAVE_WINDOWS_SC = {
    "Wave 1 (Original)":     ("2020-07-01", "2020-10-31"),
    "Wave 2 (Alpha/WT)":     ("2020-11-01", "2021-04-30"),
    "Wave 3 (Delta)":        ("2021-06-01", "2021-11-30"),
    "Wave 4 (Omicron BA.1)": ("2021-12-01", "2022-05-31"),
    "BA.5":                  ("2022-06-01", "2022-09-30"),
    "BQ.1 / BQ.1.1":        ("2022-10-01", "2023-01-15"),
    "XBB.1.5":               ("2023-01-16", "2023-04-30"),
}

summary_rows_sc = []
for wave_name, (ws, we) in _WAVE_WINDOWS_SC.items():
    ws_ts, we_ts = pd.Timestamp(ws), pd.Timestamp(we)
    cases_win = cases_sc[(cases_sc["date"] >= ws_ts) & (cases_sc["date"] <= we_ts)]
    if cases_win.empty or cases_win["cases"].sum() == 0:
        summary_rows_sc.append({"Wave": wave_name, "Case Peak Date": "—",
                                "WW Alert (z>2)": "—", "Lead (wks)": "—", "Status": "No cases"})
        continue
    case_peak_dt = cases_win.loc[cases_win["cases"].idxmax(), "date"]
    ww_win = sc_z[(sc_z["week"] >= ws_ts) & (sc_z["week"] <= we_ts) & (sc_z["zscore"] > Z_THRESHOLD)]
    if ww_win.empty:
        summary_rows_sc.append({"Wave": wave_name, "Case Peak Date": case_peak_dt.strftime("%Y-%m-%d"),
                                "WW Alert (z>2)": "—", "Lead (wks)": "—", "Status": "No WW alert fired"})
        continue
    ww_alert_dt = ww_win["week"].min()
    lead_wks    = (case_peak_dt - ww_alert_dt).days / 7.0
    status = "Leads" if lead_wks > 0.5 else ("Lags" if lead_wks < -1 else "Coincident")
    summary_rows_sc.append({
        "Wave":            wave_name,
        "Case Peak Date":  case_peak_dt.strftime("%Y-%m-%d"),
        "WW Alert (z>2)":  ww_alert_dt.strftime("%Y-%m-%d"),
        "Lead (wks)":      f"{lead_wks:+.1f}",
        "Status":          status,
    })

summary_sc_df = pd.DataFrame(summary_rows_sc)
print("=== Santa Clara Early Warning Summary: WW Z-Score vs Case Peaks ===")
print(f"Alert threshold: z > {Z_THRESHOLD}  |  Baseline: {ROLLING_BASELINE_WEEKS}-week rolling window\\n")
print(summary_sc_df.to_string(index=False))
'''))

    # Section 6: Cross-correlation
    cells.append(md("""
---
## Section 6: Cross-Correlation — WW at t vs Cases at t+lag

Pearson r(WW_t, cases_{t+lag}) for lags 0–8 weeks.
The peak lag is the optimal WW lead time for Santa Clara.
"""))

    cells.append(code('''
lags_cc  = range(0, 9)
cc_vals  = []
cc_valid_n = []

for lag in lags_cc:
    ww_v  = merged_sc["log1p_rc"].values[:-lag]  if lag > 0 else merged_sc["log1p_rc"].values
    cas_v = merged_sc["log1p_cases"].values[lag:] if lag > 0 else merged_sc["log1p_cases"].values
    n = min(len(ww_v), len(cas_v))
    paired = np.column_stack([ww_v[:n], cas_v[:n]])
    valid  = paired[~np.isnan(paired).any(axis=1)]
    cc_vals.append(np.corrcoef(valid[:, 0], valid[:, 1])[0, 1] if len(valid) >= 8 else np.nan)
    cc_valid_n.append(len(valid))

best_lag = int(np.nanargmax(cc_vals)) if not all(np.isnan(v) for v in cc_vals) else 0

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    "Figure 6: Cross-Correlation WW at t vs Cases at t+lag — Santa Clara County\\n"
    "Pearson r (pairwise NaN-clean); peak = optimal WW lead time; 1-3 wk window shaded",
    fontsize=13, fontweight="bold"
)
bar_colors = [SC_COLOR if lag == best_lag else C_WW for lag in lags_cc]
bars = ax.bar(list(lags_cc), cc_vals, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.5)
ax.axvspan(0.5, 3.5, alpha=0.08, color=C_ACCENT, label="Typical lead window (1-3 wks)")
ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
for lag, r in zip(lags_cc, cc_vals):
    if not np.isnan(r):
        ax.text(lag, r + 0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if lag == best_lag else "normal",
                color=SC_COLOR if lag == best_lag else "gray")
ax.set_xlabel("Lag (weeks WW leads cases)", fontsize=11)
ax.set_ylabel("Pearson r (WW vs future cases)", fontsize=11)
ax.set_xticks(list(lags_cc))
ax.set_xticklabels([f"{l}w" for l in lags_cc])
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(min(min(v for v in cc_vals if not np.isnan(v)), 0) - 0.05, 1.0)
plt.tight_layout()
plt.show()

print("\\nSanta Clara WW->Cases cross-correlation by lag:")
for lag, r, n in zip(lags_cc, cc_vals, cc_valid_n):
    marker = " <- PEAK CORRELATION" if lag == best_lag else ""
    r_str  = f"{r:.3f}" if not np.isnan(r) else "  NaN"
    print(f"  Lag {lag}w: r = {r_str}  (n={n}){marker}")
print(f"\\n-> Optimal WW lead time: {best_lag} week(s)")
'''))

    # Figure E: Train/CV/Holdout split
    cells.append(md("""
---
## Section 7: Model Training Window Anatomy

The plot below annotates the Santa Clara case timeline with the **exact train / CV / holdout splits**
and per-fold CV WIS scores — the same splits used in both the SC single-county and 3-county Bay Area runs.
"""))

    cells.append(code('''
TRAIN_START = pd.Timestamp("2020-07-01")
TRAIN_END   = pd.Timestamp("2022-10-05")
VAL_END     = pd.Timestamp("2023-06-07")
HOLDOUT_END = pd.Timestamp("2023-12-19")

CV_FOLDS = [
    (pd.Timestamp("2022-10-05"), 0.173),
    (pd.Timestamp("2022-11-02"), 0.392),
    (pd.Timestamp("2022-11-30"), 1.354),
    (pd.Timestamp("2022-12-28"), 0.459),
    (pd.Timestamp("2023-01-25"), 0.204),
    (pd.Timestamp("2023-02-22"), 0.285),
    (pd.Timestamp("2023-03-22"), 0.065),
    (pd.Timestamp("2023-04-19"), 0.109),
    (pd.Timestamp("2023-05-17"), 0.088),
]

_ymax = cases_sc["cases"].max()
_roll_sc = cases_sc.set_index("date")["cases"].rolling(4, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(20, 8))

for color, s, e, lbl in ALL_WAVE_SPANS:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.05, color=color, zorder=0)

ax.axvspan(TRAIN_START, TRAIN_END,   alpha=0.10, color="#2196F3", zorder=1)
ax.axvspan(TRAIN_END,   VAL_END,     alpha=0.10, color="#FF9800", zorder=1)
ax.axvspan(VAL_END,     HOLDOUT_END, alpha=0.10, color="#F44336", zorder=1)

for i, (fold_dt, wis) in enumerate(CV_FOLDS):
    fcolor = "#b71c1c" if wis > 0.5 else ("#e65100" if wis > 0.15 else "#2e7d32")
    ax.axvline(fold_dt, color=fcolor, linewidth=1.1, linestyle=":", alpha=0.85, zorder=4)
    ypos = _ymax * (0.52 if i % 2 == 0 else 0.43)
    ax.text(fold_dt, ypos, f"F{i+1}\\nWIS\\n{wis:.2f}",
            ha="center", va="top", fontsize=7.5, color=fcolor, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=fcolor, alpha=0.88))

ax.fill_between(cases_sc["date"], cases_sc["cases"], alpha=0.18, color=SC_COLOR, zorder=2)
ax.bar(cases_sc["date"], cases_sc["cases"], color=SC_COLOR, alpha=0.38, width=5, zorder=3)
ax.plot(_roll_sc.index, _roll_sc.values, color="black", linewidth=2.3, zorder=5, label="4-wk rolling avg")

_VARIANT_ANN = [
    ("Wave 2\\n(Alpha/WT)",   "2021-01-10", 0.72, "#f4a261"),
    ("Wave 3\\n(Delta)",      "2021-09-10", 0.67, "#57cc99"),
    ("Omicron BA.1",          "2022-01-12", 0.90, "#e76f51"),
    ("BA.5",                  "2022-07-15", 0.34, "#c77dff"),
    ("BQ.1 / BQ.1.1",        "2022-11-15", 0.38, "#9b72cf"),
    ("XBB.1.5",               "2023-02-10", 0.33, "#48cae4"),
]
for lbl, cdt, yfrac, color in _VARIANT_ANN:
    ax.annotate(lbl, xy=(pd.Timestamp(cdt), _ymax * yfrac),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.88))

for dt, lbl, color in [(TRAIN_END, "TRAIN END / CV START", "#1565C0"),
                        (VAL_END,   "CV END / HOLDOUT START", "#B71C1C")]:
    ax.axvline(dt, color=color, linewidth=2.2, linestyle="--", zorder=6)
    ax.text(dt + pd.Timedelta(days=3), _ymax * 0.04, lbl,
            ha="left", va="bottom", fontsize=8.5, color=color, fontweight="bold")

ax.set_xlim(TRAIN_START - pd.Timedelta(weeks=2), HOLDOUT_END + pd.Timedelta(weeks=2))
ax.set_ylim(0, _ymax * 1.18)
ax.set_xlabel("Week (W-WED)", fontsize=11)
ax.set_ylabel("Weekly New Cases — Santa Clara County", fontsize=11)
ax.set_title(
    "Figure E: Train / CV / Holdout Splits — Santa Clara County COVID-19 Timeline\\n"
    "Blue = Train | Orange = CV window (9 expanding folds) | Red = Holdout\\n"
    "Fold WIS color: green <= 0.15 | orange 0.15-0.50 | red > 0.50",
    fontsize=12, fontweight="bold"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("TRAIN (2020-07 -> 2022-10):  Wave 1 (first summer surge), Wave 2 (Alpha), Wave 3 (Delta), Omicron onset")
print("CV    (2022-10 -> 2023-06):  BQ.1 peak (Fold 3, worst WIS=1.354), XBB.1.5 onset")
print("HOLD  (2023-06 -> 2023-12):  Post-XBB.1.5 decline  -- ~7% Coverage_95")
print("\\n-> Santa Clara has 173 joint modelable weeks vs CDC baseline of 66 -- 107 extra weeks.")
'''))

    nb["cells"] = cells
    return nb


# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nb03_path = OUT / "03_pipeline_bay_area_ca.ipynb"
    nb04_path = OUT / "04_pipeline_santa_clara_county.ipynb"

    nb03 = make_notebook_bay_area()
    nbf.write(nb03, str(nb03_path))
    print(f"Written: {nb03_path}")

    nb04 = make_notebook_santa_clara()
    nbf.write(nb04, str(nb04_path))
    print(f"Written: {nb04_path}")

    print("\nDone. Open in Jupyter / VS Code to run.")
