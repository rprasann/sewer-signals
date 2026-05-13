"""
Generate project overview figures from EDA data.
Saves 6 PNG files to the assets/ directory.
Run from the project root:  python assets/generate_overview_figures.py
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import find_peaks
from scipy import stats

sns.set_theme(style='darkgrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a1a',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#ccc',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#eee',
    'grid.color':       '#2a2a2a',
    'legend.facecolor': '#1a1a1a',
    'legend.edgecolor': '#444',
})

OUT_DIR  = Path(__file__).parent
DATA_DIR = Path(__file__).parents[1] / 'data' / 'raw'
WW_FILE    = DATA_DIR / 'CDC_Wastewater_Data_for_SARS-CoV-2_20260505.csv'
CASES_FILE = DATA_DIR / 'Weekly_United_States_COVID-19_Cases_and_Deaths_by_County_-_ARCHIVED_20260502.csv'

BAY_AREA_FIPS = {
    'Alameda':       '6001', 'Contra Costa':  '6013', 'Marin':         '6041',
    'Napa':          '6055', 'San Francisco': '6075', 'San Mateo':     '6081',
    'Santa Clara':   '6085', 'Solano':        '6095', 'Sonoma':        '6097',
}
FIPS_TO_NAME    = {v.zfill(5): k for k, v in BAY_AREA_FIPS.items()}
BAY_FIPS_PADDED = set(FIPS_TO_NAME.keys())

OVERLAP_START = pd.Timestamp('2022-02-07')
OVERLAP_END   = pd.Timestamp('2023-05-10')
TARGET_UNIT   = "copies/g dry sludge"

ALL_COUNTIES = sorted(FIPS_TO_NAME.values())
COUNTY_COLORS = dict(zip(ALL_COUNTIES, sns.color_palette('tab10', 9)))

# ── Wave band definitions ─────────────────────────────────────────────────────
WAVES = {
    'BA.2\n(Spring \'22)':             (pd.Timestamp('2022-02-07'), pd.Timestamp('2022-06-01')),
    'BA.4/5\n(Summer \'22)':           (pd.Timestamp('2022-06-01'), pd.Timestamp('2022-10-01')),
    'BQ.1 / XBB.1.5\n(Winter \'22-23)': (pd.Timestamp('2022-10-01'), pd.Timestamp('2023-05-10')),
}
WAVE_COLORS = {
    'BA.2\n(Spring \'22)':               '#4e9af1',
    'BA.4/5\n(Summer \'22)':             '#f4a261',
    'BQ.1 / XBB.1.5\n(Winter \'22-23)': '#e76f51',
}

print("Loading data...")

# ── Load & filter wastewater ──────────────────────────────────────────────────
ww_raw = pd.read_csv(WW_FILE, dtype=str, low_memory=False)
ww_raw.columns = ww_raw.columns.str.strip().str.lower().str.replace(' ', '_')
ww_raw['sample_collect_date'] = pd.to_datetime(ww_raw['sample_collect_date'], errors='coerce')
for col in ['pcr_target_avg_conc', 'lod_sewage', 'rec_eff_percent', 'population_served']:
    if col in ww_raw.columns:
        ww_raw[col] = pd.to_numeric(ww_raw[col].str.replace(',', '', regex=False), errors='coerce')

ww_ca = ww_raw[ww_raw['state_territory'] == 'ca'].copy()
ww_ca['county_fips_list'] = ww_ca['county_fips'].str.split(',')
ww_ca = ww_ca.explode('county_fips_list')
ww_ca['county_fips_list'] = ww_ca['county_fips_list'].str.strip().str.zfill(5)
ww_bay = ww_ca[ww_ca['county_fips_list'].isin(BAY_FIPS_PADDED)].copy()
ww_bay.rename(columns={'county_fips_list': 'fips'}, inplace=True)
ww_bay['county'] = ww_bay['fips'].map(FIPS_TO_NAME)
ww_bay = ww_bay[
    (ww_bay['sample_collect_date'] >= OVERLAP_START) &
    (ww_bay['sample_collect_date'] <= OVERLAP_END)
].copy()

unit_col = 'pcr_target_units'
ww_g = ww_bay[ww_bay[unit_col].str.contains('copies/g', case=False, na=False)].copy()
if 'pcr_target_detect' in ww_g.columns:
    ww_g = ww_g[ww_g['pcr_target_detect'].str.lower() == 'yes'].copy()

ww_active = ww_g

# ── Load & filter cases ───────────────────────────────────────────────────────
cases_raw = pd.read_csv(CASES_FILE, dtype=str, low_memory=False)
cases_raw.columns = cases_raw.columns.str.strip().str.lower().str.replace(' ', '_')
cases_raw['date'] = pd.to_datetime(
    cases_raw['end_date'] if 'end_date' in cases_raw.columns else cases_raw['date'], errors='coerce')
for col in ['new_cases', 'new_deaths', 'cumulative_cases', 'cumulative_deaths']:
    if col in cases_raw.columns:
        cases_raw[col] = pd.to_numeric(cases_raw[col].str.replace(',', '', regex=False), errors='coerce')

cases_ca = cases_raw[cases_raw['state'] == 'CA'].copy()
cases_ca['fips_code'] = cases_ca['fips_code'].astype(str).str.zfill(5)
cases_bay = cases_ca[cases_ca['fips_code'].isin(BAY_FIPS_PADDED)].copy()
cases_bay['county'] = cases_bay['fips_code'].map(FIPS_TO_NAME)
cases_bay = cases_bay[
    (cases_bay['date'] >= OVERLAP_START) & (cases_bay['date'] <= OVERLAP_END)
].copy()

# ── Weekly WW aggregation ─────────────────────────────────────────────────────
ww_weekly = (
    ww_active.set_index('sample_collect_date')
    .groupby('county')['pcr_target_avg_conc']
    .resample('W-WED').median()
    .reset_index()
)
ww_weekly.columns = ['county', 'week', 'conc']

def build_merged(county_name):
    src = ww_active[ww_active['county'] == county_name].copy()
    if len(src) == 0:
        return pd.DataFrame()
    ww_w = (src.set_index('sample_collect_date')['pcr_target_avg_conc']
            .resample('W-WED').median().reset_index())
    ww_w.columns = ['date', 'ww_conc']
    cas_w = cases_bay[cases_bay['county'] == county_name][['date', 'new_cases']].copy()
    cas_w['new_cases'] = cas_w['new_cases'].clip(lower=0)
    return pd.merge(ww_w, cas_w, on='date', how='inner').dropna(subset=['ww_conc', 'new_cases'])

modelable_counties = sorted(
    set(ww_active['county'].dropna()) & set(cases_bay['county'].dropna())
)
print(f"Modelable counties: {len(modelable_counties)}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: County time-series 3×3 grid (WW + cases overlay)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure A: county 3x3 grid...")
county_order = [
    'Alameda', 'Contra Costa', 'Marin',
    'Napa', 'San Francisco', 'San Mateo',
    'Santa Clara', 'Solano', 'Sonoma'
]

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'Weekly SARS-CoV-2 WW Concentration — All 9 Bay Area Counties\n'
    f'(copies/g dry sludge, W-WED resampled, log scale)',
    fontsize=14, fontweight='bold', color='#eee'
)

for ax, county in zip(axes.flat, county_order):
    color  = COUNTY_COLORS.get(county, 'steelblue')
    subset = ww_weekly[ww_weekly['county'] == county].dropna(subset=['conc'])
    if len(subset) > 0:
        ax.plot(subset['week'], subset['conc'], color=color, linewidth=1.8)
        ax.fill_between(subset['week'], subset['conc'], alpha=0.18, color=color)
        ax.set_yscale('log')
        ax.set_ylabel('copies/g (log)', fontsize=7, color='#aaa')
    ax.set_title(county, fontweight='bold', color=color, fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    # shade variant waves
    for wave_name, (ws, we) in WAVES.items():
        wc = WAVE_COLORS[wave_name]
        ax.axvspan(ws, we, alpha=0.07, color=wc, zorder=0)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_a_county_ww_timeseries.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_a_county_ww_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: Bay Area aggregate Z-score overlay (the "lead time" money shot)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B: aggregate z-score overlay...")

ww_liq = ww_bay[ww_bay[unit_col].str.contains('copies/l', case=False, na=False)].copy()
if 'pcr_target_detect' in ww_liq.columns:
    ww_liq = ww_liq[ww_liq['pcr_target_detect'].str.lower() == 'yes'].copy()

ww_agg_raw = (
    ww_liq.set_index('sample_collect_date')['pcr_target_avg_conc']
    .resample('W-WED').median().reset_index()
)
ww_agg_raw.columns = ['date', 'ww_agg']
ww_agg_raw = ww_agg_raw[(ww_agg_raw['date'] >= OVERLAP_START) & (ww_agg_raw['date'] <= OVERLAP_END)]
cases_agg = cases_bay.groupby('date')['new_cases'].sum().clip(lower=0).reset_index()
cases_agg.columns = ['date', 'cases_agg']
agg = pd.merge(ww_agg_raw, cases_agg, on='date', how='inner').sort_values('date').reset_index(drop=True).dropna()

def zscore_series(s):
    return (s - s.mean()) / (s.std() + 1e-9)

agg['ww_z']    = zscore_series(np.log1p(agg['ww_agg']))
agg['cases_z'] = zscore_series(agg['cases_agg'])

ww_peaks,    _ = find_peaks(agg['ww_z'].values,    height=0.5, distance=4)
cases_peaks, _ = find_peaks(agg['cases_z'].values, height=0.5, distance=4)

fig, ax = plt.subplots(figsize=(15, 5))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

ax.plot(agg['date'], agg['ww_z'],    color='#38BDF8', linewidth=2.2, label='Z-score  log(WW concentration)',  zorder=3)
ax.plot(agg['date'], agg['cases_z'], color='#F97316', linewidth=2.2, label='Z-score  weekly new cases',  zorder=3)
ax.fill_between(agg['date'], agg['ww_z'],    alpha=0.15, color='#38BDF8', zorder=2)
ax.fill_between(agg['date'], agg['cases_z'], alpha=0.15, color='#F97316', zorder=2)

arrow_kw = dict(arrowstyle='->', lw=1.5)
for pk_idx in ww_peaks:
    pk_date, pk_val = agg['date'].iloc[pk_idx], agg['ww_z'].iloc[pk_idx]
    ax.annotate('WW\npeak', xy=(pk_date, pk_val), xytext=(pk_date, pk_val + 0.7),
                arrowprops={**arrow_kw, 'color': '#38BDF8'},
                ha='center', fontsize=8, color='#38BDF8', fontweight='bold')
for pk_idx in cases_peaks:
    pk_date, pk_val = agg['date'].iloc[pk_idx], agg['cases_z'].iloc[pk_idx]
    ax.annotate('Cases\npeak', xy=(pk_date, pk_val), xytext=(pk_date, pk_val + 0.7),
                arrowprops={**arrow_kw, 'color': '#F97316'},
                ha='center', fontsize=8, color='#F97316', fontweight='bold')

for wave_name, (ws, we) in WAVES.items():
    wc = WAVE_COLORS[wave_name]
    ax.axvspan(ws, we, alpha=0.07, color=wc, zorder=0)
    mid = ws + (we - ws) / 2
    ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -2.8,
            wave_name.replace('\n', ' '), ha='center', fontsize=7, color=wc, alpha=0.8)

ax.axhline(0, color='#444', linestyle='--', linewidth=1)
ax.set_xlabel('Week (Wednesday-anchored)', color='#ccc')
ax.set_ylabel('Z-score', color='#ccc')
ax.set_title(
    'Bay Area Aggregate: Wastewater Signal Leads Clinical Case Reports\n'
    'Z-scored log(WW concentration) vs Z-scored weekly new cases — all 9 counties',
    fontsize=12, fontweight='bold', color='#eee'
)
ax.legend(fontsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_b_aggregate_zscore_leadtime.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_b_aggregate_zscore_leadtime.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C: Lag-correlation profiles — all counties
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure C: lag-correlation profiles...")

LAG_RANGE = range(-4, 9)
lag_profiles = {}
lag_results  = []

for county in modelable_counties:
    merged = build_merged(county)
    if len(merged) < 10:
        continue
    merged = merged.sort_values('date').reset_index(drop=True)
    log_ww = np.log1p(merged['ww_conc'].values)
    cases  = merged['new_cases'].values
    n = len(merged)
    profile  = []
    best_r, best_lag = -np.inf, 0
    for lag in LAG_RANGE:
        if lag > 0:
            r = np.corrcoef(log_ww[:n-lag], cases[lag:])[0,1] if n-lag >= 5 else np.nan
        elif lag < 0:
            k = abs(lag)
            r = np.corrcoef(cases[:n-k], log_ww[k:])[0,1] if n-k >= 5 else np.nan
        else:
            r = np.corrcoef(log_ww, cases)[0,1]
        profile.append(r)
        if not np.isnan(r) and r > best_r:
            best_r, best_lag = r, lag
    lag_profiles[county] = profile
    lag_results.append({'county': county, 'best_lag_wks': best_lag, 'max_pearson_r': round(best_r, 3)})

lag_df = pd.DataFrame(lag_results).sort_values('best_lag_wks', ascending=False)

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

for county, profile in lag_profiles.items():
    ax.plot(list(LAG_RANGE), profile, marker='o', markersize=4,
            linewidth=1.8, color=COUNTY_COLORS.get(county), label=county, alpha=0.9)

median_best = int(lag_df['best_lag_wks'].median())
ax.axvline(0, color='#555', linestyle=':', linewidth=1.5)
ax.axvline(median_best, color='#F97316', linestyle='--', linewidth=2.0, alpha=0.8)
ax.text(median_best + 0.15, 0.45, f'Median best lag\n+{median_best} week(s)',
        ha='left', va='top', fontsize=9, color='#F97316', fontweight='bold')

ax.set_xlabel('Lag (weeks) — positive = WW leads cases', color='#ccc')
ax.set_ylabel('Pearson r  [log(WW) vs new cases]', color='#ccc')
ax.set_title(
    'Wastewater Leads Clinical Reports: Lag-Correlation Profiles\n'
    'All 9 Bay Area Counties — copies/g dry sludge',
    fontweight='bold', color='#eee'
)
ax.legend(loc='lower right', ncol=3, fontsize=8)
ax.set_xticks(list(LAG_RANGE))
ax.set_xlim(-4.5, 8.5)
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_c_lag_correlation_profiles.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_c_lag_correlation_profiles.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D: Dual-axis merged WW + cases — 4 illustrative counties
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure D: merged dual-axis 4 counties...")

illustrative = ['San Francisco', 'Santa Clara', 'Sonoma', 'Alameda']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'Wastewater (blue) vs Weekly New COVID-19 Cases (orange)\n'
    'Wednesday-Aligned W-WED Spine — Illustrative Counties',
    fontsize=13, fontweight='bold', color='#eee'
)

for ax, county in zip(axes.flat, illustrative):
    merged = build_merged(county)
    color  = COUNTY_COLORS.get(county, 'steelblue')
    ax.set_facecolor('#1a1a1a')

    if len(merged) == 0:
        ax.text(0.5, 0.5, f'No joint data for {county}', ha='center', va='center',
                transform=ax.transAxes, color='#888')
        continue

    ax2 = ax.twinx()
    ax2.bar(merged['date'], merged['new_cases'], width=5, color='#F97316', alpha=0.35, label='New cases')
    ax2.plot(merged['date'], merged['new_cases'].rolling(4, min_periods=1).mean(),
             color='#F97316', linewidth=2.0, alpha=0.85, label='4-wk rolling avg')
    ax2.set_ylabel('Weekly New Cases', color='#F97316', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#F97316', labelsize=8)
    ax2.spines['right'].set_edgecolor('#F97316')

    ax.plot(merged['date'], merged['ww_conc'], color='#38BDF8', linewidth=2.2, zorder=5,
            label=f'WW ({TARGET_UNIT})')
    ax.set_yscale('log')
    ax.set_ylabel(f'WW (log, copies/g)', color='#38BDF8', fontsize=8)
    ax.tick_params(axis='y', labelcolor='#38BDF8', labelsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.set_title(f'{county}  [n={len(merged)} weeks]', fontweight='bold', color=color)

    for wave_name, (ws, we) in WAVES.items():
        wc = WAVE_COLORS[wave_name]
        ax.axvspan(ws, we, alpha=0.07, color=wc, zorder=0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_d_merged_ww_cases_dualaxis.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_d_merged_ww_cases_dualaxis.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E: Early-warning Gantt — alert → case peak lead time
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure E: early-warning Gantt...")

gants = []
for county in modelable_counties:
    merged = build_merged(county)
    if len(merged) < 6:
        continue
    merged = merged.sort_values('date').reset_index(drop=True)
    merged['ww_smooth']   = merged['ww_conc'].rolling(4, min_periods=2).mean()
    merged['ww_baseline'] = merged['ww_smooth'].shift(4)
    merged['surge_alert'] = merged['ww_smooth'] > merged['ww_baseline'] * 1.25
    merged['alert_onset'] = merged['surge_alert'] & ~merged['surge_alert'].shift(1, fill_value=False)

    for wave_name, (w_start, w_end) in WAVES.items():
        wave_data = merged[(merged['date'] >= w_start) & (merged['date'] < w_end)]
        if len(wave_data) < 3 or wave_data['new_cases'].isna().all():
            continue
        alert_rows = wave_data[wave_data['alert_onset']]
        alert_date = alert_rows['date'].iloc[0] if len(alert_rows) > 0 else None
        peak_date  = wave_data.loc[wave_data['new_cases'].idxmax(), 'date']
        lead_wks   = (peak_date - alert_date).days // 7 if alert_date is not None and alert_date < peak_date else 0
        gants.append({'county': county, 'wave': wave_name,
                      'alert_date': alert_date, 'peak_date': peak_date, 'lead_wks': lead_wks})

gants_df = pd.DataFrame(gants)

if len(gants_df) > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    counties_ordered_gantt = sorted(gants_df['county'].unique())
    county_y   = {c: i for i, c in enumerate(counties_ordered_gantt)}
    wave_list  = list(WAVES.keys())
    bar_height = 0.25
    wave_offsets = {w: (i - 1) * bar_height for i, w in enumerate(wave_list)}

    for _, row in gants_df.iterrows():
        if row['alert_date'] is None:
            continue
        y     = county_y[row['county']] + wave_offsets.get(row['wave'], 0)
        color = WAVE_COLORS.get(row['wave'], '#888')
        width_days = (row['peak_date'] - row['alert_date']).days
        ax.barh(y, width=width_days, left=row['alert_date'],
                height=bar_height * 0.85, color=color, alpha=0.75, edgecolor='#222')
        mid = row['alert_date'] + (row['peak_date'] - row['alert_date']) / 2
        if row['lead_wks'] > 0:
            ax.text(mid, y, f'+{row["lead_wks"]}w',
                    ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')

    ax.legend(
        handles=[mpatches.Patch(color=c, label=w.replace('\n', ' ')) for w, c in WAVE_COLORS.items()],
        fontsize=9, loc='upper left'
    )
    ax.set_yticks(list(county_y.values()))
    ax.set_yticklabels(list(county_y.keys()), color='#ccc')
    ax.set_xlabel('Date — bar spans from WW alert onset to clinical case peak', color='#ccc')
    ax.set_title(
        'Early Warning Timeline: Wastewater Alert → Clinical Case Peak per Wave\n'
        'Bar width = lead time (weeks). Label = advance warning weeks.',
        fontsize=12, fontweight='bold', color='#eee'
    )
    ax.set_xlim(OVERLAP_START, OVERLAP_END)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_e_early_warning_gantt.png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("  → fig_e_early_warning_gantt.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F: Cross-county WW correlation heatmap (justifies global model)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure F: cross-county correlation heatmap...")

ww_all = (
    ww_active.set_index('sample_collect_date')
    .groupby('county')['pcr_target_avg_conc']
    .resample('W-WED').median()
    .reset_index()
)
ww_all.columns = ['county', 'week', 'conc']
ww_all = ww_all.dropna(subset=['conc'])
ww_all = ww_all[(ww_all['week'] >= OVERLAP_START) & (ww_all['week'] <= OVERLAP_END)]

pivot      = ww_all.pivot_table(index='week', columns='county', values='conc')
corr_matrix = np.log1p(pivot).corr()

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

sns.heatmap(
    corr_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
    vmin=0, vmax=1, ax=ax, linewidths=0.5,
    annot_kws={'size': 9, 'color': '#111'},
    linecolor='#222'
)
ax.set_title(
    'Cross-County WW Correlation Matrix — Pearson r of log(concentration)\n'
    'High off-diagonal r → counties share epidemic wave dynamics → global TFT is justified',
    fontsize=11, fontweight='bold', color='#eee'
)
ax.tick_params(axis='x', rotation=40, labelsize=9, colors='#ccc')
ax.tick_params(axis='y', rotation=0, labelsize=9, colors='#ccc')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_f_crosscounty_correlation_heatmap.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_f_crosscounty_correlation_heatmap.png")

print("\nAll 6 figures saved to assets/")
