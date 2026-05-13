"""
Generate the three additional figures (5, 10, 13) not produced in the first pass.
Run from project root:  python assets/generate_extra_figures.py
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
OVERLAP_START   = pd.Timestamp('2022-02-07')
OVERLAP_END     = pd.Timestamp('2023-05-10')
ALL_COUNTIES    = sorted(FIPS_TO_NAME.values())
COUNTY_COLORS   = dict(zip(ALL_COUNTIES, sns.color_palette('tab10', 9)))

COUNTY_ORDER = [
    'Alameda', 'Contra Costa', 'Marin',
    'Napa', 'San Francisco', 'San Mateo',
    'Santa Clara', 'Solano', 'Sonoma'
]

WAVES = {
    "BA.2 (Spring '22)":             (pd.Timestamp('2022-02-07'), pd.Timestamp('2022-06-01')),
    "BA.4/5 (Summer '22)":           (pd.Timestamp('2022-06-01'), pd.Timestamp('2022-10-01')),
    "BQ.1 / XBB.1.5 (Winter '22-23)": (pd.Timestamp('2022-10-01'), pd.Timestamp('2023-05-10')),
}
WAVE_COLORS = {
    "BA.2 (Spring '22)":               '#4e9af1',
    "BA.4/5 (Summer '22)":             '#f4a261',
    "BQ.1 / XBB.1.5 (Winter '22-23)": '#e76f51',
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
ww_raw = pd.read_csv(WW_FILE, dtype=str, low_memory=False)
ww_raw.columns = ww_raw.columns.str.strip().str.lower().str.replace(' ', '_')
ww_raw['sample_collect_date'] = pd.to_datetime(ww_raw['sample_collect_date'], errors='coerce')
ww_raw['pcr_target_avg_conc'] = pd.to_numeric(
    ww_raw['pcr_target_avg_conc'].str.replace(',', '', regex=False), errors='coerce')

ww_ca  = ww_raw[ww_raw['state_territory'] == 'ca'].copy()
ww_ca['county_fips_list'] = ww_ca['county_fips'].str.split(',')
ww_ca  = ww_ca.explode('county_fips_list')
ww_ca['county_fips_list'] = ww_ca['county_fips_list'].str.strip().str.zfill(5)
ww_bay = ww_ca[ww_ca['county_fips_list'].isin(BAY_FIPS_PADDED)].copy()
ww_bay.rename(columns={'county_fips_list': 'fips'}, inplace=True)
ww_bay['county'] = ww_bay['fips'].map(FIPS_TO_NAME)
ww_bay = ww_bay[(ww_bay['sample_collect_date'] >= OVERLAP_START) &
                (ww_bay['sample_collect_date'] <= OVERLAP_END)].copy()

unit_col = 'pcr_target_units'
ww_g   = ww_bay[ww_bay[unit_col].str.contains('copies/g', case=False, na=False)].copy()
ww_liq = ww_bay[ww_bay[unit_col].str.contains('copies/l', case=False, na=False)].copy()
for df in [ww_g, ww_liq]:
    if 'pcr_target_detect' in df.columns:
        df.drop(df[df['pcr_target_detect'].str.lower() != 'yes'].index, inplace=True)

cases_raw = pd.read_csv(CASES_FILE, dtype=str, low_memory=False)
cases_raw.columns = cases_raw.columns.str.strip().str.lower().str.replace(' ', '_')
cases_raw['date'] = pd.to_datetime(
    cases_raw.get('end_date', cases_raw.get('date')), errors='coerce')
cases_raw['new_cases'] = pd.to_numeric(
    cases_raw['new_cases'].str.replace(',', '', regex=False), errors='coerce')
cases_ca  = cases_raw[cases_raw['state'] == 'CA'].copy()
cases_ca['fips_code'] = cases_ca['fips_code'].astype(str).str.zfill(5)
cases_bay = cases_ca[cases_ca['fips_code'].isin(BAY_FIPS_PADDED)].copy()
cases_bay['county'] = cases_bay['fips_code'].map(FIPS_TO_NAME)
cases_bay = cases_bay[(cases_bay['date'] >= OVERLAP_START) &
                      (cases_bay['date'] <= OVERLAP_END)].copy()

def shade_waves(ax):
    for name, (ws, we) in WAVES.items():
        ax.axvspan(ws, we, alpha=0.07, color=WAVE_COLORS[name], zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Weekly new cases 3×3 grid
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 5: cases 3×3 grid...")
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'Weekly New COVID-19 Cases — All 9 Bay Area Counties\n'
    '(Overlap window 2022-02-09 → 2023-05-10)',
    fontsize=14, fontweight='bold', color='#eee'
)

for ax, county in zip(axes.flat, COUNTY_ORDER):
    color  = COUNTY_COLORS.get(county, 'steelblue')
    subset = cases_bay[cases_bay['county'] == county].sort_values('date')
    ax.set_facecolor('#1a1a1a')
    shade_waves(ax)
    if len(subset) > 0:
        ax.bar(subset['date'], subset['new_cases'].clip(lower=0),
               color=color, alpha=0.45, width=5)
        rolling = subset['new_cases'].clip(lower=0).rolling(4, min_periods=1).mean()
        ax.plot(subset['date'], rolling, color='#F97316', linewidth=2.0, label='4-wk avg')
        ax.set_ylabel('New cases / week', fontsize=7, color='#aaa')
        ax.legend(fontsize=7)
    ax.set_title(county, fontweight='bold', color=color, fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_5_cases_grid.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_5_cases_grid.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Rate-of-change: sludge vs liquid (Santa Clara copies/g, SF copies/l)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 10: rate-of-change sludge vs liquid...")

def build_merged_from(ww_src_df, county_name):
    src = ww_src_df[ww_src_df['county'] == county_name].copy()
    if len(src) == 0:
        return pd.DataFrame()
    ww_w = (src.set_index('sample_collect_date')['pcr_target_avg_conc']
            .resample('W-WED').median().reset_index())
    ww_w.columns = ['date', 'ww_conc']
    cas_w = cases_bay[cases_bay['county'] == county_name][['date', 'new_cases']].copy()
    cas_w['new_cases'] = cas_w['new_cases'].clip(lower=0)
    return pd.merge(ww_w, cas_w, on='date', how='inner').dropna(subset=['ww_conc', 'new_cases'])

panels = [
    ('Santa Clara', ww_g,   'copies/g dry sludge',  '#38BDF8'),
    ('San Francisco', ww_liq, 'copies/l wastewater', '#C084FC'),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'Rate-of-Change Comparison: Sludge Track vs Liquid Track\n'
    'Δ WW / week (blue/violet) vs Δ new cases / week (orange dashed)',
    fontsize=13, fontweight='bold', color='#eee'
)

for ax, (county, ww_src_df, unit_label, ww_color) in zip(axes, panels):
    merged = build_merged_from(ww_src_df, county)
    ax.set_facecolor('#1a1a1a')
    shade_waves(ax)
    if len(merged) < 8:
        ax.set_title(f'{county} — insufficient data')
        continue
    merged = merged.sort_values('date').reset_index(drop=True)
    merged['dww']    = merged['ww_conc'].pct_change().clip(-2, 5)
    merged['dcases'] = merged['new_cases'].pct_change().clip(-2, 5)
    merged = merged.dropna(subset=['dww', 'dcases'])

    ax2 = ax.twinx()
    ax.plot(merged['date'],  merged['dww'],    color=ww_color,   linewidth=2.0, label=f'Δ WW ({unit_label})')
    ax2.plot(merged['date'], merged['dcases'], color='#F97316', linewidth=1.8, linestyle='--', label='Δ new cases / wk')
    ax.fill_between(merged['date'], merged['dww'],    alpha=0.12, color=ww_color)
    ax.axhline(0, color='#555', linewidth=0.8, linestyle=':')

    ax.set_ylabel(f'Δ WW / week  ({unit_label})', color=ww_color, fontsize=9)
    ax2.set_ylabel('Δ new cases / week', color='#F97316', fontsize=9)
    ax.tick_params(axis='y', labelcolor=ww_color, labelsize=8)
    ax2.tick_params(axis='y', labelcolor='#F97316', labelsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.set_title(f'{county}  —  {unit_label}', fontweight='bold',
                 color=COUNTY_COLORS.get(county, 'white'))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax2.spines['right'].set_edgecolor('#F97316')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_10_rate_of_change.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_10_rate_of_change.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13 — Normalized wave synchrony overlay (all 9 counties)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 13: normalized wave synchrony overlay...")

ww_all = (
    ww_g.set_index('sample_collect_date')
    .groupby('county')['pcr_target_avg_conc']
    .resample('W-WED').median()
    .reset_index()
)
ww_all.columns = ['county', 'week', 'conc']
ww_all = ww_all.dropna(subset=['conc'])
ww_all = ww_all[(ww_all['week'] >= OVERLAP_START) & (ww_all['week'] <= OVERLAP_END)]

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

for county in COUNTY_ORDER:
    s = ww_all[ww_all['county'] == county].sort_values('week')
    if len(s) < 5:
        continue
    conc_norm = (s['conc'] - s['conc'].min()) / (s['conc'].max() - s['conc'].min() + 1e-9)
    ax.plot(s['week'], conc_norm,
            label=county, color=COUNTY_COLORS.get(county), linewidth=1.8, alpha=0.88)

shade_waves(ax)
# annotate wave bands
for name, (ws, we) in WAVES.items():
    mid   = ws + (we - ws) / 2
    color = WAVE_COLORS[name]
    ax.text(mid, -0.08, name, ha='center', va='top', fontsize=8,
            color=color, fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.set_xlabel('Week (Wednesday-anchored)', color='#ccc')
ax.set_ylabel('Normalized WW Concentration [0–1 per county]', color='#ccc')
ax.set_title(
    'Epidemic Wave Synchrony — All 9 Bay Area Counties\n'
    'Min-max normalized copies/g dry sludge  |  Shared peaks confirm global model is appropriate',
    fontsize=13, fontweight='bold', color='#eee'
)
ax.legend(loc='upper right', ncol=3, fontsize=9)
ax.set_ylim(-0.05, 1.15)
for spine in ax.spines.values():
    spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_13_wave_synchrony.png', bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  → fig_13_wave_synchrony.png")

print("\nDone.")
