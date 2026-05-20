"""Generate cross-run pipeline comparison charts (Plotly HTML)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "documents"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
mc_eval = json.load(open(ROOT / "data/processed/eval_summary.json"))
mc_cv   = pd.read_csv(ROOT / "data/processed/cv_results.csv")
sc_eval = json.load(open(ROOT / "data/processed/santa_clara/eval_summary.json"))
sc_cv   = pd.read_csv(ROOT / "data/processed/santa_clara/cv_results.csv")

fold_labels = [f"Fold {i+1}" for i in range(len(mc_cv))]
_ORANGE = "#F97316"
_BLUE   = "#38BDF8"
_AMBER  = "#FDE68A"
_VIOLET = "#C084FC"

# ---------------------------------------------------------------------------
# Chart 1: WIS per CV fold (both runs)
# ---------------------------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=fold_labels, y=mc_cv["mean_wis"].round(3),
    name="9-County (Run 3)", marker_color=_ORANGE,
    text=mc_cv["mean_wis"].round(3), textposition="outside",
))
fig1.add_trace(go.Bar(
    x=fold_labels, y=sc_cv["mean_wis"].round(3),
    name="Santa Clara (Run 4)", marker_color=_BLUE,
    text=sc_cv["mean_wis"].round(3), textposition="outside",
))
fig1.update_layout(
    title="Mean WIS per CV Fold — Cross-Run Comparison",
    yaxis_title="Mean WIS (lower = better)",
    barmode="group",
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font=dict(color="#F8FAFC"), legend=dict(bgcolor="#1E293B"),
    xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#334155"),
)
fig1.write_html(str(OUT / "chart_wis_per_fold.html"))
print("Saved: chart_wis_per_fold.html")

# ---------------------------------------------------------------------------
# Chart 2: Coverage_95 per CV fold
# ---------------------------------------------------------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=fold_labels, y=(mc_cv["coverage_95"] * 100).round(1),
    mode="lines+markers", name="9-County (Run 3)",
    line=dict(color=_ORANGE, width=2), marker=dict(size=8),
))
fig2.add_trace(go.Scatter(
    x=fold_labels, y=(sc_cv["coverage_95"] * 100).round(1),
    mode="lines+markers", name="Santa Clara (Run 4)",
    line=dict(color=_BLUE, width=2), marker=dict(size=8, symbol="diamond"),
))
fig2.add_hline(y=95, line_dash="dot", line_color=_AMBER,
               annotation_text="Nominal 95%", annotation_font_color=_AMBER)
fig2.update_layout(
    title="95% PI Coverage per CV Fold",
    yaxis_title="Empirical Coverage (%)",
    yaxis=dict(range=[0, 115], gridcolor="#334155"),
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font=dict(color="#F8FAFC"), legend=dict(bgcolor="#1E293B"),
    xaxis=dict(gridcolor="#1E293B"),
)
fig2.write_html(str(OUT / "chart_coverage_per_fold.html"))
print("Saved: chart_coverage_per_fold.html")

# ---------------------------------------------------------------------------
# Chart 3: Holdout scorecard (radar)
# ---------------------------------------------------------------------------
metrics_labels = ["WIS (inv)", "Coverage 50%", "Coverage 95%", "SMAPE (inv)"]

def norm_inv(v, hi): return max(0, 1 - v / hi)  # lower-is-better, normalise to [0,1]
def norm(v):         return min(1.0, v)           # already [0,1]

mc_vals = [
    norm_inv(mc_eval["mean_wis"], 2.0),
    norm(mc_eval["coverage_50"]),
    norm(mc_eval["coverage_95"]),
    norm_inv(mc_eval["smape"], 2.0),
]
sc_vals = [
    norm_inv(sc_eval["mean_wis"], 2.0),
    norm(sc_eval["coverage_50"]),
    norm(sc_eval["coverage_95"]),
    norm_inv(sc_eval["smape"], 2.0),
]

fig3 = go.Figure()
for vals, name, color in [(mc_vals, "9-County Holdout", _ORANGE),
                          (sc_vals, "Santa Clara Holdout", _BLUE)]:
    closed = vals + [vals[0]]
    closed_labels = metrics_labels + [metrics_labels[0]]
    # Convert hex to rgba
    h = color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    fill = f"rgba({r},{g},{b},0.15)"
    fig3.add_trace(go.Scatterpolar(
        r=closed, theta=closed_labels, fill="toself",
        name=name, line_color=color, fillcolor=fill,
    ))
fig3.update_layout(
    polar=dict(
        bgcolor="#1E293B",
        radialaxis=dict(range=[0, 1], gridcolor="#334155", tickfont_color="#94A3B8"),
        angularaxis=dict(gridcolor="#334155"),
    ),
    title="Holdout Scorecard (normalised, higher = better)",
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font=dict(color="#F8FAFC"), legend=dict(bgcolor="#1E293B"),
)
fig3.write_html(str(OUT / "chart_holdout_scorecard.html"))
print("Saved: chart_holdout_scorecard.html")

# ---------------------------------------------------------------------------
# Chart 4: WIS heatmap per county × CV fold (9-county run only)
# ---------------------------------------------------------------------------
wis_county_cols = [c for c in mc_cv.columns if c.startswith("wis_")]
county_labels   = [c.replace("wis_", "") for c in wis_county_cols]

z = mc_cv[wis_county_cols].values.T  # [counties, folds]
# Replace NaN with 0 for display
z_display = np.where(np.isnan(z), 0, z)

fig4 = go.Figure(go.Heatmap(
    z=z_display,
    x=fold_labels,
    y=county_labels,
    colorscale="Plasma",
    colorbar=dict(title="WIS", tickfont=dict(size=11)),
    hovertemplate="County: %{y}<br>Fold: %{x}<br>WIS: %{z:.3f}<extra></extra>",
    zmin=0, zmax=1.5,
))
fig4.update_layout(
    title="WIS per County × CV Fold (9-County Run)",
    xaxis_title="CV Fold", yaxis_title="County FIPS",
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font=dict(color="#F8FAFC"),
    xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
    height=400,
)
fig4.write_html(str(OUT / "chart_wis_heatmap_counties.html"))
print("Saved: chart_wis_heatmap_counties.html")

print("\nAll charts saved to documents/")
