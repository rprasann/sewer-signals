"""
Sewer Surveillance Dashboard — Bay Area Wastewater Surveillance
Attention-based outbreak forecasting with COVID-19 wastewater data

Professional public-health light mode, California blue + orange palette.

Usage
-----
    from src.visualization.dashboard import create_app

    app = create_app(
        processed_df=processed,
        forecast_df=forecast,
        eval_result=eval_result,
        model=model,
        cv_results=cv_df,
    )
    app.run(debug=False, port=8050)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table

from src.config import (
    BAY_AREA_FIPS,
    COUNTY_COL,
    DASH_HOST,
    DASH_PORT,
    DATA_START_DATE,
    DATA_END_DATE,
    FIPS_TO_COUNTY,
    NWSS_DATE_COL,
    TARGET_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from src.evaluation.metrics import QuantileColumns
from src.visualization.attention_plots import (
    extract_attention_weights,
    extract_vsn_weights,
    plot_attention_heatmap,
    plot_vsn_importance,
)


# ---------------------------------------------------------------------------
# California Public Health — Light Mode Color Tokens
# ---------------------------------------------------------------------------

_BG       = "#F4F7FB"        # page background (cool blue-gray)
_SURFACE  = "#FFFFFF"        # card surface
_SURFACE2 = "#EBF0F9"        # alternate rows / secondary surface
_BORDER   = "#C5D0E4"        # borders, dividers
_TEXT     = "#162032"        # primary text (dark navy)
_MUTED    = "#576880"        # secondary / muted text
_ACCENT   = "#E8821C"        # California golden orange — forecast + brand
_BLUE     = "#1A4FA0"        # California navy blue — training + WW signal
_GREEN    = "#15803D"        # good metrics, low WIS
_AMBER    = "#B45309"        # caution — validation period
_RED      = "#B91C1C"        # alert — holdout period
_VIOLET   = "#6D28D9"        # CV SMAPE line

_PLOT_BG  = "#FFFFFF"
_PAPER_BG = "#FFFFFF"
_GRID     = "rgba(0,0,0,0.06)"

_FONT_BODY = {"family": "Inter, system-ui, sans-serif", "size": 14, "color": _TEXT}

# Per-county palette for all-counties spaghetti (light background friendly)
_COUNTY_COLORS = [
    "#1A4FA0",  # CA blue
    "#E8821C",  # CA orange
    "#15803D",  # green
    "#7C3AED",  # violet
    "#B91C1C",  # red
    "#0284C7",  # sky blue
    "#047857",  # emerald
    "#92400E",  # warm amber
    "#6D28D9",  # deep violet
]

# Bay Area county centroids (lat, lon)
_COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    "06001": (37.6017, -121.7195),  # Alameda
    "06013": (37.9147, -121.9235),  # Contra Costa
    "06041": (38.0834, -122.7633),  # Marin
    "06055": (38.5025, -122.2654),  # Napa
    "06075": (37.7749, -122.4194),  # San Francisco
    "06081": (37.4350, -122.3185),  # San Mateo
    "06085": (37.3328, -121.8956),  # Santa Clara
    "06095": (38.2494, -121.9018),  # Solano
    "06097": (38.5780, -122.8759),  # Sonoma
}

# Biological translation table static content
_BIO_TABLE_DATA = [
    {
        "metric":      "Coverage (95%)",
        "what_it_is":  "% of weeks the actual case count fell inside our 95% predicted range",
        "bio_label":   "The Safety Net",
        "bio_meaning": "How reliably our predicted range catches real outbreaks. ≥95% = well-calibrated model.",
        "target":      "≥ 95%",
    },
    {
        "metric":      "WIS (Error)",
        "what_it_is":  "Weighted Interval Score — combines sharpness and calibration into one number",
        "bio_label":   "The Precision Score",
        "bio_meaning": "How much the forecast 'missed' the actual surge magnitude. Lower = more accurate.",
        "target":      "Lower is better",
    },
    {
        "metric":      "Sensitivity",
        "what_it_is":  "True Positive Rate — fraction of real outbreaks we correctly flagged",
        "bio_label":   "The Early Warning Score",
        "bio_meaning": "Our ability to trigger an alert before the clinical surge hits.",
        "target":      "≥ 0.80",
    },
    {
        "metric":      "Specificity",
        "what_it_is":  "True Negative Rate — fraction of quiet periods we correctly stayed silent",
        "bio_label":   "The False Alarm Rate",
        "bio_meaning": "Our ability to remain silent when there is no outbreak. False alarms erode clinical trust.",
        "target":      "≥ 0.85",
    },
    {
        "metric":      "Lead Time",
        "what_it_is":  "Days between our alert and the confirmed clinical case surge",
        "bio_label":   "The Decision Window",
        "bio_meaning": "How many days of advance warning we give. 7–21 days = actionable for hospital planning.",
        "target":      "7–21 days",
    },
]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _card(children, style: dict | None = None) -> html.Div:
    base = {
        "backgroundColor": _SURFACE,
        "border": f"1px solid {_BORDER}",
        "borderRadius": "10px",
        "padding": "16px",
        "height": "100%",
        "boxShadow": "0 1px 4px rgba(22,32,50,0.08)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _section_label(text: str, sub: str = "") -> html.Div:
    children = [
        html.H6(text, style={"color": _TEXT, "fontWeight": "600", "marginBottom": "2px"}),
    ]
    if sub:
        children.append(
            html.P(sub, style={"color": _MUTED, "fontSize": "12px", "marginBottom": "12px"})
        )
    return html.Div(children)


def _empty_fig(message: str = "No data", height: int = 350) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color=_MUTED),
    )
    fig.update_layout(
        height=height, paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
        font=_FONT_BODY, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _base_layout(title: str, height: int, margin: dict) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=_TEXT)),
        height=height,
        margin=margin,
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
        font=_FONT_BODY,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=_BORDER, borderwidth=1,
            font=dict(size=11, color=_TEXT),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
    )


def _axis(title_text: str, **kwargs) -> dict:
    return dict(
        title=dict(text=title_text, font=dict(size=12, color=_MUTED)),
        tickfont=dict(size=11, color=_MUTED),
        gridcolor=_GRID, showgrid=True, zeroline=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Biological Translation Table
# ---------------------------------------------------------------------------

def _build_bio_table(eval_result=None) -> dash_table.DataTable:
    def _fmt(v, pct: bool = False, days: bool = False) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if pct:
            return f"{v * 100:.1f}%"
        if days:
            return f"{v:.1f} days"
        return f"{v:.3f}"

    live: dict[str, str] = {}
    if eval_result is not None:
        lt = eval_result.lead_time
        live = {
            "Coverage (95%)": _fmt(eval_result.coverage_95, pct=True),
            "WIS (Error)":    _fmt(eval_result.mean_wis),
            "Sensitivity":    _fmt(lt.sensitivity, pct=True),
            "Specificity":    _fmt(lt.specificity, pct=True),
            "Lead Time":      _fmt(lt.mean_lead_days, days=True),
        }

    has_live = bool(live)
    rows = [{**r, "actual": live.get(r["metric"], "—")} for r in _BIO_TABLE_DATA]

    columns = [
        {"name": "Technical Metric",   "id": "metric"},
        {"name": "Clinical Label",     "id": "bio_label"},
        {"name": "Biological Meaning", "id": "bio_meaning"},
        {"name": "Target",             "id": "target"},
    ]
    if has_live:
        columns.insert(2, {"name": "Actual (Holdout)", "id": "actual"})

    cond = [
        {"if": {"row_index": "odd"}, "backgroundColor": _SURFACE2},
        {"if": {"column_id": "bio_label"}, "color": _ACCENT, "fontWeight": "600"},
        {"if": {"column_id": "target"},    "color": _BLUE,   "fontWeight": "600"},
    ]
    if has_live:
        cond.append({"if": {"column_id": "actual"}, "color": _GREEN, "fontWeight": "700"})

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_header={
            "backgroundColor": _SURFACE2, "color": _TEXT,
            "fontWeight": "600", "fontSize": "13px",
            "border": f"1px solid {_BORDER}",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
        },
        style_cell={
            "backgroundColor": _SURFACE, "color": _TEXT,
            "border": f"1px solid {_BORDER}",
            "padding": "10px 14px", "fontSize": "13px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "whiteSpace": "normal", "height": "auto", "textAlign": "left",
        },
        style_data_conditional=cond,
        tooltip_data=[
            {col["id"]: {"value": row.get(col["id"], ""), "type": "text"} for col in columns}
            for row in rows
        ],
        tooltip_delay=400, tooltip_duration=None,
        page_action="none", sort_action="native",
    )


# ---------------------------------------------------------------------------
# Clickable Bay Area map  (Scattermapbox — open-street-map, no token)
# ---------------------------------------------------------------------------

def create_map(
    forecast_df: pd.DataFrame,
    q_cols: QuantileColumns,
    selected_fips: str | None = None,
) -> go.Figure:
    """Bay Area bubble map coloured by 8-week median forecast density.
    Uses Scattermapbox with open-street-map tiles for proper Bay Area zoom.
    Click a bubble to select that county; hero and timeline update.
    """
    rows = []
    for fips, (lat, lon) in _COUNTY_CENTROIDS.items():
        sub = forecast_df[forecast_df["unique_id"] == fips]
        density = float(sub[q_cols.q50].sum()) if (not sub.empty and q_cols.q50 in sub.columns) else 0.0
        rows.append({
            "fips": fips,
            "name": FIPS_TO_COUNTY.get(fips, fips),
            "lat": lat, "lon": lon, "density": density,
        })

    df_map = pd.DataFrame(rows)
    mx = df_map["density"].max()
    df_map["norm"] = df_map["density"] / mx if mx > 0 else 0.0

    def _bubble_color(norm: float, fips: str) -> str:
        if fips == selected_fips:
            return _ACCENT
        return _GREEN if norm < 0.33 else (_AMBER if norm < 0.66 else _RED)

    sizes     = [max(22, int(52 * n)) for n in df_map["norm"]]
    colors    = [_bubble_color(n, f) for n, f in zip(df_map["norm"], df_map["fips"])]
    opacities = [1.0 if f == selected_fips or selected_fips is None else 0.55
                 for f in df_map["fips"]]

    fig = go.Figure(go.Scattermapbox(
        lat=df_map["lat"],
        lon=df_map["lon"],
        text=df_map["name"],
        customdata=df_map[["density", "fips"]].values,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color=_TEXT),
        marker=dict(size=sizes, color=colors, opacity=opacities),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "8-wk Forecast Σ: %{customdata[0]:.2f}<br>"
            "<i>Click to focus</i><extra></extra>"
        ),
    ))

    # Halo ring for selected county
    if selected_fips and selected_fips in df_map["fips"].values:
        sel = df_map[df_map["fips"] == selected_fips].iloc[0]
        sel_idx = df_map["fips"].tolist().index(selected_fips)
        fig.add_trace(go.Scattermapbox(
            lat=[sel["lat"]], lon=[sel["lon"]],
            mode="markers",
            marker=dict(size=sizes[sel_idx] + 18, color=_ACCENT, opacity=0.22),
            hoverinfo="skip",
            showlegend=False,
        ))

    subtitle = (
        f"— {FIPS_TO_COUNTY.get(selected_fips, '')} selected · click to deselect"
        if selected_fips else "· click a county to focus"
    )
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=37.7, lon=-122.2),
            zoom=7.8,
        ),
        title=dict(text=f"Forecasted Case Density {subtitle}", font=dict(size=13, color=_TEXT)),
        paper_bgcolor=_PAPER_BG,
        font=_FONT_BODY,
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Hero forecast plot  (8-week horizon)
# ---------------------------------------------------------------------------

def create_hero_plot(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    county_fips: str | None,
    q_cols: QuantileColumns,
    context_weeks: int = 26,
) -> go.Figure:
    """8-week forecast hero chart.

    county_fips=None → all-counties spaghetti of medians.
    county_fips=str  → single county:
      - 50% / 95% PI ribbons (80% skipped if corrupt/negative)
      - context actuals (26 weeks before forecast)
      - holdout actuals during forecast window (diamond markers, CA blue)
    """
    fig = go.Figure()
    all_fips = sorted(forecast_df["unique_id"].unique())

    if county_fips is None:
        # ── All-counties median spaghetti ────────────────────────────────────
        for i, fips in enumerate(all_fips):
            sub = forecast_df[forecast_df["unique_id"] == fips].sort_values("ds")
            if sub.empty or q_cols.q50 not in sub.columns:
                continue
            color = _COUNTY_COLORS[i % len(_COUNTY_COLORS)]
            name  = FIPS_TO_COUNTY.get(fips, fips)
            fig.add_trace(go.Scatter(
                x=sub["ds"], y=sub[q_cols.q50],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>Forecast: %{{y:.2f}}<extra></extra>",
            ))
        title = "8-Week Forecast — All Bay Area Counties  ·  click map to focus"

    else:
        # ── Single-county ribbon chart ────────────────────────────────────────
        fcast  = forecast_df[forecast_df["unique_id"] == county_fips].sort_values("ds")
        actual = actual_df[actual_df[COUNTY_COL] == county_fips].sort_values(NWSS_DATE_COL)

        if not fcast.empty:
            fcast_start   = pd.Timestamp(fcast["ds"].min())
            context_cutoff = fcast_start - pd.Timedelta(weeks=context_weeks)
            actual_pre    = actual[
                (actual[NWSS_DATE_COL] >= context_cutoff) &
                (actual[NWSS_DATE_COL] < fcast_start)
            ]
            actual_holdout = actual[actual[NWSS_DATE_COL] >= fcast_start]
        else:
            actual_pre    = actual
            actual_holdout = pd.DataFrame()

        def _band(lo_col: str, hi_col: str, fill_color: str, name: str) -> None:
            if lo_col not in fcast.columns or hi_col not in fcast.columns:
                return
            lo_vals = fcast[lo_col]
            hi_vals = fcast[hi_col]
            # Skip band if quantile values are corrupt (e.g. q10/q90 went negative)
            if lo_vals.median() < 0 or hi_vals.median() < 0:
                logger.debug("Skipping {} band — negative quantile values.", name)
                return
            x = list(fcast["ds"]) + list(fcast["ds"])[::-1]
            y = list(lo_vals) + list(hi_vals)[::-1]
            fig.add_trace(go.Scatter(
                x=x, y=y, fill="toself",
                fillcolor=fill_color, line=dict(color="rgba(0,0,0,0)"),
                name=name, showlegend=True, hoverinfo="skip",
            ))

        # Widest band first so narrower bands render on top
        _band(q_cols.q025, q_cols.q975, "rgba(232,130,28,0.13)", "95% PI")
        if q_cols.q10 is not None and q_cols.q90 is not None:
            _band(q_cols.q10,  q_cols.q90,  "rgba(232,130,28,0.24)", "80% PI")
        _band(q_cols.q25,  q_cols.q75,  "rgba(232,130,28,0.38)", "50% PI")

        if q_cols.q50 in fcast.columns:
            fig.add_trace(go.Scatter(
                x=fcast["ds"], y=fcast[q_cols.q50],
                name="Median Forecast",
                line=dict(color=_ACCENT, width=3),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.2f}<extra></extra>",
            ))

        # Context actuals (pre-forecast)
        if not actual_pre.empty and TARGET_COL in actual_pre.columns:
            fig.add_trace(go.Scatter(
                x=actual_pre[NWSS_DATE_COL], y=actual_pre[TARGET_COL],
                name="Observed (context)",
                mode="lines+markers",
                line=dict(color=_TEXT, width=2),
                marker=dict(size=5, color=_TEXT),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Observed: %{y:.2f}<extra></extra>",
            ))

        # Holdout actuals DURING forecast window — direct forecast vs. reality comparison
        if not actual_holdout.empty and TARGET_COL in actual_holdout.columns:
            fig.add_trace(go.Scatter(
                x=actual_holdout[NWSS_DATE_COL], y=actual_holdout[TARGET_COL],
                name="Observed (holdout)",
                mode="lines+markers",
                line=dict(color=_BLUE, width=2.5),
                marker=dict(size=8, color=_BLUE, symbol="diamond"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Holdout actual: %{y:.2f}<extra></extra>",
            ))

        # Forecast-start separator — bare vline (annotation kwargs break Plotly 6 on date axes)
        if not fcast.empty:
            fig.add_vline(
                x=pd.Timestamp(fcast["ds"].min()).to_pydatetime(),
                line=dict(color=_MUTED, dash="longdash", width=1),
            )

        county_name = FIPS_TO_COUNTY.get(county_fips, county_fips)
        title = f"8-Week Forecast — {county_name}  ·  click map to deselect"

    fig.update_layout(
        **_base_layout(title, height=440, margin=dict(l=70, r=30, t=65, b=60)),
        xaxis=_axis("Date"),
        yaxis=_axis("Weekly New Cases (log1p)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Full timeline  (train / val / holdout / forecast periods)
# ---------------------------------------------------------------------------

def create_timeline_chart(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    county_fips: str | None,
    q_cols: QuantileColumns,
) -> go.Figure:
    """Full historical + forecast timeline with period shading.

    Training (blue) → Validation (amber) → Holdout (red) → Forecast (orange).
    """
    fig = go.Figure()

    # Use Python datetime objects — pd.Timestamp / ISO strings both break
    # Plotly 6's shapeannotation._mean for date axes.
    _dt = lambda s: datetime.strptime(s, "%Y-%m-%d")
    train_end  = _dt(TRAIN_END_DATE)
    val_end    = _dt(VAL_END_DATE)
    data_end   = _dt(DATA_END_DATE)
    data_start = _dt(DATA_START_DATE)

    def _vrect(x0, x1, color: str, label: str) -> None:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color, opacity=0.10,
            layer="below", line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_color=_MUTED,
            annotation_font_size=10,
        )

    _vrect(data_start, train_end, _BLUE,   "Training")
    _vrect(train_end,  val_end,   _AMBER,  "Validation")
    _vrect(val_end,    data_end,  _RED,    "Holdout")

    fcast_end_ts = forecast_df["ds"].max() if not forecast_df.empty else pd.Timestamp(DATA_END_DATE)
    fcast_end    = pd.Timestamp(fcast_end_ts).to_pydatetime()
    if fcast_end > data_end:
        _vrect(data_end, fcast_end, _ACCENT, "Forecast")

    if county_fips is not None:
        # ── Single county: actuals + forecast ribbon + 95% PI ────────────────
        act = actual_df[actual_df[COUNTY_COL] == county_fips].sort_values(NWSS_DATE_COL)
        if not act.empty and TARGET_COL in act.columns:
            fig.add_trace(go.Scatter(
                x=act[NWSS_DATE_COL], y=act[TARGET_COL],
                name=f"{FIPS_TO_COUNTY.get(county_fips, county_fips)} — Observed",
                mode="lines+markers",
                line=dict(color=_TEXT, width=2),
                marker=dict(size=4, color=_TEXT),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Observed: %{y:.2f}<extra></extra>",
            ))

        fcast = forecast_df[forecast_df["unique_id"] == county_fips].sort_values("ds")
        if not fcast.empty and q_cols.q50 in fcast.columns:
            # 95% PI ribbon (guard against negative quantiles)
            if (q_cols.q025 in fcast.columns and q_cols.q975 in fcast.columns and
                    fcast[q_cols.q025].median() >= 0 and fcast[q_cols.q975].median() >= 0):
                x_band = list(fcast["ds"]) + list(fcast["ds"])[::-1]
                y_band = list(fcast[q_cols.q025]) + list(fcast[q_cols.q975])[::-1]
                fig.add_trace(go.Scatter(
                    x=x_band, y=y_band, fill="toself",
                    fillcolor="rgba(232,130,28,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% PI", hoverinfo="skip",
                ))
            # 50% PI ribbon
            if (q_cols.q25 in fcast.columns and q_cols.q75 in fcast.columns and
                    fcast[q_cols.q25].median() >= 0 and fcast[q_cols.q75].median() >= 0):
                x_band = list(fcast["ds"]) + list(fcast["ds"])[::-1]
                y_band = list(fcast[q_cols.q25]) + list(fcast[q_cols.q75])[::-1]
                fig.add_trace(go.Scatter(
                    x=x_band, y=y_band, fill="toself",
                    fillcolor="rgba(232,130,28,0.32)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="50% PI", hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=fcast["ds"], y=fcast[q_cols.q50],
                name="Forecast Median",
                line=dict(color=_ACCENT, width=2.5, dash="dot"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.2f}<extra></extra>",
            ))
        title_suffix = FIPS_TO_COUNTY.get(county_fips, county_fips)

    else:
        # ── All counties: per-county thin lines + Bay Area average ───────────
        bay_dates = actual_df.groupby(NWSS_DATE_COL)[TARGET_COL].mean().reset_index()
        bay_dates = bay_dates.sort_values(NWSS_DATE_COL)

        for i, fips in enumerate(sorted(actual_df[COUNTY_COL].unique())):
            act = actual_df[actual_df[COUNTY_COL] == fips].sort_values(NWSS_DATE_COL)
            if act.empty or TARGET_COL not in act.columns:
                continue
            color = _COUNTY_COLORS[i % len(_COUNTY_COLORS)]
            name  = FIPS_TO_COUNTY.get(fips, fips)
            fig.add_trace(go.Scatter(
                x=act[NWSS_DATE_COL], y=act[TARGET_COL],
                name=name, mode="lines",
                line=dict(color=color, width=1.2),
                opacity=0.45,
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
            ))

        # Bay Area average as bold overlay
        if not bay_dates.empty:
            fig.add_trace(go.Scatter(
                x=bay_dates[NWSS_DATE_COL], y=bay_dates[TARGET_COL],
                name="Bay Area Avg",
                mode="lines",
                line=dict(color=_TEXT, width=3),
                opacity=0.9,
                hovertemplate="<b>Bay Area Avg</b><br>%{x|%b %d, %Y}<br>%{y:.2f}<extra></extra>",
            ))
        title_suffix = "All Bay Area Counties"

    # Period boundary vlines — bare lines (annotation kwargs crash Plotly 6 on date axes)
    for ts in [train_end, val_end, data_end]:
        fig.add_vline(x=ts, line=dict(color=_BORDER, dash="dot", width=1))

    fig.update_layout(
        **_base_layout(
            f"Full Data Timeline — {title_suffix}",
            height=340,
            margin=dict(l=70, r=30, t=55, b=55),
        ),
        xaxis=_axis("Date", range=[data_start, fcast_end]),
        yaxis=_axis("Weekly New Cases (log1p)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Suggested visual 1 — Per-county holdout WIS ranking
# ---------------------------------------------------------------------------

def _build_county_wis_chart(eval_result) -> go.Figure:
    wis = eval_result.wis_per_county
    if not wis:
        return _empty_fig("No per-county WIS data", height=300)

    names  = [FIPS_TO_COUNTY.get(f, f) for f in wis]
    values = list(wis.values())
    pairs  = sorted(zip(values, names), reverse=True)
    values, names = zip(*pairs)
    colors = [_GREEN if v < 0.2 else (_AMBER if v < 0.5 else _RED) for v in values]

    fig = go.Figure(go.Bar(
        x=list(values), y=list(names),
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="<b>%{y}</b><br>WIS: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("County WIS — Which Forecasts to Trust", height=320,
                        margin=dict(l=110, r=30, t=55, b=50)),
        xaxis=_axis("WIS (lower = more accurate)"),
        yaxis=dict(tickfont=dict(size=11, color=_TEXT), gridcolor=_GRID),
        showlegend=False,
    )
    fig.add_vline(x=0.2, line=dict(color=_GREEN, dash="dot", width=1),
                  annotation_text="good", annotation_font_color=_GREEN,
                  annotation_position="top right")
    fig.add_vline(x=0.5, line=dict(color=_AMBER, dash="dot", width=1),
                  annotation_text="fair", annotation_font_color=_AMBER,
                  annotation_position="top right")
    return fig


# ---------------------------------------------------------------------------
# Suggested visual 2 — CV fold performance over time
# ---------------------------------------------------------------------------

def _build_cv_chart(cv_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cv_df["cutoff_date"], y=cv_df["mean_wis"],
        name="WIS", mode="lines+markers",
        line=dict(color=_ACCENT, width=2), marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>WIS: %{y:.3f}<extra></extra>",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=cv_df["cutoff_date"], y=cv_df["smape"] * 100,
        name="SMAPE %", mode="lines+markers",
        line=dict(color=_VIOLET, width=2, dash="dashdot"), marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>SMAPE: %{y:.1f}%%<extra></extra>",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=cv_df["cutoff_date"], y=cv_df["coverage_95"] * 100,
        name="Coverage 95%", mode="lines+markers",
        line=dict(color=_BLUE, width=2, dash="dot"), marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Coverage: %{y:.1f}%%<extra></extra>",
        yaxis="y2",
    ))
    fig.add_hline(y=95, yref="y2",
                  line=dict(color=_BLUE, dash="longdash", width=1),
                  annotation_text="95% target", annotation_font_color=_BLUE,
                  annotation_position="top right")

    fig.update_layout(
        **_base_layout("CV Fold Performance — Temporal Stability", height=320,
                        margin=dict(l=70, r=70, t=55, b=60)),
        xaxis=_axis("CV Cutoff Date"),
        yaxis=dict(
            title=dict(text="WIS / SMAPE %", font=dict(size=12, color=_ACCENT)),
            tickfont=dict(size=11, color=_MUTED),
            gridcolor=_GRID, zeroline=False, side="left",
        ),
        yaxis2=dict(
            title=dict(text="Coverage 95% (%)", font=dict(size=12, color=_BLUE)),
            tickfont=dict(size=11, color=_BLUE),
            overlaying="y", side="right",
            range=[0, 105], zeroline=False, showgrid=False,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app builder
# ---------------------------------------------------------------------------

def create_app(
    processed_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model=None,
    sludge_df: Optional[pd.DataFrame] = None,
    liquid_df: Optional[pd.DataFrame] = None,
    q_cols: Optional[QuantileColumns] = None,
    eval_result=None,
    cv_results: Optional[pd.DataFrame] = None,
) -> dash.Dash:
    """Build and return the Sewer Surveillance production dashboard."""
    if q_cols is None:
        try:
            q_cols = QuantileColumns.auto_detect(forecast_df)
        except ValueError:
            q_cols = QuantileColumns()

    _vsn_weights: dict = {}
    if model is not None:
        try:
            raw = extract_vsn_weights(model)
            if raw:
                _vsn_weights = raw
                logger.info("VSN weights pre-extracted.")
        except Exception as exc:
            logger.warning("VSN extraction skipped: {}", exc)

    has_suggestions = eval_result is not None or (cv_results is not None and not cv_results.empty)

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.FLATLY,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        ],
        title="Sewer Surveillance — Bay Area Wastewater Surveillance",
    )

    # ── Layout ──────────────────────────────────────────────────────────────────
    app.layout = dbc.Container(
        fluid=True,
        style={"backgroundColor": _BG, "minHeight": "100vh", "padding": "20px 28px"},
        children=[

            # ── Row 0: Header ──────────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Sewer Surveillance", style={
                            "fontSize": "28px", "fontWeight": "700",
                            "color": _ACCENT, "letterSpacing": "-0.03em",
                            "fontFamily": "Inter, system-ui, sans-serif",
                        }),
                        html.Span(" | ", style={"color": _BORDER, "margin": "0 10px"}),
                        html.Span(
                            "Attention-based outbreak forecasting with COVID-19 wastewater data",
                            style={"fontSize": "16px", "color": _MUTED, "fontWeight": "400"},
                        ),
                    ], style={"display": "flex", "alignItems": "baseline"}),
                    html.Div(
                        id="selected-county-badge",
                        style={"color": _MUTED, "fontSize": "12px", "marginTop": "4px"},
                        children="9-County Bay Area  ·  8-week horizon  ·  click a county on the map to focus",
                    ),
                ]),
            ], className="mb-4"),

            # ── Row 1: Hero + Map ──────────────────────────────────────────────
            dbc.Row([
                dbc.Col(
                    _card([dcc.Graph(
                        id="hero-chart",
                        config={"displayModeBar": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                "displaylogo": False},
                    )]),
                    width=8,
                ),
                dbc.Col(
                    _card([dcc.Graph(
                        id="map-chart",
                        config={"displayModeBar": False},
                    )]),
                    width=4,
                ),
            ], className="mb-4"),

            # ── Row 2: Full timeline ───────────────────────────────────────────
            dbc.Row([
                dbc.Col(_card([
                    _section_label(
                        "Full Data Timeline",
                        "Three-year history with training, validation, and holdout periods annotated. "
                        "Orange PI ribbons = 8-week model forecast. Blue diamonds = holdout actuals.",
                    ),
                    dcc.Graph(
                        id="timeline-chart",
                        config={"displayModeBar": False},
                    ),
                ])),
            ], className="mb-4"),

            # ── Row 3: Biological Translation Table ───────────────────────────
            dbc.Row([
                dbc.Col(_card([
                    _section_label(
                        "Biological Translation Layer",
                        "What the model's numbers mean for clinicians and public health officials.",
                    ),
                    _build_bio_table(eval_result=eval_result),
                ])),
            ], className="mb-4"),

            # ── Row 4: VSN + Attention ─────────────────────────────────────────
            dbc.Row([
                dbc.Col(
                    _card([dcc.Graph(id="vsn-chart", config={"displayModeBar": False})]),
                    width=6,
                ),
                dbc.Col(
                    _card([dcc.Graph(id="attention-chart", config={"displayModeBar": False})]),
                    width=6,
                ),
            ], className="mb-4"),

            # ── Row 5: Suggested visuals (county WIS + CV folds) ──────────────
            *([] if not has_suggestions else [
                dbc.Row([
                    _section_label(
                        "Diagnostic Panels",
                        "Left: which county forecasts to trust most.  "
                        "Right: is the model's performance stable across training windows?",
                    ),
                ], className="mb-2"),
                dbc.Row([
                    *([] if eval_result is None else [
                        dbc.Col(
                            _card([dcc.Graph(
                                id="county-wis-chart",
                                figure=_build_county_wis_chart(eval_result),
                                config={"displayModeBar": False},
                            )]),
                            width=5,
                        ),
                    ]),
                    *([] if cv_results is None or cv_results.empty else [
                        dbc.Col(
                            _card([dcc.Graph(
                                id="cv-chart",
                                figure=_build_cv_chart(cv_results),
                                config={"displayModeBar": False},
                            )]),
                            width=7,
                        ),
                    ]),
                ], className="mb-4"),
            ]),

            # ── Footer ─────────────────────────────────────────────────────────
            dbc.Row(dbc.Col(html.P(
                "Data: CA Open Data Portal  ·  Model: Temporal Fusion Transformer  ·  "
                "Signals: log1p-transformed weekly new COVID-19 cases  ·  9-County Bay Area",
                style={"color": _MUTED, "fontSize": "11px", "textAlign": "center",
                       "paddingTop": "8px", "borderTop": f"1px solid {_BORDER}"},
            ))),

            dcc.Store(id="selected-county", data=None),
        ],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    @app.callback(
        Output("selected-county", "data"),
        Input("map-chart", "clickData"),
        State("selected-county", "data"),
    )
    def on_map_click(click_data, current_fips):
        if click_data is None:
            return current_fips
        try:
            fips = click_data["points"][0]["customdata"][1]
        except (KeyError, IndexError, TypeError):
            return current_fips
        return None if fips == current_fips else fips

    @app.callback(
        Output("hero-chart",            "figure"),
        Output("map-chart",             "figure"),
        Output("timeline-chart",        "figure"),
        Output("selected-county-badge", "children"),
        Input("selected-county", "data"),
    )
    def update_main_panels(county_fips):
        county_name = FIPS_TO_COUNTY.get(county_fips, county_fips) if county_fips else None
        badge = (
            f"Focused on: {county_name}  ·  8-week horizon  ·  click the county again to deselect"
            if county_name else
            "9-County Bay Area  ·  8-week horizon  ·  click a county on the map to focus"
        )
        hero     = create_hero_plot(processed_df, forecast_df, county_fips, q_cols)
        map_fig  = create_map(forecast_df, q_cols, selected_fips=county_fips)
        timeline = create_timeline_chart(processed_df, forecast_df, county_fips, q_cols)
        return hero, map_fig, timeline, badge

    @app.callback(
        Output("vsn-chart", "figure"),
        Input("selected-county", "data"),
    )
    def update_vsn(county_fips):
        if _vsn_weights:
            weights = _vsn_weights
            title   = "VSN Feature Importance"
        else:
            from src.models.tft_model import HIST_COVARIATES
            rng = np.random.default_rng(42)
            w   = rng.dirichlet(np.ones(len(HIST_COVARIATES)) * 0.6)
            weights = {"historical": w}
            title = "VSN Feature Importance (illustrative)"
        return plot_vsn_importance(weights, role="historical", title=title)

    @app.callback(
        Output("attention-chart", "figure"),
        Input("selected-county", "data"),
    )
    def update_attention(county_fips):
        weights = None
        if model is not None:
            try:
                weights = extract_attention_weights(model)
            except Exception:
                pass
        if weights is None:
            T = 16
            weights = np.zeros((4, T))
            for i in range(4):
                w = np.exp(-0.12 * np.arange(T)[::-1])
                w += 0.25 * np.exp(-0.5 * ((np.arange(T) - (T - 3 + i)) / 2.0) ** 2)
                weights[i] = w / w.sum()
        label = " (illustrative)" if model is None else ""
        return plot_attention_heatmap(
            weights,
            county=county_fips or "Bay Area",
            title=f"Temporal Attention Weights{label}",
        )

    return app


# ---------------------------------------------------------------------------
# Demo launcher (synthetic data)
# ---------------------------------------------------------------------------

def _build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng    = np.random.default_rng(0)
    dates  = pd.date_range("2020-07-01", periods=175, freq="W-WED")
    fips_list = list(BAY_AREA_FIPS.values())

    rows = []
    for ds in dates:
        t = (ds - dates[0]).days / 365
        for fips in fips_list:
            base = 2.5 + 1.5 * math.sin(2 * math.pi * t) + float(rng.normal(0, 0.3))
            rows.append({COUNTY_COL: fips, NWSS_DATE_COL: ds, TARGET_COL: max(0.0, base)})
    processed_df = pd.DataFrame(rows)

    forecast_rows = []
    for ds in dates[-8:]:
        for fips in fips_list:
            med = float(rng.exponential(2.0))
            forecast_rows.append({
                "unique_id": fips, "ds": ds,
                "TFT-lo-95.0": max(0.0, med - 1.2),
                "TFT-lo-80.0": max(0.0, med - 0.8),
                "TFT-lo-50.0": max(0.0, med - 0.4),
                "TFT-median":  med,
                "TFT-hi-50.0": med + 0.4,
                "TFT-hi-80.0": med + 0.8,
                "TFT-hi-95.0": med + 1.2,
            })
    return processed_df, pd.DataFrame(forecast_rows)


def run_demo(host: str = DASH_HOST, port: int = DASH_PORT, debug: bool = True) -> None:
    processed_df, forecast_df = _build_demo_data()
    app = create_app(processed_df=processed_df, forecast_df=forecast_df)
    logger.info("Sewer Surveillance demo dashboard → http://{}:{}", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_demo()
