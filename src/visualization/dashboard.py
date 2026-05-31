"""
Sewer Surveillance Dashboard — Bay Area Wastewater Surveillance
Attention-based outbreak forecasting with COVID-19 wastewater data.

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
        runs_dir=RUNS_DIR,   # optional — enables run selector
    )
    app.run(debug=False, port=8050)
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from loguru import logger

from src.config import (
    BAY_AREA_FIPS,
    COUNTY_COL,
    DASH_HOST,
    DASH_PORT,
    DATA_END_DATE,
    DATA_START_DATE,
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

_BG       = "#F4F7FB"
_SURFACE  = "#FFFFFF"
_SURFACE2 = "#EBF0F9"
_BORDER   = "#C5D0E4"
_TEXT     = "#162032"
_MUTED    = "#576880"
_ACCENT   = "#E8821C"
_BLUE     = "#1A4FA0"
_GREEN    = "#15803D"
_AMBER    = "#B45309"
_RED      = "#B91C1C"
_VIOLET   = "#6D28D9"

_PLOT_BG  = "#FFFFFF"
_PAPER_BG = "#FFFFFF"
_GRID     = "rgba(0,0,0,0.06)"

_FONT_BODY = {"family": "Inter, system-ui, sans-serif", "size": 14, "color": _TEXT}

_COUNTY_COLORS = [
    "#1A4FA0", "#E8821C", "#15803D", "#7C3AED",
    "#B91C1C", "#0284C7", "#047857", "#92400E", "#6D28D9",
]

# County centroids — loaded from the active geography at module use time.
# Falls back to Bay Area defaults when no geography has been loaded.
_BAY_AREA_CENTROIDS: dict[str, tuple[float, float]] = {
    "06001": (37.6017, -121.7195),
    "06013": (37.9147, -121.9235),
    "06041": (38.0834, -122.7633),
    "06055": (38.5025, -122.2654),
    "06075": (37.7749, -122.4194),
    "06081": (37.4350, -122.3185),
    "06085": (37.3328, -121.8956),
    "06095": (38.2494, -121.9018),
    "06097": (38.5780, -122.8759),
}


def _get_centroids() -> dict[str, tuple[float, float]]:
    """Return centroids for the currently active geography."""
    import src.config as _cfg
    geo = getattr(_cfg, "ACTIVE_GEOGRAPHY", None)
    if geo is not None and geo.centroids:
        return geo.centroids
    return _BAY_AREA_CENTROIDS


def _get_map_center() -> tuple[float, float, float]:
    """Return (lat, lon, zoom) for the currently active geography."""
    import src.config as _cfg
    geo = getattr(_cfg, "ACTIVE_GEOGRAPHY", None)
    if geo is not None:
        return geo.map_center_lat, geo.map_center_lon, geo.map_zoom
    return 37.7, -122.2, 7.8


def _get_geography_label() -> str:
    """Return display name for the currently active geography (e.g. 'SF Bay Area')."""
    import src.config as _cfg
    geo = getattr(_cfg, "ACTIVE_GEOGRAPHY", None)
    if geo is not None:
        return geo.name
    return "Bay Area"

_BIO_TABLE_DATA = [
    {
        "metric": "WIS",
        "what_it_is": "Weighted Interval Score — sharpness + calibration combined (Bracher 2021)",
        "bio_label": "Overall Forecast Quality",
        "bio_meaning": "Penalises both missed magnitude and interval width simultaneously. "
                       "0 = perfect.  Captures the full distribution, not just the median.",
        "target": "< 0.20 = excellent  |  < 0.50 = acceptable",
    },
    {
        "metric": "MAE",
        "what_it_is": "Mean Absolute Error of the median (P₅₀) forecast vs. actuals",
        "bio_label": "Point Forecast Accuracy",
        "bio_meaning": "Average distance from the forecast median to reality (log1p scale). "
                       "+0.19 log1p ≈ model predicts ~1.4× too many cases.",
        "target": "< 0.30 = good  |  < 0.10 = excellent",
    },
    {
        "metric": "Coverage 50%",
        "what_it_is": "Fraction of weeks where the actual fell inside the 25th–75th percentile band",
        "bio_label": "Median Calibration (critical)",
        "bio_meaning": "The most revealing calibration signal. Near 0% = systematic bias — "
                       "the median is displaced. Should be ≈ 50% for an unbiased model.",
        "target": "≈ 50%  (current ~11% = severe upward bias)",
    },
    {
        "metric": "Coverage 95%",
        "what_it_is": "Fraction of weeks where actual fell inside the 2.5th–97.5th percentile band",
        "bio_label": "Interval Width Signal",
        "bio_meaning": "100% is NOT necessarily good — it can mean intervals are trivially wide "
                       "(>3 log1p), catching everything by brute force, not precision.",
        "target": "≈ 95%  (100% = too wide / uninformative)",
    },
    {
        "metric": "Pinball q0.10",
        "what_it_is": "Pinball (quantile) loss at the 10th percentile — lower-tail calibration",
        "bio_label": "Lower-Tail Bias Direction",
        "bio_meaning": "If Pinball q0.10 >> Pinball q0.90, the lower tail is too high — "
                       "the model over-predicts even at the pessimistic lower bound. "
                       "Direction of required shift: downward.",
        "target": "Should ≈ Pinball q0.90  (symmetry = unbiased)",
    },
    {
        "metric": "Pinball q0.90",
        "what_it_is": "Pinball (quantile) loss at the 90th percentile — upper-tail calibration",
        "bio_label": "Upper-Tail Sharpness",
        "bio_meaning": "The mirror of q0.10. When q0.10 ≈ q0.90 the distribution is symmetric "
                       "and unbiased. q0.10 >> q0.90 = model predicts too many cases (shift DOWN). "
                       "q0.90 >> q0.10 = model predicts too few (shift UP).",
        "target": "Should ≈ Pinball q0.10  (symmetry = unbiased)",
    },
    {
        "metric": "Pinball Ratio",
        "what_it_is": "Pinball q0.10 ÷ Pinball q0.90 — directional bias signal",
        "bio_label": "Bias Direction & Magnitude",
        "bio_meaning": "The single most actionable calibration diagnostic. Ratio > 1 means the "
                       "lower tail is over-predicted — the entire distribution sits too high. "
                       "Ratio = 1.0 is unbiased. run_008 holdout: 7.81× (severe upward bias); "
                       "by Nov 2023 the rolling folds reach 0.90× (near-symmetric).",
        "target": "~1.0 = unbiased  |  > 2.0 = actionable  |  > 5.0 = severe",
    },
]
# Note: Precision, Recall, F1, and TTD are computed by the evaluation engine but
# are not shown in this table. The 2023 holdout is post-XBB endemic period with
# zero actual outbreak onsets above the p75 training threshold — detection metrics
# are undefined by construction, not a model failure. They become meaningful when
# evaluated against a window containing an active surge (e.g. Dec 2022 BQ.1/XBB.1.5).


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
    children = [html.H6(text, style={"color": _TEXT, "fontWeight": "600", "marginBottom": "2px"})]
    if sub:
        children.append(html.P(sub, style={"color": _MUTED, "fontSize": "12px", "marginBottom": "12px"}))
    return html.Div(children)


def _empty_fig(message: str = "No data", height: int = 350) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color=_MUTED))
    fig.update_layout(height=height, paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
                      font=_FONT_BODY, margin=dict(l=20, r=20, t=40, b=20))
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
            x=0.01, y=0.99, xanchor="left", yanchor="top",
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

def _build_bio_table(eval_dict: dict | None = None) -> dash_table.DataTable:
    def _fmt(v, pct: bool = False, days: bool = False) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if pct:
            return f"{v * 100:.1f}%"
        if days:
            return f"{v:.1f} days"
        return f"{v:.3f}"

    live: dict[str, str] = {}
    if eval_dict:
        live = {
            "WIS":           _fmt(eval_dict.get("mean_wis")),
            "MAE":           _fmt(eval_dict.get("mae")),
            "Coverage 50%":  _fmt(eval_dict.get("coverage_50"), pct=True),
            "Coverage 95%":  _fmt(eval_dict.get("coverage_95"), pct=True),
            "Pinball q0.10": _fmt(eval_dict.get("pinball_q010")),
            "Pinball q0.90": _fmt(eval_dict.get("pinball_q090")),
            "Pinball Ratio": _fmt(eval_dict.get("pinball_ratio")),
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
        data=rows, columns=columns,
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_header={"backgroundColor": _SURFACE2, "color": _TEXT, "fontWeight": "600",
                      "fontSize": "13px", "border": f"1px solid {_BORDER}",
                      "textTransform": "uppercase", "letterSpacing": "0.05em"},
        style_cell={"backgroundColor": _SURFACE, "color": _TEXT, "border": f"1px solid {_BORDER}",
                    "padding": "10px 14px", "fontSize": "13px",
                    "fontFamily": "Inter, system-ui, sans-serif",
                    "whiteSpace": "normal", "height": "auto", "textAlign": "left"},
        style_data_conditional=cond,
        page_action="none", sort_action="native",
    )


# ---------------------------------------------------------------------------
# Forecasted Case Density Map  (Scattermapbox — open-street-map)
# ---------------------------------------------------------------------------

def create_map(
    forecast_df: pd.DataFrame,
    q_cols: QuantileColumns,
    selected_fips: str | None = None,
) -> go.Figure:
    """Bay Area bubble map coloured by 8-week median forecast density.

    Only counties present in forecast_df are shown as active bubbles;
    remaining Bay Area counties appear as small gray placeholders so the
    map always renders the full Bay Area geography.
    """
    if forecast_df.empty or q_cols.q50 not in forecast_df.columns:
        return _empty_fig("No forecast data available", height=400)

    # Build per-county density
    forecast_fips = set(forecast_df["unique_id"].astype(str).unique())
    rows = []
    for fips, (lat, lon) in _get_centroids().items():
        sub = forecast_df[forecast_df["unique_id"] == fips]
        density = float(sub[q_cols.q50].sum()) if not sub.empty else 0.0
        in_run  = fips in forecast_fips
        rows.append({
            "fips": fips, "name": FIPS_TO_COUNTY.get(fips, fips),
            "lat": lat, "lon": lon, "density": density, "in_run": in_run,
        })

    df_map = pd.DataFrame(rows)
    active  = df_map[df_map["in_run"]]
    mx = active["density"].max() if not active.empty else 1.0
    df_map["norm"] = df_map["density"] / (mx if mx > 0 else 1.0)

    def _color(row) -> str:
        if not row["in_run"]:
            return "#CCCCCC"
        if row["fips"] == selected_fips:
            return _ACCENT
        n = row["norm"]
        return _GREEN if n < 0.33 else (_AMBER if n < 0.66 else _RED)

    def _size(row) -> int:
        if not row["in_run"]:
            return 14
        return max(22, int(52 * row["norm"]))

    def _opacity(row) -> float:
        if not row["in_run"]:
            return 0.35
        return 1.0 if (row["fips"] == selected_fips or selected_fips is None) else 0.55

    df_map["color"]   = df_map.apply(_color, axis=1)
    df_map["size"]    = df_map.apply(_size, axis=1)
    df_map["opacity"] = df_map.apply(_opacity, axis=1)

    fig = go.Figure(go.Scattermapbox(
        lat=df_map["lat"],
        lon=df_map["lon"],
        text=df_map["name"],
        customdata=df_map[["density", "fips", "in_run"]].values,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color=_TEXT),
        marker=dict(
            size=df_map["size"].tolist(),
            color=df_map["color"].tolist(),
            opacity=df_map["opacity"].tolist(),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "8-wk Forecast Σ: %{customdata[0]:.2f}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
    ))

    # Halo ring for selected county
    if selected_fips and selected_fips in df_map["fips"].values:
        sel = df_map[df_map["fips"] == selected_fips].iloc[0]
        sel_idx = df_map.index[df_map["fips"] == selected_fips][0]
        fig.add_trace(go.Scattermapbox(
            lat=[sel["lat"]], lon=[sel["lon"]],
            mode="markers",
            marker=dict(size=df_map.loc[sel_idx, "size"] + 18, color=_ACCENT, opacity=0.22),
            hoverinfo="skip", showlegend=False,
        ))

    subtitle = (
        f"— {FIPS_TO_COUNTY.get(selected_fips, '')} selected · click to deselect"
        if selected_fips else "· click a county to focus"
    )
    n_active = int(df_map["in_run"].sum())
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=_get_map_center()[0], lon=_get_map_center()[1]),
                    zoom=_get_map_center()[2]),
        title=dict(
            text=f"Forecasted Case Density ({n_active} counties) {subtitle}",
            font=dict(size=13, color=_TEXT),
        ),
        paper_bgcolor=_PAPER_BG,
        font=_FONT_BODY,
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Classifier timeline  (Z-score + triggered bands per county)
# ---------------------------------------------------------------------------

_SURGE_BAND_COLOR    = "rgba(234,179,8,0.08)"    # California gold — Stage 2 active (not orange)
_SURGE_MARKER_COLOR  = "rgba(161,123,0,0.9)"     # darker gold for annotation text
_SUPPRESS_BAND_COLOR = "rgba(100,100,100,0.10)"  # gray — suppressed

# Hero chart — PI fills and observed line colors (Phase 6 color refresh)
# Old scheme had PI bands, median, and TFT-active regions all in orange shades.
# New: PI bands → California navy blue; TFT-active → gold; median stays orange (the only orange).
_PI_95_FILL   = "rgba(26,79,160,0.11)"   # California navy, very light — 95% PI band
_PI_50_FILL   = "rgba(26,79,160,0.28)"   # California navy, medium — 50% PI band
_OBS_CONTEXT  = "#64748B"                # slate gray — pre-forecast context line
_OBS_HOLDOUT  = "#0C4A6E"               # Pacific deep blue — holdout actuals


def create_classifier_timeline(
    clf_df: pd.DataFrame,
    county_fips: str | None = None,
) -> go.Figure:
    """Z-score time-series per county with triggered/suppressed shading.

    Orange background = Stage 2 (TFT) invoked.
    No shading = classifier suppressed; flat quiet prior returned.
    """
    if clf_df is None or clf_df.empty:
        return _empty_fig(
            "No classifier data — run with --two-stage to see gatekeeper activity.",
            height=280,
        )

    fig      = go.Figure()
    all_fips = sorted(clf_df["unique_id"].astype(str).unique())
    fips_list = [county_fips] if county_fips and county_fips in all_fips else all_fips

    for i, fips in enumerate(fips_list):
        sub  = clf_df[clf_df["unique_id"] == fips].sort_values("date")
        name = FIPS_TO_COUNTY.get(fips, fips)
        color = _COUNTY_COLORS[i % len(_COUNTY_COLORS)]

        if sub.empty:
            continue

        # Z-score line
        if "z_score" in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["z_score"].clip(-4, 8),
                name=f"{name} — Z-score",
                line=dict(color=color, width=1.8),
                mode="lines",
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>Z-score: %{{y:.2f}}<extra></extra>",
            ))

        # Scatter markers where triggered
        triggered = sub[sub["triggered"] == True]  # noqa: E712
        if not triggered.empty and "z_score" in triggered.columns:
            fig.add_trace(go.Scatter(
                x=triggered["date"],
                y=triggered["z_score"].clip(-4, 8),
                name=f"{name} — TFT active",
                mode="markers",
                marker=dict(
                    color=_SURGE_MARKER_COLOR, size=10,
                    symbol="circle", line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>"
                    "Z-score: %{y:.2f}<br><b>Stage 2 (TFT) ACTIVE</b><extra></extra>"
                ),
                showlegend=False,
            ))

        # Background shading for triggered runs
        in_run   = False
        run_start = None
        dates    = sub["date"].tolist()
        trigs    = sub["triggered"].tolist()
        for date, trig in zip(dates, trigs):
            if trig and not in_run:
                run_start = date
                in_run    = True
            elif not trig and in_run:
                fig.add_vrect(
                    x0=run_start, x1=date,
                    fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
                    annotation_text="TFT", annotation_position="top left",
                    annotation_font=dict(color=_SURGE_MARKER_COLOR, size=9),
                )
                in_run = False
        if in_run:  # close open run at end of series
            fig.add_vrect(
                x0=run_start, x1=dates[-1],
                fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
            )

    # Z-threshold reference line
    fig.add_hline(
        y=1.5, line=dict(color=_SURGE_MARKER_COLOR, dash="dot", width=1.2),
        annotation_text="Z threshold (1.5)",
        annotation_font=dict(color=_SURGE_MARKER_COLOR, size=10),
        annotation_position="right",
    )
    fig.add_hline(y=0, line=dict(color=_MUTED, dash="solid", width=0.5))

    fig.update_layout(
        **_base_layout("Outbreak Classifier — Z-score vs Quiet Baseline", height=280),
        xaxis=_axis("Date"),
        yaxis=_axis("Z-score  (σ above training quiet-period mean)"),
        showlegend=(county_fips is None),
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
    clf_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """8-week forecast hero chart.

    county_fips=None → all-counties spaghetti of medians.
    county_fips=str  → single county with 50%/95% PI ribbons and actuals.

    clf_df: optional OutbreakClassifier output.  When provided and a county
    is selected, orange background shading marks Stage 2 (TFT) active windows;
    suppressed forecast rows (``_suppressed == True``) are rendered as a
    dashed gray line instead of the solid orange median.
    """
    if forecast_df.empty or q_cols.q50 not in forecast_df.columns:
        return _empty_fig("No forecast data — run the pipeline first", height=440)

    fig = go.Figure()
    all_fips = sorted(forecast_df["unique_id"].astype(str).unique())

    if county_fips is None:
        # ── All-counties median spaghetti ─────────────────────────────────────
        for i, fips in enumerate(all_fips):
            sub = forecast_df[forecast_df["unique_id"] == fips].sort_values("ds")
            if sub.empty:
                continue
            color = _COUNTY_COLORS[i % len(_COUNTY_COLORS)]
            name  = FIPS_TO_COUNTY.get(fips, fips)
            fig.add_trace(go.Scatter(
                x=sub["ds"], y=sub[q_cols.q50],
                name=name, mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>Median: %{{y:.2f}}<extra></extra>",
            ))
        title = f"8-Week Forecast — {len(all_fips)} {'County' if len(all_fips)==1 else 'Counties'}  ·  click map to focus"

    else:
        # ── Single-county PI ribbon chart ─────────────────────────────────────
        fcast  = forecast_df[forecast_df["unique_id"] == county_fips].sort_values("ds")
        county_col = COUNTY_COL if COUNTY_COL in actual_df.columns else "unique_id"
        actual = actual_df[actual_df[county_col] == county_fips].sort_values(NWSS_DATE_COL)

        if fcast.empty:
            return _empty_fig(f"No forecast for {FIPS_TO_COUNTY.get(county_fips, county_fips)}", height=440)

        fcast_start    = pd.Timestamp(fcast["ds"].min())
        context_cutoff = fcast_start - pd.Timedelta(weeks=context_weeks)
        actual_pre     = actual[(actual[NWSS_DATE_COL] >= context_cutoff) & (actual[NWSS_DATE_COL] < fcast_start)]
        actual_holdout = actual[actual[NWSS_DATE_COL] >= fcast_start]

        # PI bands — widest first so narrower bands render on top
        def _band(lo_col: str, hi_col: str, fill_color: str, name: str) -> None:
            if lo_col not in fcast.columns or hi_col not in fcast.columns:
                return
            lo_v, hi_v = fcast[lo_col], fcast[hi_col]
            x_fill = list(fcast["ds"]) + list(fcast["ds"])[::-1]
            y_fill = list(lo_v) + list(hi_v)[::-1]
            fig.add_trace(go.Scatter(
                x=x_fill, y=y_fill, fill="toself",
                fillcolor=fill_color, line=dict(color="rgba(0,0,0,0)"),
                name=name, showlegend=True, hoverinfo="skip",
            ))

        _band(q_cols.q025, q_cols.q975, _PI_95_FILL, "95% PI")
        _band(q_cols.q25,  q_cols.q75,  _PI_50_FILL, "50% PI")

        # Median forecast line
        fig.add_trace(go.Scatter(
            x=fcast["ds"], y=fcast[q_cols.q50],
            name="Median Forecast",
            line=dict(color=_ACCENT, width=3),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.2f}<extra></extra>",
        ))

        # Context actuals (pre-forecast window)
        if not actual_pre.empty and TARGET_COL in actual_pre.columns:
            fig.add_trace(go.Scatter(
                x=actual_pre[NWSS_DATE_COL], y=actual_pre[TARGET_COL],
                name="Observed (context)", mode="lines+markers",
                line=dict(color=_OBS_CONTEXT, width=2), marker=dict(size=5, color=_OBS_CONTEXT),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Observed: %{y:.2f}<extra></extra>",
            ))

        # Holdout actuals during forecast window — direct comparison
        if not actual_holdout.empty and TARGET_COL in actual_holdout.columns:
            fig.add_trace(go.Scatter(
                x=actual_holdout[NWSS_DATE_COL], y=actual_holdout[TARGET_COL],
                name="Observed (holdout)", mode="lines+markers",
                line=dict(color=_OBS_HOLDOUT, width=2.5),
                marker=dict(size=8, color=_OBS_HOLDOUT, symbol="diamond"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Holdout actual: %{y:.2f}<extra></extra>",
            ))

        # ── Classification overlay (orange = Stage 2 active; gray dashed = quiet prior) ──
        if clf_df is not None and not clf_df.empty:
            sub_clf = clf_df[clf_df["unique_id"] == county_fips].sort_values("date")
            if not sub_clf.empty:
                # Orange shading for triggered weeks that overlap the forecast window
                in_run, run_start = False, None
                for _, row in sub_clf.iterrows():
                    if row["triggered"] and not in_run:
                        run_start, in_run = row["date"], True
                    elif not row["triggered"] and in_run:
                        fig.add_vrect(
                            x0=run_start, x1=row["date"],
                            fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
                            annotation_text="TFT active",
                            annotation_font=dict(color=_SURGE_MARKER_COLOR, size=9),
                            annotation_position="top left",
                        )
                        in_run = False
                if in_run:
                    fig.add_vrect(
                        x0=run_start, x1=sub_clf["date"].iloc[-1],
                        fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
                    )

        # Suppressed forecast rows rendered as a separate dashed gray line
        if "_suppressed" in fcast.columns:
            suppressed = fcast[fcast["_suppressed"] == True]  # noqa: E712
            active     = fcast[fcast["_suppressed"] != True]  # noqa: E712
            if not suppressed.empty and q_cols.q50 in suppressed.columns:
                fig.add_trace(go.Scatter(
                    x=suppressed["ds"], y=suppressed[q_cols.q50],
                    name="Quiet prior (suppressed)",
                    line=dict(color="rgba(160,160,160,0.7)", width=2, dash="dot"),
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Quiet prior: %{y:.2f}<extra></extra>",
                ))
            if not active.empty and q_cols.q50 in active.columns:
                # Replace the default median trace with just the active subset
                # (already added above as "Median Forecast" — this is a supplementary note)
                pass  # the full fcast trace above shows both; suppressed just gets a style note

        # Forecast-start separator line
        fig.add_vline(
            x=fcast_start.to_pydatetime(),
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
    clf_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Full historical + forecast timeline with period shading.

    When clf_df is provided and a county is selected, classifier-triggered
    windows are highlighted with an orange overlay on the full 3-year timeline.
    """
    fig = go.Figure()

    def _dt(s):
        return datetime.strptime(s, "%Y-%m-%d")
    train_end  = _dt(TRAIN_END_DATE)
    val_end    = _dt(VAL_END_DATE)
    data_end   = _dt(DATA_END_DATE)
    data_start = _dt(DATA_START_DATE)

    def _vrect(x0, x1, color: str, label: str) -> None:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=color, opacity=0.10,
            layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_color=_MUTED, annotation_font_size=10,
        )

    _vrect(data_start, train_end, _BLUE,  "Training")
    _vrect(train_end,  val_end,   _AMBER, "Validation")
    _vrect(val_end,    data_end,  _RED,   "Holdout")

    fcast_end_ts = forecast_df["ds"].max() if not forecast_df.empty else pd.Timestamp(DATA_END_DATE)
    fcast_end    = pd.Timestamp(fcast_end_ts).to_pydatetime()
    if fcast_end > data_end:
        _vrect(data_end, fcast_end, _ACCENT, "Forecast")

    county_col = COUNTY_COL if COUNTY_COL in actual_df.columns else "unique_id"

    if county_fips is not None:
        act = actual_df[actual_df[county_col] == county_fips].sort_values(NWSS_DATE_COL)
        if not act.empty and TARGET_COL in act.columns:
            fig.add_trace(go.Scatter(
                x=act[NWSS_DATE_COL], y=act[TARGET_COL],
                name=f"{FIPS_TO_COUNTY.get(county_fips, county_fips)} — Observed",
                mode="lines+markers",
                line=dict(color=_TEXT, width=2), marker=dict(size=4, color=_TEXT),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Observed: %{y:.2f}<extra></extra>",
            ))

        fcast = forecast_df[forecast_df["unique_id"] == county_fips].sort_values("ds")
        if not fcast.empty and q_cols.q50 in fcast.columns:
            if q_cols.q025 in fcast.columns and q_cols.q975 in fcast.columns:
                x_b = list(fcast["ds"]) + list(fcast["ds"])[::-1]
                y_b = list(fcast[q_cols.q025]) + list(fcast[q_cols.q975])[::-1]
                fig.add_trace(go.Scatter(x=x_b, y=y_b, fill="toself",
                                         fillcolor=_PI_95_FILL,
                                         line=dict(color="rgba(0,0,0,0)"),
                                         name="95% PI", hoverinfo="skip"))
            if q_cols.q25 in fcast.columns and q_cols.q75 in fcast.columns:
                x_b = list(fcast["ds"]) + list(fcast["ds"])[::-1]
                y_b = list(fcast[q_cols.q25]) + list(fcast[q_cols.q75])[::-1]
                fig.add_trace(go.Scatter(x=x_b, y=y_b, fill="toself",
                                         fillcolor=_PI_50_FILL,
                                         line=dict(color="rgba(0,0,0,0)"),
                                         name="50% PI", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=fcast["ds"], y=fcast[q_cols.q50],
                name="Forecast Median",
                line=dict(color=_ACCENT, width=2.5, dash="dot"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.2f}<extra></extra>",
            ))
        title_suffix = FIPS_TO_COUNTY.get(county_fips, county_fips)

    else:
        bay = actual_df.groupby(NWSS_DATE_COL)[TARGET_COL].mean().reset_index().sort_values(NWSS_DATE_COL) \
              if TARGET_COL in actual_df.columns else pd.DataFrame()
        for i, fips in enumerate(sorted(actual_df[county_col].unique())):
            act = actual_df[actual_df[county_col] == fips].sort_values(NWSS_DATE_COL)
            if act.empty or TARGET_COL not in act.columns:
                continue
            fig.add_trace(go.Scatter(
                x=act[NWSS_DATE_COL], y=act[TARGET_COL],
                name=FIPS_TO_COUNTY.get(fips, fips), mode="lines",
                line=dict(color=_COUNTY_COLORS[i % len(_COUNTY_COLORS)], width=1.2), opacity=0.45,
            ))
        if not bay.empty:
            fig.add_trace(go.Scatter(
                x=bay[NWSS_DATE_COL], y=bay[TARGET_COL], name=f"{_get_geography_label()} Avg",
                mode="lines", line=dict(color=_TEXT, width=3), opacity=0.9,
            ))
        title_suffix = f"All {len(sorted(actual_df[county_col].unique()))} Counties"

    for ts in [train_end, val_end, data_end]:
        fig.add_vline(x=ts, line=dict(color=_BORDER, dash="dot", width=1))

    # ── Classifier overlay: orange bands for Stage 2 active windows ──────────
    if clf_df is not None and not clf_df.empty and county_fips is not None:
        sub_clf = clf_df[clf_df["unique_id"] == county_fips].sort_values("date")
        in_run, run_start = False, None
        for _, row in sub_clf.iterrows():
            if row["triggered"] and not in_run:
                run_start, in_run = row["date"], True
            elif not row["triggered"] and in_run:
                fig.add_vrect(
                    x0=run_start, x1=row["date"],
                    fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
                )
                in_run = False
        if in_run:
            fig.add_vrect(
                x0=run_start, x1=sub_clf["date"].iloc[-1],
                fillcolor=_SURGE_BAND_COLOR, line_width=0, layer="below",
            )

    fig.update_layout(
        **_base_layout(f"Full Data Timeline — {title_suffix}", height=340,
                        margin=dict(l=70, r=30, t=55, b=55)),
        xaxis=_axis("Date", range=[data_start, fcast_end]),
        yaxis=_axis("Weekly New Cases (log1p)"),
    )
    return fig


# ---------------------------------------------------------------------------
# CV fold chart
# ---------------------------------------------------------------------------

def _build_cv_chart(cv_df: pd.DataFrame) -> go.Figure:
    if cv_df.empty or "cutoff_date" not in cv_df.columns:
        return _empty_fig("No CV results available", height=320)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cv_df["cutoff_date"], y=cv_df["mean_wis"], name="WIS",
        mode="lines+markers", line=dict(color=_ACCENT, width=2), marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>WIS: %{y:.3f}<extra></extra>", yaxis="y1",
    ))
    if "smape" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=cv_df["cutoff_date"], y=cv_df["smape"] * 100, name="SMAPE %",
            mode="lines+markers", line=dict(color=_VIOLET, width=2, dash="dashdot"), marker=dict(size=6),
            hovertemplate="<b>%{x|%b %Y}</b><br>SMAPE: %{y:.1f}%%<extra></extra>", yaxis="y1",
        ))
    if "coverage_95" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=cv_df["cutoff_date"], y=cv_df["coverage_95"] * 100, name="Coverage 95%",
            mode="lines+markers", line=dict(color=_BLUE, width=2, dash="dot"), marker=dict(size=6),
            hovertemplate="<b>%{x|%b %Y}</b><br>Coverage: %{y:.1f}%%<extra></extra>", yaxis="y2",
        ))
        fig.add_hline(y=95, yref="y2", line=dict(color=_BLUE, dash="longdash", width=1),
                      annotation_text="95% target", annotation_font_color=_BLUE,
                      annotation_position="top right")

    fig.update_layout(
        **_base_layout("CV Fold Performance — Temporal Stability", height=320,
                        margin=dict(l=70, r=70, t=55, b=60)),
        xaxis=_axis("CV Cutoff Date"),
        yaxis=dict(title=dict(text="WIS / SMAPE %", font=dict(size=12, color=_ACCENT)),
                   tickfont=dict(size=11, color=_MUTED), gridcolor=_GRID, zeroline=False, side="left"),
        yaxis2=dict(title=dict(text="Coverage 95% (%)", font=dict(size=12, color=_BLUE)),
                    tickfont=dict(size=11, color=_BLUE), overlaying="y", side="right",
                    range=[0, 105], zeroline=False, showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Run metrics summary bar  (compact scorecard strip)
# ---------------------------------------------------------------------------

def _build_run_scorecard(eval_dict: dict) -> html.Div:
    """Compact KPI strip shown under the run selector when a run is chosen."""
    def _chip(label: str, value: str, color: str) -> html.Span:
        return html.Span([
            html.Span(label, style={"fontSize": "10px", "color": _MUTED, "marginRight": "4px"}),
            html.Span(value, style={"fontSize": "13px", "fontWeight": "700", "color": color}),
        ], style={
            "backgroundColor": _SURFACE2,
            "border": f"1px solid {_BORDER}",
            "borderRadius": "6px",
            "padding": "4px 10px",
            "marginRight": "8px",
            "display": "inline-block",
        })

    def _fmt(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else "—"
    def _fmtpct(v) -> str:
        return f"{v*100:.1f}%" if isinstance(v, float) and not math.isnan(v) else "—"

    wis   = eval_dict.get("mean_wis",     float("nan"))
    cov95 = eval_dict.get("coverage_95",  float("nan"))
    cov50 = eval_dict.get("coverage_50",  float("nan"))
    smape = eval_dict.get("smape",        float("nan"))
    auc   = eval_dict.get("auc",          float("nan"))

    cov95_color = _GREEN if (isinstance(cov95, float) and not math.isnan(cov95) and cov95 >= 0.80) \
                 else (_AMBER if (isinstance(cov95, float) and not math.isnan(cov95) and cov95 >= 0.40) \
                 else _RED)
    wis_color   = _GREEN if (isinstance(wis, float) and not math.isnan(wis) and wis < 0.2) \
                 else (_AMBER if (isinstance(wis, float) and not math.isnan(wis) and wis < 0.5) \
                 else _RED)

    return html.Div([
        _chip("WIS",       _fmt(wis),    wis_color),
        _chip("Cov95",     _fmtpct(cov95), cov95_color),
        _chip("Cov50",     _fmtpct(cov50), _MUTED),
        _chip("SMAPE",     _fmt(smape),  _MUTED),
        _chip("AUC",       _fmt(auc),    _MUTED),
    ], style={"marginTop": "6px"})


# ---------------------------------------------------------------------------
# Tier layout helper
# ---------------------------------------------------------------------------

def _tier_header(tier_num: int, title: str, description: str) -> html.Div:
    """Prominent section divider for each dashboard tier."""
    palette = {1: _ACCENT, 2: _BLUE, 3: _VIOLET}
    color   = palette.get(tier_num, _TEXT)
    return html.Div([
        html.Div([
            html.Span(
                f"TIER {tier_num}",
                style={
                    "backgroundColor": color, "color": "#fff",
                    "fontSize": "10px", "fontWeight": "700",
                    "letterSpacing": "0.12em", "padding": "2px 9px",
                    "borderRadius": "3px", "marginRight": "10px",
                    "verticalAlign": "middle", "display": "inline-block",
                },
            ),
            html.Span(title, style={
                "fontSize": "17px", "fontWeight": "700",
                "color": _TEXT, "verticalAlign": "middle",
            }),
        ], style={"marginBottom": "4px"}),
        html.P(description, style={
            "color": _MUTED, "fontSize": "13px", "marginBottom": "6px",
        }),
        html.Hr(style={"borderColor": _BORDER, "marginTop": "2px", "marginBottom": "16px"}),
    ])


# ---------------------------------------------------------------------------
# Tier 1 — Current Status Panel
# ---------------------------------------------------------------------------

def _build_current_status_panel(
    forecast_df:   pd.DataFrame,
    q_cols:        QuantileColumns,
    clf_df:        pd.DataFrame,
    county_fips:   str | None,
    eval_dict:     dict,
    processed_df:  pd.DataFrame | None = None,
    selected_date: str | None          = None,
) -> html.Div:
    """Phase badge, P50/P95 KPIs, classifier Z-score, and alert text.

    When ``selected_date`` is provided (from hero chart clickData) the panel
    shows metrics at that exact date instead of the dataset's last date.  This
    lets the audience scrub through the timeline and see phase transitions live.
    """
    # ── Resolve the target date ───────────────────────────────────────────────
    # Normalise to date-only so W-WED timestamps match regardless of time component
    def _normalize(ts) -> pd.Timestamp | None:
        try:
            return pd.Timestamp(ts).normalize()
        except Exception:
            return None

    target_dt = _normalize(selected_date) if selected_date else None

    def _row_at_date(df: pd.DataFrame, date_col: str) -> pd.Series | None:
        """Return the row (for county_fips) closest to target_dt, or None."""
        if df is None or df.empty or date_col not in df.columns:
            return None
        sub = df[df["unique_id"] == county_fips].copy() if county_fips else df.copy()
        if sub.empty:
            return None
        sub["_nd"] = pd.to_datetime(sub[date_col]).dt.normalize()
        match = sub[sub["_nd"] == target_dt] if target_dt is not None else pd.DataFrame()
        if not match.empty:
            return match.iloc[0]
        if target_dt is None:
            return sub.sort_values(date_col).iloc[-1]
        return None   # date not found — we'll show "—"

    # ── Phase status from classifier ─────────────────────────────────────────
    triggered  = False
    z_score    = float("nan")
    n_surge_wk = 0
    momentum   = float("nan")

    if clf_df is not None and not clf_df.empty:
        sub = (
            clf_df[clf_df["unique_id"] == county_fips]
            if county_fips else clf_df
        ).sort_values("date").copy()
        sub["_nd"] = pd.to_datetime(sub["date"]).dt.normalize()

        if target_dt is not None:
            row_clf = sub[sub["_nd"] == target_dt]
            row_clf = row_clf.iloc[0] if not row_clf.empty else None
        else:
            row_clf = sub.iloc[-1] if not sub.empty else None

        if row_clf is not None:
            triggered  = bool(row_clf.get("triggered", False))
            z_score    = float(row_clf.get("z_score",   float("nan")))
            momentum   = float(row_clf.get("momentum",  float("nan")))
        n_surge_wk = int(sub["triggered"].sum())

    # ── Forecast KPIs at target date ─────────────────────────────────────────
    p50 = p95 = float("nan")
    if not forecast_df.empty and q_cols.q50 in forecast_df.columns:
        sub_fc = (
            forecast_df[forecast_df["unique_id"] == county_fips]
            if county_fips else forecast_df
        ).sort_values("ds").copy()
        sub_fc["_nd"] = pd.to_datetime(sub_fc["ds"]).dt.normalize()

        row_fc = (
            sub_fc[sub_fc["_nd"] == target_dt].iloc[0]
            if (target_dt is not None and not sub_fc[sub_fc["_nd"] == target_dt].empty)
            else (sub_fc.iloc[-1] if not sub_fc.empty else None)
        )
        if row_fc is not None:
            p50 = float(row_fc.get(q_cols.q50,  float("nan")))
            p95 = float(row_fc.get(q_cols.q975, float("nan")))

    # ── Actual value + coverage at target date ───────────────────────────────
    actual_val = float("nan")
    inside_pi  = None      # True / False / None (unknown)
    if processed_df is not None and not processed_df.empty and TARGET_COL in processed_df.columns:
        county_col = COUNTY_COL if COUNTY_COL in processed_df.columns else "unique_id"
        act_sub = (
            processed_df[processed_df[county_col] == county_fips]
            if county_fips else processed_df
        ).copy()
        act_sub["_nd"] = pd.to_datetime(act_sub[NWSS_DATE_COL]).dt.normalize()
        row_act = (
            act_sub[act_sub["_nd"] == target_dt].iloc[0]
            if (target_dt is not None and not act_sub[act_sub["_nd"] == target_dt].empty)
            else (act_sub.sort_values(NWSS_DATE_COL).iloc[-1] if not act_sub.empty else None)
        )
        if row_act is not None:
            actual_val = float(row_act.get(TARGET_COL, float("nan")))
            if (not math.isnan(actual_val) and not math.isnan(p50)
                    and q_cols.q025 in forecast_df.columns):
                p025 = float(row_fc.get(q_cols.q025, float("nan"))) if row_fc is not None else float("nan")  # type: ignore[union-attr]
                p975 = p95
                if not math.isnan(p025) and not math.isnan(p975):
                    inside_pi = bool(p025 <= actual_val <= p975)

    def _fmt(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if not math.isnan(v) else "—"

    # ── Date context line (scrub mode vs default) ─────────────────────────────
    if target_dt is not None:
        date_label = target_dt.strftime("%b %d, %Y")
        mode_note  = f"📅  Viewing  {date_label}  ·  click another point or deselect to reset"
    else:
        date_label = "latest"
        mode_note  = "Click any point on the forecast chart above to explore that date's metrics."

    # ── Phase badge ──────────────────────────────────────────────────────────
    phase_label = "OUTBREAK" if triggered else "BASELINE"
    phase_color = _RED      if triggered else _GREEN
    phase_bg    = "#FEF2F2" if triggered else "#F0FDF4"
    phase_icon  = "⚠" if triggered else "✓"

    badge = html.Div([
        html.Div(phase_icon, style={"fontSize": "28px", "marginBottom": "2px"}),
        html.Div(phase_label, style={
            "fontSize": "15px", "fontWeight": "800",
            "color": phase_color, "letterSpacing": "0.08em",
        }),
        html.Div("Phase Status", style={"fontSize": "10px", "color": _MUTED, "marginTop": "2px"}),
    ], style={
        "backgroundColor": phase_bg,
        "border": f"2px solid {phase_color}",
        "borderRadius": "10px",
        "padding": "14px 20px",
        "textAlign": "center",
        "minWidth": "120px",
    })

    # ── KPI chips ────────────────────────────────────────────────────────────
    def _kpi(label: str, value: str, color: str = _TEXT, sub: str = "") -> html.Div:
        return html.Div([
            html.Div(label, style={
                "fontSize": "10px", "color": _MUTED, "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            }),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "700", "color": color, "lineHeight": "1.2"}),
            html.Div(sub, style={"fontSize": "10px", "color": _MUTED}),
        ], style={
            "textAlign": "center", "padding": "10px 18px",
            "borderRight": f"1px solid {_BORDER}",
        })

    p50_color = _RED if (not math.isnan(p50) and triggered) else _TEXT
    z_color   = _RED if (not math.isnan(z_score) and z_score >= 1.5) else _MUTED

    # Coverage chip: "✓ Inside" / "✗ Outside" / "—" depending on availability
    if inside_pi is True:
        cov_val, cov_color, cov_sub = "✓ Inside", _GREEN, "actual in 95% PI"
    elif inside_pi is False:
        cov_val, cov_color, cov_sub = "✗ Outside", _RED, "actual outside 95% PI"
    else:
        cov_val, cov_color, cov_sub = "—", _MUTED, "95% PI coverage"

    # When a date is selected show 5 KPIs (add Actual + Coverage); otherwise 4
    if target_dt is not None:
        kpi_cols = [
            dbc.Col(_kpi("Actual",        _fmt(actual_val), _BLUE if not math.isnan(actual_val) else _MUTED, "observed (log1p)"), width=2),
            dbc.Col(_kpi("Forecast P₅₀",  _fmt(p50), p50_color, "median (log1p)"), width=2),
            dbc.Col(_kpi("Forecast P₉₅",  _fmt(p95), _RED if triggered else _MUTED, "upper bound"), width=2),
            dbc.Col(_kpi("Classifier Z",  _fmt(z_score, 2), z_color, "σ above baseline"), width=3),
            dbc.Col(_kpi("Coverage",      cov_val, cov_color, cov_sub), width=3),
        ]
    else:
        kpi_cols = [
            dbc.Col(_kpi("Forecast P₅₀", _fmt(p50), p50_color, "median (log1p)"), width=3),
            dbc.Col(_kpi("Forecast P₉₅", _fmt(p95), _RED if triggered else _MUTED, "upper bound"), width=3),
            dbc.Col(_kpi("Classifier Z",  _fmt(z_score, 2), z_color, "σ above baseline"), width=3),
            dbc.Col(_kpi("Surge Weeks",   str(n_surge_wk), _RED if n_surge_wk > 0 else _MUTED, "total triggered"), width=3),
        ]

    kpis = html.Div([
        dbc.Row(kpi_cols, className="g-0"),
    ], style={"border": f"1px solid {_BORDER}", "borderRadius": "8px", "overflow": "hidden"})

    # ── Alert / narrative text ────────────────────────────────────────────────
    date_ctx = f" on {date_label}" if target_dt is not None else ""
    if triggered:
        z_str = f"Z={z_score:.2f}σ" if not math.isnan(z_score) else ""
        m_str = f"  ·  momentum={momentum:.2f}" if not math.isnan(momentum) else ""
        alert_div = html.Div([
            html.Div(f"⚠  SURGE DETECTED{date_ctx} — Stage 2 (TFT) Forecaster Invoked", style={
                "fontWeight": "700", "color": _RED, "fontSize": "14px", "marginBottom": "4px",
            }),
            html.Div(
                f"WW signal elevated above training quiet-period baseline.  {z_str}{m_str}  ·  "
                f"{n_surge_wk} triggered weeks in series.  Forecast generated by full TFT model.",
                style={"color": _MUTED, "fontSize": "12px"},
            ),
        ], style={
            "padding": "12px 16px",
            "backgroundColor": "#FEF2F2",
            "borderLeft": f"3px solid {_RED}",
            "borderRadius": "6px",
        })
    else:
        alert_div = html.Div([
            html.Div(f"✓  QUIET BASELINE{date_ctx} — OutbreakForecaster Suppressed", style={
                "fontWeight": "600", "color": _GREEN, "fontSize": "14px", "marginBottom": "4px",
            }),
            html.Div(
                "No active surge detected.  Classifier Z-score below threshold — flat quiet "
                "prior returned.  The TFT model was NOT invoked this period.",
                style={"color": _MUTED, "fontSize": "12px"},
            ),
        ], style={
            "padding": "12px 16px",
            "backgroundColor": "#F0FDF4",
            "borderLeft": f"3px solid {_GREEN}",
            "borderRadius": "6px",
        })

    county_label = FIPS_TO_COUNTY.get(county_fips, county_fips) if county_fips else "All Counties"

    # Header: show date scrub hint or static label
    header_right = html.Span(
        mode_note,
        style={"fontSize": "11px", "color": _BLUE if target_dt else _MUTED,
               "fontStyle": "italic", "marginLeft": "10px"},
    )
    return html.Div([
        html.Div([
            html.Span(
                f"Status — {county_label}",
                style={"fontSize": "13px", "fontWeight": "600", "color": _MUTED},
            ),
            header_right,
        ], style={"marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(badge, width="auto", className="pe-3"),
            dbc.Col([kpis], width=True, className="pe-3"),
            dbc.Col(alert_div, width=4),
        ], className="align-items-center g-2"),
    ])


# ---------------------------------------------------------------------------
# Tier 2 — Residual Timeline (replaces sparse scatter)
# ---------------------------------------------------------------------------

def create_residual_plot(
    processed_df: pd.DataFrame,
    forecast_df:  pd.DataFrame,
    q_cols:       QuantileColumns,
    county_fips:  str | None = None,
) -> go.Figure:
    """Scatter of predicted median vs observed actuals with 45° reference.

    Over-predictions (actual < predicted) are shown in red.
    Under-predictions (actual > predicted) are shown in blue.
    MAE and signed bias are annotated.
    """
    county_col = COUNTY_COL if COUNTY_COL in processed_df.columns else "unique_id"
    actuals = processed_df.rename(columns={
        county_col: "unique_id", NWSS_DATE_COL: "ds", TARGET_COL: "y_true",
    })[["unique_id", "ds", "y_true"]]

    fc = (
        forecast_df[forecast_df["unique_id"] == county_fips].copy()
        if county_fips else forecast_df.copy()
    )
    act = actuals[actuals["unique_id"] == county_fips].copy() if county_fips else actuals.copy()

    merged = fc.merge(act, on=["unique_id", "ds"], how="inner")
    if merged.empty or q_cols.q50 not in merged.columns:
        return _empty_fig("No overlapping forecast–actual pairs for residual analysis", height=340)

    y_true = merged["y_true"].to_numpy(dtype=float)
    y_pred = merged[q_cols.q50].to_numpy(dtype=float)
    resid  = y_true - y_pred

    colors  = [_RED  if r < 0 else _BLUE for r in resid]
    opacity = [0.80  if r < 0 else 0.70  for r in resid]
    labels  = [
        f"{FIPS_TO_COUNTY.get(uid, uid)}<br>{str(ds)[:10]}"
        for uid, ds in zip(merged["unique_id"], merged["ds"])
    ]

    # PI error bars (q025–q975 spread)
    has_pi = q_cols.q025 in merged.columns and q_cols.q975 in merged.columns
    err_lo  = (y_pred - merged[q_cols.q025].values) if has_pi else None
    err_hi  = (merged[q_cols.q975].values - y_pred) if has_pi else None

    fig = go.Figure()

    # 45° perfect-prediction line
    lo = float(min(min(y_true), min(y_pred))) - 0.1
    hi = float(max(max(y_true), max(y_pred))) + 0.1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], name="Perfect prediction",
        mode="lines", line=dict(color=_BORDER, dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    # Scatter points with optional PI error bars
    error_y = dict(type="data", array=err_hi.tolist(), arrayminus=err_lo.tolist(),
                   visible=True, color="rgba(150,150,150,0.4)", thickness=1.5) if has_pi else None

    fig.add_trace(go.Scatter(
        x=y_pred, y=y_true,
        mode="markers", name="Forecast vs. Actual",
        marker=dict(
            color=colors, opacity=opacity, size=8,
            line=dict(color="rgba(255,255,255,0.6)", width=0.5),
        ),
        error_y=error_y,
        text=labels,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Predicted P₅₀: %{x:.3f}<br>"
            "Actual: %{y:.3f}<br>"
            "Residual: %{customdata:.3f}<extra></extra>"
        ),
        customdata=resid,
    ))

    # Annotations
    mae  = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    n    = len(resid)
    over_pct  = 100 * int((resid < 0).sum()) / n
    under_pct = 100 - over_pct

    for txt, x_ref, y_ref, color in [
        (f"MAE = {mae:.3f}", 0.02, 0.99, _TEXT),
        (f"Bias = {bias:+.3f}", 0.02, 0.92, _RED if bias < -0.1 else (_BLUE if bias > 0.1 else _GREEN)),
        (f"Over-pred: {over_pct:.0f}%  Under: {under_pct:.0f}%", 0.02, 0.85, _MUTED),
    ]:
        fig.add_annotation(
            text=txt, xref="paper", yref="paper", x=x_ref, y=y_ref,
            showarrow=False, font=dict(size=11, color=color), align="left",
        )

    title = "Residual Analysis — Predicted vs. Actual"
    if county_fips:
        title += f"  ({FIPS_TO_COUNTY.get(county_fips, county_fips)})"

    fig.update_layout(
        **_base_layout(title, height=360, margin=dict(l=70, r=30, t=55, b=65)),
        xaxis=_axis("Predicted Median P₅₀ (log1p, unscaled)"),
        yaxis=_axis("Observed Actual (log1p, unscaled)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Tier 2 — Residual Timeline (time-series of forecast errors)
# ---------------------------------------------------------------------------

def create_residual_timeline(
    processed_df: pd.DataFrame,
    forecast_df:  pd.DataFrame,
    q_cols:       QuantileColumns,
    county_fips:  str | None = None,
) -> go.Figure:
    """Residual time-series: (actual − predicted_median) as vertical coloured bars.

    Orange bars = under-prediction (actual > forecast median; model too conservative).
    Red bars    = over-prediction  (actual < forecast median; model too high).
    A thick bias line shows the mean displacement across the whole window.

    This replaces the sparse scatter: with only 56–189 points a scatter loses
    all temporal structure, while bars immediately reveal the systematic upward
    bias pattern and whether it improves over the holdout period.
    """
    county_col = COUNTY_COL if COUNTY_COL in processed_df.columns else "unique_id"
    actuals = processed_df.rename(columns={
        county_col: "unique_id", NWSS_DATE_COL: "ds", TARGET_COL: "y_true",
    })[["unique_id", "ds", "y_true"]]

    fc  = (forecast_df[forecast_df["unique_id"] == county_fips].copy()
           if county_fips else forecast_df.copy())
    act = (actuals[actuals["unique_id"] == county_fips].copy()
           if county_fips else actuals.copy())

    merged = fc.merge(act, on=["unique_id", "ds"], how="inner")
    if merged.empty or q_cols.q50 not in merged.columns:
        return _empty_fig("No overlapping forecast–actual pairs", height=320)

    merged  = merged.sort_values("ds")
    y_true  = merged["y_true"].to_numpy(dtype=float)
    y_pred  = merged[q_cols.q50].to_numpy(dtype=float)
    resid   = y_true - y_pred           # positive = actual ABOVE forecast (under-pred)
    bias    = float(resid.mean())
    mae_val = float(np.mean(np.abs(resid)))

    colors  = [_ACCENT if r >= 0 else _RED for r in resid]
    dates   = merged["ds"].tolist()
    county_names = [FIPS_TO_COUNTY.get(u, u) for u in merged["unique_id"]]

    fig = go.Figure()

    # Zero reference line
    fig.add_hline(y=0, line=dict(color=_BORDER, width=1.5))

    # Residual bars
    fig.add_trace(go.Bar(
        x=dates, y=resid,
        marker_color=colors, opacity=0.80,
        name="Residual (actual − P₅₀)",
        text=[f"{c}" for c in county_names],
        hovertemplate=(
            "<b>%{text}</b><br>%{x|%b %d, %Y}<br>"
            "Residual: %{y:+.3f}<br>"
            "<i>positive = actual above forecast</i><extra></extra>"
        ),
    ))

    # Bias reference line
    line_color = _RED if bias < -0.05 else (_ACCENT if bias > 0.05 else _GREEN)
    fig.add_hline(
        y=bias,
        line=dict(color=line_color, dash="dash", width=2),
        annotation_text=f"Mean bias {bias:+.3f}",
        annotation_font=dict(color=line_color, size=11),
        annotation_position="right",
    )

    # MAE band (±MAE shading)
    fig.add_hrect(
        y0=-mae_val, y1=mae_val,
        fillcolor="rgba(150,150,150,0.06)", line_width=0,
        annotation_text=f"±MAE ({mae_val:.3f})",
        annotation_font=dict(color=_MUTED, size=10),
        annotation_position="top right",
    )

    title = "Residual Timeline — Actual minus Forecast Median"
    if county_fips:
        title += f"  ({FIPS_TO_COUNTY.get(county_fips, county_fips)})"
    subtitle = (
        "Orange = model too LOW (under-pred).  Red = model too HIGH (over-pred).  "
        f"Dominant red = systematic upward bias (+{abs(bias):.2f} log1p ≈ "
        f"{(10**abs(bias)-1)*100:.0f}% case overestimate)."
        if bias < -0.05 else
        "Orange = model too LOW.  Red = model too HIGH."
    )

    fig.update_layout(
        **_base_layout(title, height=340, margin=dict(l=70, r=120, t=55, b=65)),
        xaxis=_axis("Date"),
        yaxis=_axis("Residual  (actual − predicted,  log1p)"),
        bargap=0.15,
        annotations=[
            dict(
                text=subtitle, xref="paper", yref="paper",
                x=0, y=-0.17, showarrow=False,
                font=dict(size=11, color=_MUTED), align="left",
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Tier 2 — Covariate Feature Timeline
# ---------------------------------------------------------------------------

def create_covariate_timeline(
    processed_df: pd.DataFrame,
    county_fips:  str | None = None,
) -> go.Figure:
    """Three-panel time-series of the engineered WW features.

    Panel 1 — WW Velocity & Acceleration
      vel_concentration   : absolute Δ log1p_concentration per week
      accel_concentration : 2nd derivative (Δ velocity) — inflection detector

    Panel 2 — WW Relative Decay Rate
      relative_decay_rate : 7-day % change on smoothed signal
                            Positive = growing, Negative = declining

    Panel 3 — WW-to-Case Momentum Divergence (OutbreakClassifier gate)
      ww_momentum_lead    : vel_concentration[t] − (cases[t-1] − cases[t-2])
                            Positive surge = WW accelerating while cases still flat
                            This is the signal that arms the classifier gate.

    The momentum lead panel has the classifier threshold overlaid so the audience
    can see exactly when Stage 2 (TFT) would be invoked.
    """
    from plotly.subplots import make_subplots

    county_col = COUNTY_COL if COUNTY_COL in processed_df.columns else "unique_id"
    sub = (
        processed_df[processed_df[county_col] == county_fips].sort_values(NWSS_DATE_COL).copy()
        if county_fips else
        processed_df.groupby(NWSS_DATE_COL, as_index=False).mean(numeric_only=True).sort_values(NWSS_DATE_COL).copy()
    )

    feat_vel    = "vel_concentration"
    feat_accel  = "accel_concentration"
    feat_decay  = "relative_decay_rate"
    feat_mom    = "ww_momentum_lead"

    missing = [f for f in (feat_vel, feat_accel, feat_decay, feat_mom) if f not in sub.columns]
    if missing or sub.empty:
        return _empty_fig(
            f"Covariate features not available — run with phase-aware training. Missing: {missing}",
            height=480,
        )

    dates = sub[NWSS_DATE_COL]
    county_label = FIPS_TO_COUNTY.get(county_fips, county_fips) if county_fips else f"{_get_geography_label()} avg"

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "WW Velocity & Acceleration",
            "WW Relative Decay Rate  (7-day % change on smoothed signal)",
            "Momentum Divergence — WW leads Cases  (OutbreakClassifier Gate)",
        ],
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.33, 0.27, 0.40],
    )

    # ── Panel 1: Velocity + Acceleration ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=sub[feat_vel],
        name="Velocity (Δ log1p_conc/wk)",
        line=dict(color=_BLUE, width=1.8),
        hovertemplate="%{x|%b %d, %Y}<br>Velocity: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=sub[feat_accel],
        name="Acceleration (Δ velocity)",
        line=dict(color=_ACCENT, width=1.5, dash="dot"),
        hovertemplate="%{x|%b %d, %Y}<br>Accel: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Zero line for panel 1 — use a trace so row/col are unambiguous
    fig.add_trace(go.Scatter(
        x=[dates.iloc[0], dates.iloc[-1]], y=[0, 0],
        mode="lines", line=dict(color=_BORDER, width=1),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── Panel 2: Relative Decay Rate ─────────────────────────────────────────
    decay_vals = sub[feat_decay].clip(-2, 2)
    decay_colors = [_BLUE if v >= 0 else _RED for v in decay_vals]
    fig.add_trace(go.Bar(
        x=dates, y=decay_vals,
        name="Relative Decay Rate",
        marker_color=decay_colors, opacity=0.75,
        hovertemplate="%{x|%b %d, %Y}<br>Decay rate: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # ── Panel 3: Momentum Divergence ─────────────────────────────────────────
    mom_vals = sub[feat_mom]

    fig.add_trace(go.Scatter(
        x=dates, y=mom_vals,
        name="ww_momentum_lead",
        line=dict(color=_VIOLET, width=2.5),
        fill="tozeroy", fillcolor="rgba(109,40,217,0.08)",
        hovertemplate="%{x|%b %d, %Y}<br>Momentum: %{y:.3f}<extra>WW–Case divergence</extra>",
    ), row=3, col=1)

    # Momentum gate threshold at y=0 — trace-based to avoid row/col type friction
    fig.add_trace(go.Scatter(
        x=[dates.iloc[0], dates.iloc[-1]], y=[0, 0],
        mode="lines", line=dict(color=_VIOLET, dash="longdash", width=1.5),
        name="Gate threshold (≥ 0)", showlegend=True, hoverinfo="skip",
    ), row=3, col=1)

    # Shade positive momentum regions via add_shape (proper subplot API)
    in_surge, surge_start = False, None
    for date, val in zip(dates, mom_vals):
        if val > 0 and not in_surge:
            surge_start, in_surge = date, True
        elif val <= 0 and in_surge:
            fig.add_shape(
                type="rect", xref="x3", yref="paper",
                x0=surge_start, x1=date, y0=0.0, y1=0.33,
                fillcolor="rgba(109,40,217,0.10)", line_width=0, layer="below",
            )
            in_surge = False
    if in_surge:
        fig.add_shape(
            type="rect", xref="x3", yref="paper",
            x0=surge_start, x1=dates.iloc[-1], y0=0.0, y1=0.33,
            fillcolor="rgba(109,40,217,0.10)", line_width=0, layer="below",
        )

    fig.update_layout(
        title=dict(
            text=f"Engineered WW Feature Timeline — {county_label}",
            font=dict(size=15, color=_TEXT),
        ),
        height=520,
        paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
        font=_FONT_BODY,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)", bordercolor=_BORDER, borderwidth=1,
            font=dict(size=11, color=_TEXT), x=0.01, y=0.99,
        ),
        margin=dict(l=70, r=120, t=70, b=60),
    )
    for row_i, y_title in enumerate([
        "Δ log1p_conc",
        "7-day % Δ (winsorised ±2)",
        "vel_conc − Δ(cases_lag)",
    ], start=1):
        fig.update_yaxes(
            title_text=y_title, title_font=dict(size=11, color=_MUTED),
            tickfont=dict(size=10, color=_MUTED),
            gridcolor=_GRID, zeroline=False,
            row=row_i, col=1,
        )
    fig.update_xaxes(
        tickfont=dict(size=10, color=_MUTED), gridcolor=_GRID,
        row=3, col=1,
    )
    return fig


# ---------------------------------------------------------------------------
# Tier 3 — Reliability Diagram (Calibration Curve)
# ---------------------------------------------------------------------------

def create_reliability_diagram(
    processed_df: pd.DataFrame,
    forecast_df:  pd.DataFrame,
    q_cols:       QuantileColumns,
    county_fips:  str | None = None,
) -> go.Figure:
    """Quantile reliability diagram: nominal quantile vs empirical coverage.

    For a well-calibrated model the curve lies on the y=x diagonal.
    Deviation above the diagonal = overconfident (PI too narrow).
    Deviation below the diagonal = underconfident (PI too wide).

    Green shaded band = ±5% acceptable tolerance.
    """
    county_col = COUNTY_COL if COUNTY_COL in processed_df.columns else "unique_id"
    actuals = processed_df.rename(columns={
        county_col: "unique_id", NWSS_DATE_COL: "ds", TARGET_COL: "y_true",
    })[["unique_id", "ds", "y_true"]]

    fc  = (forecast_df[forecast_df["unique_id"] == county_fips].copy()
           if county_fips else forecast_df.copy())
    act = (actuals[actuals["unique_id"] == county_fips].copy()
           if county_fips else actuals.copy())

    merged = fc.merge(act, on=["unique_id", "ds"], how="inner")
    if merged.empty:
        return _empty_fig("No overlapping data for calibration", height=360)

    y_true = merged["y_true"].to_numpy(dtype=float)

    quantile_map = [
        (0.025, q_cols.q025,  "P₂.₅"),
        (0.10,  q_cols.q10,   "P₁₀"),
        (0.25,  q_cols.q25,   "P₂₅"),
        (0.50,  q_cols.q50,   "P₅₀"),
        (0.75,  q_cols.q75,   "P₇₅"),
        (0.90,  q_cols.q90,   "P₉₀"),
        (0.975, q_cols.q975,  "P₉₇.₅"),
    ]

    nominal    = []
    empirical  = []
    pt_labels  = []
    pt_colors  = []

    for q_val, col, label in quantile_map:
        if col is None or col not in merged.columns:
            continue
        yhat = merged[col].to_numpy(dtype=float)
        emp  = float((y_true <= yhat).mean())
        dev  = abs(emp - q_val)
        nominal.append(q_val)
        empirical.append(emp)
        pt_labels.append(label)
        pt_colors.append(
            _GREEN if dev < 0.05 else (_AMBER if dev < 0.15 else _RED)
        )

    if not nominal:
        return _empty_fig("Insufficient quantile columns for calibration", height=360)

    fig = go.Figure()

    # ±5% tolerance band
    upper_band = [min(1.0, n + 0.05) for n in nominal]
    lower_band = [max(0.0, n - 0.05) for n in nominal]
    fig.add_trace(go.Scatter(
        x=nominal + nominal[::-1],
        y=upper_band + lower_band[::-1],
        fill="toself", fillcolor="rgba(21,128,61,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±5% tolerance", hoverinfo="skip",
    ))

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfect calibration",
        mode="lines", line=dict(color=_BORDER, dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    # Model calibration curve
    fig.add_trace(go.Scatter(
        x=nominal, y=empirical, name="Model",
        mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=11, color=pt_colors, line=dict(color=_TEXT, width=1)),
        text=pt_labels,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Nominal: %{x:.3f}  (%{x:.0%})<br>"
            "Empirical: %{y:.3f}  (%{y:.0%})<br>"
            "Deviation: %{customdata:+.3f}<extra></extra>"
        ),
        customdata=[e - n for e, n in zip(empirical, nominal)],
    ))

    # Overconfident / underconfident annotation
    mean_dev = float(np.mean([abs(e - n) for e, n in zip(empirical, nominal)]))
    cal_label = (
        f"Well-calibrated  (mean dev = {mean_dev:.3f})" if mean_dev < 0.05
        else f"Miscalibrated  (mean dev = {mean_dev:.3f})"
    )
    cal_color = _GREEN if mean_dev < 0.05 else (_AMBER if mean_dev < 0.15 else _RED)
    fig.add_annotation(
        text=cal_label, xref="paper", yref="paper", x=0.98, y=0.05,
        showarrow=False, font=dict(size=11, color=cal_color), align="right",
    )

    title = "Reliability Diagram — Quantile Calibration"
    if county_fips:
        title += f"  ({FIPS_TO_COUNTY.get(county_fips, county_fips)})"

    fig.update_layout(
        **_base_layout(title, height=360, margin=dict(l=70, r=30, t=55, b=65)),
        xaxis=_axis("Nominal Quantile Level", tickformat=".0%", range=[-0.02, 1.02]),
        yaxis=_axis("Empirical Coverage (fraction of actuals ≤ predicted quantile)",
                    tickformat=".0%", range=[-0.02, 1.02]),
    )
    return fig


# ---------------------------------------------------------------------------
# Tier 3 — CV Stability: 3-panel (Accuracy / Coverage / Pinball calibration)
# ---------------------------------------------------------------------------

def _build_cv_stability_chart(cv_df: pd.DataFrame) -> go.Figure:
    """Three-panel cross-validation stability chart.

    Panel 1 — Accuracy: WIS and MAE per fold, with per-county WIS lines when available.
    Panel 2 — Coverage: Coverage 50% and 95% against reference lines.
    Panel 3 — Pinball calibration bar: average quantile loss per level across folds.
               A symmetric (flat) bar = well-calibrated.
               Tall bars on the LEFT = lower quantiles too high = model needs to shift DOWN.
    """
    from plotly.subplots import make_subplots

    if cv_df.empty or "cutoff_date" not in cv_df.columns:
        return _empty_fig("No CV results available", height=480)

    dates = pd.to_datetime(cv_df["cutoff_date"])

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Accuracy per Fold (WIS & MAE)",
            "Coverage per Fold  ·  targets: 50% and 95%",
            "Pinball Calibration  ·  symmetric = unbiased  ·  tall left bars = upward bias",
        ],
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.34, 0.30, 0.36],
    )

    # ── Panel 1: WIS + MAE ───────────────────────────────────────────────────
    if "mean_wis" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=cv_df["mean_wis"], name="Mean WIS",
            mode="lines+markers", line=dict(color=_ACCENT, width=2.5), marker=dict(size=7),
            hovertemplate="Cutoff %{x|%b %Y}<br>WIS: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

    if "mae" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=cv_df["mae"], name="MAE",
            mode="lines+markers", line=dict(color=_VIOLET, width=2, dash="dot"), marker=dict(size=6),
            hovertemplate="Cutoff %{x|%b %Y}<br>MAE: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

    # Per-county WIS lines (lighter)
    wis_county_cols = [c for c in cv_df.columns if c.startswith("wis_") and c != "mean_wis"]
    for i, col in enumerate(wis_county_cols):
        fips = col[4:]
        fig.add_trace(go.Scatter(
            x=dates, y=cv_df[col], name=FIPS_TO_COUNTY.get(fips, fips),
            mode="lines", line=dict(color=_COUNTY_COLORS[i % len(_COUNTY_COLORS)], width=1),
            opacity=0.45, hovertemplate=f"{FIPS_TO_COUNTY.get(fips,fips)}<br>WIS: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    # ── Panel 2: Coverage ────────────────────────────────────────────────────
    if "coverage_95" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=cv_df["coverage_95"] * 100, name="Coverage 95%",
            mode="lines+markers", line=dict(color=_BLUE, width=2.5), marker=dict(size=7),
            hovertemplate="Cutoff %{x|%b %Y}<br>Coverage 95%: %{y:.1f}%%<extra></extra>",
        ), row=2, col=1)

    if "coverage_50" in cv_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=cv_df["coverage_50"] * 100, name="Coverage 50%",
            mode="lines+markers", line=dict(color=_GREEN, width=2), marker=dict(size=6),
            hovertemplate="Cutoff %{x|%b %Y}<br>Coverage 50%: %{y:.1f}%%<extra></extra>",
        ), row=2, col=1)

    # Reference targets as traces (avoids row/col friction with add_hline)
    for target_pct, color in [(95, _BLUE), (50, _GREEN)]:
        fig.add_trace(go.Scatter(
            x=[dates.min(), dates.max()], y=[target_pct, target_pct],
            mode="lines", line=dict(color=color, dash="longdash", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)

    # Per-fold pinball ratio (q10/q90) — right y-axis on panel 2
    if "pinball_q010" in cv_df.columns and "pinball_q090" in cv_df.columns:
        ratio_vals = (cv_df["pinball_q010"] / cv_df["pinball_q090"].replace(0, float("nan"))).tolist()
        fig.add_trace(go.Scatter(
            x=dates, y=ratio_vals, name="Pinball ratio q10/q90",
            mode="lines+markers",
            line=dict(color=_RED, width=2, dash="dot"),
            marker=dict(size=6, color=_RED),
            yaxis="y4",
            hovertemplate="Cutoff %{x|%b %Y}<br>Pinball ratio: %{y:.2f}×<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[dates.min(), dates.max()], y=[1.0, 1.0],
            mode="lines", line=dict(color=_RED, dash="longdash", width=1),
            showlegend=False, hoverinfo="skip", yaxis="y4",
        ), row=2, col=1)

    # ── Panel 3: Pinball calibration bar ─────────────────────────────────────
    pinball_map = {
        "pinball_q0025": "q2.5%", "pinball_q010": "q10%", "pinball_q025": "q25%",
        "pinball_q050": "q50%", "pinball_q075": "q75%", "pinball_q090": "q90%",
        "pinball_q0975": "q97.5%",
    }
    pb_cols   = [c for c in pinball_map if c in cv_df.columns]
    if pb_cols:
        avg_pb = cv_df[pb_cols].mean()
        labels = [pinball_map[c] for c in pb_cols]
        values = avg_pb[pb_cols].tolist()
        # Colour: lower quantiles (q2.5%–q25%) in red when they dominate → upward bias
        colors = []
        for c in pb_cols:
            q_idx  = pb_cols.index(c)
            mirror = pb_cols[-(q_idx + 1)] if q_idx < len(pb_cols) // 2 else None
            is_high = (mirror is not None and avg_pb[c] > 1.5 * avg_pb[mirror])
            colors.append(_RED if is_high else (_AMBER if q_idx < len(pb_cols) // 2 else _BLUE))

        fig.add_trace(go.Bar(
            x=labels, y=values, name="Avg pinball loss",
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Avg pinball: %{y:.4f}<extra></extra>",
        ), row=3, col=1)

        # Symmetry reference: mean of all pinball values as horizontal line via trace
        mean_pb = float(np.mean(values))
        fig.add_trace(go.Scatter(
            x=[labels[0], labels[-1]], y=[mean_pb, mean_pb],
            mode="lines", line=dict(color=_MUTED, dash="dot", width=1.5),
            name=f"Mean ({mean_pb:.4f})", showlegend=True, hoverinfo="skip",
        ), row=3, col=1)

    fig.update_layout(
        title=dict(
            text="Cross-Validation — Temporal Stability",
            font=dict(size=15, color=_TEXT),
        ),
        height=540,
        paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
        font=_FONT_BODY,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor=_BORDER, borderwidth=1,
            font=dict(size=10, color=_TEXT), x=1.01, y=0.99, xanchor="left",
        ),
        margin=dict(l=60, r=160, t=70, b=55),
        yaxis4=dict(
            title=dict(text="Pinball ratio (q10/q90)", font=dict(size=10, color=_RED)),
            tickfont=dict(size=9, color=_RED),
            overlaying="y2", side="right", showgrid=False,
            zeroline=False,
        ),
    )
    for row_i, y_title in enumerate(["WIS / MAE", "Coverage (%)", "Avg Pinball Loss"], start=1):
        fig.update_yaxes(
            title_text=y_title, title_font=dict(size=11, color=_MUTED),
            tickfont=dict(size=10, color=_MUTED), gridcolor=_GRID, zeroline=False,
            row=row_i, col=1,
        )
    fig.update_xaxes(tickfont=dict(size=10, color=_MUTED), gridcolor=_GRID)
    return fig


# ---------------------------------------------------------------------------
# Tier 3 — Coverage & WIS compact summary card
# ---------------------------------------------------------------------------

def _build_wis_coverage_summary(eval_dict: dict, cv_df: pd.DataFrame) -> go.Figure:
    """Combined Coverage + WIS summary — horizontal grouped bar per county."""
    wis_vals = {k[4:]: v for k, v in eval_dict.items()
                if k.startswith("wis_") and isinstance(v, float)}
    cov95  = eval_dict.get("coverage_95", float("nan"))
    cov50  = eval_dict.get("coverage_50", float("nan"))
    mae    = eval_dict.get("mae",         float("nan"))
    mean_wis = eval_dict.get("mean_wis",  float("nan"))

    fig = go.Figure()

    if wis_vals:
        names  = [FIPS_TO_COUNTY.get(f, f) for f in wis_vals]
        values = list(wis_vals.values())
        pairs  = sorted(zip(values, names), reverse=True)
        values, names = zip(*pairs)
        colors = [_GREEN if v < 0.2 else (_AMBER if v < 0.5 else _RED) for v in values]
        fig.add_trace(go.Bar(
            x=list(values), y=list(names), orientation="h",
            marker=dict(color=colors, opacity=0.85),
            name="County WIS",
            hovertemplate="<b>%{y}</b><br>WIS: %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0.2, line=dict(color=_GREEN, dash="dot", width=1),
                      annotation_text="good ≤0.2", annotation_font_color=_GREEN,
                      annotation_position="top right")
        fig.add_vline(x=0.5, line=dict(color=_AMBER, dash="dot", width=1),
                      annotation_text="fair ≤0.5", annotation_font_color=_AMBER,
                      annotation_position="top right")

    def _ann(txt, xp, yp, color):
        fig.add_annotation(
            text=txt, xref="paper", yref="paper", x=xp, y=yp,
            showarrow=False, font=dict(size=12, color=color), align="right",
            bgcolor=_SURFACE2, bordercolor=_BORDER, borderwidth=1, borderpad=4,
        )

    if not math.isnan(mean_wis):
        _ann(f"Mean WIS: {mean_wis:.3f}", 0.98, 0.95,
             _GREEN if mean_wis < 0.2 else (_AMBER if mean_wis < 0.5 else _RED))
    if not math.isnan(cov95):
        _ann(f"Coverage 95%: {cov95:.0%}", 0.98, 0.82,
             _GREEN if cov95 >= 0.85 else (_AMBER if cov95 >= 0.60 else _RED))
    if not math.isnan(cov50):
        _ann(f"Coverage 50%: {cov50:.0%}", 0.98, 0.69,
             _GREEN if cov50 >= 0.40 else (_AMBER if cov50 >= 0.20 else _RED))
    if not math.isnan(mae):
        _ann(f"MAE: {mae:.3f}", 0.98, 0.56, _MUTED)

    fig.update_layout(
        **_base_layout("WIS & Coverage Summary — Holdout Evaluation", height=360,
                       margin=dict(l=110, r=30, t=55, b=50)),
        xaxis=_axis("WIS  (lower = more accurate)"),
        yaxis=dict(tickfont=dict(size=11, color=_TEXT), gridcolor=_GRID),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app builder
# ---------------------------------------------------------------------------

def create_app(
    processed_df:        pd.DataFrame,
    forecast_df:         pd.DataFrame,
    model                = None,
    sludge_df:           Optional[pd.DataFrame] = None,
    liquid_df:           Optional[pd.DataFrame] = None,
    q_cols:              Optional[QuantileColumns] = None,
    eval_result          = None,
    cv_results:          Optional[pd.DataFrame] = None,
    runs_dir:            Optional[Path] = None,
    rolling_forecast_df: Optional[pd.DataFrame] = None,
) -> dash.Dash:
    """Build and return the Sewer Surveillance production dashboard.

    runs_dir — if provided, a run selector dropdown is shown at the top.
    Selecting a different run reloads all charts from that run's artefacts.
    The initial view uses the passed-in processed_df / forecast_df (the
    most recently completed run).
    """
    from src.utils.run_manager import list_runs, load_run_data

    if q_cols is None:
        try:
            q_cols = QuantileColumns.auto_detect(forecast_df)
        except ValueError:
            q_cols = QuantileColumns()

    # Build eval_dict from eval_result object (if provided) or leave empty
    _initial_eval_dict: dict = {}
    if eval_result is not None:
        try:
            _initial_eval_dict = eval_result.to_dict()
        except Exception:
            pass

    # Build initial CV frame
    _initial_cv = cv_results if (cv_results is not None and not cv_results.empty) else pd.DataFrame()

    # Initial rolling forecast (may be empty if run was not --rolling-holdout)
    _initial_rolling = rolling_forecast_df if rolling_forecast_df is not None else pd.DataFrame()

    # VSN weights (pre-extracted if model available)
    _vsn_weights: dict = {}
    if model is not None:
        try:
            raw = extract_vsn_weights(model)
            if raw:
                _vsn_weights = raw
        except Exception as exc:
            logger.warning("VSN extraction skipped: {}", exc)

    # Available runs for the selector
    available_runs = list_runs(runs_dir) if runs_dir else []
    has_run_selector = len(available_runs) > 0

    # Dropdown options: newest first, plus "Live (current run)" at top
    run_options = [{"label": "⚡ Live  ·  current pipeline run", "value": "__live__"}]
    for r in reversed(available_runs):
        run_options.append({"label": r.dropdown_label, "value": str(r.run_dir)})

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.FLATLY,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        ],
        title=f"Sewer Surveillance — {_get_geography_label()}",
    )

    # ── Layout ──────────────────────────────────────────────────────────────────
    header_children = [
        html.Div([
            html.Span("Sewer Surveillance", style={
                "fontSize": "28px", "fontWeight": "700", "color": _ACCENT,
                "letterSpacing": "-0.03em", "fontFamily": "Inter, system-ui, sans-serif",
            }),
            html.Span(" | ", style={"color": _BORDER, "margin": "0 10px"}),
            html.Span(
                "Attention-based outbreak forecasting with COVID-19 wastewater data",
                style={"fontSize": "16px", "color": _MUTED, "fontWeight": "400"},
            ),
        ], style={"display": "flex", "alignItems": "baseline"}),
        html.Div(id="selected-county-badge",
                 style={"color": _MUTED, "fontSize": "12px", "marginTop": "4px"},
                 children=f"{_get_geography_label()}  ·  8-week horizon  ·  click a county on the map to focus"),
    ]

    run_selector_row = []
    if has_run_selector:
        run_selector_row = [dbc.Row([
            dbc.Col(_card([
                html.Div([
                    html.Label("Pipeline Run", style={
                        "fontSize": "11px", "fontWeight": "600", "color": _MUTED,
                        "textTransform": "uppercase", "letterSpacing": "0.05em",
                        "marginBottom": "6px", "display": "block",
                    }),
                    dcc.Dropdown(
                        id="run-selector",
                        options=run_options,
                        value="__live__",
                        clearable=False,
                        style={
                            "fontFamily": "Inter, system-ui, sans-serif",
                            "fontSize": "13px",
                            "border": f"1px solid {_BORDER}",
                            "borderRadius": "6px",
                        },
                    ),
                    html.Div(
                        id="run-scorecard",
                        children=_build_run_scorecard(_initial_eval_dict),
                        style={"marginTop": "6px"},
                    ),
                ]),
            ], style={"padding": "12px 16px"})),
        ], className="mb-3")]

    app.layout = dbc.Container(
        fluid=True,
        style={"backgroundColor": _BG, "minHeight": "100vh", "padding": "20px 28px"},
        children=[

            # ── Header ───────────────────────────────────────────────────────
            dbc.Row([dbc.Col(header_children)], className="mb-3"),
            *run_selector_row,

            # ════════════════════════════════════════════════════════════════
            # TIER 1 — DECISION LAYER
            # ════════════════════════════════════════════════════════════════
            dbc.Row([dbc.Col(_tier_header(
                1, "Decision Layer",
                "Immediate situational awareness for operational decision-makers. "
                "Phase status, forecast, and spatial density at a glance.",
            ))], className="mb-1"),

            # 1-A: Forecast chart + Map
            dbc.Row([
                dbc.Col(_card([
                    dcc.RadioItems(
                        id="forecast-view-toggle",
                        options=[
                            {"label": "  Final 8-week forecast", "value": "final"},
                            {"label": "  Full rolling holdout (28 weeks)", "value": "rolling"},
                        ],
                        value="final", inline=True,
                        style={"fontSize": "12px", "color": _MUTED, "marginBottom": "10px"},
                        inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                    ),
                    dcc.Graph(
                        id="hero-chart",
                        config={"displayModeBar": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                "displaylogo": False},
                    ),
                ]), width=8),
                dbc.Col(_card([dcc.Graph(
                    id="map-chart",
                    config={"displayModeBar": False},
                )]), width=4),
            ], className="mb-3"),

            # 1-B: Current Status Panel (phase badge + KPIs + alert)
            dbc.Row([dbc.Col(_card([
                html.Div(
                    id="status-panel",
                    children=_build_current_status_panel(
                        forecast_df, q_cols, pd.DataFrame(), None, _initial_eval_dict,
                        processed_df=processed_df,
                    ),
                ),
            ]))], className="mb-4"),

            # ════════════════════════════════════════════════════════════════
            # TIER 2 — DETAILED LAYER
            # ════════════════════════════════════════════════════════════════
            dbc.Row([dbc.Col(_tier_header(
                2, "Detailed Layer",
                "Scientific explanation of model behaviour: timeline, gatekeeper activity, "
                "feature importance, attention weights, and residual analysis.",
            ))], className="mb-1"),

            # 2-A: Full 3-year timeline (with classifier overlay)
            dbc.Row([dbc.Col(_card([
                _section_label(
                    "Full Data Timeline",
                    "Three-year history with train / validation / holdout splits. "
                    "Orange bands = OutbreakClassifier Stage-2 active.  "
                    "Orange PI = 8-week forecast.  Blue diamonds = holdout actuals.",
                ),
                dcc.Graph(id="timeline-chart", config={"displayModeBar": False}),
            ]))], className="mb-3"),

            # 2-B: Biological Translation Table
            dbc.Row([dbc.Col(_card([
                _section_label(
                    "Biological Translation Layer",
                    "Maps each model metric to its clinical / epidemiological meaning. "
                    "Use this panel when briefing non-technical stakeholders.",
                ),
                html.Div(
                    id="bio-table-container",
                    children=_build_bio_table(_initial_eval_dict or None),
                ),
            ]))], className="mb-3"),

            # 2-D: Feature Importance + Attention Weights (side by side)
            dbc.Row([
                dbc.Col(_card([
                    _section_label(
                        "Feature Importance — VSN",
                        "Variable Selection Network learned importance weights. "
                        "Which covariates are the TFT relying on most?",
                    ),
                    dcc.Graph(id="vsn-chart", config={"displayModeBar": False}),
                ]), width=6),
                dbc.Col(_card([
                    _section_label(
                        "Attention Weight Map",
                        "Transformer self-attention: which historical time-steps "
                        "most influence the current 8-week forecast window?",
                    ),
                    dcc.Graph(id="attention-chart", config={"displayModeBar": False}),
                ]), width=6),
            ], className="mb-3"),

            # 2-E: Covariate Feature Timeline
            dbc.Row([dbc.Col(_card([
                _section_label(
                    "Engineered WW Feature Timeline — Velocity, Acceleration & Momentum",
                    "Top: WW velocity (Δ log1p/wk) and acceleration (inflection detector).  "
                    "Middle: 7-day relative decay rate (positive = growing).  "
                    "Bottom: ww_momentum_lead — WW velocity minus lagged case velocity.  "
                    "Purple bands = positive momentum (OutbreakClassifier gate opens when momentum ≥ 0 AND Z ≥ 1.5).",
                ),
                dcc.Graph(id="covariate-chart", config={"displayModeBar": False}),
            ]))], className="mb-3"),

            # 2-F: Residual Timeline
            dbc.Row([dbc.Col(_card([
                _section_label(
                    "Residual Timeline — Actual minus Forecast Median",
                    "Orange = model too LOW (actual above forecast — under-prediction).  "
                    "Red = model too HIGH (actual below forecast — over-prediction, dominant pattern).  "
                    "Dashed line = mean bias across holdout window.",
                ),
                dcc.Graph(id="residual-chart", config={"displayModeBar": False}),
            ]))], className="mb-4"),

            # ════════════════════════════════════════════════════════════════
            # TIER 3 — DIAGNOSTIC LAYER
            # ════════════════════════════════════════════════════════════════
            dbc.Row([dbc.Col(_tier_header(
                3, "Diagnostic Layer",
                "Statistical validation of the probabilistic model. "
                "Quantifies calibration quality and temporal stability.",
            ))], className="mb-1"),

            # 3-A: Reliability Diagram + WIS/Coverage Summary
            dbc.Row([
                dbc.Col(_card([
                    _section_label(
                        "Reliability Diagram",
                        "Nominal quantile vs empirical coverage rate. "
                        "On the 45° diagonal = perfectly calibrated. "
                        "Above diagonal = overconfident. Below = underconfident.",
                    ),
                    dcc.Graph(id="reliability-chart", config={"displayModeBar": False}),
                ]), width=6),
                dbc.Col(_card([
                    _section_label(
                        "Coverage & WIS Summary",
                        "Per-county WIS with holdout aggregate metrics. "
                        "Green ≤ 0.20 = excellent.  Orange ≤ 0.50 = acceptable.  "
                        "Coverage targets: 95% PI ≥ 85%, 50% PI ≥ 40%.",
                    ),
                    dcc.Graph(id="wis-coverage-chart", config={"displayModeBar": False}),
                ]), width=6),
            ], className="mb-3"),

            # 3-B: CV Stability (3-panel)
            dbc.Row([dbc.Col(_card([
                _section_label(
                    "Cross-Validation — Temporal Stability (3 panels)",
                    "Panel 1: WIS and MAE per fold with per-county breakdown.  "
                    "Panel 2: Coverage 50% and 95% against reference lines — Coverage 50% near 0 = systematic upward bias.  "
                    "Panel 3: Pinball calibration bar — symmetric = unbiased; tall LEFT bars = lower quantiles too high = model must shift DOWN.",
                ),
                dcc.Graph(id="cv-chart", config={"displayModeBar": False}),
            ]))], className="mb-4"),

            # ── Footer ───────────────────────────────────────────────────────
            dbc.Row(dbc.Col(html.P(
                "Data: CA Open Data Portal  ·  Model: Temporal Fusion Transformer  ·  "
                "Sewer Signals — Attention-based outbreak forecasting",
                style={"color": _MUTED, "fontSize": "11px", "textAlign": "center",
                       "paddingTop": "8px", "borderTop": f"1px solid {_BORDER}"},
            ))),

            dcc.Store(id="selected-county", data=None),
            dcc.Store(id="selected-date",   data=None),
            *([dcc.Store(id="run-selector", data="__live__")] if not has_run_selector else []),
        ],
    )

    # ── Helpers inside create_app closure ────────────────────────────────────

    # Initial classification data (empty when --two-stage was not used)
    _clf_path    = (runs_dir.parent / "processed" / "classification.parquet"
                    if runs_dir is not None else None)
    _initial_clf = (pd.read_parquet(_clf_path)
                    if _clf_path is not None and _clf_path.exists()
                    else pd.DataFrame())

    def _get_run_data(run_dir_str: str) -> tuple:
        """Return (processed_df, forecast_df, q_cols, cv_df, eval_dict, rolling_df, clf_df)."""
        if run_dir_str == "__live__" or not run_dir_str:
            return (processed_df, forecast_df, q_cols,
                    _initial_cv, _initial_eval_dict, _initial_rolling, _initial_clf)
        try:
            return load_run_data(Path(run_dir_str))
        except Exception as exc:
            logger.warning("Failed to load run {}: {}", run_dir_str, exc)
            return (processed_df, forecast_df, q_cols,
                    _initial_cv, _initial_eval_dict, _initial_rolling, _initial_clf)

    # ── Map click → county selection ──────────────────────────────────────────
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

    # ── Hero chart click → date scrub ─────────────────────────────────────────
    # Captures the x-value (date) from whichever trace the user clicks.
    # Clicking the same date a second time resets (toggles off) to deselect.
    @app.callback(
        Output("selected-date", "data"),
        Input("hero-chart", "clickData"),
        State("selected-date", "data"),
    )
    def on_hero_click(click_data, current_date):
        if click_data is None:
            return current_date
        try:
            raw_x = str(click_data["points"][0]["x"])
            # Normalise to YYYY-MM-DD — Plotly may return "2022-01-05T00:00:00" etc.
            clicked = pd.Timestamp(raw_x).date().isoformat()
        except Exception:
            return current_date
        # Toggle: clicking the same date deselects it
        return None if clicked == current_date else clicked

    # ── TIER 1: Hero + Map + Badge (main panels) ──────────────────────────────
    main_inputs = [Input("selected-county", "data"), Input("forecast-view-toggle", "value")]
    if has_run_selector:
        main_inputs.append(Input("run-selector", "value"))

    @app.callback(
        Output("hero-chart",            "figure"),
        Output("map-chart",             "figure"),
        Output("timeline-chart",        "figure"),
        Output("selected-county-badge", "children"),
        *main_inputs,
    )
    def update_main_panels(*args):
        county_fips  = args[0]
        view_mode    = args[1] if len(args) > 1 else "final"
        run_dir_str  = args[2] if len(args) > 2 else "__live__"
        pd_df, fc_df, qc, _, _, rolling_df, clf_df = _get_run_data(run_dir_str)

        use_rolling   = (view_mode == "rolling") and not rolling_df.empty
        display_fc    = rolling_df if use_rolling else fc_df
        horizon_label = "28-week rolling holdout" if use_rolling else "8-week horizon"

        county_name = FIPS_TO_COUNTY.get(county_fips, county_fips) if county_fips else None
        n_counties  = display_fc["unique_id"].nunique() if not display_fc.empty else 0
        badge = (
            f"Focused: {county_name}  ·  {horizon_label}  ·  click again to deselect"
            if county_name else
            f"{n_counties} {'County' if n_counties == 1 else 'Counties'}  ·  "
            f"{horizon_label}  ·  click a county to focus"
        )

        hero     = create_hero_plot(pd_df, display_fc, county_fips, qc, clf_df=clf_df)
        map_fig  = create_map(display_fc, qc, selected_fips=county_fips)
        timeline = create_timeline_chart(pd_df, display_fc, county_fips, qc, clf_df=clf_df)
        return hero, map_fig, timeline, badge

    # ── TIER 1: Current Status Panel ─────────────────────────────────────────
    # Inputs: selected county  +  clicked date (date scrub)  +  run selector
    status_inputs = [Input("selected-county", "data"), Input("selected-date", "data")]
    if has_run_selector:
        status_inputs.append(Input("run-selector", "value"))

    @app.callback(Output("status-panel", "children"), *status_inputs)
    def update_status_panel(*args):
        county_fips   = args[0]
        selected_date = args[1]
        run_dir_str   = args[2] if len(args) > 2 else "__live__"
        pd_df, fc_df, qc, _, eval_dict_run, _, clf_df = _get_run_data(run_dir_str)
        return _build_current_status_panel(
            fc_df, qc, clf_df, county_fips, eval_dict_run,
            processed_df=pd_df,
            selected_date=selected_date,
        )

    # ── TIER 2: Classifier timeline ───────────────────────────────────────────
    # ── TIER 2: Bio table + VSN + Attention ───────────────────────────────────
    detail_inputs = [Input("selected-county", "data")]
    if has_run_selector:
        detail_inputs.append(Input("run-selector", "value"))

    @app.callback(
        Output("bio-table-container", "children"),
        Output("run-scorecard", "children") if has_run_selector else Output("vsn-chart", "figure"),
        Output("vsn-chart",     "figure")   if has_run_selector else Output("attention-chart", "figure"),
        Output("attention-chart", "figure") if has_run_selector else Output("bio-table-container", "children"),
        *detail_inputs,
    )
    def update_detailed_panels(*args):
        county_fips = args[0]
        run_dir_str = args[1] if len(args) > 1 else "__live__"
        _, _, _, _, eval_dict_run, _, _ = _get_run_data(run_dir_str)

        bio_tbl   = _build_bio_table(eval_dict_run or None)
        scorecard = _build_run_scorecard(eval_dict_run)

        # VSN importance
        if _vsn_weights:
            vsn_fig = plot_vsn_importance(_vsn_weights, role="historical",
                                          title="Feature Importance — VSN")
        else:
            from src.models.tft_model import HIST_COVARIATES
            rng = np.random.default_rng(42)
            w   = rng.dirichlet(np.ones(len(HIST_COVARIATES)) * 0.6)
            vsn_fig = plot_vsn_importance(
                {"historical": w}, role="historical",
                title="Feature Importance — VSN (illustrative)",
            )

        # Attention weights
        attn_weights = None
        if model is not None:
            try:
                attn_weights = extract_attention_weights(model)
            except Exception:
                pass
        if attn_weights is None:
            T = 16
            attn_weights = np.zeros((4, T))
            for i in range(4):
                w = np.exp(-0.12 * np.arange(T)[::-1])
                w += 0.25 * np.exp(-0.5 * ((np.arange(T) - (T - 3 + i)) / 2.0) ** 2)
                attn_weights[i] = w / w.sum()
        label = " (illustrative)" if model is None else ""
        attn_fig = plot_attention_heatmap(
            attn_weights,
            county=county_fips or "All Counties",
            title=f"Temporal Attention Weights{label}",
        )

        if has_run_selector:
            return bio_tbl, scorecard, vsn_fig, attn_fig
        else:
            return bio_tbl, vsn_fig, attn_fig, bio_tbl  # fallback

    # ── TIER 2: Covariate Feature Timeline ───────────────────────────────────
    cov_inputs = [Input("selected-county", "data")]
    if has_run_selector:
        cov_inputs.append(Input("run-selector", "value"))

    @app.callback(Output("covariate-chart", "figure"), *cov_inputs)
    def update_covariate_chart(*args):
        county_fips = args[0]
        run_dir_str = args[1] if len(args) > 1 else "__live__"
        pd_df, _, _, _, _, _, _ = _get_run_data(run_dir_str)
        return create_covariate_timeline(pd_df, county_fips)

    # ── TIER 2: Residual Timeline ─────────────────────────────────────────────
    resid_inputs = [Input("selected-county", "data")]
    if has_run_selector:
        resid_inputs.append(Input("run-selector", "value"))

    @app.callback(Output("residual-chart", "figure"), *resid_inputs)
    def update_residual(*args):
        county_fips = args[0]
        run_dir_str = args[1] if len(args) > 1 else "__live__"
        pd_df, fc_df, qc, _, _, rolling_df, _ = _get_run_data(run_dir_str)
        display_fc = rolling_df if not rolling_df.empty else fc_df
        return create_residual_timeline(pd_df, display_fc, qc, county_fips)

    # ── TIER 3: Reliability + WIS/Coverage + CV Stability ────────────────────
    diag_inputs = [Input("selected-county", "data")]
    if has_run_selector:
        diag_inputs.append(Input("run-selector", "value"))

    @app.callback(
        Output("reliability-chart",  "figure"),
        Output("wis-coverage-chart", "figure"),
        Output("cv-chart",           "figure"),
        *diag_inputs,
    )
    def update_diagnostics(*args):
        county_fips = args[0]
        run_dir_str = args[1] if len(args) > 1 else "__live__"
        pd_df, fc_df, qc, cv_df_run, eval_dict_run, rolling_df, _ = _get_run_data(run_dir_str)
        display_fc = rolling_df if not rolling_df.empty else fc_df

        reliability  = create_reliability_diagram(pd_df, display_fc, qc, county_fips)
        wis_cov      = _build_wis_coverage_summary(eval_dict_run, cv_df_run)
        cv_stability = _build_cv_stability_chart(cv_df_run)
        return reliability, wis_cov, cv_stability

    return app


# ---------------------------------------------------------------------------
# Demo launcher (synthetic data)
# ---------------------------------------------------------------------------

def _build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng       = np.random.default_rng(0)
    dates     = pd.date_range("2020-07-01", periods=175, freq="W-WED")
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
                "TFT-lo-50.0": max(0.0, med - 0.4),
                "TFT-median":  med,
                "TFT-hi-50.0": med + 0.4,
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
