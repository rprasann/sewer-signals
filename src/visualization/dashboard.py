"""
Interactive Dash dashboard for wastewater COVID-19 surveillance forecasting.

Usage
-----
    from src.visualization.dashboard import create_app

    app = create_app(
        processed_df=processed,   # WastewaterProcessor output
        forecast_df=forecast,     # WastewaterTFT.predict() output
        model=model,              # fitted WastewaterTFT (optional — enables live VI)
        sludge_df=sludge,         # optional: sludge track for Section 4.1 comparison
        liquid_df=liquid,         # optional: liquid track for Section 4.1 comparison
    )
    app.run(debug=False, port=8050)

Demo (synthetic data)
---------------------
    from src.visualization.dashboard import run_demo
    run_demo()

Dashboard sections
------------------
Row 1 : Header — county selector, forecast date selector, outbreak alert badge
Row 2 : Forecast ribbon chart (left)   |   VSN importance bar chart (right)
Row 3 : Attention heatmap (left)        |   Wave synchrony heatmap (right)
Row 4 : Two-track comparison (sludge vs. liquid) — full width
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from src.config import (
    BAY_AREA_FIPS,
    COUNTY_COL,
    DASH_HOST,
    DASH_PORT,
    FIPS_TO_COUNTY,
    NWSS_DATE_COL,
    TARGET_COL,
)
from src.evaluation.metrics import OutbreakDetector, OutbreakRecovery, QuantileColumns
from src.visualization.attention_plots import (
    extract_attention_weights,
    extract_vsn_weights,
    plot_attention_heatmap,
    plot_forecast,
    plot_two_track_comparison,
    plot_vsn_importance,
    plot_wave_synchrony,
)


# ---------------------------------------------------------------------------
# Shared styling constants
# ---------------------------------------------------------------------------

_DARK_BG   = "#16213e"
_CARD_BG   = "#1a1a2e"
_TEXT_CLR  = "#e0e0e0"
_ACCENT    = "#F97316"   # orange — matches forecast colour

_CARD_STYLE = {
    "backgroundColor": _CARD_BG,
    "border": "1px solid #2a2a4a",
    "borderRadius": "8px",
    "padding": "12px",
}

_LABEL_STYLE = {
    "color": _TEXT_CLR,
    "fontSize": "13px",
    "marginBottom": "4px",
}


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def create_app(
    processed_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model=None,
    sludge_df: Optional[pd.DataFrame] = None,
    liquid_df: Optional[pd.DataFrame] = None,
    q_cols: Optional[QuantileColumns] = None,
) -> dash.Dash:
    """Build and return the Dash application object.

    Parameters
    ----------
    processed_df : Full processor output (actuals + features).
    forecast_df  : NeuralForecast predict() output.
    model        : Fitted WastewaterTFT — enables live attention/VSN extraction.
    sludge_df    : Processed sludge-track DataFrame for Section 4.1 comparison.
    liquid_df    : Processed liquid-track DataFrame for Section 4.1 comparison.
    q_cols       : Quantile column map; auto-detected from forecast_df if None.
    """
    # ── Pre-compute static artefacts ─────────────────────────────────────────
    if q_cols is None:
        try:
            q_cols = QuantileColumns.auto_detect(forecast_df)
        except ValueError:
            q_cols = QuantileColumns()

    available_counties = sorted(processed_df[COUNTY_COL].unique().tolist())
    county_options = [
        {"label": FIPS_TO_COUNTY.get(f, f), "value": f}
        for f in available_counties
    ]

    available_dates = sorted(forecast_df["ds"].unique())
    date_options = [
        {"label": str(d)[:10], "value": str(d)[:10]}
        for d in available_dates
    ]

    # Outbreak / recovery detection on the full processed set
    onset_detector    = OutbreakDetector()
    recovery_detector = OutbreakRecovery()
    labelled_df = onset_detector.detect_df(processed_df)
    recovery_events = recovery_detector.detect_df(processed_df)

    # Wave synchrony figure (static — computed once)
    synchrony_fig = plot_wave_synchrony(processed_df)

    # Two-track comparison figure (static, if both tracks provided)
    def _has_county_col(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and not df.empty and COUNTY_COL in df.columns

    two_track_counties = (
        list(set(sludge_df[COUNTY_COL].unique()) & set(liquid_df[COUNTY_COL].unique()))
        if _has_county_col(sludge_df) and _has_county_col(liquid_df) else []
    )

    # Pre-extract VSN weights if model is provided
    _vsn_weights: dict = {}
    if model is not None:
        try:
            raw_vsn = extract_vsn_weights(model)
            if raw_vsn:
                _vsn_weights = raw_vsn
                logger.info("VSN weights extracted from fitted model.")
        except Exception as exc:
            logger.warning("VSN extraction skipped: {}", exc)

    # ── App layout ───────────────────────────────────────────────────────────
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Sewer Signals — Bay Area Wastewater Surveillance",
    )

    app.layout = dbc.Container(
        fluid=True,
        style={"backgroundColor": _DARK_BG, "minHeight": "100vh", "padding": "16px"},
        children=[

            # ── Header ──────────────────────────────────────────────────────
            dbc.Row([
                dbc.Col(html.H3(
                    "🦠 Sewer Signals — Bay Area Wastewater Surveillance",
                    style={"color": "#e0e0e0", "margin": "0"},
                ), width=6),
                dbc.Col(html.P(
                    "Temporal Fusion Transformer · PINN Growth-Rate Penalty · 9-County Bay Area",
                    style={"color": "#888", "fontSize": "13px", "marginTop": "8px"},
                ), width=6, className="text-end"),
            ], className="mb-3"),

            # ── Controls row ─────────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("County", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="county-selector",
                        options=county_options,
                        value=available_counties[0] if available_counties else None,
                        clearable=False,
                        style={"backgroundColor": _CARD_BG, "color": _TEXT_CLR,
                               "border": "1px solid #2a2a4a"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Forecast Date", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="date-selector",
                        options=date_options,
                        value=date_options[-1]["value"] if date_options else None,
                        clearable=False,
                        style={"backgroundColor": _CARD_BG, "color": _TEXT_CLR,
                               "border": "1px solid #2a2a4a"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Outbreak Alert", style=_LABEL_STYLE),
                    html.Div(id="alert-badge"),
                ], width=3),
                dbc.Col([
                    html.Label("Recovery Status", style=_LABEL_STYLE),
                    html.Div(id="recovery-badge"),
                ], width=3),
            ], className="mb-3"),

            # ── Row 2: Forecast + VSN ─────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Div(style=_CARD_STYLE, children=[
                        dcc.Graph(id="forecast-chart", config={"displayModeBar": False}),
                    ]),
                ], width=7),
                dbc.Col([
                    html.Div(style=_CARD_STYLE, children=[
                        dcc.Graph(id="vsn-chart", config={"displayModeBar": False}),
                    ]),
                ], width=5),
            ], className="mb-3"),

            # ── Row 3: Attention + Wave synchrony ────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Div(style=_CARD_STYLE, children=[
                        dcc.Graph(id="attention-chart", config={"displayModeBar": False}),
                    ]),
                ], width=6),
                dbc.Col([
                    html.Div(style=_CARD_STYLE, children=[
                        dcc.Graph(
                            id="synchrony-chart",
                            figure=synchrony_fig,
                            config={"displayModeBar": False},
                        ),
                    ]),
                ], width=6),
            ], className="mb-3"),

            # ── Row 4: Two-track comparison ───────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Div(style=_CARD_STYLE, children=[
                        dcc.Graph(id="two-track-chart", config={"displayModeBar": False}),
                    ]),
                ]),
            ], className="mb-2"),

            # Lightweight footer
            dbc.Row(dbc.Col(html.P(
                "Data: CDC NWSS · Model: TFT + PINNWastewaterLoss · "
                "Signals are log1p-transformed and RobustScaled.",
                style={"color": "#555", "fontSize": "11px", "textAlign": "center"},
            ))),
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────

    @app.callback(
        Output("forecast-chart",  "figure"),
        Output("alert-badge",     "children"),
        Output("recovery-badge",  "children"),
        Input("county-selector",  "value"),
        Input("date-selector",    "value"),
    )
    def update_forecast_and_alert(county_fips: str, date_str: str):
        if not county_fips:
            return _empty_fig("Select a county"), _no_badge(), _no_badge()

        # ── Onset dates for this county ──────────────────────────────────────
        county_labelled = labelled_df[labelled_df[COUNTY_COL] == county_fips]
        onset_dates = list(
            county_labelled.loc[county_labelled["onset"], NWSS_DATE_COL]
        )

        # ── Recovery dates for this county ───────────────────────────────────
        county_recoveries = [e for e in recovery_events if e.county == county_fips]
        recovery_dates = [
            e.recovery_date for e in county_recoveries if e.recovery_date is not None
        ]

        # ── Forecast chart ───────────────────────────────────────────────────
        fcast_fig = plot_forecast(
            actual_df=processed_df,
            forecast_df=forecast_df,
            county_fips=county_fips,
            q_cols=q_cols,
            onset_dates=onset_dates,
            recovery_dates=recovery_dates,
        )

        # ── Alert badge ──────────────────────────────────────────────────────
        #    Check whether the selected forecast date is within an active onset window
        if date_str and onset_dates:
            try:
                sel_date = pd.Timestamp(date_str)
                active_onset = any(
                    abs((sel_date - od).days) <= 21 for od in onset_dates
                )
            except Exception:
                active_onset = False
        else:
            active_onset = False

        alert_badge = dbc.Alert(
            "⚠ OUTBREAK ONSET" if active_onset else "✓ No Active Alert",
            color="danger" if active_onset else "success",
            style={"padding": "6px 14px", "marginBottom": "0", "fontSize": "13px"},
        )

        # ── Recovery badge ───────────────────────────────────────────────────
        completed = [e for e in county_recoveries if e.duration_weeks is not None]
        if completed:
            avg_wks = float(np.mean([e.duration_weeks for e in completed]))
            rec_text = f"Recovery avg: {avg_wks:.1f} wks"
            rec_colour = "info"
        else:
            rec_text = "Recovery: pending"
            rec_colour = "secondary"

        recovery_badge = dbc.Alert(
            rec_text,
            color=rec_colour,
            style={"padding": "6px 14px", "marginBottom": "0", "fontSize": "13px"},
        )

        return fcast_fig, alert_badge, recovery_badge

    @app.callback(
        Output("attention-chart", "figure"),
        Input("county-selector",  "value"),
    )
    def update_attention(county_fips: str):
        if not county_fips:
            return _empty_fig("Select a county")

        # Try live extraction; fall back to synthetic illustrative weights
        weights = None
        if model is not None:
            try:
                weights = extract_attention_weights(model)
            except Exception:
                pass

        if weights is None:
            # Synthetic illustrative heatmap (decaying attention over context)
            T = 16
            weights = np.zeros((4, T))
            for i in range(4):
                w = np.exp(-0.15 * np.arange(T)[::-1])
                w += 0.2 * np.exp(-0.5 * (np.arange(T) - (T - 4 + i)) ** 2)
                w /= w.sum()
                weights[i] = w

        return plot_attention_heatmap(
            weights, county=county_fips,
            title="Temporal Attention Weights (illustrative)" if model is None
                  else "Temporal Attention Weights",
        )

    @app.callback(
        Output("vsn-chart", "figure"),
        Input("county-selector", "value"),
    )
    def update_vsn(county_fips: str):
        if not county_fips:
            return _empty_fig("Select a county")

        # Use pre-extracted weights, or synthetic fallback
        if _vsn_weights:
            weights = _vsn_weights
        else:
            from src.models.tft_model import HIST_COVARIATES
            n = len(HIST_COVARIATES)
            # Synthetic: give more weight to lag/decay features
            w = np.array([0.22, 0.18, 0.14, 0.16, 0.20, 0.10])[:n]
            w /= w.sum()
            weights = {"historical": w}

        title = (
            "VSN Feature Importance (illustrative)" if not _vsn_weights
            else "VSN Feature Importance"
        )
        return plot_vsn_importance(weights, role="historical", title=title)

    @app.callback(
        Output("two-track-chart", "figure"),
        Input("county-selector",  "value"),
    )
    def update_two_track(county_fips: str):
        if not county_fips:
            return _empty_fig("Select a county")

        if not _has_county_col(sludge_df) or not _has_county_col(liquid_df):
            return _empty_fig(
                "Two-track comparison unavailable — "
                "pass sludge_df and liquid_df to create_app()"
            )
        if county_fips not in two_track_counties:
            return _empty_fig(
                f"No liquid-track data for {FIPS_TO_COUNTY.get(county_fips, county_fips)}"
            )

        return plot_two_track_comparison(sludge_df, liquid_df, county_fips)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_fig(message: str = "No data") -> "go.Figure":
    """Return a blank placeholder figure with a centred annotation."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#aaa"),
    )
    fig.update_layout(
        height=350, paper_bgcolor=_CARD_BG, plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT_CLR), margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _no_badge() -> dbc.Alert:
    return dbc.Alert(
        "—", color="secondary",
        style={"padding": "6px 14px", "marginBottom": "0", "fontSize": "13px"},
    )


# ---------------------------------------------------------------------------
# Demo launcher (synthetic data)
# ---------------------------------------------------------------------------

def _build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic processed_df + forecast_df for dashboard demo."""
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-02", periods=60, freq="W")
    counties = ["06001", "06075", "06085"]

    rows = []
    for ds in dates:
        for fips in counties:
            conc = float(rng.exponential(5_000))
            rows.append({
                COUNTY_COL:  fips,
                NWSS_DATE_COL: ds,
                TARGET_COL:  float(np.log1p(conc)),
                "concentration": conc,
            })
    processed_df = pd.DataFrame(rows)

    # Synthetic forecast_df (last 8 weeks)
    forecast_rows = []
    for ds in dates[-8:]:
        for fips in counties:
            med = float(rng.exponential(1.5))
            forecast_rows.append({
                "unique_id":   fips,
                "ds":          ds,
                "TFT-lo-95.0": max(0.0, med - 0.8),
                "TFT-lo-50.0": max(0.0, med - 0.3),
                "TFT-median":  med,
                "TFT-hi-50.0": med + 0.3,
                "TFT-hi-95.0": med + 0.8,
            })
    forecast_df = pd.DataFrame(forecast_rows)

    return processed_df, forecast_df


def run_demo(host: str = DASH_HOST, port: int = DASH_PORT, debug: bool = True) -> None:
    """Launch the dashboard with synthetic demo data."""
    processed_df, forecast_df = _build_demo_data()
    app = create_app(processed_df=processed_df, forecast_df=forecast_df)
    logger.info("Demo dashboard starting at http://{}:{}", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_demo()
