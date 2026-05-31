"""
Plotly visualization functions for TFT interpretability and Bay Area wave analysis.

All functions return ``plotly.graph_objects.Figure`` — renderable in Jupyter,
saveable with ``fig.write_html()`` / ``fig.write_image()``, and embeddable in the
Dash app via ``dcc.Graph(figure=fig)``.

Extraction
----------
extract_attention_weights   — hook temporal_fusion_decoder.attention → [H, T] array
extract_vsn_weights         — hook history/future/static VSN → per-role weight arrays

Plots
-----
plot_attention_heatmap      — temporal self-attention weight heatmap
plot_vsn_importance         — VSN feature importance horizontal bar chart
plot_two_track_comparison   — sludge vs. liquid decay-rate overlay (Section 4.1)
plot_wave_synchrony         — cross-county log1p-concentration Pearson heatmap
plot_forecast               — forecast ribbon + quantile shading vs. actuals
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from src.config import COUNTY_COL, FIPS_TO_COUNTY, NWSS_DATE_COL, TARGET_COL

if TYPE_CHECKING:
    from src.evaluation.metrics import QuantileColumns
    from src.models.tft_model import WastewaterTFT


# ---------------------------------------------------------------------------
# Colour palette — Operational Dashboard (high-contrast)
# ---------------------------------------------------------------------------

# Sludge track: sky blue solid lines (primary track, left axis)
_SLUDGE_COLOUR  = "#38BDF8"
# Liquid track: violet dashed lines (secondary track — visually distinct from
# both sludge blue and the orange forecast)
_LIQUID_COLOUR  = "#C084FC"
# Actuals: near-white for maximum contrast on the dark card background
_ACTUAL_COLOUR  = "#F5F5F5"
# Forecast: bright orange — maximally distinct from actuals and the blue WW signal
_MEDIAN_COLOUR  = "#F97316"
_PI50_COLOUR    = "rgba(249, 115, 22, 0.30)"   # orange 30% — inner 50% band
_PI95_COLOUR    = "rgba(249, 115, 22, 0.12)"   # orange 12% — outer 95% band
_ALERT_COLOUR   = "#F87171"   # soft red — onset markers
_CLEAR_COLOUR   = "#4ADE80"   # bright green — recovery markers
# Decay-rate axis accent (right-axis title, zero-line)
_DECAY_AXIS_CLR = "#FDE68A"   # amber — distinguishes the right axis from concentration


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

@contextmanager
def _forward_hooks(model, hook_map: dict):
    """Context manager that registers forward hooks and removes them on exit.

    ``hook_map`` : {module_name: list_to_append_captured_outputs}
    """
    handles = []
    for name, module in model.named_modules():
        if name in hook_map:
            store = hook_map[name]

            def _make_hook(s):
                def _hook(_module, _input, output):
                    s.append(output)
                return _hook

            handles.append(module.register_forward_hook(_make_hook(store)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def extract_attention_weights(
    model: "WastewaterTFT",
) -> Optional[np.ndarray]:
    """Extract and average temporal attention weights from a fitted TFT.

    Registers a forward hook on ``temporal_fusion_decoder.attention``,
    calls ``model.predict()``, then removes the hook.

    Returns
    -------
    np.ndarray of shape ``[T, T]`` — attention weights averaged over batch
    and heads (rows = query time-steps, columns = key time-steps).
    Returns ``None`` if the model is not fitted or extraction fails.
    """
    if model._nf is None:
        logger.warning("Model not fitted — returning None for attention weights.")
        return None

    tft = model._tft
    captured: dict[str, list] = {"temporal_fusion_decoder.attention": []}

    try:
        with _forward_hooks(tft, captured):
            model.predict()
    except Exception as exc:
        logger.warning("Attention extraction failed during predict(): {}", exc)
        return None

    outputs = captured["temporal_fusion_decoder.attention"]
    if not outputs:
        logger.warning("No attention outputs captured — hook may have missed the module.")
        return None

    # outputs[-1] is (attn_out, attn_prob); attn_prob: [B, n_heads, T, T]
    _, attn_prob = outputs[-1]
    weights = attn_prob.detach().float().cpu().numpy()
    return weights.mean(axis=(0, 1))   # [T, T]


def extract_vsn_weights(
    model: "WastewaterTFT",
) -> Optional[dict[str, np.ndarray]]:
    """Extract Variable Selection Network soft-weights from a fitted TFT.

    Hooks ``temporal_encoder.history_vsn`` and ``temporal_encoder.future_vsn``
    (if present), returning softmax weights averaged over batch and time.

    Returns
    -------
    Dict with keys ``"historical"`` and ``"future"`` (when present),
    each mapping to a 1-D ``np.ndarray`` of normalised importance scores
    aligned to the corresponding covariate list.
    Returns ``None`` if extraction fails.
    """
    if model._nf is None:
        return None

    tft = model._tft
    hook_targets = {
        "temporal_encoder.history_vsn": [],
        "temporal_encoder.future_vsn":  [],
    }
    present = {k for k, _ in tft.named_modules() if k in hook_targets}
    active = {k: hook_targets[k] for k in present}

    try:
        with _forward_hooks(tft, active):
            model.predict()
    except Exception as exc:
        logger.warning("VSN extraction failed: {}", exc)
        return None

    result: dict[str, np.ndarray] = {}

    hist_outputs = active.get("temporal_encoder.history_vsn", [])
    if hist_outputs:
        _, sparse = hist_outputs[-1]        # (ctx, sparse_weights)
        # sparse_weights: [B, T, n_vars]  → average over B and T
        result["historical"] = sparse.detach().float().cpu().numpy().mean(axis=(0, 1))

    futr_outputs = active.get("temporal_encoder.future_vsn", [])
    if futr_outputs:
        _, sparse = futr_outputs[-1]
        result["future"] = sparse.detach().float().cpu().numpy().mean(axis=(0, 1))

    return result or None


# ---------------------------------------------------------------------------
# VSN momentum audit (Phase 4)
# ---------------------------------------------------------------------------

# Features that should carry the burst/acceleration signal.
# Importance below _MOMENTUM_MIN_IMPORTANCE triggers a warning.
_MOMENTUM_FEATURES = (
    "vel_concentration",
    "accel_concentration",
    "vel_concentration_lag1w",
)
_MOMENTUM_MIN_IMPORTANCE = 0.067  # ~1/15 = random baseline for 15 features


def vsn_momentum_audit(
    model: "WastewaterTFT",
    feature_list: Optional[list[str]] = None,
) -> Optional[dict[str, float]]:
    """Extract VSN weights and warn when momentum features are below-chance.

    Calls ``extract_vsn_weights`` on the fitted model, then checks whether
    each momentum feature's normalised importance is above ``1/n_features``
    (the random-chance baseline).  Logs a warning for any under-weighted
    feature so the user knows before interpreting AUC results.

    Parameters
    ----------
    model        : Fitted WastewaterTFT.
    feature_list : Ordered list of HIST_COVARIATES fed to the model.
                   If None, the function cannot map weight indices to names.

    Returns
    -------
    Dict mapping feature name → normalised importance (0–1), or None on failure.
    """
    vsn = extract_vsn_weights(model)
    if vsn is None:
        logger.warning("VSN momentum audit: weight extraction returned None.")
        return None

    hist_weights = vsn.get("historical")
    if hist_weights is None:
        logger.warning("VSN momentum audit: no historical VSN weights found.")
        return None

    if feature_list is None:
        logger.warning(
            "VSN momentum audit: feature_list not provided; returning raw weights."
        )
        return {"raw": hist_weights.tolist()}

    n = min(len(feature_list), len(hist_weights))
    importance: dict[str, float] = {
        feature_list[i]: float(hist_weights[i]) for i in range(n)
    }

    chance = 1.0 / n
    for feat in _MOMENTUM_FEATURES:
        imp = importance.get(feat)
        if imp is None:
            continue
        if imp < chance:
            logger.warning(
                "VSN momentum audit: '{}' importance={:.4f} is BELOW chance ({:.4f}). "
                "Consider checking feature scaling or adding derivative features.",
                feat, imp, chance,
            )
        else:
            logger.info(
                "VSN momentum audit: '{}' importance={:.4f} (chance={:.4f}) ✓",
                feat, imp, chance,
            )

    return importance


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    weights: np.ndarray,
    horizon_labels: Optional[list[str]] = None,
    context_labels: Optional[list[str]] = None,
    title: str = "Temporal Self-Attention Weights",
    county: str = "",
) -> go.Figure:
    """Plotly heatmap of TFT temporal self-attention weights.

    Parameters
    ----------
    weights        : 2-D array ``[T_query, T_key]`` — averaged over heads and batch.
    horizon_labels : Row labels (query time-steps); auto-generated if None.
    context_labels : Column labels (key time-steps); auto-generated if None.
    title          : Figure title.
    county         : Optional county name appended to the title.
    """
    T_q, T_k = weights.shape

    if horizon_labels is None:
        horizon_labels = [f"t+{i+1}" for i in range(T_q)]
    if context_labels is None:
        context_labels = [f"t-{T_k - i - 1}" if i < T_k else "t" for i in range(T_k)]

    fig = go.Figure(go.Heatmap(
        z=weights,
        x=context_labels,
        y=horizon_labels,
        colorscale="Plasma",   # strong cold→hot contrast vs. Viridis
        colorbar=dict(title="Attention weight", thickness=15,
                      title_font=dict(size=12), tickfont=dict(size=11)),
        hoverongaps=False,
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ))

    subtitle = f" — {FIPS_TO_COUNTY.get(county, county)}" if county else ""
    fig.update_layout(
        title=dict(text=f"{title}{subtitle}", font=dict(size=18, color="#162032")),
        xaxis=dict(
            title=dict(text="Context time-step (key)", font=dict(size=14, color="#576880")),
            tickfont=dict(size=12, color="#576880"),
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title=dict(text="Horizon step (query)", font=dict(size=14, color="#576880")),
            tickfont=dict(size=12, color="#576880"),
        ),
        height=420,
        margin=dict(l=80, r=40, t=70, b=70),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F4F7FB",
        font=dict(color="#162032"),
    )
    return fig


# ---------------------------------------------------------------------------
# VSN importance bar chart
# ---------------------------------------------------------------------------

_COVARIATE_CATEGORY: dict[str, str] = {
    # Wastewater signal — primary leading indicator
    "log1p_concentration":       "Wastewater",
    "log1p_concentration_lag1w": "Wastewater",
    "log1p_concentration_lag2w": "Wastewater",
    "log1p_concentration_lag3w": "Wastewater",
    # Case momentum lags — temporal carry-forward signal
    "log1p_new_cases_lag1w":     "Case Lag",
    "log1p_new_cases_lag2w":     "Case Lag",
    "log1p_new_cases_lag3w":     "Case Lag",
    # Outbreak-phase dynamics
    "growth_rate_1w":            "Dynamics",
    "relative_decay_rate":       "Dynamics",
    "vel_concentration":          "Dynamics",
    "accel_concentration":       "Dynamics",
    "vel_concentration_lag1w":   "Dynamics",
    "outlier_flag_int":          "QC Flag",
    "sin_annual_1":  "Seasonality", "cos_annual_1": "Seasonality",
    "sin_annual_2":  "Seasonality", "cos_annual_2": "Seasonality",
    "sin_annual_3":  "Seasonality", "cos_annual_3": "Seasonality",
    "day_of_week_sin": "Calendar",  "day_of_week_cos": "Calendar",
    "month_sin":       "Calendar",  "month_cos":       "Calendar",
    "week_of_year":    "Calendar",
    "log_population":      "Static", "county_fips_encoded": "Static",
    "is_sludge":           "Static",
}

_CATEGORY_COLOURS: dict[str, str] = {
    "Wastewater":  "#38BDF8",   # sky blue
    "Case Lag":    "#F97316",   # orange — matches forecast colour for instant link
    "Dynamics":    "#4ADE80",   # green
    "Seasonality": "#34D399",   # teal
    "Calendar":    "#22D3EE",   # cyan
    "Static":      "#C084FC",   # violet
    "QC Flag":     "#F87171",   # soft red
}


def plot_vsn_importance(
    weights: dict[str, np.ndarray],
    covariate_names: Optional[dict[str, list[str]]] = None,
    role: str = "historical",
    title: str = "Variable Selection Network — Feature Importance",
) -> go.Figure:
    """Horizontal bar chart of VSN soft-weight importances.

    Parameters
    ----------
    weights        : Dict mapping role (``"historical"``, ``"future"``) to 1-D
                     weight arrays (output of ``extract_vsn_weights()`` or
                     ``WastewaterTFT.variable_importance()``).
    covariate_names: Dict mapping role → list of variable names aligned to weights.
                     Defaults to the canonical HIST/FUTURE lists from tft_model.
    role           : Which role's weights to display.
    """
    from src.models.tft_model import FUTURE_COVARIATES, HIST_COVARIATES, STATIC_COVARIATES

    _default_names = {
        "historical": HIST_COVARIATES,
        "future":     FUTURE_COVARIATES,
        "static":     STATIC_COVARIATES,
    }
    names_map = covariate_names or _default_names

    if role not in weights:
        fig = go.Figure()
        fig.add_annotation(text="No VSN weights available for this role.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#576880"))
        fig.update_layout(paper_bgcolor="#F4F7FB", plot_bgcolor="#FFFFFF",
                          font=dict(color="#162032"), height=300, title=title)
        return fig

    w = np.asarray(weights[role])
    names = names_map.get(role, [f"var_{i}" for i in range(len(w))])
    n = min(len(names), len(w))
    names, w = names[:n], w[:n]

    # Sort descending
    order = np.argsort(w)
    names_sorted = [names[i] for i in order]
    w_sorted = w[order]

    colours = [_CATEGORY_COLOURS.get(_COVARIATE_CATEGORY.get(n, ""), "#9E9E9E")
               for n in names_sorted]

    fig = go.Figure(go.Bar(
        x=w_sorted,
        y=names_sorted,
        orientation="h",
        marker_color=colours,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#162032")),
        xaxis=dict(
            title=dict(text="Importance (softmax weight)", font=dict(size=14, color="#576880")),
            tickfont=dict(size=12, color="#576880"),
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(tickfont=dict(size=12, color="#162032")),
        height=max(280, 36 * n + 80),
        margin=dict(l=180, r=40, t=65, b=55),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F4F7FB",
        font=dict(color="#162032"),
    )
    return fig


# ---------------------------------------------------------------------------
# Two-track comparison  (Section 4.1)
# ---------------------------------------------------------------------------

def plot_two_track_comparison(
    sludge_df: pd.DataFrame,
    liquid_df: pd.DataFrame,
    county_fips: str,
    date_col: str = NWSS_DATE_COL,
    signal_col: str = "concentration",
    decay_col: str = "relative_decay_rate",
    title: str = "Sludge vs. Liquid Track: Signal & Decay Rate",
) -> go.Figure:
    """Overlay sludge and liquid WW concentrations with their decay rates.

    The plot has two y-axes:
    - Left:  log1p concentration (both tracks)
    - Right: relative_decay_rate (both tracks)

    This directly supports Section 4.1's argument that the sludge track
    provides sharper decay-rate resolution than the liquid track.
    """
    def _county(df: pd.DataFrame) -> pd.DataFrame:
        if COUNTY_COL in df.columns:
            df = df[df[COUNTY_COL] == county_fips]
        return df.sort_values(date_col)

    sl = _county(sludge_df.copy())
    lq = _county(liquid_df.copy())

    county_name = FIPS_TO_COUNTY.get(county_fips, county_fips)
    fig = go.Figure()

    # ── Concentrations (left axis) — solid=sludge, dashed=liquid ────────────
    if signal_col in sl.columns:
        fig.add_trace(go.Scatter(
            x=sl[date_col], y=np.log1p(sl[signal_col]),
            name="Sludge conc. (solid)", line=dict(color=_SLUDGE_COLOUR, width=2.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Sludge conc.</extra>",
        ))
    if signal_col in lq.columns:
        fig.add_trace(go.Scatter(
            x=lq[date_col], y=np.log1p(lq[signal_col]),
            name="Liquid conc. (dashed)",
            line=dict(color=_LIQUID_COLOUR, width=2.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Liquid conc.</extra>",
        ))

    # ── Decay rates (right axis) — dotted lines, same colour family ─────────
    if decay_col in sl.columns:
        fig.add_trace(go.Scatter(
            x=sl[date_col], y=sl[decay_col],
            name="Sludge decay rate (dotted)",
            line=dict(color=_SLUDGE_COLOUR, width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Sludge decay</extra>",
        ))
    if decay_col in lq.columns:
        fig.add_trace(go.Scatter(
            x=lq[date_col], y=lq[decay_col],
            name="Liquid decay rate (dotted)",
            line=dict(color=_LIQUID_COLOUR, width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Liquid decay</extra>",
        ))

    fig.add_hline(y=0, line=dict(color=_DECAY_AXIS_CLR, width=1, dash="dot"), yref="y2")

    fig.update_layout(
        title=dict(text=f"{title} — {county_name}", font=dict(size=18, color="#F5F5F5")),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=14)),
            tickfont=dict(size=12),
        ),
        # Left axis: concentration — label coloured to clarify axis ownership
        yaxis=dict(
            title=dict(
                text="Concentration  [log1p]",
                font=dict(size=14, color="#B0D4F1"),
            ),
            tickfont=dict(size=12),
            side="left",
        ),
        # Right axis: decay rate — amber accent to distinguish from conc. axis
        yaxis2=dict(
            title=dict(
                text="Relative Decay Rate (7-day)",
                font=dict(size=14, color=_DECAY_AXIS_CLR),
            ),
            tickfont=dict(size=12, color=_DECAY_AXIS_CLR),
            overlaying="y",
            side="right",
            zeroline=True,
            zerolinecolor=_DECAY_AXIS_CLR,
            zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="rgba(22,33,62,0.85)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=13),
            x=0.01,
            y=0.99,
        ),
        height=500,
        margin=dict(l=80, r=100, t=75, b=65),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Wave synchrony heatmap  (Section 4.3)
# ---------------------------------------------------------------------------

def plot_wave_synchrony(
    processed_df: pd.DataFrame,
    fips_to_county: Optional[dict[str, str]] = None,
    signal_col: str = TARGET_COL,
    date_col: str = NWSS_DATE_COL,
    id_col: str = COUNTY_COL,
    title: str = "Cross-County Wave Synchrony — Pearson Correlation",
) -> go.Figure:
    """Cross-county Pearson correlation heatmap of the WW signal.

    Justifies the TFT's multi-head attention: highly synchronized waves across
    counties mean the model can use one county's trajectory to sharpen another's.

    Parameters
    ----------
    processed_df  : Full pipeline output with signal_col per county.
    fips_to_county: Optional FIPS → county name mapping for axis labels.
    """
    f2c = fips_to_county or FIPS_TO_COUNTY

    # Pivot to dates × counties, interpolate short gaps
    pivot = (
        processed_df
        .pivot_table(index=date_col, columns=id_col, values=signal_col, aggfunc="mean")
        .sort_index()
        .interpolate(method="linear", limit=4, axis=0)
    )
    pivot = pivot.dropna(axis=1, thresh=int(0.5 * len(pivot)))  # drop sparse counties

    if pivot.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient counties for synchrony analysis.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#aaa"))
        fig.update_layout(height=350, paper_bgcolor="#16213e",
                          font=dict(color="#e0e0e0"), title=title)
        return fig

    corr = pivot.corr(method="pearson")
    county_labels = [f2c.get(c, c) for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=county_labels,
        y=county_labels,
        zmin=-1, zmax=1,
        colorscale="RdBu_r",
        colorbar=dict(title="Pearson r", thickness=14),
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))

    n = len(county_labels)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#F5F5F5")),
        xaxis=dict(tickangle=-35, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        height=80 + 58 * n,
        width=150 + 58 * n,
        margin=dict(l=130, r=60, t=75, b=110),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )
    return fig


# ---------------------------------------------------------------------------
# Forecast vs. actuals ribbon plot
# ---------------------------------------------------------------------------

def plot_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    county_fips: str,
    q_cols: Optional["QuantileColumns"] = None,
    onset_dates: Optional[list[pd.Timestamp]] = None,
    recovery_dates: Optional[list[pd.Timestamp]] = None,
    title: Optional[str] = None,
    id_col: str = "unique_id",
    date_col: str = "ds",
    actual_id_col: str = COUNTY_COL,
    actual_date_col: str = NWSS_DATE_COL,
    actual_signal_col: str = TARGET_COL,
) -> go.Figure:
    """Forecast quantile ribbon overlaid on actuals for a single county.

    Shaded bands:
    - Dark purple: 50 % prediction interval
    - Light purple: 95 % prediction interval

    Markers:
    - Vertical red dashed lines at confirmed onset dates.
    - Vertical green dashed lines at confirmed recovery dates.
    """
    from src.evaluation.metrics import QuantileColumns

    if q_cols is None:
        try:
            q_cols = QuantileColumns.auto_detect(forecast_df)
        except ValueError:
            q_cols = QuantileColumns()

    # ── Filter to selected county ────────────────────────────────────────────
    fcast = forecast_df[forecast_df[id_col] == county_fips].sort_values(date_col)
    actual = (
        actual_df[actual_df[actual_id_col] == county_fips]
        .sort_values(actual_date_col)
    )

    county_name = FIPS_TO_COUNTY.get(county_fips, county_fips)
    plot_title = title or f"Forecast vs. Actuals — {county_name}"

    fig = go.Figure()

    # ── 95 % PI ribbon ───────────────────────────────────────────────────────
    if q_cols.q025 in fcast.columns and q_cols.q975 in fcast.columns:
        x_band = list(fcast[date_col]) + list(fcast[date_col])[::-1]
        y_band = (list(fcast[q_cols.q025]) +
                  list(fcast[q_cols.q975])[::-1])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, fill="toself",
            fillcolor=_PI95_COLOUR, line=dict(color="rgba(0,0,0,0)"),
            name="95 % PI", showlegend=True,
            hoverinfo="skip",
        ))

    # ── 50 % PI ribbon ───────────────────────────────────────────────────────
    if q_cols.q25 in fcast.columns and q_cols.q75 in fcast.columns:
        x_band = list(fcast[date_col]) + list(fcast[date_col])[::-1]
        y_band = (list(fcast[q_cols.q25]) +
                  list(fcast[q_cols.q75])[::-1])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, fill="toself",
            fillcolor=_PI50_COLOUR, line=dict(color="rgba(0,0,0,0)"),
            name="50 % PI", showlegend=True,
            hoverinfo="skip",
        ))

    # ── Median forecast ──────────────────────────────────────────────────────
    if q_cols.q50 in fcast.columns:
        fig.add_trace(go.Scatter(
            x=fcast[date_col], y=fcast[q_cols.q50],
            name="Median forecast",
            line=dict(color=_MEDIAN_COLOUR, width=2.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Median forecast</extra>",
        ))

    # ── Actuals — near-white for maximum contrast on dark background ─────────
    if not actual.empty and actual_signal_col in actual.columns:
        fig.add_trace(go.Scatter(
            x=actual[actual_date_col], y=actual[actual_signal_col],
            name="Observed (actuals)", mode="lines+markers",
            line=dict(color=_ACTUAL_COLOUR, width=2.5),
            marker=dict(size=6, color=_ACTUAL_COLOUR, symbol="circle"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>Observed</extra>",
        ))

    # ── Onset / recovery annotations ─────────────────────────────────────────
    for ts in (onset_dates or []):
        fig.add_vline(x=ts, line=dict(color=_ALERT_COLOUR, dash="dot", width=1.5),
                      annotation_text="Onset", annotation_font_color=_ALERT_COLOUR)
    for ts in (recovery_dates or []):
        fig.add_vline(x=ts, line=dict(color=_CLEAR_COLOUR, dash="dot", width=1.5),
                      annotation_text="Recovery", annotation_font_color=_CLEAR_COLOUR)

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=18, color="#F5F5F5")),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=14)),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="Weekly New Cases  [log1p, RobustScaled]",
                       font=dict(size=14)),
            tickfont=dict(size=12),
        ),
        legend=dict(
            bgcolor="rgba(22,33,62,0.85)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=13),
            x=0.01,
            y=0.99,
        ),
        height=440,
        hovermode="x unified",
        margin=dict(l=80, r=40, t=75, b=65),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )
    return fig
