"""
Logging setup, rich console reporting, and LLM summarisation utilities.

Used by both the test suite (for diagnostic output) and the main pipeline
(for operator-facing run summaries and automated public health briefings).
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    import pandas as pd
    from src.evaluation.metrics import EvalResult, LeadTimeResult

console = Console()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(level: str = "INFO") -> None:
    """Configure loguru for rich-compatible stderr output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
    )
    # Silence PyTorch Lightning's verbose startup messages (GPU/TPU availability,
    # LitLogger tip, etc.) — these clutter CV fold output without adding signal.
    for _noisy in ("pytorch_lightning", "lightning", "lightning_fabric", "lightning.pytorch"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)
    # Suppress the PL _pytree.py LeafSpec deprecation warning
    warnings.filterwarnings("ignore", message=".*LeafSpec.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*treespec.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Loss-function diagnostics
# ---------------------------------------------------------------------------

def print_pinball_asymmetry_table(quantile_levels: list[float]) -> None:
    """Print a table showing pinball asymmetry for each quantile level.

    Illustrates why low quantiles penalise overestimation heavily (they model
    lower tails) and vice versa — critical for understanding why the model
    learns calibrated uncertainty bands.
    """
    import torch
    from src.models.loss_functions import PinballLoss

    table = Table(
        title="[bold]Pinball Loss Asymmetry[/bold]  — how penalty distributes across quantiles",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
    )
    table.add_column("Quantile (q)", justify="center")
    table.add_column("Loss: ŷ = y + 1 (overpredict)", justify="right")
    table.add_column("Loss: ŷ = y − 1 (underpredict)", justify="right")
    table.add_column("Penalty ratio (over/under)", justify="right")
    table.add_column("Interpretation", justify="left")

    fn = PinballLoss(quantile_levels)
    y_true = torch.zeros(1, 1)   # scalar true value = 0

    for q in quantile_levels:
        idx = quantile_levels.index(q)
        q_tensor = torch.tensor([[quantile_levels]])

        y_over  = torch.zeros(1, 1, len(quantile_levels))
        y_under = torch.zeros(1, 1, len(quantile_levels))
        y_over[0, 0, idx]  = 1.0    # ŷ = y + 1 at this quantile slot
        y_under[0, 0, idx] = -1.0   # ŷ = y - 1

        loss_over  = float(fn(y_over,  y_true))
        loss_under = float(fn(y_under, y_true))
        ratio = loss_over / (loss_under + 1e-9)

        if q < 0.5:
            interp = "[red]Low q — overpredict penalised harder[/red]"
        elif q > 0.5:
            interp = "[green]High q — underpredict penalised harder[/green]"
        else:
            interp = "[yellow]Median — symmetric[/yellow]"

        table.add_row(
            f"q={q:.3f}",
            f"{loss_over:.4f}",
            f"{loss_under:.4f}",
            f"{ratio:.2f}×",
            interp,
        )

    console.print(table)


def print_pinn_comparison_table(scenarios: list[dict]) -> None:
    """Compare MQLoss vs PINNWastewaterLoss across biological plausibility scenarios.

    Each scenario is a dict with 'label', 'median_trajectory' (list of floats),
    and 'y_true' (list of floats). Shows how the PINN penalty adds cost only
    for impossible growth rates, leaving realistic forecasts unchanged.
    """
    import torch
    from neuralforecast.losses.pytorch import MQLoss
    from src.models.loss_functions import PINNWastewaterLoss
    from src.config import QUANTILE_LEVELS

    mql  = MQLoss(quantiles=QUANTILE_LEVELS)
    pinn = PINNWastewaterLoss(quantiles=QUANTILE_LEVELS)

    table = Table(
        title="[bold]MQLoss vs PINNWastewaterLoss[/bold]  — penalty fires only for impossible biology",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
    )
    table.add_column("Scenario", style="bold")
    table.add_column("Max weekly growth rate", justify="right")
    table.add_column("MQLoss", justify="right")
    table.add_column("PINNWastewaterLoss", justify="right")
    table.add_column("PINN overhead", justify="right")
    table.add_column("Biologically possible?", justify="center")

    for s in scenarios:
        traj = torch.tensor(s["median_trajectory"], dtype=torch.float32)
        h = len(traj)
        # Build [1, H, 1*Q] tensor with identical values across all quantiles
        y_hat_raw = traj.view(1, h, 1).expand(1, h, len(QUANTILE_LEVELS)).reshape(1, h, -1)
        y_true = torch.tensor(s["y_true"], dtype=torch.float32).view(1, h, 1)

        mapped_mql  = mql.domain_map(y_hat_raw.clone())
        mapped_pinn = pinn.domain_map(y_hat_raw.clone())

        loss_mql  = float(mql(y=y_true,  y_hat=mapped_mql))
        loss_pinn = float(pinn(y=y_true, y_hat=mapped_pinn))
        overhead  = loss_pinn - loss_mql

        # Compute actual max growth rate
        m = traj.numpy()
        rates = (m[1:] - m[:-1]) / (np.abs(m[:-1]) + 1e-8)
        max_rate = float(rates.max()) if len(rates) > 0 else 0.0

        possible = max_rate <= 2.45  # MAX_DAILY_GROWTH_RATE * 7
        badge = "[green]✓ Yes[/green]" if possible else "[red]✗ No[/red]"
        overhead_str = (
            f"[red]+{overhead:.4f}[/red]" if overhead > 0.01 else f"[dim]+{overhead:.4f}[/dim]"
        )

        table.add_row(
            s["label"],
            f"{max_rate:.2f}  (limit: 2.45)",
            f"{loss_mql:.4f}",
            f"{loss_pinn:.4f}",
            overhead_str,
            badge,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def print_eval_report(result: "EvalResult", title: str = "Evaluation Report") -> None:
    """Print a rich multi-panel evaluation report from an ``EvalResult``."""

    # ── Probabilistic metrics ──────────────────────────────────────────────
    prob_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    prob_table.add_column("Metric", style="bold")
    prob_table.add_column("Value", justify="right")
    prob_table.add_column("Target / Interpretation", style="dim")

    prob_table.add_row(
        "Mean WIS",
        f"{result.mean_wis:.4f}",
        "Lower is better  (0 = perfect, ↑ = more miscalibrated)",
    )
    prob_table.add_row(
        "Coverage 50 % PI",
        _badge_coverage(result.coverage_50, 0.50),
        "Should be ≈ 50 %  (well-calibrated uncertainty)",
    )
    prob_table.add_row(
        "Coverage 95 % PI",
        _badge_coverage(result.coverage_95, 0.95),
        "Should be ≈ 95 %  (well-calibrated uncertainty)",
    )
    prob_table.add_row(
        "SMAPE",
        f"{result.smape:.4f}",
        "0 = perfect, 2.0 = maximum, <0.2 = good for surveillance",
    )

    # ── WIS by county ─────────────────────────────────────────────────────
    county_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    county_table.add_column("County FIPS")
    county_table.add_column("Mean WIS", justify="right")
    county_table.add_column("Bar", justify="left")

    max_wis = max(result.wis_per_county.values(), default=1.0) or 1.0
    for fips, val in sorted(result.wis_per_county.items(), key=lambda kv: -kv[1]):
        bar_len = int(20 * val / max_wis)
        bar = "[red]" + "█" * bar_len + "[/red]" + "░" * (20 - bar_len)
        county_table.add_row(fips, f"{val:.4f}", bar)

    # ── Outbreak & lead-time ───────────────────────────────────────────────
    lt = result.lead_time
    outbreak_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    outbreak_table.add_column("Metric", style="bold")
    outbreak_table.add_column("Value", justify="right")
    outbreak_table.add_column("Interpretation", style="dim")

    outbreak_table.add_row("Actual onset events", str(result.n_actual_onsets), "")
    outbreak_table.add_row("Model alerts issued",  str(result.n_predicted_alerts), "")
    outbreak_table.add_row("True Positives",  str(lt.tp), "")
    outbreak_table.add_row("False Positives", str(lt.fp), "[dim]Unnecessary alerts[/dim]")
    outbreak_table.add_row("False Negatives", str(lt.fn), "[dim]Missed outbreaks[/dim]")
    outbreak_table.add_row("True Negatives",  str(lt.tn), "")
    outbreak_table.add_row(
        "Sensitivity",
        _badge_metric(lt.sensitivity, 0.70, 0.85),
        "TP / (TP+FN)  — fraction of real outbreaks caught",
    )
    outbreak_table.add_row(
        "Specificity",
        _badge_metric(lt.specificity, 0.70, 0.85),
        "TN / (TN+FP)  — fraction of non-outbreaks correctly silent",
    )
    outbreak_table.add_row(
        "AUC",
        _badge_metric(lt.auc, 0.70, 0.85),
        "0.5 = random, >0.80 = actionable, >0.90 = excellent",
    )
    outbreak_table.add_row(
        "Mean lead time",
        f"{lt.mean_lead_days:.1f} days" if not np.isnan(lt.mean_lead_days) else "N/A",
        "Positive = model alerts BEFORE clinical onset (target: 7–21 days)",
    )

    # ── Render ─────────────────────────────────────────────────────────────
    console.rule(f"[bold white] {title} [/bold white]")
    cutoff_str = str(result.cutoff_date.date()) if result.cutoff_date else "all data"
    console.print(
        f"  [dim]Cutoff:[/dim] {cutoff_str}   "
        f"[dim]Series:[/dim] {result.n_series}   "
        f"[dim]Observations:[/dim] {result.n_observations}"
    )
    console.print(Columns([
        Panel(prob_table,    title="[bold]Probabilistic Metrics[/bold]", border_style="blue"),
        Panel(county_table,  title="[bold]WIS by County[/bold]",        border_style="blue"),
    ]))
    console.print(Panel(outbreak_table, title="[bold]Outbreak Detection & Lead Time[/bold]", border_style="magenta"))
    _print_key_takeaways(result)
    console.print()


def print_outbreak_timeline(
    df_with_onsets: "pd.DataFrame",
    date_col: str,
    county_col: str,
    onset_col: str = "onset",
    signal_col: str | None = None,
    max_rows: int = 40,
) -> None:
    """Print a chronological timeline of detected onset events."""
    import pandas as pd

    events = df_with_onsets[df_with_onsets[onset_col]].sort_values(date_col)

    if events.empty:
        console.print("[dim]No onset events detected in this window.[/dim]")
        return

    table = Table(
        title="[bold]Outbreak Onset Timeline[/bold]",
        box=box.ROUNDED, show_header=True, header_style="bold red",
    )
    table.add_column("Date", style="bold")
    table.add_column("County FIPS")
    if signal_col and signal_col in events.columns:
        table.add_column(f"Signal ({signal_col})", justify="right")

    for _, row in events.head(max_rows).iterrows():
        cols = [str(row[date_col])[:10], str(row[county_col])]
        if signal_col and signal_col in events.columns:
            cols.append(f"{row[signal_col]:.3f}")
        table.add_row(*cols)

    if len(events) > max_rows:
        table.add_row(f"[dim]… {len(events) - max_rows} more rows[/dim]", "", "")

    console.print(table)


def print_cv_summary(cv_df: "pd.DataFrame") -> None:
    """Print a compact cross-validation results table."""
    if cv_df.empty:
        console.print("[red]No CV folds completed.[/red]")
        return

    table = Table(
        title=f"[bold]Expanding-Window Cross-Validation  ({len(cv_df)} folds)[/bold]",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
    )
    for col in ["cutoff_date", "mean_wis", "coverage_50", "coverage_95",
                "smape", "sensitivity", "specificity", "auc", "mean_lead_days"]:
        if col in cv_df.columns:
            table.add_column(col.replace("_", " ").title(), justify="right")

    for _, row in cv_df.iterrows():
        vals = []
        for col in ["cutoff_date", "mean_wis", "coverage_50", "coverage_95",
                    "smape", "sensitivity", "specificity", "auc", "mean_lead_days"]:
            if col not in cv_df.columns:
                continue
            v = row[col]
            if col == "cutoff_date":
                vals.append(str(v)[:10] if v is not None else "—")
            elif isinstance(v, float) and np.isnan(v):
                vals.append("—")
            else:
                vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        table.add_row(*vals)

    # Summary row
    numeric_cols = [c for c in ["mean_wis", "smape", "auc"] if c in cv_df.columns]
    means = {c: cv_df[c].mean() for c in numeric_cols}
    summary_vals = []
    for col in ["cutoff_date", "mean_wis", "coverage_50", "coverage_95",
                "smape", "sensitivity", "specificity", "auc", "mean_lead_days"]:
        if col not in cv_df.columns:
            continue
        if col == "cutoff_date":
            summary_vals.append("[bold]MEAN[/bold]")
        elif col in means:
            summary_vals.append(f"[bold]{means[col]:.3f}[/bold]")
        else:
            summary_vals.append("—")
    table.add_section()
    table.add_row(*summary_vals)

    console.print(table)

    # Trend note
    if "mean_wis" in cv_df.columns and len(cv_df) >= 3:
        wis_trend = np.polyfit(range(len(cv_df)), cv_df["mean_wis"].fillna(0), 1)[0]
        if wis_trend > 0.01:
            console.print("[red]⚠  WIS is trending upward across folds — model may struggle on later waves.[/red]")
        elif wis_trend < -0.01:
            console.print("[green]✓  WIS is trending downward — model improves as training window expands.[/green]")
        else:
            console.print("[dim]WIS stable across folds.[/dim]")


# ---------------------------------------------------------------------------
# LLM public health summarisation
# ---------------------------------------------------------------------------

def generate_public_health_summary(
    forecast_df: "pd.DataFrame",
    eval_result: "EvalResult",
    vsn_weights: dict,
    county_fips: list[str] | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a natural-language public health bulletin via a local LM Studio server.

    Sends a structured prompt to the LM Studio OpenAI-compatible endpoint at
    ``base_url`` (default: ``LOCAL_LLM_BASE_URL`` from config).  No cloud API
    key is required — LM Studio runs fully offline.

    Prompt contains:
    - Per-county median weekly new-case trajectories and 95 % PI widths
    - Top-5 VSN feature importances from the TFT encoder
    - Outbreak detection performance (sensitivity, AUC, mean lead time)
    - Recovery duration statistics

    Parameters
    ----------
    forecast_df  : Inverse-transformed forecast DataFrame (weekly new case counts
                   in quantile columns, after expm1).  Expected columns: ``unique_id``,
                   ``ds``, and the NeuralForecast quantile column names.
    eval_result  : EvalResult from ``evaluate()`` on the hold-out test set.
    vsn_weights  : Output of ``WastewaterTFT.variable_importance()``.
    county_fips  : FIPS codes to include in the bulletin (default: all).
    base_url     : LM Studio server URL (overrides ``LOCAL_LLM_BASE_URL``).
    model        : Model identifier as shown in LM Studio (overrides
                   ``LOCAL_LLM_MODEL``).

    Returns
    -------
    str — The generated summary text.

    Raises
    ------
    RuntimeError if the LM Studio server is unreachable or returns an error.
    """
    from openai import OpenAI
    from src.config import FIPS_TO_COUNTY, LLM_MAX_TOKENS, LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL

    _base_url = base_url or LOCAL_LLM_BASE_URL
    _model    = model    or LOCAL_LLM_MODEL

    # ── Forecast summary per county ───────────────────────────────────────────
    fips_subset = set(county_fips) if county_fips else None
    forecast_lines: list[str] = []
    for uid, grp in forecast_df.groupby("unique_id"):
        if fips_subset and str(uid) not in fips_subset:
            continue
        grp = grp.sort_values("ds")
        county_name = FIPS_TO_COUNTY.get(str(uid), str(uid))

        q50_col  = next((c for c in grp.columns if c.endswith("-median")),  None)
        q025_col = next((c for c in grp.columns if c.endswith("-lo-95.0")), None)
        q975_col = next((c for c in grp.columns if c.endswith("-hi-95.0")), None)
        if q50_col is None:
            continue

        medians = grp[q50_col].fillna(0).round(0).astype(int).tolist()[:4]
        dates   = grp["ds"].dt.strftime("%Y-%m-%d").tolist()[:4]
        pi_str  = ""
        if q025_col and q975_col:
            avg_width = float((grp[q975_col] - grp[q025_col]).mean())
            pi_str = f"  [95 % PI avg width ≈ {int(avg_width):,} new cases]"

        pts = ", ".join(f"{d}={m:,}" for d, m in zip(dates, medians))
        forecast_lines.append(f"  {county_name}: {pts}{pi_str}")

    # ── VSN importances ───────────────────────────────────────────────────────
    vsn_lines: list[str] = []
    for role, df_vi in vsn_weights.items():
        if not hasattr(df_vi, "head"):
            continue
        top5 = df_vi.dropna(subset=["importance"]).head(5)
        if top5.empty:
            continue
        items = ", ".join(
            f"{row['variable']} ({row['importance']:.3f})"
            for _, row in top5.iterrows()
        )
        vsn_lines.append(f"  {role.title()}: {items}")

    # ── Evaluation summary ────────────────────────────────────────────────────
    lt = eval_result.lead_time
    rec_str = (
        f"{eval_result.mean_recovery_weeks:.1f} weeks"
        if not np.isnan(eval_result.mean_recovery_weeks)
        else "N/A (insufficient data)"
    )
    eval_section = (
        f"  Weighted Interval Score (WIS)  = {eval_result.mean_wis:.4f}  "
        f"(lower is better)\n"
        f"  Symmetric MAPE (SMAPE)         = {eval_result.smape:.4f}\n"
        f"  50 % PI coverage               = {eval_result.coverage_50:.1%}  "
        f"(target 50 %)\n"
        f"  95 % PI coverage               = {eval_result.coverage_95:.1%}  "
        f"(target 95 %)\n"
        f"  Sensitivity                    = {lt.sensitivity:.3f}\n"
        f"  Specificity                    = {lt.specificity:.3f}\n"
        f"  AUC                            = {lt.auc:.3f}\n"
        f"  Mean lead time                 = {lt.mean_lead_days:.1f} days  "
        f"(positive = alerted before clinical onset)\n"
        f"  Mean outbreak recovery         = {rec_str}"
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    forecast_block = (
        "\n".join(forecast_lines) if forecast_lines else "  (no forecast data available)"
    )
    vsn_block = "\n".join(vsn_lines) if vsn_lines else "  (not available)"

    prompt = f"""\
You are an epidemiologist preparing a public health briefing on COVID-19 trends
for officials across the nine-county San Francisco Bay Area.

The data below comes from a Temporal Fusion Transformer (TFT) that predicts weekly
new COVID-19 cases using CDC NWSS wastewater concentration signals as leading indicators.
All nine Bay Area counties are covered using the copies/g dry sludge measurement track.

## 8-Week Quantile Forecast — First 4 Weeks (weekly new COVID-19 cases)

{forecast_block}

## Variable Importance — TFT Variable Selection Networks (top 5 per role)

{vsn_block}

## Model Performance on Hold-out Test Set

{eval_section}

---

Write a 3–4 paragraph public health summary suitable for a weekly surveillance bulletin.

Your summary must:
1. Characterise the current COVID-19 case trajectory (rising, falling, or stable) for the
   Bay Area overall, and highlight any county with a notably elevated or rapidly changing forecast.
2. Interpret the model's uncertainty (prediction interval widths) in plain language — what does
   a wide vs. narrow interval mean for decision-making?
3. Explain what the lead-time metric means for public health response readiness, and whether
   the current value (positive = wastewater signal alerted before clinical case confirmation) is actionable.
4. Note which model features are most predictive (VSN importances) and what that implies about
   the current drivers of COVID-19 trends in the Bay Area.

Avoid statistical jargon.  Write for a non-technical public health decision-maker.
Do not fabricate specific policy recommendations beyond what the data shows.\
"""

    # ── LM Studio API call (OpenAI-compatible) ────────────────────────────────
    logger.info("Calling LM Studio at {} with model {} …", _base_url, _model)
    try:
        client = OpenAI(
            base_url=_base_url,
            api_key="lm-studio",   # required by the openai client; ignored by LM Studio
        )
        response = client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.4,       # lower temp → more factual, less hallucination
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("LM Studio call failed: {}", exc)
        raise RuntimeError(
            f"LM Studio at {_base_url} returned an error: {exc}\n"
            "Make sure LM Studio is running and a model is loaded."
        ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _badge_coverage(val: float, target: float, tol: float = 0.10) -> str:
    pct = f"{val:.1%}"
    if abs(val - target) <= tol:
        return f"[green]{pct}[/green]"
    elif abs(val - target) <= 2 * tol:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


def _badge_metric(val: float, warn: float, good: float) -> str:
    if np.isnan(val):
        return "[dim]N/A[/dim]"
    formatted = f"{val:.3f}"
    if val >= good:
        return f"[green]{formatted}[/green]"
    elif val >= warn:
        return f"[yellow]{formatted}[/yellow]"
    return f"[red]{formatted}[/red]"


def _print_key_takeaways(result: "EvalResult") -> None:
    lt = result.lead_time
    lines: list[str] = []

    # Coverage calibration
    cov50_err = abs(result.coverage_50 - 0.50)
    cov95_err = abs(result.coverage_95 - 0.95)
    if cov50_err > 0.15 or cov95_err > 0.10:
        lines.append(
            f"[yellow]⚠ Prediction intervals are miscalibrated[/yellow] "
            f"(50 % PI={result.coverage_50:.0%}, 95 % PI={result.coverage_95:.0%}). "
            "Consider wider intervals or recalibration."
        )
    else:
        lines.append("[green]✓ Prediction intervals are well-calibrated.[/green]")

    # AUC
    if not np.isnan(lt.auc):
        if lt.auc >= 0.85:
            lines.append(f"[green]✓ AUC={lt.auc:.3f} — model is a strong outbreak detector.[/green]")
        elif lt.auc >= 0.70:
            lines.append(f"[yellow]△ AUC={lt.auc:.3f} — moderate discriminative power.[/yellow]")
        else:
            lines.append(f"[red]✗ AUC={lt.auc:.3f} — model barely better than random for outbreak detection.[/red]")

    # Lead time vs. target window (7–21 days)
    if not np.isnan(lt.mean_lead_days):
        if 7 <= lt.mean_lead_days <= 21:
            lines.append(
                f"[green]✓ Mean lead time {lt.mean_lead_days:.1f} days is within the "
                f"target 7–21 day biosurveillance window.[/green]"
            )
        elif lt.mean_lead_days > 21:
            lines.append(
                f"[yellow]△ Lead time {lt.mean_lead_days:.1f} days exceeds 21 days — "
                "model may be overly sensitive (early false alarms).[/yellow]"
            )
        else:
            lines.append(
                f"[red]✗ Lead time {lt.mean_lead_days:.1f} days — model is alerting "
                "too late for clinical intervention.[/red]"
            )

    # False negatives
    if lt.fn > 0:
        lines.append(
            f"[red]✗ {lt.fn} missed outbreak(s). "
            "Consider lowering the alert threshold or extending the forecast horizon.[/red]"
        )

    text = "\n".join(f"  {l}" for l in lines)
    console.print(Panel(text, title="[bold]Key Takeaways[/bold]", border_style="yellow"))
