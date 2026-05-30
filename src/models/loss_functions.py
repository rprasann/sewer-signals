"""
Custom loss functions for wastewater probabilistic forecasting.

Architecture
------------
PINNWastewaterLoss (MQLoss subclass)
  ├── domain_map()         — median-anchored cumulative softplus → [B,H,N,Q]
  ├── __call__()           — Pinball (MQLoss) + underdispersion penalty
  └── _underdispersion_penalty() — penalises PI narrower than volatility floor

Design notes (Phase 4 domain_map)
-----------------------------------
The median (Q[0.50]) is the unconstrained raw output.  All other quantiles are
built symmetrically via cumulative softplus:

    lower: Q[0.25] = median − softplus(d2)
           Q[0.10] = median − softplus(d2) − softplus(d1)
           Q[0.025]= median − softplus(d2) − softplus(d1) − softplus(d0)

    upper: Q[0.75] = median + softplus(d3)
           Q[0.90] = median + softplus(d3) + softplus(d4)
           Q[0.975]= median + softplus(d3) + softplus(d4) + softplus(d5)

Why median as anchor (not Q[0.025]):
  • The pinball gradient at Q[0.50] is τ = 0.5 — the largest among all
    quantile levels.  Every training sample contributes equally, so the
    median is the most strongly constrained output and stays close to the
    true conditional median of the data.
  • With a Q[0.025] anchor the optimizer has to balance two weak signals
    (τ = 0.025 pinball + underdispersion penalty on PI width).  The
    resulting equilibrium shifts the anchor — and therefore ALL quantiles —
    far above the data distribution, producing the "forecast jump" at the
    context boundary.
  • With a median anchor the underdispersion penalty controls the spread
    (increment sizes) without moving the level.  The median tracks the
    data; the PI width floats symmetrically around it.

GROWTH_RATE_LAMBDA is kept at 0.0 to isolate the underdispersion penalty and
monotonicity dynamics.  The dynamic step-change cap infrastructure is wired and
ready; set GROWTH_RATE_LAMBDA > 0 in config.py once median magnitude is stable.

Standalone modules (PinballLoss, GrowthRatePenalty) are kept for unit tests and
any direct PyTorch usage outside NeuralForecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from neuralforecast.losses.pytorch import MQLoss

from src.config import (
    GROWTH_RATE_LAMBDA,
    MAX_DAILY_GROWTH_RATE,
    MAX_WEEKLY_STEP_CHANGE,
    MIN_PI_WIDTH,
    MIN_PI_WIDTH_FLOOR,
    MIN_PI_WIDTH_MULTIPLIER,
    QUANTILE_LEVELS,
    STEP_CHANGE_MULTIPLIER,
    UNDERDISPERSION_K,
    UNDERDISPERSION_LAMBDA,
)

# SARS-CoV-2 doubling time ~2 days → max daily growth rate ≈ ln(2)/2 ≈ 0.347
# Processor resamples to weekly → scale by 7 for the per-step threshold
_DATA_FREQ_DAYS = 7
_MAX_WEEKLY_GROWTH_RATE = MAX_DAILY_GROWTH_RATE * _DATA_FREQ_DAYS  # ≈ 2.45


# ---------------------------------------------------------------------------
# Standalone modules (pure PyTorch; useful for unit tests and ablations)
# ---------------------------------------------------------------------------

class PinballLoss(nn.Module):
    """Pinball (quantile) loss averaged across all quantile levels.

    Args:
        quantile_levels: Sorted list of quantile probabilities, e.g. [0.025, …, 0.975].
    """

    def __init__(self, quantile_levels: list[float] = QUANTILE_LEVELS) -> None:
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (B, H, Q)  predicted quantiles
        y_true : (B, H)     ground truth

        Returns
        -------
        Scalar mean pinball loss.
        """
        y_true = y_true.unsqueeze(-1)              # (B, H, 1)
        errors = y_true - y_pred                   # (B, H, Q)
        q = self.quantiles.view(1, 1, -1)          # (1, 1, Q)
        loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
        return loss.mean()


class GrowthRatePenalty(nn.Module):
    """PINN soft constraint on the median forecast's step-over-step growth rate.

    Violations above *max_rate* are penalised quadratically to keep gradients
    smooth everywhere.

    Args:
        lam:        Weight of the penalty term relative to the pinball loss.
        max_rate:   Threshold for a single time step (same units as model output).
        median_idx: Column index of the 0.5-quantile in the last dim of y_pred.
    """

    def __init__(
        self,
        lam: float = GROWTH_RATE_LAMBDA,
        max_rate: float = _MAX_WEEKLY_GROWTH_RATE,
        median_idx: int = 2,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.max_rate = max_rate
        self.median_idx = median_idx

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (B, H, Q)  — median at [:, :, median_idx]

        Returns
        -------
        Scalar penalty term.
        """
        median = y_pred[:, :, self.median_idx]          # (B, H)
        denom = median[:, :-1].abs() + 1e-6
        step_rate = (median[:, 1:] - median[:, :-1]) / denom   # (B, H-1)
        violation = torch.clamp(step_rate - self.max_rate, min=0.0)
        return self.lam * (violation ** 2).mean()


# ---------------------------------------------------------------------------
# NeuralForecast-compatible loss  (the one passed to TFT)
# ---------------------------------------------------------------------------

class PINNWastewaterLoss(MQLoss):
    """MQLoss + PINN growth-rate penalty (no Softplus activation).

    Designed to be passed as the ``loss`` argument to
    ``neuralforecast.models.TFT``.  It satisfies the full ``BasePointLoss``
    contract, including ``domain_map`` and ``__call__``.

    domain_map
    ~~~~~~~~~~
    Median-anchored cumulative softplus.  Q[0.50] is the unconstrained raw
    output; lower quantiles are built downward from it and upper quantiles
    upward, each via softplus increments.  Monotonicity is guaranteed by
    construction and the median stays tightly anchored to the data
    distribution (τ=0.5 pinball gradient dominates).  See module docstring.

    Growth-rate penalty
    ~~~~~~~~~~~~~~~~~~~
    A weak soft constraint on the median forecast's step-over-step growth rate.
    On weekly data the threshold is ``MAX_DAILY_GROWTH_RATE × 7 ≈ 2.45``
    (relative change per step in scaled space).  The penalty is:

        λ · mean( max(0, rate_t − max_rate)² )

    With ``growth_lambda=0.005`` (reduced from 0.05) the term contributes
    < 0.5 % of total loss at typical training steps, preserving the biological-
    plausibility signal in VSN attention weights without flattening PI spread.

    Args:
        quantiles:               Sorted quantile levels; must include 0.5.
        growth_lambda:           Weight of the PINN penalty (0.0 = disabled).
        max_daily_growth_rate:   Per-day biological ceiling; scaled by data_freq_days.
        data_freq_days:          Days per model time-step (7 for weekly data).
        max_step_change:         Static floor / fallback for the growth-rate cap (scaled units).
        step_change_multiplier:  Phase 4 — dyn_cap = multiplier × σ(y_insample[-4:]),
                                 clamped to min=max_step_change.  Parallel to
                                 min_pi_width_multiplier for the underdispersion penalty.
        horizon_weight:          Optional per-step horizon weighting passed to MQLoss.
        underdispersion_k:       Phase 4 — ratio K so effective_lambda = K × pinball_loss.
                                 Keeps the penalty proportional to base loss magnitude.
        min_pi_width_multiplier: Phase 4 — min_width_t = multiplier × σ(y_insample[-4:]).
        min_pi_width_floor:      Absolute minimum width floor (scaled units).
        underdispersion_lambda:  Phase 3 legacy (ignored when underdispersion_k > 0).
        min_pi_width:            Phase 3 legacy (ignored when min_pi_width_multiplier > 0).
    """

    def __init__(
        self,
        quantiles: list[float] = QUANTILE_LEVELS,
        growth_lambda: float = GROWTH_RATE_LAMBDA,
        max_daily_growth_rate: float = MAX_DAILY_GROWTH_RATE,
        data_freq_days: int = _DATA_FREQ_DAYS,
        max_step_change: float = MAX_WEEKLY_STEP_CHANGE,
        step_change_multiplier: float = STEP_CHANGE_MULTIPLIER,
        horizon_weight=None,
        underdispersion_k: float = UNDERDISPERSION_K,
        min_pi_width_multiplier: float = MIN_PI_WIDTH_MULTIPLIER,
        min_pi_width_floor: float = MIN_PI_WIDTH_FLOOR,
        underdispersion_lambda: float = UNDERDISPERSION_LAMBDA,
        min_pi_width: float = MIN_PI_WIDTH,
    ) -> None:
        super().__init__(quantiles=quantiles, horizon_weight=horizon_weight)
        self.growth_lambda = growth_lambda
        self.max_step_growth_rate: float = max_daily_growth_rate * data_freq_days
        # Phase 4: dynamic step-change cap — floor when y_insample unavailable
        self.max_step_change: float = max_step_change
        self.step_change_multiplier: float = step_change_multiplier
        # Phase 4 adaptive parameters
        self.underdispersion_k = underdispersion_k
        self.min_pi_width_multiplier = min_pi_width_multiplier
        self.min_pi_width_floor = min_pi_width_floor
        # Phase 3 legacy (kept for backward compat / ablations)
        self.underdispersion_lambda = underdispersion_lambda
        self.min_pi_width = min_pi_width

        sorted_qs = sorted(quantiles)
        if 0.5 not in sorted_qs:
            raise ValueError("quantiles must include 0.5 for the growth-rate penalty.")
        self.median_idx: int = sorted_qs.index(0.5)

        # Outer 95% PI indices for underdispersion penalty (None if not present)
        self._lower_idx: int | None = sorted_qs.index(0.025) if 0.025 in sorted_qs else None
        self._upper_idx: int | None = sorted_qs.index(0.975) if 0.975 in sorted_qs else None

        # Fires once on the first forward pass to confirm whether NeuralForecast
        # passes y_insample to our __call__.  Determines whether Phase 4 dynamic
        # features (_dynamic_min_width, _growth_penalty dynamic cap) are live.
        self._y_insample_logged: bool = False

    # ------------------------------------------------------------------
    # domain_map — called by NeuralForecast *before* __call__
    # ------------------------------------------------------------------

    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Reshape raw logits to ``[B, H, N, Q]`` with median-anchored monotonicity.

        Q[0.50] is the unconstrained raw output — the pinball gradient at τ=0.5
        is the strongest of all quantile levels, keeping the median tightly
        anchored to the true data distribution.

        Lower quantiles are built downward from the median via reverse-cumsum of
        softplus increments; upper quantiles are built upward the same way:

            Q[0.25]  = median − softplus(d₂)
            Q[0.10]  = median − softplus(d₂) − softplus(d₁)
            Q[0.025] = median − softplus(d₂) − softplus(d₁) − softplus(d₀)

            Q[0.75]  = median + softplus(d₃)
            Q[0.90]  = median + softplus(d₃) + softplus(d₄)
            Q[0.975] = median + softplus(d₃) + softplus(d₄) + softplus(d₅)

        Monotonicity (Q[i] ≤ Q[i+1]) is guaranteed by construction because every
        increment is strictly positive.  At random initialisation (all delta
        logits = 0, softplus(0) ≈ 0.693), the PI spans 2 × 3 × 0.693 ≈ 4.16
        scaled units, centred symmetrically on the median.  The underdispersion
        penalty then widens/narrows this spread without displacing the level.
        """
        reshaped = super().domain_map(y_hat)            # [B, H, N, Q] — reshape only

        # Median: unconstrained anchor at position median_idx (=3 for 7-quantile setup)
        median = reshaped[..., self.median_idx : self.median_idx + 1]  # [B, H, N, 1]

        # ── Lower quantiles (Q[0.025], Q[0.10], Q[0.25]) ────────────────────
        # lower_deltas[..., 0] → leftmost (Q[0.025]), furthest from median
        # lower_deltas[..., 2] → rightmost (Q[0.25]),  closest to median
        lower_deltas = reshaped[..., : self.median_idx]                # [B, H, N, 3]
        lower_incs   = F.softplus(lower_deltas)                        # all > 0

        # Reverse-prefix-sum: lower_cumsum[..., k] = sum of incs from k to end
        # so that Q[0.025] = median − (inc₀+inc₁+inc₂) and Q[0.25] = median − inc₂
        lower_cumsum = torch.flip(
            torch.cumsum(torch.flip(lower_incs, dims=[-1]), dim=-1),
            dims=[-1],
        )                                                               # [B, H, N, 3]
        lower_qs = median - lower_cumsum                               # [B, H, N, 3]

        # ── Upper quantiles (Q[0.75], Q[0.90], Q[0.975]) ────────────────────
        upper_deltas = reshaped[..., self.median_idx + 1 :]            # [B, H, N, 3]
        upper_incs   = F.softplus(upper_deltas)                        # all > 0
        upper_qs = median + torch.cumsum(upper_incs, dim=-1)           # [B, H, N, 3]

        return torch.cat([lower_qs, median, upper_qs], dim=-1)         # [B, H, N, Q]

    # ------------------------------------------------------------------
    # __call__ — receives y_hat already shaped [B, H, N, Q]
    # ------------------------------------------------------------------

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample=None,
        mask=None,
    ) -> torch.Tensor:
        """Pinball loss + PINN growth-rate penalty + adaptive underdispersion penalty.

        Parameters
        ----------
        y          : ``[B, H, N]``    ground-truth outsample values
        y_hat      : ``[B, H, N, Q]`` predicted quantiles (post domain_map)
        y_insample : ``[B, T, N]``    historical target values (used for dynamic PI width)
        """
        if not self._y_insample_logged:
            if y_insample is not None:
                logger.info(
                    "y_insample RECEIVED — Phase 4 dynamic features LIVE "
                    "(shape={}, dtype={}).",
                    tuple(y_insample.shape),
                    y_insample.dtype,
                )
            else:
                logger.warning(
                    "y_insample ABSENT — Phase 4 dynamic features inactive; "
                    "_dynamic_min_width and _growth_penalty using static fallbacks "
                    "(min_pi_width={}, max_step_change={}).",
                    self.min_pi_width,
                    self.max_step_change,
                )
            self._y_insample_logged = True

        pinball = super().__call__(
            y=y, y_hat=y_hat, y_insample=y_insample, mask=mask
        )
        # K-ratio scales the penalty proportionally to training loss magnitude, keeping
        # it relevant early in training.  The clamp ensures it never fades to zero at
        # convergence — without the floor, at train_loss≈0.02 the penalty drops to ~0.01,
        # too weak to maintain PI width against the pinball gradient.
        effective_lambda = torch.clamp(
            self.underdispersion_k * pinball.detach(),
            min=self.underdispersion_lambda,
        )
        dyn_min_width = self._dynamic_min_width(y_insample, y_hat)
        pinn = self._growth_penalty(y_hat, y_insample=y_insample)
        underdispersion = self._underdispersion_penalty(y_hat, effective_lambda, dyn_min_width)
        return pinball + pinn + underdispersion

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _growth_penalty(
        self,
        y_hat: torch.Tensor,
        y_insample=None,
    ) -> torch.Tensor:
        """Soft quadratic penalty on absolute median step-changes above a dynamic cap.

        The cap adapts to per-sample insample volatility, mirroring _dynamic_min_width:

            dyn_cap_t = STEP_CHANGE_MULTIPLIER × σ(y_insample[-4:])
                        clamped to min=max_step_change  (static floor / fallback)

        Behaviour by regime:
          - calm inter-wave (σ ≈ 0.3): cap ≈ 3×0.3 = 0.9 → clamped to 1.5 (tight)
          - surge onset     (σ ≈ 0.8): cap ≈ 3×0.8 = 2.4 → wider, tracks legitimate moves
          - peak            (σ ≈ 1.5): cap ≈ 3×1.5 = 4.5 → wide open at peak volatility

        Falls back to ``max_step_change`` (static) when ``y_insample`` is None.
        Returns 0 when growth_lambda == 0.

        Parameters
        ----------
        y_hat      : ``[B, H, N, Q]`` — monotone quantile predictions (post domain_map)
        y_insample : ``[B, T, N]`` or ``[B, T]`` — historical target values (optional)
        """
        if self.growth_lambda == 0.0:
            return torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype).squeeze()

        # --- Dynamic cap: step_change_multiplier × local volatility ---------------
        if y_insample is not None and self.step_change_multiplier > 0:
            ys = y_insample.squeeze(-1) if y_insample.dim() == 3 else y_insample  # [B, T]
            n_steps = min(4, ys.shape[1])
            vol = ys[:, -n_steps:].float().std(dim=1, keepdim=True)   # [B, 1]
            vol = torch.nan_to_num(vol, nan=0.0)
            dyn_cap = (self.step_change_multiplier * vol).clamp(min=self.max_step_change)
            dyn_cap = dyn_cap.unsqueeze(-1)    # [B, 1, 1] — broadcasts over H-1 and N
        else:
            dyn_cap = self.max_step_change     # scalar fallback

        median = y_hat[:, :, :, self.median_idx]               # (B, H, N)
        step_change = median[:, 1:, :] - median[:, :-1, :]     # (B, H-1, N) absolute Δ

        # Asymmetric gate — upward steps only; decays to zero at outbreak scale.
        #
        # Symmetric hard ceiling (old): penalises step_change > dyn_cap equally in
        # both directions.  This suppressed legitimate surges alongside hallucinations.
        #
        # Asymmetric sigmoid gate (new):
        #   gate ≈ 1  when upward << dyn_cap  (noise zone: small compounding steps)
        #   gate ≈ 0  when upward >> dyn_cap  (outbreak zone: genuine surge, unconstrained)
        # Result: the model is incentivised to either stay flat OR commit to a
        # genuine outbreak-scale jump — ambiguous noise-level growth is penalised.
        upward = step_change.clamp(min=0.0)
        gate   = torch.sigmoid(4.0 * (dyn_cap - upward) / (dyn_cap + 1e-6))
        return self.growth_lambda * (upward * gate).pow(2).mean()

    def _dynamic_min_width(
        self,
        y_insample,
        y_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample dynamic minimum PI width using two volatility sources.

        1. ``case_vol`` — std of last 4 insample case steps (lagging, confirmed).

        2. ``forecast_vol`` — std of predicted median across H forecast steps (leading).
           When ww_momentum_lead or vel_concentration has driven the model to predict
           a volatile trajectory, this rises even if recent cases are quiet — proxying
           WW-signal-driven uncertainty without requiring WW data in the loss function.
           Computed with .detach() so no gradients flow back through this path.

        Floor = max(case_vol, forecast_vol) × multiplier, clamped to min_pi_width.
        This couples PI width to BOTH lagging case history AND leading forecast
        trajectory — the structural gap where PI stayed narrow during WW-driven
        surge onset (cases quiet, WW rising) is now addressed.
        """
        device = y_hat.device
        dtype  = y_hat.dtype

        if y_insample is None or self.min_pi_width_multiplier <= 0:
            return torch.tensor(self.min_pi_width, device=device, dtype=dtype)

        # ── 1. Case history vol (lagging) ─────────────────────────────────────
        ys = y_insample.squeeze(-1) if y_insample.dim() == 3 else y_insample  # [B, T]
        n_steps  = min(4, ys.shape[1])
        case_vol = ys[:, -n_steps:].float().std(dim=1, keepdim=True)           # [B, 1]
        case_vol = torch.nan_to_num(case_vol, nan=0.0)

        # ── 2. Forecast trajectory vol (WW-proxy, leading) ────────────────────
        forecast_median = y_hat[:, :, :, self.median_idx]      # [B, H, N]
        forecast_median = forecast_median.squeeze(-1)           # [B, H]
        forecast_vol    = forecast_median.std(dim=1, keepdim=True).detach()    # [B, 1]
        forecast_vol    = torch.nan_to_num(forecast_vol, nan=0.0)

        effective_vol = torch.maximum(case_vol, forecast_vol)
        dyn = (self.min_pi_width_multiplier * effective_vol).clamp(min=self.min_pi_width)
        return dyn.unsqueeze(-1)                               # [B, 1, 1] for H×N broadcast

    def _underdispersion_penalty(
        self,
        y_hat: torch.Tensor,
        effective_lambda: torch.Tensor,
        min_width,
    ) -> torch.Tensor:
        """Quadratic penalty when the predicted 95% PI is narrower than min_width.

        Phase 4: ``effective_lambda`` is proportional to the current pinball loss
        magnitude so the penalty stays scaled throughout training.  ``min_width``
        is per-sample (``[B, 1, 1]``) derived from insample volatility.

        Parameters
        ----------
        y_hat           : ``[B, H, N, Q]`` — quantile predictions
        effective_lambda: Scalar or tensor — adaptive penalty weight
        min_width       : Scalar or ``[B, 1, 1]`` — dynamic minimum PI width

        Returns
        -------
        Scalar penalty term (zero when outer PI quantiles are unavailable).
        """
        if self._lower_idx is None or self._upper_idx is None:
            return torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype).squeeze()
        q_lower = y_hat[:, :, :, self._lower_idx]             # (B, H, N)
        q_upper = y_hat[:, :, :, self._upper_idx]             # (B, H, N)
        pi_width = q_upper - q_lower                          # (B, H, N)
        shortfall = torch.clamp(min_width - pi_width, min=0.0)
        return effective_lambda * (shortfall ** 2).mean()
