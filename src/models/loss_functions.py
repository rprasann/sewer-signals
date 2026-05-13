"""
Custom loss functions for wastewater probabilistic forecasting.

Architecture
------------
PINNWastewaterLoss (MQLoss subclass)
  ├── domain_map()   — identity reshape only → raw logits → [B,H,N,Q]
  ├── __call__()     — Pinball (MQLoss) + PINN growth-rate penalty
  └── _growth_penalty() — weak regulariser on median step-rate violations

Design notes (calibration pivot)
---------------------------------
Softplus was removed from domain_map.  The target is log1p_new_cases scaled
by the processor's RobustScaler; values are legitimately negative (below-median
weeks).  Forcing non-negativity via Softplus collapsed all quantile outputs
toward the Softplus floor → near-zero quantile spread → 0% PI coverage.
Non-negativity is enforced post-hoc in _build_decoded_forecast via expm1+clip.

GROWTH_RATE_LAMBDA is reduced from 0.05 → 0.005.  The original lambda was
calibrated for copies/g WW concentrations; for RobustScaled case counts it
acted as a strong smoothing prior that suppressed outbreak spikes.  The reduced
lambda keeps the biological-plausibility spirit and preserves the penalty signal
in VSN attention weights without flattening prediction intervals.

Standalone modules (PinballLoss, GrowthRatePenalty) are kept for unit tests and
any direct PyTorch usage outside NeuralForecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralforecast.losses.pytorch import MQLoss

from src.config import (
    GROWTH_RATE_LAMBDA,
    MAX_DAILY_GROWTH_RATE,
    MIN_PI_WIDTH,
    QUANTILE_LEVELS,
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
    Performs the standard MQLoss reshape ``[B, H, N*Q] → [B, H, N, Q]`` with
    no activation.  Softplus was removed because the target (log1p_new_cases)
    is RobustScaler-standardised and can be legitimately negative; see module
    docstring for full rationale.

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
        quantiles:            Sorted quantile levels; must include 0.5.
        growth_lambda:        Weight of the PINN penalty (default 0.005).
        max_daily_growth_rate: Per-day biological ceiling; scaled by data_freq_days.
        data_freq_days:       Days per model time-step (7 for weekly data).
        horizon_weight:       Optional per-step horizon weighting passed to MQLoss.
    """

    def __init__(
        self,
        quantiles: list[float] = QUANTILE_LEVELS,
        growth_lambda: float = GROWTH_RATE_LAMBDA,
        max_daily_growth_rate: float = MAX_DAILY_GROWTH_RATE,
        data_freq_days: int = _DATA_FREQ_DAYS,
        horizon_weight=None,
        underdispersion_lambda: float = UNDERDISPERSION_LAMBDA,
        min_pi_width: float = MIN_PI_WIDTH,
    ) -> None:
        super().__init__(quantiles=quantiles, horizon_weight=horizon_weight)
        self.growth_lambda = growth_lambda
        self.max_step_growth_rate: float = max_daily_growth_rate * data_freq_days
        self.underdispersion_lambda = underdispersion_lambda
        self.min_pi_width = min_pi_width

        sorted_qs = sorted(quantiles)
        if 0.5 not in sorted_qs:
            raise ValueError("quantiles must include 0.5 for the growth-rate penalty.")
        self.median_idx: int = sorted_qs.index(0.5)

        # Outer 95% PI indices for underdispersion penalty (None if not present)
        self._lower_idx: int | None = sorted_qs.index(0.025) if 0.025 in sorted_qs else None
        self._upper_idx: int | None = sorted_qs.index(0.975) if 0.975 in sorted_qs else None

    # ------------------------------------------------------------------
    # domain_map — called by NeuralForecast *before* __call__
    # ------------------------------------------------------------------

    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Reshape raw logits to ``[B, H, N, Q]`` — no activation applied.

        Softplus removed; see module docstring for rationale.
        """
        return super().domain_map(y_hat)

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
        """Pinball loss + PINN growth-rate penalty.

        Parameters
        ----------
        y     : ``[B, H, N]``    ground-truth outsample values
        y_hat : ``[B, H, N, Q]`` predicted quantiles (post domain_map)
        """
        pinball = super().__call__(
            y=y, y_hat=y_hat, y_insample=y_insample, mask=mask
        )
        pinn = self._growth_penalty(y_hat)
        underdispersion = self._underdispersion_penalty(y_hat)
        return pinball + pinn + underdispersion

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _growth_penalty(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty on median step-rate violations.

        Parameters
        ----------
        y_hat : ``[B, H, N, Q]`` — quantile predictions (post Softplus)
        """
        median = y_hat[:, :, :, self.median_idx]            # (B, H, N)
        denom = median[:, :-1, :].abs() + 1e-6
        step_rate = (median[:, 1:, :] - median[:, :-1, :]) / denom  # (B, H-1, N)
        violation = torch.clamp(step_rate - self.max_step_growth_rate, min=0.0)
        return self.growth_lambda * (violation ** 2).mean()

    def _underdispersion_penalty(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty when the predicted 95% PI is narrower than min_pi_width.

        Encourages the model to maintain a minimum quantile spread, preventing
        the overconfident-smoother failure mode (0% PI coverage) where the
        upper quantile stays consistently below the actuals.

        Parameters
        ----------
        y_hat : ``[B, H, N, Q]`` — quantile predictions

        Returns
        -------
        Scalar penalty term (zero when outer PI quantiles are unavailable).
        """
        if self._lower_idx is None or self._upper_idx is None:
            return torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype).squeeze()
        q_lower = y_hat[:, :, :, self._lower_idx]           # (B, H, N)
        q_upper = y_hat[:, :, :, self._upper_idx]           # (B, H, N)
        pi_width = q_upper - q_lower                        # (B, H, N); expect ≥ min_pi_width
        shortfall = torch.clamp(self.min_pi_width - pi_width, min=0.0)
        return self.underdispersion_lambda * (shortfall ** 2).mean()
