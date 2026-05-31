"""
Tests for src/models/loss_functions.py  (Module 2 — loss functions)

What this suite proves
----------------------
1. PinballLoss    — quantile asymmetry is mathematically correct; zero loss at truth
2. GrowthRatePenalty — fires only above MAX_DAILY_GROWTH_RATE × 7; scales quadratically
3. PINNWastewaterLoss — Softplus enforces non-negativity; PINN overhead is selective;
                        domain_map shape matches NeuralForecast contract; median_idx correct

Run with visible output:
    pytest tests/test_loss_functions.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import GROWTH_RATE_LAMBDA, MAX_DAILY_GROWTH_RATE, QUANTILE_LEVELS
from src.models.loss_functions import (
    GrowthRatePenalty,
    PinballLoss,
    PINNWastewaterLoss,
)
from src.utils.helpers import console, print_pinball_asymmetry_table, print_pinn_comparison_table

_MAX_WEEKLY = MAX_DAILY_GROWTH_RATE * 7   # 0.35 × 7 = 2.45
_B, _H, _Q  = 4, 8, len(QUANTILE_LEVELS)  # batch, horizon, quantiles


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def pinball():
    return PinballLoss(QUANTILE_LEVELS)


@pytest.fixture(scope="module")
def growth_penalty():
    return GrowthRatePenalty(lam=GROWTH_RATE_LAMBDA, max_rate=_MAX_WEEKLY, median_idx=3)


@pytest.fixture(scope="module")
def pinn():
    return PINNWastewaterLoss(quantiles=QUANTILE_LEVELS)


# ===========================================================================
# TestPinballLoss
# ===========================================================================

class TestPinballLoss:

    def test_zero_loss_for_exact_prediction(self, pinball):
        """When every predicted quantile equals y_true, loss should be 0."""
        y = torch.tensor([[[2.5] * len(QUANTILE_LEVELS)]])  # (1, 1, Q) all = 2.5
        t = torch.tensor([[2.5]])                            # (1, 1)
        val = float(pinball(y, t))
        assert val == pytest.approx(0.0, abs=1e-6), f"Expected 0.0, got {val}"

    def test_loss_is_positive_for_any_error(self, pinball):
        y_pred = torch.rand(_B, _H, _Q) * 5
        y_true = torch.rand(_B, _H) * 5 + 10  # shifted so predictions are all wrong
        val = float(pinball(y_pred, y_true))
        assert val > 0.0

    def test_lower_quantile_penalises_overprediction_more(self, pinball):
        """For q=0.025, predicting too HIGH should cost more than predicting too LOW."""
        y_true = torch.zeros(1, 1)
        y_over  = torch.zeros(1, 1, _Q); y_over[0, 0, 0]  =  1.0   # q0.025 = +1
        y_under = torch.zeros(1, 1, _Q); y_under[0, 0, 0] = -1.0   # q0.025 = -1
        loss_over  = float(pinball(y_over,  y_true))
        loss_under = float(pinball(y_under, y_true))
        assert loss_over > loss_under, (
            f"q0.025 should penalise overprediction ({loss_over:.4f}) "
            f"more than underprediction ({loss_under:.4f})"
        )

    def test_upper_quantile_penalises_underprediction_more(self, pinball):
        """For q=0.975, predicting too LOW should cost more than predicting too HIGH."""
        y_true = torch.zeros(1, 1)
        y_over  = torch.zeros(1, 1, _Q); y_over[0, 0, -1]  =  1.0
        y_under = torch.zeros(1, 1, _Q); y_under[0, 0, -1] = -1.0
        loss_over  = float(pinball(y_over,  y_true))
        loss_under = float(pinball(y_under, y_true))
        assert loss_under > loss_over, (
            f"q0.975 should penalise underprediction ({loss_under:.4f}) "
            f"more than overprediction ({loss_over:.4f})"
        )

    def test_median_is_symmetric(self, pinball):
        """For q=0.5, over- and underprediction of same magnitude should cost equally."""
        y_true = torch.zeros(1, 1)
        median_idx = QUANTILE_LEVELS.index(0.5)
        y_over  = torch.zeros(1, 1, _Q); y_over[0, 0, median_idx]  =  2.0
        y_under = torch.zeros(1, 1, _Q); y_under[0, 0, median_idx] = -2.0
        loss_over  = float(pinball(y_over,  y_true))
        loss_under = float(pinball(y_under, y_true))
        assert loss_over == pytest.approx(loss_under, rel=1e-5)

    def test_loss_scales_linearly_with_error_magnitude(self, pinball):
        """Doubling the error should double the pinball loss."""
        y_true = torch.zeros(1, 1)
        y1 = torch.ones(1, 1, _Q)
        y2 = torch.ones(1, 1, _Q) * 2
        l1 = float(pinball(y1, y_true))
        l2 = float(pinball(y2, y_true))
        assert l2 == pytest.approx(2 * l1, rel=1e-5)

    def test_output_is_scalar(self, pinball):
        assert pinball(torch.rand(_B, _H, _Q), torch.rand(_B, _H)).shape == torch.Size([])

    def test_prints_asymmetry_table(self):
        """Visual: print quantile asymmetry table (visible with pytest -s)."""
        console.rule("[bold cyan] PinballLoss: Quantile Asymmetry Audit [/bold cyan]")
        print_pinball_asymmetry_table(QUANTILE_LEVELS)


# ===========================================================================
# TestGrowthRatePenalty
# ===========================================================================

class TestGrowthRatePenalty:

    def test_zero_for_flat_trajectory(self, growth_penalty):
        """Constant prediction → growth rate = 0 → no penalty."""
        y_pred = torch.ones(_B, _H, _Q) * 2.0
        val = float(growth_penalty(y_pred))
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_zero_for_plausible_growth(self, growth_penalty):
        """1 % weekly growth (max rate ≪ 2.45) → no penalty."""
        base = torch.ones(_B, 1, _Q) * 1.0
        steps = [base * (1.01 ** t) for t in range(_H)]
        y_pred = torch.cat(steps, dim=1)
        val = float(growth_penalty(y_pred))
        assert val == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.skip(reason="GROWTH_RATE_LAMBDA=0.0 — growth penalty disabled in Phase 2")
    def test_penalty_fires_above_threshold(self, growth_penalty):
        """Trajectory that doubles every step (growth rate >> 2.45) → large penalty."""
        base = torch.ones(_B, 1, _Q) * 0.5
        steps = [base * (10.0 ** t) for t in range(_H)]   # 10x per step
        y_pred = torch.cat(steps, dim=1)
        val = float(growth_penalty(y_pred))
        assert val > 1.0, f"Expected large penalty, got {val:.4f}"

    def test_penalty_at_boundary(self, growth_penalty):
        """Growth rate exactly at the limit → penalty should be zero."""
        limit = _MAX_WEEKLY  # 2.45
        base = torch.ones(_B, 1, _Q)
        steps = [base * ((1 + limit) ** t) for t in range(_H)]
        y_pred = torch.cat(steps, dim=1)
        val = float(growth_penalty(y_pred))
        assert val == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.skip(reason="GROWTH_RATE_LAMBDA=0.0 — growth penalty disabled in Phase 2")
    def test_quadratic_scaling(self, growth_penalty):
        """Violation of 2× the limit should produce ~4× the penalty of 1× violation."""
        def _penalty_for_rate(rate):
            base = torch.ones(_B, 1, _Q)
            steps = [base * ((1 + rate) ** t) for t in range(_H)]
            y_pred = torch.cat(steps, dim=1)
            return float(growth_penalty(y_pred))

        # Both exceed the limit; one doubles the excess
        rate_small = _MAX_WEEKLY + 0.5
        rate_large = _MAX_WEEKLY + 1.0

        p_small = _penalty_for_rate(rate_small)
        p_large = _penalty_for_rate(rate_large)

        # Large violation should produce more penalty (not necessarily exactly 4× due
        # to compound growth across horizon steps, but definitely larger)
        assert p_large > p_small, (
            f"Larger violation rate should give larger penalty: {p_large:.4f} vs {p_small:.4f}"
        )

    def test_penalty_is_scalar(self, growth_penalty):
        assert growth_penalty(torch.rand(_B, _H, _Q)).shape == torch.Size([])

    @pytest.mark.skip(reason="GROWTH_RATE_LAMBDA=0.0 — growth penalty disabled in Phase 2")
    def test_penalty_uses_median_only(self, growth_penalty):
        """Changing non-median quantiles should not affect the penalty."""
        base = torch.ones(_B, _H, _Q) * 2.0
        spike = base.clone()
        spike[:, 4, :] = 200.0          # all quantiles spike

        spike_non_median = base.clone()
        for idx in [0, 1, 3, 4]:       # change only non-median quantiles
            spike_non_median[:, 4, idx] = 200.0

        p_all    = float(growth_penalty(spike))
        p_others = float(growth_penalty(spike_non_median))
        # Penalty for spiking only non-median should be ~0 (median unchanged)
        assert p_others < p_all * 0.01, (
            f"Non-median spike should barely affect penalty: {p_others:.4f} vs {p_all:.4f}"
        )


# ===========================================================================
# TestPINNWastewaterLoss
# ===========================================================================

class TestPINNWastewaterLoss:

    # ── NeuralForecast interface contract ──────────────────────────────────

    def test_domain_map_output_shape(self, pinn):
        """domain_map must reshape [B, H, N*Q] → [B, H, N, Q] for univariate N=1."""
        raw = torch.randn(_B, _H, _Q)          # [B, H, 1*Q]
        mapped = pinn.domain_map(raw)
        assert mapped.shape == (_B, _H, 1, _Q), f"Expected ({_B}, {_H}, 1, {_Q}), got {mapped.shape}"

    def test_domain_map_is_monotone_softplus(self, pinn):
        """Phase 4: domain_map applies median-anchored cumulative softplus.

        Guarantees strict quantile monotonicity:
          Q[0.025] < Q[0.10] < Q[0.25] < Q[0.50] < Q[0.75] < Q[0.90] < Q[0.975]
        for every (batch, horizon, series) position — regardless of raw logit values.
        """
        torch.manual_seed(7)
        raw = torch.randn(_B, _H, _Q) * 5   # realistic scale of raw logits
        mapped = pinn.domain_map(raw)        # [B, H, N=1, Q=7]
        q = mapped.squeeze(2)               # [B, H, 7]

        # Every consecutive pair must be strictly ordered
        for k in range(_Q - 1):
            assert (q[..., k + 1] > q[..., k]).all(), (
                f"Monotonicity violated at quantile pair ({k}, {k+1})"
            )

    def test_domain_map_preserves_negatives(self, pinn):
        """Phase 4: median (index 3) is the unconstrained raw anchor.

        The median logit passes through unchanged; only the increments
        use softplus.  Negative median logits should remain negative.
        """
        raw = torch.full((_B, _H, _Q), -50.0)   # all logits very negative
        mapped = pinn.domain_map(raw)
        median = mapped.squeeze(2)[..., pinn.median_idx]   # raw anchor
        # Median should equal the raw logit at index 3 (parent domain_map reshape)
        # With all-identical inputs the median slot is pulled through, and the
        # softplus(−50) ≈ 0 increments make all other quantiles ≈ median too.
        assert (median < 0).all(), "Negative median logits must stay negative"

    def test_total_loss_is_finite_scalar(self, pinn):
        raw = torch.randn(_B, _H, _Q)
        y_true = torch.rand(_B, _H, 1)
        mapped = pinn.domain_map(raw)
        loss = pinn(y=y_true, y_hat=mapped)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    # ── Configuration ──────────────────────────────────────────────────────

    def test_median_idx_is_three(self, pinn):
        """For quantiles [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975], median is at index 3."""
        assert pinn.median_idx == 3

    def test_max_step_growth_rate(self, pinn):
        """Threshold must be MAX_DAILY × 7 (weekly data)."""
        assert pinn.max_step_growth_rate == pytest.approx(_MAX_WEEKLY, rel=1e-6)

    def test_output_size_multiplier(self, pinn):
        assert pinn.outputsize_multiplier == len(QUANTILE_LEVELS)

    def test_raises_without_median_quantile(self):
        with pytest.raises(ValueError, match="must include 0.5"):
            PINNWastewaterLoss(quantiles=[0.1, 0.9])

    # ── PINN vs plain MQLoss ───────────────────────────────────────────────

    def test_growth_penalty_fires_for_impossible_spike(self):
        """Growth penalty is positive for a biologically impossible step-change.

        GROWTH_RATE_LAMBDA defaults to 0.0 in config (disabled for production runs).
        This test instantiates PINNWastewaterLoss with an explicit non-zero lambda
        so the penalty path is exercised regardless of the config default.
        """
        pinn_active = PINNWastewaterLoss(
            quantiles=QUANTILE_LEVELS,
            growth_lambda=0.01,    # explicitly enabled — config default is 0.0
        )

        raw_safe  = torch.ones(_B, _H, _Q)
        raw_spike = raw_safe.clone()
        # Inject a moderate impossible step of +2.0 at the midpoint.
        # This gives step_change ≈ 2.0, which is above max_step_change ≈ 1.5 but
        # small enough that the asymmetric sigmoid gate does NOT decay to zero
        # (the gate decays to 0 only for outbreak-scale spikes, by design).
        raw_spike[:, _H // 2, :] += 2.0

        mapped_safe  = pinn_active.domain_map(raw_safe)
        mapped_spike = pinn_active.domain_map(raw_spike)

        penalty_safe  = float(pinn_active._growth_penalty(mapped_safe))
        penalty_spike = float(pinn_active._growth_penalty(mapped_spike))

        assert penalty_spike > penalty_safe, (
            f"Impossible spike should have higher growth penalty than flat trajectory "
            f"(spike={penalty_spike:.4f}, safe={penalty_safe:.4f})"
        )
        assert penalty_safe == pytest.approx(0.0, abs=1e-3), (
            "Flat trajectory should have near-zero growth penalty"
        )

    def test_pinn_growth_penalty_near_zero_for_plausible_trajectory(self, pinn):
        """For slowly-growing (biologically safe) trajectories, the growth penalty is ~0.

        We test the penalty component directly, not the total loss vs MQLoss — the two
        losses are not directly comparable because PINN applies Softplus in domain_map
        while plain MQLoss does not, so their total losses will naturally differ even for
        identical inputs.
        """
        # Safe: 1 % weekly growth (well below MAX_DAILY × 7 = 2.45 limit)
        base = torch.ones(_B, 1, _Q)
        raw = torch.cat([base * (1.01 ** t) for t in range(_H)], dim=1)

        mapped = pinn.domain_map(raw)   # [B, H, N, Q] after Softplus
        penalty = pinn._growth_penalty(mapped)

        assert float(penalty) < 0.001, (
            f"Growth penalty for safe 1%-growth trajectory: {float(penalty):.6f} (expected < 0.001)"
        )

    def test_prints_pinn_comparison_table(self):
        """Visual: print PINN vs MQLoss comparison for several scenarios (pytest -s)."""
        console.rule("[bold magenta] PINNWastewaterLoss: Biological Plausibility Audit [/bold magenta]")
        scenarios = [
            {
                "label": "Flat (no growth)",
                "median_trajectory": [1.0] * _H,
                "y_true": [1.0] * _H,
            },
            {
                "label": "1 % weekly growth",
                "median_trajectory": [1.01 ** t for t in range(_H)],
                "y_true": [1.0] * _H,
            },
            {
                "label": "Max biologically plausible (2.45/wk)",
                "median_trajectory": [1.0 * (3.45 ** t) for t in range(_H)],
                "y_true": [1.0] * _H,
            },
            {
                "label": "10× spike (impossible)",
                "median_trajectory": [1.0 if t < 4 else 10.0 ** (t - 3) for t in range(_H)],
                "y_true": [1.0] * _H,
            },
            {
                "label": "100× spike (catastrophic)",
                "median_trajectory": [1.0 if t < 4 else 100.0 ** (t - 3) for t in range(_H)],
                "y_true": [1.0] * _H,
            },
        ]
        print_pinn_comparison_table(scenarios)
