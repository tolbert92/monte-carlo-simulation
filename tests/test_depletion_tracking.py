
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.10: Intraperiod Depletion Tests

Tests Stage 2.9 portfolio depletion event tracking.

Definitions:
- depletion_events: zero wealth at period close
- intraperiod_depletion_events: zero wealth at a modeled
  accounting checkpoint before period close
- ever_depleted: either event in any simulated period

These tests cover discrete accounting checkpoints only.
They do not model continuous-time price trajectories.

Run this file:
    python -m unittest discover \
        -s tests -p "test_depletion_tracking.py" -v

Run all tests:
    python -m unittest discover \
        -s tests -p "test_*.py" -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monte_carlo.portfolio_engine import (  # noqa: E402
    PortfolioEvolutionConfig,
    evolve_portfolio,
)


def make_config(**overrides):
    """Create a valid two-asset portfolio configuration."""
    values = {
        "asset_ids": ("EQUITY", "BONDS"),
        "initial_portfolio_value": 100000.0,
        "initial_weights": (0.60, 0.40),
        "cash_flow_timing": "end_of_period",
        "periodic_fee_rate": 0.0,
        "prevent_negative_balances": True,
    }
    values.update(overrides)
    return PortfolioEvolutionConfig(**values)


def make_returns(
    paths=1,
    periods=1,
    equity_return=0.0,
    bond_return=0.0,
):
    """Build a three-dimensional two-asset return array."""
    returns = np.zeros(
        (paths, periods, 2),
        dtype=np.float64,
    )
    returns[:, :, 0] = equity_return
    returns[:, :, 1] = bond_return
    return returns


class TestDepletionTracking(unittest.TestCase):
    """Sixteen tests for discrete depletion event tracking."""

    def assert_events(
        self,
        result,
        closing,
        intraperiod,
        ever,
    ):
        """Compare all three depletion outputs exactly."""
        np.testing.assert_array_equal(
            result.depletion_events,
            np.asarray(closing, dtype=bool),
        )
        np.testing.assert_array_equal(
            result.intraperiod_depletion_events,
            np.asarray(intraperiod, dtype=bool),
        )
        np.testing.assert_array_equal(
            result.ever_depleted,
            np.asarray(ever, dtype=bool),
        )

    def assert_reconciled(self, result):
        """Verify period-by-period accounting conservation."""
        beginning = result.portfolio_balances[:, :-1]
        ending = result.portfolio_balances[:, 1:]

        expected = (
            beginning
            + result.actual_contributions
            - result.actual_withdrawals
            + result.investment_gains
            - result.fees_paid
        )

        np.testing.assert_allclose(
            ending,
            expected,
            rtol=1e-12,
            atol=1e-8,
        )

        np.testing.assert_allclose(
            result.accounting_residuals,
            0.0,
            rtol=0,
            atol=1e-8,
        )

    def test_01_positive_portfolio_has_no_depletion(self):
        result = evolve_portfolio(
            make_returns(periods=3),
            make_config(),
        )

        self.assert_events(
            result,
            [[False, False, False]],
            [[False, False, False]],
            [False],
        )

    def test_02_total_investment_loss_depletes(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=-1.0,
                bond_return=-1.0,
            ),
            make_config(),
        )

        self.assert_events(
            result,
            [[True]],
            [[True]],
            [True],
        )

        self.assertEqual(
            result.portfolio_balances[0, 1],
            0.0,
        )

    def test_03_closing_contribution_restores_loss(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=-1.0,
                bond_return=-1.0,
            ),
            make_config(),
            contributions=[10000.0],
        )

        self.assert_events(
            result,
            [[False]],
            [[True]],
            [True],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            10000.0,
        )

    def test_04_full_end_period_withdrawal(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(),
            withdrawals=[100000.0],
        )

        self.assert_events(
            result,
            [[True]],
            [[False]],
            [True],
        )

    def test_05_full_beginning_period_withdrawal(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(
                cash_flow_timing="beginning_of_period",
            ),
            withdrawals=[100000.0],
        )

        self.assert_events(
            result,
            [[True]],
            [[True]],
            [True],
        )

    def test_06_partial_withdrawal_no_depletion(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(),
            withdrawals=[25000.0],
        )

        self.assert_events(
            result,
            [[False]],
            [[False]],
            [False],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            75000.0,
        )

    def test_07_recovery_in_following_period(self):
        result = evolve_portfolio(
            make_returns(periods=2),
            make_config(),
            withdrawals=[100000.0, 0.0],
            contributions=[0.0, 15000.0],
        )

        self.assert_events(
            result,
            [[True, False]],
            [[False, True]],
            [True],
        )

        np.testing.assert_allclose(
            result.portfolio_balances[0],
            [100000.0, 0.0, 15000.0],
        )

    def test_08_recovery_does_not_erase_history(self):
        result = evolve_portfolio(
            make_returns(periods=3),
            make_config(),
            withdrawals=[100000.0, 0.0, 0.0],
            contributions=[0.0, 20000.0, 0.0],
        )

        self.assert_events(
            result,
            [[True, False, False]],
            [[False, True, False]],
            [True],
        )

    def test_09_multiple_paths_are_independent(self):
        returns = np.array([
            [[0.05, 0.05]],
            [[-1.0, -1.0]],
            [[-0.20, -0.20]],
        ])

        result = evolve_portfolio(
            returns,
            make_config(),
            contributions=[10000.0],
        )

        self.assert_events(
            result,
            [[False], [False], [False]],
            [[False], [True], [False]],
            [False, True, False],
        )

        np.testing.assert_allclose(
            result.portfolio_balances[:, 1],
            [115000.0, 10000.0, 90000.0],
        )

    def test_10_depletion_arrays_have_correct_shapes(self):
        result = evolve_portfolio(
            make_returns(paths=4, periods=6),
            make_config(),
        )

        self.assertEqual(
            result.depletion_events.shape,
            (4, 6),
        )

        self.assertEqual(
            result.intraperiod_depletion_events.shape,
            (4, 6),
        )

        self.assertEqual(
            result.ever_depleted.shape,
            (4,),
        )

        self.assertEqual(
            result.intraperiod_depletion_events.dtype,
            np.dtype(bool),
        )

    def test_11_metadata_identifies_depletion_policy(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(),
        )

        metadata = result.to_metadata()

        self.assertEqual(
            metadata["depletion_policy"],
            "closing_and_intraperiod_accounting_checkpoints",
        )

        self.assertTrue(
            metadata["intraperiod_depletion_tracking"]
        )

        self.assertEqual(
            metadata["balance_shape"],
            [1, 2, 2],
        )

    def test_12_accounting_reconciles_after_recovery(self):
        result = evolve_portfolio(
            make_returns(
                paths=2,
                periods=3,
                equity_return=-1.0,
                bond_return=-1.0,
            ),
            make_config(),
            contributions=[5000.0, 2000.0, 3000.0],
            withdrawals=[0.0, 1000.0, 500.0],
        )

        self.assert_reconciled(result)

        self.assertTrue(
            np.all(result.portfolio_balances >= 0)
        )

        self.assertTrue(
            np.all(result.ever_depleted)
        )

    def test_13_zero_initial_wealth_with_contribution(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(initial_portfolio_value=0.0),
            contributions=[10000.0],
        )

        self.assert_events(
            result,
            [[False]],
            [[True]],
            [True],
        )

        np.testing.assert_allclose(
            result.asset_balances[0, 1],
            [6000.0, 4000.0],
        )

    def test_14_beginning_contribution_avoids_depletion(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(
                initial_portfolio_value=0.0,
                cash_flow_timing="beginning_of_period",
            ),
            contributions=[10000.0],
        )

        self.assert_events(
            result,
            [[False]],
            [[False]],
            [False],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            10000.0,
        )

    def test_15_full_periodic_fee_depletes_portfolio(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(periodic_fee_rate=1.0),
        )

        self.assert_events(
            result,
            [[True]],
            [[True]],
            [True],
        )

        self.assertAlmostEqual(
            result.fees_paid[0, 0],
            100000.0,
        )

    def test_16_event_union_matches_ever_depleted(self):
        returns = make_returns(paths=3, periods=2)

        returns[0, 0, :] = -1.0
        returns[1, 1, :] = -1.0

        contributions = np.array([
            [10000.0, 0.0],
            [0.0, 5000.0],
            [0.0, 0.0],
        ])

        result = evolve_portfolio(
            returns,
            make_config(),
            contributions=contributions,
        )

        expected_union = np.any(
            result.depletion_events
            | result.intraperiod_depletion_events,
            axis=1,
        )

        np.testing.assert_array_equal(
            result.ever_depleted,
            expected_union,
        )

        np.testing.assert_array_equal(
            result.ever_depleted,
            [True, True, False],
        )

        self.assert_reconciled(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
