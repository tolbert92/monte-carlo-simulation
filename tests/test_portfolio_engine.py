
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.6: Portfolio Evolution Engine Tests

Tests portfolio accounting and numerical behavior.

Run independently:
    python -m unittest discover \
        -s tests -p "test_portfolio_engine.py" -v

Run with existing tests:
    python -m unittest discover \
        -s tests -p "test_*.py" -v

This suite tests the current Stage 2.5 implementation.
It does not certify taxes, rebalancing, inflation,
retirement income, or financial recommendations.
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
    PortfolioEngineError,
    evolve_portfolio,
    validate_portfolio_config,
)


def make_config(**overrides):
    """Create a valid two-asset configuration."""

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
    """Create a two-asset return array."""

    result = np.zeros(
        (paths, periods, 2),
        dtype=np.float64,
    )

    result[:, :, 0] = equity_return
    result[:, :, 1] = bond_return

    return result


class TestPortfolioEvolution(unittest.TestCase):
    """Twenty-four portfolio accounting and validation tests."""

    def assert_engine_error(self, code, operation):
        """Assert that an operation raises the expected error."""

        with self.assertRaises(PortfolioEngineError) as context:
            operation()

        self.assertEqual(context.exception.code, code)

    def assert_reconciled(self, result):
        """Verify the accounting identity for every path."""

        start = result.portfolio_balances[:, :-1]
        end = result.portfolio_balances[:, 1:]

        expected = (
            start
            + result.actual_contributions
            - result.actual_withdrawals
            + result.investment_gains
            - result.fees_paid
        )

        np.testing.assert_allclose(
            end,
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

    def test_01_zero_returns_preserve_initial_balance(self):
        result = evolve_portfolio(
            make_returns(periods=5),
            make_config(),
        )

        np.testing.assert_allclose(
            result.portfolio_balances[0],
            [100000.0] * 6,
        )

        self.assert_reconciled(result)

    def test_02_positive_returns_compound(self):
        result = evolve_portfolio(
            make_returns(
                periods=3,
                equity_return=0.10,
                bond_return=0.10,
            ),
            make_config(),
        )

        expected = [
            100000.0,
            110000.0,
            121000.0,
            133100.0,
        ]

        np.testing.assert_allclose(
            result.portfolio_balances[0],
            expected,
            rtol=1e-12,
        )

    def test_03_negative_returns_reduce_wealth(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=-0.20,
                bond_return=-0.10,
            ),
            make_config(),
        )

        # Equity: 60,000 * 0.8 = 48,000
        # Bonds: 40,000 * 0.9 = 36,000
        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            84000.0,
        )

    def test_04_asset_balances_are_tracked_separately(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.10,
                bond_return=-0.05,
            ),
            make_config(),
        )

        np.testing.assert_allclose(
            result.asset_balances[0, 1],
            [66000.0, 38000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            104000.0,
        )

    def test_05_end_of_period_contribution(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.10,
                bond_return=0.10,
            ),
            make_config(),
            contributions=[6000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            116000.0,
        )

        self.assertAlmostEqual(
            result.actual_contributions[0, 0],
            6000.0,
        )

    def test_06_beginning_of_period_contribution(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.10,
                bond_return=0.10,
            ),
            make_config(
                cash_flow_timing="beginning_of_period",
            ),
            contributions=[6000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            116600.0,
        )

    def test_07_end_of_period_withdrawal(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.10,
                bond_return=0.10,
            ),
            make_config(),
            withdrawals=[20000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            90000.0,
        )

        self.assertAlmostEqual(
            result.actual_withdrawals[0, 0],
            20000.0,
        )

    def test_08_beginning_of_period_withdrawal(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.10,
                bond_return=0.10,
            ),
            make_config(
                cash_flow_timing="beginning_of_period",
            ),
            withdrawals=[20000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            88000.0,
        )

    def test_09_excess_withdrawal_is_capped(self):
        result = evolve_portfolio(
            make_returns(),
            make_config(),
            withdrawals=[150000.0],
        )

        self.assertAlmostEqual(
            result.actual_withdrawals[0, 0],
            100000.0,
        )

        self.assertAlmostEqual(
            result.unfulfilled_withdrawals[0, 0],
            50000.0,
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            0.0,
        )

    def test_10_full_withdrawal_marks_depletion(self):
        result = evolve_portfolio(
            make_returns(periods=2),
            make_config(),
            withdrawals=[100000.0, 0.0],
        )

        self.assertTrue(
            result.depletion_events[0, 0]
        )

        self.assertTrue(
            result.ever_depleted[0]
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 2],
            0.0,
        )

    def test_11_contribution_restores_depleted_portfolio(self):
        result = evolve_portfolio(
            make_returns(periods=2),
            make_config(),
            withdrawals=[100000.0, 0.0],
            contributions=[0.0, 10000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            0.0,
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 2],
            10000.0,
        )

        np.testing.assert_allclose(
            result.asset_balances[0, 2],
            [6000.0, 4000.0],
        )

        self.assertTrue(result.ever_depleted[0])

    def test_12_periodic_fees_reduce_portfolio(self):
        result = evolve_portfolio(
            make_returns(periods=2),
            make_config(periodic_fee_rate=0.01),
        )

        np.testing.assert_allclose(
            result.portfolio_balances[0],
            [100000.0, 99000.0, 98010.0],
        )

        np.testing.assert_allclose(
            result.fees_paid[0],
            [1000.0, 990.0],
        )

    def test_13_asset_weights_drift_without_rebalancing(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=0.20,
                bond_return=0.0,
            ),
            make_config(),
        )

        equity = result.asset_balances[0, 1, 0]
        total = result.portfolio_balances[0, 1]

        self.assertAlmostEqual(equity, 72000.0)
        self.assertAlmostEqual(total, 112000.0)

        self.assertAlmostEqual(
            equity / total,
            72000.0 / 112000.0,
        )

        self.assertNotAlmostEqual(
            equity / total,
            0.60,
        )

    def test_14_accounting_identity_reconciles(self):
        result = evolve_portfolio(
            make_returns(
                paths=3,
                periods=4,
                equity_return=0.08,
                bond_return=-0.02,
            ),
            make_config(periodic_fee_rate=0.002),
            contributions=[1000.0] * 4,
            withdrawals=[500.0] * 4,
        )

        self.assert_reconciled(result)

    def test_15_multiple_paths_are_independent(self):
        returns = np.array([
            [[0.10, 0.10]],
            [[-0.10, -0.10]],
            [[0.00, 0.00]],
        ])

        result = evolve_portfolio(
            returns,
            make_config(),
        )

        np.testing.assert_allclose(
            result.portfolio_balances[:, 1],
            [110000.0, 90000.0, 100000.0],
        )

    def test_16_path_specific_cash_flows(self):
        result = evolve_portfolio(
            make_returns(paths=2),
            make_config(),
            contributions=np.array([
                [1000.0],
                [5000.0],
            ]),
            withdrawals=np.array([
                [0.0],
                [2000.0],
            ]),
        )

        np.testing.assert_allclose(
            result.portfolio_balances[:, 1],
            [101000.0, 103000.0],
        )

    def test_17_full_negative_return_depletes_portfolio(self):
        result = evolve_portfolio(
            make_returns(
                equity_return=-1.0,
                bond_return=-1.0,
            ),
            make_config(),
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            0.0,
        )

        self.assertTrue(result.ever_depleted[0])

    def test_18_returns_below_negative_one_are_rejected(self):
        self.assert_engine_error(
            "RETURN_BELOW_NEGATIVE_ONE",
            lambda: evolve_portfolio(
                make_returns(equity_return=-1.01),
                make_config(),
            ),
        )

    def test_19_invalid_initial_weights_rejected(self):
        self.assert_engine_error(
            "INVALID_PORTFOLIO_WEIGHTS",
            lambda: validate_portfolio_config(
                make_config(initial_weights=(0.70, 0.40))
            ),
        )

    def test_20_invalid_cash_flow_dimensions_rejected(self):
        self.assert_engine_error(
            "INVALID_CASH_FLOW_DIMENSIONS",
            lambda: evolve_portfolio(
                make_returns(periods=3),
                make_config(),
                contributions=[1000.0, 2000.0],
            ),
        )

    def test_21_negative_cash_flow_rejected(self):
        self.assert_engine_error(
            "NEGATIVE_CASH_FLOW",
            lambda: evolve_portfolio(
                make_returns(),
                make_config(),
                contributions=[-100.0],
            ),
        )

    def test_22_nonfinite_inputs_rejected(self):
        self.assert_engine_error(
            "NONFINITE_INPUT",
            lambda: evolve_portfolio(
                make_returns(equity_return=float("nan")),
                make_config(),
            ),
        )

        self.assert_engine_error(
            "NONFINITE_INPUT",
            lambda: evolve_portfolio(
                make_returns(),
                make_config(),
                withdrawals=[float("inf")],
            ),
        )

    def test_23_invalid_fee_and_timing_rejected(self):
        self.assert_engine_error(
            "INVALID_FEE_RATE",
            lambda: validate_portfolio_config(
                make_config(periodic_fee_rate=1.1)
            ),
        )

        self.assert_engine_error(
            "INVALID_CASH_FLOW_TIMING",
            lambda: validate_portfolio_config(
                make_config(cash_flow_timing="middle")
            ),
        )

    def test_24_metadata_and_end_period_depletion_policy(self):
        """
        The current implementation records depletion using
        end-of-period wealth, not intraperiod wealth.

        A portfolio wiped out by returns and then restored
        by an end-of-period contribution is therefore not
        flagged as depleted in this version.

        This is a documented limitation to resolve before
        retirement sustainability analytics.
        """

        result = evolve_portfolio(
            make_returns(
                equity_return=-1.0,
                bond_return=-1.0,
            ),
            make_config(),
            contributions=[10000.0],
        )

        self.assertAlmostEqual(
            result.portfolio_balances[0, 1],
            10000.0,
        )

        self.assertFalse(
            result.depletion_events[0, 0]
        )

        metadata = result.to_metadata()

        self.assertEqual(
            metadata["balance_shape"],
            [1, 2, 2],
        )

        self.assertEqual(
            metadata["number_of_periods"],
            1,
        )

        self.assertEqual(
            metadata["rebalancing"],
            "none",
        )

        self.assert_reconciled(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
