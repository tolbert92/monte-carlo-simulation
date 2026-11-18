
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.9: Portfolio Evolution Engine

Adds intraperiod depletion tracking while preserving the
Stage 2.5 portfolio accounting interface.

Accounting conventions
----------------------
1. Holdings are measured in nominal portfolio currency.
2. Initial asset weights sum to one.
3. Returns are arithmetic simple returns.
4. Contributions and withdrawals are nonnegative.
5. Cash flows occur at the beginning or end of a period.
6. Withdrawals cannot exceed available wealth.
7. Contributions follow current asset proportions, or
   initial target weights when holdings are zero.
8. Withdrawals reduce holdings proportionally.
9. Periodic fees apply after investment returns.
10. No automatic rebalancing is performed.
11. A contribution can restore a depleted portfolio.
12. Arithmetic returns below -100% are rejected.
13. All input and output amounts must be finite.

Depletion conventions
---------------------
depletion_events:
    True when closing portfolio wealth equals zero.

intraperiod_depletion_events:
    True when wealth reaches zero during the accounting
    sequence before end-of-period contributions can restore
    the portfolio.

ever_depleted:
    True if either event occurs in any period.

For beginning-of-period cash flows, depletion is checked
after withdrawals and after investment returns/fees.

For end-of-period cash flows, depletion is checked after
returns/fees and again at the closing balance.

This module does not model continuous-time asset paths.
Intraperiod refers to observable accounting checkpoints.

Input return shape:
    (paths, periods, assets)

Output balance shape:
    (paths, periods + 1, assets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACCOUNTING_TOLERANCE = 1e-8


class PortfolioEngineError(ValueError):
    """Stable error-code exception for portfolio evolution."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise PortfolioEngineError(code, message)


@dataclass(frozen=True)
class PortfolioEvolutionConfig:
    """Configuration for a portfolio simulation run."""

    asset_ids: tuple[str, ...]
    initial_portfolio_value: float
    initial_weights: tuple[float, ...]
    cash_flow_timing: str = "end_of_period"
    periodic_fee_rate: float = 0.0
    prevent_negative_balances: bool = True


@dataclass(frozen=True)
class PortfolioEvolutionResult:
    """Balances, cash flows, and accounting observations."""

    asset_ids: tuple[str, ...]
    asset_balances: np.ndarray
    portfolio_balances: np.ndarray
    requested_contributions: np.ndarray
    requested_withdrawals: np.ndarray
    actual_contributions: np.ndarray
    actual_withdrawals: np.ndarray
    unfulfilled_withdrawals: np.ndarray
    investment_gains: np.ndarray
    fees_paid: np.ndarray
    depletion_events: np.ndarray
    ever_depleted: np.ndarray
    accounting_residuals: np.ndarray

    # New Stage 2.9 field, placed last for compatibility.
    intraperiod_depletion_events: np.ndarray | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible descriptive metadata."""

        paths, observations, assets = self.asset_balances.shape

        return {
            "asset_ids": list(self.asset_ids),
            "number_of_paths": paths,
            "number_of_periods": observations - 1,
            "number_of_assets": assets,
            "balance_shape": list(self.asset_balances.shape),
            "cash_flow_policy": "withdrawals_capped_at_available",
            "rebalancing": "none",
            "accounting_basis": "nominal",
            "depletion_policy": (
                "closing_and_intraperiod_accounting_checkpoints"
            ),
            "intraperiod_depletion_tracking": (
                self.intraperiod_depletion_events is not None
            ),
        }


def _numeric_array(
    values: Any,
    name: str,
) -> np.ndarray:
    """Convert input to finite float64 values."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail(
            "INVALID_NUMERIC_INPUT",
            f"{name} must contain numeric values.",
        )

    if not np.all(np.isfinite(array)):
        _fail(
            "NONFINITE_INPUT",
            f"{name} contains NaN or infinity.",
        )

    return array


def validate_portfolio_config(
    config: PortfolioEvolutionConfig,
) -> None:
    """Validate configuration before simulation."""

    if not isinstance(config, PortfolioEvolutionConfig):
        _fail(
            "INVALID_CONFIGURATION",
            "Expected PortfolioEvolutionConfig.",
        )

    asset_count = len(config.asset_ids)

    if asset_count == 0:
        _fail(
            "EMPTY_PORTFOLIO",
            "At least one asset is required.",
        )

    if (
        any(
            not isinstance(asset_id, str) or not asset_id
            for asset_id in config.asset_ids
        )
        or len(set(config.asset_ids)) != asset_count
    ):
        _fail(
            "INVALID_ASSET_IDS",
            "Asset IDs must be unique nonempty strings.",
        )

    weights = _numeric_array(
        config.initial_weights,
        "Initial weights",
    )

    if weights.shape != (asset_count,):
        _fail(
            "INVALID_WEIGHT_DIMENSIONS",
            "Initial weights must match the asset count.",
        )

    if np.any(weights < 0) or np.any(weights > 1):
        _fail(
            "INVALID_PORTFOLIO_WEIGHTS",
            "Initial weights must be between zero and one.",
        )

    if not np.isclose(
        np.sum(weights),
        1.0,
        atol=ACCOUNTING_TOLERANCE,
        rtol=0,
    ):
        _fail(
            "INVALID_PORTFOLIO_WEIGHTS",
            "Initial portfolio weights must sum to one.",
        )

    initial_value = _numeric_array(
        config.initial_portfolio_value,
        "Initial portfolio value",
    )

    if initial_value.ndim != 0 or initial_value < 0:
        _fail(
            "INVALID_INITIAL_VALUE",
            "Initial portfolio value must be nonnegative.",
        )

    if config.cash_flow_timing not in (
        "beginning_of_period",
        "end_of_period",
    ):
        _fail(
            "INVALID_CASH_FLOW_TIMING",
            "Cash flow timing must be beginning_of_period "
            "or end_of_period.",
        )

    fee = _numeric_array(
        config.periodic_fee_rate,
        "Periodic fee rate",
    )

    if fee.ndim != 0 or not 0 <= fee <= 1:
        _fail(
            "INVALID_FEE_RATE",
            "Periodic fee rate must be between zero and one.",
        )

    if config.prevent_negative_balances is not True:
        _fail(
            "UNSUPPORTED_BALANCE_POLICY",
            "This stage requires nonnegative balances.",
        )


def _cash_flow_array(
    values: Any,
    paths: int,
    periods: int,
    name: str,
) -> np.ndarray:
    """Normalize a cash-flow schedule to (paths, periods)."""

    if values is None:
        return np.zeros((paths, periods), dtype=np.float64)

    array = _numeric_array(values, name)

    if array.ndim == 1 and array.shape == (periods,):
        array = np.broadcast_to(
            array,
            (paths, periods),
        ).copy()

    elif array.ndim == 2 and array.shape == (paths, periods):
        array = array.copy()

    else:
        _fail(
            "INVALID_CASH_FLOW_DIMENSIONS",
            f"{name} must have shape (periods,) "
            "or (paths, periods).",
        )

    if np.any(array < 0):
        _fail(
            "NEGATIVE_CASH_FLOW",
            f"{name} cannot contain negative amounts.",
        )

    return array


def _allocate_contributions(
    holdings: np.ndarray,
    amounts: np.ndarray,
    initial_weights: np.ndarray,
) -> np.ndarray:
    """Allocate contributions proportionally to asset holdings."""

    wealth = holdings.sum(axis=1)

    proportions = np.empty_like(holdings)
    positive = wealth > 0

    if np.any(positive):
        proportions[positive] = (
            holdings[positive]
            / wealth[positive, None]
        )

    if np.any(~positive):
        proportions[~positive] = initial_weights

    return holdings + amounts[:, None] * proportions


def _apply_withdrawals(
    holdings: np.ndarray,
    requested: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fulfill withdrawals without creating negative holdings."""

    available = holdings.sum(axis=1)
    actual = np.minimum(requested, available)

    proportions = np.zeros_like(holdings)
    positive = available > 0

    if np.any(positive):
        proportions[positive] = (
            holdings[positive]
            / available[positive, None]
        )

    updated = holdings - actual[:, None] * proportions
    updated = np.maximum(updated, 0.0)

    return updated, actual


def evolve_portfolio(
    simple_returns: Any,
    config: PortfolioEvolutionConfig,
    contributions: Any = None,
    withdrawals: Any = None,
) -> PortfolioEvolutionResult:
    """
    Evolve holdings across Monte Carlo paths and periods.

    Returns have shape (paths, periods, assets).

    Beginning-of-period sequence:
        contribution -> withdrawal -> returns -> fee

    End-of-period sequence:
        returns -> fee -> contribution -> withdrawal

    Accounting identity:
        ending wealth
        = starting wealth
        + actual contributions
        - actual withdrawals
        + investment gains
        - fees

    Depletion observations are made at discrete accounting
    checkpoints, not at unobserved times between returns.
    """

    validate_portfolio_config(config)

    returns = _numeric_array(
        simple_returns,
        "Simple returns",
    )

    if returns.ndim != 3:
        _fail(
            "INVALID_RETURN_DIMENSIONS",
            "Returns must have shape (paths, periods, assets).",
        )

    paths, periods, assets = returns.shape

    if paths < 1 or periods < 1:
        _fail(
            "EMPTY_RETURN_ARRAY",
            "At least one path and period are required.",
        )

    if assets != len(config.asset_ids):
        _fail(
            "ASSET_DIMENSION_MISMATCH",
            "Return asset dimension must match configuration.",
        )

    if np.any(returns < -1):
        _fail(
            "RETURN_BELOW_NEGATIVE_ONE",
            "Arithmetic returns below -100% are invalid "
            "for a long-only portfolio.",
        )

    contribution_schedule = _cash_flow_array(
        contributions,
        paths,
        periods,
        "Contributions",
    )

    withdrawal_schedule = _cash_flow_array(
        withdrawals,
        paths,
        periods,
        "Withdrawals",
    )

    weights = np.asarray(
        config.initial_weights,
        dtype=np.float64,
    )

    balances = np.zeros(
        (paths, periods + 1, assets),
        dtype=np.float64,
    )

    balances[:, 0, :] = (
        config.initial_portfolio_value * weights
    )

    actual_contributions = np.zeros(
        (paths, periods),
        dtype=np.float64,
    )

    actual_withdrawals = np.zeros(
        (paths, periods),
        dtype=np.float64,
    )

    investment_gains = np.zeros(
        (paths, periods),
        dtype=np.float64,
    )

    fees_paid = np.zeros(
        (paths, periods),
        dtype=np.float64,
    )

    depletion_events = np.zeros(
        (paths, periods),
        dtype=bool,
    )

    # New: captures depletion before closing cash flows.
    intraperiod_depletion_events = np.zeros(
        (paths, periods),
        dtype=bool,
    )

    accounting_residuals = np.zeros(
        (paths, periods),
        dtype=np.float64,
    )

    beginning = (
        config.cash_flow_timing == "beginning_of_period"
    )

    for period in range(periods):
        holdings = balances[:, period, :].copy()
        starting_wealth = holdings.sum(axis=1)

        requested_contribution = (
            contribution_schedule[:, period]
        )

        requested_withdrawal = (
            withdrawal_schedule[:, period]
        )

        # Record whether wealth reaches zero at any
        # observable checkpoint before period close.
        depleted_during_period = np.zeros(
            paths,
            dtype=bool,
        )

        if beginning:
            holdings = _allocate_contributions(
                holdings,
                requested_contribution,
                weights,
            )

            actual_contributions[:, period] = (
                requested_contribution
            )

            holdings, fulfilled = _apply_withdrawals(
                holdings,
                requested_withdrawal,
            )

            actual_withdrawals[:, period] = fulfilled

            # Checkpoint: after beginning-period withdrawal.
            depleted_during_period |= (
                holdings.sum(axis=1) <= 0
            )

        pre_return_wealth = holdings.sum(axis=1)

        holdings_after_returns = (
            holdings * (1.0 + returns[:, period, :])
        )

        gains = (
            holdings_after_returns.sum(axis=1)
            - pre_return_wealth
        )

        investment_gains[:, period] = gains
        holdings = holdings_after_returns

        fee_amounts = (
            holdings.sum(axis=1)
            * config.periodic_fee_rate
        )

        holdings *= (
            1.0 - config.periodic_fee_rate
        )

        fees_paid[:, period] = fee_amounts

        # Checkpoint: after investment returns and fees.
        # This occurs before end-period contributions.
        depleted_during_period |= (
            holdings.sum(axis=1) <= 0
        )

        if not beginning:
            holdings = _allocate_contributions(
                holdings,
                requested_contribution,
                weights,
            )

            actual_contributions[:, period] = (
                requested_contribution
            )

            holdings, fulfilled = _apply_withdrawals(
                holdings,
                requested_withdrawal,
            )

            actual_withdrawals[:, period] = fulfilled

        if not np.all(np.isfinite(holdings)):
            _fail(
                "NONFINITE_PORTFOLIO_OUTPUT",
                "Portfolio evolution produced nonfinite balances.",
            )

        if np.any(holdings < -ACCOUNTING_TOLERANCE):
            _fail(
                "NEGATIVE_PORTFOLIO_BALANCE",
                "Portfolio evolution produced negative holdings.",
            )

        holdings = np.maximum(holdings, 0.0)
        ending_wealth = holdings.sum(axis=1)

        expected_ending_wealth = (
            starting_wealth
            + actual_contributions[:, period]
            - actual_withdrawals[:, period]
            + investment_gains[:, period]
            - fees_paid[:, period]
        )

        residual = (
            ending_wealth - expected_ending_wealth
        )

        accounting_residuals[:, period] = residual

        allowed_error = (
            ACCOUNTING_TOLERANCE
            * np.maximum(
                1.0,
                np.maximum(
                    np.abs(ending_wealth),
                    np.abs(expected_ending_wealth),
                ),
            )
        )

        if np.any(np.abs(residual) > allowed_error):
            _fail(
                "ACCOUNTING_RECONCILIATION_FAILURE",
                "Portfolio accounting identity did not reconcile.",
            )

        balances[:, period + 1, :] = holdings

        # Preserve original end-of-period depletion semantics.
        depletion_events[:, period] = (
            ending_wealth <= 0
        )

        # New: record exhaustion before the period closes.
        intraperiod_depletion_events[:, period] = (
            depleted_during_period
        )

    portfolio_balances = balances.sum(axis=2)

    unfulfilled_withdrawals = (
        withdrawal_schedule - actual_withdrawals
    )

    # New: include both closing and intraperiod depletion.
    ever_depleted = np.any(
        depletion_events | intraperiod_depletion_events,
        axis=1,
    )

    return PortfolioEvolutionResult(
        asset_ids=config.asset_ids,
        asset_balances=balances,
        portfolio_balances=portfolio_balances,
        requested_contributions=contribution_schedule,
        requested_withdrawals=withdrawal_schedule,
        actual_contributions=actual_contributions,
        actual_withdrawals=actual_withdrawals,
        unfulfilled_withdrawals=unfulfilled_withdrawals,
        investment_gains=investment_gains,
        fees_paid=fees_paid,
        depletion_events=depletion_events,
        ever_depleted=ever_depleted,
        accounting_residuals=accounting_residuals,
        intraperiod_depletion_events=(
            intraperiod_depletion_events
        ),
    )
