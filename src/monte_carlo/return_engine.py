
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.1: Mathematical Engine Foundation

Deterministic, correlated multi-asset return generation.

Supported models:
    - multivariate_normal
    - lognormal

Supported return conventions:
    - arithmetic_simple for multivariate_normal
    - log for lognormal

The expected returns and volatilities must be expressed
at the same frequency as the simulation time step.

No portfolio balances, contributions, withdrawals,
taxes, or investment recommendations are calculated here.

This module uses NumPy for all numerical calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SUPPORTED_MODELS = frozenset({
    "multivariate_normal",
    "lognormal",
})

SUPPORTED_FREQUENCIES = frozenset({
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "annually",
})

DEFAULT_TOLERANCE = 1e-10


class ReturnEngineError(ValueError):
    """Base exception for return-engine validation errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ReturnEngineConfig:
    """Immutable validated engine configuration."""

    asset_ids: tuple[str, ...]
    expected_returns: tuple[float, ...]
    volatilities: tuple[float, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    model: str
    return_convention: str
    parameter_frequency: str
    time_step: str
    number_of_paths: int
    number_of_periods: int
    random_seed: int


@dataclass(frozen=True)
class ReturnSimulationResult:
    """Generated returns and reproducibility metadata."""

    asset_ids: tuple[str, ...]
    model: str
    return_convention: str
    number_of_paths: int
    number_of_periods: int
    random_seed: int
    simple_returns: np.ndarray
    model_returns: np.ndarray

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible simulation metadata."""
        return {
            "asset_ids": list(self.asset_ids),
            "model": self.model,
            "return_convention": self.return_convention,
            "number_of_paths": self.number_of_paths,
            "number_of_periods": self.number_of_periods,
            "random_seed": self.random_seed,
            "shape": list(self.simple_returns.shape),
        }


def _fail(code: str, message: str) -> None:
    raise ReturnEngineError(code, message)


def _finite_vector(
    values: tuple[float, ...],
    name: str,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail(
            "INVALID_NUMERIC_INPUT",
            f"{name} must contain numeric values.",
        )

    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        _fail(
            "INVALID_NUMERIC_INPUT",
            f"{name} must be a finite one-dimensional vector.",
        )

    return vector


def _finite_matrix(
    values: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail(
            "INVALID_CORRELATION_MATRIX",
            "Correlation matrix must contain numeric values.",
        )

    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        _fail(
            "INVALID_CORRELATION_MATRIX",
            "Correlation matrix must be finite and rectangular.",
        )

    return matrix


def validate_config(
    config: ReturnEngineConfig,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Validate configuration before generating any returns."""

    if not isinstance(config, ReturnEngineConfig):
        _fail(
            "INVALID_CONFIGURATION",
            "Expected ReturnEngineConfig.",
        )

    if config.model not in SUPPORTED_MODELS:
        _fail(
            "UNSUPPORTED_MODEL",
            f"Model '{config.model}' is not implemented.",
        )

    if (
        config.parameter_frequency not in SUPPORTED_FREQUENCIES
        or config.time_step not in SUPPORTED_FREQUENCIES
    ):
        _fail(
            "INVALID_FREQUENCY",
            "Unsupported parameter frequency or time step.",
        )

    if config.parameter_frequency != config.time_step:
        _fail(
            "FREQUENCY_MISMATCH",
            "Stage 2.1 requires matching parameter frequency "
            "and simulation time step. No implicit scaling "
            "is permitted.",
        )

    expected_convention = (
        "arithmetic_simple"
        if config.model == "multivariate_normal"
        else "log"
    )

    if config.return_convention != expected_convention:
        _fail(
            "RETURN_CONVENTION_MISMATCH",
            f"Model '{config.model}' requires "
            f"'{expected_convention}' returns.",
        )

    if not isinstance(config.number_of_paths, int) or isinstance(
        config.number_of_paths, bool
    ) or not 1 <= config.number_of_paths <= 1_000_000:
        _fail(
            "INVALID_PATH_COUNT",
            "Number of paths must be an integer from "
            "1 through 1,000,000.",
        )

    if not isinstance(config.number_of_periods, int) or isinstance(
        config.number_of_periods, bool
    ) or config.number_of_periods < 1:
        _fail(
            "INVALID_PERIOD_COUNT",
            "Number of periods must be a positive integer.",
        )

    if not isinstance(config.random_seed, int) or isinstance(
        config.random_seed, bool
    ) or not 0 <= config.random_seed <= 4294967295:
        _fail(
            "INVALID_RANDOM_SEED",
            "Seed must be an integer between 0 and 4294967295.",
        )

    if (
        not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        _fail(
            "INVALID_TOLERANCE",
            "Tolerance must be finite and positive.",
        )

    asset_count = len(config.asset_ids)

    if asset_count == 0:
        _fail(
            "EMPTY_PORTFOLIO",
            "At least one asset is required.",
        )

    if (
        any(not isinstance(asset_id, str) or not asset_id
            for asset_id in config.asset_ids)
        or len(set(config.asset_ids)) != asset_count
    ):
        _fail(
            "INVALID_ASSET_IDS",
            "Asset IDs must be unique nonempty strings.",
        )

    means = _finite_vector(
        config.expected_returns,
        "Expected returns",
    )
    volatilities = _finite_vector(
        config.volatilities,
        "Volatilities",
    )
    correlation = _finite_matrix(
        config.correlation_matrix
    )

    if (
        means.shape != (asset_count,)
        or volatilities.shape != (asset_count,)
    ):
        _fail(
            "ASSET_DIMENSION_MISMATCH",
            "Return and volatility vectors must match "
            "the asset count.",
        )

    if correlation.shape != (asset_count, asset_count):
        _fail(
            "INVALID_CORRELATION_DIMENSIONS",
            "Correlation matrix dimensions must match "
            "the asset count.",
        )

    if np.any(volatilities < 0):
        _fail(
            "NEGATIVE_VOLATILITY",
            "Volatility cannot be negative.",
        )

    if (
        config.return_convention == "arithmetic_simple"
        and np.any(means < -1)
    ):
        _fail(
            "INVALID_EXPECTED_RETURN",
            "Arithmetic expected returns cannot be below -1.",
        )

    if not np.allclose(
        correlation,
        correlation.T,
        atol=tolerance,
        rtol=0,
    ):
        _fail(
            "INVALID_CORRELATION_SYMMETRY",
            "Correlation matrix must be symmetric.",
        )

    if not np.allclose(
        np.diag(correlation),
        1.0,
        atol=tolerance,
        rtol=0,
    ):
        _fail(
            "INVALID_CORRELATION_DIAGONAL",
            "Correlation diagonal must equal one.",
        )

    if np.any(np.abs(correlation) > 1 + tolerance):
        _fail(
            "INVALID_CORRELATION_RANGE",
            "Correlation coefficients must be between -1 and 1.",
        )

    eigenvalues = np.linalg.eigvalsh(
        (correlation + correlation.T) / 2.0
    )

    if np.min(eigenvalues) < -tolerance:
        _fail(
            "INVALID_CORRELATION_PSD",
            "Correlation matrix is not positive semidefinite.",
        )


def build_covariance_matrix(
    config: ReturnEngineConfig,
) -> np.ndarray:
    """Build covariance matrix in the input return convention."""

    validate_config(config)

    volatility = np.asarray(
        config.volatilities,
        dtype=np.float64,
    )
    correlation = np.asarray(
        config.correlation_matrix,
        dtype=np.float64,
    )

    covariance = (
        np.outer(volatility, volatility) * correlation
    )

    return (covariance + covariance.T) / 2.0


def _psd_factor(
    covariance: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """
    Produce a factor F such that F @ F.T approximates
    the covariance matrix.

    An eigenvalue factorization supports singular PSD
    covariance matrices, including zero-volatility assets.

    Tiny negative eigenvalues caused by floating-point
    roundoff are clipped to zero. Materially negative
    eigenvalues are rejected.
    """

    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    if np.min(eigenvalues) < -tolerance:
        _fail(
            "INVALID_COVARIANCE_PSD",
            "Covariance matrix is not positive semidefinite.",
        )

    nonnegative = np.maximum(eigenvalues, 0.0)

    return eigenvectors @ np.diag(np.sqrt(nonnegative))


def simulate_returns(
    config: ReturnEngineConfig,
) -> ReturnSimulationResult:
    """
    Generate correlated asset returns.

    Output shape:
        (number_of_paths, number_of_periods, number_of_assets)

    multivariate_normal:
        model_returns = arithmetic simple returns
        simple_returns = model_returns

    lognormal:
        model_returns = normally distributed log returns
        simple_returns = exp(log returns) - 1

    A fresh RNG is initialized from the supplied seed
    for every invocation, ensuring reproducibility with
    the same NumPy environment and configuration.
    """

    validate_config(config)

    means = np.asarray(
        config.expected_returns,
        dtype=np.float64,
    )
    covariance = build_covariance_matrix(config)

    factor = _psd_factor(covariance)

    rng = np.random.default_rng(config.random_seed)

    shape = (
        config.number_of_paths,
        config.number_of_periods,
        len(config.asset_ids),
    )

    standard_normal = rng.standard_normal(
        size=shape,
        dtype=np.float64,
    )

    model_returns = (
        standard_normal @ factor.T
    ) + means

    if config.model == "lognormal":
        with np.errstate(over="ignore", invalid="ignore"):
            simple_returns = np.expm1(model_returns)
    else:
        simple_returns = model_returns.copy()

    if (
        not np.all(np.isfinite(model_returns))
        or not np.all(np.isfinite(simple_returns))
    ):
        _fail(
            "NONFINITE_SIMULATION_OUTPUT",
            "Simulation produced nonfinite returns.",
        )

    return ReturnSimulationResult(
        asset_ids=config.asset_ids,
        model=config.model,
        return_convention=config.return_convention,
        number_of_paths=config.number_of_paths,
        number_of_periods=config.number_of_periods,
        random_seed=config.random_seed,
        simple_returns=simple_returns,
        model_returns=model_returns,
    )


def config_from_request(
    request: Mapping[str, Any],
) -> ReturnEngineConfig:
    """
    Adapt a versioned Monte Carlo request to Stage 2.1.

    Asset order is taken directly from portfolio.assets.
    Expected return and volatility maps are aligned by
    asset_id, not by dictionary insertion order.

    The correlation matrix must use the portfolio asset
    ordering established by the request contract.
    """

    if request.get("schema_version") != "monte_carlo.request.v1":
        _fail(
            "INVALID_SCHEMA_VERSION",
            "Expected monte_carlo.request.v1.",
        )

    try:
        assets = request["portfolio"]["assets"]
        assumptions = request["market_assumptions"]
        simulation = request["simulation"]

        asset_ids = tuple(
            asset["asset_id"] for asset in assets
        )

        return_keys = set(
            assumptions["expected_returns"]
        )
        volatility_keys = set(
            assumptions["volatilities"]
        )

        if (
            return_keys != set(asset_ids)
            or volatility_keys != set(asset_ids)
        ):
            _fail(
                "ASSET_ASSUMPTION_MISMATCH",
                "Market assumption asset IDs must match "
                "portfolio asset IDs.",
            )

        expected_returns = tuple(
            assumptions["expected_returns"][asset_id]
            for asset_id in asset_ids
        )

        volatilities = tuple(
            assumptions["volatilities"][asset_id]
            for asset_id in asset_ids
        )

        correlation = tuple(
            tuple(row)
            for row in assumptions["correlation_matrix"]
        )

        frequency = simulation["time_step"]
        periods_per_year = {
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
            "quarterly": 4,
            "annually": 1,
        }

        if frequency not in periods_per_year:
            _fail(
                "INVALID_FREQUENCY",
                f"Unsupported time step: {frequency}",
            )

        number_of_periods = (
            simulation["horizon_years"]
            * periods_per_year[frequency]
        )

        config = ReturnEngineConfig(
            asset_ids=asset_ids,
            expected_returns=expected_returns,
            volatilities=volatilities,
            correlation_matrix=correlation,
            model=simulation["model"],
            return_convention=assumptions["return_convention"],
            parameter_frequency=assumptions[
                "parameter_frequency"
            ],
            time_step=frequency,
            number_of_paths=simulation["number_of_paths"],
            number_of_periods=number_of_periods,
            random_seed=simulation["random_seed"],
        )

    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReturnEngineError):
            raise

        _fail(
            "INVALID_REQUEST_STRUCTURE",
            f"Cannot construct engine configuration: {exc}",
        )

    validate_config(config)
    return config


def simulate_request(
    request: Mapping[str, Any],
) -> ReturnSimulationResult:
    """
    Run Stage 2.1 return generation from a request.

    The caller is responsible for JSON Schema validation
    before using this adapter.
    """

    return simulate_returns(
        config_from_request(request)
    )
