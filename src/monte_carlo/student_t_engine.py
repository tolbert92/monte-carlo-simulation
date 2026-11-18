
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.13A: Correlated Student's t Return Engine

Generates correlated multivariate Student's t arithmetic
simple returns with reproducible NumPy random streams.

Mathematical model
------------------
For target covariance Sigma and degrees of freedom nu > 2:

    Sigma = diag(volatility) @ correlation @ diag(volatility)
    L @ L.T = Sigma
    Z ~ N(0, I)
    U ~ ChiSquare(nu)

    returns = mean + sqrt((nu - 2) / U) * (L @ Z)

One chi-square draw is shared by all assets within each
path-period, producing multivariate tail dependence.

The scaling gives:
    E[returns] = mean
    Cov[returns] = Sigma

Subject to the usual theoretical assumptions.

Conventions
-----------
- Shape: (paths, periods, assets)
- Means: arithmetic expected returns per period
- Volatilities: standard deviations per period
- Correlations: Pearson correlation matrix
- Degrees of freedom: finite, strictly greater than 2
- Random seed: integer from 0 to 2**32 - 1
- Long-only compatibility: simple returns >= -1
- No clipping, resampling, or covariance repair
- No global NumPy random-state modification
- No conversion from annual to periodic assumptions

Limitations
-----------
A Student's t distribution has unbounded support. It can
generate simple returns below -100%, which are invalid for
the existing long-only portfolio evolution engine.

Such draws are rejected explicitly. No silent clipping or
conditional resampling is performed.

Finite degrees of freedom above 2 guarantee a theoretical
covariance, but the fourth moment is finite only for nu > 4.

The module models one-period shocks, not a calibrated
volatility process or continuous-time market behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CORRELATION_TOLERANCE = 1e-10
EIGENVALUE_TOLERANCE = 1e-10
MAX_SEED = 2**32 - 1


class StudentTReturnEngineError(ValueError):
    """Validation or generation error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise StudentTReturnEngineError(code, message)


@dataclass(frozen=True)
class StudentTReturnConfig:
    """
    Configuration for correlated Student's t return generation.

    All expected returns and volatilities refer to the
    simulation period, not necessarily one calendar year.
    """

    asset_ids: tuple[str, ...]
    expected_returns: tuple[float, ...]
    volatilities: tuple[float, ...]
    correlation_matrix: Any
    degrees_of_freedom: float
    number_of_paths: int
    number_of_periods: int
    random_seed: int


@dataclass(frozen=True)
class StudentTReturnResult:
    """Generated returns and documented model parameters."""

    asset_ids: tuple[str, ...]
    simple_returns: np.ndarray
    expected_returns: np.ndarray
    volatilities: np.ndarray
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    degrees_of_freedom: float
    random_seed: int
    model: str = "student_t"

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible model metadata."""

        paths, periods, assets = self.simple_returns.shape

        return {
            "model": self.model,
            "asset_ids": list(self.asset_ids),
            "number_of_paths": paths,
            "number_of_periods": periods,
            "number_of_assets": assets,
            "return_shape": list(self.simple_returns.shape),
            "degrees_of_freedom": self.degrees_of_freedom,
            "random_seed": self.random_seed,
            "return_convention": "arithmetic_simple",
            "distribution": "multivariate_student_t",
            "covariance_scaling": "variance_normalized",
            "shared_chi_square_per_period": True,
            "covariance_repair": False,
            "return_clipping": False,
            "invalid_draw_policy": "reject_entire_run",
        }


def _finite_array(values: Any, name: str) -> np.ndarray:
    """Convert input to a finite float64 NumPy array."""

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


def _positive_integer(value: Any, name: str) -> int:
    """Require a positive Python or NumPy integer."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        _fail(
            "INVALID_SIMULATION_DIMENSIONS",
            f"{name} must be a positive integer.",
        )

    return int(value)


def _validate_seed(value: Any) -> int:
    """Require a reproducible unsigned 32-bit seed."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value <= MAX_SEED
    ):
        _fail(
            "INVALID_RANDOM_SEED",
            "random_seed must be an integer between "
            f"0 and {MAX_SEED}.",
        )

    return int(value)


def _validate_correlation(
    matrix: Any,
    asset_count: int,
) -> np.ndarray:
    """Validate a correlation matrix without repairing it."""

    correlation = _finite_array(
        matrix,
        "Correlation matrix",
    )

    if correlation.shape != (asset_count, asset_count):
        _fail(
            "INVALID_CORRELATION_DIMENSIONS",
            "Correlation matrix must have shape "
            "(assets, assets).",
        )

    if not np.allclose(
        correlation,
        correlation.T,
        atol=CORRELATION_TOLERANCE,
        rtol=0,
    ):
        _fail(
            "ASYMMETRIC_CORRELATION_MATRIX",
            "Correlation matrix must be symmetric.",
        )

    if not np.allclose(
        np.diag(correlation),
        1.0,
        atol=CORRELATION_TOLERANCE,
        rtol=0,
    ):
        _fail(
            "INVALID_CORRELATION_DIAGONAL",
            "Correlation diagonal entries must equal one.",
        )

    if np.any(
        correlation < -1.0 - CORRELATION_TOLERANCE
    ) or np.any(
        correlation > 1.0 + CORRELATION_TOLERANCE
    ):
        _fail(
            "INVALID_CORRELATION_RANGE",
            "Correlation values must be within [-1, 1].",
        )

    # A symmetric matrix may contain a negative eigenvalue
    # even when all individual correlations are in range.
    eigenvalues = np.linalg.eigvalsh(correlation)

    if np.min(eigenvalues) < -EIGENVALUE_TOLERANCE:
        _fail(
            "CORRELATION_NOT_PSD",
            "Correlation matrix is not positive semidefinite.",
        )

    # Do not symmetrize or alter supplied correlations.
    return correlation.copy()


def validate_student_t_config(
    config: StudentTReturnConfig,
) -> None:
    """Validate the complete Student's t configuration."""

    if not isinstance(config, StudentTReturnConfig):
        _fail(
            "INVALID_CONFIGURATION",
            "Expected StudentTReturnConfig.",
        )

    asset_count = len(config.asset_ids)

    if asset_count < 1:
        _fail(
            "EMPTY_PORTFOLIO",
            "At least one asset is required.",
        )

    if (
        any(
            not isinstance(asset_id, str)
            or not asset_id.strip()
            for asset_id in config.asset_ids
        )
        or len(set(config.asset_ids)) != asset_count
    ):
        _fail(
            "INVALID_ASSET_IDS",
            "Asset IDs must be unique nonempty strings.",
        )

    means = _finite_array(
        config.expected_returns,
        "Expected returns",
    )

    volatilities = _finite_array(
        config.volatilities,
        "Volatilities",
    )

    if means.shape != (asset_count,):
        _fail(
            "INVALID_MEAN_DIMENSIONS",
            "Expected returns must match asset count.",
        )

    if volatilities.shape != (asset_count,):
        _fail(
            "INVALID_VOLATILITY_DIMENSIONS",
            "Volatilities must match asset count.",
        )

    if np.any(means < -1.0):
        _fail(
            "INVALID_EXPECTED_RETURN",
            "Expected arithmetic simple returns "
            "cannot be below -100%.",
        )

    if np.any(volatilities < 0):
        _fail(
            "NEGATIVE_VOLATILITY",
            "Volatilities must be nonnegative.",
        )

    degrees = _finite_array(
        config.degrees_of_freedom,
        "Degrees of freedom",
    )

    if degrees.ndim != 0 or degrees <= 2.0:
        _fail(
            "INVALID_DEGREES_OF_FREEDOM",
            "Degrees of freedom must be finite and > 2.",
        )

    _positive_integer(
        config.number_of_paths,
        "number_of_paths",
    )

    _positive_integer(
        config.number_of_periods,
        "number_of_periods",
    )

    _validate_seed(config.random_seed)

    _validate_correlation(
        config.correlation_matrix,
        asset_count,
    )


def build_target_covariance(
    config: StudentTReturnConfig,
) -> np.ndarray:
    """
    Construct the target covariance matrix.

    Covariance is based on the specified periodic standard
    deviations and correlation matrix.
    """

    validate_student_t_config(config)

    sigma = _finite_array(
        config.volatilities,
        "Volatilities",
    )

    correlation = _validate_correlation(
        config.correlation_matrix,
        len(config.asset_ids),
    )

    covariance = (
        sigma[:, None]
        * correlation
        * sigma[None, :]
    )

    if not np.all(np.isfinite(covariance)):
        _fail(
            "NONFINITE_COVARIANCE",
            "Covariance calculation produced nonfinite values.",
        )

    return covariance


def _covariance_factor(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Return L satisfying L @ L.T approximately covariance.

    Eigen-decomposition supports positive-semidefinite,
    including singular, covariance matrices.

    Tiny negative eigenvalues caused by floating-point
    roundoff are treated as zero only within tolerance.
    Materially negative eigenvalues are rejected.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    tolerance = EIGENVALUE_TOLERANCE * scale

    if np.min(eigenvalues) < -tolerance:
        _fail(
            "COVARIANCE_NOT_PSD",
            "Target covariance matrix is not PSD.",
        )

    nonnegative = np.maximum(eigenvalues, 0.0)

    factor = (
        eigenvectors * np.sqrt(nonnegative)[None, :]
    )

    if not np.all(np.isfinite(factor)):
        _fail(
            "NONFINITE_COVARIANCE_FACTOR",
            "Covariance factor contains nonfinite values.",
        )

    return factor


def generate_student_t_returns(
    config: StudentTReturnConfig,
) -> StudentTReturnResult:
    """
    Generate correlated multivariate Student's t returns.

    Output:
        simple_returns.shape == (paths, periods, assets)

    Random generation:
        - independent standard normal vectors
        - one independent chi-square draw per path-period
        - common chi-square scale across assets

    Reproducibility:
        A local NumPy Generator is initialized from the
        supplied seed. Repeated calls with the same inputs
        and NumPy version produce identical results.

    Long-only compatibility:
        If any arithmetic return is below -1, the entire
        run fails. No clipping or resampling occurs.
    """

    validate_student_t_config(config)

    paths = int(config.number_of_paths)
    periods = int(config.number_of_periods)
    assets = len(config.asset_ids)

    means = _finite_array(
        config.expected_returns,
        "Expected returns",
    )

    volatilities = _finite_array(
        config.volatilities,
        "Volatilities",
    )

    correlation = _validate_correlation(
        config.correlation_matrix,
        assets,
    )

    covariance = build_target_covariance(config)
    factor = _covariance_factor(covariance)

    degrees = float(config.degrees_of_freedom)
    seed = _validate_seed(config.random_seed)

    rng = np.random.default_rng(seed)

    gaussian = rng.standard_normal(
        size=(paths, periods, assets),
    )

    chi_square = rng.chisquare(
        df=degrees,
        size=(paths, periods),
    )

    if not np.all(np.isfinite(chi_square)):
        _fail(
            "NONFINITE_CHI_SQUARE",
            "Chi-square generation produced nonfinite values.",
        )

    if np.any(chi_square <= 0):
        _fail(
            "INVALID_CHI_SQUARE_DRAW",
            "Chi-square draws must be strictly positive.",
        )

    correlated_gaussian = (
        gaussian @ factor.T
    )

    scale = np.sqrt(
        (degrees - 2.0) / chi_square
    )

    generated = (
        means[None, None, :]
        + correlated_gaussian * scale[:, :, None]
    )

    if not np.all(np.isfinite(generated)):
        _fail(
            "NONFINITE_GENERATED_RETURNS",
            "Generated returns contain NaN or infinity.",
        )

    if np.any(generated < -1.0):
        invalid_count = int(
            np.count_nonzero(generated < -1.0)
        )

        _fail(
            "RETURN_BELOW_NEGATIVE_ONE",
            f"{invalid_count} generated simple returns "
            "were below -100%. The long-only portfolio "
            "engine cannot accept these draws. "
            "No clipping or resampling was performed.",
        )

    return StudentTReturnResult(
        asset_ids=config.asset_ids,
        simple_returns=generated,
        expected_returns=means.copy(),
        volatilities=volatilities.copy(),
        correlation_matrix=correlation.copy(),
        covariance_matrix=covariance.copy(),
        degrees_of_freedom=degrees,
        random_seed=seed,
    )
