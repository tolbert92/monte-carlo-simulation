
"""
Skill #8 — Monte Carlo Simulation
Gate 2, Stage 2.2: Return Engine Numerical Tests

Tests the Stage 2.1 mathematical return engine.

Coverage:
- Reproducible random number generation
- Seed sensitivity
- Output shape and metadata
- Zero-volatility deterministic returns
- Expected return and volatility sampling
- Independent, positive, and negative correlations
- Singular positive-semidefinite covariance
- Invalid correlation rejection
- Lognormal return conversion
- Asset ordering and request adaptation
- Unsupported models and return conventions
- Frequency and numerical input validation

Run:
    python -m unittest discover -s tests -p "test_return_engine.py" -v

These tests validate return generation, not portfolio evolution,
cash flows, taxes, or investment recommendations.
"""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monte_carlo.return_engine import (  # noqa: E402
    ReturnEngineConfig,
    ReturnEngineError,
    build_covariance_matrix,
    config_from_request,
    simulate_request,
    simulate_returns,
    validate_config,
)


def make_config(**overrides):
    """Create a small, valid two-asset test configuration."""

    values = {
        "asset_ids": ("EQUITY", "BONDS"),
        "expected_returns": (0.07, 0.03),
        "volatilities": (0.16, 0.06),
        "correlation_matrix": (
            (1.0, 0.2),
            (0.2, 1.0),
        ),
        "model": "multivariate_normal",
        "return_convention": "arithmetic_simple",
        "parameter_frequency": "annually",
        "time_step": "annually",
        "number_of_paths": 100,
        "number_of_periods": 5,
        "random_seed": 59108,
    }

    values.update(overrides)

    return ReturnEngineConfig(**values)


def load_fixture(filename):
    path = ROOT / "examples" / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class TestReturnEngine(unittest.TestCase):
    """Twenty numerical and validation tests."""

    def assert_engine_error(self, code, operation):
        """Verify the stable error code, not just an exception."""

        with self.assertRaises(ReturnEngineError) as context:
            operation()

        self.assertEqual(context.exception.code, code)

    def test_01_same_seed_produces_identical_returns(self):
        config = make_config()

        first = simulate_returns(config)
        second = simulate_returns(config)

        np.testing.assert_array_equal(
            first.simple_returns,
            second.simple_returns,
        )

        np.testing.assert_array_equal(
            first.model_returns,
            second.model_returns,
        )

    def test_02_different_seeds_change_returns(self):
        first = simulate_returns(
            make_config(random_seed=123)
        )

        second = simulate_returns(
            make_config(random_seed=456)
        )

        self.assertFalse(
            np.array_equal(
                first.simple_returns,
                second.simple_returns,
            )
        )

    def test_03_output_shape_and_metadata(self):
        config = make_config(
            number_of_paths=12,
            number_of_periods=8,
        )

        result = simulate_returns(config)

        self.assertEqual(
            result.simple_returns.shape,
            (12, 8, 2),
        )

        self.assertEqual(
            result.model_returns.shape,
            (12, 8, 2),
        )

        metadata = result.to_metadata()

        self.assertEqual(metadata["shape"], [12, 8, 2])
        self.assertEqual(metadata["random_seed"], 59108)
        self.assertEqual(
            metadata["asset_ids"],
            ["EQUITY", "BONDS"],
        )

    def test_04_zero_volatility_is_deterministic(self):
        config = make_config(
            expected_returns=(0.05, 0.02),
            volatilities=(0.0, 0.0),
            number_of_paths=500,
            number_of_periods=10,
        )

        result = simulate_returns(config)

        expected = np.broadcast_to(
            np.array([0.05, 0.02]),
            result.simple_returns.shape,
        )

        np.testing.assert_allclose(
            result.simple_returns,
            expected,
            atol=1e-14,
            rtol=0,
        )

    def test_05_sample_means_approach_inputs(self):
        config = make_config(
            number_of_paths=20000,
            number_of_periods=5,
            random_seed=2026,
        )

        result = simulate_returns(config)

        observed = result.simple_returns.mean(
            axis=(0, 1)
        )

        np.testing.assert_allclose(
            observed,
            config.expected_returns,
            atol=0.003,
            rtol=0,
        )

    def test_06_sample_volatilities_approach_inputs(self):
        config = make_config(
            number_of_paths=20000,
            number_of_periods=5,
            random_seed=2027,
        )

        result = simulate_returns(config)

        observed = result.simple_returns.std(
            axis=(0, 1)
        )

        np.testing.assert_allclose(
            observed,
            config.volatilities,
            atol=0.003,
            rtol=0,
        )

    def test_07_identity_correlation_is_near_zero(self):
        config = make_config(
            correlation_matrix=(
                (1.0, 0.0),
                (0.0, 1.0),
            ),
            number_of_paths=20000,
            number_of_periods=5,
        )

        result = simulate_returns(config)

        samples = result.simple_returns.reshape(-1, 2)

        observed = np.corrcoef(
            samples,
            rowvar=False,
        )[0, 1]

        self.assertLess(abs(observed), 0.03)

    def test_08_positive_correlation_is_preserved(self):
        config = make_config(
            correlation_matrix=(
                (1.0, 0.75),
                (0.75, 1.0),
            ),
            number_of_paths=20000,
            number_of_periods=5,
        )

        result = simulate_returns(config)

        samples = result.simple_returns.reshape(-1, 2)

        observed = np.corrcoef(
            samples,
            rowvar=False,
        )[0, 1]

        self.assertAlmostEqual(
            observed,
            0.75,
            delta=0.03,
        )

    def test_09_negative_correlation_is_preserved(self):
        config = make_config(
            correlation_matrix=(
                (1.0, -0.65),
                (-0.65, 1.0),
            ),
            number_of_paths=20000,
            number_of_periods=5,
        )

        result = simulate_returns(config)

        samples = result.simple_returns.reshape(-1, 2)

        observed = np.corrcoef(
            samples,
            rowvar=False,
        )[0, 1]

        self.assertAlmostEqual(
            observed,
            -0.65,
            delta=0.03,
        )

    def test_10_covariance_matrix_matches_formula(self):
        config = make_config()

        covariance = build_covariance_matrix(config)

        expected = np.array([
            [0.16 ** 2, 0.16 * 0.06 * 0.2],
            [0.16 * 0.06 * 0.2, 0.06 ** 2],
        ])

        np.testing.assert_allclose(
            covariance,
            expected,
            atol=1e-14,
            rtol=0,
        )

    def test_11_singular_psd_matrix_is_accepted(self):
        config = make_config(
            correlation_matrix=(
                (1.0, 1.0),
                (1.0, 1.0),
            ),
            volatilities=(0.1, 0.1),
            expected_returns=(0.04, 0.04),
        )

        validate_config(config)

        result = simulate_returns(config)

        np.testing.assert_allclose(
            result.simple_returns[:, :, 0],
            result.simple_returns[:, :, 1],
            atol=1e-12,
            rtol=0,
        )

    def test_12_invalid_psd_matrix_is_rejected(self):
        config = make_config(
            asset_ids=("A", "B", "C"),
            expected_returns=(0.05, 0.04, 0.03),
            volatilities=(0.2, 0.1, 0.05),
            correlation_matrix=(
                (1.0, 0.9, 0.9),
                (0.9, 1.0, -0.9),
                (0.9, -0.9, 1.0),
            ),
        )

        self.assert_engine_error(
            "INVALID_CORRELATION_PSD",
            lambda: simulate_returns(config),
        )

    def test_13_lognormal_conversion_is_correct(self):
        config = make_config(
            model="lognormal",
            return_convention="log",
            expected_returns=(0.05, 0.02),
        )

        result = simulate_returns(config)

        np.testing.assert_allclose(
            result.simple_returns,
            np.expm1(result.model_returns),
            atol=1e-14,
            rtol=0,
        )

        self.assertTrue(
            np.all(result.simple_returns > -1)
        )

    def test_14_request_adapter_aligns_asset_ids(self):
        request = load_fixture(
            "valid-baseline-request.json"
        )

        modified = copy.deepcopy(request)

        modified["portfolio"]["assets"] = [
            modified["portfolio"]["assets"][2],
            modified["portfolio"]["assets"][0],
            modified["portfolio"]["assets"][1],
        ]

        # Reorder the matrix consistently with the asset list.
        original = np.asarray(
            request["market_assumptions"][
                "correlation_matrix"
            ]
        )

        order = [2, 0, 1]

        modified["market_assumptions"][
            "correlation_matrix"
        ] = original[np.ix_(order, order)].tolist()

        config = config_from_request(modified)

        self.assertEqual(
            config.asset_ids,
            ("CASH", "EQUITY", "BONDS"),
        )

        self.assertEqual(
            config.expected_returns,
            (0.02, 0.07, 0.035),
        )

        self.assertEqual(
            config.volatilities,
            (0.005, 0.16, 0.06),
        )

    def test_15_request_fixture_executes(self):
        request = load_fixture(
            "deterministic-reference-case.json"
        )

        result = simulate_request(request)

        self.assertEqual(
            result.simple_returns.shape,
            (100, 5, 1),
        )

        np.testing.assert_array_equal(
            result.simple_returns,
            np.zeros((100, 5, 1)),
        )

    def test_16_unsupported_model_is_rejected(self):
        config = make_config(
            model="student_t"
        )

        self.assert_engine_error(
            "UNSUPPORTED_MODEL",
            lambda: simulate_returns(config),
        )

    def test_17_return_convention_mismatch_rejected(self):
        config = make_config(
            model="lognormal",
            return_convention="arithmetic_simple",
        )

        self.assert_engine_error(
            "RETURN_CONVENTION_MISMATCH",
            lambda: simulate_returns(config),
        )

    def test_18_frequency_mismatch_is_rejected(self):
        config = make_config(
            parameter_frequency="annually",
            time_step="monthly",
        )

        self.assert_engine_error(
            "FREQUENCY_MISMATCH",
            lambda: simulate_returns(config),
        )

    def test_19_nonfinite_inputs_are_rejected(self):
        config = make_config(
            expected_returns=(float("nan"), 0.03),
        )

        self.assert_engine_error(
            "INVALID_NUMERIC_INPUT",
            lambda: simulate_returns(config),
        )

        config_infinity = make_config(
            volatilities=(0.16, float("inf")),
        )

        self.assert_engine_error(
            "INVALID_NUMERIC_INPUT",
            lambda: simulate_returns(config_infinity),
        )

    def test_20_invalid_seed_and_path_count_rejected(self):
        invalid_seed = make_config(
            random_seed=-1
        )

        self.assert_engine_error(
            "INVALID_RANDOM_SEED",
            lambda: simulate_returns(invalid_seed),
        )

        invalid_paths = make_config(
            number_of_paths=0
        )

        self.assert_engine_error(
            "INVALID_PATH_COUNT",
            lambda: simulate_returns(invalid_paths),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
