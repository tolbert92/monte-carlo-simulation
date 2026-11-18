
"""
Skill #8 — Monte Carlo Simulation
Gate 1: Contract and Semantic Validation Tests

Tests:
- JSON Schema Draft 2020-12 validity
- Request and result fixture compliance
- Portfolio weight reconciliation
- Market assumption alignment
- Correlation matrix mathematical validity
- Simulation parameter validation
- Financial goal and cash-flow consistency
- Regime-transition matrix validity
- Result metrics and probability consistency
- Deterministic reference fixture expectations

These tests do not execute a Monte Carlo simulation engine.
"""

import copy
import json
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
EXAMPLES = ROOT / "examples"

REQUEST_SCHEMA_FILE = (
    SCHEMAS / "monte_carlo.request.v1.schema.json"
)
RESULT_SCHEMA_FILE = (
    SCHEMAS / "monte_carlo.result.v1.schema.json"
)

TOLERANCE = 1e-8


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=FormatChecker()
    )


def validate_schema(instance, validator):
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: str(error.path)
    )
    return [error.message for error in errors]


def semantic_errors(request):
    """
    Validate cross-field financial and mathematical rules.

    Returns a list of stable error codes.

    This is a Gate 1 reference validator, not the final
    production validation engine.
    """
    errors = []

    portfolio = request["portfolio"]
    assets = portfolio["assets"]
    assumptions = request["market_assumptions"]
    simulation = request["simulation"]

    asset_ids = [asset["asset_id"] for asset in assets]
    asset_set = set(asset_ids)
    weights = [asset["target_weight"] for asset in assets]

    # Portfolio integrity
    if len(asset_ids) != len(asset_set):
        errors.append("DUPLICATE_ASSET_ID")

    if not np.isclose(
        sum(weights), 1.0, atol=TOLERANCE, rtol=0
    ):
        errors.append("INVALID_PORTFOLIO_WEIGHTS")

    # Market assumption alignment
    for field in ("expected_returns", "volatilities"):
        keys = set(assumptions[field])
        if keys != asset_set:
            errors.append(
                f"ASSET_ALIGNMENT_{field.upper()}"
            )

    # Correlation matrix integrity
    matrix = np.asarray(
        assumptions["correlation_matrix"],
        dtype=float
    )
    n_assets = len(assets)

    if matrix.shape != (n_assets, n_assets):
        errors.append("INVALID_CORRELATION_DIMENSIONS")
    elif not np.all(np.isfinite(matrix)):
        errors.append("NONFINITE_CORRELATION")
    else:
        if not np.allclose(
            matrix, matrix.T,
            atol=TOLERANCE,
            rtol=0
        ):
            errors.append("INVALID_CORRELATION_SYMMETRY")

        if not np.allclose(
            np.diag(matrix), 1.0,
            atol=TOLERANCE,
            rtol=0
        ):
            errors.append("INVALID_CORRELATION_DIAGONAL")

        if np.any(np.abs(matrix) > 1 + TOLERANCE):
            errors.append("INVALID_CORRELATION_RANGE")

        if np.allclose(
            matrix, matrix.T,
            atol=TOLERANCE,
            rtol=0
        ):
            eigenvalues = np.linalg.eigvalsh(
                (matrix + matrix.T) / 2
            )
            if np.min(eigenvalues) < -TOLERANCE:
                errors.append("INVALID_CORRELATION_PSD")

    # Simulation parameters
    if simulation["number_of_paths"] < 1:
        errors.append("INVALID_PATH_COUNT")

    if simulation["horizon_years"] < 1:
        errors.append("INVALID_HORIZON")

    # Dates and horizon
    start_date = date.fromisoformat(
        request["as_of_date"]
    )
    horizon_year = (
        start_date.year + simulation["horizon_years"]
    )

    for goal in request["goals"]:
        target_date = date.fromisoformat(
            goal["target_date"]
        )
        if (
            target_date < start_date
            or target_date.year > horizon_year
        ):
            errors.append("GOAL_OUTSIDE_HORIZON")

    for flow in request.get("cash_flows", []):
        flow_start = date.fromisoformat(
            flow["start_date"]
        )
        if (
            flow_start < start_date
            or flow_start.year > horizon_year
        ):
            errors.append("CASH_FLOW_OUTSIDE_HORIZON")

        if "end_date" in flow:
            flow_end = date.fromisoformat(
                flow["end_date"]
            )
            if (
                flow_end < flow_start
                or flow_end.year > horizon_year
            ):
                errors.append("INVALID_CASH_FLOW_END")

    # Regime model integrity
    parameters = simulation.get(
        "model_parameters", {}
    )

    if simulation["model"] == "regime_switching":
        regimes = parameters["regimes"]
        transitions = np.asarray(
            parameters["regime_transition_matrix"],
            dtype=float
        )
        n_regimes = len(regimes)

        if transitions.shape != (
            n_regimes, n_regimes
        ):
            errors.append(
                "INVALID_REGIME_MATRIX_DIMENSIONS"
            )
        elif not np.all(np.isfinite(transitions)):
            errors.append("NONFINITE_REGIME_MATRIX")
        else:
            if np.any(
                (transitions < 0)
                | (transitions > 1)
            ):
                errors.append(
                    "INVALID_REGIME_PROBABILITIES"
                )

            if not np.allclose(
                transitions.sum(axis=1),
                1.0,
                atol=TOLERANCE,
                rtol=0
            ):
                errors.append(
                    "INVALID_REGIME_ROW_SUM"
                )

    # Rebalancing target weights
    rebalancing = request.get(
        "rebalancing", {"policy": "none"}
    )

    if rebalancing["policy"] == "glide_path":
        for entry in rebalancing["glide_path"]:
            target_weights = entry["target_weights"]

            if set(target_weights) != asset_set:
                errors.append(
                    "GLIDE_PATH_ASSET_MISMATCH"
                )

            if not np.isclose(
                sum(target_weights.values()),
                1.0,
                atol=TOLERANCE,
                rtol=0
            ):
                errors.append(
                    "INVALID_GLIDE_PATH_WEIGHTS"
                )

    # Scenario weights are optional.
    scenarios = request.get("scenarios", [])
    weighted = [
        scenario["scenario_weight"]
        for scenario in scenarios
        if "scenario_weight" in scenario
    ]

    if weighted and len(weighted) == len(scenarios):
        if not np.isclose(
            sum(weighted),
            1.0,
            atol=TOLERANCE,
            rtol=0
        ):
            errors.append("INVALID_SCENARIO_WEIGHTS")

    return sorted(set(errors))


def result_semantic_errors(result):
    """Validate cross-field result consistency."""
    errors = []

    config = result["simulation_configuration"]

    if (
        config["completed_paths"]
        > config["requested_paths"]
    ):
        errors.append("COMPLETED_PATHS_EXCEED_REQUEST")

    warning_count = sum(
        finding["severity"] == "warning"
        for finding in result["validation_findings"]
    )

    declared_warnings = result["diagnostics"].get(
        "warnings_count"
    )

    if (
        declared_warnings is not None
        and declared_warnings != warning_count
    ):
        errors.append("WARNING_COUNT_MISMATCH")

    if result["status"] in (
        "completed",
        "completed_with_warnings"
    ):
        if config["completed_paths"] == 0:
            errors.append("NO_COMPLETED_PATHS")

        for finding in result["validation_findings"]:
            if finding["severity"] == "critical":
                errors.append(
                    "CRITICAL_FINDING_IN_COMPLETED_RUN"
                )

        terminal = result["metrics"]["terminal_wealth"]
        percentiles = terminal["percentiles"]

        values = [
            percentiles[key]
            for key in ("p5", "p25", "p50", "p75", "p95")
        ]

        if all(value is not None for value in values):
            if any(
                values[i] > values[i + 1]
                for i in range(len(values) - 1)
            ):
                errors.append("UNORDERED_PERCENTILES")

    periods = [
        point["period"]
        for point in result.get("time_series", [])
    ]

    if periods != sorted(set(periods)):
        errors.append("INVALID_TIME_SERIES_ORDER")

    return sorted(set(errors))


class TestSchemaContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.request_schema = load_json(
            REQUEST_SCHEMA_FILE
        )
        cls.result_schema = load_json(
            RESULT_SCHEMA_FILE
        )

        cls.request_validator = make_validator(
            cls.request_schema
        )
        cls.result_validator = make_validator(
            cls.result_schema
        )

        cls.baseline = load_json(
            EXAMPLES / "valid-baseline-request.json"
        )
        cls.deterministic = load_json(
            EXAMPLES / "deterministic-reference-case.json"
        )
        cls.invalid_corr = load_json(
            EXAMPLES / "invalid-correlation-request.json"
        )
        cls.result = load_json(
            EXAMPLES / "valid-baseline-result.json"
        )

    def test_01_schema_documents_are_valid(self):
        Draft202012Validator.check_schema(
            self.request_schema
        )
        Draft202012Validator.check_schema(
            self.result_schema
        )

    def test_02_request_schema_version(self):
        self.assertEqual(
            self.baseline["schema_version"],
            "monte_carlo.request.v1"
        )

    def test_03_valid_baseline_request(self):
        self.assertEqual(
            validate_schema(
                self.baseline,
                self.request_validator
            ),
            []
        )

    def test_04_deterministic_request(self):
        self.assertEqual(
            validate_schema(
                self.deterministic,
                self.request_validator
            ),
            []
        )

    def test_05_invalid_correlation_is_structurally_valid(self):
        self.assertEqual(
            validate_schema(
                self.invalid_corr,
                self.request_validator
            ),
            []
        )

    def test_06_valid_result_contract(self):
        self.assertEqual(
            validate_schema(
                self.result,
                self.result_validator
            ),
            []
        )

    def test_07_reject_wrong_schema_version(self):
        modified = copy.deepcopy(self.baseline)
        modified["schema_version"] = "invalid.version"

        self.assertTrue(
            validate_schema(
                modified,
                self.request_validator
            )
        )

    def test_08_reject_missing_required_field(self):
        modified = copy.deepcopy(self.baseline)
        del modified["simulation"]

        self.assertTrue(
            validate_schema(
                modified,
                self.request_validator
            )
        )

    def test_09_reject_invalid_probability_range(self):
        modified = copy.deepcopy(self.result)
        modified["goals"][0][
            "achievement_probability"
        ] = 1.5

        self.assertTrue(
            validate_schema(
                modified,
                self.result_validator
            )
        )

    def test_10_baseline_semantic_validation(self):
        self.assertEqual(
            semantic_errors(self.baseline),
            []
        )

    def test_11_deterministic_semantic_validation(self):
        self.assertEqual(
            semantic_errors(self.deterministic),
            []
        )

    def test_12_reject_non_psd_correlation(self):
        errors = semantic_errors(
            self.invalid_corr
        )

        self.assertIn(
            "INVALID_CORRELATION_PSD",
            errors
        )

    def test_13_reject_incorrect_portfolio_weights(self):
        modified = copy.deepcopy(self.baseline)
        modified["portfolio"]["assets"][0][
            "target_weight"
        ] = 0.7

        self.assertIn(
            "INVALID_PORTFOLIO_WEIGHTS",
            semantic_errors(modified)
        )

    def test_14_reject_asset_assumption_mismatch(self):
        modified = copy.deepcopy(self.baseline)
        del modified["market_assumptions"][
            "expected_returns"
        ]["CASH"]

        self.assertIn(
            "ASSET_ALIGNMENT_EXPECTED_RETURNS",
            semantic_errors(modified)
        )

    def test_15_reject_duplicate_asset_ids(self):
        modified = copy.deepcopy(self.baseline)
        modified["portfolio"]["assets"][1][
            "asset_id"
        ] = "EQUITY"

        self.assertIn(
            "DUPLICATE_ASSET_ID",
            semantic_errors(modified)
        )

    def test_16_reject_invalid_correlation_dimensions(self):
        modified = copy.deepcopy(self.baseline)
        modified["market_assumptions"][
            "correlation_matrix"
        ] = [[1, 0], [0, 1]]

        self.assertIn(
            "INVALID_CORRELATION_DIMENSIONS",
            semantic_errors(modified)
        )

    def test_17_reject_goal_outside_horizon(self):
        modified = copy.deepcopy(self.baseline)
        modified["goals"][0][
            "target_date"
        ] = "2060-11-17"

        self.assertIn(
            "GOAL_OUTSIDE_HORIZON",
            semantic_errors(modified)
        )

    def test_18_reject_invalid_regime_probabilities(self):
        modified = copy.deepcopy(self.baseline)
        modified["simulation"]["model"] = (
            "regime_switching"
        )

        regime = {
            "regime_id": "NORMAL",
            "expected_returns": {
                "EQUITY": 0.07,
                "BONDS": 0.035,
                "CASH": 0.02
            },
            "volatilities": {
                "EQUITY": 0.16,
                "BONDS": 0.06,
                "CASH": 0.005
            },
            "correlation_matrix": [
                [1, 0.2, 0],
                [0.2, 1, 0.1],
                [0, 0.1, 1]
            ]
        }

        second_regime = copy.deepcopy(regime)
        second_regime["regime_id"] = "STRESS"

        modified["simulation"]["model_parameters"] = {
            "regimes": [
                regime,
                second_regime
            ],
            "regime_transition_matrix": [
                [0.8, 0.3],
                [0.2, 0.8]
            ]
        }

        self.assertEqual(
            validate_schema(
                modified,
                self.request_validator
            ),
            []
        )

        self.assertIn(
            "INVALID_REGIME_ROW_SUM",
            semantic_errors(modified)
        )

    def test_19_valid_result_semantics(self):
        self.assertEqual(
            result_semantic_errors(self.result),
            []
        )

    def test_20_reject_unordered_percentiles(self):
        modified = copy.deepcopy(self.result)

        modified["metrics"]["terminal_wealth"][
            "percentiles"
        ]["p25"] = 600000

        self.assertIn(
            "UNORDERED_PERCENTILES",
            result_semantic_errors(modified)
        )

    def test_21_reject_completed_paths_over_request(self):
        modified = copy.deepcopy(self.result)

        modified["simulation_configuration"][
            "completed_paths"
        ] = 20000

        self.assertIn(
            "COMPLETED_PATHS_EXCEED_REQUEST",
            result_semantic_errors(modified)
        )

    def test_22_reject_warning_count_mismatch(self):
        modified = copy.deepcopy(self.result)
        modified["diagnostics"]["warnings_count"] = 2

        self.assertIn(
            "WARNING_COUNT_MISMATCH",
            result_semantic_errors(modified)
        )

    def test_23_deterministic_reference_expectation(self):
        assets = self.deterministic[
            "portfolio"
        ]["assets"]

        initial_value = sum(
            asset["initial_value"]
            for asset in assets
        )

        annual_contribution = (
            self.deterministic["cash_flows"][0]["amount"]
        )

        years = self.deterministic[
            "simulation"
        ]["horizon_years"]

        expected_terminal_value = (
            initial_value
            + annual_contribution * years
        )

        self.assertEqual(
            expected_terminal_value,
            15000
        )

        self.assertEqual(
            self.deterministic["market_assumptions"][
                "volatilities"
            ]["CASH"],
            0
        )

        self.assertEqual(
            self.deterministic["market_assumptions"][
                "expected_returns"
            ]["CASH"],
            0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
