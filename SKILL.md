
---
name: monte-carlo-simulation
description: Validated, reproducible multi-asset Monte Carlo portfolio simulation, financial goal and downside-risk analysis, lifecycle cash-flow modeling, and scenario comparison with standalone and multi-agent operation.
version: 1.0.0
status: specification
---

# Skill #8 — Monte Carlo Simulation

## 1. Purpose and Scope

Provide a reproducible, auditable, model-aware Monte Carlo portfolio analysis capability for the IE 591 AI Portfolio Optimization project.

Simulate multi-asset portfolios across accumulation and withdrawal phases, estimate distributions of hypothetical outcomes, assess financial goals, stress test scenarios, and provide validated results for visualization and downstream decision support.

This document is the approved implementation specification, not evidence of a working numerical engine.

**Financial disclosure:** All outcomes and probabilities are conditional, model-dependent hypothetical estimates, not forecasts, guarantees, personalized investment advice, or authoritative tax guidance.

## 2. Approved Design Decisions

| Decision | Approved Option C |
|---|---|
| 1 — Scope | Comprehensive Monte Carlo Portfolio Intelligence |
| 2 — Inputs | Hybrid Input Architecture with Assumption Governance |
| 3 — Engine | Comprehensive Multi-Model Monte Carlo Engine |
| 4 — Analytics | Comprehensive Probabilistic Portfolio Intelligence |
| 5 — Scenarios | Comprehensive Scenario & Economic Regime Intelligence |
| 6 — Lifecycle | Comprehensive Financial Lifecycle Simulation |
| 7 — Reporting | Comprehensive Modular Reporting & Visualization Architecture |
| 8 — Governance | Comprehensive Statistical Validation & Model Governance |
| 9 — Integration | Hybrid Multi-Agent Orchestration with Independent Execution |
| 10 — QA | Comprehensive Multi-Gate Verification & Professor Acceptance Framework |

## 3. Core Operating Principles

1. The Python engine, not an LLM, performs calculations. Optional AI agents coordinate workflows and explain validated results.
2. Standalone file-driven execution must not require another agent or an OpenAI API key.
3. Version all contracts and retain assumption provenance, model configuration, input snapshot, run ID, seed, and diagnostic results.
4. Distinguish observed historical data, calibrated parameters, user overrides, stress assumptions, and simulated future outcomes.
5. Never silently impute essential inputs, invent scenario probabilities, repair invalid covariance matrices, or substitute a model without explicit policy, recorded transformation, and warning or user approval.
6. Numerical results are authoritative; visualization sampling must never alter the canonical analytical outputs.
7. An invalid configuration fails closed. Noncritical limitations must be visible to the user and downstream agents.

## 4. Inputs and Versioned Contracts

### 4.1 Supported Input Modes

- **Standalone:** Local or uploaded JSON and CSV; all necessary assumptions supplied explicitly.
- **Agent-driven:** Versioned JSON objects from upstream agents; contracts validated at the receiving boundary.
- **Hybrid:** Agent-sourced values supplemented by explicit user overrides with provenance and conflict resolution.

### 4.2 Canonical Simulation Request

Schema identifier: `monte_carlo.request.v1`

A request must identify:

- `schema_version`, `request_id`, `as_of_date`, `base_currency`, and `input_mode`.
- `portfolio`: Asset identifiers, asset class, beginning balances or weights, cash balance, target weights, account types where applicable.
- `market_assumptions`: Time basis, expected return convention, volatility, correlations or covariance, inflation assumptions, calibration data and sources.
- `simulation`: Model identifier, horizon, time-step frequency, number of paths, seed, selected outputs, and compute limits.
- `cash_flows`: Contribution and withdrawal schedules, external income, fees, transaction costs, dividend/interest treatment, and timing conventions.
- `rebalancing`: None, scheduled, threshold, or glide-path policy; execution timing and costs.
- `goals`: Target wealth, target dates, minimum balance, depletion definition, or withdrawal sustainability criteria.
- `scenarios`: Baseline, stress, historical analogues, custom shocks, regime assumptions, and optional defensible weights.
- `governance`: Provenance, user overrides, model version, warnings, and permissions for optional transformations.

Requiredness is conditional on the selected model and features.

A concrete JSON Schema in `schemas/` must enforce structural rules, while semantic validators enforce numerical and cross-field constraints.

The contract must explicitly document conventions for:

- Annualized versus periodic parameters
- Nominal versus real values
- Log versus arithmetic returns
- Beginning- versus end-of-period cash flows

### 4.3 Agent Input Sources

| Upstream Skill | Planned Inputs |
|---|---|
| #1 Individual Investment Profile Builder | Goals, horizon, risk tolerance, constraints |
| #2 Market Data Analyst | Observed returns, timestamps, sources, calibration candidates |
| #3 Portfolio Optimization | Proposed allocations, constraints, optimized portfolios |
| #4 Portfolio Risk Assessment | Risk context, exposure constraints, reference metrics |
| #6 Historical Portfolio Backtesting | Historical observations and calibrated reference comparisons |

No upstream output is accepted solely because it originates from an agent.

Validate version, units, timestamps, completeness, and provenance.

## 5. Statistical Simulation Engine

### 5.1 Supported Model Families

**Parametric Monte Carlo**
- Correlated multivariate arithmetic-normal returns
- Lognormal price dynamics
- Correctly specified log-return conventions

**Fat-Tailed Distributions**
- Multivariate Student's t or documented alternatives
- Valid degrees of freedom
- Finite-moment requirements for requested statistics
- Correlation calibration

**Historical Bootstrap**
- Resampling complete cross-asset return vectors
- Retaining contemporaneous asset dependence

**Block Bootstrap**
- Contiguous observations or blocks
- Preservation of some serial dependence
- Disclosure of block-length assumptions

**Regime Switching**
- Conditional return distributions
- Valid stochastic transition matrices
- Regime persistence and duration

**Stress Testing**
- Deterministic or conditional shocks
- Recovery rules
- Custom economic environments
- No unsupported inference of event likelihood from severity

### 5.2 Simulation Procedure

1. Validate request, units, source freshness, calibration window, model support, and resource limits.
2. Normalize assumptions to a declared time step and return convention, with documented conversions.
3. Validate covariance and correlation matrices for symmetry and positive semidefiniteness.
4. Fail closed on invalid matrices unless a separately authorized, logged repair procedure is applied.
5. Initialize a reproducible random number generator using the recorded seed and algorithm/version.
6. Generate cross-asset return vectors or scenario/regime sequences without look-ahead bias.
7. Apply asset-level returns, distributions, dividends, external cash flows, taxes if configured, fees, rebalancing, and transaction costs in a documented event order.
8. Reconcile beginning balance plus investment gains/losses plus inflows minus outflows, fees, and taxes to ending balance, accounting explicitly for transfers and trade costs.
9. Track balances, portfolio weights, drawdowns, goal states, depletion, and requested path statistics.
10. Aggregate metrics and calculate numerical diagnostics, including convergence and sampling uncertainty.
11. Emit canonical results, warnings, metadata, and optional visualization data.

### 5.3 Computational Controls

- Configure time steps, paths, memory, and execution-time budgets.
- Prefer vectorized or batched computation while retaining reproducibility guarantees within the specified engine/version.
- Reject unsupported model/parameter combinations rather than silently substituting a simpler model.
- Document chunking, approximation, numerical tolerances, and reproducibility limits across environments.

## 6. Portfolio Lifecycle Mechanics

Support:

- Initial holdings
- Recurring and lump-sum contributions
- Employer matches when explicitly specified
- Fixed and inflation-indexed withdrawals
- Percentage-based and dynamic spending policies
- External income, including configurable pension and Social Security assumptions
- One-time expenses
- Portfolio depletion rules

### 6.1 Rebalancing

Support:

- Buy-and-hold
- Calendar-based rebalancing
- Threshold-based rebalancing
- Glide-path rebalancing

### 6.2 Investment Income and Expenses

Support:

- Dividends and interest
- Reinvestment
- Advisory fees
- Fund expenses
- Transaction costs
- Account-aware cash balances
- Configurable tax-aware withdrawals

Tax treatment is an assumption-driven model, not a live tax-law determination.

Enforce account constraints only when explicitly represented and validated.

### 6.3 Accounting Controls

Define and test:

- Event ordering
- Cash-flow dates
- Fractional holdings
- Negative-balance policy
- Nominal and real reporting
- Beginning and ending portfolio balances
- Ledger reconciliation

Record all financial events needed for reconciliation.

## 7. Analytics and Financial Goals

Calculate, when supported and requested:

### 7.1 Portfolio Outcome Metrics

- Terminal wealth mean
- Median
- Standard deviation
- Minimum and maximum simulated outcomes
- 5th, 25th, 50th, 75th, and 95th percentiles
- Time-series wealth percentile bands
- Probability of positive and negative returns

Percentile bands must be labeled simulation quantiles, not confidence intervals for the true future outcome.

### 7.2 Downside Risk Metrics

- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Expected shortfall
- Maximum drawdown distributions
- Loss-threshold probabilities
- Extreme downside scenario frequency and severity

Document sign, horizon, and confidence conventions.

### 7.3 Financial Goal Achievement

- Probability of reaching target wealth by a specified date
- Probability of meeting multiple goals
- Probability of maintaining minimum balances
- Required contribution sensitivity
- Probability of portfolio depletion
- Timing of depletion

### 7.4 Retirement Sustainability

- Portfolio longevity
- Withdrawal sustainability
- Sequence-of-returns risk
- Inflation-adjusted spending sustainability
- Alternative withdrawal policy comparisons

### 7.5 Strategy Comparisons

- Allocation comparisons
- Goal-achievement comparisons
- Downside protection comparisons
- Risk-adjusted metrics where mathematically appropriate
- Paired simulation comparisons using common random numbers when appropriate

### 7.6 Sensitivity and Uncertainty

Evaluate sensitivity to:

- Expected returns
- Volatility
- Correlations
- Inflation
- Fees
- Contributions
- Withdrawals
- Statistical model selection

Report Monte Carlo sampling error and convergence diagnostics.

## 8. Market Scenarios and Economic Regimes

### 8.1 Economic Regimes

Support:

- Bull markets
- Bear markets
- Recessions
- Recoveries
- High-inflation environments
- Elevated-volatility environments

Model regime-specific returns, volatility, correlations, persistence, and transition probabilities.

### 8.2 Historical Crisis-Inspired Scenarios

Support documented historical market data associated with:

- 2000–2002 technology-sector downturn
- 2007–2009 global financial crisis
- 2020 pandemic market shock
- 2022 inflation and interest-rate shock

Historical scenarios must identify source data, observation periods, and calibration methods.

### 8.3 Forward-Looking Stress Scenarios

Support:

- Equity market declines
- Bond market losses
- Interest-rate shocks
- Persistent inflation
- Increased correlations during crises
- Extended low-return periods
- Liquidity and volatility disruptions

### 8.4 Custom Scenarios

Allow user-defined:

- Asset-class shocks
- Scenario duration
- Recovery assumptions
- Contribution changes
- Withdrawal changes
- Spending changes
- Rebalancing policies

### 8.5 Scenario Governance

Report conditional outcomes under each scenario.

Aggregate scenario-weighted results only when valid scenario weights are supplied and their provenance and interpretation are documented.

Regime-switching transition probabilities must be nonnegative, and each row must sum to one within tolerance.

Keep regime probabilities separate from arbitrary stress-case severity assumptions.

## 9. Reporting and Visualization

### 9.1 Supported Outputs

- Interactive HTML dashboard
- Optional professional PDF report
- Canonical versioned JSON result
- CSV summary tables
- Optional full-path exports subject to size limits

### 9.2 Visualizations

Support:

- Sample simulation paths
- Portfolio growth percentile bands
- Terminal-value histograms
- Drawdown distributions
- Depletion probability charts
- Goal-achievement probability charts
- Sensitivity analysis plots
- Scenario comparisons
- Strategy comparisons

### 9.3 Canonical Result Contract

Schema identifier: `monte_carlo.result.v1`

Required result information:

- `schema_version`
- `run_id`
- `request_id`
- `status`
- `started_at`
- `completed_at`
- `engine_version`
- `model`
- `seed`
- `input_digest`
- `assumption_summary`
- `simulation_configuration`
- `metrics`
- `goals`
- `scenarios`
- `diagnostics`
- `validation_findings`
- `provenance`
- `export_manifest`

Include time-series percentile data and optional sampled paths when requested.

Clearly mark unavailable metrics and explain why they were not computed. Do not replace unavailable metrics with zero.

### 9.4 Reporting Rules

- Display prominent hypothetical and model-dependent disclosures.
- Surface material warnings in reports and downstream machine-readable results.
- Preserve complete validated metrics independently of visualization downsampling.
- Do not imply that a percentile trajectory represents a single realizable portfolio path.
- Maintain traceability from every chart and table to the canonical run and metric definition.

## 10. Validation and Model Governance

### 10.1 Critical Failures — Block Execution

Examples:

- Malformed schema
- Missing essential inputs
- Invalid portfolio weights
- Invalid cash flows
- Inconsistent units
- Invalid covariance or correlation matrices
- Unsupported return models
- Invalid regime-transition probabilities
- Nonfinite numerical values
- Impossible time horizons
- Unauthorized account behavior
- Exceeded resource limits
- Missing calibration required by the selected model

### 10.2 Warnings — Qualified Execution Only

Examples:

- Short calibration history for an otherwise supported model
- Stale but explicitly accepted data
- Extreme parameter sensitivity
- Low effective sample size
- Uncertain scenario weighting
- Numerical convergence below preferred thresholds when minimum acceptance thresholds still hold
- Optional model features disabled with explicit explanation

### 10.3 Informational Findings

Record:

- Random seed
- Model and version
- Data source identifiers
- Calibration dates
- Assumption overrides
- Transformations
- Random number generator
- Compute environment
- Export metadata

Persist an auditable manifest of source identifiers, timestamps, input digest, seed, code version, transformations, warnings, and downstream handoffs.

Model repair or fallback must never be silent.

## 11. Multi-Agent Integration and Deployment

### 11.1 Target Technology Stack

- HTML/CSS/JavaScript frontend
- Hosted Python backend
- GitHub repository and continuous integration
- Optional OpenAI-powered orchestration and explanation agents

Keep provider secrets on the server.

Standalone deterministic simulation and professor testing must work without an OpenAI API key.

### 11.2 Planned Upstream Integrations

- Skill #1 — Individual Investment Profile Builder
- Skill #2 — Market Data Analyst
- Skill #3 — Portfolio Optimization
- Skill #4 — Portfolio Risk Assessment
- Skill #6 — Historical Portfolio Backtesting

### 11.3 Planned Downstream Integrations

- Skill #5 — 3D Portfolio Visualization
- Skill #7 — Forward Testing & Paper Trading
- Skill #9 — Strategy Comparison
- Skill #10 — Decision Support

Contract adapters must check schema versions and preserve run IDs, provenance, metrics, and warnings.

If an upstream service fails, standalone mode remains available.

Never invent missing values or claim a handoff succeeded before confirmation.

### 11.4 Six-Phase Skill Workflow

**Phase 1 — Intake**
- Receive standalone or agent-provided request.
- Identify goals and required inputs.

**Phase 2 — Validate and Calibrate**
- Check schema and semantics.
- Validate assumptions and model applicability.
- Verify provenance.

**Phase 3 — Simulate**
- Run seeded stochastic paths.
- Apply financial lifecycle events.

**Phase 4 — Analyze**
- Calculate risk and probability metrics.
- Evaluate goals, scenarios, and sensitivities.
- Calculate convergence diagnostics.

**Phase 5 — Report and Export**
- Generate versioned machine-readable outputs.
- Produce optional dashboards and reports.

**Phase 6 — Handoff and Audit**
- Deliver validated results to authorized agents.
- Persist manifests and diagnostics.

## 12. Repository Structure

```text
monte-carlo-simulation/
├── SKILL.md
├── README.md
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── integration-contracts.md
│   └── validation-plan.md
├── schemas/
│   ├── monte_carlo.request.v1.schema.json
│   └── monte_carlo.result.v1.schema.json
├── examples/
│   ├── valid-baseline-request.json
│   ├── invalid-correlation-request.json
│   └── deterministic-reference-case.json
├── src/
│   └── ...  # Gate 2 implementation
├── tests/
│   ├── test_contracts.py
│   ├── test_engine_reference.py
│   ├── test_statistics.py
│   ├── test_cash_flows.py
│   ├── test_scenarios.py
│   └── test_integration.py
└── .github/workflows/
    └── validate.yml
```

The repository structure is a target, not a claim that these files have been created or tests passed.

Tests for unimplemented functionality must be reported as pending, never passed or silently skipped to inflate completion.

## 13. Verification Gates and Acceptance Criteria

### Gate 1 — Specification and Contract Validation

**Deliverables**
- SKILL.md
- README.md
- Architecture documentation
- Versioned JSON schemas
- Valid and invalid fixtures
- Contract tests
- GitHub Actions workflow

**Acceptance criteria**
- Schemas validate documented fixtures.
- Invalid cases are rejected for expected reasons.
- Implemented contract tests pass in GitHub Actions.
- Execution evidence is retained.

### Gate 2 — Mathematical Engine Verification

**Deliverables**
- Executable Python numerical engine
- Deterministic reference tests
- Statistical model implementations

**Acceptance criteria**
- Deterministic cases match independently derived results within documented tolerances.
- Random seed reproducibility is verified.
- Return distributions and covariance calculations pass.
- Accounting, rebalancing, inflation, withdrawals, and depletion tests pass.
- Unsupported models are not represented as implemented.

### Gate 3 — Scenario and Analytics Validation

**Deliverables**
- Executable stress and regime models
- Risk and financial-goal metrics
- Statistical diagnostics

**Acceptance criteria**
- Scenario fixtures pass.
- Transition probabilities are validated.
- VaR and CVaR calculations pass.
- Drawdown and quantile calculations pass.
- Goal and depletion probabilities pass.
- Sensitivity and convergence tests pass.
- Independent reference evidence is available.

### Gate 4 — Agent Integration

**Deliverables**
- Adapters for available upstream and downstream skills
- Versioned integration contracts
- End-to-end integration tests

**Acceptance criteria**
- Versioned handoffs succeed.
- Provenance is preserved.
- Missing dependencies are handled.
- Standalone fallback works.
- Unavailable integrations are explicitly marked pending.

### Gate 5 — Performance, Security, and Reliability

**Deliverables**
- Performance benchmarks
- Resource limits
- Secure credential handling
- Failure tests
- Consistent exports

**Acceptance criteria**
- Documented workload targets pass.
- Failures are bounded and diagnosed.
- Numerical outputs remain consistent.
- No provider secrets appear in client artifacts.

### Gate 6 — Professor Acceptance

**Deliverables**
- Hosted browser demonstration
- Preconfigured sample datasets
- Repeatable test guide
- Exportable simulation results
- Acceptance report

**Acceptance criteria**
- A reviewer can execute a deterministic example without an OpenAI API key.
- Calculations can be inspected.
- Reference outputs can be reproduced.
- JSON and CSV files can be downloaded.
- Test evidence and known limitations can be reviewed.

**Gate rule:** Do not mark a gate complete without observed execution evidence.

Passing Gate 1 does not imply completion of Gates 2–6.

## 14. Minimum Acceptance Test Matrix

| Test Family | Required Verification |
|---|---|
| Contracts | Valid/invalid schema fixtures, version mismatch, missing fields |
| Parameters | Units, bounds, weights, correlation PSD, transition matrix rows |
| Deterministic Engine | Zero-volatility constant return, zero-return cash flows, no-fee baseline |
| RNG | Same seed/configuration reproduces results under pinned environment |
| Statistical Properties | Empirical means, variances, correlations, fat-tail behavior within defined sampling tolerances |
| Bootstrap | Sampled vectors/blocks come from eligible historical windows; no future leakage |
| Lifecycle | Cash-flow timing, dividends, withdrawals, fees, rebalancing, taxes when supported |
| Risk/Goal Metrics | Quantiles, VaR, CVaR, drawdown, target probability, depletion boundary cases |
| Scenarios | Conditional shocks, regime transitions, no invented scenario probabilities |
| Numerical Diagnostics | Convergence, sampling errors, sensitivity, nonfinite handling |
| Reporting | JSON/CSV consistency, warnings, metadata, reproducibility manifest |
| Integration | Versioned handoff, missing agents, standalone fallback, no secrets in frontend |
| Professor Demo | Browser execution, repeatable examples, export downloads, documented results |

## 15. Definition of Done

Skill #8 is **specified** when the ten approved design decisions are incorporated into this document.

It is **contract-verified** only after Gate 1 execution evidence exists.

It is **functionally verified** only after the implemented numerical engine and analytics pass Gates 2–3.

It is **integrated and professor-ready** only after Gates 4–6 pass.

Report status by gate rather than using a single ambiguous "complete" label.

## 16. Next Implementation Steps

1. Commit this `SKILL.md` to the Skill #8 GitHub repository.
2. Create README, JSON schemas, valid/invalid fixtures, contract tests, and GitHub Actions workflow.
3. Run and inspect Gate 1 contract tests.
4. Implement the Python engine incrementally, starting with a deterministic single-asset baseline and a correlated multi-asset model.
5. Extend model families, financial lifecycle, risk analytics, scenarios, reporting, and integration, verifying each against independent reference cases.
6. Publish a browser-accessible professor demo only after required numerical and security checks pass.
