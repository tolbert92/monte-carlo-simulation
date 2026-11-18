
# Skill #8 — Monte Carlo Simulation

**IE 591 — AI Portfolio Optimization Project**

**Version:** 1.0.0  
**Status:** Specification Published | Gate 1 In Progress  
**Repository:** [monte-carlo-simulation](https://github.com/tolbert92/monte-carlo-simulation)

---

## 1. Project Overview

The Monte Carlo Simulation skill is the eighth component of a ten-skill AI Portfolio Optimization platform developed for the IE 591 engineering course.

Its purpose is to evaluate investment portfolios under uncertain future market conditions using statistical simulation, financial lifecycle modeling, probabilistic risk analysis, and scenario testing.

The intended system will simulate thousands of possible investment outcomes to help evaluate:

- Long-term portfolio growth
- Probability of achieving investment goals
- Downside risk and potential portfolio losses
- Retirement withdrawal sustainability
- Portfolio depletion risk
- Market crashes and economic stress scenarios
- Alternative investment strategies and allocations
- Sensitivity to financial and statistical assumptions

The skill is designed to operate independently or as part of a multi-agent portfolio intelligence architecture.

**Current status:** The approved design specification has been published. The functional Python simulation engine and complete application have not yet been implemented or verified.

All simulated outcomes are hypothetical, model-dependent estimates and are not guaranteed financial forecasts or personalized investment advice.

---

## 2. Project Objectives

The primary objectives are to:

1. Develop a reproducible multi-asset Monte Carlo simulation engine.
2. Support multiple statistical simulation methodologies.
3. Evaluate portfolio risk and financial goal achievement.
4. Incorporate contributions, withdrawals, fees, inflation, and rebalancing.
5. Simulate changing economic regimes and stress scenarios.
6. Produce interactive visualizations and standardized analytical reports.
7. Integrate with other portfolio intelligence skills through versioned JSON contracts.
8. Provide automated verification and independent mathematical validation.
9. Enable browser-based testing and evaluation by the course professor.
10. Preserve model assumptions, data provenance, and simulation reproducibility.

---

## 3. Approved Design Architecture

All ten design decisions were approved using Option C.

| # | Design Area | Approved Architecture |
|---|---|---|
| 1 | Simulation Scope | Comprehensive Monte Carlo Portfolio Intelligence |
| 2 | Input Architecture | Hybrid Input Architecture with Assumption Governance |
| 3 | Statistical Modeling | Comprehensive Multi-Model Monte Carlo Engine |
| 4 | Probability Analysis | Comprehensive Probabilistic Portfolio Intelligence |
| 5 | Market Scenarios | Comprehensive Scenario & Economic Regime Intelligence |
| 6 | Financial Mechanics | Comprehensive Financial Lifecycle Simulation |
| 7 | Reporting | Comprehensive Modular Reporting & Visualization Architecture |
| 8 | Validation | Comprehensive Statistical Validation & Model Governance |
| 9 | Agent Integration | Hybrid Multi-Agent Orchestration with Independent Execution |
| 10 | Testing | Comprehensive Multi-Gate Verification & Professor Acceptance Framework |

The complete approved specification is documented in [SKILL.md](SKILL.md).

---

## 4. Planned Core Capabilities

### 4.1 Multi-Asset Portfolio Simulation

The engine will support portfolios containing multiple asset classes, including:

- Equities
- Fixed-income investments
- Cash and cash equivalents
- Other supported investment categories

Portfolio simulations will account for:

- Starting balances
- Asset allocation weights
- Expected returns
- Volatility
- Asset correlations
- Investment horizons
- Rebalancing policies
- Portfolio cash flows

### 4.2 Statistical Simulation Models

The planned engine supports multiple modeling approaches.

| Model | Purpose |
|---|---|
| Parametric Monte Carlo | Simulate returns using specified statistical distributions |
| Correlated Multi-Asset Simulation | Preserve modeled relationships between asset classes |
| Fat-Tailed Distributions | Evaluate elevated probabilities of extreme outcomes |
| Historical Bootstrap | Resample historical return observations |
| Block Bootstrap | Preserve some historical serial dependence |
| Regime Switching | Model transitions between different economic environments |
| Stress Testing | Evaluate conditional outcomes under specified shocks |

Model selection will depend on validated inputs, calibration data, and statistical suitability.

Unsupported or insufficiently calibrated models must not be silently substituted.

### 4.3 Portfolio Lifecycle Modeling

The skill will model investment activity across accumulation and retirement phases.

Supported mechanics will include:

- Recurring contributions
- Lump-sum investments
- Employer contributions when configured
- Scheduled withdrawals
- Inflation-adjusted withdrawals
- Dynamic spending strategies
- Dividends and interest
- Investment management fees
- Transaction costs
- Portfolio rebalancing
- External income
- Configurable tax assumptions
- Portfolio depletion

All financial events must follow documented accounting and timing conventions.

### 4.4 Financial Goal Evaluation

The system will estimate model-dependent probabilities for:

- Achieving a target portfolio value
- Reaching investment goals by specified dates
- Maintaining a minimum portfolio balance
- Sustaining retirement withdrawals
- Avoiding portfolio depletion
- Meeting multiple financial objectives

### 4.5 Risk Analytics

Planned analytical outputs include:

- Mean and median portfolio values
- Standard deviation
- Portfolio outcome percentiles
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum drawdown distributions
- Loss-threshold probabilities
- Portfolio depletion probability
- Sequence-of-returns risk
- Sensitivity analysis
- Monte Carlo convergence diagnostics

Percentile bands describe simulated outcome distributions, not guaranteed future portfolio values.

---

## 5. Market Scenarios and Stress Testing

The skill will support both stochastic economic regime modeling and explicitly defined stress scenarios.

### Economic Regimes

- Bull markets
- Bear markets
- Recessions
- Recoveries
- High-inflation environments
- High-volatility environments

### Historical Crisis-Inspired Scenarios

- 2000–2002 technology-sector downturn
- 2007–2009 global financial crisis
- 2020 pandemic market shock
- 2022 inflation and interest-rate shock

Historical scenario implementations must use documented data sources and calibration periods.

### Custom Stress Scenarios

Users will be able to configure:

- Equity market shocks
- Bond market losses
- Interest-rate changes
- Inflation shocks
- Changes in asset correlations
- Extended low-return environments
- Recovery assumptions
- Contribution and withdrawal changes

Stress scenarios describe conditional outcomes. They do not automatically establish the probability of the stressed event occurring.

---

## 6. System Architecture

The planned technology architecture is:

| Layer | Technology | Responsibility |
|---|---|---|
| Presentation | HTML, CSS, JavaScript | User interface and visualizations |
| Application | Hosted Python backend | Request handling and workflow execution |
| Analytics | Python statistical libraries | Monte Carlo calculations and financial analytics |
| AI Orchestration | Optional OpenAI-powered agents | Workflow coordination and interpretation |
| Integration | Versioned JSON contracts | Agent-to-agent data exchange |
| Source Control | GitHub | Version control and project documentation |
| Validation | Python tests and GitHub Actions | Automated verification |

**Important:** Numerical calculations will be performed by deterministic or stochastic Python code, not generated directly by a language model.

The core simulation engine must operate without requiring an OpenAI API key.

---

## 7. Six-Phase Simulation Workflow

### Phase 1 — Intake

Receive a simulation request from a standalone JSON/CSV file or a supported upstream agent.

Identify:

- Portfolio composition
- Financial objectives
- Investment horizon
- Market assumptions
- Cash-flow requirements
- Simulation settings

### Phase 2 — Validate and Calibrate

Validate:

- Input schemas
- Portfolio weights
- Return and volatility assumptions
- Correlation matrices
- Model compatibility
- Calibration data
- Data provenance
- Resource requirements

### Phase 3 — Simulate

Execute the selected statistical model.

Generate simulated portfolio paths and apply:

- Investment returns
- Contributions
- Withdrawals
- Fees
- Rebalancing
- Inflation
- Other configured financial events

### Phase 4 — Analyze

Calculate:

- Portfolio outcome distributions
- Downside-risk metrics
- Goal-achievement probabilities
- Portfolio depletion probabilities
- Scenario comparisons
- Sensitivity measures
- Convergence diagnostics

### Phase 5 — Report and Export

Generate validated outputs in supported formats.

Planned formats include:

- JSON
- CSV
- Interactive HTML
- PDF reports

### Phase 6 — Handoff and Audit

Preserve:

- Simulation run identifier
- Random seed
- Model version
- Input snapshot
- Assumption provenance
- Validation findings
- Execution diagnostics

Transfer validated results to authorized downstream agents.

---

## 8. Multi-Agent Integration

Skill #8 is designed as an independent analytical component within the broader IE 591 portfolio intelligence platform.

### Planned Upstream Skills

| Skill | Planned Contribution |
|---|---|
| #1 — Individual Investment Profile Builder | Investor goals, horizon, and risk preferences |
| #2 — Market Data Analyst | Historical observations and market assumptions |
| #3 — Portfolio Optimization | Candidate portfolio allocations |
| #4 — Portfolio Risk Assessment | Risk context and constraints |
| #6 — Historical Portfolio Backtesting | Historical results and calibration references |

### Planned Downstream Skills

| Skill | Planned Use |
|---|---|
| #5 — 3D Portfolio Visualization | Visualization of simulation outputs |
| #7 — Forward Testing & Paper Trading | Comparison with forward-testing outcomes |
| #9 — Strategy Comparison | Evaluation of alternative strategies |
| #10 — Decision Support | Interpretation of probabilities, risks, and goals |

All integrations will use validated, versioned JSON contracts.

These interfaces are planned and are not yet verified as operational.

---

## 9. Input and Output Contracts

### Canonical Input Contract

`monte_carlo.request.v1`

The input contract will define:

- Portfolio holdings and weights
- Market assumptions
- Statistical model configuration
- Simulation horizon and path count
- Random seed
- Cash-flow schedules
- Rebalancing rules
- Financial goals
- Market scenarios
- Provenance and governance metadata

### Canonical Output Contract

`monte_carlo.result.v1`

The output contract will define:

- Simulation identifiers
- Execution status
- Model configuration
- Portfolio outcome metrics
- Risk analytics
- Financial goal probabilities
- Scenario results
- Statistical diagnostics
- Validation findings
- Data provenance
- Export metadata

### Planned Schema Files

```text
schemas/
├── monte_carlo.request.v1.schema.json
└── monte_carlo.result.v1.schema.json
```

Schema files will be created and validated during Gate 1.

---

## 10. Validation and Model Governance

The skill will use three validation levels.

| Level | Meaning | Required Behavior |
|---|---|---|
| Critical | Invalid or mathematically inconsistent configuration | Block execution |
| Warning | Noncritical limitation or uncertainty | Disclose and qualify results |
| Information | Execution metadata or assumption details | Record for auditability |

Examples of critical validation failures include:

- Invalid portfolio weights
- Unsupported statistical models
- Invalid covariance matrices
- Missing essential inputs
- Invalid regime-transition probabilities
- Nonfinite numerical values
- Impossible simulation horizons

Model assumptions and limitations must remain visible in reports and downstream outputs.

---

## 11. Planned Repository Structure

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
│   └── ...
├── tests/
│   ├── test_contracts.py
│   ├── test_engine_reference.py
│   ├── test_statistics.py
│   ├── test_cash_flows.py
│   ├── test_scenarios.py
│   └── test_integration.py
└── .github/
    └── workflows/
        └── validate.yml
```

This is the target structure. Files not yet committed or implemented must not be represented as complete.

---

## 12. Six-Gate Verification Framework

### Gate 1 — Specification and Contract Validation

**Objective:** Verify documentation, schemas, fixtures, and contract validation.

Required deliverables:

- Approved SKILL.md
- Complete README.md
- JSON input/output schemas
- Valid and invalid fixtures
- Automated contract tests
- GitHub Actions workflow

**Acceptance:** All implemented contract tests pass in GitHub Actions, and the results are verified.

### Gate 2 — Mathematical Engine Verification

**Objective:** Implement and verify actual numerical calculations.

Required testing includes:

- Deterministic reference calculations
- Random seed reproducibility
- Statistical distribution behavior
- Correlated multi-asset returns
- Portfolio accounting
- Rebalancing
- Contributions and withdrawals
- Inflation and fees

**Acceptance:** Numerical results match independently verified reference calculations within documented tolerances.

### Gate 3 — Scenario and Analytics Validation

**Objective:** Verify probabilistic analytics and scenario modeling.

Required testing includes:

- Stress scenarios
- Economic regimes
- VaR and CVaR
- Maximum drawdown
- Goal-achievement probabilities
- Portfolio depletion
- Sensitivity analysis
- Convergence diagnostics

**Acceptance:** Analytics and scenarios pass independent reference tests.

### Gate 4 — Agent Integration

**Objective:** Verify standardized agent-to-agent communication.

Required testing includes:

- Input contract compatibility
- Output contract compatibility
- Data provenance preservation
- Missing-service handling
- Standalone execution

**Acceptance:** Implemented integration workflows pass end-to-end tests.

### Gate 5 — Performance, Security, and Reliability

**Objective:** Verify operational quality.

Required testing includes:

- Simulation performance
- Resource limits
- Numerical stability
- Error handling
- Secure credential handling
- Export consistency

**Acceptance:** Documented performance and reliability targets pass.

### Gate 6 — Professor Validation

**Objective:** Enable independent academic evaluation.

Required capabilities include:

- Browser-accessible demonstration
- Sample portfolios
- Repeatable simulations
- Documented test instructions
- JSON and CSV downloads
- Mathematical reference tests
- Acceptance evidence

**Acceptance:** A reviewer can execute, inspect, and reproduce documented examples without supplying an OpenAI API key.

---

## 13. Gate 1 Implementation Roadmap

| Step | Deliverable | Status |
|---|---|---|
| 1 | Publish SKILL.md | Complete |
| 2 | Complete README.md | In Progress |
| 3 | Create input JSON Schema | Pending |
| 4 | Create output JSON Schema | Pending |
| 5 | Create example fixtures | Pending |
| 6 | Implement contract tests | Pending |
| 7 | Configure GitHub Actions | Pending |
| 8 | Execute and verify tests | Pending |

A passing Gate 1 contract test does not establish that the simulation engine is implemented.

---

## 14. Testing and Execution

### Current Execution Status

The repository currently contains the approved specification and initial documentation.

The following capabilities are planned but not yet available:

- Functional Monte Carlo calculations
- Statistical model execution
- Scenario simulation
- Automated numerical verification
- Interactive dashboard
- Hosted professor demonstration

### Gate 1 Testing

After the schemas, fixtures, and tests are committed, GitHub Actions will automatically execute the contract-validation suite.

The workflow must:

1. Check out repository files.
2. Configure the supported Python environment.
3. Install pinned test dependencies.
4. Validate example inputs and outputs.
5. Run automated contract tests.
6. Fail when required tests fail.
7. Display test results in GitHub Actions.

Exact execution commands and dependency versions will be documented when the tests are implemented.

### Future Numerical Execution

Gate 2 will introduce the functional Python engine and its command-line or application entry point.

The final implementation must support reproducible simulations using documented configuration files and random seeds.

---

## 15. Professor Validation Strategy

The completed application is intended to allow the professor to:

1. Open a hosted web application.
2. Select a sample investment portfolio.
3. Configure simulation assumptions.
4. Run Monte Carlo analysis.
5. Review portfolio distributions and financial-goal probabilities.
6. Compare market scenarios.
7. Download numerical results.
8. Reproduce deterministic reference tests.
9. Review validation evidence and model limitations.

Core numerical testing will not require the professor to create an OpenAI account or provide an API key.

Optional AI-powered explanation features may depend on credentials configured securely by the application operator.

---

## 16. Project Status

| Component | Current Status |
|---|---|
| Design Decisions | 10 of 10 Approved |
| SKILL.md | Published |
| README.md | In Progress |
| JSON Schemas | Pending |
| Contract Tests | Pending |
| GitHub Actions | Pending |
| Python Simulation Engine | Not Implemented |
| Agent Integration | Not Verified |
| Browser Application | Not Implemented |
| Professor Acceptance | Pending |

Status will be updated as verified implementation evidence becomes available.

---

## 17. Relationship to the IE 591 Project

This repository represents Skill #8 of the ten-skill portfolio intelligence architecture.

The overall project is intended to demonstrate the integration of:

- Financial data analysis
- Portfolio optimization
- Portfolio risk assessment
- Historical backtesting
- Forward testing
- Monte Carlo simulation
- Strategy comparison
- Decision support
- Interactive visualization
- AI-assisted workflow orchestration

The engineering emphasis is on modular design, repeatability, mathematical verification, traceability, and independent testing.

---

## 18. Disclaimer

This project is developed for academic and engineering demonstration purposes.

Monte Carlo simulation results depend on statistical assumptions, historical data, model selection, and user-defined parameters.

Past performance does not guarantee future results. Simulated probabilities are not guarantees of future investment outcomes.

The software is not a substitute for professional investment, financial planning, legal, or tax advice.

---

## 19. Next Development Milestone

**Gate 1 — Step 3: Create the canonical Monte Carlo request JSON Schema.**

The next file will be:

`schemas/monte_carlo.request.v1.schema.json`

It will establish the versioned input contract used by standalone simulations and future upstream agents.

---

**IE 591 — AI Portfolio Optimization Project**  
**Skill #8 — Monte Carlo Simulation**  
**Specification Version 1.0.0**
