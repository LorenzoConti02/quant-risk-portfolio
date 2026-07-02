# 📊 Quant Risk Portfolio Analyzer

**A production-ready Python engine for multi-asset portfolio risk measurement — Value at Risk (Historical, Parametric, Monte Carlo), Conditional VaR, historical stress testing, and regulatory-style backtesting.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-13%20passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

---

## 🎯 Project Overview

This project implements an institutional-grade **market risk measurement pipeline** for a multi-asset portfolio (equities, government bonds, gold), inspired by the risk management frameworks used by banks and asset managers under **Basel III / FRTB** guidelines.

Given a portfolio of tickers and weights, the engine:

1. Pulls real historical price data (Yahoo Finance API)
2. Computes **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)** using **three independent methodologies**
3. Runs **historical stress tests** against real market crash scenarios (1987, 2008, 2020, rate shocks)
4. **Backtests** the VaR model statistically (Kupiec Proportion of Failures test) — the same test regulators use to validate bank internal models
5. Produces a full visual + Markdown risk report, ready to share

The goal is to demonstrate the full skill stack required for **Quant Risk, Equity Research, and Asset Management** roles: financial theory, statistical rigor, and clean production Python.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-methodology VaR** | Historical Simulation, Parametric (Variance-Covariance), Monte Carlo (Cholesky-correlated simulation) |
| **Conditional VaR (CVaR)** | Expected Shortfall in the tail beyond VaR — the metric increasingly favored over VaR under Basel FRTB |
| **Historical Stress Testing** | 5 pre-built scenarios (Black Monday 1987, GFC 2008, COVID Crash 2020, Rate Shock, Geopolitical Shock) — fully customizable |
| **Model Validation (Backtesting)** | Rolling-window Kupiec POF test to statistically validate whether the VaR model is well-calibrated |
| **Robust Data Pipeline** | Local CSV caching, automatic fallback to synthetic (GBM) data if the API is unreachable, full exception handling |
| **Automated Visual Report** | 6 charts (cumulative returns, drawdown, return distribution with VaR overlay, correlation heatmap, stress test, VaR/CVaR comparison) + Markdown summary |
| **Unit-tested** | 13 automated tests validating statistical correctness (e.g. Monte Carlo VaR converging to the closed-form parametric solution) |

---

## 🧠 Theoretical & Engineering Background

### 1. Value at Risk — three lenses on the same question

VaR answers: *"What is the maximum loss I should expect not to exceed, X% of the time, over a given horizon?"* Each method makes a different trade-off between assumptions and computational cost — which is exactly why a serious risk desk never relies on just one.

**Historical Simulation** — fully non-parametric. It takes the empirical distribution of past portfolio returns and reads off the desired percentile directly:

```
VaR_α = -Percentile(R_portfolio, 1-α)
```

*Pro:* captures real fat tails and skewness. *Con:* assumes the past repeats itself and needs a long history.

**Parametric / Variance-Covariance (Delta-Normal)** — assumes portfolio returns are Normally distributed, `R ~ N(μ, σ²)`, where portfolio variance comes from the full covariance matrix:

```
σ_p² = wᵀ Σ w
VaR_α = -(μ·h + z_α · σ_p · √h)
```

with `z_α` the Normal quantile (e.g. -1.645 at 95%) and `h` the horizon in days (√t scaling rule). *Pro:* closed-form, instantaneous. *Con:* underestimates risk when returns have fat tails (which they almost always do).

**Monte Carlo Simulation** — simulates thousands of correlated asset return scenarios via **Cholesky decomposition** of the covariance matrix (`Σ = LLᵀ`), aggregates them into portfolio returns using the portfolio weights, then reads the empirical percentile of the simulated distribution. This is the most flexible method — it extends naturally to non-linear instruments (options) and non-Normal distributions.

### 2. Conditional VaR (CVaR / Expected Shortfall)

VaR tells you the threshold; it says nothing about how bad things get *beyond* that threshold. CVaR fixes this — it's the **average loss in the worst (1-α)% of cases**:

```
CVaR_α = E[R | R ≤ -VaR_α]
```

CVaR is a *coherent* risk measure (it satisfies subadditivity, unlike VaR), which is why the Basel Committee's FRTB framework moved the standard from VaR to **Expected Shortfall**.

### 3. Stress Testing

VaR is inherently backward-looking and probabilistic. Stress testing complements it by asking: *"What happens to my portfolio if a specific, severe (but not necessarily historically observed) scenario materializes tomorrow?"* This engine applies deterministic shocks to each asset and aggregates the impact through the current portfolio weights — a simplified version of the scenario analysis used in ICAAP and CCAR-style regulatory stress tests.

### 4. Backtesting — Kupiec Proportion of Failures Test

No risk model is credible without validation. The **Kupiec test** checks whether the observed number of VaR violations over a rolling window is statistically consistent with the expected violation rate `(1-α)`, using a likelihood-ratio statistic:

```
LR = -2·ln[ (1-p)^(n-x)·p^x / (1-x̂/n)^(n-x)·(x̂/n)^x ]  ~  χ²(1)
```

If the p-value exceeds 0.05, the model is **not rejected** at the 95% level — this is the exact mechanism regulators use to decide whether a bank's internal VaR model can be used for capital requirement purposes (traffic-light approach).

---

## 🏗️ Repository Structure

```
quant-risk-portfolio/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config.py                  # Central configuration: tickers, weights, risk params
├── main.py                    # Pipeline entry point (orchestrator)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Market data ingestion, caching, synthetic fallback
│   ├── analytics.py           # RiskEngine: VaR, CVaR, stress test, Kupiec backtest
│   ├── visualization.py       # Chart generation (matplotlib / seaborn)
│   └── report_generator.py    # Markdown report assembly
├── tests/
│   ├── __init__.py
│   └── test_analytics.py      # 13 unit tests on statistical correctness
├── data_cache/                # Local CSV cache of downloaded prices (gitignored)
└── outputs/                   # Generated charts + risk_report.md (gitignored)
```

---

## ⚙️ Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/quant-risk-portfolio.git
cd quant-risk-portfolio

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the full pipeline with real market data

```bash
python main.py
```

This downloads 5 years of historical prices for the configured portfolio via Yahoo Finance, computes all risk metrics, generates 6 charts, and writes a full Markdown report to `outputs/risk_report.md`.

### Run in offline / demo mode (synthetic data)

Useful if you don't have internet access, if Yahoo Finance rate-limits you, or if you just want a fast deterministic demo:

```bash
python main.py --synthetic
```

### Customize the portfolio

Edit `config.py`:

```python
TICKERS = ["AAPL", "MSFT", "JPM", "EUNL.DE", "TLT", "GLD"]
WEIGHTS = {"AAPL": 0.20, "MSFT": 0.20, "JPM": 0.15,
           "EUNL.DE": 0.20, "TLT": 0.15, "GLD": 0.10}
PORTFOLIO_VALUE = 1_000_000
CONFIDENCE_LEVELS = [0.95, 0.99]
```

Any Yahoo Finance ticker works (stocks, ETFs, indices). Weights must sum to 1.0 — this is enforced by an explicit validation check that raises a `ValueError` otherwise.

### Run the test suite

```bash
pytest tests/ -v
```

### Use the engine programmatically

```python
from src.data_loader import DataLoader
from src.analytics import RiskEngine

loader = DataLoader(["AAPL", "MSFT"], "2020-01-01", "2025-01-01")
prices = loader.fetch_prices()
returns = DataLoader.compute_log_returns(prices)

engine = RiskEngine(returns, weights={"AAPL": 0.6, "MSFT": 0.4},
                     portfolio_value=1_000_000)

var_95 = engine.historical_var(confidence=0.95)
print(var_95)
# [Historical Simulation] VaR 95% (1d): 1.83% | 18,347  CVaR: 2.41% | 24,103
```

---

## 📈 Example Output

Running the pipeline generates the following artifacts in `outputs/`:

- `cumulative_returns.png` — portfolio equity curve
- `drawdown.png` — historical drawdown chart
- `var_distribution.png` — return histogram with VaR thresholds overlaid across methods
- `correlation_heatmap.png` — asset correlation matrix
- `stress_test.png` — horizontal bar chart of scenario impacts
- `var_comparison.png` — VaR vs CVaR across methodologies
- `risk_report.md` — full tabular summary (composition, performance, VaR/CVaR, stress tests, backtest verdict)

Sample console output:

```
[Historical Simulation]         VaR 95% (1d): 1.57%  | CVaR: 2.09%
[Parametric (Variance-Cov)]     VaR 95% (1d): 1.64%  | CVaR: 2.05%
[Monte Carlo Simulation]        VaR 95% (1d): 1.64%  | CVaR: 2.05%

Kupiec Backtest (95%, 250-day window):
  Violations observed: 52  |  Expected: 50.5  |  p-value: 0.824  →  Model ACCEPTED
```

The near-perfect convergence between the three VaR methodologies on Gaussian-like data — and the divergence you'd see once you swap in real, fat-tailed market data — is itself the point: it's a live demonstration of *when* each method's assumptions hold and when they break down.

---

## 🚀 Possible Future Extensions

- **Dockerization** — package the pipeline into a container for reproducible deployment and CI/CD integration.
- **ESG-adjusted risk overlay** — integrate an ESG scoring API (e.g. Refinitiv, MSCI) to analyze how portfolio VaR shifts under ESG-tilted allocations, and use an LLM to summarize ESG controversy news flow as a qualitative risk signal.
- **Cloud & database migration** — move price caching from CSV to PostgreSQL/TimescaleDB and deploy the pipeline as a scheduled job (Airflow / AWS Lambda) with a lightweight API (FastAPI) exposing risk metrics on demand.
- **GARCH volatility modeling** — replace the constant-covariance assumption with a GARCH(1,1) or EWMA-based dynamic covariance estimator for more realistic time-varying VaR.
- **Portfolio optimization module** — extend the engine with a Markowitz / Black-Litterman optimizer that treats the computed VaR as an explicit constraint rather than just a diagnostic.

---

## 👤 Author

**Lorenzo Conti** — MSc Management Engineering, Politecnico di Torino
Data & Sustainable Finance Analyst @IntesaSanpaolo

[LinkedIn](https://www.linkedin.com/in/lorenzo-conti02) · [GitHub](https://github.com/LorenzoConti02)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
