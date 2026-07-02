"""
test_analytics.py
------------------
Test unitari per il modulo analytics.py. Utilizza dati sintetici
per garantire riproducibilita' e indipendenza da connessioni di rete.
Esegui con: pytest tests/ -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analytics import RiskEngine


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(123)
    n_days = 1000
    tickers = ["A", "B", "C"]
    cov = np.array([[0.0004, 0.0001, 0.00005],
                     [0.0001, 0.0003, 0.00003],
                     [0.00005, 0.00003, 0.0002]])
    mu = np.array([0.0004, 0.0003, 0.0002])
    data = rng.multivariate_normal(mu, cov, size=n_days)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def engine(synthetic_returns):
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    return RiskEngine(synthetic_returns, weights, portfolio_value=1_000_000)


def test_weights_must_sum_to_one(synthetic_returns):
    with pytest.raises(ValueError):
        RiskEngine(synthetic_returns, {"A": 0.5, "B": 0.3, "C": 0.1}, portfolio_value=1_000_000)


def test_missing_ticker_in_returns(synthetic_returns):
    with pytest.raises(ValueError):
        RiskEngine(synthetic_returns, {"A": 0.5, "B": 0.3, "D": 0.2}, portfolio_value=1_000_000)


def test_empty_returns_raises():
    with pytest.raises(ValueError):
        RiskEngine(pd.DataFrame(), {"A": 1.0}, portfolio_value=1_000_000)


def test_historical_var_positive(engine):
    res = engine.historical_var(confidence=0.95)
    assert res.var_pct > 0
    assert res.cvar_pct >= res.var_pct  # CVaR (perdita media in coda) >= VaR


def test_parametric_var_positive(engine):
    res = engine.parametric_var(confidence=0.99)
    assert res.var_pct > 0


def test_monte_carlo_var_converges_to_parametric(engine):
    """Con rendimenti gaussiani, il VaR Monte Carlo deve convergere al VaR parametrico."""
    mc = engine.monte_carlo_var(confidence=0.95, n_simulations=100_000, seed=1)
    param = engine.parametric_var(confidence=0.95)
    assert abs(mc.var_pct - param.var_pct) < 0.01  # tolleranza 1 punto percentuale


def test_var_increases_with_confidence(engine):
    var_95 = engine.historical_var(confidence=0.95)
    var_99 = engine.historical_var(confidence=0.99)
    assert var_99.var_pct >= var_95.var_pct


def test_var_scales_with_sqrt_time(engine):
    """VaR a 10 giorni deve essere circa sqrt(10) volte il VaR a 1 giorno (parametrico)."""
    var_1d = engine.parametric_var(confidence=0.95, horizon_days=1)
    var_10d = engine.parametric_var(confidence=0.95, horizon_days=10)
    ratio = var_10d.var_pct / var_1d.var_pct
    assert abs(ratio - np.sqrt(10)) < 0.5


def test_stress_test_output_shape(engine):
    scenarios = {"Crash Test": {"A": -0.10, "B": -0.05, "C": 0.02}}
    df = engine.stress_test(scenarios)
    assert len(df) == 1
    expected_impact = -0.10 * 0.5 + -0.05 * 0.3 + 0.02 * 0.2
    assert np.isclose(df.iloc[0]["Portfolio Impact (%)"], expected_impact)


def test_performance_summary_keys(engine):
    perf = engine.performance_summary()
    expected_keys = {"Annualized Return", "Annualized Volatility", "Sharpe Ratio",
                     "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Skewness", "Kurtosis"}
    assert expected_keys.issubset(perf.keys())


def test_max_drawdown_is_negative_or_zero(engine):
    perf = engine.performance_summary()
    assert perf["Max Drawdown"] <= 0


def test_kupiec_backtest_runs(engine):
    result = engine.kupiec_backtest(confidence=0.95, window=250)
    assert "violations" in result
    assert result["n_tests"] > 0
    assert 0 <= result["observed_violation_rate"] <= 1


def test_kupiec_backtest_insufficient_data():
    rng = np.random.default_rng(1)
    short_returns = pd.DataFrame(
        rng.normal(0, 0.01, (100, 2)),
        columns=["A", "B"],
        index=pd.bdate_range(end=pd.Timestamp.today(), periods=100),
    )
    eng = RiskEngine(short_returns, {"A": 0.5, "B": 0.5}, portfolio_value=1_000_000)
    with pytest.raises(ValueError):
        eng.kupiec_backtest(window=250)
