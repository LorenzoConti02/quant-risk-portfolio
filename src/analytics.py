"""
analytics.py
------------
Motore quantitativo per il calcolo di:
  - Metriche di portafoglio (rendimento, volatilita', Sharpe, Sortino, Max Drawdown)
  - Value at Risk (VaR) con tre metodologie: Storica, Parametrica (Varianza-Covarianza), Monte Carlo
  - Conditional VaR / Expected Shortfall (CVaR)
  - Stress Testing su scenari storici/ipotetici
  - Backtesting del VaR (Kupiec Proportion of Failures Test)

Convenzioni:
  - I rendimenti sono LOG-RETURNS giornalieri.
  - Il VaR e il CVaR sono espressi come PERDITE (numero positivo = perdita attesa).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Contenitore strutturato per i risultati del VaR."""
    method: str
    confidence: float
    horizon_days: int
    var_pct: float          # VaR in percentuale del portafoglio
    var_value: float        # VaR in valuta
    cvar_pct: float = None
    cvar_value: float = None

    def __repr__(self):
        return (f"[{self.method}] VaR {self.confidence:.0%} ({self.horizon_days}d): "
                f"{self.var_pct:.2%} | {self.var_value:,.0f}  "
                f"CVaR: {self.cvar_pct:.2%} | {self.cvar_value:,.0f}")


class RiskEngine:
    """
    Classe principale per l'analisi del rischio di un portafoglio multi-asset.
    """

    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 portfolio_value: float = 1_000_000, trading_days: int = 252,
                 risk_free_rate: float = 0.02):
        self._validate_inputs(returns, weights)
        self.returns = returns[list(weights.keys())].copy()
        self.weights = np.array([weights[t] for t in self.returns.columns])
        self.tickers = list(self.returns.columns)
        self.portfolio_value = portfolio_value
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate

        self.portfolio_returns = self._compute_portfolio_returns()
        self.cov_matrix = self.returns.cov()

    # ------------------------------------------------------------------
    # VALIDAZIONE
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(returns: pd.DataFrame, weights: Dict[str, float]) -> None:
        if returns is None or returns.empty:
            raise ValueError("Il DataFrame dei rendimenti e' vuoto.")
        if not weights:
            raise ValueError("Il dizionario dei pesi e' vuoto.")
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-3):
            raise ValueError(f"I pesi devono sommare a 1.0 (attuale: {total_weight:.4f}).")
        missing = set(weights.keys()) - set(returns.columns)
        if missing:
            raise ValueError(f"Ticker nei pesi assenti nei dati di rendimento: {missing}")

    # ------------------------------------------------------------------
    # RENDIMENTI DI PORTAFOGLIO
    # ------------------------------------------------------------------
    def _compute_portfolio_returns(self) -> pd.Series:
        port_ret = self.returns.values @ self.weights
        return pd.Series(port_ret, index=self.returns.index, name="Portfolio")

    # ------------------------------------------------------------------
    # METRICHE DI PERFORMANCE
    # ------------------------------------------------------------------
    def performance_summary(self) -> Dict[str, float]:
        """Calcola le metriche standard di performance e rischio, annualizzate."""
        r = self.portfolio_returns
        ann_return = r.mean() * self.trading_days
        ann_vol = r.std() * np.sqrt(self.trading_days)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        downside = r[r < 0]
        downside_vol = downside.std() * np.sqrt(self.trading_days) if len(downside) > 0 else np.nan
        sortino = (ann_return - self.risk_free_rate) / downside_vol if downside_vol else np.nan

        cum_returns = (1 + r).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

        return {
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Skewness": stats.skew(r),
            "Kurtosis": stats.kurtosis(r),
        }

    # ------------------------------------------------------------------
    # 1) VAR STORICO (Historical Simulation)
    # ------------------------------------------------------------------
    def historical_var(self, confidence: float = 0.95, horizon_days: int = 1) -> VaRResult:
        """
        Metodo non parametrico: usa direttamente la distribuzione empirica
        dei rendimenti storici, senza assumere normalita'.
        VaR = -percentile(alpha) dei rendimenti scalati sull'orizzonte.
        """
        alpha = 1 - confidence
        scaled_returns = self.portfolio_returns * np.sqrt(horizon_days)

        var_pct = -np.percentile(scaled_returns, alpha * 100)
        tail_losses = scaled_returns[scaled_returns <= -var_pct]
        cvar_pct = -tail_losses.mean() if len(tail_losses) > 0 else var_pct

        return VaRResult(
            method="Historical Simulation",
            confidence=confidence,
            horizon_days=horizon_days,
            var_pct=var_pct,
            var_value=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_value=cvar_pct * self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # 2) VAR PARAMETRICO (Variance-Covariance / Delta-Normal)
    # ------------------------------------------------------------------
    def parametric_var(self, confidence: float = 0.95, horizon_days: int = 1) -> VaRResult:
        """
        Assume che i rendimenti di portafoglio seguano N(mu, sigma^2).
        VaR = -(mu*h + z_alpha * sigma * sqrt(h)), dove sigma^2 = w' * Cov * w
        """
        mu = self.portfolio_returns.mean()
        sigma = np.sqrt(self.weights @ self.cov_matrix.values @ self.weights)

        z = stats.norm.ppf(1 - confidence)  # z negativo per code sinistre
        var_pct = -(mu * horizon_days + z * sigma * np.sqrt(horizon_days))

        # CVaR parametrico in forma chiusa per distribuzione normale:
        # CVaR = -(mu - sigma * phi(z) / alpha)
        phi_z = stats.norm.pdf(z)
        cvar_pct = -(mu * horizon_days - sigma * np.sqrt(horizon_days) * phi_z / (1 - confidence))

        return VaRResult(
            method="Parametric (Variance-Covariance)",
            confidence=confidence,
            horizon_days=horizon_days,
            var_pct=var_pct,
            var_value=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_value=cvar_pct * self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # 3) VAR MONTE CARLO
    # ------------------------------------------------------------------
    def monte_carlo_var(self, confidence: float = 0.95, horizon_days: int = 1,
                         n_simulations: int = 50_000, seed: int = 42) -> VaRResult:
        """
        Simula n_simulations scenari di rendimenti multivariati correlati
        via decomposizione di Cholesky della matrice di covarianza,
        poi aggrega al livello di portafoglio.
        """
        rng = np.random.default_rng(seed)
        mu_vec = self.returns.mean().values
        cov = self.cov_matrix.values

        # Regolarizzazione per garantire semi-definita positiva (stabilita' numerica)
        cov_reg = cov + np.eye(len(cov)) * 1e-10
        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            logger.warning("Matrice di covarianza non PSD, applico eigenvalue clipping.")
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            eigvals = np.clip(eigvals, 1e-10, None)
            cov_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(cov_reg)

        z = rng.standard_normal((n_simulations, len(mu_vec)))
        simulated_asset_returns = mu_vec * horizon_days + (z @ L.T) * np.sqrt(horizon_days)
        simulated_portfolio_returns = simulated_asset_returns @ self.weights

        alpha = 1 - confidence
        var_pct = -np.percentile(simulated_portfolio_returns, alpha * 100)
        tail = simulated_portfolio_returns[simulated_portfolio_returns <= -var_pct]
        cvar_pct = -tail.mean() if len(tail) > 0 else var_pct

        return VaRResult(
            method="Monte Carlo Simulation",
            confidence=confidence,
            horizon_days=horizon_days,
            var_pct=var_pct,
            var_value=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_value=cvar_pct * self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # STRESS TESTING
    # ------------------------------------------------------------------
    def stress_test(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Applica shock istantanei ai singoli asset e misura l'impatto
        sul valore di portafoglio, dato lo schema di pesi corrente.

        Parameters
        ----------
        scenarios : dict
            {"Nome Scenario": {"TICKER": shock_percentuale, ...}, ...}
        """
        results = []
        for name, shocks in scenarios.items():
            shock_vector = np.array([shocks.get(t, 0.0) for t in self.tickers])
            portfolio_shock = shock_vector @ self.weights
            results.append({
                "Scenario": name,
                "Portfolio Impact (%)": portfolio_shock,
                "Portfolio Impact (Value)": portfolio_shock * self.portfolio_value,
            })
        df = pd.DataFrame(results).sort_values("Portfolio Impact (%)")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # BACKTESTING DEL VAR — KUPIEC POF TEST
    # ------------------------------------------------------------------
    def kupiec_backtest(self, confidence: float = 0.95, window: int = 250) -> Dict[str, float]:
        """
        Backtest rolling del VaR storico: conta quante volte la perdita
        realizzata supera il VaR previsto (violazioni) e verifica se il
        tasso di violazione e' statisticamente coerente con il livello
        di confidenza atteso (Kupiec Proportion of Failures test).
        """
        r = self.portfolio_returns
        if len(r) <= window:
            raise ValueError("Serie storica troppo corta per il backtest richiesto.")

        alpha = 1 - confidence
        violations = 0
        n_tests = 0

        for i in range(window, len(r)):
            window_returns = r.iloc[i - window:i]
            var_threshold = -np.percentile(window_returns, alpha * 100)
            actual_return = r.iloc[i]
            if actual_return < -var_threshold:
                violations += 1
            n_tests += 1

        observed_rate = violations / n_tests
        # Statistica Likelihood Ratio di Kupiec
        if violations == 0 or violations == n_tests:
            lr_stat = 0.0
        else:
            lr_stat = -2 * (
                np.log(((1 - alpha) ** (n_tests - violations)) * (alpha ** violations))
                - np.log(((1 - observed_rate) ** (n_tests - violations)) * (observed_rate ** violations))
            )
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        return {
            "n_tests": n_tests,
            "violations": violations,
            "expected_violations": round(alpha * n_tests, 1),
            "observed_violation_rate": observed_rate,
            "expected_violation_rate": alpha,
            "LR_statistic": lr_stat,
            "p_value": p_value,
            "model_accepted_at_95%": p_value > 0.05,
        }
