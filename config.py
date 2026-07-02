"""
config.py
---------
Configurazione centralizzata del progetto Quant Risk Portfolio Analyzer.
Modifica questi parametri per adattare l'analisi al tuo portafoglio.
"""

from datetime import datetime, timedelta

# ------------------------------------------------------------------
# UNIVERSO DI INVESTIMENTO
# ------------------------------------------------------------------
# Ticker Yahoo Finance del portafoglio multi-asset (azioni, ETF, obbligazionario, oro)
TICKERS = [
    "AAPL",     # Equity - Tech US
    "MSFT",     # Equity - Tech US
    "JPM",      # Equity - Financials US
    "EUNL.DE",  # ETF - MSCI World (iShares Core)
    "TLT",      # ETF - US Treasury Bond 20+ anni
    "GLD",      # ETF - Oro
]

# Pesi del portafoglio (devono sommare a 1.0)
WEIGHTS = {
    "AAPL": 0.20,
    "MSFT": 0.20,
    "JPM": 0.15,
    "EUNL.DE": 0.20,
    "TLT": 0.15,
    "GLD": 0.10,
}

# ------------------------------------------------------------------
# ORIZZONTE TEMPORALE
# ------------------------------------------------------------------
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")  # 5 anni storici

# ------------------------------------------------------------------
# PARAMETRI DI RISCHIO
# ------------------------------------------------------------------
CONFIDENCE_LEVELS = [0.95, 0.99]      # Livelli di confidenza per VaR/CVaR
VAR_HORIZON_DAYS = 1                  # Orizzonte del VaR (1 giorno)
PORTFOLIO_VALUE = 1_000_000           # Valore nozionale del portafoglio
RISK_FREE_RATE = 0.035                # Tasso risk-free annuo (per Sharpe Ratio)
TRADING_DAYS = 252

# ------------------------------------------------------------------
# MONTE CARLO
# ------------------------------------------------------------------
N_SIMULATIONS = 50_000
RANDOM_SEED = 42

# ------------------------------------------------------------------
# STRESS TEST SCENARIOS
# ------------------------------------------------------------------
# Shock istantanei applicati ai rendimenti giornalieri dei singoli asset (%),
# ispirati a eventi storici reali
STRESS_SCENARIOS = {
    "Black Monday 1987 (-20% Equity)": {
        "AAPL": -0.20, "MSFT": -0.20, "JPM": -0.20,
        "EUNL.DE": -0.20, "TLT": 0.03, "GLD": 0.02,
    },
    "GFC 2008 (Equity Crash + Flight to Quality)": {
        "AAPL": -0.09, "MSFT": -0.09, "JPM": -0.15,
        "EUNL.DE": -0.08, "TLT": 0.04, "GLD": 0.03,
    },
    "COVID Crash Marzo 2020": {
        "AAPL": -0.13, "MSFT": -0.15, "JPM": -0.18,
        "EUNL.DE": -0.11, "TLT": 0.05, "GLD": -0.02,
    },
    "Rialzo Tassi Aggressivo (Rate Shock +150bps)": {
        "AAPL": -0.05, "MSFT": -0.05, "JPM": 0.03,
        "EUNL.DE": -0.04, "TLT": -0.12, "GLD": -0.03,
    },
    "Shock Geopolitico / Flight to Safety": {
        "AAPL": -0.06, "MSFT": -0.06, "JPM": -0.07,
        "EUNL.DE": -0.05, "TLT": 0.02, "GLD": 0.06,
    },
}

# ------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
CACHE_DIR = "data_cache"
