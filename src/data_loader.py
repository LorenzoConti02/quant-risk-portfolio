"""
data_loader.py
--------------
Modulo responsabile del recupero e della pulizia dei dati di mercato.
Utilizza yfinance come fonte primaria, con caching locale su CSV per
ridurre le chiamate API e garantire riproducibilita'. Include un
generatore di dati sintetici (GBM multivariato) da usare come fallback
offline o in ambienti di test/CI senza accesso a Internet.
"""

import os
import logging
from typing import List

import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Eccezione custom per errori di caricamento dati."""
    pass


class DataLoader:
    """
    Gestisce il download, la validazione e il caching dei prezzi storici
    per un paniere di ticker tramite Yahoo Finance.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 cache_dir: str = "data_cache"):
        if not tickers:
            raise ValueError("La lista dei ticker non puo' essere vuota.")
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self) -> str:
        tickers_hash = "_".join(sorted(self.tickers))
        filename = f"prices_{tickers_hash}_{self.start_date}_{self.end_date}.csv".replace(":", "-")
        return os.path.join(self.cache_dir, filename)

    def fetch_prices(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        """
        Scarica i prezzi Adjusted Close per tutti i ticker.

        Returns
        -------
        pd.DataFrame
            DataFrame con indice Date e colonne = ticker, valori = Adj Close.
        """
        cache_file = self._cache_path()

        if use_cache and not force_refresh and os.path.exists(cache_file):
            logger.info(f"Carico dati da cache locale: {cache_file}")
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Cache corrotta ({e}), procedo con nuovo download.")

        logger.info(f"Download dati da Yahoo Finance per: {self.tickers}")
        try:
            raw = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as e:
            raise DataLoaderError(f"Errore durante il download da Yahoo Finance: {e}")

        if raw is None or raw.empty:
            raise DataLoaderError(
                "Nessun dato restituito. Controlla i ticker o la connessione di rete."
            )

        # Gestione del caso singolo ticker vs multi-ticker (yfinance cambia struttura colonne)
        if len(self.tickers) == 1:
            prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
        else:
            try:
                prices = pd.concat(
                    {t: raw[t]["Close"] for t in self.tickers if t in raw.columns.get_level_values(0)},
                    axis=1,
                )
            except (KeyError, TypeError):
                # Fallback per strutture MultiIndex differenti tra versioni di yfinance
                prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

        prices = prices.dropna(how="all")
        missing_tickers = set(self.tickers) - set(prices.columns)
        if missing_tickers:
            logger.warning(f"Ticker non trovati o senza dati: {missing_tickers}")

        prices = prices.ffill().dropna(how="any")

        if prices.empty:
            raise DataLoaderError("Il DataFrame dei prezzi e' vuoto dopo la pulizia.")

        prices.to_csv(cache_file)
        logger.info(f"Dati salvati in cache: {cache_file}")
        return prices

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calcola i rendimenti logaritmici giornalieri: r_t = ln(P_t / P_t-1)."""
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        return returns

    @staticmethod
    def generate_synthetic_prices(tickers: List[str], n_days: int = 1260,
                                   seed: int = 42) -> pd.DataFrame:
        """
        Genera una serie storica sintetica di prezzi (GBM multivariato correlato).
        Utile per demo, testing offline o CI/CD quando l'API di mercato
        non e' raggiungibile (es. rate limit, assenza di rete).
        """
        rng = np.random.default_rng(seed)
        n_assets = len(tickers)

        # Matrice di correlazione plausibile
        base_corr = 0.35
        corr = np.full((n_assets, n_assets), base_corr)
        np.fill_diagonal(corr, 1.0)
        vol = rng.uniform(0.15, 0.30, n_assets)  # volatilita' annua per asset
        cov = np.outer(vol, vol) * corr / 252

        mu = rng.uniform(0.02, 0.10, n_assets) / 252  # drift giornaliero
        daily_returns = rng.multivariate_normal(mu, cov, size=n_days)

        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)
