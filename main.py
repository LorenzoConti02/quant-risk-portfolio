"""
main.py
-------
Entry point del progetto Quant Risk Portfolio Analyzer.

Esegue la pipeline completa:
  1. Caricamento dati di mercato (Yahoo Finance, con fallback sintetico)
  2. Calcolo rendimenti e metriche di performance
  3. Calcolo VaR/CVaR con 3 metodologie (Storica, Parametrica, Monte Carlo)
  4. Stress Testing su scenari storici
  5. Backtesting del modello VaR (Kupiec Test)
  6. Generazione grafici e report Markdown

Uso:
    python main.py                # dati reali da Yahoo Finance
    python main.py --synthetic    # dati sintetici (demo/offline/CI)
"""

import argparse
import logging
import sys

import config
from src.data_loader import DataLoader, DataLoaderError
from src.analytics import RiskEngine
from src.visualization import ReportVisualizer
from src.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run(use_synthetic: bool = False) -> None:
    loader = DataLoader(config.TICKERS, config.START_DATE, config.END_DATE, cache_dir=config.CACHE_DIR)

    if use_synthetic:
        logger.info("Modalita' SINTETICA attiva: genero dati simulati.")
        prices = DataLoader.generate_synthetic_prices(config.TICKERS)
    else:
        try:
            prices = loader.fetch_prices()
        except DataLoaderError as e:
            logger.error(f"Impossibile scaricare dati reali ({e}). Passo a dati sintetici.")
            prices = DataLoader.generate_synthetic_prices(config.TICKERS)

    returns = DataLoader.compute_log_returns(prices)
    logger.info(f"Dataset rendimenti: {returns.shape[0]} osservazioni, {returns.shape[1]} asset.")

    engine = RiskEngine(
        returns=returns,
        weights=config.WEIGHTS,
        portfolio_value=config.PORTFOLIO_VALUE,
        trading_days=config.TRADING_DAYS,
        risk_free_rate=config.RISK_FREE_RATE,
    )

    performance = engine.performance_summary()
    logger.info("Metriche di performance calcolate.")

    var_results = []
    for conf in config.CONFIDENCE_LEVELS:
        var_results.append(engine.historical_var(confidence=conf, horizon_days=config.VAR_HORIZON_DAYS))
        var_results.append(engine.parametric_var(confidence=conf, horizon_days=config.VAR_HORIZON_DAYS))
        var_results.append(engine.monte_carlo_var(
            confidence=conf, horizon_days=config.VAR_HORIZON_DAYS,
            n_simulations=config.N_SIMULATIONS, seed=config.RANDOM_SEED,
        ))

    for res in var_results:
        logger.info(res)

    stress_df = engine.stress_test(config.STRESS_SCENARIOS)
    logger.info("Stress test completato.")

    try:
        backtest = engine.kupiec_backtest(confidence=0.95, window=250)
        logger.info(f"Backtest Kupiec: {backtest}")
    except ValueError as e:
        logger.warning(f"Backtest non eseguito: {e}")
        backtest = {"info": "Serie storica insufficiente per il backtest (minimo 500 osservazioni)."}

    visualizer = ReportVisualizer(output_dir=config.OUTPUT_DIR)
    visualizer.plot_cumulative_returns(engine.portfolio_returns)
    visualizer.plot_drawdown(engine.portfolio_returns)
    visualizer.plot_return_distribution(
        engine.portfolio_returns,
        [r for r in var_results if r.confidence == 0.95],
    )
    visualizer.plot_correlation_heatmap(returns)
    visualizer.plot_stress_test(stress_df)
    visualizer.plot_var_comparison([r for r in var_results if r.confidence == 0.95])
    logger.info(f"Grafici salvati in: {config.OUTPUT_DIR}/")

    reporter = ReportGenerator(output_dir=config.OUTPUT_DIR)
    report_path = reporter.build(
        tickers=list(config.WEIGHTS.keys()),
        weights=config.WEIGHTS,
        performance=performance,
        var_results=var_results,
        stress_df=stress_df,
        backtest=backtest,
        portfolio_value=config.PORTFOLIO_VALUE,
    )
    logger.info(f"Report generato: {report_path}")

    print("\n" + "=" * 70)
    print("QUANT RISK PORTFOLIO ANALYZER — Esecuzione completata con successo")
    print("=" * 70)
    print(f"Report: {report_path}")
    print(f"Grafici: {config.OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Quant Risk Portfolio Analyzer")
    parser.add_argument("--synthetic", action="store_true",
                        help="Usa dati di mercato sintetici invece di Yahoo Finance")
    args = parser.parse_args()

    try:
        run(use_synthetic=args.synthetic)
    except Exception as e:
        logger.exception(f"Errore fatale nell'esecuzione della pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
