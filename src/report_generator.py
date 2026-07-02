"""
report_generator.py
--------------------
Genera un report sintetico in Markdown con tutti i risultati dell'analisi
di rischio, pronto per essere condiviso, allegato a una presentazione o
convertito in PDF.
"""

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.analytics import VaRResult


class ReportGenerator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build(self, tickers: List[str], weights: Dict[str, float],
              performance: Dict[str, float], var_results: List[VaRResult],
              stress_df: pd.DataFrame, backtest: Dict, portfolio_value: float) -> str:

        lines = []
        lines.append("# Risk Report — Multi-Asset Portfolio")
        lines.append(f"_Generated on {datetime.today().strftime('%Y-%m-%d %H:%M')}_\n")

        lines.append("## 1. Portfolio Composition")
        lines.append(f"Notional value: **{portfolio_value:,.0f}**\n")
        lines.append("| Ticker | Weight |")
        lines.append("|---|---|")
        for t in tickers:
            lines.append(f"| {t} | {weights[t]:.1%} |")
        lines.append("")

        lines.append("## 2. Performance Metrics")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in performance.items():
            if "Ratio" in k or "Skew" in k or "Kurt" in k:
                lines.append(f"| {k} | {v:.3f} |")
            else:
                lines.append(f"| {k} | {v:.2%} |")
        lines.append("")

        lines.append("## 3. Value at Risk (VaR) & Conditional VaR (CVaR)")
        lines.append("| Method | Confidence | VaR % | VaR (value) | CVaR % | CVaR (value) |")
        lines.append("|---|---|---|---|---|---|")
        for res in var_results:
            lines.append(
                f"| {res.method} | {res.confidence:.0%} | {res.var_pct:.2%} | "
                f"{res.var_value:,.0f} | {res.cvar_pct:.2%} | {res.cvar_value:,.0f} |"
            )
        lines.append("")

        lines.append("## 4. Stress Testing — Historical & Hypothetical Scenarios")
        lines.append("| Scenario | Impact (%) | Impact (Value) |")
        lines.append("|---|---|---|")
        for _, row in stress_df.iterrows():
            lines.append(
                f"| {row['Scenario']} | {row['Portfolio Impact (%)']:.2%} | "
                f"{row['Portfolio Impact (Value)']:,.0f} |"
            )
        lines.append("")

        lines.append("## 5. VaR Backtesting (Kupiec POF Test)")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in backtest.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        if "model_accepted_at_95%" in backtest:
            verdict = "PASSED" if backtest.get("model_accepted_at_95%") else "REJECTED"
            lines.append(f"**Kupiec Test Verdict (95% conf.): Model {verdict}**\n")

        report_text = "\n".join(lines)
        path = os.path.join(self.output_dir, "risk_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return path
