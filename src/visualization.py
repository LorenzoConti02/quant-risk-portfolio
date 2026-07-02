"""
visualization.py
-----------------
Funzioni di plotting per il report di rischio: rendimento cumulato,
drawdown, distribuzione dei rendimenti con VaR, matrice di correlazione,
stress test e confronto tra metodologie di VaR.
"""

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


class ReportVisualizer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, fig, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_cumulative_returns(self, returns: pd.Series, filename: str = "cumulative_returns.png") -> str:
        cum = (1 + returns).cumprod()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cum.index, cum.values, color="#1f4e79", linewidth=1.6)
        ax.set_title("Rendimento Cumulato del Portafoglio", fontsize=13, fontweight="bold")
        ax.set_ylabel("Valore (base 1.0)")
        ax.set_xlabel("Data")
        return self._save(fig, filename)

    def plot_drawdown(self, returns: pd.Series, filename: str = "drawdown.png") -> str:
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#c0392b", alpha=0.6)
        ax.set_title("Drawdown Storico del Portafoglio", fontsize=13, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        return self._save(fig, filename)

    def plot_return_distribution(self, returns: pd.Series, var_results: List,
                                  filename: str = "var_distribution.png") -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(returns * 100, bins=80, kde=True, color="#2874a6", ax=ax, stat="density")

        colors = ["#e74c3c", "#8e44ad", "#f39c12"]
        for i, res in enumerate(var_results):
            ax.axvline(-res.var_pct * 100, color=colors[i % len(colors)], linestyle="--",
                       linewidth=1.8, label=f"{res.method} VaR {res.confidence:.0%}: {res.var_pct:.2%}")

        ax.set_title("Distribuzione dei Rendimenti Giornalieri e Value at Risk", fontsize=13, fontweight="bold")
        ax.set_xlabel("Rendimento Giornaliero (%)")
        ax.legend(fontsize=8)
        return self._save(fig, filename)

    def plot_correlation_heatmap(self, returns: pd.DataFrame, filename: str = "correlation_heatmap.png") -> str:
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, ax=ax,
                    cbar_kws={"label": "Correlazione"})
        ax.set_title("Matrice di Correlazione degli Asset", fontsize=13, fontweight="bold")
        return self._save(fig, filename)

    def plot_stress_test(self, stress_df: pd.DataFrame, filename: str = "stress_test.png") -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in stress_df["Portfolio Impact (%)"]]
        ax.barh(stress_df["Scenario"], stress_df["Portfolio Impact (%)"] * 100, color=colors)
        ax.set_title("Stress Test — Impatto sul Portafoglio per Scenario", fontsize=13, fontweight="bold")
        ax.set_xlabel("Impatto sul Portafoglio (%)")
        ax.axvline(0, color="black", linewidth=0.8)
        return self._save(fig, filename)

    def plot_var_comparison(self, var_results: List, filename: str = "var_comparison.png") -> str:
        methods = [r.method for r in var_results]
        vars_pct = [r.var_pct * 100 for r in var_results]
        cvars_pct = [r.cvar_pct * 100 for r in var_results]

        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, vars_pct, width, label="VaR", color="#2874a6")
        ax.bar(x + width / 2, cvars_pct, width, label="CVaR (Expected Shortfall)", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Perdita Attesa (%)")
        ax.set_title("Confronto Metodologie VaR vs CVaR", fontsize=13, fontweight="bold")
        ax.legend()
        return self._save(fig, filename)
