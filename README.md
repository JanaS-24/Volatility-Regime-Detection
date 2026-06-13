# Volatility Regime Detection in Equity Markets using Unsupervised Machine Learning

## Overview
This project applies unsupervised machine learning (K-Means and Gaussian Mixture Models) 
to identify latent volatility regimes in the FTSE 100 index using ~20 years of daily data 
(2006-2025). The goal is to demonstrate how data-driven clustering can uncover market 
states without relying on restrictive parametric assumptions (e.g. GARCH, HMMs).

## Key Question
Can unsupervised clustering identify economically meaningful volatility regimes, and how 
does this information support risk management and portfolio allocation decisions?

## Methodology
- **Data**: Daily FTSE 100 prices (2006–2025) via Yahoo Finance
- **Feature Engineering**: Log returns, 21-day rolling volatility, squared returns
- **Standardisation**: Features scaled to zero mean/unit variance
- **Clustering**: K-Means (primary) and GMM (robustness check), k=3
- **Regimes identified**: Low, High, and Extreme volatility

## Key Findings
- **Regime 0 (Low Volatility)**: 4,363 days, ~12.5% annualised vol, positive average returns
- **Regime 1 (High Volatility)**: 650 days, ~33% annualised vol, negative average returns
- **Regime 2 (Extreme Volatility)**: 10 days, ~146% annualised vol — captures crisis events 
  (2008 GFC, 2020 COVID crash)
- High-risk regimes are short-lived but highly damaging — average duration of 7 days vs 
  53 days for low-volatility periods
- A simple regime-based allocation strategy (reducing exposure by 50% during high-risk 
  regimes) **improved risk-adjusted returns**: annualised return rose from 2.7% to 3.5%, 
  while volatility fell from 17.7% to 13.4%

## Implications
Demonstrates a practical, interpretable framework for risk monitoring and dynamic 
portfolio allocation — relevant to risk management teams in asset management and 
banking.

## Tools Used
Python | Pandas | NumPy | scikit-learn (KMeans, GaussianMixture) | Matplotlib | 
yfinance | SciPy

## Files
- `volatility_regimes.ipynb` — Full analysis notebook
- `ftse_features_with_regimes.csv` — Output dataset with regime labels
- `ftse_regime_summary.csv` — Regime summary statistics

## Limitations
- Single index (FTSE 100) — generalisability across markets not tested
- Regime detection is retrospective (ex-post), not predictive
- Results depend on choice of features and number of clusters (k=3)
