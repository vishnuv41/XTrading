# Quantitative Methodology & Machine Learning Protocol

## Strategy Specification

Phase 13 candidate formulation:
> **50/50 XGBoost + CatBoost continuous-return regression → 66 causal MTF features → 250-prediction rolling Top-1% rank gate → 1.5 ATR SL / 3.0 ATR TP → 48-bar timeout → 4-bar cooldown → 10 BPS fees/slippage**

---

## 1. Feature Engineering (66 Multi-Timeframe Features)

The feature pipeline transforms raw 1m and 1h OHLCV price series into 66 stationary, non-lookahead causal features designed to capture multi-scale market dynamics:

- **Momentum & Trend**: Exponential Moving Average Spreads ($\frac{\text{EMA}_{12} - \text{EMA}_{26}}{\text{Close}}$), MACD Histograms, Relative Strength Index (RSI 7, 14, 21), Supertrend direction/distance.
- **Volatility Regimes**: Normalized Average True Range ($\frac{\text{ATR}_{14}}{\text{Close}}$), Bollinger Band Width, Historical Volatility Ratios ($V_{1\text{h}} / V_{24\text{h}}$), Volatility Skew.
- **Volume & Flow**: Volume Z-Scores, On-Balance Volume (OBV) Slope, Volume-Weighted Average Price (VWAP) Distance.
- **Multi-Timeframe Interactions**: 1m to 1h trend agreement, multi-period return momentum across 1h, 4h, 12h, and 24h lookbacks.

---

## 2. Model Ensemble Architecture

- **Regressors**: Equal-weighted ensemble (50% XGBoost Regressor + 50% CatBoost Regressor).
- **Target Formulation**: Continuous return regression predicting expected triple-barrier return adjusted for volatility.
- **No Synthetic Fallback**: Realized predictions strictly use model inferenced values against historical state.

---

## 3. Signal Generation & 250-Bar Rolling Rank Gate

Raw regression predictions $\hat{y}_t$ are evaluated against a rolling 250-prediction historical window:

$$\text{Percentile}(\hat{y}_t) = \frac{\text{Count}(\hat{y}_{\text{roll\_250}} < \hat{y}_t)}{250}$$

- **Signal Gate**: Top 1.0% Percentile ($p \ge 0.9900$).
- **Action**: Trade entries trigger ONLY when predicted return exceeds the 99th percentile threshold of the rolling 250-prediction window.

---

## 4. Risk & Execution Rules

- **Position Sizing**: Fixed percentage risk heat per trade ($1.0\%$ of total account equity).
- **Stop Loss (SL)**: $1.5 \times \text{ATR}_{14}$ from entry price.
- **Take Profit (TP)**: $3.0 \times \text{ATR}_{14}$ from entry price (Target 2.0 R:R ratio).
- **Timeout Exit**: Maximum holding period of 48 1-hour candles.
- **Trade Cooldown**: Minimum 4-bar cooldown required after a trade exit before a new position can open on the same symbol.
- **Frictions**: 10 Basis Points (0.10%) taker fee per leg + conservative fill slippage assumptions.
- **Intrabar SL Priority**: In bars where High $\ge$ TP and Low $\le$ SL simultaneously, Stop Loss fill is strictly prioritized.
