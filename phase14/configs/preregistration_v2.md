# Phase 14B — Pre-Registration Specification Document (v2)

*Status*: PRE-REGISTERED INFORMATION AUDIT SPECIFICATION  
*Date*: 2026-09-10  
*Purpose*: Pre-register evaluation metrics, target formulations, forecast horizons, feature groups, and derivatives market data before running Phase 14B empirical audit.

> [!IMPORTANT]
> **Zero Phase 13 Contamination**: Phase 14B information audits are conducted strictly on historical development data (`BTC/USDT`, `ETH/USDT`). Phase 13 prospective trading remains 100% frozen.

---

## 1. Track 1 — Target Formulations

We pre-register 4 distinct target formulations:
1. **Target 1 (Continuous Return)**: $y_{t+h} = \ln(p_{t+h} / p_t)$
2. **Target 2 (Directional Classification)**: 
   $$y_{t+h} = \begin{cases} +1 & \text{if } y_{t+h} > +\delta \\ -1 & \text{if } y_{t+h} < -\delta \\ 0 & \text{otherwise} \end{cases}$$
3. **Target 3 (Volatility-Adjusted Return)**: $y_{t+h} = \frac{\ln(p_{t+h} / p_t)}{\sigma_{\text{ATR}, 14}}$
4. **Target 4 (Triple-Barrier Label)**: $y_{t+h} \in \{+1 \text{ (TP hit first)}, -1 \text{ (SL hit first)}, 0 \text{ (Timeout)}\}$

---

## 2. Track 2 — Forecast Horizons Grid ($h$)

We pre-register 5 forecast horizons:
$$h \in [30\text{m}, 1\text{h}, 2\text{h}, 4\text{h}, 8\text{h}]$$

---

## 3. Track 3 — Feature Groups

We pre-register 5 feature groups:
1. **Group A (Trend & Moving Averages)**: EMA, SMA, Supertrend, Ichimoku Cloud.
2. **Group B (Momentum & Oscillators)**: RSI, Stochastic, MACD, ROC.
3. **Group C (Volatility & Channels)**: ATR, Bollinger Bands, Donchian Channels, Keltner Channels.
4. **Group D (Volume & Flow)**: OBV, VWAP, CMF, Volume Ratio.
5. **Group E (Multi-Timeframe)**: Resampled 4H / 1D indicators.

---

## 4. Track 4 — Crypto Derivatives Market Data

We pre-register 4 derivatives feature categories:
1. **Funding Rate**: Binance Futures 8H funding rate & 24h rolling funding sum.
2. **Open Interest (OI)**: 1H OI change %, OI/Volume ratio.
3. **Premium / Basis**: Spot vs. Futures premium spread $\frac{p_{\text{futures}} - p_{\text{spot}}}{p_{\text{spot}}}$.
4. **Liquidation Spikes**: Estimated Long/Short liquidation volume spikes.
